#!/usr/bin/env python
"""
Treina um ou todos os modelos do FitoVision.

Uso:
    # Pipeline binario (novo — healthy vs anomalous)
    python train.py --data ./data --binary --model mobilenet_v2
    python train.py --data ./data --binary --model all --epochs 20

    # Pipeline multi-classe (original — 6 classes FitoVision)
    python train.py --data ./data --model all
    python train.py --data ./data --model mobilenet_v2 --epochs 40 --lr 5e-4
    python train.py --data ./plantvillage --model all --mapping plantvillage_map.json

    # Google Colab (mais workers, batch maior)
    python train.py --data ./data --binary --model all --batch-size 64 --workers 2

Saida:
    weights/<model_name>.pth         — melhor modelo (multi-classe)
    weights/<model_name>_binary.pth  — melhor modelo (binario)
    logs/<model_name>_history.json   — historico de treino
    logs/test_split.json             — test split para o evaluate.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset import make_loaders, make_binary_folder_loaders, BinaryFolderDataset, BINARY_CLASSES, VAL_TRANSFORMS
from app.config import CLASS_NAMES, AVAILABLE_MODELS


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Modelo desconhecido: {name}")
    return m


def train_one_model(
    model_name: str,
    train_loader,
    val_loader,
    num_classes: int,
    epochs: int,
    lr: float,
    weights_dir: Path,
    logs_dir: Path,
    device: torch.device,
    patience: int = 10,
    weights_suffix: str = "",
) -> float:
    print(f"\n{'='*60}")
    print(f"  Modelo : {model_name}")
    print(f"  Device : {device}  |  Épocas : {epochs}  |  LR : {lr}")
    print(f"{'='*60}")

    model = build_model(model_name, num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # label_smoothing ajuda na calibração das probabilidades — útil para o TCC
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc    = 0.0
    patience_counter = 0
    history         = []
    t_start         = time.time()

    for epoch in range(1, epochs + 1):
        # ---------- Treino ----------
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"  Epoch {epoch:3d}/{epochs} treino", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            t_loss    += loss.item() * imgs.size(0)
            t_correct += (out.argmax(1) == labels).sum().item()
            t_total   += imgs.size(0)

        scheduler.step()

        # ---------- Validação ----------
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    out  = model(imgs)
                    loss = criterion(out, labels)
                v_loss    += loss.item() * imgs.size(0)
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total   += imgs.size(0)

        t_acc = t_correct / t_total
        v_acc = v_correct / v_total

        record = {
            "epoch":      epoch,
            "train_loss": t_loss / t_total,
            "train_acc":  t_acc,
            "val_loss":   v_loss / v_total,
            "val_acc":    v_acc,
        }
        history.append(record)

        marker = " *" if v_acc > best_val_acc else ""
        print(
            f"  Epoch {epoch:3d} | "
            f"loss={record['train_loss']:.4f} acc={t_acc:.4f} | "
            f"val_loss={record['val_loss']:.4f} val_acc={v_acc:.4f}{marker}"
        )

        if v_acc > best_val_acc:
            best_val_acc     = v_acc
            patience_counter = 0
            save_path = weights_dir / f"{model_name}{weights_suffix}.pth"
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping após {epoch} épocas (patience={patience}).")
                break

    elapsed = time.time() - t_start
    log = {
        "model":        model_name,
        "best_val_acc": best_val_acc,
        "elapsed_s":    elapsed,
        "history":      history,
    }
    log_path = logs_dir / f"{model_name}_history.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n  Melhor val_acc : {best_val_acc:.4f}  ({elapsed/60:.1f} min)")
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Treino FitoVision")
    parser.add_argument("--data",        required=True,
                        help="Diretório raiz do dataset")
    parser.add_argument("--model",       default="all",
                        help=f"Modelo: all | {' | '.join(AVAILABLE_MODELS)}")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--val-split",   type=float, default=0.15)
    parser.add_argument("--test-split",  type=float, default=0.15)
    # No Windows, num_workers>0 pode causar problemas; use 2+ no Colab/Linux
    parser.add_argument("--workers",     type=int,   default=0)
    parser.add_argument("--binary",      action="store_true",
                        help="Pipeline binario: healthy vs anomalous (pre-split em train/val/test/)")
    parser.add_argument("--mapping",     default=None,
                        help="JSON de mapeamento pasta->classe (ex: plantvillage_map.json)")
    parser.add_argument("--weights-dir", default="./weights")
    parser.add_argument("--logs-dir",    default="./logs")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nFitoVision -- Treino")
    print(f"Modo        : {'BINARIO (healthy vs anomalous)' if args.binary else 'MULTI-CLASSE (6 classes)'}")
    print(f"Dispositivo : {device}")

    weights_dir = Path(args.weights_dir)
    logs_dir    = Path(args.logs_dir)
    weights_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"[ERRO] Directorio de dados nao encontrado: {data_path}")

    # ── Pipeline binario ─────────────────────────────────────────────────────
    if args.binary:
        class_names = BINARY_CLASSES  # ["healthy", "anomalous"]
        for split in ("train", "val", "test"):
            for label in class_names:
                d = data_path / split / label
                if not d.exists():
                    sys.exit(
                        f"[ERRO] Pasta ausente: {d}\n"
                        "       Execute download_datasets.py primeiro."
                    )

        print(f"\nCarregando dataset binario de: {data_path}")
        train_loader, val_loader, test_loader = make_binary_folder_loaders(
            data_dir=data_path,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        num_classes     = 2
        weights_suffix  = "_binary"
        test_split_path = logs_dir / "test_split_binary.json"
        # Salva caminhos do test set para o evaluate.py
        test_ds = BinaryFolderDataset(data_path / "test", transform=VAL_TRANSFORMS)
        with open(test_split_path, "w") as f:
            json.dump([[str(p), lbl] for p, lbl in test_ds.samples], f)

    # ── Pipeline multi-classe ─────────────────────────────────────────────────
    else:
        class_names    = CLASS_NAMES
        folder_map     = None
        if args.mapping:
            with open(args.mapping) as f:
                folder_map = json.load(f)

        if folder_map is None:
            missing = [c for c in CLASS_NAMES if not any((data_path / c).glob("*"))]
            if missing:
                sys.exit(
                    f"[ERRO] Classes sem imagens em '{data_path}': {missing}\n"
                    "       Cada classe precisa de uma pasta com seu nome."
                )

        print(f"\nCarregando dataset multi-classe de: {data_path}")
        train_loader, val_loader, _, test_samples = make_loaders(
            data_dir=data_path,
            class_names=CLASS_NAMES,
            batch_size=args.batch_size,
            val_frac=args.val_split,
            test_frac=args.test_split,
            num_workers=args.workers,
            folder_class_map=folder_map,
            seed=args.seed,
        )
        num_classes     = len(CLASS_NAMES)
        weights_suffix  = ""
        test_split_path = logs_dir / "test_split.json"
        with open(test_split_path, "w") as f:
            json.dump([[str(p), lbl] for p, lbl in test_samples], f)

    print(f"Test split salvo em: {test_split_path}")

    models_to_train = AVAILABLE_MODELS if args.model == "all" else [args.model]

    results: dict[str, float] = {}
    for model_name in models_to_train:
        results[model_name] = train_one_model(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            epochs=args.epochs,
            lr=args.lr,
            weights_dir=weights_dir,
            logs_dir=logs_dir,
            device=device,
            patience=args.patience,
            weights_suffix=weights_suffix,
        )

    print("\n\n=== Resumo de Validacao ===")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        pth = weights_dir / f"{name}{weights_suffix}.pth"
        status = "SALVO" if pth.exists() else "FALHOU"
        print(f"  {name:20s}: {acc:.4f}  [{status}]")

    print(f"\nPesos em   : {weights_dir}/")
    print(f"Logs em    : {logs_dir}/")
    print(f"\nProximo passo: python evaluate.py --test-split {test_split_path}")


if __name__ == "__main__":
    main()
