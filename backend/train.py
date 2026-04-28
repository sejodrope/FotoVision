#!/usr/bin/env python
"""
Treina um ou todos os modelos do FitoVision.

Uso:
    # Treinar todos os modelos (dataset organizado por classe)
    python train.py --data ./data --model all

    # Treinar só um modelo com learning rate personalizado
    python train.py --data ./data --model mobilenet_v2 --epochs 40 --lr 5e-4

    # Usar mapeamento PlantVillage -> classes FitoVision
    python train.py --data ./plantvillage --model all --mapping plantvillage_map.json

    # Google Colab (mais workers, batch maior)
    python train.py --data ./data --model all --batch-size 64 --workers 2

Saída:
    weights/<model_name>.pth  — melhor modelo por val_accuracy
    logs/<model_name>_history.json  — histórico de treino
    logs/test_split.json  — índice do test split para o evaluate.py
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
from dataset import make_loaders
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
            save_path = weights_dir / f"{model_name}.pth"
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
    parser.add_argument("--mapping",     default=None,
                        help="JSON de mapeamento pasta->classe (ex: plantvillage_map.json)")
    parser.add_argument("--weights-dir", default="./weights")
    parser.add_argument("--logs-dir",    default="./logs")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nFitoVision — Treino")
    print(f"Dispositivo : {device}")

    weights_dir = Path(args.weights_dir)
    logs_dir    = Path(args.logs_dir)
    weights_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    folder_map = None
    if args.mapping:
        with open(args.mapping) as f:
            folder_map = json.load(f)

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"[ERRO] Directório de dados não encontrado: {data_path}")

    if folder_map is None:
        # Valida que todas as classes têm imagens antes de iniciar o treino
        missing = [cls for cls in CLASS_NAMES if not any((data_path / cls).glob("*"))]
        if missing:
            sys.exit(
                f"[ERRO] Classes sem imagens no directório '{data_path}': {missing}\n"
                "       Certifique-se que cada classe tem uma pasta com o respectivo nome."
            )

    print(f"\nCarregando dataset de: {args.data}")
    train_loader, val_loader, _, test_samples = make_loaders(
        data_dir=args.data,
        class_names=CLASS_NAMES,
        batch_size=args.batch_size,
        val_frac=args.val_split,
        test_frac=args.test_split,
        num_workers=args.workers,
        folder_class_map=folder_map,
        seed=args.seed,
    )

    # Guarda o test split para o evaluate.py usar o mesmo conjunto
    test_split_path = logs_dir / "test_split.json"
    with open(test_split_path, "w") as f:
        json.dump([[str(p), lbl] for p, lbl in test_samples], f)
    print(f"Test split guardado em: {test_split_path}")

    models_to_train = AVAILABLE_MODELS if args.model == "all" else [args.model]

    results: dict[str, float] = {}
    for model_name in models_to_train:
        results[model_name] = train_one_model(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=len(CLASS_NAMES),
            epochs=args.epochs,
            lr=args.lr,
            weights_dir=weights_dir,
            logs_dir=logs_dir,
            device=device,
            patience=args.patience,
        )

    print("\n\n=== Resumo de Validação ===")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        status = "CALIBRADO" if (weights_dir / f"{name}.pth").exists() else "FALHOU"
        print(f"  {name:20s}: {acc:.4f}  [{status}]")

    print(f"\nPesos guardados em  : {weights_dir}/")
    print(f"Logs guardados em   : {logs_dir}/")
    print(f"\nPróximo passo: python evaluate.py --test-split {test_split_path}")


if __name__ == "__main__":
    main()
