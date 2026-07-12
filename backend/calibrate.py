#!/usr/bin/env python
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Calibração de confiança por Temperature Scaling (Guo et al., 2017).

O PROBLEMA
──────────
Redes profundas modernas são sistematicamente EXCESSIVAMENTE CONFIANTES: um
modelo que acerta 85% das vezes reporta "97% de confiança". Numa foto fora do
domínio de treino (uma folha de espécie nunca vista, uma foto de telemóvel com
fundo de terra, ou algo que nem é uma planta), o softmax continua a devolver
valores próximos de 1,0 — porque o softmax não tem como dizer "não sei".

Foi isso que aconteceu no teste com fotos aleatórias: números altos e errados.

A SOLUÇÃO (duas camadas, ambas implementadas aqui)
──────────────────────────────────────────────────
1. TEMPERATURE SCALING — aprende um único escalar T > 0 no conjunto de VALIDAÇÃO
   e divide os logits por ele antes do softmax. Não altera a predição (argmax é
   invariante), apenas achata as probabilidades para que "80% de confiança"
   signifique de facto "acerto 80% das vezes". Mede-se com o ECE
   (Expected Calibration Error).

2. LIMIAR DE ABSTENÇÃO — a partir das probabilidades calibradas, escolhe o limiar
   abaixo do qual o sistema deve dizer "inconclusivo — tire outra foto" em vez de
   arriscar um diagnóstico. É a diferença entre um sistema honesto e um que mente
   com confiança.

USO
───
    python calibrate.py --data ./data --binary
    python calibrate.py --data ./data --binary --model efficientnet_b0

SAÍDA
─────
    weights/<model>_binary_calibration.json   — {"temperature": T, "threshold": τ}
    results/calibration_<model>.png           — diagrama de fiabilidade (antes/depois)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from dataset import BinaryFolderDataset, VAL_TRANSFORMS, BINARY_CLASSES

BASE_DIR = Path(__file__).parent


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    elif name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "vit_b_16":
        m = models.vit_b_16(weights=None)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Modelo desconhecido: {name}")
    return m


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device):
    """Devolve (logits, labels) de todo o conjunto, em CPU."""
    model.eval()
    all_logits, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="  logits", leave=False):
        out = model(imgs.to(device))
        all_logits.append(out.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    ECE: média ponderada de |confiança - accuracy| por bin de confiança.
    ECE = 0 ⇒ perfeitamente calibrado. ECE alto ⇒ o modelo mente sobre a sua certeza.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Optimiza T minimizando a NLL no conjunto de validação (LBFGS, como no paper).
    T > 1 achata as probabilidades (corrige excesso de confiança);
    T < 1 aguça-as (corrige falta de confiança).
    """
    log_t = torch.zeros(1, requires_grad=True)  # parametriza T = exp(log_t) > 0
    nll = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / torch.exp(log_t), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_t).item())


def pick_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    """
    Escolhe o limiar de confiança para abstenção.

    Critério: o menor limiar τ tal que, entre as predições com confiança >= τ,
    a accuracy seja >= 95%. Ou seja: "quando o sistema se pronuncia, acerta 95%
    das vezes". As restantes são devolvidas como 'inconclusivo'.

    Devolve (τ, estatísticas de cobertura).
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)

    best_tau = 0.5
    stats = {"coverage": 1.0, "selective_accuracy": float(correct.mean())}

    for tau in np.arange(0.50, 1.00, 0.01):
        mask = conf >= tau
        if mask.sum() < max(20, 0.05 * len(conf)):
            break                                    # cobertura pequena demais para ser útil
        sel_acc = correct[mask].mean()
        if sel_acc >= 0.95:
            best_tau = float(tau)
            stats = {
                "coverage": float(mask.mean()),
                "selective_accuracy": float(sel_acc),
            }
            break

    return best_tau, stats


def plot_reliability(
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    out: Path,
    n_bins: int = 15,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, probs, title in (
        (axes[0], probs_before, "Antes (softmax cru)"),
        (axes[1], probs_after,  "Depois (temperature scaling)"),
    ):
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        correct = (pred == labels).astype(float)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        centers, accs, confs = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf > lo) & (conf <= hi)
            if not mask.any():
                continue
            centers.append((lo + hi) / 2)
            accs.append(correct[mask].mean())
            confs.append(conf[mask].mean())

        ax.plot([0, 1], [0, 1], "--", color="#94a3b8", label="calibração perfeita")
        ax.bar(centers, accs, width=1.0 / n_bins * 0.9, color="#22c55e",
               edgecolor="white", label="accuracy observada")
        ax.plot(confs, accs, "o-", color="#ef4444", label="confiança vs accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confiança reportada")
        ax.set_ylabel("Accuracy real")
        ece = expected_calibration_error(probs, labels, n_bins)
        ax.set_title(f"{title}\nECE = {ece:.4f}")
        ax.legend(loc="upper left", fontsize=8)

    plt.suptitle(f"Diagrama de fiabilidade — {model_name}", fontsize=13)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Diagrama de fiabilidade: {out}")


def main():
    parser = argparse.ArgumentParser(description="Calibração de confiança — FitoVision")
    parser.add_argument("--data", default=str(BASE_DIR / "data"))
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--binary", action="store_true", default=True)
    parser.add_argument("--weights-dir", default=str(BASE_DIR / "weights"))
    parser.add_argument("--results-dir", default=str(BASE_DIR / "results"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_dir = Path(args.weights_dir)
    results_dir = Path(args.results_dir)
    suffix = "_binary" if args.binary else ""

    weight_path = weights_dir / f"{args.model}{suffix}.pth"
    if not weight_path.exists():
        sys.exit(f"[ERRO] Pesos não encontrados: {weight_path}\n"
                 f"       Treine primeiro: python train.py --data ./data --binary --model {args.model}")

    print(f"\nFitoVision — Calibração de confiança")
    print(f"Modelo      : {args.model}")
    print(f"Dispositivo : {device}")

    # A calibração é feita no conjunto de VALIDAÇÃO — nunca no de teste, que tem
    # de permanecer intocado para a avaliação final.
    val_dir = Path(args.data) / "val"
    if not val_dir.exists():
        sys.exit(f"[ERRO] '{val_dir}' não existe.")

    val_ds = BinaryFolderDataset(val_dir, transform=VAL_TRANSFORMS)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)
    print(f"Validação   : {len(val_ds)} imagens  {val_ds.class_distribution()}")

    model = build_model(args.model, len(BINARY_CLASSES)).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))

    print("\nColectando logits do conjunto de validação...")
    logits, labels = collect_logits(model, val_loader, device)

    probs_before = torch.softmax(logits, dim=1).numpy()
    labels_np = labels.numpy()
    ece_before = expected_calibration_error(probs_before, labels_np)

    print("\nAjustando temperatura (LBFGS sobre a NLL)...")
    T = fit_temperature(logits, labels)

    probs_after = torch.softmax(logits / T, dim=1).numpy()
    ece_after = expected_calibration_error(probs_after, labels_np)

    tau, sel_stats = pick_threshold(probs_after, labels_np)

    mean_conf_before = float(probs_before.max(axis=1).mean())
    mean_conf_after = float(probs_after.max(axis=1).mean())
    acc = float((probs_after.argmax(axis=1) == labels_np).mean())

    print("\n" + "=" * 60)
    print("  RESULTADO DA CALIBRAÇÃO")
    print("=" * 60)
    print(f"  Temperatura óptima (T)     : {T:.4f}")
    if T > 1.05:
        print(f"                               (T > 1 ⇒ o modelo estava EXCESSIVAMENTE")
        print(f"                                confiante; as probabilidades foram achatadas)")
    print()
    print(f"  Accuracy (validação)       : {acc:.4f}")
    print(f"  Confiança média ANTES      : {mean_conf_before:.4f}")
    print(f"  Confiança média DEPOIS     : {mean_conf_after:.4f}")
    print()
    print(f"  ECE antes                  : {ece_before:.4f}")
    print(f"  ECE depois                 : {ece_after:.4f}   "
          f"({100 * (ece_before - ece_after) / ece_before:+.1f}%)" if ece_before > 0 else "")
    print()
    print(f"  Limiar de abstenção (τ)    : {tau:.2f}")
    print(f"    → cobertura              : {sel_stats['coverage']:.1%} das fotos recebem diagnóstico")
    print(f"    → accuracy quando decide  : {sel_stats['selective_accuracy']:.1%}")
    print(f"    → as restantes {1 - sel_stats['coverage']:.1%} são devolvidas como 'inconclusivo'")
    print("=" * 60)

    calib = {
        "model": args.model,
        "temperature": round(T, 6),
        "threshold": round(tau, 4),
        "ece_before": round(ece_before, 6),
        "ece_after": round(ece_after, 6),
        "val_accuracy": round(acc, 6),
        "coverage_at_threshold": round(sel_stats["coverage"], 4),
        "selective_accuracy": round(sel_stats["selective_accuracy"], 4),
    }

    out_json = weights_dir / f"{args.model}{suffix}_calibration.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)
    print(f"\n  Calibração salva: {out_json}")
    print("  (a API carrega este ficheiro automaticamente)")

    plot_reliability(probs_before, probs_after, labels_np, args.model,
                     results_dir / f"calibration_{args.model}.png")


if __name__ == "__main__":
    main()
