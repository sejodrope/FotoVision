#!/usr/bin/env python
"""
Avalia todos os modelos treinados e gera métricas para o TCC.

Uso:
    # Usando o test split salvo pelo train.py (recomendado — garante mesmo conjunto)
    python evaluate.py --test-split ./logs/test_split.json

    # Ou apontando para o dataset (faz novo split com o mesmo seed)
    python evaluate.py --data ./data --seed 42

Saída em ./results/:
    cm_<model>.png            — confusion matrix por modelo
    model_comparison.png      — gráfico comparativo de accuracy, F1 e latência
    metrics_comparison.csv    — tabela para o TCC
    metrics_full.json         — métricas detalhadas por classe
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset import PlantDataset, VAL_TRANSFORMS, make_loaders
from app.config import CLASS_NAMES, AVAILABLE_MODELS, CLASS_LABELS


def build_model(name: str, num_classes: int) -> nn.Module:
    """Constrói o modelo sem pesos pré-treinados (os pesos do ficheiro .pth serão carregados)."""
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


def measure_latency(model: nn.Module, device: torch.device, n_runs: int = 100) -> float:
    """Latência média em ms para uma imagem de 224×224 (CPU ou GPU)."""
    dummy = torch.randn(1, 3, 224, 224).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(10):  # warmup
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return (elapsed / n_runs) * 1000


def plot_confusion_matrix(
    cm: np.ndarray,
    display_names: list[str],
    model_name: str,
    save_path: Path,
):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=display_names, yticklabels=display_names, ax=ax,
    )
    ax.set_xlabel("Predito", fontsize=11)
    ax.set_ylabel("Real",    fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate_model(
    model_name: str,
    weight_path: Path,
    test_loader: DataLoader,
    device: torch.device,
    results_dir: Path,
) -> dict:
    model = build_model(model_name, len(CLASS_NAMES)).to(device)
    state = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc=f"  {model_name}", leave=False):
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro",   zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro",    zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    lat  = measure_latency(model, device)

    display_names = [CLASS_LABELS[c] for c in CLASS_NAMES]
    plot_confusion_matrix(cm, display_names, model_name,
                          results_dir / f"cm_{model_name}.png")

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    per_class = {c: report[c] for c in CLASS_NAMES}

    return {
        "model":            model_name,
        "accuracy":         acc,
        "f1_macro":         f1,
        "precision_macro":  prec,
        "recall_macro":     rec,
        "latency_ms":       lat,
        "per_class":        per_class,
        "confusion_matrix": cm.tolist(),
    }


def plot_comparison(results: list[dict], results_dir: Path):
    names = [r["model"] for r in results]
    accs  = [r["accuracy"]   for r in results]
    f1s   = [r["f1_macro"]   for r in results]
    lats  = [r["latency_ms"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def bar_plot(ax, values, title, ylabel, color, fmt=".3f"):
        bars = ax.bar(names, values, color=color, edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_ylabel(ylabel, fontsize=10)
        if max(values) <= 1.0:
            ax.set_ylim(0, 1.05)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:{fmt}}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)

    bar_plot(axes[0], accs, "Accuracy (test set)",    "Accuracy",  "#22c55e")
    bar_plot(axes[1], f1s,  "F1 Score Macro (test set)", "F1",    "#3b82f6")
    bar_plot(axes[2], lats, "Latência média (ms/img)", "ms",      "#f97316", fmt=".1f")

    plt.suptitle("FitoVision — Comparação de Modelos", fontsize=14, y=1.02)
    plt.tight_layout()
    out = results_dir / "model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico comparativo : {out}")


def plot_per_class_f1(results: list[dict], results_dir: Path):
    """Gráfico de F1 por classe para cada modelo — útil para o capítulo de análise do TCC."""
    display_names = [CLASS_LABELS[c] for c in CLASS_NAMES]
    x = np.arange(len(CLASS_NAMES))
    width = 0.8 / len(results)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#22c55e", "#3b82f6", "#f97316", "#8b5cf6"]

    for i, r in enumerate(results):
        f1s = [r["per_class"][c]["f1-score"] for c in CLASS_NAMES]
        offset = (i - len(results) / 2 + 0.5) * width
        ax.bar(x + offset, f1s, width=width, label=r["model"],
               color=colors[i % len(colors)], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("F1 por Classe — Comparação de Modelos", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out = results_dir / "f1_per_class.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  F1 por classe       : {out}")


def main():
    parser = argparse.ArgumentParser(description="Avaliação FitoVision")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data",       help="Diretório do dataset (novo split)")
    group.add_argument("--test-split", help="JSON do test split salvo pelo train.py")

    parser.add_argument("--weights-dir",    default="./weights")
    parser.add_argument("--results-dir",    default="./results")
    parser.add_argument("--batch-size",     type=int,   default=32)
    parser.add_argument("--workers",        type=int,   default=0)
    parser.add_argument("--mapping",        default=None)
    parser.add_argument("--val-split",      type=float, default=0.15)
    parser.add_argument("--test-split-frac",type=float, default=0.15, dest="test_frac")
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_dir = Path(args.weights_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    print(f"\nFitoVision — Avaliação")
    print(f"Dispositivo : {device}")

    # Monta test_loader
    if args.test_split:
        with open(args.test_split) as f:
            raw = json.load(f)
        test_samples = [(Path(p), int(lbl)) for p, lbl in raw]
        test_ds = PlantDataset.__new__(PlantDataset)
        test_ds.class_names = CLASS_NAMES
        test_ds.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
        test_ds.transform = VAL_TRANSFORMS
        test_ds.samples = test_samples
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers)
        print(f"Test set    : {len(test_samples)} imagens (do train.py)")
    else:
        folder_map = None
        if args.mapping:
            with open(args.mapping) as f:
                folder_map = json.load(f)
        _, _, test_loader, _ = make_loaders(
            data_dir=args.data,
            class_names=CLASS_NAMES,
            batch_size=args.batch_size,
            val_frac=args.val_split,
            test_frac=args.test_frac,
            num_workers=args.workers,
            folder_class_map=folder_map,
            seed=args.seed,
        )

    # Avalia cada modelo com pesos disponíveis
    all_results = []
    for model_name in AVAILABLE_MODELS:
        weight_path = weights_dir / f"{model_name}.pth"
        if not weight_path.exists():
            print(f"\n[SKIP] {model_name} — {weight_path} não encontrado")
            continue
        print(f"\nAvaliando {model_name}...")
        result = evaluate_model(model_name, weight_path, test_loader, device, results_dir)
        all_results.append(result)
        print(
            f"  accuracy={result['accuracy']:.4f}  "
            f"f1={result['f1_macro']:.4f}  "
            f"latência={result['latency_ms']:.1f}ms"
        )

    if not all_results:
        print(f"\nNenhum modelo treinado encontrado em {weights_dir}/")
        print("Execute primeiro: python train.py --data ./data --model all")
        return

    # Tabela comparativa
    rows = [{
        "Modelo":         r["model"],
        "Accuracy":       f"{r['accuracy']:.4f}",
        "F1 (macro)":     f"{r['f1_macro']:.4f}",
        "Precision":      f"{r['precision_macro']:.4f}",
        "Recall":         f"{r['recall_macro']:.4f}",
        "Latência (ms)":  f"{r['latency_ms']:.1f}",
    } for r in all_results]

    df = pd.DataFrame(rows)
    print("\n\n=== Comparação de Modelos ===")
    print(df.to_string(index=False))

    csv_path = results_dir / "metrics_comparison.csv"
    df.to_csv(csv_path, index=False)

    json_path = results_dir / "metrics_full.json"
    with open(json_path, "w") as f:
        # confusion_matrix como lista de listas (já convertido no evaluate_model)
        json.dump(all_results, f, indent=2)

    print(f"\nResultados guardados em {results_dir}/:")
    print(f"  CSV        : {csv_path.name}")
    print(f"  JSON       : {json_path.name}")

    plot_comparison(all_results, results_dir)
    if len(all_results) > 1:
        plot_per_class_f1(all_results, results_dir)

    print(f"\n  Confusion matrices por modelo em: {results_dir}/cm_*.png")


if __name__ == "__main__":
    main()
