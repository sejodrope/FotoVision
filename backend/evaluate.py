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
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset import dataset_from_samples, VAL_TRANSFORMS, make_loaders, BINARY_CLASSES, BinaryFolderDataset
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


def _load_temperature(weights_dir: Path, model_name: str, suffix: str) -> float:
    """Lê a temperatura de calibração, se calibrate.py já correu. 1.0 = sem correcção."""
    path = weights_dir / f"{model_name}{suffix}_calibration.json"
    if not path.exists():
        return 1.0
    try:
        with open(path, encoding="utf-8") as f:
            return float(json.load(f).get("temperature", 1.0))
    except (OSError, ValueError):
        return 1.0


def evaluate_model(
    model_name: str,
    weight_path: Path,
    test_loader: DataLoader,
    device: torch.device,
    results_dir: Path,
    class_names: list[str] | None = None,
    temperature: float = 1.0,
) -> dict:
    if class_names is None:
        class_names = CLASS_NAMES
    num_classes   = len(class_names)
    display_names = [CLASS_LABELS.get(c, c) for c in class_names]

    model = build_model(model_name, num_classes).to(device)
    state = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc=f"  {model_name}", leave=False):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits / temperature, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(probs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc  = accuracy_score(y_true, y_pred)
    # ─── Accuracy balanceada ──────────────────────────────────────────────────
    # Média do recall por classe. Sob desbalanceamento, a accuracy simples premia
    # o modelo que prevê sempre a classe maioritária; a balanceada não. É este o
    # número que deve ir para o TCC.
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro",   zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro",    zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    lat  = measure_latency(model, device)

    # AUC (só faz sentido no binário) e ECE (quão honesta é a confiança reportada)
    auc = None
    if num_classes == 2 and len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_prob[:, 1]))
    ece = _ece(y_prob, y_true)
    mean_conf = float(y_prob.max(axis=1).mean())

    plot_confusion_matrix(cm, display_names, model_name,
                          results_dir / f"cm_{model_name}.png")

    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    per_class = {c: report[c] for c in class_names}

    return {
        "model":            model_name,
        "accuracy":         acc,
        "balanced_accuracy": bacc,
        "f1_macro":         f1,
        "precision_macro":  prec,
        "recall_macro":     rec,
        "roc_auc":          auc,
        "ece":              ece,
        "mean_confidence":  mean_conf,
        "temperature":      temperature,
        "latency_ms":       lat,
        "per_class":        per_class,
        "confusion_matrix": cm.tolist(),
    }


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error — desvio médio entre confiança reportada e acerto real."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(conf)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


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


def plot_per_class_f1(results: list[dict], results_dir: Path, class_names: list[str]):
    """Gráfico de F1 por classe para cada modelo — útil para o capítulo de análise do TCC."""
    display_names = [CLASS_LABELS.get(c, c) for c in class_names]
    x = np.arange(len(class_names))
    width = 0.8 / len(results)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#22c55e", "#3b82f6", "#f97316", "#8b5cf6"]

    for i, r in enumerate(results):
        f1s = [r["per_class"][c]["f1-score"] for c in class_names]
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
    parser.add_argument("--binary",         action="store_true",
                        help="Avalia pipeline binário (healthy vs anomalous)")
    args = parser.parse_args()

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_dir = Path(args.weights_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    active_classes = BINARY_CLASSES if args.binary else CLASS_NAMES
    weights_suffix = "_binary" if args.binary else ""

    print(f"\nFitoVision — Avaliação {'BINÁRIA' if args.binary else 'MULTI-CLASSE'}")
    print(f"Dispositivo : {device}")
    print(f"Classes     : {active_classes}")

    # Monta test_loader
    if args.test_split:
        with open(args.test_split) as f:
            raw = json.load(f)
        test_samples = [(Path(p), int(lbl)) for p, lbl in raw]
        test_ds = dataset_from_samples(active_classes, test_samples, VAL_TRANSFORMS)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers)
        print(f"Test set    : {len(test_samples)} imagens (do train.py)")
    elif args.binary:
        from pathlib import Path as P
        test_ds = BinaryFolderDataset(P(args.data) / "test", transform=VAL_TRANSFORMS)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers)
        print(f"Test set    : {len(test_ds)} imagens")
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

    # Aviso de vazamento: se o split não tem metadados, foi gerado pela versão
    # antiga (shuffle por ficheiro) e estas métricas estarão infladas.
    data_root = Path(args.data) if args.data else Path("./data")
    if not (data_root / "split_metadata.json").exists():
        print(
            "\n  ⚠  AVISO: não encontrei 'data/split_metadata.json'.\n"
            "     Se o split foi gerado pela versão antiga, o conjunto de teste contém\n"
            "     duplicados do treino e TODAS as métricas abaixo estão infladas.\n"
            "     Verifique com:  python audit_leakage.py\n"
        )

    # Avalia cada modelo com pesos disponíveis
    all_results = []
    for model_name in AVAILABLE_MODELS:
        weight_path = weights_dir / f"{model_name}{weights_suffix}.pth"
        if not weight_path.exists():
            print(f"\n[SKIP] {model_name} — {weight_path} não encontrado")
            continue
        temperature = _load_temperature(weights_dir, model_name, weights_suffix)
        print(f"\nAvaliando {model_name}..." +
              (f"  (T={temperature:.3f})" if temperature != 1.0 else "  (sem calibração)"))
        result = evaluate_model(model_name, weight_path, test_loader, device,
                                results_dir, class_names=active_classes,
                                temperature=temperature)
        all_results.append(result)
        print(
            f"  accuracy={result['accuracy']:.4f}  "
            f"bal_acc={result['balanced_accuracy']:.4f}  "
            f"f1={result['f1_macro']:.4f}  "
            f"ECE={result['ece']:.4f}  "
            f"latência={result['latency_ms']:.1f}ms"
        )

    if not all_results:
        print(f"\nNenhum modelo treinado encontrado em {weights_dir}/")
        print("Execute primeiro: python train.py --data ./data --model all")
        return

    # Tabela comparativa. 'Bal. Acc' e 'F1' são os números honestos sob
    # desbalanceamento; 'ECE' diz se a confiança reportada é de confiar.
    rows = [{
        "Modelo":         r["model"],
        "Accuracy":       f"{r['accuracy']:.4f}",
        "Bal. Acc":       f"{r['balanced_accuracy']:.4f}",
        "F1 (macro)":     f"{r['f1_macro']:.4f}",
        "Precision":      f"{r['precision_macro']:.4f}",
        "Recall":         f"{r['recall_macro']:.4f}",
        "AUC":            f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "—",
        "ECE":            f"{r['ece']:.4f}",
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
        plot_per_class_f1(all_results, results_dir, active_classes)

    print(f"\n  Confusion matrices por modelo em: {results_dir}/cm_*.png")

    # Nota metodológica para o TCC
    print("\n" + "-" * 60)
    print("  Nota: 'Bal. Acc' (accuracy balanceada) é a métrica a reportar sob")
    print("  desbalanceamento de classes. 'ECE' mede se a confiança é honesta:")
    print("  ECE alto ⇒ o modelo diz '99%' quando na verdade acerta bem menos.")
    print("  Se ECE > 0.05, execute: python calibrate.py --data ./data --binary")
    print("-" * 60)


if __name__ == "__main__":
    main()
