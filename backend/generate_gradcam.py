#!/usr/bin/env python3
"""
generate_gradcam.py -- Grad-CAM visualization for all 3 FitoVision binary models.

Run from the backend/ directory:
    .venv/Scripts/python.exe generate_gradcam.py
"""
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
WEIGHTS_DIR = BASE_DIR / "weights"
LOGS_DIR    = BASE_DIR / "logs"
OUT_DIR     = BASE_DIR / "results" / "gradcam"

SEED           = 42
N_TP_PER_CLASS = 4   # 4 TP healthy + 4 TP anomalous per model
N_FN           = 2   # false negatives per model

MODELS_CFG = [
    {
        "name":      "efficientnet_b0",
        "label":     "EfficientNet-B0",
        "weights":   "efficientnet_b0_binary.pth",
        # features[-1] e o 1x1 conv de projecao para 1280 canais: o CAM nessa
        # camada colapsa num unico ponto do grid 7x7 (mapa degenerado, sempre
        # 1/49 celulas acima de 0.5). O ultimo bloco MBConv (features[-2])
        # produz mapas espacialmente distribuidos e interpretaveis.
        "target_fn": lambda m: [m.features[-2]],
    },
    {
        "name":      "mobilenet_v2",
        "label":     "MobileNetV2",
        "weights":   "mobilenet_v2_binary.pth",
        "target_fn": lambda m: [m.features[-1]],
    },
    {
        "name":      "resnet50",
        "label":     "ResNet-50",
        "weights":   "resnet50_binary.pth",
        "target_fn": lambda m: [m.layer4[-1]],
    },
]

_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Model builders ───────────────────────────────────────────────────────────
def _build(name: str) -> nn.Module:
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.last_channel, 2)
    elif name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
    else:
        raise ValueError(f"Modelo desconhecido: {name}")
    return m


def load_model(cfg: dict) -> nn.Module:
    m = _build(cfg["name"])
    path = WEIGHTS_DIR / cfg["weights"]
    if not path.exists():
        print(f"  ERRO: pesos não encontrados em {path}", file=sys.stderr)
        sys.exit(1)
    print(f"  Carregando {cfg['label']} de {path.name}...", end=" ", flush=True)
    state = torch.load(path, map_location=device, weights_only=True)
    m.load_state_dict(state)
    m.to(device).eval()
    print("OK")
    return m


# ─── Inference + Grad-CAM ─────────────────────────────────────────────────────
def predict(model: nn.Module, img: Image.Image) -> tuple[int, float]:
    t = _tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(t), 1)[0]
    idx = int(probs.argmax())
    return idx, float(probs[idx])


def make_overlay(cam: GradCAM, img: Image.Image, class_idx: int) -> np.ndarray:
    t = _tf(img).unsqueeze(0).to(device)
    grayscale = cam(input_tensor=t, targets=[ClassifierOutputTarget(class_idx)])[0]
    rgb = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0
    return show_cam_on_image(rgb, grayscale, use_rgb=True)  # uint8 RGB


def side_by_side(img: Image.Image, overlay: np.ndarray) -> np.ndarray:
    orig = np.array(img.resize((224, 224)))
    return np.concatenate([orig, overlay], axis=1)  # 224 × 448


# ─── Sampling ─────────────────────────────────────────────────────────────────
def try_load(rel_path: str) -> Image.Image | None:
    full = BASE_DIR / rel_path
    if not full.exists():
        return None
    try:
        return Image.open(full).convert("RGB")
    except Exception:
        return None


def collect_samples(cfg: dict, model: nn.Module, test_data: list) -> dict:
    rng = random.Random(SEED)
    target_layers = cfg["target_fn"](model)

    healthy_paths   = [p for p, lbl in test_data if lbl == 0]
    anomalous_paths = [p for p, lbl in test_data if lbl == 1]
    rng.shuffle(healthy_paths)
    rng.shuffle(anomalous_paths)

    tp_healthy   = []
    tp_anomalous = []
    fn_samples   = []

    print(f"  Selecionando TP healthy...", end=" ", flush=True)
    checked = 0
    with GradCAM(model=model, target_layers=target_layers) as cam:
        for p in healthy_paths:
            if len(tp_healthy) >= N_TP_PER_CLASS:
                break
            img = try_load(p)
            if img is None:
                continue
            checked += 1
            pred, conf = predict(model, img)
            if pred == 0:
                ov = make_overlay(cam, img, pred)
                tp_healthy.append({"path": p, "img": img, "overlay": ov, "conf": conf})

        print(f"{len(tp_healthy)}/{N_TP_PER_CLASS} (verificadas: {checked})")

        print(f"  Selecionando TP anomalous + FN...", end=" ", flush=True)
        checked = 0
        for p in anomalous_paths[:600]:
            if len(tp_anomalous) >= N_TP_PER_CLASS and len(fn_samples) >= N_FN:
                break
            img = try_load(p)
            if img is None:
                continue
            checked += 1
            pred, conf = predict(model, img)
            if pred == 1 and len(tp_anomalous) < N_TP_PER_CLASS:
                ov = make_overlay(cam, img, pred)
                tp_anomalous.append({"path": p, "img": img, "overlay": ov, "conf": conf})
            elif pred == 0 and len(fn_samples) < N_FN:
                ov = make_overlay(cam, img, pred)
                fn_samples.append({"path": p, "img": img, "overlay": ov, "conf": conf})

    print(f"TP={len(tp_anomalous)}/{N_TP_PER_CLASS} FN={len(fn_samples)}/{N_FN} (verificadas: {checked})")

    return {"tp_healthy": tp_healthy, "tp_anomalous": tp_anomalous, "fn": fn_samples}


# ─── Individual saves ─────────────────────────────────────────────────────────
def save_model_images(cfg: dict, samples: dict) -> int:
    model_dir = OUT_DIR / cfg["name"]
    model_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, s in enumerate(samples["tp_healthy"], 1):
        Image.fromarray(side_by_side(s["img"], s["overlay"])).save(
            model_dir / f"healthy_{i:02d}.png"
        )
        saved += 1
    for i, s in enumerate(samples["tp_anomalous"], 1):
        Image.fromarray(side_by_side(s["img"], s["overlay"])).save(
            model_dir / f"anomalous_{i:02d}.png"
        )
        saved += 1
    for i, s in enumerate(samples["fn"], 1):
        Image.fromarray(side_by_side(s["img"], s["overlay"])).save(
            model_dir / f"fn_{i:02d}.png"
        )
        saved += 1
    return saved


# ─── Comparison grids ─────────────────────────────────────────────────────────
DARK_BG   = "#0f1117"
CELL_W    = 4.2   # inches per pair (original + cam)
CELL_H    = 2.3   # inches per row


def _draw_grid(title: str, rows_data: list, n_cols: int, out_path: Path):
    n_rows = len(rows_data)
    fig_w  = n_cols * CELL_W + 0.8   # +0.8 for row labels
    fig_h  = n_rows * CELL_H + 0.5   # +0.5 for title
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(title, fontsize=11, fontweight="bold", color="white", y=0.98)

    # Ensure axes is always 2D
    if n_rows == 1 and n_cols * 2 == 2:
        axes = np.array([[axes[0], axes[1]]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]

    for ri, row in enumerate(rows_data):
        samples = row["samples"]
        for ci in range(n_cols):
            ax_o = axes[ri, ci * 2]
            ax_c = axes[ri, ci * 2 + 1]
            for ax in (ax_o, ax_c):
                ax.axis("off")
                ax.set_facecolor(DARK_BG)
            if ri == 0:
                ax_o.set_title(f"Original {ci+1}", fontsize=7, color="#aaa", pad=2)
                ax_c.set_title(f"Grad-CAM {ci+1}", fontsize=7, color="#aaa", pad=2)
            if ci < len(samples):
                s = samples[ci]
                ax_o.imshow(np.array(s["img"].resize((224, 224))))
                ax_c.imshow(s["overlay"])
                ax_c.text(
                    3, 218, f"{s['conf']:.1%}",
                    fontsize=6.5, color="white", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.65),
                )
            else:
                ax_o.text(0.5, 0.5, "—", ha="center", va="center",
                          color="#555", fontsize=14, transform=ax_o.transAxes)

        axes[ri, 0].set_ylabel(
            row["label"], fontsize=9, fontweight="bold",
            color="white", rotation=90, labelpad=6,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Salvo: {out_path.name}  ({out_path.stat().st_size // 1024} KB)")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device : {device}")
    print(f"Saída  : {OUT_DIR.resolve()}\n")

    print("Lendo test_split_binary.json...", end=" ", flush=True)
    with open(LOGS_DIR / "test_split_binary.json") as f:
        test_data = json.load(f)
    n_healthy   = sum(1 for _, lbl in test_data if lbl == 0)
    n_anomalous = sum(1 for _, lbl in test_data if lbl == 1)
    print(f"{len(test_data)} imagens  (healthy={n_healthy}, anomalous={n_anomalous})\n")

    all_results: dict = {}
    t_global = time.time()

    for cfg in MODELS_CFG:
        print(f"{'-'*52}")
        print(f"Modelo: {cfg['label']}")
        t0    = time.time()
        model = load_model(cfg)
        samps = collect_samples(cfg, model, test_data)
        n_saved = save_model_images(cfg, samps)
        elapsed = time.time() - t0
        n_total = len(samps["tp_healthy"]) + len(samps["tp_anomalous"]) + len(samps["fn"])
        print(f"  {n_total} mapas gerados ({n_saved} PNGs) em {elapsed:.1f}s")
        all_results[cfg["name"]] = {"cfg": cfg, "samples": samps}
        del model
        torch.cuda.empty_cache()

    # ── Comparison grids ──────────────────────────────────────────────────────
    print(f"\n{'-'*52}")
    print("Gerando grids comparativos...\n")

    rows_healthy = [
        {"label": all_results[c["name"]]["cfg"]["label"],
         "samples": all_results[c["name"]]["samples"]["tp_healthy"]}
        for c in MODELS_CFG
    ]
    _draw_grid(
        "Grad-CAM — Folhas Saudáveis (classificação correta) · 3 Modelos × 4 imagens",
        rows_healthy, N_TP_PER_CLASS,
        OUT_DIR / "comparison_healthy.png",
    )

    rows_anomalous = [
        {"label": all_results[c["name"]]["cfg"]["label"],
         "samples": all_results[c["name"]]["samples"]["tp_anomalous"]}
        for c in MODELS_CFG
    ]
    _draw_grid(
        "Grad-CAM — Folhas Anômalas (classificação correta) · 3 Modelos × 4 imagens",
        rows_anomalous, N_TP_PER_CLASS,
        OUT_DIR / "comparison_anomalous.png",
    )

    rows_fn = [
        {"label": all_results[c["name"]]["cfg"]["label"],
         "samples": all_results[c["name"]]["samples"]["fn"]}
        for c in MODELS_CFG
    ]
    _draw_grid(
        "Grad-CAM — Falsos Negativos: Anômala classificada como Saudável · 3 Modelos",
        rows_fn, N_FN,
        OUT_DIR / "false_negatives.png",
    )

    # ── Report ────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_global
    print(f"\n{'='*52}")
    print(f"Concluído em {total_elapsed:.1f}s\n")
    print("Mapas Grad-CAM gerados por modelo:")
    grand_total = 0
    for cfg in MODELS_CFG:
        s = all_results[cfg["name"]]["samples"]
        n = len(s["tp_healthy"]) + len(s["tp_anomalous"]) + len(s["fn"])
        grand_total += n
        print(f"  {cfg['label']:20s} {n:2d}  "
              f"(healthy={len(s['tp_healthy'])}, "
              f"anomalous={len(s['tp_anomalous'])}, "
              f"FN={len(s['fn'])})")
    print(f"  {'TOTAL':20s} {grand_total:2d}\n")

    print("Ficheiros de comparação:")
    for name in ("comparison_healthy.png", "comparison_anomalous.png", "false_negatives.png"):
        p = OUT_DIR / name
        print(f"  {p.resolve()}")

    print("\nImagens mais recomendadas para o TCC/slides:")
    print("  1. comparison_anomalous.png — mostra onde cada modelo detecta a anomalia")
    print("  2. false_negatives.png      — mostra onde o modelo 'falha' (análise de erro)")
    print("  3. comparison_healthy.png   — confirma que o modelo foca no limbo foliar")


if __name__ == "__main__":
    main()
