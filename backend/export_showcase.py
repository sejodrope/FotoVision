"""
Exporta imagens showcase (saudavel + anomala) com GradCAM para a apresentação.
Uso: python export_showcase.py
Saída: backend/showcase/
"""
import sys
import shutil
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from app.ml.inference import load_binary_model_by_name, predict_binary_with_model
from app.ml.preprocessing import preprocess_image
from app.ml.gradcam import generate_gradcam
from PIL import Image
import base64
import io

OUT_DIR = Path(__file__).parent / "showcase"
OUT_DIR.mkdir(exist_ok=True)


def save_gradcam(b64: str, path: Path):
    data = base64.b64decode(b64.split(",")[1])
    Image.open(io.BytesIO(data)).save(path)


def find_best(folder: Path, model, expected_label: str, n: int = 5):
    results = []
    for f in folder.iterdir():
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        try:
            img_bytes = f.read_bytes()
            tensor, orig = preprocess_image(img_bytes)
            r = predict_binary_with_model(model, tensor)
            if r["label"] == expected_label and r["confidence"] > 0.90:
                results.append((r["confidence"], f, tensor, orig))
        except Exception:
            continue
        if len(results) >= n * 3:
            break
    results.sort(reverse=True)
    return results[:n]


def main():
    model = load_binary_model_by_name("efficientnet_b0")

    data_root = Path(__file__).parent / "data" / "test"

    # Alface saudável (shuvokumarbasak tem as melhores imagens de alface saudável)
    print("Procurando imagens de alface saudável...")
    healthy_folder = data_root / "healthy"
    healthy_cands = [f for f in healthy_folder.iterdir() if "shuvoku" in f.name.lower()]
    healthy_top = find_best_from_list(healthy_cands[:500], model, "healthy", n=5)

    print("Procurando imagens de alface anômala...")
    anom_folder = data_root / "anomalous"
    anom_cands = [f for f in anom_folder.iterdir()
                  if "ashishjstar" in f.name.lower() and "Healthy" not in f.name]
    anom_top = find_best_from_list(anom_cands[:200], model, "anomalous", n=5)

    print(f"\nExportando {len(healthy_top)} saudáveis + {len(anom_top)} anômalas...")

    for i, (conf, f, tensor, orig) in enumerate(healthy_top):
        dest = OUT_DIR / f"saudavel_{i+1:02d}_conf{conf:.0%}.jpg"
        shutil.copy2(f, dest)
        pred_idx = 0  # healthy = index 0
        b64 = generate_gradcam(model, "efficientnet_b0", tensor, orig, pred_idx)
        save_gradcam(b64, OUT_DIR / f"saudavel_{i+1:02d}_gradcam.png")
        print(f"  saudavel_{i+1:02d} | conf={conf:.1%}")

    for i, (conf, f, tensor, orig) in enumerate(anom_top):
        dest = OUT_DIR / f"anomala_{i+1:02d}_conf{conf:.0%}.jpg"
        shutil.copy2(f, dest)
        pred_idx = 1  # anomalous = index 1
        b64 = generate_gradcam(model, "efficientnet_b0", tensor, orig, pred_idx)
        save_gradcam(b64, OUT_DIR / f"anomala_{i+1:02d}_gradcam.png")
        print(f"  anomala_{i+1:02d} | conf={conf:.1%}")

    print(f"\nDone! Imagens em: {OUT_DIR}")


def find_best_from_list(files, model, expected_label: str, n: int = 5):
    results = []
    for f in files:
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        try:
            img_bytes = f.read_bytes()
            tensor, orig = preprocess_image(img_bytes)
            r = predict_binary_with_model(model, tensor)
            if r["label"] == expected_label and r["confidence"] > 0.90:
                results.append((r["confidence"], f, tensor, orig))
        except Exception:
            continue
        if len(results) >= n * 3:
            break
    results.sort(reverse=True)
    return results[:n]


if __name__ == "__main__":
    main()
