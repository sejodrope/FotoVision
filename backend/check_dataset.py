#!/usr/bin/env python
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Diagnóstico do dataset binário do FitoVision.

Verifica a estrutura data/train|val|test/healthy|anomalous/ e emite:
  ✅  tudo OK para treino
  ⚠   alertas de classes com poucos exemplos ou desbalanceamento

Também salva um grid visual 4×4 de amostras em data/sample_grid.png.

Uso:
    python check_dataset.py
    python check_dataset.py --data ./data --no-grid
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

SPLITS        = ("train", "val", "test")
LABELS        = ("healthy", "anomalous")
IMAGE_EXT     = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MIN_PER_CLASS = 200
MAX_RATIO     = 3.0


# ─── Contagem ────────────────────────────────────────────────────────────────

def count_images(data_dir: Path) -> dict[str, dict[str, int]]:
    """Conta imagens por split e label. Retorna {split: {label: count}}."""
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        counts[split] = {}
        for label in LABELS:
            folder = data_dir / split / label
            if folder.is_dir():
                n = sum(1 for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXT)
            else:
                n = 0
            counts[split][label] = n
    return counts


# ─── Tabela de contagem ───────────────────────────────────────────────────────

def print_table(counts: dict[str, dict[str, int]]):
    print("\n" + "=" * 58)
    print("  DISTRIBUIÇÃO DO DATASET — FitoVision (binário)")
    print("=" * 58)
    print(f"  {'Split':<8}  {'healthy':>8}  {'anomalous':>10}  {'total':>7}  {'ratio':>7}")
    print("  " + "-" * 46)

    grand = defaultdict(int)
    for split in SPLITS:
        h = counts[split]["healthy"]
        a = counts[split]["anomalous"]
        t = h + a
        ratio_str = f"{max(h,a)/min(h,a):.1f}:1" if h > 0 and a > 0 else "—"
        grand["healthy"]   += h
        grand["anomalous"] += a
        missing = ""
        if h == 0 or a == 0:
            missing = "  ⚠"
        print(f"  {split:<8}  {h:>8}  {a:>10}  {t:>7}  {ratio_str:>7}{missing}")

    print("  " + "-" * 46)
    h, a = grand["healthy"], grand["anomalous"]
    ratio_str = f"{max(h,a)/min(h,a):.1f}:1" if h > 0 and a > 0 else "—"
    print(f"  {'TOTAL':<8}  {h:>8}  {a:>10}  {h+a:>7}  {ratio_str:>7}")
    print("=" * 58)


# ─── Alertas ──────────────────────────────────────────────────────────────────

def emit_alerts(counts: dict[str, dict[str, int]]) -> bool:
    """Emite alertas e retorna True se tudo OK."""
    issues: list[str] = []

    # Alerta por split+label com < MIN_PER_CLASS
    for split in SPLITS:
        for label in LABELS:
            n = counts[split][label]
            if n == 0:
                issues.append(f"  ⚠  {split}/{label}: VAZIO — adicione imagens antes de treinar.")
            elif n < MIN_PER_CLASS:
                issues.append(
                    f"  ⚠  {split}/{label}: apenas {n} imagens "
                    f"(mínimo recomendado: {MIN_PER_CLASS})."
                )

    # Alerta de desbalanceamento no train
    h_train = counts["train"]["healthy"]
    a_train = counts["train"]["anomalous"]
    if h_train > 0 and a_train > 0:
        ratio = max(h_train, a_train) / min(h_train, a_train)
        if ratio > MAX_RATIO:
            dominant  = "healthy" if h_train > a_train else "anomalous"
            minority  = "anomalous" if h_train > a_train else "healthy"
            issues.append(
                f"  ⚠  Desbalanceamento no treino: {ratio:.1f}:1 "
                f"('{dominant}' domina '{minority}')."
            )

    print()
    if issues:
        for issue in issues:
            print(issue)
        print()
        _print_suggestions(counts)
        return False
    else:
        print("  ✅ Dataset OK para treino — distribuição e volume adequados.")
        return True


def _print_suggestions(counts: dict[str, dict[str, int]]):
    print("  Sugestões:")
    h_train = counts["train"]["healthy"]
    a_train = counts["train"]["anomalous"]

    for label in LABELS:
        total_label = sum(counts[s][label] for s in SPLITS)
        if total_label < MIN_PER_CLASS:
            print(f"    → Busque mais imagens '{label}' no Kaggle/Roboflow e adicione em")
            print(f"      KAGGLE_DATASETS ou ROBOFLOW_DATASETS no download_datasets.py")

    if h_train > 0 and a_train > 0:
        ratio = max(h_train, a_train) / min(h_train, a_train)
        if ratio > MAX_RATIO:
            minority = "anomalous" if h_train > a_train else "healthy"
            print(f"    → Considere augmentation extra para '{minority}'")
            print(f"      ou use WeightedRandomSampler no treino (já suportado pelo dataset.py).")


# ─── Grid visual ─────────────────────────────────────────────────────────────

def save_sample_grid(data_dir: Path, output_path: Path, seed: int = 42):
    """
    Salva um grid 4×4 com amostras aleatórias:
        linhas 0-1: healthy  (2 linhas × 4 colunas = 8 imagens)
        linhas 2-3: anomalous (2 linhas × 4 colunas = 8 imagens)
    """
    try:
        from PIL import Image
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"  [grid] Dependência ausente: {e} — grid não gerado.")
        return

    rng   = random.Random(seed)
    cols  = 4
    rows_per_label = 2
    n_per_label = cols * rows_per_label  # 8 imagens por label

    label_images: dict[str, list[Path]] = {}
    for label in LABELS:
        candidates: list[Path] = []
        for split in SPLITS:
            folder = data_dir / split / label
            if folder.is_dir():
                candidates.extend(
                    f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXT
                )
        rng.shuffle(candidates)
        label_images[label] = candidates[:n_per_label]

    total_rows = len(LABELS) * rows_per_label  # 4 rows
    fig, axes = plt.subplots(total_rows, cols, figsize=(cols * 3, total_rows * 3))
    fig.suptitle("FitoVision — Amostras do Dataset (healthy / anomalous)", fontsize=13)

    for row_idx in range(total_rows):
        label = LABELS[row_idx // rows_per_label]
        img_offset = (row_idx % rows_per_label) * cols
        imgs = label_images[label]

        for col_idx in range(cols):
            ax = axes[row_idx][col_idx]
            ax.axis("off")
            img_idx = img_offset + col_idx
            if img_idx >= len(imgs):
                ax.set_title(f"{label} (sem img)", fontsize=7)
                continue
            try:
                img = Image.open(imgs[img_idx]).convert("RGB").resize((224, 224))
                ax.imshow(img)
                split_name = imgs[img_idx].parts[-3] if len(imgs[img_idx].parts) >= 3 else ""
                ax.set_title(f"{label}\n[{split_name}]", fontsize=7)
            except Exception:
                ax.set_title(f"{label} (erro)", fontsize=7)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Grid salvo: {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Diagnóstico do dataset FitoVision")
    parser.add_argument("--data", default=str(DATA_DIR),
                        help="Diretório raiz do dataset (default: ./data)")
    parser.add_argument("--no-grid", action="store_true",
                        help="Não gera o grid visual")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data)

    if not data_dir.exists():
        print(f"[ERRO] Diretório '{data_dir}' não encontrado.")
        print("       Execute primeiro: python download_datasets.py")
        return

    counts = count_images(data_dir)
    print_table(counts)
    ok = emit_alerts(counts)

    if not args.no_grid:
        grid_path = data_dir / "sample_grid.png"
        print(f"\n  Gerando grid de amostras → {grid_path}")
        save_sample_grid(data_dir, grid_path, seed=args.seed)

    if ok:
        print("\n  Sequência de treino:")
        print(f"    python train.py --data {data_dir} --model mobilenet_v2")


if __name__ == "__main__":
    main()
