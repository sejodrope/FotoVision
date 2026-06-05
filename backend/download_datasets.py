#!/usr/bin/env python
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Pipeline de download e organização de datasets para o FitoVision.

Culturas-alvo: alface, rúcula, espinafre, acelga, couve (hortaliças folhosas)
Classificação: BINÁRIA — healthy (saudável) vs anomalous (anômalo)

Uso:
    python download_datasets.py                # download completo + organização
    python download_datasets.py --skip-download  # só organiza raw/ existente
    python download_datasets.py --dry-run       # mostra o que faria sem executar

Dependências: pip install kaggle roboflow
Variáveis de ambiente (no .env ou no sistema):
    KAGGLE_USERNAME   — seu usuário Kaggle
    KAGGLE_KEY        — sua chave Kaggle (kaggle.json → "key")
    ROBOFLOW_API_KEY  — chave privada Roboflow (opcional)
"""

import argparse
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ─── CONFIGURAÇÃO: Kaggle ────────────────────────────────────────────────────

# Adicione aqui os slugs "usuario/nome-dataset" encontrados no Kaggle
KAGGLE_DATASETS = [
    "abdallahalidev/plantvillage-dataset",
    # Exemplos de datasets de hortaliças encontrados no Kaggle:
    # "ashishmotwani/lettuce",
    # "smaranjitghose/plant-disease-classification-merged-dataset",
    # "vipoooool/new-plant-diseases-dataset",
    # "ajinkyakadam2003/plant-diseases-comprehensive-dataset",
]

# Subpastas do PlantVillage relevantes para o pipeline binário.
# PlantVillage não tem alface/rúcula, mas tem imagens folha que servem como
# proxy visual para treino inicial (healthy vs. doença genérica de folha).
PV_KEEP_FOLDERS = {
    "Tomato___healthy",
    "Pepper,_bell___healthy",
    "Potato___healthy",
    "Corn_(maize)___healthy",
    "Tomato___Late_blight",
    "Tomato___Early_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Potato___Late_blight",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Cherry_(including_sour)___Powdery_mildew",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
}

# ─── CONFIGURAÇÃO: Roboflow ──────────────────────────────────────────────────

# Formato: (workspace_id, project_id, numero_da_versao)
# Encontre em universe.roboflow.com → pesquise "lettuce disease classification"
ROBOFLOW_DATASETS: list[tuple[str, str, int]] = [
    # Exemplos — substitua pelos slugs que encontrar:
    # ("workspace_id", "lettuce-disease-classification", 1),
    # ("workspace_id", "leafy-vegetable-disease", 2),
]

# ─── CONFIGURAÇÃO: Organização binária ───────────────────────────────────────

HEALTHY_KEYWORDS = ["healthy", "saudavel", "saudável", "normal", "good"]
ANOMALOUS_KEYWORDS = [
    "disease", "blight", "mildew", "spot", "rot", "virus",
    "anomaly", "doenca", "doença", "bacterial", "fungal",
    "downy", "powdery", "mosaic", "chlorosis", "scorch",
    "rust", "leaf_curl", "wilt", "necrosis", "angular",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SPLIT_SEED  = 42
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
# test_frac  = 1.0 - TRAIN_FRAC - VAL_FRAC = 0.15

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR  = DATA_DIR / "raw"
STAGING  = DATA_DIR / "staging"


# ─── Kaggle ──────────────────────────────────────────────────────────────────

def _kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[ERRO] 'kaggle' não instalado. Execute: pip install kaggle")
        sys.exit(1)

    username = os.getenv("KAGGLE_USERNAME", "").strip()
    key      = os.getenv("KAGGLE_KEY", "").strip()

    if not username or not key:
        print("[ERRO] KAGGLE_USERNAME e KAGGLE_KEY devem estar no .env")
        print("       Veja .env.example para o formato correto.")
        sys.exit(1)

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"]      = key

    api = KaggleApi()
    api.authenticate()
    return api


def download_from_kaggle(dry_run: bool = False):
    if not KAGGLE_DATASETS:
        print("[Kaggle] Nenhum dataset em KAGGLE_DATASETS — pulando.")
        return

    api = _kaggle_api()

    for slug in KAGGLE_DATASETS:
        dest = RAW_DIR / slug.replace("/", "_")
        if dest.exists() and any(dest.iterdir()):
            print(f"[Kaggle] '{slug}' já existe em {dest} — ignorando.")
            continue
        print(f"[Kaggle] Baixando '{slug}' → {dest} ...")
        if dry_run:
            print(f"[Kaggle] [DRY-RUN] seria baixado para {dest}")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(slug, path=str(dest), unzip=True)
        print(f"[Kaggle] '{slug}' baixado.")


# ─── Roboflow ─────────────────────────────────────────────────────────────────

def download_from_roboflow(dry_run: bool = False):
    if not ROBOFLOW_DATASETS:
        print("[Roboflow] Nenhum dataset em ROBOFLOW_DATASETS — pulando.")
        return

    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print("[Roboflow] ROBOFLOW_API_KEY não definido — pulando.")
        return

    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERRO] 'roboflow' não instalado. Execute: pip install roboflow")
        return

    rf = Roboflow(api_key=api_key)

    for workspace_id, project_id, version in ROBOFLOW_DATASETS:
        dest = RAW_DIR / f"roboflow_{project_id}_v{version}"
        if dest.exists() and any(dest.iterdir()):
            print(f"[Roboflow] '{project_id}' já existe — ignorando.")
            continue
        print(f"[Roboflow] Baixando '{workspace_id}/{project_id}' v{version} → {dest} ...")
        if dry_run:
            print(f"[Roboflow] [DRY-RUN] seria baixado para {dest}")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        project  = rf.workspace(workspace_id).project(project_id)
        project.version(version).download("folder", location=str(dest))
        print(f"[Roboflow] '{project_id}' baixado.")


# ─── PlantVillage ─────────────────────────────────────────────────────────────

def _find_pv_root(pv_dir: Path) -> Path | None:
    """Localiza a pasta com as subpastas 'Cultura___doença' dentro do PlantVillage."""
    for candidate in [pv_dir, *pv_dir.rglob("*")]:
        if not candidate.is_dir():
            continue
        subdirs = {d.name for d in candidate.iterdir() if d.is_dir()}
        if any("___" in d for d in subdirs):
            return candidate
    return None


def integrate_plantvillage(dry_run: bool = False) -> dict[str, int]:
    """
    Filtra PlantVillage para apenas as pastas em PV_KEEP_FOLDERS e copia
    para staging/plantvillage/healthy/ ou staging/plantvillage/anomalous/.
    """
    counts: dict[str, int] = {"healthy": 0, "anomalous": 0}

    pv_dir = RAW_DIR / "abdallahalidev_plantvillage-dataset"
    if not pv_dir.exists():
        print("[PlantVillage] Dataset não encontrado em raw/ — pulando.")
        return counts

    pv_root = _find_pv_root(pv_dir)
    if pv_root is None:
        print("[PlantVillage] Estrutura de pastas não reconhecida — pulando.")
        return counts

    print(f"[PlantVillage] Raiz: {pv_root}")

    # Monta mapeamento pasta → label binário
    pv_binary_map: dict[str, str] = {}
    for folder in PV_KEEP_FOLDERS:
        label = "healthy" if folder.lower().endswith("___healthy") else "anomalous"
        pv_binary_map[folder] = label

    staging_pv = STAGING / "plantvillage"
    for label in ("healthy", "anomalous"):
        (staging_pv / label).mkdir(parents=True, exist_ok=True)

    for folder_name, label in pv_binary_map.items():
        src = pv_root / folder_name
        if not src.is_dir():
            continue
        dest_dir = staging_pv / label
        for img in src.iterdir():
            if img.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            dest = dest_dir / f"pv_{folder_name[:30]}_{img.name}"
            if not dest.exists() and not dry_run:
                shutil.copy2(img, dest)
            counts[label] += 1

    print(f"[PlantVillage] healthy={counts['healthy']} | anomalous={counts['anomalous']}")
    return counts


# ─── Organização binária ───────────────────────────────────────────────────────

def _infer_label(name: str) -> str | None:
    """Infere label binário a partir do nome de uma pasta ou arquivo."""
    n = name.lower()
    for kw in HEALTHY_KEYWORDS:
        if kw in n:
            return "healthy"
    for kw in ANOMALOUS_KEYWORDS:
        if kw in n:
            return "anomalous"
    return None


def organize_binary(dry_run: bool = False) -> dict[str, int]:
    """
    Percorre staging/ recursivamente e organiza imagens em:
        staging/healthy/
        staging/anomalous/
    usando palavras-chave nos nomes das pastas.
    """
    counts: dict[str, int] = {"healthy": 0, "anomalous": 0}

    if not STAGING.exists():
        print("[organize_binary] staging/ não existe — nada a organizar.")
        return counts

    for label in ("healthy", "anomalous"):
        (STAGING / label).mkdir(parents=True, exist_ok=True)

    for source_dir in sorted(STAGING.iterdir()):
        if not source_dir.is_dir() or source_dir.name in ("healthy", "anomalous"):
            continue

        # Caso 1: source tem subpastas healthy/ e anomalous/ (já organizado)
        has_binary_sub = (source_dir / "healthy").is_dir() or (source_dir / "anomalous").is_dir()
        if has_binary_sub:
            for label in ("healthy", "anomalous"):
                src_sub = source_dir / label
                if not src_sub.is_dir():
                    continue
                dest_dir = STAGING / label
                for img in src_sub.iterdir():
                    if img.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    dest = dest_dir / f"{source_dir.name}_{img.name}"
                    if not dest.exists() and not dry_run:
                        shutil.copy2(img, dest)
                    counts[label] += 1
            continue

        # Caso 2: source/<class_folder>/ ou source/<class_folder>/images/
        for class_folder in sorted(source_dir.iterdir()):
            if not class_folder.is_dir():
                continue
            label = _infer_label(class_folder.name)
            if label is None:
                continue
            dest_dir = STAGING / label
            # Imagens direto na pasta OU em subpasta /images/
            search_dirs = [class_folder]
            if (class_folder / "images").is_dir():
                search_dirs.append(class_folder / "images")
            for sd in search_dirs:
                for img in sd.iterdir():
                    if img.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    dest = dest_dir / f"{source_dir.name}_{class_folder.name}_{img.name}"
                    if not dest.exists() and not dry_run:
                        shutil.copy2(img, dest)
                    counts[label] += 1

    total = counts["healthy"] + counts["anomalous"]
    print(f"[organize_binary] staging/healthy={counts['healthy']} | staging/anomalous={counts['anomalous']} | total={total}")
    return counts


# ─── Split 70/15/15 ────────────────────────────────────────────────────────────

def split_dataset(dry_run: bool = False) -> dict[str, dict[str, int]]:
    """
    Faz split estratificado 70/15/15 das imagens em staging/healthy/ e staging/anomalous/
    para data/train/, data/val/ e data/test/.
    """
    rng = random.Random(SPLIT_SEED)
    split_counts: dict[str, dict[str, int]] = {
        "train": {"healthy": 0, "anomalous": 0},
        "val":   {"healthy": 0, "anomalous": 0},
        "test":  {"healthy": 0, "anomalous": 0},
    }

    for label in ("healthy", "anomalous"):
        src = STAGING / label
        if not src.is_dir():
            print(f"[split] staging/{label}/ não encontrada — pulando.")
            continue

        images = sorted(f for f in src.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        rng.shuffle(images)
        n        = len(images)
        n_val    = max(1, round(n * VAL_FRAC))
        n_test   = max(1, round(n * (1.0 - TRAIN_FRAC - VAL_FRAC)))
        n_train  = n - n_val - n_test

        splits = {
            "train": images[:n_train],
            "val":   images[n_train:n_train + n_val],
            "test":  images[n_train + n_val:],
        }

        for split_name, imgs in splits.items():
            dest_dir = DATA_DIR / split_name / label
            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                dest = dest_dir / img.name
                if not dest.exists() and not dry_run:
                    shutil.copy2(img, dest)
                split_counts[split_name][label] += 1

    return split_counts


# ─── Relatório ────────────────────────────────────────────────────────────────

def print_report(split_counts: dict[str, dict[str, int]]):
    print("\n" + "=" * 56)
    print("  RELATÓRIO FINAL DO DATASET — FitoVision")
    print("=" * 56)
    print(f"  {'Split':<8}  {'healthy':>8}  {'anomalous':>10}  {'total':>7}")
    print("  " + "-" * 40)

    grand = {"healthy": 0, "anomalous": 0}
    for split in ("train", "val", "test"):
        h = split_counts[split]["healthy"]
        a = split_counts[split]["anomalous"]
        t = h + a
        grand["healthy"]   += h
        grand["anomalous"] += a
        print(f"  {split:<8}  {h:>8}  {a:>10}  {t:>7}")

    print("  " + "-" * 40)
    h, a = grand["healthy"], grand["anomalous"]
    print(f"  {'TOTAL':<8}  {h:>8}  {a:>10}  {h+a:>7}")
    print("=" * 56)

    # Alertas
    print()
    if h == 0 and a == 0:
        print("  ⚠  Dataset vazio! Adicione slugs em KAGGLE_DATASETS ou")
        print("     ROBOFLOW_DATASETS e execute novamente.")
        return

    if h > 0 and a > 0:
        ratio = max(h, a) / min(h, a)
        if ratio > 3.0:
            dominant = "healthy" if h > a else "anomalous"
            print(f"  ⚠  Desbalanceamento {ratio:.1f}:1 — '{dominant}' domina.")
            print(f"     → Considere augmentation extra na classe minoritária.")
            print(f"     → Ou busque mais imagens da classe minoritária no Kaggle/Roboflow.")
        else:
            print(f"  ✅ Balanceamento OK ({ratio:.2f}:1)")
    else:
        missing = "healthy" if h == 0 else "anomalous"
        print(f"  ⚠  Classe '{missing}' tem 0 imagens.")
        print(f"     → Busque mais datasets no Kaggle/Roboflow.")

    for split in ("train", "val", "test"):
        for label in ("healthy", "anomalous"):
            n = split_counts[split][label]
            if 0 < n < 200:
                print(f"  ⚠  {split}/{label} tem apenas {n} imagens (mínimo recomendado: 200).")

    print(f"\n  Próximos passos:")
    print(f"    1. python check_dataset.py        → diagnóstico visual")
    print(f"    2. python train.py --data ./data --model mobilenet_v2")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download e organização de datasets FitoVision")
    parser.add_argument("--skip-download", action="store_true",
                        help="Pula downloads, organiza apenas o raw/ existente")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostra o que faria sem copiar ou baixar nada")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        print("─── 1/5  Download Kaggle ────────────────────────────────────")
        download_from_kaggle(dry_run=args.dry_run)

        print("\n─── 2/5  Download Roboflow ──────────────────────────────────")
        download_from_roboflow(dry_run=args.dry_run)
    else:
        print("[--skip-download] Pulando downloads Kaggle/Roboflow.")

    print("\n─── 3/5  Integrar PlantVillage (filtro binário) ─────────────")
    integrate_plantvillage(dry_run=args.dry_run)

    print("\n─── 4/5  Organizar staging → healthy / anomalous ────────────")
    organize_binary(dry_run=args.dry_run)

    print("\n─── 5/5  Split 70/15/15 ─────────────────────────────────────")
    split_counts = split_dataset(dry_run=args.dry_run)

    print_report(split_counts)


if __name__ == "__main__":
    main()
