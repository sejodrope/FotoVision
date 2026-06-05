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

# Slugs "usuario/nome-dataset" — apenas datasets com imagens de plantas/folhas
KAGGLE_DATASETS = [
    "abdallahalidev/plantvillage-dataset",
    "ashishjstar/lettuce-diseases",
    "shuvokumarbasak2030/lettuce-disease-multi-transformation-dataset",
    # misrakahmed/vegetable-image-dataset removido: classifica tipo de vegetal,
    # nao tem labels de doenca — inutilizavel para pipeline binario healthy/anomalous
    "nirmalsankalana/plant-diseases-training-dataset",
]

# Datasets Kaggle para IGNORAR no staging (nao tem labels de doenca utilizaveis)
KAGGLE_STAGING_SKIP = {
    "misrakahmed_vegetable-image-dataset",
}

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

# Formato: (workspace_id, project_id)  — versão detectada automaticamente
# Extraídos de universe.roboflow.com/{workspace}/{project}
ROBOFLOW_DATASETS: list[tuple[str, str]] = [
    # ── Alface (lettuce) ───────────────────────────────────────────────────
    ("prabhat-shukla-xnveb",                    "lettuce-disease"),
    ("cv-tomato-disease",                        "lettuce-disease-auek1"),
    ("justin-issac-zwnqi",                       "lettuce-disease-classification-dryyj"),
    ("yssp",                                     "disease-lettuce-h69zb"),
    ("university-11lwh",                         "lettuce-disease-detection"),
    ("disease-detection-rqwrr",                  "lettuce-disease-detection-pvv71"),
    ("lettuce-iyoop",                            "lettuce-disease-cge82"),
    ("the-stove",                                "lettuce-kgxfw"),
    ("lettuce-cjpxw",                            "lettuce-ub8x5"),
    ("asia-in6sj",                               "lettuce-d3ixa"),
    ("pravalika-odkyr",                          "lettuce-qcfff"),
    ("nanyang-technological-university-ytc7m",   "lettuce-disease-detection-wbnpq"),
    ("sathwik-chandra",                          "lettuce-leaf-disease-detection"),
    # ── Folhosas genéricas ─────────────────────────────────────────────────
    ("hung-nguyen-g3wvq",                        "leaf-disease-gm9cg"),
    ("oral-cncer",                               "anthracnose-spinach"),
    ("chlorotechdatasets",                       "leafy-vegetables-7kpre"),
    ("cxd",                                      "leafy-vegetables-iz0st"),
    ("tsqs-workspace",                           "leafy-vegetable-crops"),
    ("final-year-project-zmta6",                 "plant-leaf-disease-detection"),
    ("new-workspace-iuwda",                      "plant-village-9vp8g"),
]

# ─── CONFIGURAÇÃO: Organização binária ───────────────────────────────────────

HEALTHY_KEYWORDS = ["healthy", "saudavel", "saudável", "normal", "good"]
ANOMALOUS_KEYWORDS = [
    "disease", "blight", "mildew", "spot", "rot", "virus", "viral",
    "anomaly", "doenca", "doença", "bacterial", "fungal",
    "downy", "powdery", "mosaic", "chlorosis", "scorch",
    "rust", "leaf_curl", "wilt", "necrosis", "angular",
    # patógenos e pragas adicionais (PlantVillage + shuvokumarbasak2030)
    "scab", "mite", "mold", "anthracnose", "blast", "nematode",
    "mottle", "greening", "measles", "weed", "phytophthora",
    "tungro", "streak", "septoria", "leafroll", "target",
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _count_images(path: Path) -> int:
    return sum(1 for f in path.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)


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

    ok = failed = skipped = 0

    for slug in KAGGLE_DATASETS:
        dest = RAW_DIR / slug.replace("/", "_")
        if dest.exists() and any(dest.iterdir()):
            print(f"[Kaggle] '{slug}' ja existe — ignorando.")
            skipped += 1
            continue

        print(f"[Kaggle] {slug} ...", end=" ", flush=True)

        if dry_run:
            print("[DRY-RUN]")
            continue

        try:
            dest.mkdir(parents=True, exist_ok=True)
            api.dataset_download_files(slug, path=str(dest), unzip=True)
            n = _count_images(dest)
            print(f"OK ({n} imgs)")
            ok += 1
        except Exception as exc:
            print(f"ERRO — {exc}")
            failed += 1

    print(f"[Kaggle] Concluido: {ok} baixados, {skipped} ja existentes, {failed} falhas.")


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

    ok = failed = skipped = 0

    for workspace_id, project_id in ROBOFLOW_DATASETS:
        dest = RAW_DIR / f"roboflow_{project_id}"
        if dest.exists() and any(dest.rglob("*")):
            print(f"[Roboflow] '{project_id}' ja existe — ignorando.")
            skipped += 1
            continue

        print(f"[Roboflow] {workspace_id}/{project_id} ...", end=" ", flush=True)

        if dry_run:
            print("[DRY-RUN]")
            continue

        try:
            project = rf.workspace(workspace_id).project(project_id)

            # Auto-detecta versão mais recente
            try:
                versions = project.versions()
                version_num = max(v.version for v in versions) if versions else 1
            except Exception:
                version_num = 1

            dest.mkdir(parents=True, exist_ok=True)

            # Tenta "folder" (classificação). Se for detecção, usa "yolov8".
            fmt = "folder"
            try:
                project.version(version_num).download(fmt, location=str(dest))
            except Exception as exc_fmt:
                if "object-detection" in str(exc_fmt) or "invalid format" in str(exc_fmt):
                    fmt = "yolov8"
                    project.version(version_num).download(fmt, location=str(dest))
                else:
                    raise

            n = _count_images(dest)
            print(f"v{version_num} [{fmt}] OK ({n} imgs)")
            ok += 1

        except Exception as exc:
            msg = str(exc)
            # Simplifica mensagem de erro longa (JSON do Roboflow)
            if "error" in msg and len(msg) > 120:
                msg = msg[:120] + "..."
            print(f"ERRO — {msg}")
            if dest.exists() and not any(dest.rglob("*")):
                shutil.rmtree(dest, ignore_errors=True)
            failed += 1

    print(f"[Roboflow] Concluido: {ok} baixados, {skipped} ja existentes, {failed} falhas.")


# ─── Staging: Kaggle genérico ─────────────────────────────────────────────────

PV_SLUG = "abdallahalidev_plantvillage-dataset"


def stage_kaggle_raw(dry_run: bool = False) -> int:
    """
    Copia datasets Kaggle de raw/ para staging/, preservando a estrutura de pastas.
    O PlantVillage é tratado por integrate_plantvillage(); Roboflow por stage_roboflow_raw().
    O organize_binary() posterior classifica pelas palavras-chave nos nomes das pastas.
    Datasets em KAGGLE_STAGING_SKIP são ignorados (sem labels de doença utilizáveis).
    """
    total = 0
    for source_dir in sorted(RAW_DIR.iterdir()):
        if not source_dir.is_dir():
            continue
        if source_dir.name == PV_SLUG:
            continue
        if source_dir.name.startswith("roboflow_"):
            continue
        if source_dir.name in KAGGLE_STAGING_SKIP:
            print(f"  [{source_dir.name}] na lista de skip — ignorando.")
            continue

        staging_dest = STAGING / source_dir.name
        if staging_dest.exists() and any(
            f for f in staging_dest.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS
        ):
            print(f"  [{source_dir.name}] ja staged — ignorando.")
            continue

        print(f"  [{source_dir.name}] copiando para staging/ ...")
        n = 0
        for img in source_dir.rglob("*"):
            if img.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rel  = img.relative_to(source_dir)
            dest = staging_dest / rel
            if not dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copy2(img, dest)
            n += 1
        print(f"    {n} imagens copiadas")
        total += n
    return total


# ─── Staging: Roboflow (yolov8 → binário) ────────────────────────────────────

def stage_roboflow_raw(dry_run: bool = False) -> dict[str, int]:
    """
    Converte datasets Roboflow baixados em formato yolov8 para o formato binário:
      - imagem COM arquivo de label não-vazio → anomalous  (tem doença anotada)
      - imagem SEM label ou label vazio       → healthy

    Para datasets classification (formato "folder"), o organize_binary() já trata.
    """
    counts = {"healthy": 0, "anomalous": 0}

    for rf_dir in sorted(RAW_DIR.iterdir()):
        if not rf_dir.is_dir() or not rf_dir.name.startswith("roboflow_"):
            continue

        # Detecta se é formato yolov8 (tem pasta "labels/")
        has_labels_dir = any(rf_dir.rglob("labels"))
        if not has_labels_dir:
            continue  # formato "folder" (classificação) — organizado por organize_binary

        staging_dest = STAGING / rf_dir.name
        if staging_dest.exists() and any(
            f for f in staging_dest.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS
        ):
            print(f"  [{rf_dir.name}] ja staged — ignorando.")
            continue

        print(f"  [{rf_dir.name}] yolov8 → binário ...")
        n_h = n_a = 0

        # Localiza todas as pastas "images/" dentro do dataset
        for images_dir in rf_dir.rglob("images"):
            if not images_dir.is_dir():
                continue
            labels_dir = images_dir.parent / "labels" / images_dir.name
            if not labels_dir.exists():
                labels_dir = images_dir.parent.parent / "labels" / images_dir.name

            for img in images_dir.iterdir():
                if img.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                # Procura arquivo de label correspondente
                label_file = labels_dir / (img.stem + ".txt") if labels_dir.exists() else None
                has_annotation = (
                    label_file is not None
                    and label_file.exists()
                    and label_file.stat().st_size > 10
                )

                label = "anomalous" if has_annotation else "healthy"
                dest_dir = staging_dest / label
                if not dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest = dest_dir / f"{images_dir.name}_{img.name}"
                    if not dest.exists():
                        shutil.copy2(img, dest)
                counts[label] += 1
                if label == "healthy":
                    n_h += 1
                else:
                    n_a += 1

        print(f"    healthy={n_h} | anomalous={n_a}")

    return counts


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


def _collect_labeled_dirs(root: Path) -> list[tuple[Path, str]]:
    """
    BFS a partir de root: retorna (dir, label) para cada diretório cujo nome
    indica label binário. Não desce dentro de um diretório já rotulado.
    """
    results: list[tuple[Path, str]] = []
    queue: list[Path] = [d for d in sorted(root.iterdir()) if d.is_dir()]
    while queue:
        d = queue.pop(0)
        label = _infer_label(d.name)
        if label is not None:
            results.append((d, label))
        else:
            queue.extend(d2 for d2 in sorted(d.iterdir()) if d2.is_dir())
    return results


def organize_binary(dry_run: bool = False) -> dict[str, int]:
    """
    Percorre staging/ recursivamente (qualquer profundidade) e organiza imagens em:
        staging/healthy/
        staging/anomalous/
    usando palavras-chave nos nomes das pastas.

    Suporta:
      staging/<dataset>/healthy|anomalous/          (já organizado — plantvillage)
      staging/<dataset>/<class>/                    (nível 2)
      staging/<dataset>/<sub>/<class>/              (nível 3 — ashishjstar, nirmalsankalana)
      staging/<dataset>/<sub>/<sub>/<class>/        (nível 4+ — shuvokumarbasak2030)
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

        # Caso 1: source tem subpastas healthy/ e/ou anomalous/ diretas (já organizado)
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

        # Caso 2: BFS recursivo — encontra pastas rotuladas a qualquer profundidade
        labeled_dirs = _collect_labeled_dirs(source_dir)
        if not labeled_dirs:
            continue

        for labeled_dir, label in labeled_dirs:
            dest_dir = STAGING / label
            rel_parts = labeled_dir.relative_to(source_dir).parts
            prefix = source_dir.name + "_" + "_".join(rel_parts)

            for img in labeled_dir.rglob("*"):
                if not img.is_file() or img.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                img_rel = img.relative_to(labeled_dir)
                flat = "_".join(img_rel.parts[:-1] + (img.name,)) if len(img_rel.parts) > 1 else img.name
                dest = dest_dir / f"{prefix}_{flat}"
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
        print("─── 1/7  Download Kaggle ────────────────────────────────────")
        download_from_kaggle(dry_run=args.dry_run)

        print("\n─── 2/7  Download Roboflow ──────────────────────────────────")
        download_from_roboflow(dry_run=args.dry_run)
    else:
        print("[--skip-download] Pulando downloads Kaggle/Roboflow.")

    print("\n─── 3/7  PlantVillage → staging (filtro binario) ────────────")
    integrate_plantvillage(dry_run=args.dry_run)

    print("\n─── 4/7  Kaggle outros → staging ────────────────────────────")
    n_kag = stage_kaggle_raw(dry_run=args.dry_run)
    print(f"  Total copiado do Kaggle: {n_kag} imagens")

    print("\n─── 5/7  Roboflow yolov8 → staging binario ──────────────────")
    stage_roboflow_raw(dry_run=args.dry_run)

    print("\n─── 6/7  Organizar staging → healthy / anomalous ────────────")
    organize_binary(dry_run=args.dry_run)

    print("\n─── 7/7  Split 70/15/15 ─────────────────────────────────────")
    # Limpa splits anteriores para garantir consistência com o staging atual
    if not args.dry_run:
        for split_name in ("train", "val", "test"):
            split_dir = DATA_DIR / split_name
            if split_dir.exists():
                shutil.rmtree(split_dir)
    split_counts = split_dataset(dry_run=args.dry_run)

    print_report(split_counts)


if __name__ == "__main__":
    main()
