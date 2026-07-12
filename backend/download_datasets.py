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
import json
import os
import random
import re
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

# Regras de inferência de label a partir do nome da pasta.
#
# ATENÇÃO — estas regras são heurísticas e são a segunda maior fonte de erro do
# pipeline (a primeira é o vazamento de dados). Foram endurecidas em relação à
# versão anterior:
#
#   • Removidos os termos genéricos "normal", "good" e "target": casavam com
#     nomes de pasta que nada têm a ver com sanidade da folha (ex.: "target_size",
#     "normalized") e produziam rótulos errados em silêncio.
#   • Removido "weed": uma erva daninha não é uma folha doente. Manter esse termo
#     ensinava o modelo a classificar uma ESPÉCIE DE PLANTA como "anomalous",
#     não uma anomalia fitossanitária — ruído puro para a tarefa deste TCC.
#   • O casamento passou a ser por TOKEN (fronteiras de palavra), não por
#     substring solta, para evitar falsos positivos do tipo "spot" dentro de
#     "spotlight".
#
# Toda a atribuição é registada em data/label_map_audit.json para conferência
# manual — nenhum rótulo é inferido em silêncio.

HEALTHY_KEYWORDS = ["healthy", "saudavel", "saudavel", "sadia", "sana"]

ANOMALOUS_KEYWORDS = [
    # sintomas / classes de doença
    "disease", "diseased", "blight", "mildew", "spot", "rot", "virus", "viral",
    "anomaly", "anomalous", "doenca", "bacterial", "fungal", "fungus",
    "downy", "powdery", "mosaic", "chlorosis", "scorch",
    "rust", "curl", "wilt", "necrosis", "angular",
    "scab", "mite", "mold", "anthracnose", "blast", "nematode",
    "mottle", "greening", "measles", "phytophthora",
    "tungro", "streak", "septoria", "leafroll", "deficiency",
]

# Pastas cujo nome casa com estes termos são DESCARTADAS (não são nem healthy
# nem anomalous para a tarefa: são outra espécie, ou metadados do dataset).
IGNORE_KEYWORDS = [
    "weed", "weeds", "shepherd", "purse",   # ervas daninhas — outra planta, não doença
    "background", "soil", "unknown", "other", "misc",
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

def _read_yolo_class_names(rf_dir: Path) -> list[str] | None:
    """Lê os nomes das classes do data.yaml de um dataset YOLO. Devolve None se não achar."""
    for yaml_path in rf_dir.rglob("data.yaml"):
        try:
            text = yaml_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        # data.yaml do Roboflow: names: ['Bacterial', 'Downy_mildew', 'Healthy', ...]
        m = re.search(r"^names:\s*\[(.*?)\]", text, re.MULTILINE | re.DOTALL)
        if m:
            raw = m.group(1)
            return [n.strip().strip("'\"") for n in raw.split(",") if n.strip()]

        # Formato em bloco:  names:\n  - Bacterial\n  - Healthy
        m = re.search(r"^names:\s*$", text, re.MULTILINE)
        if m:
            names: list[str] = []
            for line in text[m.end():].splitlines():
                stripped = line.strip()
                if stripped.startswith("- "):
                    names.append(stripped[2:].strip().strip("'\""))
                elif stripped and not line.startswith((" ", "\t", "-")):
                    break
            if names:
                return names

    return None


def stage_roboflow_raw(dry_run: bool = False) -> dict[str, int]:
    """
    Converte datasets Roboflow em formato yolov8 para o formato binário.

    ─── Correcção de um erro grave de rotulagem ──────────────────────────────
    A versão anterior fazia:
        imagem COM ficheiro de anotação → anomalous
        imagem SEM anotação             → healthy

    Isto está errado nos dois sentidos:
      • Num dataset de detecção, uma imagem cuja bounding box marca uma folha
        SAUDÁVEL (classe 'Healthy' — presente na maioria destes datasets) tem
        ficheiro de anotação, e era rotulada como DOENTE.
      • Uma imagem que simplesmente não foi anotada (esquecimento do anotador,
        ou imagem de fundo) era rotulada como SAUDÁVEL.

    A versão correcta lê os NOMES DAS CLASSES do data.yaml, mapeia cada class_id
    para healthy/anomalous com as mesmas regras de _infer_label(), e decide pelo
    conteúdo real das anotações:
      • alguma box de classe anómala  → anomalous
      • só boxes de classe saudável   → healthy
      • sem anotações                 → DESCARTADA (não se pode afirmar nada)
      • classe não reconhecida        → DESCARTADA
    """
    counts = {"healthy": 0, "anomalous": 0, "descartadas": 0}

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

        class_names = _read_yolo_class_names(rf_dir)
        if not class_names:
            print(f"  [{rf_dir.name}] ⚠  data.yaml não encontrado/legível — "
                  f"dataset INTEIRO descartado (rotular sem as classes seria adivinhar).")
            continue

        # class_id → "healthy" | "anomalous" | None (desconhecida)
        id_to_label: dict[int, str | None] = {}
        for i, cname in enumerate(class_names):
            lb = _infer_label(cname)
            id_to_label[i] = lb if lb in ("healthy", "anomalous") else None

        known = {i: lb for i, lb in id_to_label.items() if lb}
        unknown = [class_names[i] for i, lb in id_to_label.items() if not lb]
        print(f"  [{rf_dir.name}] classes: {class_names}")
        print(f"      mapeadas: {[f'{class_names[i]}→{lb}' for i, lb in known.items()]}")
        if unknown:
            print(f"      ⚠  não reconhecidas (imagens com estas classes serão descartadas): {unknown}")

        if not known:
            print(f"      ⚠  nenhuma classe mapeável — dataset descartado.")
            continue

        n_h = n_a = n_drop = 0

        for images_dir in rf_dir.rglob("images"):
            if not images_dir.is_dir():
                continue
            labels_dir = images_dir.parent / "labels"
            if not labels_dir.is_dir():
                labels_dir = images_dir.parent.parent / "labels" / images_dir.name
            if not labels_dir.is_dir():
                continue

            for img in images_dir.iterdir():
                if img.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                label_file = labels_dir / (img.stem + ".txt")
                if not label_file.exists():
                    n_drop += 1          # sem anotação ⇒ não se sabe nada sobre ela
                    continue

                try:
                    lines = [
                        ln.strip() for ln in
                        label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
                        if ln.strip()
                    ]
                except OSError:
                    n_drop += 1
                    continue

                if not lines:
                    n_drop += 1          # anotação vazia ⇒ idem
                    continue

                box_labels: list[str | None] = []
                for ln in lines:
                    try:
                        cid = int(ln.split()[0])
                    except (ValueError, IndexError):
                        continue
                    box_labels.append(id_to_label.get(cid))

                if not box_labels or any(b is None for b in box_labels):
                    n_drop += 1          # contém classe não mapeável ⇒ ambígua
                    continue

                # Qualquer box anómala domina: a folha tem uma anomalia.
                label = "anomalous" if "anomalous" in box_labels else "healthy"

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

        n_drop_total = n_drop
        counts["descartadas"] += n_drop_total
        print(f"      → healthy={n_h} | anomalous={n_a} | descartadas={n_drop_total}")

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

def _tokens(name: str) -> set[str]:
    """Divide um nome de pasta em tokens minúsculos ('Tomato___Late_blight' → {tomato, late, blight})."""
    return {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}


def _infer_label(name: str) -> str | None:
    """
    Infere o label binário a partir do nome de uma pasta.

    Devolve "healthy", "anomalous", "ignore" (pasta a descartar) ou None
    (não reconhecido — o chamador desce mais um nível ou avisa).

    Casamento por token, não por substring: "spot" casa com 'Septoria_leaf_spot'
    mas não com 'spotlight'. 'healthy' tem precedência porque nomes como
    'Tomato___healthy' não contêm termos de doença.
    """
    toks = _tokens(name)

    if toks & set(IGNORE_KEYWORDS):
        return "ignore"
    if toks & set(HEALTHY_KEYWORDS):
        return "healthy"
    if toks & set(ANOMALOUS_KEYWORDS):
        return "anomalous"
    return None


def _collect_labeled_dirs(root: Path) -> tuple[list[tuple[Path, str]], list[Path]]:
    """
    BFS a partir de root: devolve (labeled_dirs, unrecognized_leaf_dirs).

    labeled_dirs        — [(dir, "healthy"|"anomalous")] para cada pasta rotulável.
    unrecognized_leaf_dirs — pastas-folha com imagens cujo nome não casou com
                             nenhuma regra. Estas são REPORTADAS, não adivinhadas:
                             rotular à sorte é como o dataset ganhou ruído.

    Não desce dentro de uma pasta já rotulada.
    """
    labeled: list[tuple[Path, str]] = []
    unrecognized: list[Path] = []

    queue: list[Path] = [d for d in sorted(root.iterdir()) if d.is_dir()]
    while queue:
        d = queue.pop(0)
        label = _infer_label(d.name)

        if label == "ignore":
            continue
        if label is not None:
            labeled.append((d, label))
            continue

        subdirs = [d2 for d2 in sorted(d.iterdir()) if d2.is_dir()]
        if subdirs:
            queue.extend(subdirs)
        else:
            # Pasta-folha sem label reconhecido: só reporta se contiver imagens
            has_imgs = any(
                f.suffix.lower() in IMAGE_EXTENSIONS for f in d.iterdir() if f.is_file()
            )
            if has_imgs:
                unrecognized.append(d)

    return labeled, unrecognized


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

    # Auditoria: toda pasta de origem → label atribuído. Salvo em disco para que a
    # rotulagem seja verificável, e não uma caixa-preta de heurísticas.
    audit: dict[str, dict] = {"assigned": [], "unrecognized": []}

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
                n = 0
                for img in src_sub.iterdir():
                    if img.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    # Prefixo com a fonte: permite reconstruir a proveniência de
                    # cada imagem depois do flatten (necessário para o split
                    # leave-one-source-out).
                    dest = dest_dir / f"{source_dir.name}__{img.name}"
                    if not dest.exists() and not dry_run:
                        shutil.copy2(img, dest)
                    counts[label] += 1
                    n += 1
                audit["assigned"].append(
                    {"source": source_dir.name, "folder": label, "label": label, "n_images": n}
                )
            continue

        # Caso 2: BFS recursivo — encontra pastas rotuladas a qualquer profundidade
        labeled_dirs, unrecognized = _collect_labeled_dirs(source_dir)

        for u in unrecognized:
            n_imgs = sum(
                1 for f in u.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            )
            audit["unrecognized"].append(
                {
                    "source": source_dir.name,
                    "folder": str(u.relative_to(source_dir)),
                    "n_images": n_imgs,
                }
            )

        if not labeled_dirs:
            continue

        for labeled_dir, label in labeled_dirs:
            dest_dir = STAGING / label
            rel_parts = labeled_dir.relative_to(source_dir).parts
            prefix = source_dir.name + "__" + "_".join(rel_parts)

            n = 0
            for img in labeled_dir.rglob("*"):
                if not img.is_file() or img.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                img_rel = img.relative_to(labeled_dir)
                flat = "_".join(img_rel.parts[:-1] + (img.name,)) if len(img_rel.parts) > 1 else img.name
                dest = dest_dir / f"{prefix}_{flat}"
                if not dest.exists() and not dry_run:
                    shutil.copy2(img, dest)
                counts[label] += 1
                n += 1

            audit["assigned"].append(
                {
                    "source": source_dir.name,
                    "folder": str(labeled_dir.relative_to(source_dir)),
                    "label": label,
                    "n_images": n,
                }
            )

    total = counts["healthy"] + counts["anomalous"]
    print(f"[organize_binary] staging/healthy={counts['healthy']} | staging/anomalous={counts['anomalous']} | total={total}")

    # Relatório de rotulagem
    if audit["unrecognized"]:
        n_unrec = sum(u["n_images"] for u in audit["unrecognized"])
        print(f"\n  ⚠  {len(audit['unrecognized'])} pastas com {n_unrec} imagens NÃO reconhecidas")
        print("     (descartadas — nenhum label foi adivinhado). Reveja-as em:")
        for u in audit["unrecognized"][:8]:
            print(f"       {u['source']}/{u['folder']}  ({u['n_images']} imgs)")
        if len(audit["unrecognized"]) > 8:
            print(f"       ... e mais {len(audit['unrecognized']) - 8}")

    if not dry_run:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        audit_path = DATA_DIR / "label_map_audit.json"
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, ensure_ascii=False)
        print(f"\n  Mapa de rotulagem (para conferir à mão): {audit_path}")

    return counts


# ─── Split 70/15/15 ────────────────────────────────────────────────────────────

def _source_of(path: Path) -> str:
    """Extrai o dataset de origem do nome do ficheiro ('plantvillage__foo.jpg' → 'plantvillage')."""
    name = path.name
    return name.split("__", 1)[0] if "__" in name else "desconhecido"


def split_dataset(dry_run: bool = False, max_distance: int = 4) -> dict[str, dict[str, int]]:
    """
    Split 70/15/15 AGRUPADO POR IDENTIDADE VISUAL, estratificado por label.

    ─── Por que não um shuffle simples ───────────────────────────────────────
    A versão anterior sorteava ao nível do FICHEIRO. Como o pool contém cópias
    transformadas da mesma foto (o dataset 'multi-transformation' é literalmente
    isso) e re-uploads do PlantVillage sob outros nomes, o sorteio colocava a
    rotação de uma foto no treino e o flip da MESMA foto no teste.

    Resultado: o modelo memorizava a foto e a accuracy de teste ia a ~99% sem
    qualquer generalização — exactamente o sintoma observado em fotos reais.

    ─── O que esta versão faz ────────────────────────────────────────────────
    1. Calcula um hash perceptual invariante a rotação/flip de cada imagem.
    2. Agrupa near-duplicates (union-find sobre os hashes).
    3. Sorteia GRUPOS INTEIROS para train/val/test.
       ⇒ Todas as variantes de uma mesma foto caem no MESMO split.
    4. Descarta imagens corrompidas (nunca as substitui por um quadrado preto
       com o label original — isso ensinava "imagem preta → healthy").
    5. Descarta grupos com labels contraditórios (a mesma foto aparece como
       healthy numa fonte e anomalous noutra ⇒ pelo menos um dos rótulos está
       errado; treinar com ambos é injectar ruído).

    O split resultante é mais pequeno e a accuracy será MAIS BAIXA — porque
    passa a ser real.
    """
    from imagehash_utils import group_by_similarity, group_stats, hash_images

    # ── 1. Recolhe todas as imagens do staging com o seu label ────────────────
    all_images: list[Path] = []
    label_of: dict[Path, str] = {}

    for label in ("healthy", "anomalous"):
        src = STAGING / label
        if not src.is_dir():
            print(f"[split] staging/{label}/ não encontrada — pulando.")
            continue
        imgs = sorted(f for f in src.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        for img in imgs:
            all_images.append(img)
            label_of[img] = label

    if not all_images:
        print("[split] Nenhuma imagem em staging/ — nada a dividir.")
        return {s: {"healthy": 0, "anomalous": 0} for s in ("train", "val", "test")}

    print(f"[split] {len(all_images)} imagens em staging/")

    # ── 2. Hash + agrupamento ─────────────────────────────────────────────────
    print("[split] Calculando hashes perceptuais (pode demorar alguns minutos)...")
    orbits, corrupted = hash_images(all_images)

    if corrupted:
        print(f"[split] ⚠  {len(corrupted)} imagens corrompidas DESCARTADAS "
              f"(não entram em nenhum split).")

    print(f"[split] Agrupando near-duplicates (max_distance={max_distance})...")
    groups = group_by_similarity(orbits, max_distance=max_distance)
    stats = group_stats(groups)

    print(f"[split] {stats['n_images']} imagens → {stats['n_groups']} fotos distintas "
          f"({stats['redundancy_pct']:.1f}% de redundância, maior grupo = "
          f"{stats['largest_group']} cópias)")

    # ── 3. Agrupa por group_id e resolve o label de cada grupo ────────────────
    group_members: dict[int, list[Path]] = defaultdict(list)
    for img, gid in groups.items():
        group_members[gid].append(img)

    group_label: dict[int, str] = {}
    conflicted: list[int] = []

    for gid, members in group_members.items():
        labels = {label_of[m] for m in members}
        if len(labels) > 1:
            conflicted.append(gid)      # mesma foto com dois rótulos ⇒ descartar
        else:
            group_label[gid] = labels.pop()

    if conflicted:
        n_conf_imgs = sum(len(group_members[g]) for g in conflicted)
        print(f"[split] ⚠  {len(conflicted)} grupos ({n_conf_imgs} imagens) com labels "
              f"CONTRADITÓRIOS — descartados por ambiguidade.")

    # ── 4. Split ao nível do GRUPO, estratificado por label ───────────────────
    rng = random.Random(SPLIT_SEED)
    split_counts: dict[str, dict[str, int]] = {
        "train": {"healthy": 0, "anomalous": 0},
        "val":   {"healthy": 0, "anomalous": 0},
        "test":  {"healthy": 0, "anomalous": 0},
    }
    split_groups: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    # Proveniência por split — para checar se o teste tem fontes que o treino não tem
    split_sources: dict[str, dict[str, int]] = {
        "train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int),
    }

    if not dry_run:
        for split_name in ("train", "val", "test"):
            for label in ("healthy", "anomalous"):
                (DATA_DIR / split_name / label).mkdir(parents=True, exist_ok=True)

    for label in ("healthy", "anomalous"):
        gids = sorted(g for g, lb in group_label.items() if lb == label)
        rng.shuffle(gids)

        n = len(gids)
        n_val = max(1, round(n * VAL_FRAC)) if n else 0
        n_test = max(1, round(n * (1.0 - TRAIN_FRAC - VAL_FRAC))) if n else 0
        n_train = n - n_val - n_test

        assignment = {
            "train": gids[:n_train],
            "val":   gids[n_train:n_train + n_val],
            "test":  gids[n_train + n_val:],
        }

        for split_name, split_gids in assignment.items():
            split_groups[split_name] += len(split_gids)
            dest_dir = DATA_DIR / split_name / label

            for gid in split_gids:
                for img in group_members[gid]:
                    dest = dest_dir / img.name
                    if not dest.exists() and not dry_run:
                        shutil.copy2(img, dest)
                    split_counts[split_name][label] += 1
                    split_sources[split_name][_source_of(img)] += 1

    print(f"[split] Grupos por split: treino={split_groups['train']} | "
          f"val={split_groups['val']} | teste={split_groups['test']}")

    # ── 5. Relatório de proveniência ──────────────────────────────────────────
    print("\n[split] Proveniência por split (nº de imagens por dataset de origem):")
    all_sources = sorted({s for sp in split_sources.values() for s in sp})
    for s in all_sources:
        tr = split_sources["train"].get(s, 0)
        va = split_sources["val"].get(s, 0)
        te = split_sources["test"].get(s, 0)
        print(f"    {s[:42]:<42}  treino={tr:>7}  val={va:>6}  teste={te:>6}")

    if not dry_run:
        meta = {
            "seed": SPLIT_SEED,
            "max_hamming_distance": max_distance,
            "n_images_staging": len(all_images),
            "n_corrupted_dropped": len(corrupted),
            "n_conflicted_groups_dropped": len(conflicted),
            "n_distinct_photos": stats["n_groups"],
            "redundancy_pct": round(stats["redundancy_pct"], 2),
            "largest_group": stats["largest_group"],
            "groups_per_split": split_groups,
            "images_per_split": {k: dict(v) for k, v in split_counts.items()},
            "sources_per_split": {k: dict(v) for k, v in split_sources.items()},
        }
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(DATA_DIR / "split_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"\n[split] Metadados do split: {DATA_DIR / 'split_metadata.json'}")

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
    print(f"    1. python audit_leakage.py        → confirma que o split não vaza")
    print(f"    2. python check_dataset.py        → diagnóstico visual")
    print(f"    3. python train.py --data ./data --binary --model efficientnet_b0")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download e organização de datasets FitoVision")
    parser.add_argument("--skip-download", action="store_true",
                        help="Pula downloads, organiza apenas o raw/ existente")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostra o que faria sem copiar ou baixar nada")
    parser.add_argument("--max-distance", type=int, default=4,
                        help="Distância de Hamming para agrupar near-duplicates no split "
                             "(0 = só duplicados exactos; 4 = default, apanha augmentation)")
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

    print("\n─── 7/7  Split 70/15/15 AGRUPADO (group-aware) ──────────────")
    # Limpa splits anteriores para garantir consistência com o staging atual
    if not args.dry_run:
        for split_name in ("train", "val", "test"):
            split_dir = DATA_DIR / split_name
            if split_dir.exists():
                shutil.rmtree(split_dir)
    split_counts = split_dataset(dry_run=args.dry_run, max_distance=args.max_distance)

    print_report(split_counts)


if __name__ == "__main__":
    main()
