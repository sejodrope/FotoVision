#!/usr/bin/env python
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Empacota o dataset e o código para o treino do ViT no Google Colab.

O Google Drive lida mal com 175 mil ficheiros soltos (o upload leva dias e a
leitura pelo Colab é lenta), mas lida bem com poucos ficheiros grandes. Este
script gera, em <projecto>/colab_export/:

    fitovision_dados_train.zip   ~8,3 GB
    fitovision_dados_val.zip     ~1,7 GB
    fitovision_dados_test.zip    ~1,5 GB   (+ split_metadata, manifesto do merge e
                                            colab_manifest.json com a contagem)
    fitovision_codigo.zip        < 1 MB    (scripts, app/ e as 13 fotos de campo)
    FitoVision_ViT_Colab.ipynb             (cópia do notebook)

Sem compressão (JPEG e PNG já são comprimidos): só agrupa. As imagens vão
exactamente como estão em data/ — nada é redimensionado, para o ViT treinar nos
mesmos píxeis das CNNs.

Uso:
    python empacota_colab.py                 # tudo
    python empacota_colab.py --only codigo   # só o código (depois de mexer num script)
    python empacota_colab.py --ciclo5        # data_c5/ (com a fonte retida) em colab_export_c5/
"""

import argparse
import json
import shutil
import time
import zipfile
from pathlib import Path

BACKEND = Path(__file__).parent
PROJECT = BACKEND.parent
DATA    = BACKEND / "data"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS  = ("train", "val", "test")
LABELS  = ("healthy", "anomalous")


def imagens(split: str, label: str, data: Path = DATA) -> list[Path]:
    return sorted(f for f in (data / split / label).iterdir()
                  if f.suffix.lower() in IMAGE_EXTENSIONS)


def empacotar_dados(out: Path, splits: tuple[str, ...], data: Path = DATA,
                    extras: tuple[str, ...] = ("split_metadata.json", "ciclo4_merge_manifest.csv")):
    # O manifesto conta SEMPRE todas as partições, mesmo que só uma seja reempacotada.
    todas = tuple(s for s in (*SPLITS, "retida") if (data / s).exists())
    contagem = {s: {l: len(imagens(s, l, data)) for l in LABELS} for s in todas}
    manifesto = data / "colab_manifest.json"
    manifesto.write_text(json.dumps(
        {"gerado_em": time.strftime("%Y-%m-%d %H:%M:%S"), "imagens": contagem}, indent=2),
        encoding="utf-8")

    for split in splits:
        destino = out / f"fitovision_dados_{split}.zip"
        t0 = time.time()
        n = 0
        with zipfile.ZipFile(destino, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for label in LABELS:
                for f in imagens(split, label, data):
                    z.write(f, f"{data.name}/{split}/{label}/{f.name}")
                    n += 1
                    if n % 20000 == 0:
                        print(f"    {split}: {n} imagens...", flush=True)
            if split == "test":
                for extra in (*extras, "colab_manifest.json"):
                    z.write(data / extra, f"{data.name}/{extra}")
        print(f"  {destino.name}: {n} imagens, {destino.stat().st_size / 1e9:.2f} GB "
              f"({time.time() - t0:.0f} s)", flush=True)


def empacotar_codigo(out: Path):
    destino = out / "fitovision_codigo.zip"
    ficheiros = [*BACKEND.glob("*.py"), BACKEND / "requirements.txt",
                 *(BACKEND / "app").rglob("*.py"),
                 *(f for f in (BACKEND / "campo").rglob("*") if f.is_file())]
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(ficheiros):
            z.write(f, f.relative_to(BACKEND).as_posix())
    print(f"  {destino.name}: {len(ficheiros)} ficheiros, "
          f"{destino.stat().st_size / 1e6:.2f} MB", flush=True)

    notebook = PROJECT / "colab" / "FitoVision_ViT_Colab.ipynb"
    if notebook.exists():
        shutil.copy2(notebook, out / notebook.name)
        print(f"  {notebook.name}: copiado", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Empacota dataset e código para o Colab")
    ap.add_argument("--out", default=str(PROJECT / "colab_export"))
    ap.add_argument("--only", choices=("codigo", "dados", *SPLITS, "retida"), default=None)
    ap.add_argument("--ciclo5", action="store_true",
                    help="empacota data_c5/ (treino, validação, teste e fonte retida) em colab_export_c5/")
    args = ap.parse_args()

    out = Path(args.out if not args.ciclo5 or args.out != str(PROJECT / "colab_export")
               else PROJECT / "colab_export_c5")
    out.mkdir(parents=True, exist_ok=True)
    print(f"A empacotar em {out}", flush=True)

    if args.only in (None, "dados", *SPLITS, "retida"):
        if args.ciclo5:
            todas = (*SPLITS, "retida")
            empacotar_dados(out, (args.only,) if args.only in todas else todas, data=BACKEND / "data_c5",
                            extras=("split_metadata.json", "manifesto_ciclo5.csv"))
        else:
            empacotar_dados(out, (args.only,) if args.only in SPLITS else SPLITS)
    if args.only in (None, "codigo"):
        empacotar_codigo(out)

    total = sum(f.stat().st_size for f in out.iterdir() if f.is_file())
    print(f"\nPronto: {total / 1e9:.2f} GB em {out}")
    print("Envie o conteúdo desta pasta para uma pasta 'FitoVision_Colab' no Google Drive.")
    print("Passo a passo: docs/GUIA_TREINO_VIT_COLAB.md")


if __name__ == "__main__":
    main()
