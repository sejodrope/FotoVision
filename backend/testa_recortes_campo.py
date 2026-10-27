#!/usr/bin/env python
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Robustez ao enquadramento nas fotos de campo: recorta o centro de cada foto (metade e um
quarto do lado) e roda as mesmas versões de roda_campo.py sobre os recortes.

Por quê: o lote de 27/10/2026 tem vários pés por foto, tirada de cima. Se a resposta mudar
quando o quadro se fecha num pé só, o resultado da foto inteira dependia do enquadramento.
Os recortes são derivados das fotos de campo: servem só para esta análise, ficam numa pasta
temporária e nunca entram em treino nem em recalibração.

Uso:
    python testa_recortes_campo.py                  # fotos em backend/campo_novo
Saída:
    results/campo_novo/recortes/recortes_<versao>.csv e recortes.md
"""

import csv
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

from roda_campo import CLASSES, PYTHON, VERSOES

BACKEND = Path(__file__).resolve().parent
ESCALAS = [(0.5, "m50"), (0.25, "c25")]  # fração do lado, rótulo


def recorta(raiz: Path, destino: Path) -> int:
    n = 0
    for f in sorted(raiz.glob("*/*/*.jpg")):
        classe, cultura = f.parent.name, f.parent.parent.name
        if classe not in CLASSES:
            continue
        im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        W, H = im.size
        pasta = destino / cultura / classe
        pasta.mkdir(parents=True, exist_ok=True)
        for frac, tag in ESCALAS:
            w, h = int(W * frac), int(H * frac)
            x, y = (W - w) // 2, (H - h) // 2
            c = im.crop((x, y, x + w, y + h))
            c.thumbnail((1200, 1200))
            c.save(pasta / f"{f.stem}_{tag}.jpg", quality=92)
            n += 1
    return n


def main():
    raiz = BACKEND / "campo_novo"
    out = BACKEND / "results" / "campo_novo" / "recortes"
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        print(f"{recorta(raiz, Path(tmp))} recortes")
        for versao, modelo, pesos in VERSOES:
            if not (pesos / f"{modelo}_binary_calibration.json").exists():
                continue
            env = {**os.environ, "BINARY_MODEL": modelo, "WEIGHTS_DIR": str(pesos)}
            subprocess.run([PYTHON, "test_campo.py", "--dir", tmp, "--out", str(out / f"recortes_{versao}.csv")],
                           cwd=str(BACKEND), env=env, check=True, stdout=subprocess.DEVNULL)

    # resumo: acertos por horta (prefixo do nome da foto) e escala
    horta = {}
    with open(raiz / "metadados_campo.csv", encoding="utf-8-sig", newline="") as f:
        horta = {Path(r["ficheiro"]).stem: r["horta"] for r in csv.DictReader(f) if r.get("ficheiro")}
    L = ["# Campo — recortes centrais (robustez ao enquadramento)", "",
         "Acertos entre os diagnósticos emitidos; `m50` = metade do lado, `c25` = um quarto.", "",
         "| Versão | Horta | Escala | Acertos | Inconclusivos |", "|---|---|---|---|---|"]
    for versao, _, _ in VERSOES:
        p = out / f"recortes_{versao}.csv"
        if not p.exists():
            continue
        tab = defaultdict(lambda: [0, 0, 0])
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                nome = Path(r["ficheiro"].replace("\\", "/")).stem
                foto, tag = nome.rsplit("_", 1)
                t = tab[(horta.get(foto, "?"), tag)]
                if r["resposta"] in CLASSES:
                    t[0] += r["resposta"] == r["verdade"]
                    t[1] += 1
                else:
                    t[2] += 1
        for (h, tag), (k, n, inc) in sorted(tab.items()):
            L.append(f"| {versao} | {h} | {tag} | {k} de {n} | {inc} |")
    (out / "recortes.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
