#!/usr/bin/env python
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Ciclo 4 — merge dos lotes preparados na Fase 6 com o split de produção.

Lotes (dataset/_work/final/, ver dataset/_reports/PHASE6_REPORT.md):
  - spinach           (Malabar Spinach, 2 fontes Mendeley) — split da Fase 6 mantido
  - lettuce_downy_v1  (Roboflow)                           — split REFEITO aqui

Por que refazer o split do lettuce_downy_v1: os 1.142 ficheiros são recortes de
anotação (`__annN`) e variantes Roboflow (`.rf.<hash>`) de apenas 49 fotos
originais. A Fase 6 dividiu por recorte, e 45 das 49 fotos tinham recortes em
mais de um split — vazamento que a auditoria por phash não apanha, porque recortes
diferentes da mesma foto não são quase-duplicados. Aqui o split é agrupado pela
foto original (prefixo antes de `.rf.`/`__ann`), seed=42, ~70/15/15 por imagens.

O split existente em data/{train,val,test} NÃO é refeito: os ficheiros novos são
apenas acrescentados (prefixo `<fonte>__`), de modo que o teste do ciclo 3 continua
contido no teste do ciclo 4 e os dois ciclos são comparáveis nesse subconjunto.

Saídas:
  data/{train,val,test}/{healthy,anomalous}/<fonte>__<ficheiro>
  data/ciclo4_merge_manifest.csv   — origem → destino de cada ficheiro
  data/split_metadata.json         — ganha a chave "ciclo4_merge"

Uso:
    python merge_ciclo4.py            # dry-run: só mostra contagens
    python merge_ciclo4.py --apply
"""

import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

BACKEND = Path(__file__).parent
ROOT    = BACKEND.parent
DATA    = BACKEND / "data"
FINAL   = ROOT / "dataset" / "_work" / "final"
REPORTS = ROOT / "dataset" / "_reports"

LABELS = {"saudavel": "healthy", "anomala": "anomalous"}
SPLITS = ("train", "val", "test")
SEED   = 42


def _read_manifest(name: str) -> list[dict]:
    with open(REPORTS / f"final_manifest_{name}.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _photo_key(filename: str) -> str:
    """Foto original de um ficheiro Roboflow: '27_jpg.rf.<hash>__ann1.jpg' → '27_jpg'."""
    return re.split(r"\.rf\.|__ann", filename)[0]


def _grouped_split(rows: list[dict]) -> dict[str, str]:
    """Atribui cada foto original a um split, ~70/15/15 por nº de imagens."""
    groups = defaultdict(list)
    for r in rows:
        groups[_photo_key(Path(r["caminho"]).name)].append(r)
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    total = len(rows)
    targets = {"test": 0.15 * total, "val": 0.15 * total}
    counts = Counter()
    assign = {}
    for k in keys:
        # Preenche teste e val primeiro; o resto vai para treino.
        split = next((s for s in ("test", "val") if counts[s] < targets[s]), "train")
        assign[k] = split
        counts[split] += len(groups[k])
    return {r["caminho"]: assign[_photo_key(Path(r["caminho"]).name)] for r in rows}


def plan() -> list[dict]:
    moves = []
    for r in _read_manifest("spinach"):
        src = ROOT / r["caminho"].replace("\\", "/")
        moves.append({
            "src": src, "split": r["split"], "label": LABELS[r["classe"]],
            "dest_name": f"mendeley_{r['origem_dataset']}__{src.name}",
            "fonte": r["origem_dataset"], "grupo": src.name,
        })
    downy = _read_manifest("lettuce_downy_v1")
    new_split = _grouped_split(downy)
    for r in downy:
        src = ROOT / r["caminho"].replace("\\", "/")
        moves.append({
            "src": src, "split": new_split[r["caminho"]], "label": LABELS[r["classe"]],
            "dest_name": f"roboflow_lettuce_downy_v1__{src.name}",
            "fonte": "lettuce_downy_v1", "grupo": _photo_key(src.name),
        })
    return moves


def main():
    ap = argparse.ArgumentParser(description="Merge dos lotes da Fase 6 (ciclo 4)")
    ap.add_argument("--apply", action="store_true", help="copia os ficheiros (sem isto é dry-run)")
    args = ap.parse_args()

    moves = plan()
    missing = [m["src"] for m in moves if not m["src"].exists()]
    if missing:
        sys.exit(f"[ERRO] {len(missing)} ficheiros do manifesto não existem, ex.: {missing[:3]}")

    # Garantia: nenhuma foto original em mais de um split
    by_group = defaultdict(set)
    for m in moves:
        by_group[(m["fonte"], m["grupo"])].add(m["split"])
    crossing = [g for g, s in by_group.items() if len(s) > 1]
    if crossing:
        sys.exit(f"[ERRO] {len(crossing)} fotos originais em mais de um split, ex.: {crossing[:3]}")

    table = Counter((m["fonte"], m["split"], m["label"]) for m in moves)
    print(f"{'fonte':<22}{'split':<7}{'classe':<11}{'imagens':>8}")
    for (fonte, split, label), n in sorted(table.items()):
        print(f"{fonte:<22}{split:<7}{label:<11}{n:>8}")
    groups = Counter((m["fonte"], m["split"]) for m in {(m["fonte"], m["grupo"]): m for m in moves}.values())
    print("\nfotos originais distintas do lettuce_downy_v1 por split:",
          {s: groups[("lettuce_downy_v1", s)] for s in SPLITS})

    if not args.apply:
        print("\n[dry-run] nada copiado. Use --apply.")
        return

    copied = skipped = 0
    for m in moves:
        dest = DATA / m["split"] / m["label"] / m["dest_name"]
        m["dest"] = dest
        if dest.exists():
            skipped += 1
            continue
        shutil.copy2(m["src"], dest)
        copied += 1

    with open(DATA / "ciclo4_merge_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["origem", "destino", "fonte", "grupo", "split", "classe"])
        for m in moves:
            w.writerow([m["src"].relative_to(ROOT).as_posix(), m["dest"].relative_to(BACKEND).as_posix(),
                        m["fonte"], m["grupo"], m["split"], m["label"]])

    meta_path = DATA / "split_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["ciclo4_merge"] = {
        "descricao": "Lotes da Fase 6 acrescentados ao split do ciclo 3, sem refazer o split existente",
        "seed": SEED,
        "images": {f"{f}/{s}/{l}": n for (f, s, l), n in sorted(table.items())},
        "lettuce_downy_v1_resplit": "agrupado por foto original (49 fotos → recortes/variantes Roboflow)",
        "manifest": "data/ciclo4_merge_manifest.csv",
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ncopiados: {copied}  já existentes: {skipped}")
    print(f"manifesto: {DATA / 'ciclo4_merge_manifest.csv'}")


if __name__ == "__main__":
    main()
