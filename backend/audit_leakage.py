#!/usr/bin/env python
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Auditoria de vazamento de dados (data leakage) entre os splits.

O QUE ISTO DETECTA
──────────────────
Uma imagem de teste que seja duplicado exacto — ou variante por augmentation
(rotação, flip, jitter) — de uma imagem de treino. Nesse caso o modelo já viu
aquela foto: a accuracy medida sobre ela não mede generalização, mede memória.

Esta é a causa provável de uma accuracy de teste ~99% que não se reproduz em
fotos reais.

USO
───
    python audit_leakage.py                      # audita ./data
    python audit_leakage.py --data ./data
    python audit_leakage.py --max-distance 0     # só duplicados exactos (mais estrito)

SAÍDA
─────
    results/leakage_report.json   — números para citar no TCC
    Lista de exemplos de pares (teste ↔ treino) para inspecção visual.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from imagehash_utils import (
    IMAGE_EXTENSIONS,
    group_by_similarity,
    group_stats,
    hash_images,
)

BASE_DIR = Path(__file__).parent
SPLITS = ("train", "val", "test")
LABELS = ("healthy", "anomalous")


def collect_split_images(data_dir: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for split in SPLITS:
        imgs: list[Path] = []
        for label in LABELS:
            folder = data_dir / split / label
            if folder.is_dir():
                imgs.extend(
                    f for f in folder.iterdir()
                    if f.suffix.lower() in IMAGE_EXTENSIONS
                )
        out[split] = imgs
    return out


def label_of(path: Path) -> str:
    return path.parent.name


_BITS = None


def cruzamentos_rapidos(orbits: dict, split_of: dict, max_distance: int,
                        lote: int = 2000, max_pares: int = 20_000_000) -> dict:
    """
    Só o que decide o veredicto: para cada imagem de val e teste, há alguma de treino a
    distância de órbita <= max_distance? Mesmo critério de orbit_distance (qualquer das
    oito orientações contra qualquer das oito), mas exacto e vectorizado: pelo princípio
    da casa dos pombos, duas hashes de 64 bits a distância <= 4 coincidem inteiras em pelo
    menos uma de 5 fatias, e só os pares que coincidem numa fatia têm a distância medida.
    O agrupador completo compara par a par em Python e não termina com ~150 mil imagens
    que têm milhares de folhas quase iguais sobre fundo branco.
    """
    import numpy as np
    global _BITS
    if _BITS is None:
        _BITS = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    n_fatias = max_distance + 1
    larg = [64 // n_fatias + (1 if k < 64 % n_fatias else 0) for k in range(n_fatias)]
    desloc = np.cumsum([0] + larg[:-1])

    def fatia(h, k):
        return (h >> np.uint64(desloc[k])) & np.uint64((1 << larg[k]) - 1)

    treino = [p for p in orbits if split_of[p] == "train"]
    T = np.array([orbits[p] for p in treino], dtype=np.uint64).reshape(-1)    # N*8 hashes
    dono_T = np.repeat(np.arange(len(treino)), 8)
    rot_treino = np.array([label_of(p) for p in treino])
    indices = []
    for k in range(n_fatias):
        chave = fatia(T, k)
        ordem = np.argsort(chave, kind="stable")
        indices.append((chave[ordem], ordem))

    out = {}
    for split in ("val", "test"):
        alvos = [p for p in orbits if split_of[p] == split]
        A = np.array([orbits[p] for p in alvos], dtype=np.uint64).reshape(-1)
        dono_A = np.repeat(np.arange(len(alvos)), 8)
        achados = []                                         # pares (alvo, treino) a distância <= max
        for i in range(0, len(A), lote * 8):
            a, da = A[i:i + lote * 8], dono_A[i:i + lote * 8]
            for k, (chaves, ordem) in enumerate(indices):
                ca = fatia(a, k)
                lo = np.searchsorted(chaves, ca, "left")
                hi = np.searchsorted(chaves, ca, "right")
                cont = hi - lo
                # expande os intervalos em pares, aos pedaços para caber na memória
                fim = np.cumsum(cont)
                ini = 0
                while ini < len(a):
                    lim = np.searchsorted(fim, (fim[ini - 1] if ini else 0) + max_pares, "right")
                    lim = max(lim, ini + 1)
                    c = cont[ini:lim]
                    if c.sum():
                        r = np.repeat(np.arange(ini, lim), c)
                        base = np.repeat(lo[ini:lim] - np.concatenate(([0], np.cumsum(c)[:-1])), c)
                        t = ordem[base + np.arange(c.sum())]
                        x = a[r] ^ T[t]
                        d = _BITS[x.view(np.uint8).reshape(-1, 8)].sum(axis=1)
                        ok = d <= max_distance
                        if ok.any():
                            achados.append(np.stack([da[r[ok]], dono_T[t[ok]]], axis=1))
                    ini = lim
            feitos = min(i // 8 + lote, len(alvos))
            n_viz = len(np.unique(np.concatenate(achados)[:, 0])) if achados else 0
            print(f"    {split}: {feitos}/{len(alvos)} ({n_viz} com quase-duplicata no treino)", flush=True)
        vazadas, conflitos, pares = [], 0, []
        if achados:
            P = np.unique(np.concatenate(achados), axis=0)
            for j in np.unique(P[:, 0]):
                viz = P[P[:, 0] == j, 1]
                p = alvos[j]
                vazadas.append(p)
                pares.append((p, treino[viz[0]]))
                if (rot_treino[viz] != label_of(p)).any():
                    conflitos += 1
        out[split] = (vazadas, conflitos, pares)
    return out


def main():
    parser = argparse.ArgumentParser(description="Auditoria de data leakage — FitoVision")
    parser.add_argument("--data", default=str(BASE_DIR / "data"))
    parser.add_argument("--results-dir", default=str(BASE_DIR / "results"))
    parser.add_argument("--max-distance", type=int, default=4,
                        help="Distância de Hamming máxima para considerar near-duplicate "
                             "(0 = só duplicados exactos; 4 = default, apanha augmentation)")
    parser.add_argument("--examples", type=int, default=15,
                        help="Quantos pares de exemplo listar")
    parser.add_argument("--hash-cache", default=None,
                        help="Cache de hashes a usar (padrão: data/hash_cache.jsonl). O ciclo 5 "
                             "usa um próprio, semeado por prepara_ciclo5.py, para não refazer os hashes")
    parser.add_argument("--rapido", action="store_true",
                        help="Só val/teste contra treino, em bloco com numpy (sem as estatísticas de "
                             "redundância interna); o agrupador completo não termina no data_c5")
    args = parser.parse_args()

    data_dir = Path(args.data)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        sys.exit(f"[ERRO] '{data_dir}' não existe. Execute download_datasets.py primeiro.")

    print("\n" + "=" * 64)
    print("  AUDITORIA DE DATA LEAKAGE — FitoVision")
    print("=" * 64)

    split_images = collect_split_images(data_dir)
    for split in SPLITS:
        print(f"  {split:<6}: {len(split_images[split]):>7} imagens")

    all_images = [p for split in SPLITS for p in split_images[split]]
    if not all_images:
        sys.exit("\n[ERRO] Nenhuma imagem encontrada. Verifique a estrutura data/<split>/<label>/.")

    split_of: dict[Path, str] = {}
    for split in SPLITS:
        for p in split_images[split]:
            split_of[p] = split

    print(f"\n  Calculando hashes perceptuais de {len(all_images)} imagens...")
    orbits, corrupted = (hash_images(all_images, cache_path=Path(args.hash_cache)) if args.hash_cache
                         else hash_images(all_images))

    if corrupted:
        print(f"\n  ⚠  {len(corrupted)} imagens corrompidas/ilegíveis (serão ignoradas):")
        for p in corrupted[:5]:
            print(f"       {p}")
        if len(corrupted) > 5:
            print(f"       ... e mais {len(corrupted) - 5}")

    if args.rapido:
        auditoria_rapida(args, orbits, split_of, split_images, corrupted, results_dir)
        return

    print(f"\n  Agrupando near-duplicates (max_distance={args.max_distance})...")
    groups = group_by_similarity(orbits, max_distance=args.max_distance)

    stats = group_stats(groups)
    print("\n" + "-" * 64)
    print("  REDUNDÂNCIA INTERNA DO DATASET")
    print("-" * 64)
    print(f"  Imagens (legíveis)         : {stats['n_images']:>8}")
    print(f"  Fotos distintas (grupos)   : {stats['n_groups']:>8}")
    print(f"  Grupos com duplicados      : {stats['n_duplicate_groups']:>8}")
    print(f"  Imagens redundantes        : {stats['n_redundant_images']:>8}")
    print(f"  Maior grupo (nº de cópias) : {stats['largest_group']:>8}")
    print(f"  Redundância                : {stats['redundancy_pct']:>7.1f}%")

    # ── Grupos que cruzam splits = vazamento ──────────────────────────────────
    group_to_splits: dict[int, set[str]] = defaultdict(set)
    group_to_paths: dict[int, list[Path]] = defaultdict(list)
    for p, gid in groups.items():
        group_to_splits[gid].add(split_of[p])
        group_to_paths[gid].append(p)

    leaked_test: list[Path] = []
    leaked_val: list[Path] = []
    leak_pairs: list[tuple[Path, Path]] = []

    for gid, splits_present in group_to_splits.items():
        if "train" not in splits_present:
            continue
        members = group_to_paths[gid]
        train_members = [p for p in members if split_of[p] == "train"]
        for p in members:
            if split_of[p] == "test":
                leaked_test.append(p)
                if len(leak_pairs) < args.examples:
                    leak_pairs.append((p, train_members[0]))
            elif split_of[p] == "val":
                leaked_val.append(p)

    n_test = len(split_images["test"])
    n_val = len(split_images["val"])
    test_leak_pct = (100.0 * len(leaked_test) / n_test) if n_test else 0.0
    val_leak_pct = (100.0 * len(leaked_val) / n_val) if n_val else 0.0

    print("\n" + "-" * 64)
    print("  VAZAMENTO ENTRE SPLITS")
    print("-" * 64)
    print(f"  Imagens de TESTE com duplicado no treino : {len(leaked_test):>7} / {n_test}  ({test_leak_pct:.1f}%)")
    print(f"  Imagens de VAL   com duplicado no treino : {len(leaked_val):>7} / {n_val}  ({val_leak_pct:.1f}%)")

    # Vazamento com troca de label = ruído de rotulagem (mesma foto, labels diferentes)
    label_conflicts = 0
    for gid, members in group_to_paths.items():
        labels = {label_of(p) for p in members}
        if len(labels) > 1:
            label_conflicts += 1

    print(f"  Grupos com labels CONTRADITÓRIOS        : {label_conflicts:>7}"
          f"   (mesma foto rotulada healthy E anomalous)")

    # ── Veredicto ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    if test_leak_pct >= 20:
        print("  ❌ VAZAMENTO GRAVE")
        print()
        print(f"  {test_leak_pct:.1f}% do conjunto de teste é composto por imagens que o modelo")
        print("  já viu no treino (duplicados exactos ou variantes por augmentation).")
        print()
        print("  A accuracy reportada NÃO mede generalização — mede memorização.")
        print("  É por isso que o modelo falha em fotos novas.")
        print()
        print("  CORREÇÃO: refazer o split agrupando por identidade visual")
        print("            → python download_datasets.py --skip-download")
        print("              (o split passou a ser group-aware)")
    elif test_leak_pct >= 5:
        print("  ⚠  VAZAMENTO MODERADO")
        print(f"     {test_leak_pct:.1f}% do teste tem duplicado no treino. As métricas estão")
        print("     optimistas. Recomenda-se refazer o split.")
    else:
        print("  ✅ SEM VAZAMENTO SIGNIFICATIVO")
        print(f"     Apenas {test_leak_pct:.1f}% do teste tem duplicado no treino.")
        print("     As métricas de teste são confiáveis.")
    print("=" * 64)

    if leak_pairs:
        print("\n  Exemplos de pares vazados (teste ↔ treino):")
        for test_p, train_p in leak_pairs:
            print(f"    TESTE  {test_p.parent.name}/{test_p.name}")
            print(f"    TREINO {train_p.parent.name}/{train_p.name}")
            print()

    report = {
        "max_distance": args.max_distance,
        "n_images": stats["n_images"],
        "n_distinct_photos": stats["n_groups"],
        "redundancy_pct": round(stats["redundancy_pct"], 2),
        "largest_group": stats["largest_group"],
        "n_corrupted": len(corrupted),
        "test": {
            "total": n_test,
            "leaked": len(leaked_test),
            "leaked_pct": round(test_leak_pct, 2),
        },
        "val": {
            "total": n_val,
            "leaked": len(leaked_val),
            "leaked_pct": round(val_leak_pct, 2),
        },
        "label_conflict_groups": label_conflicts,
        "verdict": (
            "GRAVE" if test_leak_pct >= 20
            else "MODERADO" if test_leak_pct >= 5
            else "OK"
        ),
    }

    out = results_dir / "leakage_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Relatório salvo: {out}")


def auditoria_rapida(args, orbits, split_of, split_images, corrupted, results_dir: Path):
    print(f"\n  Val e teste contra o treino, em bloco (max_distance={args.max_distance})...")
    res = cruzamentos_rapidos(orbits, split_of, args.max_distance)
    n = {s: len(split_images[s]) for s in ("val", "test")}
    pct = {s: (100.0 * len(res[s][0]) / n[s]) if n[s] else 0.0 for s in n}
    print("\n" + "-" * 64)
    print("  VAZAMENTO ENTRE SPLITS (modo rápido)")
    print("-" * 64)
    print(f"  Imagens de TESTE com duplicado no treino : {len(res['test'][0]):>7} / {n['test']}  ({pct['test']:.1f}%)")
    print(f"  Imagens de VAL   com duplicado no treino : {len(res['val'][0]):>7} / {n['val']}  ({pct['val']:.1f}%)")
    print(f"  ... e rótulo diferente do vizinho de treino: teste {res['test'][1]}, val {res['val'][1]}")
    verdict = "GRAVE" if pct["test"] >= 20 else "MODERADO" if pct["test"] >= 5 else "OK"
    print(f"\n  Veredicto: {verdict}")
    for test_p, train_p in res["test"][2][:args.examples]:
        print(f"    TESTE  {test_p.parent.name}/{test_p.name}\n    TREINO {train_p.parent.name}/{train_p.name}\n")
    report = {
        "modo": "rapido (val/teste contra treino; sem redundância interna)",
        "max_distance": args.max_distance,
        "n_images": len(orbits),
        "n_corrupted": len(corrupted),
        "test": {"total": n["test"], "leaked": len(res["test"][0]), "leaked_pct": round(pct["test"], 2),
                 "label_conflicts": res["test"][1]},
        "val": {"total": n["val"], "leaked": len(res["val"][0]), "leaked_pct": round(pct["val"], 2),
                "label_conflicts": res["val"][1]},
        "exemplos_teste": [[f"{a.parent.name}/{a.name}", f"{b.parent.name}/{b.name}"]
                           for a, b in res["test"][2][:200]],
        "verdict": verdict,
    }
    out = results_dir / "leakage_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Relatório salvo: {out}")


if __name__ == "__main__":
    main()
