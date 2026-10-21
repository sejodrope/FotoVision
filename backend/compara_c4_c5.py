#!/usr/bin/env python
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
"""
Ciclo 4 × ciclo 5, por subconjunto, nas mesmas imagens de teste (as que não mudaram de
partição). Complementa o avalia_ciclo5.py, que dá o McNemar só no total: aqui saem a
acurácia balanceada, a especificidade e a sensibilidade de cada ciclo em cada subconjunto,
e o McNemar exato de cada um.

Uso:
    python compara_c4_c5.py                       # modelos com previsões nos dois ciclos
    python compara_c4_c5.py --modelos efficientnet_b0
Saída:
    results/ciclo5/ciclo4_x_ciclo5.md / .json
"""

import argparse
import csv
import json
from pathlib import Path

from scipy.stats import binomtest

BASE = Path(__file__).parent
MODELS = ("efficientnet_b0", "vit_b_16", "mobilenet_v2", "resnet50")
SUBCONJUNTOS = ("coletanea_lavoura", "coletanea_demais", "plantvillage", "espinafre")


def taxas(linhas: list[dict], chave: str) -> dict:
    s = [r for r in linhas if r["v"] == "healthy"]
    a = [r for r in linhas if r["v"] == "anomalous"]
    acerto = lambda g: sum(r[chave] == r["v"] for r in g) / len(g) if g else float("nan")
    espec, sens = acerto(s), acerto(a)
    return {"bal": (espec + sens) / 2, "espec": espec, "sens": sens}


def comparar(modelo: str, manifesto: dict) -> dict | None:
    p4 = BASE / "results" / "arquiteturas" / "predicoes" / f"{modelo}.csv"
    p5 = BASE / "results" / "ciclo5" / "predicoes" / f"{modelo}.csv"
    if not (p4.exists() and p5.exists()):
        return None
    with open(p4, encoding="utf-8") as f:
        c4 = {r["chave"].split("/", 1)[1]: r["pred"] for r in csv.DictReader(f)}
    with open(p5, encoding="utf-8") as f:
        c5 = [r for r in csv.DictReader(f) if r["split"] == "test" and r["arquivo"] in c4]
    out = {}
    for sub in (*SUBCONJUNTOS, "todas_comuns"):
        linhas = [{"v": r["verdade"], "p5": r["pred"], "p4": c4[r["arquivo"]]}
                  for r in c5 if sub == "todas_comuns" or manifesto.get(r["arquivo"]) == sub]
        so4 = sum(r["p4"] == r["v"] != r["p5"] for r in linhas)
        so5 = sum(r["p5"] == r["v"] != r["p4"] for r in linhas)
        out[sub] = {"n": len(linhas), "ciclo4": taxas(linhas, "p4"), "ciclo5": taxas(linhas, "p5"),
                    "so_ciclo4_acerta": so4, "so_ciclo5_acerta": so5,
                    "p_mcnemar": float(binomtest(so4, so4 + so5).pvalue) if so4 + so5 else 1.0}
    return out


def main():
    ap = argparse.ArgumentParser(description="Ciclo 4 × ciclo 5 por subconjunto")
    ap.add_argument("--modelos", nargs="+", default=list(MODELS))
    args = ap.parse_args()
    with open(BASE / "data_c5" / "manifesto_ciclo5.csv", encoding="utf-8") as f:
        manifesto = {r["arquivo"]: r["tipo"] for r in csv.DictReader(f)}
    res = {m: r for m in args.modelos if (r := comparar(m, manifesto))}
    pct = lambda x: f"{100 * x:.1f}"
    md = ["# Ciclo 4 × ciclo 5 nas mesmas imagens de teste", "",
          "Gerado por `backend/compara_c4_c5.py`. Bal. / espec. / sens. em %; McNemar exato "
          "sobre o acerto por imagem.", ""]
    for m, subs in res.items():
        md += [f"## {m}", "", "| Subconjunto | N | Ciclo 4 | Ciclo 5 | Só c4 / só c5 | p |",
               "|---|---|---|---|---|---|"]
        for sub, d in subs.items():
            c4, c5 = d["ciclo4"], d["ciclo5"]
            md.append(f"| {sub} | {d['n']} | {pct(c4['bal'])} / {pct(c4['espec'])} / {pct(c4['sens'])} | "
                      f"{pct(c5['bal'])} / {pct(c5['espec'])} / {pct(c5['sens'])} | "
                      f"{d['so_ciclo4_acerta']} / {d['so_ciclo5_acerta']} | {d['p_mcnemar']:.3g} |")
        md.append("")
    out = BASE / "results" / "ciclo5"
    (out / "ciclo4_x_ciclo5.md").write_text("\n".join(md), encoding="utf-8")
    (out / "ciclo4_x_ciclo5.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
