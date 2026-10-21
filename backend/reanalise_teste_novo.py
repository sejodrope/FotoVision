"""
Reanálise de 08/10/2026 — o que o conjunto de teste realmente mede.

Não treina nem roda inferência: só relê artefatos já gerados e responde a três
perguntas da revisão crítica (§11 do docs/CORRECOES_METODOLOGICAS.md):

  1. O teste vem das mesmas fontes do treino? (data/split_metadata.json)
  2. O ganho do ciclo 4 no "teste novo" sobrevive sem os recortes de míldio,
     que vêm de só 5 fotos e não são observações independentes?
     (results/ciclo4/comparacao_pareada_teste_novo.csv)
  3. Em campo, a confiança avisa quando o modelo erra?
     (results/campo_resultados*.csv)

Uso:
    python reanalise_teste_novo.py
Saída:
    results/ciclo4/reanalise_por_lote.json
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score

BASE = Path(__file__).resolve().parent
MODELOS = ["efficientnet_b0", "resnet50", "mobilenet_v2"]
# As duas fontes tratadas em 08/10 como "derivadas do PlantVillage". A auditoria de 13/10
# (auditoria_consistencia.py) mostrou que a segunda é uma coletânea: 37% dela são culturas
# que o PlantVillage não tem, parte fotografada em lavoura. O "pct_derivado_plantvillage"
# calculado abaixo (70,1%) superestima; o valor por cultura é no máximo 47,7%.
FONTES_PLANTVILLAGE = {"plantvillage", "nirmalsankalana_plant-diseases-training-dataset"}
CAMPO = {
    "ciclo1": "campo_resultados.csv",
    "ciclo2": "campo_resultados_v2.csv",
    "ciclo3": "campo_resultados_v3.csv",
    "ciclo4": "campo_resultados_v4.csv",
}


def composicao_por_fonte() -> dict:
    meta = json.loads((BASE / "data/split_metadata.json").read_text(encoding="utf-8"))
    out = {}
    for split, fontes in meta["sources_per_split"].items():
        total = sum(fontes.values())
        out[split] = {
            "total": total,
            "pct_por_fonte": {k: round(100 * v / total, 1) for k, v in sorted(fontes.items())},
            "pct_derivado_plantvillage": round(
                100 * sum(v for k, v in fontes.items() if k in FONTES_PLANTVILLAGE) / total, 1
            ),
        }
    return out


def holm(ps: list[float]) -> list[float]:
    ordem = sorted(range(len(ps)), key=lambda i: ps[i])
    ajustado = [0.0] * len(ps)
    anterior = 0.0
    for pos, i in enumerate(ordem):
        anterior = max(anterior, min(1.0, ps[i] * (len(ps) - pos)))
        ajustado[i] = anterior
    return ajustado


def lote(ficheiro: str) -> str:
    return "espinafre" if "spinach" in ficheiro else "mildio"


def teste_novo_por_lote() -> dict:
    linhas = list(csv.DictReader(
        (BASE / "results/ciclo4/comparacao_pareada_teste_novo.csv").open(encoding="utf-8")))
    grupo = {
        r["destino"].replace("/", "\\"): r["grupo"]
        for r in csv.DictReader((BASE / "data/ciclo4_merge_manifest.csv").open(encoding="utf-8"))
    }
    out = {}
    for nome in ("espinafre", "mildio", "todos"):
        rs = [r for r in linhas if nome == "todos" or lote(r["ficheiro"]) == nome]
        y = [r["verdade"] for r in rs]
        res, ps = {}, []
        for m in MODELOS:
            a = [int(r[f"{m}_ciclo3_acertou"]) for r in rs]
            b = [int(r[f"{m}_ciclo4_acertou"]) for r in rs]
            so3 = sum(1 for x, z in zip(a, b) if x and not z)
            so4 = sum(1 for x, z in zip(a, b) if z and not x)
            p = binomtest(min(so3, so4), so3 + so4, 0.5).pvalue if so3 + so4 else 1.0
            ps.append(p)
            # Com uma só classe (míldio é 100% anômalo) a acurácia balanceada
            # degenera na revocação dessa classe.
            if len(set(y)) > 1:
                ba3 = balanced_accuracy_score(y, [r[f"{m}_ciclo3_pred"] for r in rs])
                ba4 = balanced_accuracy_score(y, [r[f"{m}_ciclo4_pred"] for r in rs])
            else:
                ba3, ba4 = sum(a) / len(a), sum(b) / len(b)
            res[m] = {
                "bal_acc_ciclo3": round(ba3, 4), "bal_acc_ciclo4": round(ba4, 4),
                "so_ciclo3_acerta": so3, "so_ciclo4_acerta": so4, "p_mcnemar_exato": p,
            }
        for m, ph in zip(MODELOS, holm(ps)):
            res[m]["p_holm"] = ph
        out[nome] = {"n_imagens": len(rs), "modelos": res}

    fotos = defaultdict(int)
    for r in linhas:
        if lote(r["ficheiro"]) == "mildio":
            fotos[grupo.get(r["ficheiro"], "?")] += 1
    out["mildio"]["n_fotos_originais"] = len(fotos)
    out["mildio"]["recortes_por_foto"] = dict(fotos)
    out["mildio"]["aviso"] = (
        "Recortes da mesma foto não são independentes; o McNemar por imagem "
        "superestima a evidência. Tratar como indicação."
    )
    return out


def confianca_em_campo() -> dict:
    out = {}
    for ciclo, nome in CAMPO.items():
        caminho = BASE / "results" / nome
        if not caminho.exists():
            continue
        rs = list(csv.DictReader(caminho.open(encoding="utf-8")))
        conf = [float(r["confianca"]) for r in rs]
        erros = [r for r in rs if r["resposta"] in ("healthy", "anomalous")
                 and r["resposta"] != r["verdade"]]
        out[ciclo] = {
            "n": len(rs),
            "confianca_min": min(conf),
            "confianca_media": round(sum(conf) / len(conf), 4),
            "abstencoes": sum(1 for r in rs if r["resposta"] == "inconclusive"),
            "erros_emitidos": len(erros),
            "confianca_dos_erros": [float(r["confianca"]) for r in erros],
        }
    return out


def main() -> None:
    resultado = {
        "composicao_por_fonte": composicao_por_fonte(),
        "teste_novo_por_lote": teste_novo_por_lote(),
        "confianca_em_campo": confianca_em_campo(),
    }
    destino = BASE / "results/ciclo4/reanalise_por_lote.json"
    destino.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Fontes derivadas do PlantVillage por partição:")
    for split, d in resultado["composicao_por_fonte"].items():
        print(f"  {split:5s} {d['pct_derivado_plantvillage']:5.1f}%  (n={d['total']})")
    print("\nCiclo 3 -> ciclo 4, acurácia balanceada e p de Holm:")
    for nome, d in resultado["teste_novo_por_lote"].items():
        print(f"  {nome} (n={d['n_imagens']})")
        for m, r in d["modelos"].items():
            print(f"    {m:16s} {r['bal_acc_ciclo3']:.3f} -> {r['bal_acc_ciclo4']:.3f}"
                  f"   p_holm={r['p_holm']:.1e}")
    print("\nConfiança em campo:")
    for ciclo, d in resultado["confianca_em_campo"].items():
        print(f"  {ciclo}: média {d['confianca_media']:.3f}, abstenções {d['abstencoes']}/{d['n']},"
              f" confiança dos erros {d['confianca_dos_erros']}")
    print(f"\nGravado em {destino}")


if __name__ == "__main__":
    main()
