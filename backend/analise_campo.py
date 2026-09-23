#!/usr/bin/env python
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Análise estatística do teste de campo — plano de análise da Parcial I do TCC-II.

Recebe os CSV que o test_campo.py produz e calcula tudo o que o artigo promete,
sem precisar de statsmodels (só numpy + scipy):

  • Acurácia balanceada + IC 95% por bootstrap estratificado (10.000 reamostragens)
  • IC de Wilson 95% sobre a acurácia bruta entre os diagnósticos emitidos
    (abstenções excluídas — o MESMO critério para todas as versões, ver §10.7 do
    CORRECOES_METODOLOGICAS.md, onde critérios diferentes invalidaram a comparação)
  • Taxa de abstenção e de rejeição por ExG
  • ECE (15 faixas) sobre os diagnósticos emitidos
  • Curva risco–cobertura (varre o limiar de confiança) + AURC
  • McNemar exacto entre versões, nas MESMAS fotos, com correcção de Holm
  • H1 cultura × acerto      → Fisher/Freeman-Halton
  • H2 cromático × estrutural → Fisher 2×2
  • H3 distância (close < média < ampla) → Cochran-Armitage de tendência

As hipóteses H1-H3 precisam do CSV de metadados por foto (--metadados); sem ele
essas secções são saltadas com um aviso. Modelo em campo/metadados_campo.csv.

USO
───
    # uma versão
    python analise_campo.py --csv results/campo_resultados_v4.csv

    # comparando versões nas mesmas fotos (McNemar + Holm)
    python analise_campo.py --csv results/campo_resultados_v3.csv:ciclo3 \\
                            results/campo_resultados_v4.csv:ciclo4

    # com metadados, para testar H1-H3
    python analise_campo.py --csv results/campo_resultados_v4.csv \\
                            --metadados campo/metadados_campo.csv

SAÍDA
─────
    results/analise_campo.md    — relatório pronto para colar no TCC
    results/analise_campo.json  — os mesmos números em JSON
    results/risco_cobertura_<versão>.csv
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, fisher_exact, norm

BASE_DIR = Path(__file__).parent
SEED     = 42
N_BOOT   = 10_000
N_BINS   = 15
LEAF_TRUTHS = ("healthy", "anomalous")
DISTANCIAS  = ("close", "media", "ampla")   # ordem ordinal para a tendência


# ─────────────────────────────────────────────────────────────────────────────
# Leitura
# ─────────────────────────────────────────────────────────────────────────────
def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["ficheiro"] = r["ficheiro"].replace("\\", "/")
        r["cultura"] = (r.get("cultura") or "").strip().lower()
        try:
            r["confianca"] = float(r["confianca"]) if r.get("confianca") else None
        except ValueError:
            r["confianca"] = None
    return rows


def load_metadados(path: Path) -> dict[str, dict]:
    meta = {}
    for r in load_csv(path):
        meta[Path(r["ficheiro"]).name] = r
    return meta


def emitted(rows: list[dict]) -> list[dict]:
    """Só folhas reais com diagnóstico emitido (exclui abstenção e not_a_leaf)."""
    return [r for r in rows
            if r["verdade"] in LEAF_TRUTHS and r["resposta"] in LEAF_TRUTHS]


def is_correct(r: dict) -> bool:
    return r["resposta"] == r["verdade"]


# ─────────────────────────────────────────────────────────────────────────────
# Estatística
# ─────────────────────────────────────────────────────────────────────────────
def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """IC de Wilson para uma proporção — fiável com N pequeno, ao contrário do normal."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    d = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / d
    meio = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, centro - meio), min(1.0, centro + meio))


def balanced_accuracy(rows: list[dict]) -> float:
    """Média das acurácias por classe verdadeira (só diagnósticos emitidos)."""
    per_class = []
    for truth in LEAF_TRUTHS:
        sub = [r for r in rows if r["verdade"] == truth]
        if sub:
            per_class.append(sum(is_correct(r) for r in sub) / len(sub))
    return float(np.mean(per_class)) if per_class else float("nan")


def bootstrap_bal_acc(rows: list[dict], n_boot: int = N_BOOT) -> dict:
    """IC percentil por bootstrap ESTRATIFICADO (reamostra dentro de cada classe,
    preservando o nº de fotos por classe — com N pequeno isso evita reamostragens
    degeneradas onde uma das classes desaparece)."""
    rng = np.random.default_rng(SEED)
    por_classe = {t: np.array([is_correct(r) for r in rows if r["verdade"] == t])
                  for t in LEAF_TRUTHS}
    por_classe = {t: v for t, v in por_classe.items() if len(v)}
    if not por_classe:
        return {"ponto": float("nan"), "ic95": [float("nan")] * 2, "n_boot": 0}
    amostras = np.empty(n_boot)
    for i in range(n_boot):
        accs = [rng.choice(v, size=len(v), replace=True).mean() for v in por_classe.values()]
        amostras[i] = float(np.mean(accs))
    return {
        "ponto": balanced_accuracy(rows),
        "ic95": [float(np.percentile(amostras, 2.5)), float(np.percentile(amostras, 97.5))],
        "n_boot": n_boot,
    }


def ece(rows: list[dict], n_bins: int = N_BINS) -> float:
    """Expected Calibration Error: |confiança média − acerto médio| por faixa,
    ponderado pelo nº de fotos. 0 = confiança honesta."""
    conf = np.array([r["confianca"] for r in rows if r["confianca"] is not None])
    acer = np.array([is_correct(r) for r in rows if r["confianca"] is not None], dtype=float)
    if not len(conf):
        return float("nan")
    limites = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(limites[:-1], limites[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(acer[m].mean() - conf[m].mean())
    return float(total)


def risco_cobertura(rows: list[dict]) -> list[dict]:
    """Para cada limiar de confiança: que fracção das fotos recebe diagnóstico
    (cobertura) e qual a acurácia nessas (acurácia selectiva). AURC = área sob a
    curva de risco — quanto menor, melhor a ordenação por confiança."""
    dados = [(r["confianca"], is_correct(r)) for r in rows if r["confianca"] is not None]
    if not dados:
        return []
    dados.sort(key=lambda x: -x[0])
    n = len(dados)
    curva, acertos = [], 0
    for i, (conf, ok) in enumerate(dados, start=1):
        acertos += ok
        curva.append({
            "limiar": round(conf, 4),
            "cobertura": i / n,
            "acuracia_selectiva": acertos / i,
            "risco": 1 - acertos / i,
            "n": i,
        })
    return curva


def aurc(curva: list[dict]) -> float:
    if not curva:
        return float("nan")
    return float(np.mean([p["risco"] for p in curva]))


def mcnemar_exacto(a: list[dict], b: list[dict]) -> dict:
    """McNemar exacto (binomial) entre duas versões nas MESMAS fotos.
    Só entram fotos com diagnóstico emitido nas DUAS versões — senão a comparação
    mistura mudança de acerto com mudança de abstenção."""
    ia = {r["ficheiro"]: r for r in emitted(a)}
    ib = {r["ficheiro"]: r for r in emitted(b)}
    comuns = sorted(set(ia) & set(ib))
    b01 = sum(1 for f in comuns if is_correct(ia[f]) and not is_correct(ib[f]))
    b10 = sum(1 for f in comuns if not is_correct(ia[f]) and is_correct(ib[f]))
    discordantes = b01 + b10
    p = binomtest(b01, discordantes, 0.5).pvalue if discordantes else 1.0
    return {
        "n_pareado": len(comuns),
        "so_A_acerta": b01,
        "so_B_acerta": b10,
        "discordantes": discordantes,
        "p": float(p),
        "excluidas_por_abstencao": len(set(ia) ^ set(ib)),
    }


def holm(ps: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Correcção de Holm-Bonferroni — controla o erro tipo I ao testar várias
    hipóteses no mesmo conjunto de fotos."""
    itens = sorted(ps.items(), key=lambda kv: kv[1])
    m = len(itens)
    out, maior = {}, 0.0
    for i, (nome, p) in enumerate(itens):
        ajustado = min(1.0, max(maior, (m - i) * p))
        maior = ajustado
        out[nome] = {"p": p, "p_holm": ajustado, "significativo": ajustado < alpha}
    return out


def cochran_armitage(tabela: list[tuple[str, int, int]]) -> dict:
    """Tendência em proporções ordenadas (close < média < ampla).
    tabela: [(nome, acertos, total)] na ordem ordinal."""
    niveis = np.arange(len(tabela), dtype=float)
    acertos = np.array([t[1] for t in tabela], dtype=float)
    totais = np.array([t[2] for t in tabela], dtype=float)
    n = totais.sum()
    if n == 0 or acertos.sum() in (0, n):
        return {"z": float("nan"), "p": float("nan"), "nota": "sem variação suficiente"}
    p_geral = acertos.sum() / n
    media_x = (niveis * totais).sum() / n
    t = (acertos * (niveis - media_x)).sum()
    var = p_geral * (1 - p_geral) * (totais * (niveis - media_x) ** 2).sum()
    if var <= 0:
        return {"z": float("nan"), "p": float("nan"), "nota": "variância nula"}
    z = t / math.sqrt(var)
    return {"z": float(z), "p": float(2 * norm.sf(abs(z)))}


# ─────────────────────────────────────────────────────────────────────────────
# Blocos do relatório
# ─────────────────────────────────────────────────────────────────────────────
def resumo_versao(nome: str, rows: list[dict]) -> dict:
    folhas = [r for r in rows if r["verdade"] in LEAF_TRUTHS]
    emit = emitted(rows)
    n_abst = sum(1 for r in folhas if r["resposta"] == "inconclusive")
    n_exg = sum(1 for r in folhas if r["resposta"] == "not_a_leaf")
    acertos = sum(is_correct(r) for r in emit)
    nao_folha = [r for r in rows if r["verdade"] == "nao_folha"]

    res = {
        "versao": nome,
        "n_fotos": len(rows),
        "n_folhas": len(folhas),
        "n_emitidos": len(emit),
        "n_abstencoes": n_abst,
        "n_rejeitadas_exg": n_exg,
        "acuracia_bruta": acertos / len(emit) if emit else float("nan"),
        "acuracia_bruta_ic95_wilson": list(wilson_ci(acertos, len(emit))),
        "acuracia_balanceada": bootstrap_bal_acc(emit),
        "taxa_abstencao": n_abst / len(folhas) if folhas else float("nan"),
        "taxa_abstencao_ic95_wilson": list(wilson_ci(n_abst, len(folhas))),
        "ece": ece(emit),
        "aurc": aurc(risco_cobertura(emit)),
        "por_classe": {},
        "por_cultura": {},
    }
    for truth in LEAF_TRUTHS:
        sub = [r for r in emit if r["verdade"] == truth]
        k = sum(is_correct(r) for r in sub)
        res["por_classe"][truth] = {"acertos": k, "n": len(sub),
                                    "ic95_wilson": list(wilson_ci(k, len(sub)))}
    culturas = sorted({r["cultura"] for r in emit if r["cultura"]})
    for c in culturas:
        sub = [r for r in emit if r["cultura"] == c]
        k = sum(is_correct(r) for r in sub)
        res["por_cultura"][c] = {
            "acertos": k, "n": len(sub),
            "acuracia_bruta": k / len(sub),
            "ic95_wilson": list(wilson_ci(k, len(sub))),
            "acuracia_balanceada": balanced_accuracy(sub),
        }
    if nao_folha:
        ok = sum(1 for r in nao_folha if r["resposta"] == "not_a_leaf")
        res["guarda_dominio"] = {"rejeitadas_corretamente": ok, "n": len(nao_folha),
                                 "ic95_wilson": list(wilson_ci(ok, len(nao_folha)))}
    return res


def hipoteses(rows: list[dict], meta: dict[str, dict]) -> dict:
    """H1 cultura, H2 cromático×estrutural, H3 distância — só sobre anomalous
    emitidas (é onde o tipo de dano faz sentido)."""
    emit = emitted(rows)
    for r in emit:
        m = meta.get(Path(r["ficheiro"]).name, {})
        r["categoria_dano"] = (m.get("categoria_dano") or "").strip().lower()
        r["distancia"] = (m.get("distancia") or "").strip().lower()
        if not r["cultura"]:
            r["cultura"] = (m.get("cultura") or "").strip().lower()

    out = {}

    # H1 — a acurácia depende da cultura?
    culturas = sorted({r["cultura"] for r in emit if r["cultura"]})
    if len(culturas) >= 2:
        tab = [[sum(1 for r in emit if r["cultura"] == c and is_correct(r)),
                sum(1 for r in emit if r["cultura"] == c and not is_correct(r))]
               for c in culturas]
        arr = np.array(tab).T.tolist()   # linhas = acerto/erro, colunas = cultura
        stat, p = fisher_exact(arr)
        out["H1_cultura"] = {"culturas": culturas, "tabela_acerto_erro": tab,
                             "teste": "Fisher/Freeman-Halton", "p": float(p)}
    else:
        out["H1_cultura"] = {"nota": "precisa de ≥2 culturas com diagnóstico emitido"}

    # H2 — dano cromático vs estrutural (só anomalous)
    anom = [r for r in emit if r["verdade"] == "anomalous"]
    cats = {"cromatico", "estrutural"}
    sub = [r for r in anom if r["categoria_dano"] in cats]
    if len({r["categoria_dano"] for r in sub}) == 2:
        tab = [[sum(1 for r in sub if r["categoria_dano"] == c and is_correct(r)),
                sum(1 for r in sub if r["categoria_dano"] == c and not is_correct(r))]
               for c in ("cromatico", "estrutural")]
        stat, p = fisher_exact(tab)
        out["H2_cromatico_vs_estrutural"] = {
            "tabela": {"cromatico": tab[0], "estrutural": tab[1]},
            "teste": "Fisher 2×2", "p": float(p), "odds_ratio": float(stat),
            "acuracia": {
                "cromatico": tab[0][0] / sum(tab[0]) if sum(tab[0]) else float("nan"),
                "estrutural": tab[1][0] / sum(tab[1]) if sum(tab[1]) else float("nan"),
            },
        }
    else:
        out["H2_cromatico_vs_estrutural"] = {
            "nota": "precisa de fotos anomalous marcadas como cromatico E estrutural"}

    # H3 — a acurácia cai com a distância?
    tab = [(d,
            sum(1 for r in emit if r["distancia"] == d and is_correct(r)),
            sum(1 for r in emit if r["distancia"] == d))
           for d in DISTANCIAS]
    if sum(t[2] for t in tab) and sum(1 for t in tab if t[2]) >= 2:
        res = cochran_armitage([t for t in tab if t[2]])
        res["tabela"] = {d: {"acertos": k, "n": n} for d, k, n in tab}
        res["teste"] = "Cochran-Armitage (tendência)"
        out["H3_distancia"] = res
    else:
        out["H3_distancia"] = {"nota": "precisa de fotos marcadas com distância em ≥2 níveis"}
    return out


def pct(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1%}"


def relatorio_md(versoes: list[dict], comparacoes: dict, hip: dict | None) -> str:
    L = ["# Análise estatística do teste de campo", "",
         "Gerado por `backend/analise_campo.py` — plano de análise da Parcial I.", ""]

    L += ["## Desempenho por versão", "",
          "| Versão | Emitidos | Acurácia bruta (IC95 Wilson) | Bal. acc (IC95 bootstrap) | Abstenção | ECE | AURC |",
          "|---|---|---|---|---|---|---|"]
    for v in versoes:
        ba = v["acuracia_balanceada"]
        L.append(
            f"| {v['versao']} | {v['n_emitidos']}/{v['n_folhas']} | "
            f"{pct(v['acuracia_bruta'])} [{pct(v['acuracia_bruta_ic95_wilson'][0])}, "
            f"{pct(v['acuracia_bruta_ic95_wilson'][1])}] | "
            f"{pct(ba['ponto'])} [{pct(ba['ic95'][0])}, {pct(ba['ic95'][1])}] | "
            f"{pct(v['taxa_abstencao'])} | {v['ece']:.3f} | {v['aurc']:.3f} |")
    L += ["", "IC de Wilson sobre a acurácia bruta **entre os diagnósticos emitidos** — "
          "abstenções excluídas, mesmo critério em todas as versões.", ""]

    for v in versoes:
        L += [f"### {v['versao']} — detalhe", ""]
        for truth, d in v["por_classe"].items():
            L.append(f"- **{truth}**: {d['acertos']}/{d['n']} "
                     f"[{pct(d['ic95_wilson'][0])}, {pct(d['ic95_wilson'][1])}]")
        if v["por_cultura"]:
            L += ["", "| Cultura | Acertos | Acurácia bruta | IC95 Wilson | Bal. acc |",
                  "|---|---|---|---|---|"]
            for c, d in v["por_cultura"].items():
                L.append(f"| {c} | {d['acertos']}/{d['n']} | {pct(d['acuracia_bruta'])} | "
                         f"[{pct(d['ic95_wilson'][0])}, {pct(d['ic95_wilson'][1])}] | "
                         f"{pct(d['acuracia_balanceada'])} |")
        if "guarda_dominio" in v:
            g = v["guarda_dominio"]
            L += ["", f"- Guarda de domínio (não-folha): {g['rejeitadas_corretamente']}/{g['n']} "
                      f"rejeitadas corretamente "
                      f"[{pct(g['ic95_wilson'][0])}, {pct(g['ic95_wilson'][1])}]"]
        L.append("")

    if comparacoes:
        L += ["## Comparação entre versões (McNemar exacto + Holm)", "",
              "| Par | Fotos pareadas | Só A acerta | Só B acerta | p | p (Holm) | Significativo |",
              "|---|---|---|---|---|---|---|"]
        for par, d in comparacoes.items():
            L.append(f"| {par} | {d['n_pareado']} | {d['so_A_acerta']} | {d['so_B_acerta']} | "
                     f"{d['p']:.4f} | {d['p_holm']:.4f} | "
                     f"{'sim' if d['significativo'] else 'não'} |")
        L += ["", "Só entram fotos com diagnóstico emitido nas duas versões.", ""]

    if hip:
        L += ["## Hipóteses", ""]
        h1 = hip["H1_cultura"]
        if "p" in h1:
            L += [f"**H1 — acurácia depende da cultura?** {h1['teste']}, "
                  f"p = {h1['p']:.4f} ({'diferença detectada' if h1['p'] < 0.05 else 'sem diferença detectada'}).",
                  f"Culturas: {', '.join(h1['culturas'])}.", ""]
        else:
            L += [f"**H1** — {h1['nota']}.", ""]
        h2 = hip["H2_cromatico_vs_estrutural"]
        if "p" in h2:
            L += [f"**H2 — dano cromático vs estrutural** (só anomalous): "
                  f"cromático {pct(h2['acuracia']['cromatico'])}, "
                  f"estrutural {pct(h2['acuracia']['estrutural'])}; "
                  f"{h2['teste']}, p = {h2['p']:.4f} "
                  f"({'diferença detectada' if h2['p'] < 0.05 else 'sem diferença detectada'}).", ""]
        else:
            L += [f"**H2** — {h2['nota']}.", ""]
        h3 = hip["H3_distancia"]
        if "p" in h3 and not math.isnan(h3.get("p", float("nan"))):
            L += [f"**H3 — acurácia cai com a distância?** {h3['teste']}, "
                  f"z = {h3['z']:.3f}, p = {h3['p']:.4f} "
                  f"({'tendência detectada' if h3['p'] < 0.05 else 'sem tendência detectada'}).", ""]
        else:
            L += [f"**H3** — {h3.get('nota', 'sem dados suficientes')}.", ""]

    L += ["---", "",
          "**Leitura com N pequeno:** ICs largos e p altos significam *não sabemos*, "
          "não *não há efeito*. Ver §10.7 do `docs/CORRECOES_METODOLOGICAS.md`.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Análise estatística do teste de campo")
    ap.add_argument("--csv", nargs="+", required=True,
                    help="CSV do test_campo.py; use caminho:nome para nomear a versão")
    ap.add_argument("--metadados", help="CSV de metadados por foto (H1-H3)")
    ap.add_argument("--out-dir", default=str(BASE_DIR / "results"))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    entradas = []
    for spec in args.csv:
        caminho, _, nome = spec.rpartition(":")
        # Em Windows "C:/..." tem ':' — se o que sobrou não parece um nome, é caminho.
        if not caminho or len(nome) <= 1:
            caminho, nome = spec, Path(spec).stem
        p = Path(caminho)
        if not p.exists():
            sys.exit(f"[ERRO] CSV não encontrado: {p}")
        entradas.append((nome, load_csv(p)))

    meta = load_metadados(Path(args.metadados)) if args.metadados else {}
    if args.metadados and not meta:
        print(f"[aviso] metadados vazios: {args.metadados}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    versoes = []
    for nome, rows in entradas:
        v = resumo_versao(nome, rows)
        versoes.append(v)
        curva = risco_cobertura(emitted(rows))
        if curva:
            cp = out_dir / f"risco_cobertura_{nome}.csv"
            with open(cp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(curva[0]))
                w.writeheader()
                w.writerows(curva)
            try:
                v["curva_risco_cobertura"] = str(cp.relative_to(BASE_DIR))
            except ValueError:      # --out-dir fora do backend/
                v["curva_risco_cobertura"] = str(cp)

    comparacoes = {}
    if len(entradas) >= 2:
        brutos = {}
        for i in range(len(entradas)):
            for j in range(i + 1, len(entradas)):
                (na, ra), (nb, rb) = entradas[i], entradas[j]
                brutos[f"{na} vs {nb}"] = mcnemar_exacto(ra, rb)
        ajustados = holm({k: v["p"] for k, v in brutos.items()})
        for k, v in brutos.items():
            comparacoes[k] = {**v, **ajustados[k]}

    hip = hipoteses(entradas[0][1], meta) if meta else None

    md = relatorio_md(versoes, comparacoes, hip)
    (out_dir / "analise_campo.md").write_text(md, encoding="utf-8")
    (out_dir / "analise_campo.json").write_text(
        json.dumps({"versoes": versoes, "comparacoes": comparacoes, "hipoteses": hip},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(md)
    print(f"\nRelatório: {out_dir / 'analise_campo.md'}")
    print(f"JSON     : {out_dir / 'analise_campo.json'}")


if __name__ == "__main__":
    main()
