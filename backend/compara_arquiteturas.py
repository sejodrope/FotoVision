#!/usr/bin/env python
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Comparação pareada entre arquitecturas no conjunto de teste (pergunta Q1 do TCC-II).

O metrics_comparison.csv dá um ponto estimado por modelo; aqui cada modelo responde
às MESMAS imagens e a diferença é testada foto a foto (McNemar exacto + Holm), com
IC de 95% por bootstrap estratificado. Serve para as três CNNs e para o ViT.

Duas etapas, que podem correr em máquinas diferentes:

  1. Previsão — um CSV por modelo, uma linha por imagem do teste, em --pred-dir.
     Modelos que já têm CSV são saltados (use --force para refazer).
  2. Comparação — lê todos os CSVs de --pred-dir e escreve
         results/arquiteturas/comparacao_arquiteturas.json / .md

O teste é dividido em três recortes, pelo manifesto do merge do ciclo 4:
    original   — as 23.967 imagens das seis fontes do treino (intra-fonte)
    espinafre  — 484 imagens do lote Mendeley
    mildio     — 173 recortes de 5 fotos (não independentes; só descritivo)

Uso:
    python compara_arquiteturas.py                    # prevê o que falta e compara
    python compara_arquiteturas.py --compare-only
    python compara_arquiteturas.py --predict-only --weights-dir ./weights/vit
"""

import argparse
import csv
import itertools
import json
import re
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from compara_ciclos_teste_novo import bal_acc, holm, wilson_ci

BASE_DIR = Path(__file__).parent
MODELS = ("efficientnet_b0", "mobilenet_v2", "resnet50", "vit_b_16")
CLASSES = ("healthy", "anomalous")
RECORTES = ("original", "espinafre", "mildio")
SEED = 42
N_BOOT = 10_000
CAMPOS = ["chave", "verdade", "recorte", "pred", "conf", "p_anomalous"]


def chave(path: str) -> str:
    """'healthy/<ficheiro>' — igual no Windows e no Linux, ao contrário do caminho."""
    return "/".join(re.split(r"[\\/]", str(path))[-2:])


def recorte_de(nome: str, novos: set[str]) -> str:
    if nome not in novos:
        return "original"
    return "espinafre" if "spinach" in nome else "mildio"


def ficheiros_novos(data_dir: Path) -> set[str]:
    with open(data_dir / "ciclo4_merge_manifest.csv", encoding="utf-8") as f:
        return {Path(r["destino"]).name for r in csv.DictReader(f) if r["split"] == "test"}


# ── Etapa 1: previsão ─────────────────────────────────────────────────────────

def prever(model_name: str, weights_dir: Path, data_dir: Path, out: Path,
           batch_size: int, workers: int) -> None:
    import torch
    from torch.utils.data import DataLoader
    from dataset import VAL_TRANSFORMS, BinaryFolderDataset
    from evaluate import _load_temperature, build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = BinaryFolderDataset(data_dir / "test", transform=VAL_TRANSFORMS)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    novos = ficheiros_novos(data_dir)

    model = build_model(model_name, num_classes=2)
    model.load_state_dict(torch.load(weights_dir / f"{model_name}_binary.pth",
                                     map_location=device, weights_only=True))
    model.to(device).eval()
    T = _load_temperature(weights_dir, model_name, "_binary")
    print(f"  {model_name}: {len(ds)} imagens, T={T:.4f}, pesos em {weights_dir}", flush=True)

    probs = []
    with torch.no_grad():
        for x, _ in loader:
            probs.append(torch.softmax(model(x.to(device)) / T, dim=1).cpu().numpy())
    probs = np.concatenate(probs)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CAMPOS)
        for (path, label), p in zip(ds.samples, probs):
            w.writerow([chave(path), CLASSES[label], recorte_de(Path(path).name, novos),
                        CLASSES[int(p.argmax())], f"{p.max():.6f}", f"{p[1]:.6f}"])

    calib = weights_dir / f"{model_name}_binary_calibration.json"
    meta = json.loads(calib.read_text(encoding="utf-8")) if calib.exists() else {}
    out.with_suffix(".json").write_text(json.dumps(
        {"model": model_name, "temperature": T, "threshold": meta.get("threshold")},
        indent=2), encoding="utf-8")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ── Etapa 2: comparação ───────────────────────────────────────────────────────

def boot_bal_acc(correct: np.ndarray, labels: np.ndarray) -> list[float]:
    """
    IC percentil por bootstrap estratificado por classe verdadeira.

    Reamostrar n imagens de uma classe com reposição e contar acertos é uma
    Binomial(n, taxa de acerto observada) — sorteia-se a contagem directamente,
    o que dá o mesmo resultado sem materializar 10.000 × 24.000 índices.
    """
    rng = np.random.default_rng(SEED)
    recalls = []
    for c in np.unique(labels):
        ok = correct[labels == c]
        recalls.append(rng.binomial(len(ok), ok.mean(), size=N_BOOT) / len(ok))
    amostras = np.mean(recalls, axis=0)
    return [float(np.percentile(amostras, 2.5)), float(np.percentile(amostras, 97.5))]


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def resumo_modelo(d: dict, tau: float | None) -> dict:
    from scipy.stats import mannwhitneyu

    labels, correct, conf = d["labels"], d["correct"], d["conf"]
    pred = np.where(correct == 1, labels, 1 - labels)
    n = len(labels)
    out = {
        "n": n,
        "acertos": int(correct.sum()),
        "acuracia": float(correct.mean()),
        "acuracia_ic95_wilson": wilson_ci(int(correct.sum()), n),
        "bal_acc": bal_acc(correct, labels),
        "falsos_negativos": int(((labels == 1) & (pred == 0)).sum()),
        "falsos_positivos": int(((labels == 0) & (pred == 1)).sum()),
        "ece": ece(conf, correct),
        "confianca_media_acertos": float(conf[correct == 1].mean()) if correct.any() else None,
        "confianca_media_erros": float(conf[correct == 0].mean()) if (correct == 0).any() else None,
    }
    if len(np.unique(labels)) > 1:
        out["bal_acc_ic95"] = boot_bal_acc(correct, labels)
    # A confiança separa acertos de erros? (Mann-Whitney; a AUROC é U/(n1·n2))
    n_ok, n_err = int(correct.sum()), int((correct == 0).sum())
    if n_ok and n_err:
        u = mannwhitneyu(conf[correct == 1], conf[correct == 0], alternative="greater")
        out["mann_whitney"] = {"U": float(u.statistic), "p": float(u.pvalue),
                               "auroc_confianca": float(u.statistic / (n_ok * n_err))}
    if tau is not None:
        emit = conf >= tau
        out["abstencao"] = {
            "tau": tau,
            "cobertura": float(emit.mean()),
            "acuracia_seletiva": float(correct[emit].mean()) if emit.any() else None,
            "erros": n_err,
            "erros_interceptados": int(((correct == 0) & ~emit).sum()),
            "acertos_descartados": int(((correct == 1) & ~emit).sum()),
        }
    return out


def carregar(pred_dir: Path) -> tuple[dict, dict]:
    """{modelo: {chave: linha}} e {modelo: limiar} dos CSVs presentes em pred_dir."""
    linhas, taus = {}, {}
    for m in MODELS:
        p = pred_dir / f"{m}.csv"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            linhas[m] = {r["chave"]: r for r in csv.DictReader(f)}
        meta = p.with_suffix(".json")
        taus[m] = json.loads(meta.read_text(encoding="utf-8")).get("threshold") if meta.exists() else None
    return linhas, taus


def comparar(pred_dir: Path, out_dir: Path) -> dict:
    from scipy.stats import binomtest

    linhas, taus = carregar(pred_dir)
    if not linhas:
        sys.exit(f"[ERRO] Nenhum CSV de previsões em {pred_dir}.")
    modelos = list(linhas)
    chaves = sorted(set.intersection(*(set(v) for v in linhas.values())))
    for m in modelos:
        if len(linhas[m]) != len(chaves):
            sys.exit(f"[ERRO] {m} tem {len(linhas[m])} imagens e a intersecção tem {len(chaves)} — "
                     "os modelos não viram o mesmo teste. Refaça as previsões com --force.")
    ref = linhas[modelos[0]]

    resultado = {"modelos": modelos, "n_bootstrap": N_BOOT, "seed": SEED, "recortes": {}}
    for recorte in (*RECORTES, "completo"):
        ks = [k for k in chaves if recorte == "completo" or ref[k]["recorte"] == recorte]
        if not ks:
            continue
        labels = np.array([CLASSES.index(ref[k]["verdade"]) for k in ks])
        dados = {}
        for m in modelos:
            correct = np.array([int(linhas[m][k]["pred"] == linhas[m][k]["verdade"]) for k in ks])
            conf = np.array([float(linhas[m][k]["conf"]) for k in ks])
            dados[m] = {"labels": labels, "correct": correct, "conf": conf}

        bloco = {"n": len(ks),
                 "n_por_classe": {c: int((labels == i).sum()) for i, c in enumerate(CLASSES)},
                 "modelos": {m: resumo_modelo(dados[m], taus[m]) for m in modelos},
                 "mcnemar": {}}
        ps = {}
        for a, b in itertools.combinations(modelos, 2):
            ca, cb = dados[a]["correct"], dados[b]["correct"]
            so_a = int(((ca == 1) & (cb == 0)).sum())
            so_b = int(((ca == 0) & (cb == 1)).sum())
            p = binomtest(so_a, so_a + so_b, 0.5).pvalue if so_a + so_b else 1.0
            nome = f"{a} vs {b}"
            bloco["mcnemar"][nome] = {"so_primeiro_acerta": so_a, "so_segundo_acerta": so_b,
                                      "p": float(p)}
            ps[nome] = float(p)
        for nome, a in holm(ps).items():
            bloco["mcnemar"][nome].update(p_holm=a["p_holm"], significativo=a["significativo"])
        if recorte == "mildio":
            bloco["aviso"] = ("173 recortes de 5 fotografias: não são observações independentes. "
                              "Só descritivo; os valores de p superestimam a evidência.")
        resultado["recortes"][recorte] = bloco

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparacao_arquiteturas.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# Comparação pareada entre arquitecturas", ""]
    for recorte, b in resultado["recortes"].items():
        md += [f"## Recorte: {recorte} (N = {b['n']})", ""]
        if "aviso" in b:
            md += [f"> {b['aviso']}", ""]
        md += ["| Modelo | Acur. bal. | IC 95% | FN | FP | ECE | Erros | Interceptados (conf < τ) | Cobertura |",
               "|---|---|---|---|---|---|---|---|---|"]
        for m, r in b["modelos"].items():
            ic = r.get("bal_acc_ic95")
            ab = r.get("abstencao")
            md.append(
                f"| {m} | {r['bal_acc']:.4f} | "
                + (f"[{ic[0]:.4f}; {ic[1]:.4f}]" if ic else "—")
                + f" | {r['falsos_negativos']} | {r['falsos_positivos']} | {r['ece']:.4f} | "
                + (f"{ab['erros']} | {ab['erros_interceptados']} | {ab['cobertura']:.4f} |"
                   if ab else f"{r['n'] - r['acertos']} | — | — |"))
        md += ["", "| Par | Só o 1.º acerta | Só o 2.º acerta | p (McNemar exacto) | p (Holm) | Signif. |",
               "|---|---|---|---|---|---|"]
        for nome, r in b["mcnemar"].items():
            md.append(f"| {nome} | {r['so_primeiro_acerta']} | {r['so_segundo_acerta']} | "
                      f"{r['p']:.2e} | {r['p_holm']:.2e} | {'sim' if r['significativo'] else 'não'} |")
        md.append("")
    (out_dir / "comparacao_arquiteturas.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"Gravado em {out_dir}")
    return resultado


def main():
    ap = argparse.ArgumentParser(description="Comparação pareada entre arquitecturas")
    ap.add_argument("--data", default=str(BASE_DIR / "data"))
    ap.add_argument("--weights-dir", action="append",
                    help="Pasta de pesos; pode repetir. Padrão: ./weights e ./weights/vit")
    ap.add_argument("--pred-dir", default=str(BASE_DIR / "results" / "arquiteturas" / "predicoes"))
    ap.add_argument("--out-dir", default=str(BASE_DIR / "results" / "arquiteturas"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="Refaz previsões já existentes")
    ap.add_argument("--predict-only", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    if not args.compare_only:
        dirs = [Path(d) for d in (args.weights_dir or
                                  [BASE_DIR / "weights", BASE_DIR / "weights" / "vit"])]
        for m in MODELS:
            wdir = next((d for d in dirs if (d / f"{m}_binary.pth").exists()), None)
            out = pred_dir / f"{m}.csv"
            if wdir is None:
                continue
            if out.exists() and not args.force:
                print(f"  {m}: previsões já existem em {out}, a saltar", flush=True)
                continue
            prever(m, wdir, Path(args.data), out, args.batch_size, args.workers)
    if not args.predict_only:
        comparar(pred_dir, Path(args.out_dir))


if __name__ == "__main__":
    main()
