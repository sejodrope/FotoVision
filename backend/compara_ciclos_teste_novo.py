#!/usr/bin/env python
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Ciclo 3 vs ciclo 4 no teste NOVO (espinafre + lettuce downy), com teste pareado.

O metrics_comparison.csv dá o ponto estimado de cada ciclo, mas não diz se a
diferença é real ou ruído — e foi exactamente essa confusão que o §10.7 teve de
corrigir. Aqui a comparação é pareada, foto a foto: cada modelo vê as MESMAS
imagens, e o McNemar exacto pergunta só pelas fotos em que os dois discordam.

Saídas:
    results/ciclo4/comparacao_pareada_teste_novo.csv   — uma linha por imagem
    results/ciclo4/comparacao_pareada_teste_novo.json  — McNemar + Holm + ICs

Uso:
    python compara_ciclos_teste_novo.py
    python compara_ciclos_teste_novo.py --split logs/test_split_binary_novo.json
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest, norm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from dataset import BINARY_CLASSES, VAL_TRANSFORMS, dataset_from_samples
from evaluate import _load_temperature
from train import build_model

BASE_DIR = Path(__file__).parent
MODELS = ("efficientnet_b0", "resnet50", "mobilenet_v2")
SEED = 42
N_BOOT = 10_000


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    d = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / d
    meio = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return [max(0.0, centro - meio), min(1.0, centro + meio)]


def bal_acc(correct: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean([correct[labels == c].mean()
                          for c in np.unique(labels) if (labels == c).any()]))


def boot_bal_acc(correct: np.ndarray, labels: np.ndarray) -> list[float]:
    """IC percentil por bootstrap estratificado por classe verdadeira."""
    rng = np.random.default_rng(SEED)
    idx_por_classe = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    amostras = np.empty(N_BOOT)
    for i in range(N_BOOT):
        amostras[i] = float(np.mean([
            correct[rng.choice(idx, size=len(idx), replace=True)].mean()
            for idx in idx_por_classe]))
    return [float(np.percentile(amostras, 2.5)), float(np.percentile(amostras, 97.5))]


def holm(ps: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    itens = sorted(ps.items(), key=lambda kv: kv[1])
    m = len(itens)
    out, maior = {}, 0.0
    for i, (nome, p) in enumerate(itens):
        ajustado = min(1.0, max(maior, (m - i) * p))
        maior = ajustado
        out[nome] = {"p": p, "p_holm": ajustado, "significativo": ajustado < alpha}
    return out


@torch.no_grad()
def prever(model_name: str, weights_dir: Path, loader: DataLoader,
           device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Devolve (previsões, confiança calibrada) — mesma temperatura da produção."""
    model = build_model(model_name, num_classes=2)
    state = torch.load(weights_dir / f"{model_name}_binary.pth", map_location=device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.to(device).eval()
    T = _load_temperature(weights_dir, model_name, "_binary") or 1.0
    preds, confs = [], []
    for x, _ in loader:
        logits = model(x.to(device)) / T
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        preds.append(pred.cpu().numpy())
        confs.append(conf.cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(preds), np.concatenate(confs)


def main():
    ap = argparse.ArgumentParser(description="Ciclo 3 vs ciclo 4 no teste novo")
    ap.add_argument("--split", default=str(BASE_DIR / "logs" / "test_split_binary_novo.json"))
    ap.add_argument("--out-dir", default=str(BASE_DIR / "results" / "ciclo4"))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.split, encoding="utf-8") as f:
        amostras = [(Path(p), int(l)) for p, l in json.load(f)]
    ds = dataset_from_samples(BINARY_CLASSES, amostras, VAL_TRANSFORMS)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    labels = np.array([l for _, l in amostras])
    print(f"Teste novo: {len(amostras)} imagens "
          f"({int((labels == 0).sum())} healthy, {int((labels == 1).sum())} anomalous)")

    resultados, linhas = {}, {}
    for m in MODELS:
        por_ciclo = {}
        for ciclo, wdir in (("ciclo3", BASE_DIR / "weights" / "ciclo3"),
                            ("ciclo4", BASE_DIR / "weights")):
            preds, confs = prever(m, wdir, loader, device)
            correct = (preds == labels).astype(int)
            por_ciclo[ciclo] = {"preds": preds, "confs": confs, "correct": correct}
            print(f"  {m:<16} {ciclo}: bal_acc={bal_acc(correct, labels):.4f}")

        c3, c4 = por_ciclo["ciclo3"]["correct"], por_ciclo["ciclo4"]["correct"]
        b01 = int(((c3 == 1) & (c4 == 0)).sum())   # só o ciclo 3 acerta
        b10 = int(((c3 == 0) & (c4 == 1)).sum())   # só o ciclo 4 acerta
        disc = b01 + b10
        p = binomtest(b01, disc, 0.5).pvalue if disc else 1.0

        resultados[m] = {
            "n": len(labels),
            "ciclo3": {"bal_acc": bal_acc(c3, labels),
                       "bal_acc_ic95": boot_bal_acc(c3, labels),
                       "acertos": int(c3.sum()),
                       "acuracia_bruta_ic95_wilson": wilson_ci(int(c3.sum()), len(c3))},
            "ciclo4": {"bal_acc": bal_acc(c4, labels),
                       "bal_acc_ic95": boot_bal_acc(c4, labels),
                       "acertos": int(c4.sum()),
                       "acuracia_bruta_ic95_wilson": wilson_ci(int(c4.sum()), len(c4))},
            "mcnemar": {"so_ciclo3_acerta": b01, "so_ciclo4_acerta": b10,
                        "discordantes": disc, "p": float(p)},
        }
        linhas[m] = por_ciclo

    ajustados = holm({m: r["mcnemar"]["p"] for m, r in resultados.items()})
    for m, a in ajustados.items():
        resultados[m]["mcnemar"].update({"p_holm": a["p_holm"], "significativo": a["significativo"]})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "comparacao_pareada_teste_novo.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ficheiro", "verdade"] +
                   [f"{m}_{c}_{campo}" for m in MODELS for c in ("ciclo3", "ciclo4")
                    for campo in ("pred", "conf", "acertou")])
        for i, (p_img, lbl) in enumerate(amostras):
            linha = [str(p_img), BINARY_CLASSES[lbl]]
            for m in MODELS:
                for c in ("ciclo3", "ciclo4"):
                    d = linhas[m][c]
                    linha += [BINARY_CLASSES[d["preds"][i]], f"{d['confs'][i]:.4f}",
                              int(d["correct"][i])]
            w.writerow(linha)

    json_path = out_dir / "comparacao_pareada_teste_novo.json"
    json_path.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("  CICLO 3 vs CICLO 4 NO TESTE NOVO — McNemar exacto pareado + Holm")
    print("=" * 72)
    print(f"  {'Modelo':<17}{'ciclo3':<9}{'ciclo4':<9}{'só c3':<7}{'só c4':<7}"
          f"{'p':<10}{'p Holm':<10}{'signif.'}")
    for m, r in resultados.items():
        mc = r["mcnemar"]
        print(f"  {m:<17}{r['ciclo3']['bal_acc']:<9.4f}{r['ciclo4']['bal_acc']:<9.4f}"
              f"{mc['so_ciclo3_acerta']:<7}{mc['so_ciclo4_acerta']:<7}"
              f"{mc['p']:<10.2e}{mc['p_holm']:<10.2e}"
              f"{'sim' if mc['significativo'] else 'não'}")
    print("=" * 72)
    print(f"  Por imagem : {csv_path}")
    print(f"  Números    : {json_path}")


if __name__ == "__main__":
    main()
