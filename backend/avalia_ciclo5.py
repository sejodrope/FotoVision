#!/usr/bin/env python
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Avaliação do ciclo 5 — por subconjunto, e não só pela média.

A auditoria de 13/10/2026 mostrou que a média do teste escondia onde o modelo errava
(docs/CORRECOES_METODOLOGICAS.md §14.1). Aqui cada modelo é avaliado em:

    teste inteiro, coletânea (lavoura × demais), PlantVillage, alface (uma fotografia
    por foto original), espinafre, pastas antes não reconhecidas, recortes de míldio
    (só descritivo) e a FONTE RETIDA de espinafre, que não entrou no treino.

O subconjunto "comparavel_com_ciclo4" são as imagens de teste da coletânea e do
PlantVillage, que mantiveram a partição: nelas o ciclo 5 é comparado ao ciclo 4 foto a
foto (McNemar exato), com as previsões já salvas em results/arquiteturas/predicoes/.

Duas etapas, como em compara_arquiteturas.py:
  1. Previsão — um CSV por modelo em results/ciclo5/predicoes/ (teste + fonte retida).
  2. Avaliação — results/ciclo5/avaliacao_ciclo5.json / .md

Uso:
    python avalia_ciclo5.py                      # todos os modelos com pesos em weights/ciclo5
    python avalia_ciclo5.py --so-avaliar
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))

from compara_arquiteturas import MODELS, boot_bal_acc, ece
from compara_ciclos_teste_novo import bal_acc

BASE = Path(__file__).resolve().parent
DATA = BASE / "data_c5"
PESOS = BASE / "weights" / "ciclo5"
OUT = BASE / "results" / "ciclo5"
CLASSES = ("healthy", "anomalous")
CAMPOS = ["arquivo", "split", "verdade", "pred", "conf", "p_anomalous"]

SUBCONJUNTOS = {
    "teste_inteiro": lambda r: r["split"] == "test",
    "coletanea_lavoura": lambda r: r["tipo"] == "coletanea_lavoura",
    "coletanea_demais": lambda r: r["tipo"] == "coletanea_demais",
    "plantvillage": lambda r: r["tipo"] == "plantvillage",
    "alface_uma_por_foto": lambda r: r["split"] == "test" and r["tipo"].startswith("alface"),
    "espinafre": lambda r: r["tipo"] == "espinafre",
    "antes_nao_reconhecidas": lambda r: r["tipo"] == "nao_reconhecida",
    "mildio_recortes_descritivo": lambda r: r["tipo"] == "mildio_recorte",
    "espinafre_fonte_retida": lambda r: r["split"] == "retida",
    "comparavel_com_ciclo4": lambda r: r["tipo"] in ("coletanea_lavoura", "coletanea_demais", "plantvillage"),
}


def prever(modelo: str, batch_size: int, workers: int) -> None:
    import torch
    from torch.utils.data import ConcatDataset, DataLoader
    from dataset import VAL_TRANSFORMS, BinaryFolderDataset
    from evaluate import _load_temperature, build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    partes = [(s, BinaryFolderDataset(DATA / s, transform=VAL_TRANSFORMS)) for s in ("test", "retida")]
    loader = DataLoader(ConcatDataset([d for _, d in partes]), batch_size=batch_size, shuffle=False,
                        num_workers=workers)
    model = build_model(modelo, num_classes=2)
    model.load_state_dict(torch.load(PESOS / f"{modelo}_binary.pth", map_location=device, weights_only=True))
    model.to(device).eval()
    T = _load_temperature(PESOS, modelo, "_binary")
    print(f"  {modelo}: {sum(len(d) for _, d in partes)} imagens, T={T:.4f}", flush=True)

    probs = []
    with torch.no_grad():
        for x, _ in loader:
            probs.append(torch.softmax(model(x.to(device)) / T, dim=1).cpu().numpy())
    probs = np.concatenate(probs)

    out = OUT / "predicoes" / f"{modelo}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    amostras = [(s, p, lbl) for s, d in partes for p, lbl in d.samples]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CAMPOS)
        for (split, path, label), p in zip(amostras, probs):
            w.writerow([Path(path).name, split, CLASSES[label], CLASSES[int(p.argmax())],
                        f"{p.max():.6f}", f"{p[1]:.6f}"])
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def carregar(modelo: str, manifesto: dict) -> list[dict]:
    with open(OUT / "predicoes" / f"{modelo}.csv", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    for r in linhas:
        r["tipo"] = manifesto[r["arquivo"]]
        r["conf"] = float(r["conf"])
        r["certo"] = r["pred"] == r["verdade"]
    return linhas


def medir(linhas: list[dict], tau: float) -> dict:
    verdade = np.array([r["verdade"] for r in linhas])
    certo = np.array([r["certo"] for r in linhas])
    conf = np.array([r["conf"] for r in linhas])
    duas_classes = len(set(verdade)) == 2
    return {
        "n": len(linhas),
        "healthy": int((verdade == "healthy").sum()),
        "anomalous": int((verdade == "anomalous").sum()),
        # Com uma classe só, a acurácia balanceada é a revocação dessa classe.
        "bal_acc": round(bal_acc(certo, verdade), 4),
        "ic95": [round(v, 4) for v in boot_bal_acc(certo, verdade)] if duas_classes and len(linhas) >= 30 else None,
        "fn": int((~certo & (verdade == "anomalous")).sum()),
        "fp": int((~certo & (verdade == "healthy")).sum()),
        "ece": round(ece(conf, certo), 4),
        "erros_interceptados": int((~certo & (conf < tau)).sum()),
        "abstencao_pct": round(100 * float((conf < tau).mean()), 1),
    }


def contra_ciclo4(modelo: str, linhas: list[dict]) -> dict | None:
    """McNemar exato ciclo 4 × ciclo 5 nas imagens de teste que não mudaram de partição."""
    antigo = BASE / "results" / "arquiteturas" / "predicoes" / f"{modelo}.csv"
    if not antigo.exists():
        return None
    with open(antigo, encoding="utf-8") as f:
        c4 = {r["chave"].split("/", 1)[1]: r["pred"] == r["verdade"] for r in csv.DictReader(f)}
    comuns = [r for r in linhas if SUBCONJUNTOS["comparavel_com_ciclo4"](r) and r["arquivo"] in c4]
    so_c4 = sum(1 for r in comuns if c4[r["arquivo"]] and not r["certo"])
    so_c5 = sum(1 for r in comuns if not c4[r["arquivo"]] and r["certo"])
    return {"n": len(comuns), "so_ciclo4_acerta": so_c4, "so_ciclo5_acerta": so_c5,
            "p_mcnemar_exato": float(binomtest(so_c4, so_c4 + so_c5, 0.5).pvalue) if so_c4 + so_c5 else 1.0}


def avaliar(modelos: list[str]) -> dict:
    with open(DATA / "manifesto_ciclo5.csv", encoding="utf-8") as f:
        manifesto = {r["arquivo"]: r["tipo"] for r in csv.DictReader(f)}
    resultado = {}
    for modelo in modelos:
        calib = json.loads((PESOS / f"{modelo}_binary_calibration.json").read_text(encoding="utf-8"))
        linhas = carregar(modelo, manifesto)
        bloco = {"temperatura": calib["temperature"], "tau": calib["threshold"], "subconjuntos": {}}
        for nome, filtro in SUBCONJUNTOS.items():
            sub = [r for r in linhas if filtro(r)]
            if sub:
                bloco["subconjuntos"][nome] = medir(sub, calib["threshold"])
        bloco["ciclo4_x_ciclo5"] = contra_ciclo4(modelo, linhas)
        resultado[modelo] = bloco
    return resultado


def markdown(resultado: dict) -> str:
    L = ["# Ciclo 5 — avaliação por subconjunto", "",
         "Gerado por `backend/avalia_ciclo5.py`. Acurácia balanceada, IC de 95% por bootstrap", "estratificado.", ""]
    for modelo, b in resultado.items():
        L += [f"## {modelo} (T = {b['temperatura']:.3f}, τ = {b['tau']})", "",
              "| Subconjunto | N | Acur. bal. | IC 95% | FN | FP | ECE | Interceptados | Abstenção |",
              "|---|---|---|---|---|---|---|---|---|"]
        for nome, m in b["subconjuntos"].items():
            ic = f"[{m['ic95'][0]:.4f}; {m['ic95'][1]:.4f}]" if m["ic95"] else "—"
            L.append(f"| {nome} | {m['n']} | {m['bal_acc']:.4f} | {ic} | {m['fn']} | {m['fp']} | {m['ece']:.4f} | "
                     f"{m['erros_interceptados']} de {m['fn'] + m['fp']} | {m['abstencao_pct']}% |")
        c = b["ciclo4_x_ciclo5"]
        if c:
            L += ["", f"Ciclo 4 × ciclo 5 nas {c['n']} imagens de teste que não mudaram de partição: só o ciclo 4 acerta "
                      f"{c['so_ciclo4_acerta']}, só o ciclo 5 acerta {c['so_ciclo5_acerta']}; McNemar exato, "
                      f"p = {c['p_mcnemar_exato']:.3g}."]
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Avaliação do ciclo 5 por subconjunto")
    ap.add_argument("--modelos", nargs="+", help="padrão: os que têm pesos em weights/ciclo5")
    ap.add_argument("--so-avaliar", action="store_true", help="não refaz as previsões")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    modelos = args.modelos or [m for m in MODELS if (PESOS / f"{m}_binary.pth").exists()]
    if not modelos:
        sys.exit(f"[ciclo5] Nenhum peso em {PESOS}. Rode run_ciclo5.py primeiro.")
    if not args.so_avaliar:
        for m in modelos:
            prever(m, args.batch_size, args.workers)
    resultado = avaliar(modelos)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "avaliacao_ciclo5.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "avaliacao_ciclo5.md").write_text(markdown(resultado), encoding="utf-8")
    print(markdown(resultado))


if __name__ == "__main__":
    main()
