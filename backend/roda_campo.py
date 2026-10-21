#!/usr/bin/env python
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Validação em campo, num comando: confere as fotos, roda cada modelo disponível às
cegas e analisa.

  1. confere_fotos_campo.py — procedência, cópias, metadados. Para se houver problema.
  2. test_campo.py com cada conjunto de pesos que existir:
         ciclo4    weights/            EfficientNet-B0 (o modelo servido)
         ciclo5    weights/ciclo5/     EfficientNet-B0 (pipeline corrigido), se já treinado
         vit       weights/vit/        ViT-B/16
         ciclo5_vit weights/ciclo5/    ViT-B/16 do ciclo 5, se já treinado
  3. analise_campo.py — acurácia balanceada, Wilson, McNemar entre as versões, ECE,
     risco-cobertura e as hipóteses H1 a H3.
  4. O que a auditoria de 13/10 mostrou que faltava olhar: ECE com e sem a temperatura
     ajustada na validação, erros acima do limiar e o resultado por horta.

Os modelos não são alterados: é só inferência. Nada daqui entra em treino.

Uso:
    python roda_campo.py                         # fotos em backend/campo_novo
    python roda_campo.py --dir ./campo_novo --mesmo-com-problemas
Saída:
    results/campo_novo/
"""

import argparse
import csv
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from compara_arquiteturas import ece
from compara_ciclos_teste_novo import wilson_ci

BACKEND = Path(__file__).resolve().parent
_venv_python = BACKEND / ".venv" / "Scripts" / "python.exe"
PYTHON = str(_venv_python) if _venv_python.exists() else sys.executable
CLASSES = ("healthy", "anomalous")
# (rótulo, modelo, pasta dos pesos)
VERSOES = [
    ("ciclo4", "efficientnet_b0", BACKEND / "weights"),
    ("ciclo5", "efficientnet_b0", BACKEND / "weights" / "ciclo5"),
    ("vit", "vit_b_16", BACKEND / "weights" / "vit"),
    ("ciclo5_vit", "vit_b_16", BACKEND / "weights" / "ciclo5"),
]


def run(args: list[str], env: dict | None = None) -> int:
    print(f"\n  $ {' '.join(args)}", flush=True)
    return subprocess.run([PYTHON, *args], cwd=str(BACKEND), env={**os.environ, **(env or {})}).returncode


def calibracao_e_hortas(csvs: dict[str, Path], raiz: Path) -> str:
    planilha = raiz / "metadados_campo.csv"
    horta = {}
    if planilha.exists():
        with open(planilha, encoding="utf-8-sig", newline="") as f:
            horta = {Path(r["ficheiro"]).name: (r.get("horta") or "").strip().lower() for r in csv.DictReader(f)
                     if (r.get("ficheiro") or "").strip()}
    L = ["# Campo — calibração e resultado por horta", "",
         "## Calibração: com e sem a temperatura ajustada na validação", "",
         "| Versão | T | τ | Diagnósticos | ECE com T | ECE sem T | Erros | Erros com confiança ≥ τ |",
         "|---|---|---|---|---|---|---|---|"]
    por_horta = {}
    for versao, caminho in csvs.items():
        modelo, pesos = next((m, p) for v, m, p in VERSOES if v == versao)
        calib = json.loads((pesos / f"{modelo}_binary_calibration.json").read_text(encoding="utf-8"))
        T, tau = calib["temperature"], calib["threshold"]
        with open(caminho, encoding="utf-8") as f:
            linhas = [r for r in csv.DictReader(f) if r["verdade"] in CLASSES]
        emitidos = [r for r in linhas if r["resposta"] in CLASSES]
        certo = np.array([r["resposta"] == r["verdade"] for r in emitidos])
        conf = np.clip(np.array([float(r["confianca"]) for r in emitidos]), 1e-6, 1 - 1e-6)
        # Em classificação binária, a confiança sem temperatura é sigmoid(T · logit(c)).
        sem_T = 1 / (1 + np.exp(-T * np.log(conf / (1 - conf))))
        L.append(f"| {versao} | {T:.3f} | {tau} | {len(emitidos)} de {len(linhas)} | {ece(conf, certo):.3f} | "
                 f"{ece(sem_T, certo):.3f} | {int((~certo).sum())} | {int((~certo & (conf >= tau)).sum())} |")
        tab = defaultdict(lambda: [0, 0])
        for r in emitidos:
            h = horta.get(Path(r["ficheiro"].replace("\\", "/")).name) or "(sem horta)"
            tab[h][0] += r["resposta"] == r["verdade"]
            tab[h][1] += 1
        por_horta[versao] = tab
    L += ["", "## Acertos por horta (diagnósticos emitidos)", ""]
    hortas = sorted({h for t in por_horta.values() for h in t})
    L += ["| Horta | " + " | ".join(csvs) + " |", "|---|" + "---|" * len(csvs)]
    for h in hortas:
        cel = []
        for versao in csvs:
            k, n = por_horta[versao].get(h, (0, 0))
            lo, hi = wilson_ci(k, n)
            cel.append(f"{k} de {n} [{lo:.0%}; {hi:.0%}]" if n else "—")
        L.append(f"| {h} | " + " | ".join(cel) + " |")
    L += ["", "A separação entre conjunto de avaliação e conjunto de adaptação é por horta. O conjunto",
          "de avaliação nunca entra em treino nem em recalibração.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Validação em campo: conferência, inferência e análise")
    ap.add_argument("--dir", default="./campo_novo")
    ap.add_argument("--out-dir", default="./results/campo_novo")
    ap.add_argument("--mesmo-com-problemas", action="store_true",
                    help="segue mesmo que a conferência das fotos aponte problemas")
    args = ap.parse_args()
    raiz, out = (BACKEND / args.dir).resolve(), (BACKEND / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    rc = run(["confere_fotos_campo.py", "--dir", str(raiz), "--out-dir", str(out)])
    if rc != 0 and not args.mesmo_com_problemas:
        sys.exit("\n[campo] A conferência apontou problemas (acima). Corrija-os ou rode com --mesmo-com-problemas.")

    csvs = {}
    for versao, modelo, pesos in VERSOES:
        if not (pesos / f"{modelo}_binary.pth").exists():
            print(f"\n  [{versao}] sem pesos em {pesos} — pulando", flush=True)
            continue
        destino = out / f"campo_{versao}.csv"
        run(["test_campo.py", "--dir", str(raiz), "--out", str(destino)],
            env={"BINARY_MODEL": modelo, "WEIGHTS_DIR": str(pesos)})
        if destino.exists():
            csvs[versao] = destino
    if not csvs:
        sys.exit("[campo] Nenhum modelo rodou.")

    planilha = raiz / "metadados_campo.csv"
    comando = ["analise_campo.py", "--csv", *[f"{os.path.relpath(c, BACKEND)}:{v}" for v, c in csvs.items()],
               "--out-dir", str(out)]
    if planilha.exists():
        comando += ["--metadados", str(planilha)]
    run(comando)

    texto = calibracao_e_hortas(csvs, raiz)
    (out / "calibracao_e_hortas.md").write_text(texto, encoding="utf-8")
    print("\n" + texto)
    print(f"Resultados em {out}")


if __name__ == "__main__":
    main()
