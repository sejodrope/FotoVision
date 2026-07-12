#!/usr/bin/env python
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Pipeline completo de correcção metodológica (docs/CORRECOES_METODOLOGICAS.md §7).

Sequência:
  1. Backup dos artefactos antigos (pesos, métricas, test split) — são a base da errata
  2. Auditoria do split ANTIGO  → results/leakage_report_old_split.json (a prova citável)
  3. Calibração do modelo antigo no val antigo (mantém a demo honesta durante o retreino)
  4. Limpeza do staging e dos splits antigos — ambos foram construídos com as regras
     erradas, e a reorganização só copia `if not dest.exists()`: sem limpar, os
     ficheiros mal rotulados/vazados sobreviveriam à correcção
  5. Reorganização + split agrupado (download_datasets.py --skip-download)
  6. Auditoria do split NOVO — aborta se o vazamento não for eliminado
  7. Treino de efficientnet_b0, resnet50 e mobilenet_v2 (30 epochs, F1 macro)
  8. Calibração dos três (temperature scaling + limiar de abstenção)
  9. Avaliação final no teste limpo → results/metrics_comparison.csv

Uso:
    python run_correction_pipeline.py
"""

import json
import shutil
import subprocess
import time
from pathlib import Path

BACKEND = Path(__file__).parent
DATA    = BACKEND / "data"
WEIGHTS = BACKEND / "weights"
RESULTS = BACKEND / "results"
LOGS    = BACKEND / "logs"

_venv_python = BACKEND / ".venv" / "Scripts" / "python.exe"
PYTHON = str(_venv_python) if _venv_python.exists() else sys.executable

MODELS = ("efficientnet_b0", "resnet50", "mobilenet_v2")
EPOCHS = 30
BATCH  = 64


def banner(msg: str):
    print("\n" + "=" * 64, flush=True)
    print(f"  {msg}", flush=True)
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 64, flush=True)


def run(args: list[str], fatal: bool = True) -> int:
    print(f"  $ {' '.join(args)}", flush=True)
    rc = subprocess.run([PYTHON, *args], cwd=str(BACKEND)).returncode
    if rc != 0 and fatal:
        sys.exit(f"[pipeline] ABORTADO — '{args[0]}' terminou com código {rc}.")
    return rc


def main():
    t0 = time.time()

    # ── 1. Backup ──────────────────────────────────────────────────────────
    banner("1/9  Backup dos artefactos do pipeline antigo")
    (WEIGHTS / "pre_correction").mkdir(exist_ok=True)
    (RESULTS / "pre_correction").mkdir(parents=True, exist_ok=True)
    (LOGS / "pre_correction").mkdir(parents=True, exist_ok=True)

    for f in WEIGHTS.glob("*_binary.pth"):
        dest = WEIGHTS / "pre_correction" / f.name
        if not dest.exists():
            shutil.copy2(f, dest)
            print(f"  pesos    → {dest}")
    for name in ("metrics_comparison.csv", "metrics_full.json", "model_comparison.png",
                 "cm_efficientnet_b0.png", "cm_mobilenet_v2.png", "cm_resnet50.png"):
        f = RESULTS / name
        if f.exists():
            shutil.move(str(f), str(RESULTS / "pre_correction" / name))
            print(f"  métricas → pre_correction/{name}")
    for f in LOGS.glob("*.json"):
        dest = LOGS / "pre_correction" / f.name
        if not dest.exists():
            shutil.copy2(f, dest)
            print(f"  logs     → {dest}")

    # ── 2. Auditoria do split ANTIGO (a prova para o TCC) ────────────────────
    banner("2/9  Auditoria de vazamento do split ANTIGO")
    run(["audit_leakage.py"])
    old_report = RESULTS / "leakage_report_old_split.json"
    shutil.copy2(RESULTS / "leakage_report.json", old_report)
    with open(old_report, encoding="utf-8") as f:
        old = json.load(f)
    print(f"\n  [prova] split antigo: {old['test']['leaked_pct']}% do teste vazado "
          f"(veredicto: {old['verdict']}) → {old_report}", flush=True)

    # ── 3. Calibração do modelo antigo (demo continua utilizável) ────────────
    banner("3/9  Calibração do modelo ANTIGO no val antigo")
    run(["calibrate.py", "--data", "./data", "--binary", "--model", "efficientnet_b0"],
        fatal=False)
    for src, dst in (
        (WEIGHTS / "efficientnet_b0_binary_calibration.json",
         RESULTS / "pre_correction" / "efficientnet_b0_binary_calibration_old_split.json"),
        (RESULTS / "calibration_efficientnet_b0.png",
         RESULTS / "pre_correction" / "calibration_efficientnet_b0_old_split.png"),
    ):
        if src.exists():
            shutil.copy2(src, dst)

    # ── 4. Limpeza do staging e splits antigos ────────────────────────────────
    banner("4/9  Limpeza de staging/ e splits antigos (regras de rotulagem erradas)")
    for d in ("train", "val", "test", "staging"):
        target = DATA / d
        if target.exists():
            print(f"  a remover {target} ...", flush=True)
            shutil.rmtree(target)
    print("  limpo.", flush=True)

    # ── 5. Reorganização + split agrupado ─────────────────────────────────────
    banner("5/9  Reorganização do raw/ + split agrupado por identidade visual")
    run(["download_datasets.py", "--skip-download"])

    # ── 6. Auditoria do split NOVO — gate ─────────────────────────────────────
    banner("6/9  Auditoria de vazamento do split NOVO")
    run(["audit_leakage.py"])
    with open(RESULTS / "leakage_report.json", encoding="utf-8") as f:
        new = json.load(f)
    print(f"\n  split novo: {new['test']['leaked_pct']}% do teste vazado "
          f"(veredicto: {new['verdict']})", flush=True)
    if new["verdict"] != "OK":
        sys.exit("[pipeline] ABORTADO — o split novo ainda vaza. "
                 "Treinar agora produziria métricas inválidas.")

    # ── 7. Treino ──────────────────────────────────────────────────────────
    for m in MODELS:
        banner(f"7/9  Treino {m} ({EPOCHS} epochs, batch={BATCH})")
        run(["train.py", "--data", "./data", "--binary", "--model", m,
             "--epochs", str(EPOCHS), "--batch-size", str(BATCH), "--workers", "2"])

    # ── 8. Calibração ──────────────────────────────────────────────────────
    for m in MODELS:
        banner(f"8/9  Calibração {m}")
        run(["calibrate.py", "--data", "./data", "--binary", "--model", m])

    # ── 9. Avaliação final ─────────────────────────────────────────────────
    banner("9/9  Avaliação final no teste limpo")
    run(["evaluate.py", "--test-split", str(LOGS / "test_split_binary.json"),
         "--binary", "--workers", "2"], fatal=False)

    banner("PIPELINE DE CORRECÇÃO COMPLETO")
    print(f"  Duração total : {(time.time() - t0) / 3600:.1f} h")
    print(f"  Prova antiga  : {old_report}")
    print(f"  Split novo    : {DATA / 'split_metadata.json'}")
    print(f"  Métricas      : {RESULTS / 'metrics_comparison.csv'}")
    csv = RESULTS / "metrics_comparison.csv"
    if csv.exists():
        print("\n" + csv.read_text(encoding="utf-8"))
    print("\n  Reporte a acurácia BALANCEADA e cite leakage_report_old_split.json"
          "\n  como justificação da retratação dos 99,01%.")


if __name__ == "__main__":
    main()
