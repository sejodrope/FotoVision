#!/usr/bin/env python
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Ciclo 4 — retreino com espinafre + lettuce_downy_v1 (docs/CORRECOES_METODOLOGICAS.md §10.9).

Pré-requisitos (já feitos manualmente, ver PROXIMOS_PASSOS_TCC2.md):
  - backup do ciclo 3 em weights/ciclo3/, results/ciclo3/, logs/ciclo3/
  - python merge_ciclo4.py --apply

Sequência (retomável — cada passo concluído deixa um marker em logs/):
  1. Baseline: pesos do ciclo 3 avaliados só no teste NOVO → results/ciclo4/baseline_ciclo3_teste_novo/
  2. Auditoria de vazamento do split com o merge — aborta se não for OK
  3. Treino de efficientnet_b0, resnet50 e mobilenet_v2 (30 epochs, F1 macro)
  4. Calibração dos três
  5. Avaliação em três recortes do teste:
       results/                                   — teste completo (ciclo 4)
       results/ciclo4/teste_original/             — só o teste do ciclo 3 (comparação justa)
       results/ciclo4/teste_novo/                 — só espinafre + lettuce downy

Diferente do run_correction_pipeline.py: não mexe em results/pre_correction/
(a prova da retratação) nem refaz o split.

Uso:
    python run_ciclo4.py
"""

import csv
import json
import subprocess
import time
from pathlib import Path

BACKEND = Path(__file__).parent
DATA    = BACKEND / "data"
WEIGHTS = BACKEND / "weights"
RESULTS = BACKEND / "results"
LOGS    = BACKEND / "logs"
OUT     = RESULTS / "ciclo4"

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
        sys.exit(f"[ciclo4] ABORTADO — '{args[0]}' terminou com código {rc}.")
    return rc


def _keep_awake():
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print("[ciclo4] Suspensão do sistema bloqueada enquanto o pipeline corre.", flush=True)


def _new_files() -> set[str]:
    with open(DATA / "ciclo4_merge_manifest.csv", encoding="utf-8") as f:
        return {Path(r["destino"]).name for r in csv.DictReader(f) if r["split"] == "test"}


def _write_test_subsets() -> tuple[Path, Path]:
    """Separa o teste completo em original (ciclo 3) e novo (merge do ciclo 4)."""
    new = _new_files()
    test_ds_json = LOGS / "test_split_binary.json"
    with open(test_ds_json, encoding="utf-8") as f:
        samples = json.load(f)
    original = [s for s in samples if Path(s[0]).name not in new]
    novo     = [s for s in samples if Path(s[0]).name in new]
    if len(novo) != len(new):
        sys.exit(f"[ciclo4] ABORTADO — teste novo tem {len(novo)} imagens, manifesto diz {len(new)}.")
    paths = (LOGS / "test_split_binary_original.json", LOGS / "test_split_binary_novo.json")
    for p, subset in zip(paths, (original, novo)):
        p.write_text(json.dumps(subset), encoding="utf-8")
        print(f"  {p.name}: {len(subset)} imagens", flush=True)
    return paths


def _new_test_split_from_folders() -> Path:
    """Teste novo antes do treino (o test_split_binary.json ainda é o do ciclo 3)."""
    new = _new_files()
    samples = []
    for label_idx, label in enumerate(("healthy", "anomalous")):
        for p in sorted((DATA / "test" / label).iterdir()):
            if p.name in new:
                samples.append([str(p.relative_to(BACKEND)), label_idx])
    path = LOGS / "test_split_binary_novo.json"
    path.write_text(json.dumps(samples), encoding="utf-8")
    return path


def main():
    t0 = time.time()
    _keep_awake()
    OUT.mkdir(parents=True, exist_ok=True)

    # ── 1. Baseline do ciclo 3 no teste novo ─────────────────────────────────
    marker = LOGS / "marker_ciclo4_baseline"
    if marker.exists():
        banner("1/5  Baseline do ciclo 3 no teste novo — já feito, a saltar")
    else:
        banner("1/5  Baseline: pesos do ciclo 3 no teste NOVO (espinafre + lettuce downy)")
        out = OUT / "baseline_ciclo3_teste_novo"
        out.mkdir(exist_ok=True)
        run(["evaluate.py", "--test-split", str(_new_test_split_from_folders()), "--binary",
             "--weights-dir", str(WEIGHTS / "ciclo3"), "--results-dir", str(out),
             "--workers", "2"], fatal=False)
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")

    # ── 2. Auditoria de vazamento — gate ─────────────────────────────────────
    marker = LOGS / "marker_ciclo4_audit_ok"
    if marker.exists():
        banner("2/5  Auditoria do split com merge — já aprovada, a saltar")
    else:
        banner("2/5  Auditoria de vazamento do split com merge")
        run(["audit_leakage.py"])
        with open(RESULTS / "leakage_report.json", encoding="utf-8") as f:
            rep = json.load(f)
        print(f"\n  split ciclo 4: {rep['test']['leaked_pct']}% do teste vazado "
              f"(veredicto: {rep['verdict']})", flush=True)
        (OUT / "leakage_report_ciclo4.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        if rep["verdict"] != "OK":
            sys.exit("[ciclo4] ABORTADO — o split com merge vaza. Treinar agora produziria "
                     "métricas inválidas.")
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")

    # ── 3. Treino ──────────────────────────────────────────────────────────
    for m in MODELS:
        marker = LOGS / f"marker_ciclo4_trained_{m}"
        if marker.exists():
            banner(f"3/5  Treino {m} — já concluído, a saltar")
            continue
        banner(f"3/5  Treino {m} ({EPOCHS} epochs, batch={BATCH})")
        run(["train.py", "--data", "./data", "--binary", "--model", m,
             "--epochs", str(EPOCHS), "--batch-size", str(BATCH), "--workers", "2"])
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")

    # ── 4. Calibração ──────────────────────────────────────────────────────
    for m in MODELS:
        banner(f"4/5  Calibração {m}")
        run(["calibrate.py", "--data", "./data", "--binary", "--model", m])

    # ── 5. Avaliação ───────────────────────────────────────────────────────
    banner("5/5  Avaliação: teste completo, teste original (ciclo 3) e teste novo")
    run(["evaluate.py", "--test-split", str(LOGS / "test_split_binary.json"),
         "--binary", "--workers", "2"], fatal=False)
    original, novo = _write_test_subsets()
    for split_json, name in ((original, "teste_original"), (novo, "teste_novo")):
        out = OUT / name
        out.mkdir(exist_ok=True)
        run(["evaluate.py", "--test-split", str(split_json), "--binary",
             "--results-dir", str(out), "--workers", "2"], fatal=False)

    banner("CICLO 4 COMPLETO")
    print(f"  Duração total : {(time.time() - t0) / 3600:.1f} h")
    for label, csv_path in (
        ("Teste completo (ciclo 4)",      RESULTS / "metrics_comparison.csv"),
        ("Teste original — ciclo 3",      RESULTS / "ciclo3" / "metrics_comparison.csv"),
        ("Teste original — ciclo 4",      OUT / "teste_original" / "metrics_comparison.csv"),
        ("Teste novo — pesos do ciclo 3", OUT / "baseline_ciclo3_teste_novo" / "metrics_comparison.csv"),
        ("Teste novo — ciclo 4",          OUT / "teste_novo" / "metrics_comparison.csv"),
    ):
        if csv_path.exists():
            print(f"\n  {label}\n" + csv_path.read_text(encoding="utf-8"))
    print("\n  O campo NÃO foi avaliado: o ciclo 4 fica congelado até as fotos novas chegarem"
          "\n  (avaliação cega — ver PROXIMOS_PASSOS_TCC2.md).")


if __name__ == "__main__":
    main()
