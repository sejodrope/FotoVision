#!/usr/bin/env python
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Ciclo 5 — retreino com as correções da auditoria de 13/10/2026
(docs/CORRECOES_METODOLOGICAS.md §14.8 e §15).

Pré-requisito:
    python prepara_ciclo5.py --apply        # cria data_c5/ (não mexe em data/)

Não toca em nada do ciclo 4: os pesos vão para weights/ciclo5/, os logs para
logs/ciclo5/ e os resultados para results/ciclo5/. O modelo servido pela aplicação
continua a ser o do ciclo 4 até alguém copiar os pesos novos para weights/.

Sequência (retomável — cada passo concluído deixa um marker em logs/ciclo5/):
  1. Conferência do split: nenhuma foto original de alface em mais de um lado
  2. Auditoria de quase-duplicatas (audit_leakage.py) — aborta se não for OK
  3. Treino (30 épocas, lote 64, amostragem equilibrada por cultura)
  4. Calibração
  5. Avaliação por subconjunto (avalia_ciclo5.py), incluindo a fonte retida

Uso:
    python run_ciclo5.py                                          # só o EfficientNet-B0 (~6 h)
    python run_ciclo5.py --modelos efficientnet_b0 mobilenet_v2 resnet50
"""

import argparse
import csv
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

BACKEND = Path(__file__).parent
DATA    = BACKEND / "data_c5"
WEIGHTS = BACKEND / "weights" / "ciclo5"
RESULTS = BACKEND / "results" / "ciclo5"
LOGS    = BACKEND / "logs" / "ciclo5"

_venv_python = BACKEND / ".venv" / "Scripts" / "python.exe"
PYTHON = str(_venv_python) if _venv_python.exists() else sys.executable

EPOCHS = 30
BATCH  = 64
# Receita de otimização própria do ViT, a mesma do run_vit.py (CORRECOES §12.5).
EXTRA = {"vit_b_16": ["--lr", "5e-5", "--weight-decay", "0.05", "--warmup-epochs", "2", "--grad-clip", "1.0"]}


def banner(msg: str):
    print("\n" + "=" * 64, flush=True)
    print(f"  {msg}", flush=True)
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 64, flush=True)


def run(args: list[str], fatal: bool = True) -> int:
    print(f"  $ {' '.join(args)}", flush=True)
    rc = subprocess.run([PYTHON, *args], cwd=str(BACKEND)).returncode
    if rc != 0 and fatal:
        sys.exit(f"[ciclo5] ABORTADO — '{args[0]}' terminou com código {rc}.")
    return rc


def _keep_awake():
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print("[ciclo5] Suspensão do sistema bloqueada enquanto o pipeline corre.", flush=True)


def conferir_proveniencia() -> None:
    lados = defaultdict(set)
    with open(DATA / "manifesto_ciclo5.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["foto_original"]:
                lados[r["foto_original"]].add(r["split"])
    espalhadas = [k for k, v in lados.items() if len(v) > 1]
    print(f"  fotos originais de alface: {len(lados)}; em mais de um lado: {len(espalhadas)}", flush=True)
    if espalhadas:
        sys.exit("[ciclo5] ABORTADO — há foto original de alface em mais de uma partição.")


def main():
    ap = argparse.ArgumentParser(description="Ciclo 5 — treino, calibração e avaliação")
    ap.add_argument("--modelos", nargs="+", default=["efficientnet_b0"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sem-auditoria", action="store_true",
                    help="pula a auditoria de quase-duplicatas (no Colab: ela é feita no PC, onde está o cache)")
    args = ap.parse_args()

    if not (DATA / "manifesto_ciclo5.csv").exists():
        sys.exit("[ciclo5] data_c5/ não existe. Rode: python prepara_ciclo5.py --apply")
    t0 = time.time()
    _keep_awake()
    for d in (WEIGHTS, RESULTS, LOGS):
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Proveniência ──────────────────────────────────────────────────────
    banner("1/5  Conferência do split por proveniência")
    conferir_proveniencia()

    # ── 2. Quase-duplicatas — gate ───────────────────────────────────────────
    marker = LOGS / "marker_audit_ok"
    if args.sem_auditoria:
        banner("2/5  Auditoria de quase-duplicatas — feita no PC (results/ciclo5/leakage_report.json)")
    elif marker.exists():
        banner("2/5  Auditoria de quase-duplicatas — já aprovada, a saltar")
    else:
        banner("2/5  Auditoria de quase-duplicatas do split do ciclo 5")
        run(["audit_leakage.py", "--data", "./data_c5", "--results-dir", str(RESULTS),
             "--hash-cache", str(BACKEND / "logs" / "hash_cache_c5.jsonl")])
        rep = json.loads((RESULTS / "leakage_report.json").read_text(encoding="utf-8"))
        print(f"\n  split ciclo 5: {rep['test']['leaked_pct']}% do teste com quase-duplicata no treino "
              f"(veredicto: {rep['verdict']})", flush=True)
        if rep["verdict"] != "OK":
            sys.exit("[ciclo5] ABORTADO — o split vaza. Treinar agora produziria métricas inválidas.")
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")

    # ── 3 e 4. Treino e calibração, um modelo de cada vez ─────────────────────
    # Calibrar logo depois de cada treino deixa cada modelo utilizável assim que
    # termina, mesmo que a sessão caia antes do próximo.
    for m in args.modelos:
        marker = LOGS / f"marker_trained_{m}"
        if marker.exists():
            banner(f"3/5  Treino {m} — já concluído, a saltar")
        else:
            banner(f"3/5  Treino {m} ({EPOCHS} épocas, lote {BATCH}, amostragem por cultura)")
            run(["train.py", "--data", "./data_c5", "--binary", "--model", m,
                 "--epochs", str(EPOCHS), "--batch-size", str(BATCH), "--workers", str(args.workers),
                 "--amostragem", "cultura", "--weights-dir", str(WEIGHTS), "--logs-dir", str(LOGS),
                 "--resume", *EXTRA.get(m, [])])
            marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        if not (WEIGHTS / f"{m}_binary_calibration.json").exists():
            banner(f"4/5  Calibração {m}")
            run(["calibrate.py", "--data", "./data_c5", "--binary", "--model", m,
                 "--weights-dir", str(WEIGHTS), "--results-dir", str(RESULTS), "--workers", str(args.workers)])

    # ── 5. Avaliação ─────────────────────────────────────────────────────────
    banner("5/5  Avaliação por subconjunto e na fonte retida")
    run(["avalia_ciclo5.py", "--modelos", *args.modelos, "--workers", str(args.workers)], fatal=False)

    banner("CICLO 5 COMPLETO")
    print(f"  Duração total : {(time.time() - t0) / 3600:.1f} h")
    print("\n  O modelo servido continua a ser o do ciclo 4. Para comparar os dois em campo:"
          "\n      python roda_campo.py"
          "\n  A troca do modelo servido só depois dessa comparação (PROXIMOS_PASSOS_TCC2.md).")


if __name__ == "__main__":
    main()
