#!/usr/bin/env python
"""
Pipeline sequencial FitoVision:
  1. Aguarda EfficientNet-B0 terminar
  2. Avalia EfficientNet-B0 (evaluate.py --binary)
  3. Treina ResNet50 --binary (30 epochs)
  4. Avalia EfficientNet-B0 + ResNet50 juntos
  5. Mostra tabela comparativa final

Uso:
    python run_pipeline.py

Corre em background — pode fechar o terminal principal.
O script usa a venv em .venv/Scripts/python.exe se existir.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

BACKEND      = Path(__file__).parent
LOGS         = BACKEND / "logs"
WEIGHTS      = BACKEND / "weights"
_venv_python = BACKEND / ".venv" / "Scripts" / "python.exe"
PYTHON       = str(_venv_python) if _venv_python.exists() else sys.executable

TOTAL_EPOCHS = 30
POLL_SECS    = 300   # verificar a cada 5 min
STALE_POLLS  = 2     # 2 verificações sem mudança = treino terminou (early stopping)


def _load_history(log_path: Path) -> dict | None:
    try:
        with open(log_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def wait_for_training(model_name: str) -> dict:
    log_path   = LOGS / f"{model_name}_history.json"
    last_count = -1
    stale      = 0

    print(f"\n[pipeline] A monitorizar {model_name} (poll a cada {POLL_SECS//60} min)...")

    while True:
        data = _load_history(log_path)
        if data is not None:
            count   = len(data.get("history", []))
            best    = data.get("best_val_acc", 0.0)
            elapsed = data.get("elapsed_s", 0.0) / 60

            if count != last_count:
                last_count = count
                stale      = 0
                print(f"[pipeline] {model_name}: epoch {count}/{TOTAL_EPOCHS}  "
                      f"best_val_acc={best:.4f}  elapsed={elapsed:.1f}min")
            else:
                stale += 1

            if count >= TOTAL_EPOCHS or stale >= STALE_POLLS:
                reason = "30 epochs completos" if count >= TOTAL_EPOCHS else f"early stopping (epoch {count})"
                print(f"[pipeline] {model_name}: TREINO COMPLETO — {reason}, best_val_acc={best:.4f}")
                return data

        time.sleep(POLL_SECS)


def run_evaluate(label: str = "") -> int:
    tag = f" ({label})" if label else ""
    print(f"\n[pipeline] A avaliar modelos{tag}...")
    result = subprocess.run(
        [
            PYTHON, "evaluate.py",
            "--test-split", str(LOGS / "test_split_binary.json"),
            "--binary",
            "--workers", "2",
        ],
        cwd=str(BACKEND),
    )
    if result.returncode != 0:
        print(f"[pipeline] AVISO: evaluate.py terminou com código {result.returncode}")
    return result.returncode


def run_train(model_name: str, batch_size: int = 64) -> int:
    print(f"\n[pipeline] A iniciar treino {model_name} ({TOTAL_EPOCHS} epochs, batch={batch_size})...")
    result = subprocess.run(
        [
            PYTHON, "train.py",
            "--data",       "./data",
            "--binary",
            "--model",      model_name,
            "--epochs",     str(TOTAL_EPOCHS),
            "--batch-size", str(batch_size),
            "--workers",    "2",
        ],
        cwd=str(BACKEND),
    )
    if result.returncode != 0:
        print(f"[pipeline] ERRO: {model_name} terminou com código {result.returncode}")
    return result.returncode


def main():
    print("=" * 60)
    print("  FitoVision — Pipeline Sequencial")
    print(f"  Python : {PYTHON}")
    print(f"  Logs   : {LOGS}")
    print("=" * 60)

    # ── 1. EfficientNet-B0 ─────────────────────────────────────
    eff_log = LOGS / "efficientnet_b0_history.json"
    data    = _load_history(eff_log)
    if data and len(data.get("history", [])) >= TOTAL_EPOCHS:
        print("[pipeline] EfficientNet-B0 já completo — a saltar espera.")
    else:
        wait_for_training("efficientnet_b0")

    # ── 2. Avaliar EfficientNet-B0 ─────────────────────────────
    run_evaluate("EfficientNet-B0")

    # ── 3. Treinar ResNet50 ────────────────────────────────────
    rc = run_train("resnet50", batch_size=64)
    if rc != 0:
        sys.exit(rc)

    # ── 4. Avaliação final (ambos os modelos) ──────────────────
    run_evaluate("EfficientNet-B0 + ResNet50")

    # ── 5. Sumário ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETO")
    print(f"  Resultados : {BACKEND / 'results'}/")
    print(f"  CSV        : {BACKEND / 'results' / 'metrics_comparison.csv'}")
    print("=" * 60)

    csv = BACKEND / "results" / "metrics_comparison.csv"
    if csv.exists():
        print("\n" + csv.read_text())


if __name__ == "__main__":
    main()
