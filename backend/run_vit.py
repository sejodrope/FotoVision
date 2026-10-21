#!/usr/bin/env python
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
ViT-B/16 no dataset do ciclo 4 — experimento de comparação de arquitecturas (Q1).

Mesmo split, mesmas transformações, mesma loss, mesmo critério de selecção (F1
macro na validação), mesmas 30 épocas e batch 64 das CNNs do ciclo 4. O que muda,
para a receita usual de fine-tuning de transformers:
LR 5e-5 (era 1e-3), weight decay 0,05 (era 1e-4), 2 épocas de aquecimento e
clipping do gradiente em 1,0. Pesos iniciais: ImageNet-1k, como os das CNNs.

Corre igual no Colab e no Windows. Retomável: cada passo concluído deixa um
marker em --logs-dir, e o treino retoma do último checkpoint (train.py --resume).
Guia passo a passo: docs/GUIA_TREINO_VIT_COLAB.md.

Sequência:
  0. Conferência do dataset contra o manifesto do empacotamento
  1. Treino
  2. Calibração (temperature scaling + limiar de abstenção, na validação)
  3. Avaliação em três recortes do teste: completo, original e novo
  4. Previsões por imagem, para o McNemar contra as CNNs (compara_arquiteturas.py)
  5. As 13 fotos históricas de campo, pelo pipeline de produção

Não mexe em weights/, results/ nem logs/ das CNNs: tudo vai para subpastas vit/.
A auditoria de vazamento não é refeita — o split é o mesmo já auditado no ciclo 4
(results/ciclo4/leakage_report_ciclo4.json).

Uso:
    python run_vit.py
    python run_vit.py --weights-dir /content/drive/MyDrive/FitoVision_Colab/vit/weights ...
"""

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

BACKEND = Path(__file__).parent

_venv_python = BACKEND / ".venv" / "Scripts" / "python.exe"
PYTHON = str(_venv_python) if _venv_python.exists() else sys.executable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def banner(msg: str):
    print("\n" + "=" * 64, flush=True)
    print(f"  {msg}", flush=True)
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 64, flush=True)


def run(args: list[str], fatal: bool = True, env: dict | None = None) -> int:
    print(f"  $ {' '.join(args)}", flush=True)
    # PYTHONIOENCODING: train.py e test_campo.py imprimem '→' e '✅', que rebentam
    # no Windows quando a saída é redireccionada para um ficheiro (cp1252).
    rc = subprocess.run([PYTHON, *args], cwd=str(BACKEND),
                        env={**os.environ, "PYTHONIOENCODING": "utf-8", **(env or {})}).returncode
    if rc != 0 and fatal:
        sys.exit(f"[vit] ABORTADO — '{args[0]}' terminou com código {rc}.")
    return rc


def conferir_dataset(data: Path):
    """Conta as imagens por partição e compara com o manifesto do empacota_colab.py."""
    contagem = {
        split: {label: sum(1 for f in (data / split / label).iterdir()
                           if f.suffix.lower() in IMAGE_EXTENSIONS)
                for label in ("healthy", "anomalous")}
        for split in ("train", "val", "test")
    }
    for split, c in contagem.items():
        print(f"  {split:<6} healthy={c['healthy']:>6}  anomalous={c['anomalous']:>6}", flush=True)
    manifesto = data / "colab_manifest.json"
    if not manifesto.exists():
        print("  (sem colab_manifest.json — contagem não conferida)", flush=True)
        return
    esperado = json.loads(manifesto.read_text(encoding="utf-8"))["imagens"]
    if contagem != esperado:
        sys.exit(f"[vit] ABORTADO — o dataset não confere com o manifesto.\n"
                 f"       esperado: {esperado}\n       encontrado: {contagem}\n"
                 f"       Algum zip não subiu ou não foi extraído por inteiro.")
    print("  Confere com o manifesto do empacotamento.", flush=True)


def recortes_do_teste(data: Path, logs: Path) -> dict[str, Path]:
    """Separa o teste em original (ciclo 3) e novo (merge do ciclo 4), pelo manifesto."""
    with open(data / "ciclo4_merge_manifest.csv", encoding="utf-8") as f:
        novos = {Path(r["destino"]).name for r in csv.DictReader(f) if r["split"] == "test"}
    completo = logs / "test_split_binary.json"
    amostras = json.loads(completo.read_text(encoding="utf-8"))
    partes = {
        "teste_original": [s for s in amostras if Path(s[0]).name not in novos],
        "teste_novo":     [s for s in amostras if Path(s[0]).name in novos],
    }
    if len(partes["teste_novo"]) != len(novos):
        sys.exit(f"[vit] ABORTADO — teste novo tem {len(partes['teste_novo'])} imagens, "
                 f"o manifesto diz {len(novos)}.")
    caminhos = {"teste_completo": completo}
    for nome, subset in partes.items():
        p = logs / f"test_split_binary_{nome.removeprefix('teste_')}.json"
        p.write_text(json.dumps(subset), encoding="utf-8")
        caminhos[nome] = p
    for nome, p in caminhos.items():
        print(f"  {nome}: {len(json.loads(p.read_text(encoding='utf-8')))} imagens", flush=True)
    return caminhos


def main():
    ap = argparse.ArgumentParser(description="ViT-B/16 no dataset do ciclo 4")
    ap.add_argument("--model",         default="vit_b_16")
    ap.add_argument("--data",          default=str(BACKEND / "data"))
    ap.add_argument("--weights-dir",   default=str(BACKEND / "weights" / "vit"))
    ap.add_argument("--logs-dir",      default=str(BACKEND / "logs" / "vit"))
    ap.add_argument("--results-dir",   default=str(BACKEND / "results" / "vit"))
    ap.add_argument("--epochs",        type=int,   default=30)
    ap.add_argument("--batch-size",    type=int,   default=64)
    ap.add_argument("--lr",            type=float, default=5e-5)
    ap.add_argument("--weight-decay",  type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int,   default=2)
    ap.add_argument("--grad-clip",     type=float, default=1.0)
    ap.add_argument("--workers",       type=int,   default=2)
    args = ap.parse_args()

    t0 = time.time()
    m = args.model
    data, weights = Path(args.data), Path(args.weights_dir)
    logs, results = Path(args.logs_dir), Path(args.results_dir)
    for d in (weights, logs, results):
        d.mkdir(parents=True, exist_ok=True)

    banner("0/5  Conferência do dataset")
    conferir_dataset(data)

    # ── 1. Treino ──────────────────────────────────────────────────────────
    marker = logs / f"marker_vit_trained_{m}"
    if marker.exists():
        banner(f"1/5  Treino {m} — já concluído, a saltar")
    else:
        banner(f"1/5  Treino {m} ({args.epochs} epochs, batch={args.batch_size}, lr={args.lr})")
        run(["train.py", "--data", str(data), "--binary", "--model", m,
             "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
             "--lr", str(args.lr), "--weight-decay", str(args.weight_decay),
             "--warmup-epochs", str(args.warmup_epochs), "--grad-clip", str(args.grad_clip),
             "--workers", str(args.workers), "--resume",
             "--weights-dir", str(weights), "--logs-dir", str(logs)])
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        # O checkpoint de retomada (~1 GB com o estado do optimizador) já não serve.
        (weights / f"{m}_binary_last.pt").unlink(missing_ok=True)

    # ── 2. Calibração ──────────────────────────────────────────────────────
    banner(f"2/5  Calibração {m}")
    run(["calibrate.py", "--data", str(data), "--binary", "--model", m,
         "--weights-dir", str(weights), "--results-dir", str(results),
         "--workers", str(args.workers)])

    # ── 3. Avaliação ───────────────────────────────────────────────────────
    banner("3/5  Avaliação: teste completo, teste original e teste novo")
    for nome, split_json in recortes_do_teste(data, logs).items():
        out = results / nome
        out.mkdir(exist_ok=True)
        run(["evaluate.py", "--test-split", str(split_json), "--binary",
             "--weights-dir", str(weights), "--results-dir", str(out),
             "--batch-size", str(args.batch_size), "--workers", str(args.workers)], fatal=False)

    # ── 4. Previsões por imagem ────────────────────────────────────────────
    banner("4/5  Previsões por imagem (entrada do McNemar contra as CNNs)")
    run(["compara_arquiteturas.py", "--predict-only", "--force", "--data", str(data),
         "--weights-dir", str(weights), "--pred-dir", str(results / "predicoes"),
         "--batch-size", str(args.batch_size), "--workers", str(args.workers)], fatal=False)

    # ── 5. Campo ───────────────────────────────────────────────────────────
    banner("5/5  As 13 fotos históricas de campo, pelo pipeline de produção")
    run(["test_campo.py", "--dir", str(BACKEND / "campo"),
         "--out", str(results / "campo_resultados_vit.csv")], fatal=False,
        env={"BINARY_MODEL": m, "WEIGHTS_DIR": str(weights)})

    banner("ViT COMPLETO")
    print(f"  Duração desta execução : {(time.time() - t0) / 3600:.1f} h")
    hist = logs / f"{m}_binary_history.json"
    if hist.exists():
        h = json.loads(hist.read_text(encoding="utf-8"))
        print(f"  Treino                 : {h['elapsed_s'] / 3600:.1f} h, melhor F1 macro (val) "
              f"{h['best_val_f1']:.4f} na época {h['best_epoch']}")
    for nome in ("teste_completo", "teste_original", "teste_novo"):
        csv_path = results / nome / "metrics_comparison.csv"
        if csv_path.exists():
            print(f"\n  {nome}\n" + csv_path.read_text(encoding="utf-8"))
    print("  A latência acima é a da GPU desta máquina. Para comparar com as CNNs,"
          "\n  meça de novo na RTX 3050 (ver docs/GUIA_TREINO_VIT_COLAB.md, passo 6).")


if __name__ == "__main__":
    main()
