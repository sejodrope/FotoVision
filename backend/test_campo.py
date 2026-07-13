#!/usr/bin/env python
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Teste de campo — avalia o modelo em fotografias reais (quintal, telemóvel).

Mede o GAP DE DOMÍNIO: quanto o desempenho cai das fotos de estúdio (PlantVillage,
onde o modelo treinou) para fotos reais. Ver docs/GUIA_TESTE_DE_CAMPO.md.

Este script exercita EXACTAMENTE o pipeline de produção (o mesmo do endpoint
/predict): pré-processamento idêntico ao de validação, guarda de vegetação (ExG),
calibração e política de abstenção. O que sair aqui é o que sairia na app.

ESTRUTURA ESPERADA DAS FOTOS
────────────────────────────
    <pasta>/healthy/     *.jpg   ← folhas que VOCÊ sabe estarem saudáveis
    <pasta>/anomalous/   *.jpg   ← folhas que VOCÊ sabe estarem doentes
    <pasta>/nao_folha/   *.jpg   ← fotos que não são folha (parede, mão, chão)

As pastas são o GABARITO (ground truth). Só é preciso ter as que você tiver fotos.

USO
───
    python test_campo.py --dir ./campo
    python test_campo.py --dir ./campo --out ./results/campo_resultados.csv

SAÍDA
─────
    results/campo_resultados.csv   — uma linha por foto (para a planilha do TCC)
    Matriz de confusão + acurácia balanceada de campo + taxa de abstenção, no ecrã.
"""

import argparse
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.ml.preprocessing import _transform          # mesmo transform da produção
from app.ml.inference import predict_binary
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Pastas de gabarito reconhecidas → rótulo verdadeiro
GROUND_TRUTH_DIRS = {
    "healthy": "healthy",
    "anomalous": "anomalous",
    "nao_folha": "nao_folha",
    "not_a_leaf": "nao_folha",
}


def load_and_predict(path: Path) -> dict:
    """Corre o pipeline de produção completo sobre uma foto."""
    with open(path, "rb") as f:
        image_bytes = f.read()
    image = Image.open(path).convert("RGB")
    tensor = _transform(image).unsqueeze(0)
    # image=... activa a guarda de vegetação (ExG), tal como no endpoint real.
    return predict_binary(tensor, image=image)


def main():
    parser = argparse.ArgumentParser(description="Teste de campo — FitoVision")
    parser.add_argument("--dir", default=str(BASE_DIR / "campo"),
                        help="Pasta com subpastas healthy/ anomalous/ nao_folha/")
    parser.add_argument("--out", default=str(BASE_DIR / "results" / "campo_resultados.csv"))
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.exists():
        sys.exit(f"[ERRO] Pasta não encontrada: {root}\n"
                 f"       Crie {root}/healthy/, {root}/anomalous/ e/ou {root}/nao_folha/\n"
                 f"       e coloque as fotos lá. Ver docs/GUIA_TESTE_DE_CAMPO.md.")

    # ── Recolhe as fotos com o seu rótulo verdadeiro ──────────────────────────
    items: list[tuple[Path, str]] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        truth = GROUND_TRUTH_DIRS.get(sub.name.lower())
        if truth is None:
            print(f"[aviso] pasta ignorada (nome não reconhecido): {sub.name}")
            continue
        for img in sorted(sub.iterdir()):
            if img.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((img, truth))

    if not items:
        sys.exit(f"[ERRO] Nenhuma foto encontrada em {root}. "
                 f"Ver a estrutura esperada em docs/GUIA_TESTE_DE_CAMPO.md.")

    print(f"\nTeste de campo — {len(items)} fotos de {root}\n")

    # ── Corre o modelo ────────────────────────────────────────────────────────
    rows = []
    # confusão 3x3 sobre {healthy, anomalous, absteve} para as folhas reais;
    # nao_folha é contabilizado à parte (queremos que dê not_a_leaf).
    confusion = {t: {"healthy": 0, "anomalous": 0, "inconclusive": 0, "not_a_leaf": 0}
                 for t in ("healthy", "anomalous", "nao_folha")}

    for img, truth in items:
        try:
            result = load_and_predict(img)
        except (UnidentifiedImageError, OSError) as exc:
            print(f"  [corrompida] {img.name}: {exc}")
            continue

        pred = result["label"]
        conf = result["confidence"]
        veg = result.get("vegetation_fraction")
        confusion[truth][pred] += 1

        # "acertou?" só se aplica a diagnóstico de classe sobre folha real
        if truth == "nao_folha":
            correct = "✅" if pred == "not_a_leaf" else "❌"
        elif pred in ("healthy", "anomalous"):
            correct = "✅" if pred == truth else "❌"
        else:
            correct = "—"   # inconclusive / not_a_leaf sobre folha real: nem acerto nem erro

        rows.append({
            "ficheiro": str(img.relative_to(root)),
            "verdade": truth,
            "resposta": pred,
            "confianca": f"{conf:.4f}",
            "veg_fraction": f"{veg:.3f}" if veg is not None else "",
            "acertou": correct,
        })
        print(f"  {correct}  {img.name:<32} verdade={truth:<10} → {pred:<12} "
              f"conf={conf:.2f}" + (f" veg={veg:.2f}" if veg is not None else ""))

    # ── Métricas ──────────────────────────────────────────────────────────────
    def acc_of(truth: str) -> tuple[int, int]:
        """(acertos, total_diagnosticados) para uma classe de folha real."""
        c = confusion[truth]
        decided = c["healthy"] + c["anomalous"]
        correct = c[truth]
        return correct, decided

    h_correct, h_decided = acc_of("healthy")
    a_correct, a_decided = acc_of("anomalous")

    total = len(rows)
    n_abstain = sum(confusion[t]["inconclusive"] for t in ("healthy", "anomalous"))
    n_reject_leaf = sum(confusion[t]["not_a_leaf"] for t in ("healthy", "anomalous"))
    n_decided = h_decided + a_decided
    plain_correct = h_correct + a_correct

    # Acurácia balanceada = média das acurácias por classe (não enviesa por desbalanço)
    per_class = []
    if h_decided:
        per_class.append(h_correct / h_decided)
    if a_decided:
        per_class.append(a_correct / a_decided)
    bal_acc = sum(per_class) / len(per_class) if per_class else 0.0

    print("\n" + "=" * 60)
    print("  RESULTADO DO TESTE DE CAMPO")
    print("=" * 60)
    print("\n  Matriz de confusão (linha = verdade, coluna = resposta):")
    print(f"    {'':<12}{'healthy':>10}{'anomalous':>11}{'inconcl.':>10}{'not_leaf':>10}")
    for t in ("healthy", "anomalous", "nao_folha"):
        c = confusion[t]
        print(f"    {t:<12}{c['healthy']:>10}{c['anomalous']:>11}"
              f"{c['inconclusive']:>10}{c['not_a_leaf']:>10}")

    print(f"\n  Fotos de folha diagnosticadas : {n_decided} de "
          f"{n_decided + n_abstain + n_reject_leaf}")
    if n_decided:
        print(f"  Acurácia simples (campo)      : {plain_correct/n_decided:.1%}  "
              f"({plain_correct}/{n_decided})")
        print(f"  Acurácia BALANCEADA (campo)   : {bal_acc:.1%}")
        if h_decided:
            print(f"    → healthy   : {h_correct/h_decided:.1%}  ({h_correct}/{h_decided})")
        if a_decided:
            print(f"    → anomalous : {a_correct/a_decided:.1%}  ({a_correct}/{a_decided})")
    if total:
        print(f"  Taxa de abstenção (inconcl.)  : {n_abstain/total:.1%}  ({n_abstain}/{total})")
        print(f"  Folhas reais rejeitadas (ExG) : {n_reject_leaf/total:.1%}  "
              f"({n_reject_leaf}/{total})  ← se alto, baixar min_vegetation_fraction")

    # nao_folha: quantas foram corretamente rejeitadas
    nf = confusion["nao_folha"]
    nf_total = sum(nf.values())
    if nf_total:
        print(f"\n  Guarda de domínio (fotos que NÃO são folha):")
        print(f"    corretamente rejeitadas (not_a_leaf) : {nf['not_a_leaf']}/{nf_total}")
        wrong = nf["healthy"] + nf["anomalous"]
        if wrong:
            print(f"    ⚠  {wrong} não-folhas receberam diagnóstico "
                  f"→ subir min_vegetation_fraction")

    print("\n  COMPARE com o teste no domínio (results/metrics_comparison.csv):")
    print("    EfficientNet-B0 bal. acc = 98,6%  ← estúdio (PlantVillage)")
    print("  A diferença entre 98,6% e a acurácia de campo acima É o gap de domínio,")
    print("  e é o resultado central do experimento. Ver docs/GUIA_TESTE_DE_CAMPO.md §5.")
    print("=" * 60)

    # ── CSV ───────────────────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ficheiro", "verdade", "resposta", "confianca",
                           "veg_fraction", "acertou"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Resultados por foto salvos em: {out}")
    print(f"  (abra no Excel/Sheets — é a planilha da §3 do guia)")


if __name__ == "__main__":
    main()
