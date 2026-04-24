#!/usr/bin/env python
"""
Consolida múltiplas fontes de imagens na pasta data/ unificada do FitoVision.

Cada fonte pode ser:
  - Um diretório com subpastas nomeadas pelas classes do FitoVision (modo direto)
  - Um diretório com subpastas de nomes arbitrários + um JSON de mapeamento

Uso:
    # Ver estatísticas sem copiar nada (recomendado antes de tudo)
    python prepare_data.py --dry-run \
        --source plantvillage:plantvillage_map.json \
        --source plantdoc:plantdoc_map.json \
        --source fotos_agricultores

    # Consolidar com limite de 2000 imagens por classe
    python prepare_data.py \
        --source plantvillage:plantvillage_map.json \
        --source plantdoc:plantdoc_map.json \
        --source fotos_agricultores \
        --output data \
        --max-per-class 2000

    # Sem limite (usa tudo)
    python prepare_data.py \
        --source plantvillage:plantvillage_map.json \
        --output data

Formato do argumento --source:
    path              -> modo direto (pastas nomeadas pelas classes FitoVision)
    path:map.json     -> modo mapeado (planta correspondência pasta->classe)

Estrutura esperada das fotos dos agricultores (modo direto):
    fotos_agricultores/
        saudavel/        foto1.jpg, foto2.jpg ...
        mildio/          foto3.jpg ...
        oidio/           ...
        clorose_nitrogenio/
        danos_pragas/
        estresse_hidrico/
"""

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from app.config import CLASS_NAMES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_samples(source_dir: Path, folder_map: dict[str, str] | None) -> list[tuple[Path, str]]:
    """Retorna lista de (caminho_imagem, nome_classe)."""
    mapping = folder_map if folder_map else {c: c for c in CLASS_NAMES}
    samples: list[tuple[Path, str]] = []
    for folder_name, class_name in mapping.items():
        if class_name not in CLASS_NAMES:
            continue
        folder = source_dir / folder_name
        if not folder.is_dir():
            continue
        for f in folder.iterdir():
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((f, class_name))
    return samples


def unique_name(src_path: Path, source_tag: str, idx: int) -> str:
    """Gera nome único para evitar colisões entre fontes diferentes."""
    h = hashlib.md5(str(src_path).encode()).hexdigest()[:6]
    return f"{source_tag}_{idx:06d}_{h}{src_path.suffix.lower()}"


def print_distribution(title: str, dist: dict[str, int]):
    total = sum(dist.values())
    print(f"\n{title}")
    print(f"{'Classe':<25} {'Imagens':>8}  {'% do total':>10}")
    print("-" * 46)
    for cls in CLASS_NAMES:
        n = dist.get(cls, 0)
        pct = (n / total * 100) if total > 0 else 0
        flag = "  << INSUFICIENTE" if n < 300 else ""
        print(f"  {cls:<23} {n:>8}  {pct:>9.1f}%{flag}")
    print("-" * 46)
    print(f"  {'TOTAL':<23} {total:>8}")


def main():
    parser = argparse.ArgumentParser(description="Preparação de dados FitoVision")
    parser.add_argument(
        "--source", action="append", dest="sources", required=True,
        metavar="PATH[:MAP.JSON]",
        help="Fonte de dados. Use 'path' ou 'path:mapeamento.json'. Repete para múltiplas fontes.",
    )
    parser.add_argument("--output", default="./data",
                        help="Diretório de saída unificado (default: ./data)")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Limite máximo de imagens por classe (None = sem limite)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Apenas mostra estatísticas, não copia ficheiros")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    rng = random.Random(args.seed)

    # Parse das fontes
    sources: list[tuple[Path, dict | None, str]] = []  # (dir, folder_map, tag)
    for s in args.sources:
        if ":" in s:
            parts = s.split(":", 1)
            src_dir = Path(parts[0])
            map_path = Path(parts[1])
            if not map_path.exists():
                print(f"[ERRO] Ficheiro de mapeamento não encontrado: {map_path}")
                sys.exit(1)
            with open(map_path) as f:
                raw_map = json.load(f)
            # Ignora chaves que começam com _ (como _note)
            folder_map = {k: v for k, v in raw_map.items() if not k.startswith("_")}
            tag = src_dir.name
        else:
            src_dir = Path(s)
            folder_map = None
            tag = src_dir.name

        if not src_dir.is_dir():
            print(f"[ERRO] Diretório não encontrado: {src_dir}")
            sys.exit(1)
        sources.append((src_dir, folder_map, tag))

    # Coleta amostras de todas as fontes
    all_by_class: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    per_source_dist: dict[str, dict[str, int]] = {}

    for src_dir, folder_map, tag in sources:
        samples = collect_samples(src_dir, folder_map)
        dist: dict[str, int] = defaultdict(int)
        for path, cls in samples:
            all_by_class[cls].append((path, tag))
            dist[cls] += 1
        per_source_dist[tag] = dict(dist)
        total = sum(dist.values())
        print(f"\nFonte '{tag}': {total} imagens de {src_dir}")
        for cls in CLASS_NAMES:
            if dist.get(cls, 0) > 0:
                print(f"  {cls:<25}: {dist[cls]}")

    # Distribuição bruta antes de aplicar limite
    raw_dist = {cls: len(imgs) for cls, imgs in all_by_class.items()}
    print_distribution("=== Distribuição bruta (todas as fontes) ===", raw_dist)

    # Aplica limite por classe
    selected_by_class: dict[str, list[tuple[Path, str]]] = {}
    for cls in CLASS_NAMES:
        imgs = all_by_class.get(cls, [])
        rng.shuffle(imgs)
        if args.max_per_class:
            imgs = imgs[: args.max_per_class]
        selected_by_class[cls] = imgs

    final_dist = {cls: len(imgs) for cls, imgs in selected_by_class.items()}

    if args.max_per_class:
        print_distribution(
            f"=== Distribuição final (máx {args.max_per_class}/classe) ===",
            final_dist,
        )

    if args.dry_run:
        print("\n[DRY RUN] Nenhum ficheiro foi copiado.")
        _print_recommendations(final_dist)
        return

    # Copia ficheiros para output/
    output = Path(args.output)
    for cls in CLASS_NAMES:
        (output / cls).mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for cls, imgs in selected_by_class.items():
        for i, (src_path, tag) in enumerate(imgs):
            dest_name = unique_name(src_path, tag, i)
            dest = output / cls / dest_name
            if dest.exists():
                skipped += 1
                continue
            shutil.copy2(src_path, dest)
            copied += 1

    print(f"\nConcluído: {copied} imagens copiadas, {skipped} já existentes ignoradas.")
    print(f"Dataset unificado em: {output.resolve()}")
    _print_recommendations(final_dist)


def _print_recommendations(dist: dict[str, int]):
    print("\n=== Recomendações ===")
    for cls in CLASS_NAMES:
        n = dist.get(cls, 0)
        if n == 0:
            print(f"  {cls:<25}: VAZIO — necessário recolher imagens")
        elif n < 300:
            print(f"  {cls:<25}: {n} imagens — recomendado pelo menos 500")
        elif n < 800:
            print(f"  {cls:<25}: {n} imagens — aceitável, mais é melhor")
        else:
            print(f"  {cls:<25}: {n} imagens — OK")

    print("\nPróximo passo: python train.py --data ./data --model all")


if __name__ == "__main__":
    main()
