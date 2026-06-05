#!/usr/bin/env python
"""
Módulo de pré-processamento e DataLoaders para o FitoVision.

━━━ Modo 1 — Pipeline binário (novo) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Importável como módulo. Usa albumentations + lê a estrutura já splitada
gerada por download_datasets.py:

    from prepare_data import make_binary_loaders
    train_loader, val_loader, test_loader = make_binary_loaders("./data")

Estrutura esperada em data_dir:
    data/
      train/healthy/   train/anomalous/
      val/healthy/     val/anomalous/
      test/healthy/    test/anomalous/

━━━ Modo 2 — CLI de consolidação multi-classe (original) ━━━━━━━━━━━━
Consolida várias fontes numa pasta unificada para o pipeline de 6 classes:

    python prepare_data.py --source plantvillage:plantvillage_map.json
    python prepare_data.py --source plantdoc:plantdoc_map.json --dry-run
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("fitovision.prepare_data")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

BINARY_CLASSES = ["healthy", "anomalous"]  # label 0=healthy, 1=anomalous

# ─── Albumentations transforms ────────────────────────────────────────────────

def _build_transforms(split: str):
    """
    Retorna transforms albumentations para 'train', 'val' ou 'test'.
    Treino: augmentation completo + normalização ImageNet.
    Val/Test: apenas resize + normalização.
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError as e:
        raise ImportError(
            "albumentations não instalado. Execute: pip install albumentations"
        ) from e

    imagenet_norm = dict(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
    )

    if split == "train":
        return A.Compose([
            A.Resize(256, 256),
            A.RandomCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.1),
            A.Normalize(**imagenet_norm),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(**imagenet_norm),
            ToTensorV2(),
        ])


# ─── Dataset com albumentations ───────────────────────────────────────────────

class BinaryAlbDataset(Dataset):
    """
    Dataset binário que carrega imagens de:
        split_dir/healthy/
        split_dir/anomalous/

    label: 0 = healthy, 1 = anomalous
    Usa transforms albumentations (espera array numpy H×W×C).
    """

    def __init__(self, split_dir: str | Path, transform=None):
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(BINARY_CLASSES)}
        self.samples: list[tuple[Path, int]] = []
        self._scan()

    def _scan(self):
        for cls in BINARY_CLASSES:
            cls_dir = self.split_dir / cls
            if not cls_dir.is_dir():
                continue
            idx = self.class_to_idx[cls]
            for f in cls_dir.iterdir():
                if f.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((f, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            logger.warning("Imagem ilegível ignorada: %s", path)
            img = Image.new("RGB", (224, 224), 0)

        img_np = np.array(img)

        if self.transform is not None:
            augmented = self.transform(image=img_np)
            tensor = augmented["image"]
        else:
            tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0

        return tensor, label

    def class_distribution(self) -> dict[str, int]:
        from collections import Counter
        return dict(Counter(BINARY_CLASSES[lbl] for _, lbl in self.samples))


# ─── Função principal: cria os 3 DataLoaders ─────────────────────────────────

def make_binary_loaders(
    data_dir: str | Path = "./data",
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Cria train/val/test DataLoaders a partir da estrutura:
        data_dir/train/healthy/   data_dir/train/anomalous/
        data_dir/val/...          data_dir/test/...

    Treino usa augmentation albumentations completo.
    Val e test usam apenas resize + normalize (sem augmentation).

    Retorna: (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)

    loaders = []
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Pasta '{split_dir}' não encontrada. "
                "Execute download_datasets.py primeiro."
            )
        transform = _build_transforms(split)
        ds = BinaryAlbDataset(split_dir, transform=transform)
        shuffle = split == "train"
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        loaders.append(loader)
        dist = ds.class_distribution()
        print(f"  {split:<6}: {len(ds):>6} imagens  {dist}")

    return tuple(loaders)  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════════
# ─── CLI de consolidação multi-classe (pipeline original de 6 classes) ────────
# ═══════════════════════════════════════════════════════════════════════════════

def _cli_consolidate():
    """
    Consolida múltiplas fontes de imagens na pasta data/ unificada do FitoVision.

    Uso:
        python prepare_data.py --dry-run \\
            --source plantvillage:plantvillage_map.json \\
            --source plantdoc:plantdoc_map.json

        python prepare_data.py \\
            --source plantvillage:plantvillage_map.json \\
            --output data --max-per-class 2000
    """
    import argparse
    import hashlib
    import json
    import shutil
    import sys
    from collections import defaultdict
    from pathlib import Path as _Path

    # Importa CLASS_NAMES aqui para não forçar import de app/ ao usar como módulo
    sys.path.insert(0, str(_Path(__file__).parent))
    from app.config import CLASS_NAMES  # noqa: PLC0415

    _IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def _collect(source_dir: _Path, folder_map: dict | None) -> list[tuple[_Path, str]]:
        mapping = folder_map if folder_map else {c: c for c in CLASS_NAMES}
        samples: list[tuple[_Path, str]] = []
        for fname, cls in mapping.items():
            if cls not in CLASS_NAMES:
                continue
            folder = source_dir / fname
            if not folder.is_dir():
                continue
            for f in folder.iterdir():
                if f.suffix.lower() in _IMAGE_EXT:
                    samples.append((f, cls))
        return samples

    def _unique_name(src: _Path, tag: str, i: int) -> str:
        h = hashlib.md5(str(src).encode()).hexdigest()[:6]
        return f"{tag}_{i:06d}_{h}{src.suffix.lower()}"

    def _print_dist(title: str, dist: dict[str, int]):
        total = sum(dist.values())
        print(f"\n{title}")
        print(f"{'Classe':<25} {'Imagens':>8}  {'%':>7}")
        print("-" * 44)
        for cls in CLASS_NAMES:
            n = dist.get(cls, 0)
            pct = (n / total * 100) if total else 0
            flag = "  << INSUFICIENTE" if n < 300 else ""
            print(f"  {cls:<23} {n:>8}  {pct:>6.1f}%{flag}")
        print("-" * 44)
        print(f"  {'TOTAL':<23} {total:>8}")

    def _recommendations(dist: dict[str, int]):
        print("\n=== Recomendações ===")
        for cls in CLASS_NAMES:
            n = dist.get(cls, 0)
            if n == 0:
                print(f"  {cls:<25}: VAZIO — necessário recolher imagens")
            elif n < 300:
                print(f"  {cls:<25}: {n} imgs — recomendado ≥ 500")
            elif n < 800:
                print(f"  {cls:<25}: {n} imgs — aceitável, mais é melhor")
            else:
                print(f"  {cls:<25}: {n} imgs — OK")
        print("\nPróximo passo: python train.py --data ./data --model all")

    parser = argparse.ArgumentParser(description="Preparação de dados FitoVision (multi-classe)")
    parser.add_argument("--source", action="append", dest="sources", required=True,
                        metavar="PATH[:MAP.JSON]")
    parser.add_argument("--output", default="./data")
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    rng = random.Random(args.seed)

    sources: list[tuple[_Path, dict | None, str]] = []
    for s in args.sources:
        if ":" in s:
            parts = s.split(":", 1)
            src_dir  = _Path(parts[0])
            map_path = _Path(parts[1])
            if not map_path.exists():
                print(f"[ERRO] Mapeamento não encontrado: {map_path}")
                sys.exit(1)
            with open(map_path) as f:
                raw_map = json.load(f)
            folder_map = {k: v for k, v in raw_map.items() if not k.startswith("_")}
            tag = src_dir.name
        else:
            src_dir    = _Path(s)
            folder_map = None
            tag        = src_dir.name
        if not src_dir.is_dir():
            print(f"[ERRO] Diretório não encontrado: {src_dir}")
            sys.exit(1)
        sources.append((src_dir, folder_map, tag))

    all_by_class: dict[str, list] = defaultdict(list)
    for src_dir, folder_map, tag in sources:
        samples = _collect(src_dir, folder_map)
        dist: dict[str, int] = defaultdict(int)
        for path, cls in samples:
            all_by_class[cls].append((path, tag))
            dist[cls] += 1
        print(f"\nFonte '{tag}': {sum(dist.values())} imgs de {src_dir}")
        for cls in CLASS_NAMES:
            if dist.get(cls, 0):
                print(f"  {cls:<25}: {dist[cls]}")

    raw_dist = {cls: len(imgs) for cls, imgs in all_by_class.items()}
    _print_dist("=== Distribuição bruta ===", raw_dist)

    selected: dict[str, list] = {}
    for cls in CLASS_NAMES:
        imgs = all_by_class.get(cls, [])
        rng.shuffle(imgs)
        selected[cls] = imgs[: args.max_per_class] if args.max_per_class else imgs

    final_dist = {cls: len(imgs) for cls, imgs in selected.items()}
    if args.max_per_class:
        _print_dist(f"=== Distribuição final (máx {args.max_per_class}/classe) ===", final_dist)

    if args.dry_run:
        print("\n[DRY RUN] Nenhum ficheiro copiado.")
        _recommendations(final_dist)
        return

    output = _Path(args.output)
    for cls in CLASS_NAMES:
        (output / cls).mkdir(parents=True, exist_ok=True)

    copied = skipped = 0
    for cls, imgs in selected.items():
        for i, (src_path, tag) in enumerate(imgs):
            dest = output / cls / _unique_name(src_path, tag, i)
            if dest.exists():
                skipped += 1
                continue
            shutil.copy2(src_path, dest)
            copied += 1

    print(f"\nConcluído: {copied} copiadas, {skipped} já existentes ignoradas.")
    print(f"Dataset em: {output.resolve()}")
    _recommendations(final_dist)


if __name__ == "__main__":
    _cli_consolidate()
