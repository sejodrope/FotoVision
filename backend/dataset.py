"""
Dataset utilities para o FitoVision.

Modo direto (padrão) — pastas nomeadas como as classes:
    data/
      saudavel/   img1.jpg ...
      mildio/     img2.jpg ...
      ...

Modo mapeado — para PlantVillage ou outro dataset com nomes de pasta diferentes:
    Passar folder_class_map: dict[str, str] mapeando nome_pasta -> classe_fitovision
    Ver plantvillage_map.json para exemplo.
"""

import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class PlantDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        class_names: list[str],
        transform=None,
        folder_class_map: Optional[dict[str, str]] = None,
    ):
        self.root = Path(root)
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self._scan(folder_class_map)

    def _scan(self, folder_class_map: Optional[dict[str, str]]):
        mapping = folder_class_map if folder_class_map else {c: c for c in self.class_names}
        for folder_name, class_name in mapping.items():
            if class_name not in self.class_to_idx:
                continue
            folder = self.root / folder_name
            if not folder.is_dir():
                continue
            idx = self.class_to_idx[class_name]
            for f in folder.iterdir():
                if f.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((f, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def class_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for _, label in self.samples:
            counts[self.class_names[label]] += 1
        return dict(counts)


def _stratified_split(
    samples: list,
    val_frac: float,
    test_frac: float,
    seed: int = 42,
) -> tuple[list, list, list]:
    by_class: dict[int, list] = defaultdict(list)
    for item in samples:
        by_class[item[1]].append(item)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for items in by_class.values():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))
        test.extend(items[:n_test])
        val.extend(items[n_test:n_test + n_val])
        train.extend(items[n_test + n_val:])
    return train, val, test


def _clone_with(base: PlantDataset, samples: list, transform) -> PlantDataset:
    ds = PlantDataset.__new__(PlantDataset)
    ds.root = base.root
    ds.class_names = base.class_names
    ds.class_to_idx = base.class_to_idx
    ds.transform = transform
    ds.samples = samples
    return ds


def weighted_sampler(dataset: PlantDataset) -> WeightedRandomSampler:
    dist = dataset.class_distribution()
    total = sum(dist.values())
    n_cls = len([v for v in dist.values() if v > 0])
    w = {c: total / (n_cls * n) if n > 0 else 0.0 for c, n in dist.items()}
    weights = [w[dataset.class_names[lbl]] for _, lbl in dataset.samples]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def make_loaders(
    data_dir: str | Path,
    class_names: list[str],
    batch_size: int = 32,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    num_workers: int = 0,
    folder_class_map: Optional[dict] = None,
    seed: int = 42,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, list]:
    """
    Retorna (train_loader, val_loader, test_loader, test_samples).
    test_samples é uma lista de (path, label) guardada pelo train.py para o evaluate.py.
    """
    full = PlantDataset(data_dir, class_names, transform=None, folder_class_map=folder_class_map)
    print(f"  Total de imagens : {len(full)}")
    print(f"  Distribuição     : {full.class_distribution()}")

    train_s, val_s, test_s = _stratified_split(full.samples, val_frac, test_frac, seed)
    print(f"  Splits           : treino={len(train_s)} | val={len(val_s)} | teste={len(test_s)}")

    train_ds = _clone_with(full, train_s, TRAIN_TRANSFORMS)
    val_ds   = _clone_with(full, val_s,   VAL_TRANSFORMS)
    test_ds  = _clone_with(full, test_s,  VAL_TRANSFORMS)

    pin = pin_memory and torch.cuda.is_available()
    sampler = weighted_sampler(train_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader, test_s
