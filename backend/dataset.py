"""
Dataset utilities para o FitoVision.

━━━ Pipeline binário (novo) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usa a estrutura criada por download_datasets.py:
    data/
      train/healthy/    train/anomalous/
      val/healthy/      val/anomalous/
      test/healthy/     test/anomalous/

Classe principal:  BinaryFolderDataset
Helper de loaders: make_binary_folder_loaders()

━━━ Pipeline multi-classe (original) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Modo direto — pastas nomeadas como as classes:
    data/
      saudavel/   img1.jpg ...
      mildio/     img2.jpg ...

Modo mapeado — PlantVillage ou datasets com nomes de pasta diferentes:
    folder_class_map: dict[str, str]  pasta → classe_fitovision
    Ver plantvillage_map.json para exemplo.

Classe principal: PlantDataset
Helper de loaders: make_loaders()
"""

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

logger = logging.getLogger("fitovision.dataset")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─── Transforms ───────────────────────────────────────────────────────────────
#
# Duas correcções em relação à versão anterior:
#
# 1. VAL/TEST usava Resize((224, 224)), que ESMAGA a proporção da imagem. O treino
#    usava RandomResizedCrop, que recorta preservando a proporção. Ou seja: o modelo
#    treinava em folhas com geometria correcta e era avaliado em folhas achatadas.
#    Numa foto de telemóvel (4:3 ou 16:9) a distorção é severa. Passou a
#    Resize(256) + CenterCrop(224) — a convenção do ImageNet, e geometricamente
#    consistente com o crop do treino.
#
# 2. O ColorJitter tinha saturation=0.3 e hue=0.05. Clorose, míldio e oídio são
#    definidos precisamente por DESVIO DE COR (amarelecimento, manchas brancas).
#    Perturbar matiz e saturação com essa intensidade apaga o próprio sinal que o
#    modelo deve aprender. Reduzido para saturation=0.15 / hue=0.02; brilho e
#    contraste (que modelam variação de iluminação, não de doença) foram mantidos
#    generosos.

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.333)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    # Oclusão aleatória: força o modelo a não depender de uma única região
    # (ex.: o fundo cinzento uniforme do PlantVillage).
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _load_or_resample(
    samples: list[tuple[Path, int]],
    idx: int,
    log: logging.Logger,
    max_tries: int = 8,
):
    """
    Abre samples[idx]. Se a imagem estiver corrompida, devolve OUTRA imagem do
    MESMO label em vez de um quadrado preto.

    Porquê: a versão anterior substituía a imagem ilegível por Image.new("RGB", …, 0)
    — um quadrado preto — MANTENDO o label original. Com centenas de ficheiros
    corrompidos num dataset de 200k, isso ensina o modelo a associar "imagem preta"
    a um label real, e injecta ruído puro na loss. Reamostrar dentro da mesma classe
    preserva a distribuição de labels e não inventa nenhum padrão.
    """
    path, label = samples[idx]
    for attempt in range(max_tries):
        try:
            return Image.open(path).convert("RGB"), label
        except (UnidentifiedImageError, OSError) as exc:
            if attempt == 0:
                log.warning("Imagem ilegível: %s (%s) — reamostrando na mesma classe.", path, exc)
            # Procura outro índice com o mesmo label (varredura circular determinística)
            n = len(samples)
            for step in range(1, n):
                cand = (idx + step) % n
                if samples[cand][1] == label:
                    path, _ = samples[cand]
                    break
            else:
                raise RuntimeError(
                    f"Nenhuma imagem legível para o label {label}. Dataset inutilizável."
                ) from exc

    raise RuntimeError(f"{max_tries} imagens seguidas ilegíveis para o label {label}.")


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
        img, label = _load_or_resample(self.samples, idx, logger)
        if self.transform:
            img = self.transform(img)
        return img, label

    def class_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for _, label in self.samples:
            counts[self.class_names[label]] += 1
        return dict(counts)


def dataset_from_samples(
    class_names: list[str],
    samples: list[tuple[Path, int]],
    transform=None,
) -> "PlantDataset":
    """Cria um PlantDataset a partir de uma lista de (path, label) já preparada.

    Usado pelo evaluate.py para reconstruir o test split guardado pelo train.py.
    """
    ds = PlantDataset.__new__(PlantDataset)
    ds.root = Path(".")
    ds.class_names = class_names
    ds.class_to_idx = {c: i for i, c in enumerate(class_names)}
    ds.transform = transform
    ds.samples = samples
    return ds


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
    """Retorna (train_loader, val_loader, test_loader, test_samples).

    test_samples é uma lista de (path, label) guardada pelo train.py
    para que o evaluate.py use exactamente o mesmo conjunto de teste.
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


# ══════════════════════════════════════════════════════════════════════════════
# ─── Pipeline binário — BinaryFolderDataset ──────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

BINARY_CLASSES = ["healthy", "anomalous"]  # label: 0=healthy, 1=anomalous


class BinaryFolderDataset(Dataset):
    """
    Carrega imagens de uma pasta de split pré-dividida:
        split_dir/healthy/
        split_dir/anomalous/

    label: 0 = healthy, 1 = anomalous

    Compatível com torchvision transforms (padrão) ou albumentations.
    Para usar albumentations, passe use_albumentations=True — nesse caso
    o transform deve ser um Compose do albumentations que espera dict
    {'image': numpy_array} e retorna {'image': tensor}.
    """

    def __init__(
        self,
        split_dir: str | Path,
        transform=None,
        use_albumentations: bool = False,
    ):
        self.split_dir          = Path(split_dir)
        self.transform          = transform
        self.use_albumentations = use_albumentations
        self.class_names        = BINARY_CLASSES
        self.class_to_idx       = {c: i for i, c in enumerate(BINARY_CLASSES)}
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
        img, label = _load_or_resample(self.samples, idx, logger)

        if self.transform is None:
            # Fallback mínimo: mesma geometria do VAL_TRANSFORMS (resize + center crop),
            # nunca um resize deformante.
            tensor = VAL_TRANSFORMS(img)
            return tensor, label

        if self.use_albumentations:
            import numpy as np
            img_np    = np.array(img)
            augmented = self.transform(image=img_np)
            return augmented["image"], label

        return self.transform(img), label

    def class_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for _, label in self.samples:
            counts[BINARY_CLASSES[label]] += 1
        return dict(counts)


def binary_weighted_sampler(dataset: "BinaryFolderDataset") -> WeightedRandomSampler:
    """WeightedRandomSampler que equaliza a probabilidade de amostragem healthy/anomalous."""
    dist  = dataset.class_distribution()
    total = sum(dist.values())
    present = [c for c in BINARY_CLASSES if dist.get(c, 0) > 0]
    w = {c: total / (len(present) * dist[c]) for c in present}
    weights = [w[BINARY_CLASSES[lbl]] for _, lbl in dataset.samples]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def make_binary_folder_loaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    use_albumentations: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Cria train/val/test DataLoaders para o pipeline binário a partir de:
        data_dir/train/healthy/  data_dir/train/anomalous/
        data_dir/val/...         data_dir/test/...

    O loader de treino usa WeightedRandomSampler para equalizar as classes.

    Retorna: (train_loader, val_loader, test_loader, class_weights)
             class_weights é um tensor [w_healthy, w_anomalous] para a CrossEntropyLoss.
    """
    if use_albumentations:
        # Delega para prepare_data que tem os transforms albumentations completos
        from prepare_data import make_binary_loaders
        loaders = make_binary_loaders(data_dir, batch_size=batch_size, num_workers=num_workers)
        return (*loaders, torch.tensor([1.0, 1.0]))

    data_dir = Path(data_dir)
    loaders  = []
    class_weights = torch.tensor([1.0, 1.0])

    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"'{split_dir}' não encontrada. Execute download_datasets.py primeiro."
            )
        transform = TRAIN_TRANSFORMS if split == "train" else VAL_TRANSFORMS
        ds        = BinaryFolderDataset(split_dir, transform=transform, use_albumentations=False)
        pin       = torch.cuda.is_available()
        dist      = ds.class_distribution()

        if split == "train":
            # ─── Balanceamento de classes ────────────────────────────────────
            # O pipeline binário anterior usava shuffle=True puro. Como o pool de
            # 'anomalous' (11 pastas de doença do PlantVillage + datasets de doença)
            # é muito maior que o de 'healthy' (4 pastas), o modelo aprendia o
            # atalho de prever 'anomalous' quase sempre — o que bate certo com
            # fotos aleatórias a serem classificadas como doentes.
            #
            # WeightedRandomSampler equaliza a probabilidade de amostragem por
            # classe; class_weights faz o mesmo na loss (cinto e suspensórios).
            sampler = binary_weighted_sampler(ds)
            loader  = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                                 num_workers=num_workers, pin_memory=pin,
                                 drop_last=True)

            n_h = dist.get("healthy", 0)
            n_a = dist.get("anomalous", 0)
            total = n_h + n_a
            if n_h > 0 and n_a > 0:
                class_weights = torch.tensor([total / (2 * n_h), total / (2 * n_a)],
                                             dtype=torch.float)
            ratio = (max(n_h, n_a) / min(n_h, n_a)) if n_h and n_a else float("inf")
            print(f"  {split:<6}: {len(ds):>6} imgs  {dist}  "
                  f"(ratio {ratio:.2f}:1 → sampler balanceado, "
                  f"pesos da loss = [{class_weights[0]:.2f}, {class_weights[1]:.2f}])")
        else:
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin)
            print(f"  {split:<6}: {len(ds):>6} imgs  {dist}")

        loaders.append(loader)

    return (*loaders, class_weights)  # type: ignore[return-value]
