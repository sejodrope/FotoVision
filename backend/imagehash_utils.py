#!/usr/bin/env python
"""
Hashing perceptual e agrupamento de imagens near-duplicate.

Motivação
─────────
O dataset do FitoVision agrega fontes que contêm cópias transformadas da mesma
foto de origem (ex.: 'lettuce-disease-multi-transformation-dataset' — rotações,
flips e ajustes de brilho da mesma folha) e re-uploads do PlantVillage sob nomes
de ficheiro diferentes.

Um split aleatório ao nível do ficheiro coloca a rotação de uma foto no treino e
o flip da MESMA foto no teste. O modelo memoriza a foto e a accuracy de teste
sobe artificialmente — sem qualquer capacidade de generalização.

A solução é agrupar as imagens por identidade visual e fazer o split ao nível do
GRUPO, garantindo que todas as variantes de uma mesma foto caem no mesmo split.

Estratégia (duas camadas)
─────────────────────────
1. dHash (difference hash, 64 bits) — invariante a reescala e a mudanças suaves
   de compressão/brilho. Captura re-uploads e reescalas exactas.
2. dHash da imagem canonicalizada (menor forma entre as 8 simetrias do quadrado
   d4: rotações de 90° e flips) — captura as variantes geradas por augmentation
   geométrica, que o dHash simples não apanha.

Duas imagens com o mesmo hash canónico entram no mesmo grupo. O agrupamento usa
union-find para propagar transitivamente (A~B, B~C ⇒ A~B~C).

Sem dependências externas além de Pillow + numpy.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# Tamanho da grelha do dHash: (HASH_SIZE+1) x HASH_SIZE pixels → HASH_SIZE^2 bits
HASH_SIZE = 8

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _dhash_from_gray(gray: np.ndarray) -> int:
    """dHash de uma matriz cinzenta já redimensionada para (HASH_SIZE, HASH_SIZE+1)."""
    diff = gray[:, 1:] > gray[:, :-1]
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bit)
    return bits


def _normalize(image: Image.Image) -> Image.Image:
    """
    Converte para cinzento e normaliza o contraste antes de hashear.

    O dHash compara píxeis adjacentes, por isso já é invariante a qualquer
    transformação MONÓTONA de brilho... excepto quando há SATURAÇÃO: ao multiplicar
    o brilho, os píxeis que passam de 255 são todos cortados para 255, empatam
    entre si, e as comparações que os envolvem invertem-se. Foi assim que as cópias
    com jitter de brilho escapavam ao agrupamento e o vazamento sobrevivia.

    autocontrast() reescala a imagem para a gama completa, desfazendo a maior parte
    das diferenças de exposição entre variantes da mesma foto.
    """
    return ImageOps.autocontrast(image.convert("L"), cutoff=1)


def dhash(image: Image.Image) -> int:
    """dHash 64-bit de uma imagem PIL."""
    gray = _normalize(image).resize((HASH_SIZE + 1, HASH_SIZE), Image.Resampling.LANCZOS)
    return _dhash_from_gray(np.asarray(gray, dtype=np.int16))


def orbit_hashes(image: Image.Image) -> tuple[int, ...]:
    """
    Os 8 dHashes da imagem sob as simetrias do quadrado (grupo diedral D4):
    rotações de 0°/90°/180°/270°, cada uma com e sem espelhamento.

    ─── Por que a órbita inteira, e não um "hash canónico" ──────────────────
    A tentação óbvia é colapsar a órbita num único valor com min() e comparar
    esses valores. Como D4 é um grupo, a órbita de uma imagem e a da sua rotação
    são o MESMO conjunto, logo o mínimo coincide — e para cópias geométricas
    exactas funciona.

    Mas min() é um selector DESCONTÍNUO. Quando duas orientações têm valores de
    hash próximos, basta um bit perturbado (um jitter de brilho, uma recompressão
    JPEG) para mudar QUAL orientação atinge o mínimo — e o "canónico" salta para um
    padrão de bits completamente diferente. Medido: variantes com jitter de brilho
    davam distância de Hamming ~39 em 64 (praticamente aleatória) em relação à foto
    de origem, e o agrupamento fragmentava-se. Nenhuma tolerância de distância
    resolve isso, porque o valor não está perto — está noutro sítio.

    Guardar a órbita inteira elimina a descontinuidade: duas imagens são
    consideradas variantes se ALGUMA orientação de uma estiver perto de ALGUMA
    orientação da outra. Cada elemento da órbita move-se pouco sob perturbação,
    e a comparação deixa de depender de uma escolha frágil.
    """
    # Quadrado (para as rotações de 90° serem bem definidas) e contraste
    # normalizado (para absorver diferenças de exposição).
    sq = _normalize(image).resize((32, 32), Image.Resampling.LANCZOS)
    arr = np.asarray(sq, dtype=np.int16)

    hashes: list[int] = []
    for k in range(4):
        rot = np.rot90(arr, k)
        for v in (rot, np.fliplr(rot)):
            small = np.asarray(
                Image.fromarray(v.astype(np.uint8)).resize(
                    (HASH_SIZE + 1, HASH_SIZE), Image.Resampling.LANCZOS
                ),
                dtype=np.int16,
            )
            hashes.append(_dhash_from_gray(small))

    return tuple(hashes)


def orbit_signature(orbit: tuple[int, ...]) -> tuple[int, ...]:
    """
    Chave estável de duplicado exacto: a órbita ORDENADA.

    Duas imagens relacionadas por uma simetria D4 exacta têm a mesma órbita
    enquanto CONJUNTO — logo a mesma assinatura ordenada. Ao contrário do min(),
    isto usa toda a órbita e não depende de nenhuma escolha frágil.
    """
    return tuple(sorted(orbit))


def orbit_distance(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    """Menor distância de Hamming entre qualquer orientação de `a` e qualquer de `b`."""
    best = 64
    for ha in a:
        for hb in b:
            d = hamming(ha, hb)
            if d < best:
                best = d
                if best == 0:
                    return 0
    return best


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class UnionFind:
    def __init__(self):
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# Cache persistente das órbitas em JSONL (append-only): hashear ~200k imagens
# demora horas e o processo pode ser interrompido (sleep, fecho de sessão).
# Chave = caminho|tamanho|mtime — muda se o ficheiro mudar. Valor null = corrompida.
_HASH_CACHE_PATH = Path(__file__).parent / "data" / "hash_cache.jsonl"


def _cache_key(p: Path) -> str | None:
    try:
        st = p.stat()
    except OSError:
        return None
    return f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}"


def _load_hash_cache(cache_path: Path) -> dict[str, tuple[int, ...] | None]:
    cache: dict[str, tuple[int, ...] | None] = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                try:
                    key, orbit = json.loads(line)
                except (ValueError, TypeError):
                    continue  # linha truncada por interrupção — recalcula-se
                cache[key] = tuple(orbit) if orbit is not None else None
    return cache


def hash_images(
    paths: list[Path],
    progress: bool = True,
    cache_path: Path = _HASH_CACHE_PATH,
) -> tuple[dict[Path, tuple[int, ...]], list[Path]]:
    """
    Calcula a órbita D4 de hashes de cada imagem.

    Devolve (orbits, corrupted) — onde `corrupted` lista as imagens que não abriram.
    Estas devem ser EXCLUÍDAS do dataset, nunca substituídas por um quadrado preto
    com o label original (isso ensina o modelo a associar "imagem preta" a um
    label real).
    """
    orbits: dict[Path, tuple[int, ...]] = {}
    corrupted: list[Path] = []

    cache = _load_hash_cache(cache_path)
    if cache:
        print(f"    cache de hashes: {len(cache)} entradas em {cache_path.name}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_hits = 0

    total = len(paths)
    with open(cache_path, "a", encoding="utf-8") as cache_f:
        for i, p in enumerate(paths):
            if progress and total > 500 and i % 2000 == 0:
                pct = 100.0 * i / total if total else 0.0
                print(f"    hashing {i:>7}/{total} ({pct:4.1f}%)", flush=True)

            key = _cache_key(p)
            if key is not None and key in cache:
                cached = cache[key]
                if cached is None:
                    corrupted.append(p)
                else:
                    orbits[p] = cached
                cache_hits += 1
                continue

            try:
                with Image.open(p) as img:
                    orbit = orbit_hashes(img.convert("RGB"))
                orbits[p] = orbit
            except (UnidentifiedImageError, OSError, ValueError):
                orbit = None
                corrupted.append(p)

            if key is not None:
                cache_f.write(json.dumps([key, list(orbit) if orbit else None]) + "\n")
                if i % 500 == 0:
                    cache_f.flush()

    if progress and cache_hits:
        print(f"    {cache_hits}/{total} órbitas vindas do cache", flush=True)
    return orbits, corrupted


def _bands(value: int, n_bands: int, n_bits: int = 64) -> list[tuple[int, int]]:
    """Parte um hash em n_bands fatias. Devolve [(índice_da_banda, valor_da_banda), ...]."""
    out: list[tuple[int, int]] = []
    base, extra = divmod(n_bits, n_bands)
    shift = 0
    for b in range(n_bands):
        width = base + (1 if b < extra else 0)
        out.append((b, (value >> shift) & ((1 << width) - 1)))
        shift += width
    return out


def group_by_similarity(
    orbits: dict[Path, tuple[int, ...]],
    max_distance: int = 4,
) -> dict[Path, int]:
    """
    Agrupa imagens near-duplicate. Devolve {path: group_id}.

    Une duas imagens quando:
      • têm a mesma assinatura de órbita (duplicado exacto ou simetria D4 limpa), OU
      • alguma orientação de uma está a distância de Hamming <= max_distance de
        alguma orientação da outra (rotação arbitrária, jitter de brilho,
        recompressão, ruído).

    ─── Indexação por bandas (LSH) ───────────────────────────────────────────
    Comparar todos os pares é O(n²) — inviável com 200k imagens. Uma versão
    anterior deste código indexava por um prefixo de 32 bits, o que PERDIA pares:
    dois hashes a distância 2 que diferissem num bit alto caíam em prefixos
    diferentes e nunca eram comparados. O agrupamento fragmentava-se e o vazamento
    sobrevivia ao "fix".

    A versão correcta usa o princípio das gavetas: partindo o hash em
    (max_distance + 1) bandas, dois hashes a distância <= max_distance têm de
    coincidir EXACTAMENTE em pelo menos uma banda (os <= max_distance bits que
    diferem só conseguem corromper <= max_distance bandas, sobrando uma intacta).
    Indexando por cada banda e comparando só dentro de cada bucket, nenhum par a
    distância <= max_distance escapa, e o custo fica praticamente linear.

    Cada imagem é indexada sob TODAS as suas 8 orientações, para que uma variante
    rodada colida com a original em alguma banda.
    """
    uf = UnionFind()
    paths = list(orbits.keys())
    idx_of = {p: i for i, p in enumerate(paths)}
    for p in paths:
        uf.find(idx_of[p])

    # ── 1. União por assinatura de órbita idêntica ────────────────────────────
    # Apanha duplicados exactos e variantes geométricas D4 limpas — a esmagadora
    # maioria (re-uploads, rotações de 90°, flips).
    by_sig: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for p in paths:
        by_sig[orbit_signature(orbits[p])].append(idx_of[p])
    for members in by_sig.values():
        for m in members[1:]:
            uf.union(members[0], m)

    # ── 2. União por proximidade, via LSH sobre as órbitas ────────────────────
    # Só um representante por assinatura distinta entra no índice: as restantes já
    # foram unidas no passo 1, e isto mantém o índice pequeno.
    if max_distance > 0:
        n_bands = max_distance + 1        # garante recall total até max_distance
        reps = [members[0] for members in by_sig.values()]

        band_index: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for rep in reps:
            for h in set(orbits[paths[rep]]):          # as 8 orientações
                for b_idx, b_val in _bands(h, n_bands):
                    band_index[(b_idx, b_val, 0)].append(rep)

        for bucket in band_index.values():
            if len(bucket) < 2:
                continue
            # Buckets patológicos (ex.: imagens quase uniformes partilham bandas com
            # milhares de outras) tornariam isto quadrático. Limita-os.
            if len(bucket) > 300:
                continue
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    a, b = bucket[i], bucket[j]
                    if uf.find(a) == uf.find(b):
                        continue
                    if orbit_distance(orbits[paths[a]], orbits[paths[b]]) <= max_distance:
                        uf.union(a, b)

    # Normaliza os ids de grupo para 0..n-1
    root_to_gid: dict[int, int] = {}
    groups: dict[Path, int] = {}
    for p in paths:
        root = uf.find(idx_of[p])
        if root not in root_to_gid:
            root_to_gid[root] = len(root_to_gid)
        groups[p] = root_to_gid[root]

    return groups


def save_groups(groups: dict[Path, int], out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({str(k): v for k, v in groups.items()}, f)


def load_groups(path: Path) -> dict[Path, int]:
    with open(path, "rb") as f:
        return {Path(k): v for k, v in pickle.load(f).items()}


def group_stats(groups: dict[Path, int]) -> dict:
    sizes: dict[int, int] = defaultdict(int)
    for gid in groups.values():
        sizes[gid] += 1
    counts = list(sizes.values())
    n_dup_groups = sum(1 for c in counts if c > 1)
    n_redundant = sum(c - 1 for c in counts if c > 1)
    return {
        "n_images": len(groups),
        "n_groups": len(sizes),
        "n_duplicate_groups": n_dup_groups,
        "n_redundant_images": n_redundant,
        "largest_group": max(counts) if counts else 0,
        "redundancy_pct": (100.0 * n_redundant / len(groups)) if groups else 0.0,
    }
