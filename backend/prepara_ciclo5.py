#!/usr/bin/env python
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Ciclo 5 — refaz o split com as correções da auditoria de 13/10/2026
(docs/CORRECOES_METODOLOGICAS.md §14.8 e §15).

Não mexe em data/ nem nos pesos do ciclo 4: escreve em data_c5/ com hardlinks, que não
ocupam espaço. O que muda em relação ao ciclo 4:

  1. Alface agrupada por PROVENIÊNCIA: todas as cópias e variantes de uma mesma foto
     original ficam do mesmo lado. O hash perceptual não liga uma rotação de ângulo
     livre, um recorte ou um filtro de bordas à foto de onde vieram; o id da foto, que
     a fonte de variantes guarda no nome do arquivo, liga.
  2. Validação e teste de alface só com fotografias, uma por foto original.
  3. Saem as variantes que não são fotografias (sobel, laplacian, high_pass) e as
     cópias idênticas da pasta dataset/ da mesma fonte.
  4. Imagens de alface sem foto original identificada ficam só no treino.
  5. As pastas que a rotulagem automática não reconheceu (dano por praga e uma virose)
     entram como anômalas.
  6. Um dos dois lotes de espinafre fica inteiro fora do treino: é a fonte retida.

As demais imagens (coletânea e PlantVillage) mantêm a partição do ciclo 4, para que
esse subconjunto do teste continue comparável foto a foto entre os ciclos.

Uso:
    python prepara_ciclo5.py            # só mostra as contagens
    python prepara_ciclo5.py --apply    # cria data_c5/
Saída:
    data_c5/{train,val,test,retida}/{healthy,anomalous}/
    data_c5/split_metadata.json
    data_c5/manifesto_ciclo5.csv        (uma linha por imagem: de onde veio e que tipo é)
"""

import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from dataset import cultura_do_arquivo
from imagehash_utils import group_by_similarity, hash_images, orbit_distance

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RAW = DATA / "raw"
OUT = BASE / "data_c5"
CACHE_C5 = BASE / "logs" / "hash_cache_c5.jsonl"
SEED = 42
SPLITS = ("train", "val", "test")
CLASSES = ("healthy", "anomalous")
FRACOES = (0.70, 0.15)                       # treino, validação; o resto é teste

VARIANTES = "shuvokumarbasak2030_lettuce-disease-multi-transformation-dataset__"
OUTRAS_ALFACE = ("ashishjstar", "iqrapervez", "santoshshaha")     # ordem = preferência no teste
FONTE_RETIDA = "mendeley_spinach_n56pn9fncw"
CULTURAS_DE_CAMPO = ("Cassava", "Rice", "Sugercane", "Coffee")
NAO_FOTOGRAFIAS = {"sobel", "laplacian", "high_pass"}
# Transformações que o hash perceptual ainda reconhece como a mesma foto: servem de
# ponte entre a fonte de variantes e as outras três fontes de alface.
SUAVES = {"flipped_horizontal", "flipped_vertical", "brightness_adjusted", "contrast_adjusted",
          "saturation_adjusted", "unsharp_mask", "gaussian_noise", "poisson_noise",
          "hist_equalized", "jittered"}
# (fonte, pasta em data/raw/<fonte>/, só no treino?)  — todas `anomalous`.
# A de alface não tem como rastrear a foto original: fica só no treino, como as demais
# imagens de alface sem origem identificada.
PASTAS_NAO_RECONHECIDAS = [
    ("iqrapervez2000_lettuce-disease-dataset", "Diseases/Diseases/Lettuce_disease_datasets/Pest damage", True),
    ("nirmalsankalana_plant-diseases-training-dataset", "data/Potato___pests", False),
    ("nirmalsankalana_plant-diseases-training-dataset", "data/Rose___slug_sawfly", False),
    ("nirmalsankalana_plant-diseases-training-dataset", "data/Tomato___spider_mites", False),
    ("nirmalsankalana_plant-diseases-training-dataset", "data/Sugercane___yellow_leaf", False),
]
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_BITS = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x)
    return _BITS[x.view(np.uint8).reshape(x.shape + (8,))].sum(axis=-1)

REGRAS = [
    "alface agrupada por foto original (id no nome do arquivo + ligação por hash)",
    "validação e teste de alface: uma fotografia por foto original, sem variantes",
    "fora as variantes sobel, laplacian e high_pass e as cópias idênticas de dataset/",
    "alface sem foto original identificada: só no treino",
    "pastas não reconhecidas pela rotulagem entram como anomalous",
    f"fonte retida: {FONTE_RETIDA} (fora de treino e validação)",
    "coletânea e PlantVillage mantêm a partição do ciclo 4",
]

RX_AUG = re.compile(r"^augmentation_augmentation_(?P<cls>.+?)_(?P<tr>[a-z_]+?)_(?P<id>\d+)_(?P=tr)\.\w+$")
RX_DS = re.compile(r"^dataset_dataset_(?:train|test|val)_(?P<cls>.+?)_(?P<id>\d+)_(?P<tr>[a-z_]+)\.\w+$")


def ler_split_atual() -> dict[str, tuple[str, str, Path]]:
    """{nome: (split, classe, caminho)} do ciclo 4, lido do disco."""
    out = {}
    for split in SPLITS:
        for cls in CLASSES:
            with os.scandir(DATA / split / cls) as it:
                for e in it:
                    if e.is_file():
                        out[e.name] = (split, cls, Path(e.path))
    return out


def orbitas_do_cache(nomes: set[str]) -> dict[str, tuple[int, ...]]:
    """Órbitas D4 já calculadas (data/hash_cache.jsonl), indexadas pelo nome do arquivo."""
    out = {}
    with (DATA / "hash_cache.jsonl").open(encoding="utf-8") as fh:
        for linha in fh:
            try:
                chave, orbita = json.loads(linha)
            except ValueError:
                continue
            caminho = chave.split("|")[0].replace("\\", "/")
            nome = caminho.rsplit("/", 1)[-1]
            if orbita and len(orbita) == 8 and nome in nomes and "/staging/" not in caminho:
                out[nome] = tuple(orbita)
    return out


def repartir(chaves: list, rng: random.Random) -> dict:
    """70/15/15 sobre uma lista de unidades (fotos originais ou grupos)."""
    chaves = sorted(chaves, key=str)
    rng.shuffle(chaves)
    n_tr = round(FRACOES[0] * len(chaves))
    n_va = round(FRACOES[1] * len(chaves))
    return {k: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test") for i, k in enumerate(chaves)}


# ── Alface: uma unidade por foto original ─────────────────────────────────────

def reparticao_da_alface(atual: dict, rng: random.Random) -> tuple[list[dict], dict]:
    variantes, outras = {}, []
    for nome in atual:
        if nome.startswith(VARIANTES):
            m = RX_AUG.match(nome[len(VARIANTES):]) or RX_DS.match(nome[len(VARIANTES):])
            if m:
                variantes[nome] = ((m["cls"], int(m["id"])), m["tr"], nome[len(VARIANTES):].startswith("augmentation"))
        elif nome.startswith(OUTRAS_ALFACE):
            outras.append(nome)
    fotos = sorted({k for k, _, _ in variantes.values()})

    # Liga as outras fontes às fotos originais com o agrupamento por hash do projeto,
    # usando como ponte as variantes que o hash ainda reconhece como a mesma foto.
    # Duas fotos originais que caiam no mesmo grupo passam a ser uma unidade só.
    suaves = {n for n, (_, tr, aug) in variantes.items() if aug and tr in SUAVES}
    orbitas = orbitas_do_cache(suaves | set(outras))
    grupos = group_by_similarity({n: o for n, o in orbitas.items()}, max_distance=4)
    por_grupo = defaultdict(list)
    for nome, g in grupos.items():
        por_grupo[g].append(nome)

    pai = {k: k for k in fotos}

    def raiz(k):
        while pai[k] != k:
            pai[k] = pai[pai[k]]
            k = pai[k]
        return k

    original_de = {}
    for membros_g in por_grupo.values():
        chaves = sorted({variantes[n][0] for n in membros_g if n in variantes})
        for k in chaves[1:]:
            pai[raiz(k)] = raiz(chaves[0])
        if chaves:
            for n in membros_g:
                if n not in variantes:
                    original_de[n] = chaves[0]

    unidades = defaultdict(lambda: {"fotos": [], "variantes": [], "outras": []})
    for k in fotos:
        unidades[raiz(k)]["fotos"].append(k)
    for nome, (k, tr, aug) in variantes.items():
        unidades[raiz(k)]["variantes"].append((nome, tr, aug))
    for nome, k in original_de.items():
        unidades[raiz(k)]["outras"].append(nome)

    stats = Counter()
    rotulo = {}
    for u, m in unidades.items():
        rotulos = {"healthy" if k[0] == "Healthy" else "anomalous" for k in m["fotos"]}
        if len(rotulos) == 1:
            rotulo[u] = rotulos.pop()
        else:
            stats["unidades_com_rotulo_contraditorio_fora"] += 1
    destino = {}
    for cls in CLASSES:
        destino.update(repartir([u for u in rotulo if rotulo[u] == cls], rng))

    itens = []
    for u, split in destino.items():
        m, cls = unidades[u], rotulo[u]
        foto = f"{u[0]}#{u[1]}"
        fotograficas = sorted(n for n, tr, aug in m["variantes"] if aug and tr not in NAO_FOTOGRAFIAS)
        stats["variantes_nao_fotograficas_fora"] += sum(1 for _, tr, aug in m["variantes"] if aug and tr in NAO_FOTOGRAFIAS)
        stats["copias_identicas_fora"] += sum(1 for _, _, aug in m["variantes"] if not aug)
        copias = sorted(m["outras"], key=lambda n: (next(i for i, f in enumerate(OUTRAS_ALFACE) if n.startswith(f)), n))
        if split == "train":
            itens += [dict(nome=n, split=split, classe=cls, tipo="alface_variante", foto=foto) for n in fotograficas]
            itens += [dict(nome=n, split=split, classe=cls, tipo="alface_original", foto=foto) for n in copias]
            continue
        # Validação e teste: uma imagem por unidade. A própria foto, se alguma fonte a
        # tiver; senão, a variante menos alterada (o espelho).
        escolhida = copias[0] if copias else next(
            (n for n, tr, aug in sorted(m["variantes"]) if aug and tr == "flipped_horizontal"),
            fotograficas[0] if fotograficas else None)
        if escolhida is None:
            stats["unidades_sem_imagem_fotografica"] += 1
            continue
        itens.append(dict(nome=escolhida, split=split, classe=cls,
                          tipo="alface_original" if copias else "alface_variante", foto=foto))
        stats["imagens_de_alface_fora_de_val_teste"] += len(fotograficas) + len(copias) - 1

    # Sem foto original identificada: só no treino.
    for nome in outras:
        if nome not in original_de:
            itens.append(dict(nome=nome, split="train", classe=atual[nome][1], tipo="alface_sem_origem", foto=""))
            stats["alface_sem_origem_so_no_treino"] += 1
    stats["fotos_originais"] = len(fotos)
    stats["unidades"] = len(rotulo)
    stats["unidades_doentes"] = sum(1 for u in rotulo if rotulo[u] == "anomalous")
    for sp in SPLITS:
        stats[f"unidades_{sp}"] = sum(1 for u in destino if destino[u] == sp)
        stats[f"unidades_doentes_{sp}"] = sum(1 for u in destino if destino[u] == sp and rotulo[u] == "anomalous")
    return itens, dict(stats)


# ── Pastas que a rotulagem deixou de fora ─────────────────────────────────────

def pastas_nao_reconhecidas(atual: dict, rng: random.Random, calcular_hash: bool) -> tuple[list[dict], dict]:
    itens, stats = [], Counter()
    novos = {}                                            # nome → caminho, para as que serão repartidas por hash
    for fonte, pasta, so_treino in PASTAS_NAO_RECONHECIDAS:
        raiz = RAW / fonte / pasta
        arquivos = sorted(f for f in raiz.iterdir() if f.suffix.lower() in EXTENSOES)
        stats[f"{fonte.split('_')[0]}/{pasta.rsplit('/', 1)[-1]}"] = len(arquivos)
        for f in arquivos:
            nome = f"{fonte}__{pasta.replace('/', '_')}_{f.name}"
            if so_treino:
                itens.append(dict(nome=nome, split="train", classe="anomalous", tipo="praga_alface_sem_origem",
                                  foto="", origem=f))
            else:
                novos[nome] = f

    if not calcular_hash:
        for nome, f in novos.items():
            itens.append(dict(nome=nome, split="?", classe="anomalous", tipo="nao_reconhecida", foto="", origem=f))
        return itens, dict(stats)

    # Reparte por grupo de hash, junto com o que já existe da mesma fonte e do
    # PlantVillage: se uma imagem nova for quase-duplicata de uma que já está no split,
    # segue o lado dela (ou sai, se o rótulo for o contrário).
    orb_novas, corrompidas = hash_images(list(novos.values()), cache_path=CACHE_C5)
    stats["corrompidas"] = len(corrompidas)
    caminho_nome = {f: n for n, f in novos.items()}
    orb = {caminho_nome[f]: o for f, o in orb_novas.items()}
    nomes_novos = sorted(orb)
    N = np.array([orb[n] for n in nomes_novos], dtype=np.uint64)

    # O agrupador do projeto compara par a par em Python e demora dezenas de minutos com
    # milhares de folhas quase iguais sobre fundo branco. Aqui a distância é calculada
    # em bloco: um filtro largo com numpy (uma orientação contra as oito, limiar 8) e a
    # distância exacta entre órbitas só nos pares que passam.
    def proximas(alvo: np.ndarray, base: np.ndarray):
        """Para cada linha de `alvo`, os índices de `base` a distância <= 8 no filtro."""
        for a in range(len(alvo)):
            d = _popcount(base ^ alvo[a, 0]).min(axis=1)
            yield a, np.nonzero(d <= 8)[0]

    pai = list(range(len(nomes_novos)))

    def raiz(k):
        while pai[k] != k:
            pai[k] = pai[pai[k]]
            k = pai[k]
        return k

    for a, cand in proximas(N, N):
        for b in cand:
            if b > a and raiz(a) != raiz(b) and orbit_distance(orb[nomes_novos[a]], orb[nomes_novos[b]]) <= 4:
                pai[raiz(b)] = raiz(a)
    grupo_de = {n: raiz(k) for k, n in enumerate(nomes_novos)}

    # Quase-duplicatas das novas entre as que já estão no split.
    existentes = orbitas_do_cache({n for n in atual if n.startswith(("nirmal", "plantvillage"))})
    nomes_old = sorted(existentes)
    E = np.array([existentes[n] for n in nomes_old], dtype=np.uint64)
    vizinhas = defaultdict(set)                                   # grupo de novas → existentes quase iguais
    for a, cand in proximas(N, E):
        for b in cand:
            if orbit_distance(orb[nomes_novos[a]], existentes[nomes_old[b]]) <= 4:
                vizinhas[grupo_de[nomes_novos[a]]].add(nomes_old[b])

    por_grupo = defaultdict(list)
    for nome, g in grupo_de.items():
        por_grupo[g].append(nome)
    destino_livre = repartir([g for g in por_grupo if g not in vizinhas], rng)
    for g, nomes in por_grupo.items():
        if g in vizinhas:
            if any(atual[n][1] != "anomalous" for n in vizinhas[g]):
                stats["novas_com_rotulo_contraditorio_fora"] += len(nomes)
                continue
            lados = {atual[n][0] for n in vizinhas[g]}
            split = "train" if "train" in lados else sorted(lados)[0]
            stats["novas_que_seguem_duplicata_existente"] += len(nomes)
        else:
            split = destino_livre[g]
        for nome in nomes:
            itens.append(dict(nome=nome, split=split, classe="anomalous", tipo="nao_reconhecida", foto="",
                              origem=novos[nome]))
    return itens, dict(stats)


# ── Montagem ──────────────────────────────────────────────────────────────────

def montar(apply: bool) -> dict:
    rng = random.Random(SEED)
    atual = ler_split_atual()
    itens = []

    # O que mantém a partição do ciclo 4.
    for nome, (split, cls, _) in atual.items():
        if nome.startswith(VARIANTES) or nome.startswith(OUTRAS_ALFACE):
            continue
        if nome.startswith(FONTE_RETIDA):
            itens.append(dict(nome=nome, split="retida", classe=cls, tipo="espinafre_fonte_retida", foto=""))
        elif nome.startswith("mendeley_spinach"):
            itens.append(dict(nome=nome, split=split, classe=cls, tipo="espinafre", foto=""))
        elif nome.startswith("roboflow_lettuce_downy"):
            itens.append(dict(nome=nome, split=split, classe=cls, tipo="mildio_recorte", foto=""))
        elif nome.startswith("plantvillage"):
            itens.append(dict(nome=nome, split=split, classe=cls, tipo="plantvillage", foto=""))
        else:
            tipo = "coletanea_lavoura" if cultura_do_arquivo(nome) in CULTURAS_DE_CAMPO else "coletanea_demais"
            itens.append(dict(nome=nome, split=split, classe=cls, tipo=tipo, foto=""))

    alface, st_alface = reparticao_da_alface(atual, rng)
    novas, st_novas = pastas_nao_reconhecidas(atual, rng, calcular_hash=apply)
    itens += alface + novas

    for it in itens:
        it.setdefault("origem", atual[it["nome"]][2] if it["nome"] in atual else None)
        it["cultura"] = cultura_do_arquivo(it["nome"])
        it["fonte"] = it["nome"].split("__")[0]

    # Conferência: nenhuma foto original de alface em mais de um lado.
    lados = defaultdict(set)
    for it in itens:
        if it["foto"]:
            lados[it["foto"]].add(it["split"])
    assert all(len(v) == 1 for v in lados.values()), "foto original de alface em mais de um lado"
    assert len({it["nome"] for it in itens}) == len(itens), "nome repetido"

    contagem = {s: Counter() for s in (*SPLITS, "retida", "?")}
    por_tipo = {s: Counter() for s in contagem}
    for it in itens:
        contagem[it["split"]][it["classe"]] += 1
        por_tipo[it["split"]][it["tipo"]] += 1
    meta = {
        "ciclo": 5,
        "seed": SEED,
        "base": "partição do ciclo 4 (data/), mantida para a coletânea e o PlantVillage",
        "regras": REGRAS,
        "images_per_split": {s: dict(c) for s, c in contagem.items() if c},
        "tipos_por_split": {s: dict(c) for s, c in por_tipo.items() if c},
        "alface": st_alface,
        "pastas_nao_reconhecidas": st_novas,
        "fonte_retida": FONTE_RETIDA,
        "total": len(itens),
    }

    if apply:
        if OUT.exists():
            sys.exit(f"[ciclo5] {OUT} já existe. Apague a pasta (são só hardlinks) para refazer.")
        for s in (*SPLITS, "retida"):
            for cls in CLASSES:
                (OUT / s / cls).mkdir(parents=True)
        for it in itens:
            os.link(it["origem"], OUT / it["split"] / it["classe"] / it["nome"])
        (OUT / "split_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        with open(OUT / "manifesto_ciclo5.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["arquivo", "split", "classe", "fonte", "cultura", "tipo", "foto_original"])
            for it in sorted(itens, key=lambda x: (x["split"], x["classe"], x["nome"])):
                w.writerow([it["nome"], it["split"], it["classe"], it["fonte"], it["cultura"], it["tipo"], it["foto"]])
    return meta


def semear_cache_de_hashes() -> int:
    """
    A auditoria de quase-duplicatas guarda os hashes por caminho. Os hardlinks de data_c5
    têm caminho novo, e sem isto ela refaria 150 mil hashes. Copia as órbitas já
    calculadas para as chaves novas, num cache só do ciclo 5 (logs/hash_cache_c5.jsonl).
    """
    from imagehash_utils import _cache_key, _load_hash_cache

    ja = _load_hash_cache(CACHE_C5)
    arquivos = []
    for s in (*SPLITS, "retida"):
        for cls in CLASSES:
            with os.scandir(OUT / s / cls) as it:
                arquivos += [Path(e.path) for e in it if e.is_file()]
    por_nome = orbitas_do_cache({a.name for a in arquivos})
    for fonte, pasta, _ in PASTAS_NAO_RECONHECIDAS:
        for f in (RAW / fonte / pasta).iterdir():
            orbita = ja.get(_cache_key(f))
            if orbita:
                por_nome[f"{fonte}__{pasta.replace('/', '_')}_{f.name}"] = orbita
    n = 0
    with open(CACHE_C5, "a", encoding="utf-8") as fh:
        for a in arquivos:
            chave = _cache_key(a)
            if chave not in ja and a.name in por_nome:
                fh.write(json.dumps([chave, list(por_nome[a.name])]) + "\n")
                n += 1
    print(f"  cache de hashes do ciclo 5: {n} chaves novas ({len(arquivos)} arquivos em data_c5)")
    return n


def main():
    ap = argparse.ArgumentParser(description="Refaz o split para o ciclo 5 (não mexe em data/)")
    ap.add_argument("--apply", action="store_true", help="cria data_c5/ (sem isto, só mostra as contagens)")
    ap.add_argument("--semear-cache", action="store_true",
                    help="só prepara o cache de hashes de um data_c5/ que já existe")
    args = ap.parse_args()

    if args.semear_cache:
        semear_cache_de_hashes()
        return
    meta = montar(args.apply)
    if args.apply:
        semear_cache_de_hashes()
    print(f"\nCiclo 5 — {meta['total']} imagens")
    print(f"  {'':8s} {'healthy':>8s} {'anomalous':>10s} {'total':>8s}")
    for s, c in meta["images_per_split"].items():
        print(f"  {s:8s} {c.get('healthy', 0):8d} {c.get('anomalous', 0):10d} {sum(c.values()):8d}")
    print("\n  Por tipo:")
    for s, c in meta["tipos_por_split"].items():
        print(f"  {s:8s} " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    print("\n  Alface:", meta["alface"])
    print("  Pastas não reconhecidas:", meta["pastas_nao_reconhecidas"])
    print("\n  " + ("Criado em " + str(OUT) if args.apply else "Nada foi escrito. Rode com --apply para criar data_c5/."))


if __name__ == "__main__":
    main()
