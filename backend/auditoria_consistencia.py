"""
Auditoria de consistência de 13/10/2026 — o que os documentos afirmam contra o que
os artefatos mostram.

Não treina nem roda inferência: relê o split em disco, os nomes dos arquivos, as
previsões por imagem e os CSVs de campo, e responde a cinco perguntas
(§14 do docs/CORRECOES_METODOLOGICAS.md e docs/FICHA_TECNICA_FITOVISION.md):

  1. Quantas imagens há em cada etapa e em cada partição, e de onde vêm?
     (data/split_metadata.json + contagem em data/{train,val,test})
  2. Quanto do treino é de facto derivado do PlantVillage? A coletânea
     nirmalsankalana tem culturas que o PlantVillage não tem.
  3. As variantes transformadas da mesma foto de alface ficaram do mesmo lado do
     split? O nome do arquivo guarda o id da foto original; o hash perceptual não
     vê uma rotação de ângulo livre nem um filtro de bordas como "a mesma foto".
  4. Onde estão os erros do teste? Acurácia por subconjunto, das previsões já
     salvas em results/arquiteturas/predicoes/.
  5. Os 9 erros de campo dos quatro ciclos são de quantas fotografias?

Uso:
    python auditoria_consistencia.py                 # perguntas 1 a 5
    python auditoria_consistencia.py --parametros    # + parâmetros dos modelos treinados
    python auditoria_consistencia.py --hash          # + proveniência das outras fontes de alface
Saída:
    results/auditoria/auditoria_13out.json
    results/auditoria/auditoria_13out.md
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from compara_arquiteturas import MODELS, boot_bal_acc, ece
from compara_ciclos_teste_novo import bal_acc, holm, wilson_ci

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "results" / "auditoria"
SPLITS = ("train", "val", "test")
CLASSES = ("healthy", "anomalous")

FONTES_ALFACE = ("ashishjstar", "shuvokumarbasak", "santoshshaha", "iqrapervez")
LOTES_CICLO4 = ("mendeley_spinach", "roboflow_lettuce_downy")
VARIANTES = "shuvokumarbasak2030_lettuce-disease-multi-transformation-dataset__"
# As 14 culturas do PlantVillage, com os nomes usados pela coletânea nirmalsankalana.
CULTURAS_PLANTVILLAGE = {
    "Apple", "Blueberry", "Cherry", "Corn", "Grape", "Orange", "Peach", "Bell_pepper",
    "Potato", "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato",
}
# Culturas da coletânea fotografadas na lavoura (planta no solo, fundo natural).
# Classificação feita por inspeção visual de amostras em 13/10/2026; a mandioca é o
# conjunto Cassava Leaf Disease (21.397 imagens de campo).
CULTURAS_DE_CAMPO = ("Cassava", "Rice", "Sugercane", "Coffee")
CAMPO = {
    "ciclo1": "campo_resultados.csv",
    "ciclo2": "campo_resultados_v2.csv",
    "ciclo3": "campo_resultados_v3.csv",
    "ciclo4": "campo_resultados_v4.csv",
    "vit_b_16": "vit/campo_resultados_vit.csv",
}
TAU = 0.85
# Arquivos de calibração das versões avaliadas em campo (a do ciclo 2 não foi preservada).
CALIBRACAO = {
    "ciclo1": "weights/pre_lettuce_expansion/efficientnet_b0_binary_calibration.json",
    "ciclo3": "weights/ciclo3/efficientnet_b0_binary_calibration.json",
    "ciclo4": "weights/efficientnet_b0_binary_calibration.json",
    "vit_b_16": "weights/vit/vit_b_16_binary_calibration.json",
}

RX_CULTURA = re.compile(r"__data_([^_]+(?:_pepper)?)___")
RX_VAR_AUG = re.compile(r"^augmentation_augmentation_(?P<cls>.+?)_(?P<tr>[a-z_]+?)_(?P<id>\d+)_(?P=tr)\.\w+$")
RX_VAR_DS = re.compile(r"^dataset_dataset_(?:train|test|val)_(?P<cls>.+?)_(?P<id>\d+)_(?P<tr>[a-z_]+)\.\w+$")


def fonte(nome: str) -> str:
    return nome.split("__")[0]


def cultura(nome: str) -> str | None:
    m = RX_CULTURA.search(nome)
    return m.group(1) if m else None


def original_da_variante(nome: str) -> tuple[str, int] | None:
    """(classe, id da foto original) de um arquivo da fonte de variantes transformadas."""
    if not nome.startswith(VARIANTES):
        return None
    resto = nome[len(VARIANTES):]
    m = RX_VAR_AUG.match(resto) or RX_VAR_DS.match(resto)
    return (m["cls"], int(m["id"])) if m else None


def listar_split() -> dict[str, list[tuple[str, str]]]:
    """{split: [(nome do arquivo, classe), ...]} lido do disco."""
    out = {}
    for split in SPLITS:
        itens = []
        for cls in CLASSES:
            with os.scandir(DATA / split / cls) as it:
                itens += [(e.name, cls) for e in it if e.is_file()]
        out[split] = itens
    return out


# ── 1 e 2. Tamanho e composição do dataset ────────────────────────────────────

def composicao(arquivos: dict) -> dict:
    meta = json.loads((DATA / "split_metadata.json").read_text(encoding="utf-8"))
    staging = meta["n_images_staging"]
    base_meta = sum(sum(v.values()) for v in meta["images_per_split"].values())
    out = {
        "staging_ciclo3": staging,
        "descartadas_no_split": {
            "corrompidas": meta["n_corrupted_dropped"],
            "grupos_com_rotulos_contraditorios": meta["n_conflicted_groups_dropped"],
            "imagens_descartadas_total": staging - base_meta,
        },
        "fotos_distintas_staging": meta["n_distinct_photos"],
        "por_split": {},
    }
    for split, itens in arquivos.items():
        base = [(n, c) for n, c in itens if not fonte(n).startswith(LOTES_CICLO4)]
        por_fonte = defaultdict(Counter)
        por_cultura = defaultdict(Counter)
        for n, c in itens:
            por_fonte[fonte(n)][c] += 1
        for n, c in base:
            if fonte(n).startswith("nirmal"):
                por_cultura[cultura(n) or "?"][c] += 1
        alface = Counter(c for n, c in base if fonte(n).startswith(FONTES_ALFACE))
        outras = Counter(c for n, c in base if not fonte(n).startswith(FONTES_ALFACE))
        classe = Counter(c for _, c in base)
        colet = sum(sum(v.values()) for v in por_cultura.values())
        colet_pv = sum(sum(v.values()) for k, v in por_cultura.items() if k in CULTURAS_PLANTVILLAGE)
        campo = sum(sum(por_cultura[k].values()) for k in CULTURAS_DE_CAMPO)
        pv = sum(por_fonte["plantvillage"].values())
        n_base = len(base)
        out["por_split"][split] = {
            "total_ciclo4": len(itens),
            "base_seis_fontes": n_base,
            "classes_base": dict(classe),
            "por_fonte": {k: dict(v) for k, v in sorted(por_fonte.items())},
            "alface": {**alface, "pct_da_base": round(100 * sum(alface.values()) / n_base, 1),
                       "pct_saudavel": round(100 * alface["healthy"] / sum(alface.values()), 1)},
            "nao_alface": {**outras, "pct_saudavel": round(100 * outras["healthy"] / sum(outras.values()), 1)},
            "pct_da_classe_healthy_que_e_alface": round(100 * alface["healthy"] / classe["healthy"], 1),
            "pct_da_classe_anomalous_que_e_alface": round(100 * alface["anomalous"] / classe["anomalous"], 1),
            "coletanea": {
                "total": colet,
                "culturas_do_plantvillage": colet_pv,
                "outras_culturas": colet - colet_pv,
                "por_cultura": {k: dict(v) for k, v in sorted(por_cultura.items())},
            },
            # O número antigo (70,1%) somava as duas fontes inteiras.
            "pct_duas_fontes_somadas": round(100 * (pv + colet) / n_base, 1),
            "pct_derivado_plantvillage_por_cultura": round(100 * (pv + colet_pv) / n_base, 1),
            "pct_culturas_de_campo": round(100 * campo / n_base, 1),
        }
    return out


# ── 3. Vazamento por proveniência nas variantes transformadas ─────────────────

def proveniencia_variantes(arquivos: dict) -> dict:
    originais = {s: defaultdict(int) for s in SPLITS}
    transf = set()
    for split, itens in arquivos.items():
        for n, _ in itens:
            chave = original_da_variante(n)
            if chave:
                originais[split][chave] += 1
                transf.add((RX_VAR_AUG.match(n[len(VARIANTES):]) or RX_VAR_DS.match(n[len(VARIANTES):]))["tr"])
    treino = set(originais["train"])
    out = {
        "fotos_originais": len(set().union(*[set(o) for o in originais.values()])),
        "transformacoes": sorted(transf),
        "por_split": {},
    }
    for split in ("val", "test"):
        n_img = sum(originais[split].values())
        com_irma = sum(v for k, v in originais[split].items() if k in treino)
        out["por_split"][split] = {
            "imagens": n_img,
            "fotos_originais": len(originais[split]),
            "fotos_originais_tambem_no_treino": len(set(originais[split]) & treino),
            "imagens_com_variante_no_treino": com_irma,
            "pct": round(100 * com_irma / n_img, 1),
        }
    return out


def proveniencia_por_hash(arquivos: dict) -> dict:
    """
    As outras três fontes de alface não guardam o id da foto no nome. Liga-se cada
    imagem a uma foto original pelo hash perceptual do projeto: distância <= 4 a uma
    variante pouco alterada (espelho, brilho, contraste...) da fonte de variantes.
    """
    suaves = {"flipped_horizontal", "flipped_vertical", "brightness_adjusted", "contrast_adjusted",
              "saturation_adjusted", "unsharp_mask", "gaussian_noise", "poisson_noise",
              "hist_equalized", "jittered"}
    onde = {n: s for s, itens in arquivos.items() for n, _ in itens}
    ref_chaves, ref_orbitas, outras = [], [], {}
    with (DATA / "hash_cache.jsonl").open(encoding="utf-8") as fh:
        for linha in fh:
            try:
                k, orbita = json.loads(linha)
            except ValueError:
                continue
            caminho = k.split("|")[0].replace("\\", "/")
            nome = caminho.split("/")[-1]
            if not orbita or len(orbita) != 8 or nome not in onde or "/staging/" in caminho:
                continue
            if nome.startswith(VARIANTES):
                m = RX_VAR_AUG.match(nome[len(VARIANTES):])
                if m and m["tr"] in suaves:
                    ref_chaves.append((m["cls"], int(m["id"])))
                    ref_orbitas.append(orbita)
            elif nome.startswith(FONTES_ALFACE):
                outras[nome] = orbita
    ref = np.array(ref_orbitas, dtype=np.uint64)
    bits = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def distancias(h: int) -> np.ndarray:
        x = np.ascontiguousarray(ref ^ np.uint64(h))
        return bits[x.view(np.uint8).reshape(x.shape + (8,))].sum(axis=-1).min(axis=1)

    original = {}
    for nome, orbita in outras.items():
        d = np.min([distancias(h) for h in orbita], axis=0)
        i = int(d.argmin())
        if d[i] <= 4:
            original[nome] = ref_chaves[i]
    treino = {original_da_variante(n) for n, _ in arquivos["train"]} - {None}
    treino |= {v for n, v in original.items() if onde[n] == "train"}
    out = {}
    for f in ("ashishjstar", "iqrapervez", "santoshshaha"):
        nomes = [n for n in outras if fonte(n).startswith(f)]
        ident = [n for n in nomes if n in original]
        out[f] = {"imagens": len(nomes), "com_foto_original_identificada": len(ident),
                  "pct_identificada": round(100 * len(ident) / len(nomes), 1)}
        for split in ("val", "test"):
            ns = [n for n in nomes if onde[n] == split]
            vaz = sum(1 for n in ns if n in original and original[n] in treino)
            out[f][split] = {"imagens": len(ns), "com_variante_no_treino": vaz,
                             "pct": round(100 * vaz / len(ns), 1) if ns else None}
    return out


# ── 4. Desempenho por subconjunto do teste ────────────────────────────────────

def carregar_previsoes() -> dict:
    pred = {}
    for m in MODELS:
        with open(BASE / "results/arquiteturas/predicoes" / f"{m}.csv", encoding="utf-8") as f:
            pred[m] = {r["chave"].split("/", 1)[1]: (r["verdade"], r["pred"], float(r["conf"]))
                       for r in csv.DictReader(f) if r["recorte"] == "original"}
    return pred


def por_subconjunto(pred: dict) -> dict:
    nomes = sorted(pred[MODELS[0]])
    variantes = lambda n: n.startswith(VARIANTES)                      # noqa: E731
    de_campo = lambda n: cultura(n) in CULTURAS_DE_CAMPO               # noqa: E731
    subconjuntos = {
        "teste_original": lambda n: True,
        # Dentro da mesma fonte (a coletânea), para não confundir fonte com condição de captura.
        "coletanea_culturas_de_campo": de_campo,
        "coletanea_demais_culturas": lambda n: fonte(n).startswith("nirmal") and not de_campo(n),
        "so_mandioca": lambda n: cultura(n) == "Cassava",
        "plantvillage_fonte": lambda n: fonte(n) == "plantvillage",
        "variantes_transformadas_de_alface": variantes,
        "sem_variantes_transformadas": lambda n: not variantes(n),
    }
    out = {}
    for rotulo, filtro in subconjuntos.items():
        ns = [n for n in nomes if filtro(n)]
        verdade = np.array([pred[MODELS[0]][n][0] for n in ns])
        bloco = {"n": len(ns), "healthy": int((verdade == "healthy").sum()),
                 "anomalous": int((verdade == "anomalous").sum()), "modelos": {}, "mcnemar": {}}
        certo = {}
        for m in MODELS:
            ok = np.array([pred[m][n][1] == pred[m][n][0] for n in ns])
            conf = np.array([pred[m][n][2] for n in ns])
            certo[m] = ok
            erros = int((~ok).sum())
            bloco["modelos"][m] = {
                "bal_acc": round(bal_acc(ok, verdade), 4),
                "ic95": [round(v, 4) for v in boot_bal_acc(ok, verdade)],
                "fn": int((~ok & (verdade == "anomalous")).sum()),
                "fp": int((~ok & (verdade == "healthy")).sum()),
                "erros": erros,
                "erros_interceptados": int((~ok & (conf < TAU)).sum()),
                "abstencao_pct": round(100 * float((conf < TAU).mean()), 1),
            }
        ps, pares = {}, {}
        for a, b in itertools.combinations(MODELS, 2):
            so_a = int((certo[a] & ~certo[b]).sum())
            so_b = int((~certo[a] & certo[b]).sum())
            ps[f"{a} vs {b}"] = float(binomtest(so_a, so_a + so_b, 0.5).pvalue) if so_a + so_b else 1.0
            pares[f"{a} vs {b}"] = {"so_a": so_a, "so_b": so_b}
        for par, h in holm(ps).items():
            bloco["mcnemar"][par] = {**pares[par], **h, "significativo": bool(h["significativo"])}
        out[rotulo] = bloco
    return out


# ── 5. Campo: foto a foto ─────────────────────────────────────────────────────

def campo_foto_a_foto() -> dict:
    tabela = defaultdict(dict)
    for versao, arq in CAMPO.items():
        with open(BASE / "results" / arq, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                tabela[r["ficheiro"].replace("\\", "/")][versao] = (r["verdade"], r["resposta"], float(r["confianca"]))

    def resumo(reclassificar: str | None = None, excluir: str | None = None) -> dict:
        out = {}
        for versao in CAMPO:
            acertos = emitidos = 0
            for foto, d in tabela.items():
                if foto == excluir:
                    continue
                verdade, resposta, _ = d[versao]
                if foto == reclassificar:
                    verdade = "anomalous"
                if resposta in ("inconclusive", "not_a_leaf"):
                    continue
                emitidos += 1
                acertos += resposta == verdade
            out[versao] = {"acertos": acertos, "emitidos": emitidos,
                           "ic95_wilson": [round(v, 3) for v in wilson_ci(acertos, emitidos)]}
        return out

    ciclos = [v for v in CAMPO if v.startswith("ciclo")]
    erros_por_foto = Counter()
    confiancas = []
    for foto, d in tabela.items():
        for v in ciclos:
            verdade, resposta, conf = d[v]
            if resposta in CLASSES and resposta != verdade:
                erros_por_foto[foto] += 1
                confiancas.append(conf)
    duvidosa = "healthy/alface_04.jpg"
    return {
        "calibracao_em_campo": temperatura_em_campo(tabela),
        "n_fotos": len(tabela),
        "erros_nos_quatro_ciclos": sum(erros_por_foto.values()),
        "fotos_distintas_com_erro": len(erros_por_foto),
        "erros_por_foto": dict(erros_por_foto.most_common()),
        "confianca_dos_erros": [min(confiancas), max(confiancas)],
        "como_esta": resumo(),
        # healthy/alface_04.jpg mostra uma folha com manchas escuras e é dada como
        # anômala nos quatro ciclos; o rótulo precisa de conferência.
        "foto_de_rotulo_duvidoso": duvidosa,
        "se_reclassificada_como_anomala": resumo(reclassificar=duvidosa),
        "se_excluida": resumo(excluir=duvidosa),
        "tabela": {foto: {v: {"resposta": d[v][1], "confianca": d[v][2], "verdade": d[v][0]} for v in CAMPO}
                   for foto, d in sorted(tabela.items())},
    }


def temperatura_em_campo(tabela: dict) -> dict:
    """
    As temperaturas ajustadas na validação são menores que 1, ou seja, AUMENTAM a
    confiança. Em campo isso ajuda ou atrapalha? O CSV guarda a confiança calibrada c
    da classe prevista; em classificação binária, a confiança sem temperatura é
    sigmoid(T · logit(c)). Compara-se o ECE dos diagnósticos emitidos nos dois casos.
    """
    out = {}
    for versao, arq in CALIBRACAO.items():
        T = json.loads((BASE / arq).read_text(encoding="utf-8"))["temperature"]
        emitidos = [d[versao] for d in tabela.values() if d[versao][1] in CLASSES]
        certo = np.array([resposta == verdade for verdade, resposta, _ in emitidos])
        conf = np.clip(np.array([c for *_, c in emitidos]), 1e-6, 1 - 1e-6)
        sem_T = 1 / (1 + np.exp(-T * np.log(conf / (1 - conf))))
        out[versao] = {
            "temperatura": round(T, 3),
            "n_emitidos": len(emitidos),
            "ece_com_temperatura": round(ece(conf, certo), 3),
            "ece_sem_temperatura": round(ece(sem_T, certo), 3),
            "confianca_media_com_temperatura": round(float(conf.mean()), 3),
            "confianca_media_sem_temperatura": round(float(sem_T.mean()), 3),
            "confianca_dos_erros_sem_temperatura": [round(float(v), 3) for v in sem_T[~certo]],
        }
    return out


# ── Parâmetros dos modelos treinados (opcional: precisa de torch) ─────────────

def parametros() -> dict:
    import torch.nn as nn
    from torchvision import models

    def construir(nome: str, n_classes: int) -> nn.Module:
        m = getattr(models, nome)(weights=None)
        if n_classes != 1000:
            if nome == "resnet50":
                m.fc = nn.Linear(m.fc.in_features, n_classes)
            elif nome == "vit_b_16":
                m.heads.head = nn.Linear(m.heads.head.in_features, n_classes)
            else:
                m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
        return m

    contar = lambda m: sum(p.numel() for p in m.parameters())          # noqa: E731
    return {nome: {"arquitetura_original_1000_classes": contar(construir(nome, 1000)),
                   "modelo_treinado_2_classes": contar(construir(nome, 2))} for nome in MODELS}


# ── Relatório ─────────────────────────────────────────────────────────────────

def markdown(r: dict) -> str:
    L = ["# Auditoria de consistência — 13/10/2026", "",
         "Gerado por `backend/auditoria_consistencia.py`. Não envolve treino nem inferência.", ""]
    c = r["composicao"]
    d = c["descartadas_no_split"]
    L += ["## 1. Tamanho do dataset", "",
          f"- Staging do ciclo 3: {c['staging_ciclo3']} imagens.",
          f"- Descartadas no split: {d['imagens_descartadas_total']} ({d['corrompidas']} corrompida(s) e as de "
          f"{d['grupos_com_rotulos_contraditorios']} grupos com rótulos contraditórios).", "",
          "| Partição | Base (6 fontes) | Ciclo 4 (com espinafre e míldio) |", "|---|---|---|"]
    for s in SPLITS:
        L.append(f"| {s} | {c['por_split'][s]['base_seis_fontes']} | {c['por_split'][s]['total_ciclo4']} |")
    L.append(f"| **total** | **{sum(c['por_split'][s]['base_seis_fontes'] for s in SPLITS)}** | "
             f"**{sum(c['por_split'][s]['total_ciclo4'] for s in SPLITS)}** |")
    t = c["por_split"]["train"]
    L += ["", "## 2. Composição do treino-base", "",
          f"- Alface (4 fontes): {t['alface']['pct_da_base']}% do treino; {t['alface']['pct_saudavel']}% dela é `healthy`.",
          f"- Demais culturas: {t['nao_alface']['pct_saudavel']}% são `healthy`.",
          f"- {t['pct_da_classe_healthy_que_e_alface']}% da classe `healthy` é alface; "
          f"{t['pct_da_classe_anomalous_que_e_alface']}% da classe `anomalous` é alface.",
          f"- PlantVillage + culturas do PlantVillage na coletânea: {t['pct_derivado_plantvillage_por_cultura']}% "
          f"(as duas fontes somadas dão {t['pct_duas_fontes_somadas']}%, o número citado até aqui).",
          f"- Culturas da coletânea fotografadas em campo: {t['pct_culturas_de_campo']}%.", ""]
    p = r["proveniencia_variantes"]
    L += ["## 3. Variantes transformadas da mesma foto de alface", "",
          f"{p['fotos_originais']} fotos originais, {len(p['transformacoes'])} transformações por foto.", ""]
    for s, v in p["por_split"].items():
        L.append(f"- {s}: {v['imagens_com_variante_no_treino']} de {v['imagens']} imagens ({v['pct']}%) têm outra "
                 f"variante da mesma foto original no treino.")
    if "proveniencia_outras_fontes" in r:
        for f, v in r["proveniencia_outras_fontes"].items():
            L.append(f"- {f}: {v['pct_identificada']}% das imagens derivam das mesmas fotos originais; no teste, "
                     f"{v['test']['com_variante_no_treino']} de {v['test']['imagens']} têm variante no treino.")
    L += ["", "## 4. Acurácia balanceada por subconjunto do teste original", "",
          "| Subconjunto | N | " + " | ".join(MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
    for rotulo, b in r["por_subconjunto"].items():
        celulas = [f"{b['modelos'][m]['bal_acc']:.4f} [{b['modelos'][m]['ic95'][0]:.4f}; {b['modelos'][m]['ic95'][1]:.4f}] "
                   f"({b['modelos'][m]['erros']} erros)" for m in MODELS]
        L.append(f"| {rotulo} | {b['n']} | " + " | ".join(celulas) + " |")
    k = r["campo"]
    L += ["", "## 5. Campo, foto a foto", "",
          f"{k['erros_nos_quatro_ciclos']} erros nos quatro ciclos, em {k['fotos_distintas_com_erro']} fotografias: "
          + ", ".join(f"{f} ({n})" for f, n in k["erros_por_foto"].items()) + ".", "",
          "| Foto | " + " | ".join(CAMPO) + " |", "|---|" + "---|" * len(CAMPO)]
    for foto, d in k["tabela"].items():
        cel = []
        for v in CAMPO:
            x = d[v]
            marca = "abstém" if x["resposta"] not in CLASSES else ("certo" if x["resposta"] == x["verdade"] else "**erro**")
            cel.append(f"{marca} ({x['confianca']:.3f})")
        L.append(f"| {foto} | " + " | ".join(cel) + " |")
    L += ["", "### Calibração em campo: com e sem a temperatura ajustada na validação", "",
          "| Versão | T | ECE com T | ECE sem T | Confiança média com T | sem T |", "|---|---|---|---|---|---|"]
    for v, x in k["calibracao_em_campo"].items():
        L.append(f"| {v} | {x['temperatura']} | {x['ece_com_temperatura']} | {x['ece_sem_temperatura']} | "
                 f"{x['confianca_media_com_temperatura']} | {x['confianca_media_sem_temperatura']} |")
    if "parametros" in r:
        L += ["", "## Parâmetros", "", "| Modelo | Arquitetura original (1.000 classes) | Modelo treinado (2 classes) |",
              "|---|---|---|"]
        for m, v in r["parametros"].items():
            L.append(f"| {m} | {v['arquitetura_original_1000_classes']} | {v['modelo_treinado_2_classes']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parametros", action="store_true", help="conta os parâmetros dos modelos (precisa de torch)")
    ap.add_argument("--hash", action="store_true", help="liga as outras fontes de alface às fotos originais pelo hash")
    args = ap.parse_args()

    arquivos = listar_split()
    r = {
        "composicao": composicao(arquivos),
        "proveniencia_variantes": proveniencia_variantes(arquivos),
        "por_subconjunto": por_subconjunto(carregar_previsoes()),
        "campo": campo_foto_a_foto(),
    }
    if args.hash:
        r["proveniencia_outras_fontes"] = proveniencia_por_hash(arquivos)
    if args.parametros:
        r["parametros"] = parametros()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "auditoria_13out.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "auditoria_13out.md").write_text(markdown(r), encoding="utf-8")
    print(markdown(r))
    print(f"Gravado em {OUT}")


if __name__ == "__main__":
    main()
