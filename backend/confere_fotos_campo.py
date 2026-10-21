#!/usr/bin/env python
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
"""
Conferência das fotos de campo ANTES de rodar o modelo.

As 13 fotos do TCC-I ficaram sem procedência registrada, com uma de rótulo duvidoso e
sem metadados (docs/CORRECOES_METODOLOGICAS.md §14.7). Este script impede que o lote
novo repita isso. Para cada foto confere:

  • abre? tem EXIF de câmera (marca, modelo, data)? passou por editor? resolução?
    — foto sem EXIF costuma ter vindo de aplicativo de mensagem, captura de tela ou
      internet: guarde o arquivo original do celular;
  • é cópia ou quase-cópia de outra foto do lote (mesma folha fotografada duas vezes
    conta uma vez só na estatística);
  • é quase-duplicata de alguma imagem do dataset de treino (não pode);
  • está numa pasta de gabarito válida e tem linha na planilha de metadados, com a
    horta, a distância e, se for anômala, o tipo de dano.

Estrutura esperada (a mesma do test_campo.py):
    campo_novo/<cultura>/<healthy|anomalous>/*.jpg
    campo_novo/nao_folha/*.jpg
    campo_novo/metadados_campo.csv

Uso:
    python confere_fotos_campo.py
    python confere_fotos_campo.py --dir ./campo_novo --sem-dataset   # pula a comparação com o treino
Saída:
    results/campo_novo/conferencia_fotos.md / .json
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).parent))

from imagehash_utils import orbit_hashes

BASE = Path(__file__).resolve().parent
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}
GABARITO = {"healthy": "healthy", "anomalous": "anomalous", "nao_folha": "nao_folha", "not_a_leaf": "nao_folha"}
COLUNAS = ["ficheiro", "horta", "cultura", "verdade", "tipo_sintoma", "categoria_dano", "distancia", "hora", "notas"]
DISTANCIAS = {"close", "media", "ampla"}
DANOS = {"cromatico", "estrutural", "misto"}
PALAVRAS_DE_DUVIDA = ("duvid", "dúvid", "incert", "conferir", "talvez", "?")
LADO_MINIMO = 1000            # px; abaixo disso a foto quase certamente foi recomprimida
_BITS = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x)
    return _BITS[x.view(np.uint8).reshape(x.shape + (8,))].sum(axis=-1)


def listar(raiz: Path) -> tuple[list[dict], list[str]]:
    fotos, avisos = [], []
    for sub in sorted(p for p in raiz.iterdir() if p.is_dir()):
        if sub.name.lower() in GABARITO:
            pastas = [(sub, "", GABARITO[sub.name.lower()])]
        else:
            pastas = [(d, sub.name.lower(), GABARITO[d.name.lower()]) for d in sorted(sub.iterdir())
                      if d.is_dir() and d.name.lower() in GABARITO]
            if not pastas:
                avisos.append(f"pasta ignorada (sem healthy/ anomalous/ dentro): {sub.name}")
        for pasta, cultura, verdade in pastas:
            for f in sorted(pasta.iterdir()):
                if f.suffix.lower() in EXTENSOES:
                    fotos.append({"caminho": f, "ficheiro": f.name, "cultura": cultura, "verdade": verdade})
    return fotos, avisos


def ler_exif(im: Image.Image) -> dict:
    ex = im.getexif()
    sub = ex.get_ifd(0x8769) if ex else {}
    return {
        "camera": " ".join(str(ex.get(t, "")).strip() for t in (271, 272)).strip(),
        "data": str(sub.get(36867) or ex.get(306) or ""),
        "software": str(ex.get(305) or ""),
    }


def orbitas_do_dataset() -> tuple[list[str], np.ndarray]:
    por_nome = {}
    with (BASE / "data" / "hash_cache.jsonl").open(encoding="utf-8") as fh:
        for linha in fh:
            try:
                chave, orbita = json.loads(linha)
            except ValueError:
                continue
            if orbita and len(orbita) == 8:
                por_nome[chave.split("|")[0].replace("\\", "/").rsplit("/", 1)[-1]] = orbita
    nomes = list(por_nome)
    return nomes, np.array([por_nome[n] for n in nomes], dtype=np.uint64)


def conferir(raiz: Path, comparar_com_dataset: bool) -> dict:
    fotos, avisos = listar(raiz)
    problemas, alertas = [], list(avisos)

    # ── arquivo, EXIF, hash ───────────────────────────────────────────────────
    for f in fotos:
        try:
            with Image.open(f["caminho"]) as im:
                f.update(ler_exif(im))
                f["formato"], f["largura"], f["altura"] = im.format, *im.size
                f["orbita"] = orbit_hashes(im.convert("RGB"))
            f["md5"] = hashlib.md5(f["caminho"].read_bytes()).hexdigest()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            f["erro"] = str(exc)
            problemas.append(f"{f['ficheiro']}: não abre ({exc})")
            continue
        if not f["camera"]:
            alertas.append(f"{f['ficheiro']}: sem EXIF de câmera — é o arquivo original do celular?")
        if f["software"] and any(s in f["software"].lower() for s in ("photoshop", "gimp", "lightroom", "snapseed")):
            alertas.append(f"{f['ficheiro']}: passou por editor ({f['software']})")
        if max(f["largura"], f["altura"]) < LADO_MINIMO:
            alertas.append(f"{f['ficheiro']}: {f['largura']}×{f['altura']} px — parece recomprimida")
    validas = [f for f in fotos if "orbita" in f]

    # ── cópias dentro do lote ─────────────────────────────────────────────────
    nomes = Counter(f["ficheiro"] for f in fotos)
    problemas += [f"nome repetido em pastas diferentes: {n}" for n, c in nomes.items() if c > 1]
    O = np.array([f["orbita"] for f in validas], dtype=np.uint64) if validas else np.zeros((0, 8), np.uint64)
    pares = []
    for i in range(len(validas)):
        d = popcount(O[i + 1:, :, None] ^ O[i][None, None, :]).min(axis=(1, 2)) if i + 1 < len(validas) else []
        for j, dist in enumerate(d, start=i + 1):
            if dist <= 10:
                a, b = validas[i], validas[j]
                pares.append({"a": a["ficheiro"], "b": b["ficheiro"], "distancia": int(dist),
                              "rotulos_diferentes": a["verdade"] != b["verdade"]})
    for p in pares:
        texto = f"{p['a']} e {p['b']}: distância de hash {p['distancia']}"
        if p["distancia"] <= 4:
            (problemas if p["rotulos_diferentes"] else alertas).append(
                texto + (" — MESMA foto com rótulos diferentes" if p["rotulos_diferentes"] else " — mesma foto duas vezes"))
        else:
            alertas.append(texto + " — muito parecidas; se forem a mesma folha, anote")

    # ── contra o dataset de treino ────────────────────────────────────────────
    no_dataset = []
    if comparar_com_dataset and validas:
        nomes_ds, D = orbitas_do_dataset()
        for k, f in enumerate(validas):
            if k % 25 == 0:
                print(f"    comparando com o dataset {k}/{len(validas)}", flush=True)
            melhor = np.full(len(nomes_ds), 64, dtype=np.uint8)
            for h in f["orbita"]:
                melhor = np.minimum(melhor, popcount(D ^ np.uint64(h)).min(axis=1))
            i = int(melhor.argmin())
            f["menor_distancia_ao_dataset"] = int(melhor[i])
            if melhor[i] <= 4:
                no_dataset.append({"foto": f["ficheiro"], "imagem_do_dataset": nomes_ds[i], "distancia": int(melhor[i])})
                problemas.append(f"{f['ficheiro']}: quase-duplicata de imagem do dataset ({nomes_ds[i][:70]})")

    # ── metadados ─────────────────────────────────────────────────────────────
    planilha = raiz / "metadados_campo.csv"
    meta = {}
    if planilha.exists():
        with open(planilha, encoding="utf-8-sig", newline="") as fh:
            leitor = csv.DictReader(fh)
            faltam = [c for c in COLUNAS if c not in (leitor.fieldnames or [])]
            if faltam:
                problemas.append(f"metadados_campo.csv sem as colunas: {', '.join(faltam)}")
            meta = {Path(r["ficheiro"]).name: r for r in leitor if (r.get("ficheiro") or "").strip()}
    else:
        problemas.append(f"não existe {planilha.name} em {raiz}")
    duvidosas = []
    for f in fotos:
        m = meta.get(f["ficheiro"])
        if m is None:
            problemas.append(f"{f['ficheiro']}: sem linha na planilha de metadados")
            continue
        g = lambda c: (m.get(c) or "").strip().lower()                                # noqa: E731
        f["horta"] = g("horta")
        if not f["horta"]:
            problemas.append(f"{f['ficheiro']}: sem horta (a separação entre avaliação e adaptação é por horta)")
        if g("verdade") and g("verdade") != f["verdade"]:
            problemas.append(f"{f['ficheiro']}: planilha diz '{g('verdade')}', pasta diz '{f['verdade']}'")
        if f["cultura"] and g("cultura") and g("cultura") != f["cultura"]:
            problemas.append(f"{f['ficheiro']}: planilha diz cultura '{g('cultura')}', pasta diz '{f['cultura']}'")
        if f["verdade"] != "nao_folha" and g("distancia") not in DISTANCIAS:
            alertas.append(f"{f['ficheiro']}: distância '{g('distancia')}' — use close, media ou ampla (hipótese H3)")
        if f["verdade"] == "anomalous" and g("categoria_dano") not in DANOS:
            alertas.append(f"{f['ficheiro']}: categoria_dano '{g('categoria_dano')}' — use cromatico, estrutural ou misto (H2)")
        if any(p in g("notas") for p in PALAVRAS_DE_DUVIDA):
            duvidosas.append({"foto": f["ficheiro"], "verdade": f["verdade"], "notas": m.get("notas", "")})
    alertas += [f"planilha tem linha sem foto: {n}" for n in meta if n not in nomes]

    # ── contagens ─────────────────────────────────────────────────────────────
    por_cultura = defaultdict(Counter)
    por_horta = defaultdict(Counter)
    for f in fotos:
        por_cultura[f["cultura"] or "(sem cultura)"][f["verdade"]] += 1
        por_horta[f.get("horta") or "(sem horta)"][f["verdade"]] += 1
    for cultura, c in por_cultura.items():
        if cultura != "(sem cultura)" and min(c["healthy"], c["anomalous"]) < 5:
            alertas.append(f"{cultura}: {c['healthy']} saudáveis e {c['anomalous']} anômalas — pouco para medir as duas classes")
    if not any(f["verdade"] == "nao_folha" for f in fotos):
        alertas.append("nenhuma foto em nao_folha/ — a guarda de vegetação fica sem teste")
    if len([h for h in por_horta if h != "(sem horta)"]) < 2:
        alertas.append("menos de duas hortas — não dá para separar avaliação e adaptação por horta")

    return {
        "pasta": str(raiz),
        "n_fotos": len(fotos),
        "por_cultura": {k: dict(v) for k, v in sorted(por_cultura.items())},
        "por_horta": {k: dict(v) for k, v in sorted(por_horta.items())},
        "sem_exif_de_camera": sum(1 for f in validas if not f["camera"]),
        "cameras": dict(Counter(f["camera"] or "(sem EXIF)" for f in validas)),
        "pares_parecidos": pares,
        "quase_duplicatas_do_dataset": no_dataset,
        "rotulos_com_duvida_anotada": duvidosas,
        "problemas": problemas,
        "alertas": alertas,
        "fotos": [{k: v for k, v in f.items() if k not in ("caminho", "orbita")} for f in fotos],
    }


def markdown(r: dict) -> str:
    L = [f"# Conferência das fotos de campo — {r['n_fotos']} fotos", "", f"Pasta: `{r['pasta']}`", "",
         "## Contagem", "", "| Cultura | Saudáveis | Anômalas | Não folha |", "|---|---|---|---|"]
    for k, c in r["por_cultura"].items():
        L.append(f"| {k} | {c.get('healthy', 0)} | {c.get('anomalous', 0)} | {c.get('nao_folha', 0)} |")
    L += ["", "| Horta | Saudáveis | Anômalas | Não folha |", "|---|---|---|---|"]
    for k, c in r["por_horta"].items():
        L.append(f"| {k} | {c.get('healthy', 0)} | {c.get('anomalous', 0)} | {c.get('nao_folha', 0)} |")
    L += ["", f"Câmeras (EXIF): {r['cameras']}", "",
          f"## Problemas que impedem a medição ({len(r['problemas'])})", ""]
    L += [f"- {p}" for p in r["problemas"]] or ["Nenhum."]
    L += ["", f"## Alertas ({len(r['alertas'])})", ""]
    L += [f"- {a}" for a in r["alertas"]] or ["Nenhum."]
    L += ["", f"## Rótulos com dúvida anotada na coleta ({len(r['rotulos_com_duvida_anotada'])})", ""]
    L += [f"- {d['foto']} ({d['verdade']}): {d['notas']}" for d in r["rotulos_com_duvida_anotada"]] or ["Nenhum."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Conferência das fotos de campo antes da inferência")
    ap.add_argument("--dir", default=str(BASE / "campo_novo"))
    ap.add_argument("--out-dir", default=str(BASE / "results" / "campo_novo"))
    ap.add_argument("--sem-dataset", action="store_true", help="não compara com o dataset de treino (mais rápido)")
    args = ap.parse_args()

    raiz = Path(args.dir)
    if not raiz.exists():
        sys.exit(f"[ERRO] Pasta não encontrada: {raiz}")
    r = conferir(raiz, comparar_com_dataset=not args.sem_dataset)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "conferencia_fotos.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "conferencia_fotos.md").write_text(markdown(r), encoding="utf-8")
    print(markdown(r))
    sys.exit(1 if r["problemas"] else 0)


if __name__ == "__main__":
    main()
