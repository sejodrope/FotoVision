# Fase 6 — Preparação final (espinafre + lettuce_downy_v1)

Data: 31/08/2026. Aprovado por José em "perfeito, pode avançar!!" (cobrindo
o dataset de espinafre, discutido explicitamente, + `lettuce_downy_v1`, que
eu decidi incluir por já ter sido recomendado GO no primeiro relatório —
avisar se não era essa a intenção).

Nada foi treinado. Nada em `backend/data/` (o dataset de produção) foi
tocado — este material fica em `dataset/_work/final/` até você decidir
mergear de fato (isso dispara um retreino completo, que não rodei).

---

## O que foi feito

1. **Dedup final**: para cada grupo de quase-duplicata só-interna (achado
   na Fase 4), mantive 1 representante (o primeiro em ordem alfabética,
   determinístico) e descartei o resto.
2. **Exclusão de qualquer grupo que tocasse o dataset já existente**
   (`backend/data/`) — mesmo os que confirmamos como falso positivo
   visualmente, por regra conservadora e reprodutível (se rodar de novo
   com dados diferentes, a mesma lógica protege sem precisar de checagem
   manual caso a caso).
3. **Split 70/15/15 com seed=42**, estratificado por classe, aplicado
   somente aos dados novos — o test set atual (`backend/data/test/`)
   **não foi tocado nem misturado**.
4. **Manifesto CSV** por dataset, com phash (reaproveitado dos caches já
   calculados, não recalculado).

## Resultado

| Dataset | Bruto | Excluído (dup/colisão) | Final | Train | Val | Test |
|---|---|---|---|---|---|---|
| Espinafre (2 fontes Mendeley) | 3.609 | 400 | **3.209** | 2.245 | 480 | 484 |
| `lettuce_downy_v1` (Roboflow) | 2.362 | 1.220 | **1.142** | 799 | 171 | 172 |

**Espinafre**, detalhe por classe/split:

| Split | saudavel | anomala |
|---|---|---|
| train | 865 | 1.380 |
| val | 185 | 295 |
| test | 187 | 297 |

`lettuce_downy_v1` é 100% `anomala` (dataset original não tinha classe
saudável) — 799/171/172.

Localização: `dataset/_work/final/spinach/<split>/<classe>/` e
`dataset/_work/final/lettuce_downy_v1/<split>/<classe>/`. Manifestos:
`dataset/_reports/final_manifest_spinach.csv` e
`final_manifest_lettuce_downy_v1.csv` (colunas: caminho, classe, split,
origem_dataset, resolução, phash_orbit).

Contagem em disco conferida = contagem no manifesto (3.209 e 1.142) — sem
divergência.

## Compatibilidade 224×224 / ImageNet

Não precisei propor transforms novos — o projeto já tem os canônicos em
`backend/dataset.py` (`TRAIN_TRANSFORMS`/`VAL_TRANSFORMS`): `RandomResizedCrop(224)`
+ flips/rotação/jitter de cor moderado no treino, `Resize(256)+CenterCrop(224)`
na validação/teste, ambos com `Normalize(IMAGENET_MEAN, IMAGENET_STD)`. Os
dois lotes novos são compatíveis sem ajuste — resolução de origem sempre
≥118px (espinafre é ≥246px), então o crop/resize não amplia nada além do
que o pipeline já faz normalmente.

## O que ainda falta (decisão sua, não técnica)

- [ ] **Mergear de fato com `backend/data/`?** Isto é o próximo passo real
      — hoje os dados novos estão isolados em `dataset/_work/final/`,
      prontos, mas separados. Mergear significa: copiar para
      `backend/data/{train,val,test}/{healthy,anomalous}/` (renomeando
      saudavel→healthy, anomala→anomalous para bater com a convenção do
      projeto) e então decidir se dispara um novo ciclo de treino
      (~17-22h pelos ciclos anteriores). **Não fiz isso ainda** — é uma
      mudança estrutural maior, prefiro seu OK explícito antes.
- [ ] Confirmar que quer registrar no TCC a ressalva de espécie do
      Malabar Spinach (já combinado, só reforçando aqui).
- [ ] Se mergear, o dataset ganha uma 6ª "espécie-classe" implícita
      (espinafre) e mais volume de alface downy mildew — vale atualizar
      `docs/CORRECOES_METODOLOGICAS.md` com a proveniência nova.

## Recomendação

Não dispararia o retreino agora — a prioridade continua sendo as fotos de
campo (roteiro já entregue). Sugiro: mergear estes dados preparados
**junto** com as fotos de campo novas, num único ciclo 4, em vez de gastar
17-22h de treino duas vezes.

---

## Correção (22/09/2026): o split do `lettuce_downy_v1` vazava

Ao preparar o merge do ciclo 4, conferi os nomes dos arquivos: os 1.142
arquivos do `lettuce_downy_v1` são **recortes de anotação** (`__annN`) e
**variantes Roboflow** (`.rf.<hash>`) de apenas **49 fotos originais**. O
split acima foi feito por recorte, e 45 das 49 fotos tinham recortes em mais
de um split (1.138 de 1.142 imagens afetadas). O dedup por phash não pega
isso porque recortes diferentes da mesma foto não são quase-duplicados.

O split deste lote foi **refeito agrupado pela foto original** em
`backend/merge_ciclo4.py` (seed 42): 773 / 196 / 173 imagens, vindas de
32 / 12 / 5 fotos originais (treino / val / teste). As pastas em
`dataset/_work/final/lettuce_downy_v1/` e o manifesto
`final_manifest_lettuce_downy_v1.csv` **mantêm o split antigo** e não devem
ser usados como split. O split válido está em
`backend/data/ciclo4_merge_manifest.csv`.

Consequência para o texto do TCC: este lote acrescenta **49 fotos
independentes** de downy mildew, não 1.142. O teste dele tem só 5 fotos
originais, então as métricas de teste deste lote servem apenas como
indicação.

O espinafre não tem esse problema: 0 grupos por nome em mais de um split.
