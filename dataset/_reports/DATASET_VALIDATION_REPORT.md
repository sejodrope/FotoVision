# Validação dos novos datasets Roboflow — TCC-II FitoVision

Data: 31/08/2026
Origem: `dataset/TCC 2 - novos dados/` (6 pastas exportadas do Roboflow)
Nada foi apagado, movido ou sobrescrito nos originais. Todo trabalho está em
`dataset/_work/` (crops + caches de hash) e `dataset/_reports/` (relatórios).
**Nenhum treinamento foi executado.**

---

## Resumo executivo

Os 6 datasets baixados do Roboflow são **100% de detecção de objeto (COCO)**,
não classificação. Foram recortados (bounding box → imagem de classificação)
com sua aprovação, gerando 58.605 crops. A validação de integridade e
duplicatas revelou um problema estrutural sério: **a maior parte desses dados
"novos" não é nova** — é a mesma coleção de fotos de doenças de alface que já
compõe o dataset atual do FitoVision (via os Kaggle `ashishjstar`,
`iqrapervez2000`, `santoshshaha`, `shuvokumarbasak2030`, já documentados no
histórico do projeto), agora só re-empacotada pelo Roboflow com bounding
boxes. Além disso, **nenhum dos 6 datasets contém fotos de campo reais** —
são todos recortes de estúdio/referência, então não ajudam a fechar o domain
gap que é a contribuição central do seu TCC.

**Recomendação geral: não vale a pena integrar a maior parte desses dados.**
Detalhes e exceções pontuais abaixo.

---

## Fase 1 — Formato detectado

| Pasta original | Formato | Splits (train/val/test) | Categorias no COCO |
|---|---|---|---|
| `Bok Choy- Lettuce- Spinach- Diseased.v1i.coco` | COCO detecção | 7479/705/377 | super + `Bok-Choy-Diseased`, `Lettuce-Diseased`, `Spinach-Diseased` (sem classe saudável) |
| `Lettuce Disease.v1i.coco` | COCO detecção | 1344/128/64 | super + `Bacterial`, `Downy_mildew_on_lettuce`, `Powdery_mildew_on_lettuce`, `Septoria_Blight_on_lettuce`, `Viral`, `Wilt_and_leaf_blight_on_lettuce`, `healthy` |
| `Lettuce.v1i.coco (1)` | COCO detecção | 20256/0/0 (tudo em train) | super + `Bacterial`, `Downy_mildew_on_lettuce`, `Septoria_Blight_on_lettuce`, `normal` |
| `Lettuce.v2i.coco` | COCO detecção | 20172/1216/1008 | idênticas ao v1i (1) — mesmo projeto Roboflow, versão posterior |
| `lettuce.v1i.coco` | COCO detecção | 2022/42/29 | `disease` (super) + `Downy-mildew-lettuce` (classe única) |
| `lettuce.v1i.coco (2)` | COCO detecção | 2022/42/29 | **duplicata byte-a-byte** de `lettuce.v1i.coco` (`diff -rq` = 0 diferenças) — excluída do processamento |

Nenhum dataset veio nativo em formato de classificação. Conforme combinado,
recortamos cada bounding box (com 8% de padding de contexto) em vez de usar a
imagem inteira, já que 3 dos 5 datasets únicos têm classe saudável explícita.

---

## Fase 2 — Recorte, inventário e integridade

**58.605 crops gerados, 0 arquivos corrompidos** (nem na origem nem nos
recortes).

| Dataset (nome curto) | Crops OK | saudavel | anomala | descartado fora-de-escopo | erro | % crops <224px | resolução (mediana W×H) |
|---|---|---|---|---|---|---|---|
| `lettuce_disease_v1` | 2.010 | 329 | 1.681 | 0 | 0 | 22,8% | 389×395 |
| `lettuce_v1i` | 21.571 | 6.387 | 15.184 | 0 | 35 | **79,1%** | 195×194 |
| `lettuce_v2i` | 23.847 | 7.130 | 16.717 | 0 | 35 | **79,7%** | 196×191 |
| `bokchoy_lettuce_spinach` | 8.815 | 0 | 8.815 | 3.313 (Bok Choy) | 0 | 27,6% | 365×538 |
| `lettuce_downy_v1` | 2.362 | 0 | 2.362 | 0 | 0 | 6,1% | 473×485 |
| **Total** | **58.605** | **13.846** | **44.759** | **3.313** | **70** | | |

Achados:
- Todas as imagens abrem corretamente, todas em RGB, todas salvas como JPEG.
- **`lettuce_v1i` e `lettuce_v2i` têm ~79% dos crops menores que 224px**
  (mínimo 4px!, mediana ~195px, máximo 256px). O Roboflow parece ter
  reexportado essas imagens já reduzidas para no máximo 256px — muitos
  desses recortes de bbox virariam ruído após upscale para 224×224 do ViT.
  Isso sozinho já compromete boa parte do valor desses dois datasets.
- `lettuce_downy_v1` tem a melhor qualidade de resolução (só 6,1% tiny,
  mediana 473×485).
- 70 anotações (`lettuce_v1i`/`v2i`) apontavam para bboxes com dimensão
  final <4px após crop — descartadas automaticamente (`erro_crop`).

---

## Fase 3 — Mapeamento de classes (aplicado conforme sua aprovação)

| Dataset | Categoria Roboflow | Classe FitoVision |
|---|---|---|
| Lettuce Disease / Lettuce v1i / v2i | `healthy` / `normal` | `saudavel` |
| Lettuce Disease / Lettuce v1i / v2i | `Bacterial`, `Downy_mildew_on_lettuce`, `Powdery_mildew_on_lettuce`, `Septoria_Blight_on_lettuce`, `Viral`, `Wilt_and_leaf_blight_on_lettuce` | `anomala` |
| Bok Choy-Lettuce-Spinach | `Lettuce-Diseased`, `Spinach-Diseased` | `anomala` |
| Bok Choy-Lettuce-Spinach | `Bok-Choy-Diseased` | **excluído** (espécie fora do escopo-alvo: alface/rúcula/espinafre/acelga/couve) |
| lettuce.v1i.coco | `Downy-mildew-lettuce` | `anomala` |
| todos | supercategorias "guarda-chuva" do Roboflow (ex.: `lettuce-disease`, `disease`) | ignoradas (0 anotações próprias, confirmado) |

Nenhuma categoria usada nas anotações ficou sem mapeamento (`categorias_sem_mapeamento`
vazio em todos os datasets) — nada foi descartado silenciosamente.

**Balanceamento resultante dos novos dados: 13.846 saudavel : 44.759 anomala
(24% : 76%)** — mais desbalanceado que o dataset atual (71k:139k = 34%:66%),
porque 2 dos 5 datasets (Bok Choy e Downy) não têm classe saudável alguma.

---

## Fase 4 — Duplicatas internas e colisões com o dataset existente

Usei o mesmo método de hash perceptual já validado no projeto (D4-orbit
dHash, `backend/imagehash_utils.py`, limiar de Hamming ≤4 — igual ao usado
no split atual), cruzando os 58.605 crops novos contra as 170.772 imagens de
`backend/data/{train,val,test}`.

### 🔴 Colisão com o TEST set existente — risco grave de integridade acadêmica

**17 grupos de duplicata, envolvendo 246 crops novos**, batem com imagens que
já estão no seu **test set atual**:

| Dataset novo | Crops envolvidos | Grupos |
|---|---|---|
| `bokchoy_lettuce_spinach` | 105 | 10 |
| `lettuce_v2i` | 66 | 4 |
| `lettuce_v1i` | 63 | 4 |
| `lettuce_disease_v1` | 12 | 3 |
| `lettuce_downy_v1` | **0** | 0 |

Lista completa em `dataset/_reports/dedup_leak_test.json`. Exemplos
verificados manualmente: os crops recortados de `p09`, `s04`, `let35`,
`let43`, `ba12` (Bok Choy dataset) e `h2`, `images-16-` (Lettuce v1i/v2i)
batem exatamente com fotos já presentes em `backend/data/test/` sob os nomes
`ashishjstar_lettuce-diseases`, `iqrapervez2000_lettuce-disease-dataset`,
`santoshshaha_lettuce-plant-disease-dataset` e
`shuvokumarbasak2030_lettuce-disease-multi-transformation-dataset` — as
mesmas 4 fontes Kaggle que, conforme o histórico do projeto, são
republicações da mesma coleção-base de fotos de doenças de alface.
**Se qualquer um desses 246 crops fosse usado em treino, contaminaria
diretamente o seu test set** (a foto "nova" e a foto de teste são a mesma
imagem, ou uma variante direta dela).

Uma ressalva: um dos grupos (`group_id 86590`, 86 crops de `lettuce_v1i`/`v2i`
todos do mesmo `images-16-`) bateu com 2 imagens de teste de **Tomate**
saudável. **Conferido visualmente (comparação lado a lado) — é falso
positivo do hash**, não uma colisão real: o crop de alface é um recorte
pequeno e de baixa nitidez, sem nenhuma relação visual com as folhas de
tomate. Provavelmente o dHash confundiu duas texturas verdes genéricas de
baixo detalhe. Os outros 16 grupos, porém, são inequivocamente a mesma foto
(nomes de arquivo e proveniência batendo exatamente) — **contam como
vazamento real de 160 crops** (246 − 86 do grupo falso-positivo).

### 🟡 Colisão com TRAIN/VAL existente — redundância, não vazamento de teste

**74 grupos, 1.073 crops novos** batem com imagens já em `train`/`val`
(não em `test`). Não é um risco de integridade de métricas, mas significa
que boa parte do "dado novo" já é conhecido do modelo.

| Dataset novo | Crops envolvidos | Grupos |
|---|---|---|
| `bokchoy_lettuce_spinach` | 865 | 55 |
| `lettuce_v2i` | 78 | 6 |
| `lettuce_disease_v1` | 61 | 19 |
| `lettuce_v1i` | 62 | 6 |
| `lettuce_downy_v1` | 7 | 1 |

### 🟠 Duplicatas internas (só entre os dados novos, sem tocar o existente)

**8.201 grupos, 48.767 crops** (83% dos 58.605!) são duplicatas/quase-duplicatas
*entre si* — principalmente `lettuce_v1i` × `lettuce_v2i` (o mesmo projeto
Roboflow, v1→v2: 4.770 dos 8.201 grupos cruzam 2+ datasets novos diferentes,
confirmando que v2i é majoritariamente uma reexportação de v1i com mais
imagens, não conteúdo independente).

| Dataset novo | Crops em grupos internos |
|---|---|
| `lettuce_v2i` | 20.458 |
| `lettuce_v1i` | 18.297 |
| `bokchoy_lettuce_spinach` | 7.328 |
| `lettuce_downy_v1` | 1.469 |
| `lettuce_disease_v1` | 1.215 |

### Resumo: quanto sobra de "conteúdo genuinamente novo"?

Somando as três categorias de duplicata por dataset (colide com test + colide
com train/val + duplicata só interna) e subtraindo do total:

| Dataset | Total crops | Tocado por alguma duplicata | Aparentemente único (limite superior) |
|---|---|---|---|
| `lettuce_disease_v1` | 2.010 | 1.288 (64%) | ~722 (36%) |
| `lettuce_v1i` | 21.571 | 18.422 (85%) | ~3.149 (15%) |
| `lettuce_v2i` | 23.847 | 20.602 (86%) | ~3.245 (14%) |
| `bokchoy_lettuce_spinach` | 8.815 | 8.298 (94%) | ~517 (6%) |
| `lettuce_downy_v1` | 2.362 | 1.476 (62%) | ~886 (38%), **0% toca o test** |

("Único" aqui é limite superior — inclui imagens em grupos de duplicata
interna das quais só 1 representante seria mantido; o número final depende
de como o Fase 6 resolver os grupos.)

---

## Fase 5 — Caracterização de domínio: estúdio vs. campo

Gerei grades de amostra (32 imagens cada) em `dataset/_reports/sample_grids/`
e revisei visualmente. **Nenhum dos 5 datasets contém fotos de campo real**
(sem solo, canteiro, múltiplas plantas ao ar livre, contexto de cultivo).
São todos recortes fechados de bbox — folha isolada ou lesão em close-up,
com fundo de mesa/prato/bancada ou ausente (crop muito fechado). O dataset
`bokchoy_lettuce_spinach` tem inclusive marca d'água "University of
California" em algumas imagens, sugerindo fonte de banco de referência
fitopatológica — o mesmo tipo de material controlado que já domina seu
dataset atual.

**Conclusão para o TCC: esses 6 datasets não contribuem para fechar o domain
gap.** São mais dado-proxy, majoritariamente redundante com o que você já
tem. A prioridade de fotos de campo reais (já registrada como bloqueante no
histórico do projeto) continua sendo o único caminho para isso.

---

## Recomendação go/no-go por dataset

| Dataset | Recomendação | Justificativa |
|---|---|---|
| `lettuce_v1i` (Lettuce.v1i (1)) | **NO-GO** | 85% duplicado (interno + existente), 79% dos crops <224px, é a versão mais antiga do mesmo projeto que o v2i supera |
| `lettuce_v2i` (Lettuce.v2i) | **NO-GO** | 86% duplicado, 79% dos crops <224px — mesmo problema do v1i, só que maior |
| `bokchoy_lettuce_spinach` | **NO-GO** | 94% duplicado com dados já existentes — praticamente não sobra conteúdo novo; sem classe saudável; Bok Choy já excluído |
| `lettuce_disease_v1` | **NO-GO** | 64% duplicado, mesma origem de fotos (Kaggle já usado); melhor resolução que v1i/v2i mas pouco volume líquido (~700 imagens únicas estimadas) |
| `lettuce_downy_v1` | **GO condicional, escopo pequeno** | Único com 0% de colisão com o test set; melhor qualidade de resolução (94% ≥224px); ainda tem 62% de duplicata interna, mas os ~886 crops únicos estimados (só `anomala`/Downy mildew) poderiam somar como reforço pontual para essa doença específica, se você quiser |

**Nenhum dataset contribui com fotos de campo.** Se o objetivo é fechar o
domain gap (a contribuição central do TCC), nenhum destes 6 ajuda — o
esforço deveria ir para mais fotos de campo reais, como já estava priorizado
no seu histórico do projeto.

---

## O que precisa da sua aprovação antes de qualquer próximo passo

- [ ] **Confirmar exclusão total** de `lettuce_v1i`, `lettuce_v2i`,
      `bokchoy_lettuce_spinach`, `lettuce_disease_v1` (minha recomendação:
      sim, descartar) — ou você quer que eu tente extrair só o subconjunto
      único de algum deles mesmo assim?
- [ ] **Decidir sobre `lettuce_downy_v1`**: incorporar o subconjunto único
      (~886 crops `anomala`, downy mildew) ao dataset via Fase 6, ou
      descartar também por simplicidade/consistência?
- [x] ~~Validar manualmente o grupo suspeito de falso positivo~~ — feito:
      `group_id 86590` (lettuce↔tomate) confirmado como falso positivo do
      hash. Os outros 16 grupos de colisão com o test set (160 crops) são
      vazamento real.
- [ ] Se algo for aprovado para a Fase 6 (reorganização ImageFolder + split
      70/15/15 + manifesto CSV), confirmar que o test set atual permanece
      **intocado** e que nenhum crop dos grupos de colisão com test entra em
      treino/val.

## Próximos passos sugeridos

1. Você decide os itens acima.
2. Se sobrar algo para aproveitar (provavelmente só `lettuce_downy_v1`),
   rodo a Fase 6: dedup final (1 representante por grupo interno, exclusão
   de todo grupo que toque o test set), reorganização ImageFolder, checagem
   224×224/ImageNet, manifesto CSV com proveniência.
2. Caso decida não aproveitar nada, o mais produtivo agora é retomar a
   prioridade já registrada: **conseguir mais fotos de campo reais** (meta
   20-40 para validação exploratória, 100-300 para o TCC-II).
