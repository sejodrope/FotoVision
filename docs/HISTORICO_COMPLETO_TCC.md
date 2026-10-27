# FitoVision — Histórico Completo do Projeto

### Da primeira versão (99,01%) até o estado atual — guia para a apresentação da banca

> **Como usar este documento**: ele conta a história inteira, em ordem cronológica,
> de forma que dê para entender tudo sem precisar abrir mais nada. Onde há detalhe
> técnico mais fundo (código, fórmulas, decisões de implementação), há uma referência
> para `docs/CORRECOES_METODOLOGICAS.md` e `docs/GUIA_TESTE_DE_CAMPO.md`. Todos os
> números aqui são medidos e citam o arquivo onde foram gerados — nada é estimativa
> não identificada como tal.
>
> **Resumo de uma frase, se precisar responder rápido na banca**: *"Publicamos um
> resultado de 99% que se revelou inválido por vazamento de dados; corrigimos o
> vazamento, medimos o gap real entre laboratório e campo, encontramos e corrigimos
> um bug de rotulagem, e concluímos — com rigor estatístico — que o teste de campo
> atual é pequeno demais para decidir qual versão do modelo é melhor. Isso não é uma
> fraqueza escondida: é o resultado do trabalho."*
>
> **Acréscimo de 08/10/2026 (ler a §9-B antes de apresentar):** *"O teste sem
> vazamento sai das mesmas seis fontes do treino, então os 98,5% medem acerto em
> fotos parecidas com as do treino. Numa fonte que o modelo nunca viu a acurácia
> cai 23 pontos, e em horta os erros saem com 99% de confiança. O que falta é foto
> com fundo real — para medir e, depois, para treinar."*
>
> **Acréscimo de 13/10/2026 (ler a §9-E antes de apresentar):** *"Auditei os meus
> próprios números. Dentro do teste, 81% dos erros estão nas culturas fotografadas
> em lavoura, onde a acurácia é de 86%, contra 99,7% nas fotos de estúdio da mesma
> fonte. E descobri que um quarto do teste são variantes transformadas de fotos de
> alface que também estão no treino, o que o meu detector de duplicatas não via."*
> Os fatos conferidos estão em `docs/FICHA_TECNICA_FITOVISION.md`.
>
> **Acréscimo de 14/10/2026 (ler a §9-F antes de apresentar):** *"Corrigi o que a
> auditoria achou e treinei de novo as quatro arquiteturas. Nas culturas fotografadas em
> lavoura, os falsos alarmes caíram — a especificidade subiu de 7 a 12 pontos nas CNNs —
> mas a sensibilidade caiu de 2 a 5. Não é um modelo melhor em tudo: é outro ponto de
> operação, e eu digo qual erro cada um comete. Numa fonte de espinafre que nenhum modelo
> viu, todos ficam entre 86% e 90%, dez pontos abaixo das fontes vistas."*

---

## Sumário

1. [O projeto em uma página](#1-o-projeto-em-uma-página)
2. [Linha do tempo](#2-linha-do-tempo)
3. [Fase 0 — Construção do dataset](#3-fase-0--construção-do-dataset)
4. [Fase 1 — Os "99,01%" e a retratação](#4-fase-1--os-9901-e-a-retratação)
5. [Fase 2 — Correção do vazamento (ciclo 1)](#5-fase-2--correção-do-vazamento-ciclo-1)
6. [Fase 3 — O gap de domínio: por que corrigir o vazamento não bastou](#6-fase-3--o-gap-de-domínio-por-que-corrigir-o-vazamento-não-bastou)
7. [Fase 4 — Ciclo 2: abstenção corrigida, dataset expandido, novo bug encontrado](#7-fase-4--ciclo-2-abstenção-corrigida-dataset-expandido-novo-bug-encontrado)
8. [Fase 5 — Ciclo 3: rotulagem corrigida, e uma lição sobre estatística](#8-fase-5--ciclo-3-rotulagem-corrigida-e-uma-lição-sobre-estatística)
9. Fase 6 — Ciclo 4: multi-cultura, e o primeiro ganho que passa num teste estatístico
9-B. **Fase 7 — Revisão crítica: o teste se parece com o treino (08/10/2026)**
9-C. **Fase 8 — Comparação pareada entre arquiteturas e preparação do ViT (12/10/2026)**
9-D. **Fase 9 — ViT-B/16 treinado: empata com o EfficientNet-B0 (13/10/2026)**
9-E. **Fase 10 — Auditoria de consistência: os números contra os artefatos (13/10/2026)**
9-F. **Fase 11 — Ciclo 5: o pipeline corrigido, nas quatro arquiteturas (14/10/2026)**
10. Estado atual — tudo num só lugar
11. As lições metodológicas
12. Trabalho futuro / próximos passos
13. Glossário rápido
14. Índice de artefactos (onde está cada prova)

---

## 1. O projeto em uma página

**FitoVision** é o TCC de José Pedro (Engenharia de Software, UNIVILLE): um sistema
de diagnóstico fitossanitário para hortaliças folhosas (alface, rúcula, espinafre,
acelga, couve) por visão computacional.

- **Tarefa**: classificação binária — `healthy` (saudável) vs. `anomalous` (com
  algum problema: doença, praga, deficiência nutricional).
- **Abordagem**: Transfer Learning — três arquiteturas de CNN pré-treinadas em
  ImageNet (MobileNetV2, ResNet50, EfficientNet-B0), ajustadas (*fine-tuning*
  completo) ao problema.
- **Stack**: PyTorch + FastAPI (backend), React + TypeScript (frontend), SQLite.
- **Contribuição pretendida**: não é só "um modelo que classifica folhas" — é a
  **metodologia**: como construir um pipeline de dados reprodutível, como detectar e
  corrigir vazamento de dados, como medir honestamente a diferença entre desempenho
  de laboratório e desempenho real, e como continuar melhorando com rigor
  estatístico em vez de "parece melhor, então é melhor".

---

## 2. Linha do tempo

| Data | Evento |
|---|---|
| (anterior) | Construção do dataset (210.832 imagens), treino dos 3 modelos, resultado publicado: **99,01% de acurácia** |
| (anterior) | **Retratação**: descoberto vazamento de dados — 57,3% do teste "vazado" do treino |
| 12–13/07/2026 | **Ciclo 1**: pipeline de correção — split reconstruído sem vazamento, guarda de vegetação (ExG), calibração de confiança. Retreino dos 3 modelos. |
| 26/07/2026 | José reporta que o sintoma ("qualquer imagem vira diagnóstico confiante") persiste. Investigação retomada. |
| 27/07/2026 | Achado e corrigido o **bug do limiar de abstenção** (nunca disparava). Primeiro **teste de campo real**: 87,5% (13 fotos). Início do **ciclo 2**: 3 datasets Kaggle novos de alface adicionados. |
| 27–28/07/2026 | Ciclo 2 treinado (17,5h). Teste de campo: **caiu para 77,5%**. Investigação revela **bug de rotulagem** — alface saudável marcada como doente desde sempre. Rollback para o ciclo 1 em produção. |
| 28–29/07/2026 | Bug de rotulagem corrigido. **Ciclo 3** lançado: staging refeito, retreino completo (22,2h). Teste de campo: **caiu ainda mais, para 73,3%** — mas a análise estatística mostra que a diferença **não é significativa** com a amostra actual (N=13). |
| 29/07/2026 | Ciclo 3 mantido em produção (sem bugs conhecidos). Documentação consolidada. Próximo passo definido: ampliar o teste de campo antes de qualquer novo retreino. |
| 31/08/2026 | Validação de datasets novos: os de alface do Roboflow foram quase todos descartados por duplicação (aprovado só o lote de míldio); 2 de espinafre-de-malabar aprovados. |
| 14/09/2026 | Parcial I do TCC-II entregue (questão de pesquisa, objetivos, plano de análise). |
| 22–23/09/2026 | **Ciclo 4**: espinafre + míldio incorporados. Achado e corrigido o split por recorte do míldio (1.142 imagens de 49 fotos). Ganho significativo nas culturas novas. |
| 08/10/2026 | **Revisão crítica para o Seminário 2**: o teste é das mesmas fontes do treino; fonte nunca vista cai 23 pontos; o p do ciclo 4 estava inflado por recortes da mesma foto; em campo os erros saem com 99% de confiança (§9-B). |
| 12–13/10/2026 | Comparação pareada entre arquiteturas; **ViT-B/16** treinado no Colab e empata com o EfficientNet-B0 (§9-C, §9-D). Artigo v4 e correções do orientador (v5). |
| 13/10/2026 | **Auditoria de consistência**: 81% dos erros do teste estão nas culturas fotografadas em lavoura (86,3% contra 99,7%); 23% do teste são variantes de fotos de alface com outra variante no treino; "70% PlantVillage" e "16 vezes mais parâmetros" corrigidos (§9-E). Artigo v6. |
| 14/10/2026 | **Ciclo 5** no Colab (06:06–17:52, ~22 unidades): split de alface por fotografia original, pragas rotuladas, amostragem por cultura, lote de espinafre retido; quatro arquiteturas. Menos falsos alarmes nas culturas de lavoura, com perda de sensibilidade (§9-F). Artigo v7. A coleta de campo foi adiada. |
| 16/10/2026 | **Literatura lida na íntegra** (7 de 9 trabalhos): a revisão de Shafay et al. converge com o ViT (o ViT puro é o mais sensível ao ambiente); o "98,7% → 32,8%" de Gerz e Jelali é mAP50 de detecção; Nguyen et al. descrevem o mesmo atalho pela cultura. Artigo v8, monografia v5 (`docs/REVISAO_LITERATURA_WOS.md`, Parte 4). |

---

## 3. Fase 0 — Construção do dataset

### O problema de partida

Não existe dataset público robusto de **hortaliças folhosas brasileiras**
especificamente (alface, rúcula, espinafre, acelga, couve). Decisão: usar
**PlantVillage** (54.306 imagens em 38 classes de 14 culturas, majoritariamente
tomate/batata/milho) como **proxy visual** — o modelo aprende a reconhecer *padrões visuais de doença em
folhas* de outras culturas e generaliza para as culturas-alvo — complementado por
datasets específicos de alface onde disponíveis.

### Datasets usados (dataset original, antes desta sessão)

| Fonte | Imagens | Papel |
|---|---|---|
| `abdallahalidev/plantvillage-dataset` (Kaggle) | 54.306 | Proxy visual — 15 pastas relevantes (17.585 imagens) filtradas das 38 classes |
| `ashishjstar/lettuce-diseases` (Kaggle) | 2.337 | **Única fonte específica de alface** até o ciclo 2 |
| `shuvokumarbasak2030/lettuce-disease-multi-transformation-dataset` (Kaggle) | 79.458 | Alface com augmentation pré-aplicado |
| `nirmalsankalana/plant-diseases-training-dataset` (Kaggle) | 116.147 | Coletânea de vários conjuntos, 70+ classes: PlantVillage e culturas que ele não tem (mandioca, arroz, rosa, cana, café, melancia). **Não é só PlantVillage** — ver §9-E |
| Roboflow (20 datasets tentados) | 0 úteis | Todos eram datasets de *detecção* (bounding box), não classificação — resultado negativo documentado |

**Total**: 210.832 imagens (healthy: 71.052 / anomalous: 139.780 — desbalanceamento
2:1, tratado com `WeightedRandomSampler`).

### Problemas técnicos resolvidos nesta fase

- **BFS recursivo para rotulagem**: datasets tinham estruturas de pastas com
  profundidades diferentes (2 a 4 níveis). Solução: `_collect_labeled_dirs()` —
  busca em largura que encontra pastas rotuladas (`healthy`/`anomalous`) em
  qualquer profundidade, sem entrar dentro de uma pasta já rotulada. **Esta mesma
  função tinha um bug que só foi encontrado e corrigido nesta sessão — ver Fase 4.**
- **Pydantic rejeitando `.env`**: corrigido com `extra="ignore"`.
- **PyTorch sem CUDA**: reinstalado `torch==2.5.1+cu121` (driver da RTX 3050 exige
  cu121, não cu124).

---

## 4. Fase 1 — Os "99,01%" e a retratação

### O resultado original

Treino dos 3 modelos no split 70/15/15 (seed=42) do dataset de 210.832 imagens:

| Modelo | Accuracy | F1 (macro) | Latência |
|---|---|---|---|
| **EfficientNet-B0** | **99,01%** | 98,89% | 24,1ms |
| MobileNetV2 | 98,81% | 98,67% | 14,3ms |
| ResNet50 | 98,71% | 98,56% | 16,7ms |

EfficientNet-B0 foi para produção. Números excelentes — bons demais, na verdade.

### O sintoma que motivou a investigação

Ao testar com fotografias fora do dataset, o desempenho não correspondia ao
prometido. Diagnósticos erráticos, sem relação clara com a condição real da folha.
*Excelente no papel, inútil na prática.*

### A pergunta certa

> Não *"por que o modelo é mau?"*, mas **"por que o número é bom demais?"**

### A causa: vazamento de dados (data leakage)

Investigação por hash perceptual (`imagehash_utils.py`) revelou: o split
train/val/test tinha sido feito **por arquivo**, não por **fotografia distinta**.
Como vários datasets continham cópias quase-idênticas da mesma foto (mesma planta,
mesmo ângulo, pequenas variações de compressão/crop — inclusive entre datasets
diferentes que redistribuíam PlantVillage), a mesma foto acabava em treino **e** em
teste. O modelo "decorava" fotos que reencontrava disfarçadas no teste.

**Medido, não estimado** (`results/leakage_report_old_split.json`):

```
Total de imagens:        210.832
Fotos distintas:         106.403   (redundância de 49,5%!)
Teste vazado:             57,35%
Validação vazada:          57,08%
Veredicto:                 GRAVE
```

Mais da metade do "teste" já tinha sido vista, sob outro nome de arquivo, durante o
treino. O 99,01% não media capacidade de diagnóstico — media memorização.

**Detalhe técnico da correção do split**: `imagehash_utils.py` — hash perceptual
invariante a rotações de 90°/espelhamento (grupo D4), agrupamento de
*near-duplicates* por *union-find* com LSH (Locality-Sensitive Hashing) para não
precisar comparar cada imagem com todas as outras (O(n²) seria inviável em 210k
imagens). Grupos de fotos quase-idênticas são mantidos **inteiros** dentro de um
único split (nunca divididos entre treino/val/teste) — só assim o vazamento
desaparece de fato.

> Detalhe completo da investigação (hipóteses testadas, indícios, código):
> `docs/CORRECOES_METODOLOGICAS.md` §1–§4.

---

## 5. Fase 2 — Correção do vazamento (ciclo 1)

*(12–13/07/2026)*

### O que foi corrigido, de uma vez

1. **Split agrupado por identidade visual** — grupos de fotos quase-idênticas nunca
   são divididos entre splits.
2. **Desbalanceamento de classes** — `WeightedRandomSampler` + pesos de classe na
   *loss*.
3. **Excesso de confiança do softmax (calibração)** — *Temperature Scaling* (Guo et
   al., 2017): aprende um escalar T que "achata" as probabilidades sem alterar a
   predição, para que "80% de confiança" signifique de fato 80% de acerto. Medido
   pelo **ECE** (Expected Calibration Error).
4. **Limiar de abstenção** — abaixo de certa confiança, o sistema devolve
   `inconclusive` em vez de arriscar.
5. **Guarda de vegetação (ExG — Excess Green Index)** — antes de classificar,
   verifica se a imagem sequer parece conter uma folha; se não, devolve
   `not_a_leaf`. Fórmula: `ExG = 2g − r − b` sobre canais RGB normalizados
   (Woebbecke et al., 1995).

### Resultado (split limpo, sem vazamento)

`results/leakage_report.json` do ciclo 1: **0,18% de vazamento** (vs. 57,3% antes).

| Modelo | Accuracy | Bal. Acc | F1 (macro) | ECE | Latência |
|---|---|---|---|---|---|
| **EfficientNet-B0** | **98,53%** | **98,59%** | 98,35% | 0,0062 | 8,3ms |
| ResNet50 | 97,52% | 97,89% | 97,24% | 0,0084 | 5,7ms |
| MobileNetV2 | 96,97% | 97,23% | 96,62% | 0,0074 | 5,6ms |

Queda de ~99% para ~98,5% — **e essa queda é o resultado correto**. Um número mais
baixo, mas verdadeiro, vale mais que um número mais alto e inválido.

> Detalhe técnico completo (código alterado, artefactos gerados):
> `docs/CORRECOES_METODOLOGICAS.md` §5, §8.

---

## 6. Fase 3 — O gap de domínio: por que corrigir o vazamento não bastou

Corrigir o vazamento torna a *métrica* verdadeira. Não garante que o modelo acerte
em fotos de campo. Existe um segundo problema, **independente**:

- O grosso do dataset é PlantVillage: folha única, fundo neutro de estúdio,
  iluminação controlada.
- Uma foto de telemóvel numa horta real tem terra, sombra, folhas sobrepostas,
  oclusão, luz natural variável.
- **O PlantVillage não contém alface, rúcula nem espinafre.** O que o modelo "sabe"
  sobre alface, até o ciclo 2, era **transferido** de tomate/batata/milho, não
  observado directamente (excepto pelos 2.337 exemplos de `ashishjstar`).

A literatura citada (ver `docs/CORRECOES_METODOLOGICAS.md` §6) aponta **30-60% de
acerto** em fotos de campo para modelos treinados só em PlantVillage. Isso não se
resolve com mais épocas de treino — resolve-se com dados de campo e de espécie-alvo.

Esta secção já previa, antes de qualquer teste de campo real acontecer, exactamente
os dois eixos que dominaram o resto do trabalho: (1) precisa de mais dados
específicos das culturas-alvo, e (2) precisa de um teste de campo real para medir o
gap de verdade, não só estimá-lo pela literatura.

---

## 7. Fase 4 — Ciclo 2: abstenção corrigida, dataset expandido, novo bug encontrado

*(27–28/07/2026)*

### 7.1 O sintoma reapareceu — e desta vez tinha uma causa concreta

Mesmo com a guarda ExG e a abstenção implementadas (Fase 2), José continuou vendo
diagnósticos confiantes em qualquer imagem. Investigação encontrou um **bug real**,
distinto do gap de domínio:

**`pick_threshold()` nunca conseguia fazer o sistema abster-se.** O algoritmo
buscava o *menor* limiar de confiança τ, a partir de 0,50, cuja *accuracy* batesse
95%. Como a confiança de uma classificação binária (softmax argmax de 2 classes)
**nunca é menor que 0,5** — é o piso matemático — e o modelo já acertava ~98% no val
set mesmo em τ=0,50, o algoritmo parava sempre na primeira iteração.

**Prova, antes da correção** (`weights/efficientnet_b0_binary_calibration.json`):
```json
{"threshold": 0.5, "coverage_at_threshold": 1.0, "selective_accuracy": 0.9802}
```
`coverage: 1.0` = o sistema **nunca** se abstinha. Um dos dois pilares da honestidade
do sistema estava, na prática, desligado desde a Fase 2.

**Correcção**: piso mínimo `min_tau=0,85` e meta de *accuracy* seletiva mais
exigente (98%, antes 95%) em `calibrate.py`. Recalibrados os 3 modelos — τ passou a
ser 0,85 em todos.

### 7.2 Primeiro teste de campo real da história do projeto

José reuniu 20 fotos reais de alface (horta, luz natural, terra visível). **A origem
de cada uma não foi registrada** (os arquivos não têm EXIF de celular; um é de câmera
Sony, de 2014 — §9-E.5). Triadas visualmente: **13 usadas como gabarito** (5 healthy, 8 anomalous — as mesmas 13
fotos usadas em todos os testes desde então, para comparação justa), **7
excluídas** por não serem representativas (uma vinha de banco de dados
fitopatológico com ID de catálogo, uma tinha texto sobreposto, cinco tinham
coloração ambígua que podia ser sintoma ou pigmentação natural da variedade).

**Resultado** (`results/campo_resultados.csv`, `test_campo.py`):

| Conjunto | Acurácia balanceada |
|---|---|
| Teste (estúdio, PlantVillage) | 98,6% |
| **Campo (horta real)** | **87,5%** |

Gap de 11,1pp — bem menor que os 30-60% da literatura (Fase 3), mas a amostra
(N=13) é pequena demais para conclusões fortes. Ainda assim, é o primeiro **dado
real**, não mais uma estimativa. Um erro (folha saudável marcada anomalous com 98%
de confiança) e uma abstenção correcta (confiança 0,72 → `inconclusive`, provando
que a correcção do §7.1 funciona na prática).

### 7.3 Expansão do dataset

Buscados no Kaggle datasets específicos de alface (culturas-alvo, não proxy):

- `santoshshaha/lettuce-plant-disease-dataset` (2.813 imagens)
- `iqrapervez2000/lettuce-disease-dataset` (6.992 imagens)
- `ramadhanihsaniyulfa/tomato-and-lettuce-diseases-dataset` (271 imagens — acabou
  não sendo aproveitado, estrutura de pastas não reconhecida pela rotulagem)

**Resultado negativo, documentado por rigor**: não existe dataset Kaggle de doenças
foliares específico para **rúcula, espinafre, acelga ou couve**. Confirmado por
busca, não por falta de esforço — é uma lacuna real de dados públicos, registada
como limitação e trabalho futuro.

Staging cresceu de 170.152 para 179.957 imagens. Retreino completo (17,5h).

### 7.4 O resultado — e a descoberta do bug real

**Domínio de teste**: praticamente empatado com o ciclo 1 (ver tabela completa na
§9). **Campo (mesmas 13 fotos): caiu para 77,5%.** Duas fotos de alface
claramente doente, corretas no ciclo 1, passaram a ser ditas "healthy" com 95-100%
de confiança — o oposto do esperado ao adicionar mais dados de alface.

Investigação (em vez de descartar como ruído) encontrou a causa: `ashishjstar/
lettuce-diseases` — **a única fonte de alface do projeto desde o início** — está
organizado como `Lettuce_disease_datasets/{Healthy, Bacterial, Downy_mildew, ...}/`.
O nome do container (`Lettuce_disease_datasets`) bate a palavra-chave "disease" da
lógica de rotulagem, e o BFS (Fase 0) parava ali, rotulando a pasta **inteira**
(incluindo `Healthy/`) como `anomalous`, sem nunca descer para diferenciar.

**Resultado: 1.123 imagens de alface saudável (48% deste dataset) estavam
rotuladas como doente desde o ciclo 1** — não só o ciclo 2. Os 2 datasets novos
tinham o mesmo padrão de empacotamento: +2.849 imagens. **Total: 3.972 imagens de
alface saudável mal-rotuladas**, entre os dois ciclos. A única fonte de "como é uma
alface saudável" que o modelo tinha estava etiquetada ao contrário — explica o
sintoma original melhor do que o gap de domínio isoladamente.

**Acção**: pesos do ciclo 2 revertidos (produção voltou ao ciclo 1) — este era um
caso claro para reverter: rótulos comprovadamente errados, não apenas um número de
campo pior.

> Detalhe técnico completo (código, evidência passo a passo):
> `docs/CORRECOES_METODOLOGICAS.md` §10.1–§10.6.

---

## 8. Fase 5 — Ciclo 3: rotulagem corrigida, e uma lição sobre estatística

*(28–29/07/2026)*

### 8.1 A correcção

`_collect_labeled_dirs()` corrigida: só aceita o rótulo do nome do *container*
quando **nenhuma subpasta imediata resolve para rótulo próprio**; caso contrário,
desce e usa os rótulos mais específicos dos filhos. Testado directamente contra
`ashishjstar_lettuce-diseases` antes de confiar na correcção — `Healthy/` (1.123
imagens) passou a ser separada correctamente dos 6 subtipos de doença.

Staging reconstruído do zero: `healthy` subiu de 72.252→76.224 (+3.972, exactamente
as imagens recuperadas); `anomalous` caiu de 107.705→99.902 (-7.803 — os 3.972
realocados **mais** 3.831 imagens de erva daninha, "Shepherd_purse_weeds", que o
mesmo bug também deixava vazar para `anomalous` sem passar pelo filtro de
descarte). Um bônus não previsto: o bug também contaminava a classe `anomalous`
com uma planta que não é nem sequer a cultura-alvo.

Split refeito a partir de 176.126 imagens em staging: descartadas 1 corrompida e 5.353
de 167 grupos com rótulos contraditórios, **ficam 170.772** (120.861 / 25.944 / 23.967);
vazamento por quase-duplicata de 0,04% (ainda melhor que o ciclo 1). No ciclo 2 os grupos
contraditórios eram 1.239 (32.027 imagens): o defeito de rotulagem também tirava essas
imagens do split, que tinha só 147.929.

### 8.2 O resultado — e por que ele não é o que se esperava

**Domínio de teste**: sem surpresa, praticamente igual aos ciclos anteriores (ver
tabela §9) — ~4 mil imagens corrigidas são uma fração pequena de ~176 mil.

**Campo (mesmas 13 fotos): caiu ainda mais — 73,3%.** Contra-intuitivo: corrigir um
bug real e bem documentado (quase 4 mil imagens) **não melhorou** o campo.

### 8.3 A análise que evitou uma conclusão errada

Em vez de aceitar "a correcção não funcionou" ou, pior, reverter de novo por
impulso, calculou-se o **intervalo de confiança de Wilson (95%)** para a acurácia
de campo de cada ciclo:

| Ciclo | Diagnósticos emitidos | Acertos | Proporção | IC 95% (Wilson) |
|---|---|---|---|---|
| 1 | 12/13 | 11 | 91,7% | **[64,6%, 98,5%]** |
| 2 | 13/13 | 10 | 76,9% | **[49,7%, 91,8%]** |
| 3 | 11/13 | 8 | 72,7% | **[43,4%, 90,3%]** |

> **Correcção (31/07/2026)**: a versão anterior desta tabela comparava
> ciclo 1 como 11/13 (abstenção contada como erro) com ciclo 3 como 8/11
> (abstenção excluída) — tratamentos **diferentes** para os dois ciclos, o que
> invalidava a comparação. Recalculado de forma uniforme, excluindo abstenções
> nos três ciclos. A conclusão não muda; fica mais forte, por cobrir os 3 ciclos.
> O IC incide sobre a proporção bruta de acertos entre diagnósticos emitidos,
> não sobre a acurácia balanceada (que é média de duas proporções e exigiria
> outro tratamento).

**Os três intervalos se sobrepõem na faixa 64,6%–90,3%.** Com N=13, os ciclos são
**estatisticamente indistinguíveis**. A queda de 87,5% para 73,3% (acurácia
balanceada) pode ser inteiramente ruído de amostra pequena — não uma piora real
do modelo. Nenhum intervalo tem amplitude inferior a 33pp.

**Esta é a conclusão central de toda a investigação de campo até agora**: o teste
de 13 fotos já cumpriu seu papel — expôs o bug de rotulagem (§7.4) — mas não tem
poder estatístico para **decidir entre versões de modelo**. Ampliar a amostra
deixou de ser "seria bom fazer" e passou a ser **pré-requisito** para qualquer
decisão futura.

Um padrão observado nos dados brutos (hipótese, não prova — o Grad-CAM gerado para
investigar ficou inconclusivo por limitação da própria visualização): os 2 erros
novos do ciclo 3 são danos **estruturais** (murcha, furos de praga) em folhas ainda
predominantemente verdes; os acertos entre as fotos doentes têm sinal de **cor**
claro (manchas escuras, revestimento esbranquiçado). Pode ser que o modelo dependa
mais de pistas de cor que de pistas estruturais — fica registado como hipótese para
testar quando houver mais dados.

**Decisão**: manter o ciclo 3 em produção — é a versão sem bugs conhecidos e com a
melhor média no domínio de teste. Ao contrário do ciclo 2, não há aqui uma causa
concreta para preferir uma versão anterior — só um número de campo estatisticamente
inconclusivo, que não é motivo válido para reverter.

> Detalhe técnico completo: `docs/CORRECOES_METODOLOGICAS.md` §10.7–§10.8.

---

## 9. Fase 6 — Ciclo 4: multi-cultura, e o primeiro ganho que passa num teste estatístico

*(22-23/09/2026)*

### 9.1 A decisão de treinar ANTES das fotos de campo

O §10.8 do documento metodológico dizia "não lançar o ciclo 4 antes de ampliar o
teste de campo". Essa regra foi revogada, com uma razão melhor do lado oposto: as
fotos de campo **só avaliam, nunca treinam** — é regra fixa do plano de análise.
Logo o treino nunca dependeu delas. E treinar antes traz uma vantagem: o modelo fica
**congelado antes de vermos as fotos novas**, o que torna a avaliação de campo
genuinamente cega e responde de antemão à crítica de ter ajustado o modelo olhando o
conjunto de teste.

### 9.2 O bug que apareceu ao preparar os dados: 1.142 imagens, 49 fotos

O lote de *downy mildew* aprovado na fase de validação de datasets tinha 1.142
ficheiros — que, ao conferir os nomes, revelaram-se **recortes de anotação e
variantes de apenas 49 fotografias originais**, com 45 delas espalhadas entre treino
e teste.

O detector de quase-duplicados por *hash* perceptual não apanha este caso, e a razão
importa: dois recortes da mesma foto **não são** visualmente duplicados — mostram
regiões diferentes. O critério correcto aqui não é semelhança, é **proveniência**.

> **Lição nova, a somar às outras**: *o agrupamento tem de reflectir como os dados
> foram produzidos, não só como se parecem.* Datasets de detecção convertidos em
> classificação trazem esta armadilha de origem.

O split foi refeito agrupado por foto. **No texto do TCC, este lote conta como 49
fotografias independentes, não 1.142.**

### 9.3 O que entrou e como

+4.351 imagens (3.209 de espinafre *Malabar*, 1.142 recortes de 49 fotos de alface
com *downy mildew*), acrescentadas aos splits existentes **sem refazer o split** —
assim o teste do ciclo 3 fica contido no do ciclo 4 e a comparação é justa.
Vazamento auditado: **0,04%** (veredicto OK).

### 9.4 O resultado

**No domínio antigo** (mesmas 23.967 imagens): empate — EfficientNet-B0 0,9848
contra 0,9849 do ciclo 3. Aprender culturas novas não custou o que já se sabia.

**Nas culturas novas** (657 imagens), com McNemar exacto pareado e correcção de Holm:

| Modelo | Ciclo 3 | Ciclo 4 | p (Holm) |
|---|---|---|---|
| EfficientNet-B0 | 0,811 | **0,903** | 7,1×10⁻¹⁴ |
| ResNet50 | 0,795 | **0,884** | 3,7×10⁻¹³ |
| MobileNetV2 | 0,758 | **0,901** | 3,5×10⁻¹¹ |

**Este é o primeiro ganho do projecto que sobrevive a um teste de significância.**

E o ciclo 3 não só errava mais nessas culturas — errava **com confiança alta**: ECE
de 0,175, contra 0,053 do ciclo 4.

### 9.5 O contraste que vale a defesa inteira

As mesmas 13 fotos de campo históricas, agora com o ciclo 4: acurácia balanceada sobe
de 73,3% para **83,8%** e o ECE cai de 0,274 para 0,146 — **mas o McNemar dá p=1,00**.

Ou seja: **os mesmos três modelos, comparados com N=657, dão p≈10⁻¹³; comparados com
N=13, dão p=1,00.** Não foi o modelo que mudou de qualidade entre as duas medições —
foi o instrumento de medida que tem, ou não tem, poder estatístico. É a demonstração
mais limpa que o projecto produziu de por que ampliar a amostra de campo era
bloqueante, e fecha o argumento aberto na §8.3.

> Detalhe técnico completo: `docs/CORRECOES_METODOLOGICAS.md` §10.9.
>
> **Ressalva de 08/10/2026:** os p da tabela 9.4 foram calculados sobre 657 imagens,
> das quais 173 são recortes de 5 fotos. Os números corrigidos estão na §9-B.3.

---

## 9-B. Fase 7 — Revisão crítica: o teste se parece com o treino

*(08/10/2026, preparação do Seminário 2)*

### 9-B.1 A leitura que faltava

Até aqui o resumo do projeto era "em laboratório o sistema é confiável; em campo
cai". Ao rever a apresentação, José apontou o que essa frase escondia: **o modelo
está amarrado aos dados de treino, e treino, validação e teste são muito
parecidos**. A verificação foi feita só com artefatos já existentes
(`backend/reanalise_teste_novo.py`), sem treino novo.

### 9-B.2 Três números, três situações

| Situação | Acurácia balanceada | N | O que mede |
|---|---|---|---|
| **Intra-fonte** — teste das mesmas 6 fontes do treino | 98,5% | 23.967 | Acerto em fotos parecidas com as do treino |
| **Fonte retida** — espinafre, que o ciclo 3 nunca viu | **75,4%** | 484 | Acerto fora das fontes, ainda em foto de estúdio |
| **Campo** — 13 fotos de alface (ciclo 4) | 83,8% [58–96] | 13 | Indício do uso real, sem poder estatístico |

- As seis fontes aparecem nas mesmas proporções nas três partições. ~~Cerca de 70% do
  treino e 72% do teste vêm de duas fontes derivadas do PlantVillage (folha isolada,
  fundo homogêneo).~~ **Corrigido em 13/10 (§9-E):** os 70% são a soma de duas
  *fontes*, e uma delas é uma coletânea com culturas que o PlantVillage não tem. O
  derivado do PlantVillage é no máximo 47,7% do treino, e 17,6% já são fotos de
  lavoura de outras culturas.
- O espinafre é foto de estúdio em fundo branco. Mesmo assim, só por ser de outra
  fonte (e de outra espécie), a acurácia cai 23 pontos. Os outros dois modelos
  ficam em 74,3% e 77,6%.
- O split por hash tira as duplicatas. **Não torna o teste diferente do treino.**

### 9-B.3 O ganho do ciclo 4, com a conta refeita

Dos 657 exemplos do "teste novo", 173 são recortes de apenas 5 fotos de míldio.
Recortes da mesma foto não são independentes, e o McNemar por imagem inflava o p.
Só com o espinafre (484 fotos independentes):

| Modelo | Ciclo 3 | Ciclo 4 | p (Holm) |
|---|---|---|---|
| EfficientNet-B0 | 0,754 | **0,884** | 5,9×10⁻¹² |
| ResNet50 | 0,776 | **0,862** | 1,9×10⁻⁷ |
| MobileNetV2 | 0,743 | **0,874** | 1,9×10⁻⁷ |

O ganho continua real e significativo (9 a 13 pontos). O que muda é a leitura:
**quando a fonte entra no treino, o teste dessa fonte sobe.** É um ganho
intra-fonte. Não diz nada sobre horta.

### 9-B.4 Em campo, a confiança não avisa

Nos quatro ciclos, os 9 erros emitidos nas 13 fotos de campo tiveram confiança
entre 0,949 e 0,999. (São 9 erros de **4 fotografias**; uma delas responde por 4 e
tem rótulo duvidoso — §9-E.5.) No ciclo 4, as duas fotos erradas saíram com 0,997 e 0,991 e
o sistema não se absteve nenhuma vez. O ECE é 0,0027 na validação e 0,146 em campo.
O limiar de abstenção é ajustado na validação, que é da mesma distribuição do
treino — fora dela, deixou de proteger.

### 9-B.5 O que é hipótese

A explicação para a queda em horta é o **fundo** (terra, sombra, folhas
sobrepostas contra o fundo limpo do treino). É a hipótese mais provável e a
literatura a apoia, mas o projeto ainda não a testou diretamente: o resultado do
espinafre mostra dependência da *fonte*, não isola o fundo.

> **Acréscimo de 13/10 (§9-E):** o fundo não é a única diferença. O treino já tem 17,6%
> de fotos de lavoura (de outras culturas), e nelas o modelo faz 86%. E a alface do
> treino é quase toda close-up da superfície da folha, enquanto a foto de horta mostra a
> planta inteira: o enquadramento também muda.

### 9-B.6 O que muda daqui para a frente

1. Chamar as coisas pelo nome: *intra-fonte*, *fonte retida*, *campo*.
2. Coletar fotos com fundo real em dois conjuntos separados por horta: avaliação
   (nunca treina) e adaptação.
3. Medir primeiro, às cegas, com o ciclo 4 congelado. Só depois treinar com fundo
   real (conjunto de adaptação + bases públicas de campo) e medir de novo.
4. Testar a hipótese do fundo removendo o fundo e removendo a folha.
5. Verificar calibração e abstenção em campo.

O item 3 muda a regra "fotos de campo nunca treinam" da Parcial I e **precisa do
aval do orientador**. Até lá vale a regra antiga.

> Detalhe técnico e registro consolidado dos 23 erros e ajustes do projeto:
> `docs/CORRECOES_METODOLOGICAS.md` §11.

---

## 9-C. Fase 8 — Comparação pareada entre arquiteturas e preparação do ViT

*12/10/2026. Detalhe: `docs/CORRECOES_METODOLOGICAS.md` §12.*

### 9-C.1 A pergunta Q1 ganhou resposta para as CNNs

Até aqui a comparação entre arquiteturas era um ponto estimado por modelo. Com as
previsões por imagem (`backend/compara_arquiteturas.py`), o teste passou a ser
pareado, nas 23.967 imagens do teste original:

| Modelo (ciclo 4) | Acur. bal. | IC 95% | Falsos negativos |
|---|---|---|---|
| EfficientNet-B0 | 0,9848 | [0,9832; 0,9864] | 209 |
| MobileNetV2 | 0,9789 | [0,9770; 0,9807] | 294 |
| ResNet-50 | 0,9782 | [0,9763; 0,9800] | 352 |

O EfficientNet-B0 difere dos outros dois (McNemar exato, p de Holm de 9,0×10⁻¹³ e
2,6×10⁻¹⁷). MobileNetV2 e ResNet-50 não diferem entre si (p = 0,13). Vale para este
teste e para uma execução de treino por modelo.

### 9-C.2 Uma frase que precisou de correção

"Aprender culturas novas não custou o que já se sabia" (§9.4) foi escrita olhando
o EfficientNet-B0. Comparando ciclo 3 e ciclo 4 foto a foto no teste original, o
EfficientNet-B0 e o ResNet-50 de fato empatam (p = 1,00), mas o **MobileNetV2 caiu de
0,9818 para 0,9789 (p de Holm = 0,015)**. A decisão de produção não muda, porque o
modelo servido é o EfficientNet-B0. O texto passa a dizer isso por modelo.

### 9-C.3 A abstenção, com a conta refeita

No teste original, o limiar τ = 0,85 segura 113 dos 363 erros do EfficientNet-B0
(31,1%), abstendo-se em 1,2% das imagens. Mais de dois terços dos erros saem com
confiança alta, mesmo dentro do domínio de treino. O número "138 de 404" que estava
no artigo v2 era de um ciclo antigo e foi substituído.

### 9-C.4 ViT-B/16: preparado, ainda não treinado

> Treinado em 12–13/10/2026. Resultado na §9-D.

O ViT-B/16 entra como quarta arquitetura, no mesmo dataset e split do ciclo 4,
treinado no Google Colab Pro. Dados, loss, seleção de modelo e calibração são os
das CNNs; mudam LR (5×10⁻⁵), weight decay (0,05), aquecimento (2 épocas) e clipping
do gradiente, que é a receita usual para transformers. Passo a passo em
`docs/GUIA_TREINO_VIT_COLAB.md`.

O que **não** esperar do ViT: ele não responde à questão central. O teste continua
intra-fonte, o espinafre está no treino, e 13 fotos de campo não distinguem
arquiteturas. O que responde à questão central continua sendo a coleta de campo.

---

## 9-D. Fase 9 — ViT-B/16 treinado: empata com o EfficientNet-B0

*13/10/2026. Tabelas completas: `backend/results/arquiteturas/comparacao_arquiteturas.md`.*

### 9-D.1 O treino

Google Colab Pro, GPU L4, de 12/10 19:27 a 13/10 01:43. Receita da §9-C.4, sem
alterações. As 30 épocas rodaram (a parada antecipada não disparou); melhor F1 macro
na validação de 0,9845, na época 29; 6,1 h de treino. Temperatura T = 0,620, com o ECE
na validação caindo de 0,047 para 0,002. Custo: 12,7 unidades de computação.

O treino não usou o notebook nem o Google Drive: foi automatizado pela CLI oficial
do Colab a partir do PC (`colab/cli/orq.sh`), com os dados conferidos por SHA-256 na
chegada.

### 9-D.2 Q1 com quatro arquiteturas

Teste original, 23.967 imagens:

| Modelo | Acur. bal. | IC 95% | FN | FP | ECE | Latência (RTX 3050) |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | 0,9848 | [0,9832; 0,9864] | 209 | 154 | 0,0039 | 7,2 ms |
| ViT-B/16 | 0,9848 | [0,9832; 0,9864] | 137 | 204 | 0,0018 | 19,4 ms |
| MobileNetV2 | 0,9789 | [0,9770; 0,9807] | 294 | 211 | 0,0055 | 5,3 ms |
| ResNet-50 | 0,9782 | [0,9763; 0,9800] | 352 | 184 | 0,0061 | 8,6 ms |

- EfficientNet-B0 × ViT-B/16: 158 imagens que só o primeiro acerta, 180 que só o
  segundo acerta; McNemar exato, p de Holm = 0,26. **Empate.**
- Os dois superam MobileNetV2 e ResNet-50 (p de Holm ≤ 1,4×10⁻¹²).
- Com seis comparações em vez de três, os p de Holm entre as CNNs mudaram em relação
  à §9-C.1: 1,4×10⁻¹² e 4,3×10⁻¹⁷ (eram 9,0×10⁻¹³ e 2,6×10⁻¹⁷); MobileNetV2 × ResNet-50
  passou de 0,13 para 0,26. Nenhuma conclusão mudou.
- As latências foram medidas para as quatro arquiteturas na mesma sessão, o que
  resolve a pendência da §12.6 de `CORRECOES_METODOLOGICAS.md`.

**Como dizer na banca:** *"Um transformer com 21 vezes mais parâmetros chega
exatamente ao mesmo lugar que o EfficientNet-B0 neste teste, e é 2,7 vezes mais lento.
A arquitetura não é o gargalo; os dados são."*

> **Correção de 13/10:** dizia-se "16 vezes", que é a razão entre as arquiteturas
> originais de 1.000 classes (86,6 M ÷ 5,3 M). Os modelos treinados, com saída de duas
> classes, têm 85,8 M e 4,0 M: 21 vezes (§9-E.6).

### 9-D.3 O que o ViT faz de diferente

- **Erra para o outro lado.** Menos falsos negativos (137 contra 209) e mais falsos
  positivos (204 contra 154). Para triagem, falso negativo é o erro caro, então o
  perfil é interessante; mas é medido em fotos de estúdio.
- **É confiante demais.** Só 51 dos 341 erros ficam abaixo de τ = 0,85 (15,0%, contra
  31,1% do EfficientNet-B0), com abstenção em 0,5% das imagens. A abstenção protege
  ainda menos.
- **No espinafre (N = 484)** fica em 0,844 [0,809; 0,877], sem diferença
  significativa para o EfficientNet-B0 (0,884; p de Holm = 0,51), e com o pior ECE
  dos quatro (0,11).
- **Nas 13 fotos de campo** acertou as 12 que diagnosticou e absteve-se em uma
  (Wilson: 75,8% a 100%). O intervalo se sobrepõe aos de todos os ciclos da §8: com
  N = 13 não se distingue arquitetura, como não se distinguia versão.

### 9-D.4 O que isso muda

Nada na decisão de produção: o modelo servido continua sendo o EfficientNet-B0, que
tem a mesma acurácia, é mais leve, mais rápido e mais bem interceptado pela abstenção.
O artigo passou para a v4 com esses resultados
(`docs/Entregas Finais/TCC2_Artigo_FitoVision_v4.docx`).

Ressalvas que acompanham os números: uma execução e uma semente por modelo;
hiperparâmetros de otimização próprios do ViT; teste intra-fonte; espinafre no treino.

---

## 9-E. Fase 10 — Auditoria de consistência: os números contra os artefatos

*13/10/2026. Fatos conferidos em `docs/FICHA_TECNICA_FITOVISION.md`; script
`backend/auditoria_consistencia.py`; detalhe em `docs/CORRECOES_METODOLOGICAS.md` §14.*

### 9-E.1 O que motivou

Ao preparar o guia da banca apareceram três inconsistências entre documentos: o tamanho
do dataset (176.126, 170.772 ou 175.123), o curso e a procedência das 13 fotos de campo.
Em vez de escolher um número, cada afirmação de dataset, arquitetura e método foi
conferida no artefato de origem: split em disco, logs, código de treino, pesos e
previsões por imagem. Nenhum treino e nenhuma inferência nova.

### 9-E.2 O achado principal: onde estão os erros do teste

As previsões por imagem já estavam salvas. Bastou reagrupá-las por fonte e por cultura.

| Subconjunto do teste original | N | EfficientNet-B0 | ViT-B/16 | MobileNetV2 | ResNet-50 |
|---|---|---|---|---|---|
| Coletânea, culturas fotografadas em lavoura (mandioca, arroz, cana, café) | 4.530 | **86,3%** [84,5–88,0] | 83,4% | 86,6% | 87,8% |
| Coletânea, demais culturas | 10.061 | **99,7%** [99,6–99,8] | 99,8% | 99,5% | 99,4% |
| Só mandioca | 3.150 | 80,9% | 75,7% | 81,3% | 82,8% |
| Fonte PlantVillage | 2.659 | 99,9% | 99,9% | 99,8% | 99,7% |
| Teste inteiro | 23.967 | 98,48% | 98,48% | 97,89% | 97,82% |

- **295 dos 363 erros do EfficientNet-B0 (81%) estão em 19% do teste.** Só a mandioca,
  que é foto de lavoura (conjunto Cassava Leaf Disease), responde por 271.
- **Dentro da mesma fonte, lavoura e estúdio ficam a 13 pontos um do outro.** É a mesma
  queda que a literatura descreve, medida no próprio teste, com N de milhares. Os 98,48%
  são uma média entre imagens fáceis e imagens difíceis.
- **Nas culturas de lavoura as arquiteturas quase não se distinguem**, e o ViT-B/16 tem
  a menor acurácia balanceada. Reforça "o limite está nos dados".
- Ressalva: "fotografada em lavoura" foi decidido por inspeção visual de amostras, e o
  efeito mistura condição de captura com qualidade dos rótulos dessas fontes.

### 9-E.3 O segundo achado: o detector de duplicatas não via as variantes de alface

A maior fonte de alface (`shuvokumarbasak2030`, 25% do treino) são 17 transformações de
cada uma de 1.231 fotos, com o id da foto original no nome do arquivo. O hash perceptual
junta espelhos e ajustes leves; não junta uma rotação de ângulo livre, um recorte ou um
filtro de bordas à foto de onde vieram. Resultado: **5.592 das 5.616 imagens de teste
dessa fonte (99,6%) têm outra variante da mesma foto no treino** — 23% do teste inteiro,
ou 26% contando as outras três fontes de alface, que derivam das mesmas fotos.

O "0,04% de vazamento" continua certo para o que mede. É o mesmo problema do lote de
míldio (§9.2), que tinha sido corrigido só para aquele lote. Sem essas variantes
(N = 18.351), a conclusão de Q1 se mantém, com valores de p mais modestos (de 1,7×10⁻⁴ a
1,8×10⁻⁸ em vez de 10⁻¹² a 10⁻¹⁷).

### 9-E.4 A composição do treino não era a que se dizia

- **"70% do treino deriva do PlantVillage" estava errado.** Os 70,1% somavam duas fontes,
  e uma delas (`nirmalsankalana`) é uma coletânea: 37% dela são culturas que o
  PlantVillage não tem. O derivado do PlantVillage é no máximo **47,7%**.
- **17,6% do treino já é foto de lavoura**, de mandioca, arroz, cana e café. O que falta
  é foto de lavoura de hortaliça folhosa.
- **Cultura e rótulo estão associados.** A alface é 29,9% do treino e 86,6% dela é
  saudável; nas demais culturas, 24,9%. Arroz, laranja e abóbora só têm o rótulo anômalo;
  soja, mirtilo e framboesa, só o saudável. É um atalho disponível para a rede, e é
  coerente com o erro mais frequente em campo: alface doente dada como saudável.
- **A alface do treino vem de 1.231 fotos originais, 108 delas doentes.**
- **Quatro pastas de dano por praga (7.779 imagens) ficaram fora do treino**, porque a
  rotulagem automática não reconhece "pest", "sawfly" nem "spider_mites". É o tipo de
  sintoma da hipótese H2.

### 9-E.5 As 13 fotos de campo, foto a foto

- **Os 9 erros dos quatro ciclos são de 4 fotografias.** `healthy/alface_04.jpg` erra nos
  quatro ciclos; `anomalous/alface_01.jpg`, em três.
- **`healthy/alface_04.jpg` tem rótulo duvidoso**: mostra uma folha com manchas escuras.
  Com o rótulo trocado, o ciclo 4 teria 12 de 13; os intervalos continuariam sobrepostos.
- **A temperatura ajustada na validação piora a calibração em campo**: ECE de 0,146 com
  ela e 0,094 sem ela, no ciclo 4. As temperaturas são todas menores que 1 (0,55 a 0,62),
  ou seja, aumentam a confiança.
- **Nenhuma das 13 é quase-duplicata de imagem do treino.** Mas nenhuma tem EXIF de
  celular, e uma é de câmera Sony, de 2014, editada no Photoshop. A procedência continua
  por registrar.

### 9-E.6 Correções de menor alcance

- **Parâmetros.** Os modelos treinados têm saída de duas classes: 4,0 M (EfficientNet-B0),
  2,2 M (MobileNetV2), 23,5 M (ResNet-50) e 85,8 M (ViT-B/16). Os 5,3 / 3,4 / 25,6 / 86,6 M
  são das arquiteturas originais. "16 vezes" passa a "21 vezes".
- **Tamanho do dataset.** 176.126 reunidas, 170.772 na base, 175.123 no ciclo 4.
- **Treino.** Lote de 64 (não 32); a parada antecipada não disparou nos ciclos 3 e 4.
- **Curso.** Engenharia de Software.

### 9-E.7 O que muda

O artigo passou para a v6 com essas correções. As correções de método — split por
proveniência, pastas de praga, balanceamento por cultura, calibração verificada fora das
fontes de treino — entram no ciclo 5 (`PROXIMOS_PASSOS_TCC2.md`).

**Como dizer na banca:** *"Quando reagrupei o teste por tipo de foto, o resultado que eu
procurava em campo já estava lá: 99,7% em foto de estúdio e 86% em foto de lavoura, na
mesma fonte. O 98,5% era a média dos dois."*

---

## 9-F. Fase 11 — Ciclo 5: o pipeline corrigido, nas quatro arquiteturas (14/10/2026)

*Detalhe em `docs/CORRECOES_METODOLOGICAS.md` §15; números em
`docs/FICHA_TECNICA_FITOVISION.md` §13; relatórios em `backend/results/ciclo5/`.*

### 9-F.1 O que mudou

Tudo o que a auditoria (§9-E) mandou corrigir, num split novo (`backend/data_c5/`, por
hardlinks; nada do ciclo 4 foi tocado):

- **Alface por fotografia original.** O id no nome do arquivo e o *hash* juntam as 17
  variantes de cada foto; as 1.231 fotos originais viram 1.172 unidades, repartidas
  inteiras. Validação e teste recebem **uma fotografia por original**: o teste honesto de
  alface tem 176 fotos, 15 doentes.
- **Pragas rotuladas.** 8.266 imagens de pastas que a rotulagem automática não reconhecia
  (quase todas de dano por praga) entram como anômalas.
- **Amostragem por cultura.** Cada cultura mantém a sua fatia, mas dentro dela saudável e
  anômala são sorteadas meio a meio; a perda deixa de ter pesos de classe.
- **Lote retido.** Um lote de espinafre-de-malabar (574 imagens) fica fora do treino e da
  validação. Outro lote da mesma espécie continua no treino — é uma troca de lote, não de
  espécie.
- **Mesma partição no resto.** A coletânea e o PlantVillage mantêm a partição do ciclo 4,
  o que permite comparar os ciclos foto a foto.

São 152.057 imagens. A auditoria completa de quase-duplicatas não terminava (o agrupador
compara par a par em Python); foi escrito um modo exato e vetorizado
(`audit_leakage.py --rapido`), conferido contra a força bruta, e ele deu **1,7% do teste**
com alguma imagem de treino a distância ≤ 4, contra 23% no ciclo 4 — boa parte colisões
entre culturas diferentes, não a mesma foto.

### 9-F.2 O treino

As quatro arquiteturas, no Google Colab (GPU L4), pela CLI (`colab/cli_c5/`), de 06:06 às
17:52 de 14/10, cerca de 22 unidades de computação. Mesmas receitas do ciclo 4 (CNNs) e do
ViT de 12–13/10. O envio dos 9,5 GB falhou na primeira tentativa (o Colab recolheu o
runtime no meio do envio) e passou na segunda, com uma célula de manutenção rodando. As
previsões refeitas no PC reproduziram as do Colab.

### 9-F.3 O resultado

| Modelo | F1 val. | Teste | Lavoura: c4 → c5 | Especificidade na lavoura | Alface (176) | Lote retido (574) |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | 0,972 | 97,4 | 86,3 → 88,8 | 76,6 → 83,4 | 96,7 | 87,9 |
| MobileNetV2 | 0,960 | 96,9 | 86,6 → 90,1 | 78,3 → 90,2 | 100 | 88,4 |
| ResNet-50 | 0,961 | 96,9 | 87,8 → 89,6 | 81,4 → 89,0 | 100 | 90,1 |
| ViT-B/16 | 0,972 | 96,9 | 83,4 → 84,5 | 69,8 → 72,1 | 100 | 85,8 |

1. **A correção agiu onde estava o erro.** Nas culturas fotografadas em lavoura, as três
   CNNs ganham de 1,8 a 3,5 pontos de acurácia balanceada e de 6,8 a 11,9 de
   especificidade: o atalho "foto de lavoura = doente" enfraqueceu.
2. **Com custo.** A sensibilidade nesse subconjunto cai de 1,8 a 4,9 pontos, e, contado
   imagem a imagem, o ciclo 4 acerta mais (McNemar p ≤ 0,04), porque três em cada quatro
   dessas imagens são anômalas.
3. **O ViT quase não muda** (p = 0,66) e é o pior na lavoura nos dois ciclos, embora
   empate com o EfficientNet-B0 na validação.
4. **A validação não ordena os modelos pelo que importa:** põe EfficientNet-B0 e ViT na
   frente (0,972 contra 0,96), e na lavoura e no lote retido a ordem se inverte.
5. **Lote retido: 85,8% a 90,1%**, cerca de dez pontos abaixo das fontes vistas.
6. **Campo histórico (13 fotos):** EfficientNet-B0 do ciclo 5 com 11 acertos em 12
   diagnósticos e uma abstenção; contra o ciclo 4, 1 a 1 (p = 1). Indecidível, como sempre
   com N = 13.

### 9-F.4 O que muda

O artigo passou para a v7, com o método do ciclo 5, a Tabela 3 e a discussão da troca de
sensibilidade por especificidade, em 11 páginas. O modelo servido continua o
EfficientNet-B0 do ciclo 4: a decisão espera as fotos de campo, que o `roda_campo.py`
avalia às cegas com o ciclo 4, o ViT, o EfficientNet-B0 do ciclo 5 e o ViT do ciclo 5.

**Como dizer na banca:** *"O ciclo 5 não ganhou em tudo, e eu não vou fingir que ganhou.
Ele erra menos em folha sadia de lavoura e mais em folha doente. Qual dos dois serve
melhor depende do custo de cada erro na horta — e é por isso que eu reporto
especificidade e sensibilidade, e não só uma acurácia."*

---

## 10. Estado atual — tudo num só lugar

### 10.1 Modelo em produção

**EfficientNet-B0, ciclo 4** — pesos em `backend/weights/efficientnet_b0_binary.pth`,
calibração em `backend/weights/efficientnet_b0_binary_calibration.json` (τ=0,85,
ECE 0,0027 no val). Os pesos do ciclo 3 estão preservados em
`backend/weights/ciclo3/`.

⚠ **A escolha formal entre ciclo 3 e ciclo 4 ainda não foi feita.** O ciclo 4 é
igual no domínio antigo, muito melhor nas culturas novas e melhor calibrado — mas a
decisão espera as fotos de campo novas, para não ser tomada só com dados de dataset.

**Ciclo 5 (14/10/2026):** as quatro arquiteturas estão em `backend/weights/ciclo5/`, sem
substituir o modelo servido. O candidato é o EfficientNet-B0 do ciclo 5, se a horta
mostrar o mesmo padrão da lavoura (falsos alarmes em folha sadia); a troca de
sensibilidade tem de ser dita (§9-F).

### 10.2 Tabela comparativa completa — os 4 ciclos, domínio de teste

| Modelo | Ciclo 1 | Ciclo 2 — rótulos errados | Ciclo 3 | **Ciclo 4 — activo** |
|---|---|---|---|---|
| EfficientNet-B0 | 0,9859 / 0,9835 | 0,9842 / 0,9803 | 0,9849 / 0,9844 | **0,9848 / 0,9844** |
| ResNet50 | 0,9789 / 0,9724 | 0,9761 / 0,9690 | 0,9773 / 0,9766 | **0,9782 / 0,9770** |
| MobileNetV2 | 0,9723 / 0,9662 | 0,9767 / 0,9708 | 0,9818 / 0,9805 | **0,9789 / 0,9783** |

*(bal.acc / F1 macro; o ciclo 4 medido nas MESMAS 23.967 imagens do ciclo 3, para a
comparação ser justa)*

Ciclo 3 × ciclo 4 com teste pareado (§9-C.2): EfficientNet-B0 e ResNet-50 empatam
(p = 1,00); o MobileNetV2 caiu com significância (p de Holm = 0,015). Entre
arquiteturas no ciclo 4: EfficientNet-B0 acima das outras duas (p < 10⁻¹²).

### 10.2b Culturas novas — onde o ciclo 4 se justifica

484 imagens de espinafre, McNemar exacto pareado + Holm (os 173 recortes de míldio
ficam fora por virem de só 5 fotos — ver §9-B.3):

| Modelo | Ciclo 3 (bal.acc) | Ciclo 4 (bal.acc) | p (Holm) |
|---|---|---|---|
| EfficientNet-B0 | 0,754 | **0,884** | 5,9×10⁻¹² |
| ResNet50 | 0,776 | **0,862** | 1,9×10⁻⁷ |
| MobileNetV2 | 0,743 | **0,874** | 1,9×10⁻⁷ |

### 10.3 Tabela comparativa completa — teste de campo (mesmas 13 fotos, sempre)

| Ciclo | Acurácia de campo | Abstenções | Observação |
|---|---|---|---|
| 1 (baseline, split limpo) | 87,5% | 1/13 | primeiro dado real do projeto |
| 2 (dataset expandido, rótulos com bug) | 77,5% | 0/13 | revertido — causa concreta (rótulos errados) |
| 3 (rótulos corrigidos) | 73,3% | 2/13 | mantido — diferença não é estatisticamente significativa (§8.3) |
| **4 (espinafre + downy mildew)** | **83,8%** | 0/13 | melhor ponto estimado e melhor calibração (ECE 0,274 → 0,146), mas p=1,00 — com N=13 continua indecidível (§9.5) |

### 10.4 Números-chave para citar de cor na defesa

- **Vazamento original**: 57,35% do teste antigo estava contaminado
  (`leakage_report_old_split.json`).
- **Vazamento depois da correcção**: 0,18% (ciclo 1) → 0,04% (ciclo 3) — sempre
  abaixo do ruído esperado de qualquer método de deduplicação.
- **Acurácia retratada**: 99,01% (inválida, vazamento).
- **Acurácia intra-fonte** (teste das mesmas fontes do treino): ~98,5%
  (EfficientNet-B0, ciclos 3 e 4). Não chamar de "desempenho de laboratório".
- **Quatro arquiteturas** (§9-D): ViT-B/16 e EfficientNet-B0 empatam em 98,48%
  (McNemar, p de Holm = 0,26) e superam MobileNetV2 e ResNet-50 (p ≤ 1,4×10⁻¹²). O ViT
  tem 21 vezes mais parâmetros (85,8 M contra 4,0 M nos modelos treinados), é 2,7 vezes
  mais lento e só 15% dos seus erros são interceptados pela abstenção (31% no
  EfficientNet-B0).
- **Por subconjunto do teste** (§9-E.2): 86,3% nas culturas fotografadas em lavoura
  (N = 4.530), onde estão 81% dos erros, contra 99,7% nas demais culturas da mesma
  fonte (N = 10.061).
- **Proveniência** (§9-E.3): 23% do teste são variantes transformadas de fotos de
  alface com outra variante no treino (26% com as outras fontes de alface).
- **Composição do treino** (§9-E.4): no máximo 47,7% derivado do PlantVillage; 17,6%
  de fotos de lavoura de outras culturas; 29,9% de alface, 86,6% dela saudável.
- **Fonte nunca vista** (espinafre, ciclo 3, N = 484): 75,4% — queda de 23 pontos
  mesmo em foto de estúdio.
- **Ganho do ciclo 4 no espinafre**: +9 a +13 pontos de acurácia balanceada, p de
  Holm entre 5,9×10⁻¹² e 1,9×10⁻⁷ (N = 484) — o único ganho do projecto
  estatisticamente significativo. É um ganho intra-fonte. O contraste com o p=1,00
  das 13 fotos de campo continua a ser a melhor ilustração de poder estatístico.
- **Confiança em campo**: os 9 erros emitidos nos 4 ciclos (de 4 fotografias) tiveram
  confiança entre 0,949 e 0,999; no ciclo 4, zero abstenções em 13 fotos.
- **Acurácia honesta, campo real**: 73-88% (N=13, intervalo estatístico amplo —
  não um número único fixo, e isso **é** o resultado, não uma imprecisão a esconder).
- **Bug de rotulagem corrigido**: ~3.972 imagens de alface saudável reclassificadas
  corretamente, mais 3.831 imagens de erva daninha removidas da classe `anomalous`.
- **Duração dos pipelines, 4 ciclos** (hash, split, treino, calibração e avaliação):
  14,5h (ciclo 1, execução retomada) + 17,5h (ciclo 2) + 22,2h (ciclo 3) + 22,8h
  (ciclo 4) ≈ **77 horas**, das quais 20,7h de treino no ciclo 4. O ViT levou 6,1h no
  Colab.
- **Dataset, três números, três etapas**: 176.126 reunidas no staging do ciclo 3;
  170.772 nas partições da base; 175.123 com os dois lotes do ciclo 4.
- **Dataset final (ciclo 4)**: 175.123 imagens, 101.987 fotos distintas, 0,04% de
  vazamento auditado.
- **Ciclo 5** (§9-F): 152.057 imagens, 1,7% do teste com quase-duplicata no treino.
  Lavoura (EfficientNet-B0): 86,3% → 88,8%, especificidade 76,6% → 83,4%,
  sensibilidade 96,0% → 94,2%. Nas três CNNs: especificidade +6,8 a +11,9,
  sensibilidade −1,8 a −4,9. Lote de espinafre retido: 85,8% a 90,1% nos quatro
  modelos. Alface honesta: 96,7% a 100% em 176 fotos (15 doentes).

---

### 10.5 Campo com fundo real — primeiro lote (27/10/2026)

Dez fotos numa horta, alface crespa, todas sadias, tiradas de cima com vários pés por
quadro. Às cegas, as quatro versões (EfficientNet-B0 e ViT-B/16 dos ciclos 4 e 5) dão as
10 como sadias, com confiança ≥ 0,98 (Wilson 72,2%–100%), e o resultado se mantém em
recortes centrais de ½ e ¼ do lado. O lote de casa de 18/10 (um pé colhido, toalha branca;
não é campo) sai inteiro como anômalo, inclusive as 3 folhas sadias. Leitura: fundo real
não gera falso alarme; a sensibilidade em campo e o *domain gap* continuam sem medida,
porque não houve folha doente na horta. Detalhe em CORRECOES §15.13.

## 11. As lições metodológicas

*(a colecção completa, juntando `docs/CORRECOES_METODOLOGICAS.md` §9 com o que se
aprendeu nos ciclos 2, 3 e 4)*

1. **Métricas boas demais são uma hipótese, não um resultado.** Perante 99%,
   perguntar *"o que o conjunto de teste tem que não devia ter?"* antes de comemorar.

2. **Concordância entre validação e teste não prova generalização** — se ambos
   vêm da mesma fonte contaminada, só prova que a contaminação é uniforme.

3. **Correcções também têm de ser testadas.** O primeiro hash perceptual escrito
   para resolver o vazamento tinha ele próprio um defeito, só encontrado porque um
   teste sintético falhou.

4. **Uma métrica de domínio estável não garante nada sobre o domínio real.** O
   teste de campo, mesmo pequeno, foi o único sinal que expôs tanto o gap de
   domínio quanto o bug de rotulagem — nenhum dos dois aparecia no teste de estúdio.

5. **A causa de um sintoma nem sempre é o que se está a mexer no momento.** O bug
   de rotulagem estava dormente desde o *primeiro* commit do dataset — só ficou
   visível quando o teste de campo finalmente existiu para o expor.

6. **"Mais dados" não é uma correcção — é uma hipótese que precisa de teste.**
   Adicionar 10 mil imagens de alface (ciclo 2) não melhorou o campo; corrigir a
   rotulagem de 4 mil imagens (ciclo 3) também não, no ponto estimado. Isso não
   invalida as correcções (ambas eram tecnicamente correctas e necessárias) — mostra
   que o instrumento de medição (N=13) não tinha resolução para as captar.

7. **Antes de comparar dois modelos, pergunte se o teste tem poder estatístico
   para distinguir entre eles.** Um intervalo de confiança de Wilson levou 2
   minutos para calcular e evitou uma conclusão errada (reverter o ciclo 3 por um
   número que, matematicamente, podia ser ruído). O ciclo 4 fechou a demonstração:
   os mesmos modelos dão p≈10⁻¹³ com N=657 e p=1,00 com N=13.

8. **O agrupamento tem de reflectir como os dados foram produzidos, não só como se
   parecem.** O dedup por semelhança visual não apanha recortes diferentes da mesma
   fotografia — e foi assim que 1.142 ficheiros de *downy mildew*, vindos de apenas
   49 fotos, quase entraram com vazamento entre treino e teste (§9.2). Datasets de
   detecção convertidos em classificação trazem esta armadilha de origem.

9. **Uma regra metodológica boa pode ser revogada por uma razão melhor — desde que
   a revogação fique escrita.** "Não treinar antes de ter mais campo" foi substituída
   por "treinar antes, para que a avaliação de campo seja cega" (§9.1), e as duas
   decisões estão documentadas com a sua justificação.

10. **Um teste sem vazamento ainda é um teste da mesma distribuição.** Tirar as
    duplicatas tornou os 98,5% verdadeiros, mas eles medem acerto em fotos das
    mesmas seis fontes do treino. A medida de generalização veio de outro lado: uma
    fonte inteira que o modelo nunca tinha visto (75,4%) e a horta (§9-B.2).

11. **Antes de aplicar um teste estatístico, pergunte quantas observações
    independentes existem.** 173 recortes de 5 fotos não são 173 observações. O
    p≈10⁻¹³ do ciclo 4 estava inflado; refeito só com fotos independentes, o ganho
    continua significativo, com p mais modesto (§9-B.3).

12. **Calibração vale para a distribuição em que foi ajustada.** O limiar de
    abstenção funciona na validação e não segurou nenhum dos erros de campo, que
    saíram com 95% a 99% de confiança (§9-B.4). Confiança tem de ser verificada no
    domínio de uso.

13. **Uma média esconde onde o modelo erra.** Os 98,48% juntavam 99,7% em foto de
    estúdio e 86% em foto de lavoura. As previsões por imagem estavam salvas havia dias;
    faltava reagrupá-las por fonte e por cultura (§9-E.2).

14. **Uma lição aprendida num lote tem de ser aplicada ao dataset inteiro.** O split por
    proveniência foi feito para as 49 fotos de míldio (lição 8) e não para a fonte de
    alface com 17 transformações por foto, que é 25% do treino (§9-E.3).

15. **Conferir o que uma fonte contém antes de lhe dar um nome.** "Re-upload do
    PlantVillage" era uma coletânea com mandioca, arroz e café fotografados em lavoura.
    O nome errado virou um número errado (70%) e uma explicação errada (§9-E.4).

16. **Auditar os documentos contra os artefatos, não uns contra os outros.** Três
    documentos concordavam em "16 vezes mais parâmetros" e em "70%"; os dois estavam
    errados. O que desempata é o código, o split em disco e os arquivos de resultado.

17. **Corrigir um atalho muda o ponto de operação, não só a acurácia.** Tirar a
    associação entre cultura e rótulo reduziu os falsos alarmes e aumentou as doenças
    perdidas (§9-F.3). Uma métrica só (acurácia por imagem, ou balanceada) teria contado
    metade da história; especificidade e sensibilidade, lado a lado, contam a inteira.

18. **A ordem na validação não é a ordem fora dela.** No ciclo 5, os dois modelos com o
    maior F1 de validação são os dois piores na lavoura (§9-F.3). Escolher modelo pela
    validação intra-fonte é escolher pelo que menos importa.

---

## 12. Trabalho futuro / próximos passos

Em ordem de prioridade real (não de facilidade):

0. **Revisão de 08/10/2026 (§9-B.6):** a coleta passa a ter dois destinos,
   separados por horta — avaliação (nunca treina) e adaptação (fundo real no
   treino, só depois da medição cega e com o aval do orientador). Acrescentam-se o
   teste de dependência do fundo e a verificação da calibração em campo.

0-B. ~~ViT-B/16 no Colab Pro~~ — **feito em 12–13/10/2026 (§9-D).** O mesmo caminho
   (`colab/cli/orq.sh`) serve para o retreino com as fotos de campo.

0-C. ~~Correções saídas da auditoria~~ — **feitas no ciclo 5, em 14/10/2026 (§9-F).**
   O que segue continua valendo como descrição do que foi feito: split agrupado
   também por proveniência (id da foto original); fora do teste as variantes que não
   são fotografias (`sobel`, `laplacian`, `high_pass`); pastas de dano por praga
   rotuladas como anômalas; amostragem equilibrada por cultura e rótulo; resultado
   reportado por subconjunto; temperatura e limiar verificados fora das fontes de treino.

1. **Ampliar o teste de campo — bloqueante.** Passo a passo prático em
   `docs/GUIA_CAMPO_HORTAS_COMUNITARIAS.md`. Meta imediata: 20-40 fotos novas
   (nunca reaproveitar as 13 já usadas como teste — viraria vazamento entre "campo"
   e "campo"). Meta do TCC-II: 100-300 fotos. Priorizar cobrir tanto dano
   estrutural (murcha, furos) quanto dano com sinal de cor (manchas, clorose), para
   poder testar a hipótese da §8.3.

2. ~~**Não lançar um ciclo 4 de retreino antes do passo 1.**~~ **Revogado em
   22/09/2026** (§9.1): como as fotos de campo só avaliam e nunca treinam, o ciclo 4
   não dependia delas — e treinar antes deixou o modelo congelado antes de vermos as
   fotos novas, tornando a avaliação de campo genuinamente cega. O que **continua**
   a depender do passo 1 é a decisão de qual ciclo vai para produção.

3. **Consertar a visualização de Grad-CAM** antes de reusar para diagnóstico — o
   *script* usado nesta sessão gerou um mapa de calor ilegível (concentrado na
   legenda de cores, não na folha).

4. **Registar a lacuna de rúcula/espinafre/acelga/couve** como limitação formal e
   trabalho futuro — não existe, à data, dataset público de doenças para estas
   culturas.

5. **Se o teste de campo ampliado confirmar a hipótese cor-vs-estrutura** (§8.3):
   considerar um próximo ciclo de dados focado especificamente em exemplos de dano
   por praga/murcha, que já estão sub-representados frente a doenças
   fúngicas/bacterianas (que dominam os datasets de disponíveis).

6. **Regra de ouro a nunca quebrar**: fotos de campo do TESTE nunca entram no
   TREINO. Violar isso reintroduziria o mesmo tipo de vazamento que motivou toda
   esta investigação — desta vez entre "campo" e "campo".

---

## 13. Glossário rápido

| Termo | Significado |
|---|---|
| **Vazamento de dados (*data leakage*)** | Quando a mesma informação (aqui, a mesma fotografia) aparece em treino e teste, inflando artificialmente a métrica. |
| **Acurácia balanceada** | Média do acerto em cada classe, separadamente. Não é enganada por classes desbalanceadas (ao contrário da *accuracy* simples). |
| **F1 macro** | Média harmónica de precisão e *recall*, calculada por classe e depois com média simples entre classes — pondera as duas classes igualmente. |
| **ECE (*Expected Calibration Error*)** | Mede se a confiança reportada bate com o acerto real. ECE=0 é perfeito; ECE alto significa que o modelo "mente" sobre a própria certeza. |
| **Temperature Scaling** | Técnica de calibração: divide os *logits* por um escalar T antes do *softmax*, sem alterar a predição, só a confiança relatada. |
| **Abstenção (`inconclusive`)** | Quando a confiança calibrada fica abaixo de um limiar, o sistema recusa diagnosticar em vez de arriscar. |
| **Guarda de vegetação (ExG)** | Verificação prévia por índice de cor (*Excess Green*) que rejeita imagens sem vegetação suficiente (`not_a_leaf`) antes mesmo de chamar o modelo. |
| **Gap de domínio** | Diferença de desempenho entre o domínio de treino (estúdio, PlantVillage) e o domínio real de uso (campo, telemóvel). |
| **Acurácia intra-fonte** | Acurácia medida num teste tirado das mesmas fontes do treino. Sem vazamento é uma métrica válida, mas não mede generalização. |
| **Fonte retida** | Fonte de dados inteira que o modelo nunca viu no treino, usada só para avaliar. É a forma barata de medir generalização sem ir a campo. |
| **Pseudo-replicação** | Tratar como independentes observações que vêm da mesma origem (ex.: vários recortes da mesma foto). Infla a significância estatística. |
| **Split agrupado** | Divisão treino/val/teste que mantém grupos de fotos quase-idênticas (por hash perceptual) sempre no mesmo lado, evitando vazamento. |
| **Intervalo de confiança de Wilson** | Método estatístico para estimar a faixa provável de uma proporção (aqui, acurácia) a partir de uma amostra pequena — mais robusto que a aproximação normal simples quando N é pequeno. |

---

## 14. Índice de artefactos (onde está cada prova)

| O quê | Onde |
|---|---|
| Prova do vazamento original (57,35%) | `backend/results/leakage_report_old_split.json` |
| Prova do split limpo, ciclo 1 (0,18%) | `backend/results/pre_lettuce_expansion/` *(backup)* |
| Prova do split limpo, ciclo 3 (0,04%) | `backend/results/leakage_report.json` |
| Métricas de teste — ciclo 1 | `backend/results/pre_lettuce_expansion/metrics_comparison.csv` |
| Métricas de teste — ciclo 2 (rótulos com bug) | `backend/results/ciclo2_labelbug/metrics_comparison.csv` |
| Métricas de teste — ciclo 3 (activo) | `backend/results/metrics_comparison.csv` |
| Teste de campo — ciclo 1 | `backend/results/campo_resultados.csv` |
| Teste de campo — ciclo 2 | `backend/results/campo_resultados_v2.csv` |
| Teste de campo — ciclo 3 | `backend/results/campo_resultados_v3.csv` |
| Teste de campo — ciclo 4 | `backend/results/campo_resultados_v4.csv` |
| Ciclo 4: teste original, teste novo, comparação pareada | `backend/results/ciclo4/` |
| Comparação pareada entre arquiteturas (12/10): previsões por imagem, IC, McNemar | `backend/results/arquiteturas/` (gerado por `backend/compara_arquiteturas.py`) |
| Ciclo 3 × ciclo 4 no teste original, pareado | `backend/results/arquiteturas/ciclo3_vs_ciclo4_teste_original.json` |
| ViT-B/16 (treinado em 12–13/10/2026) | `backend/weights/vit/`, `backend/results/vit/`, `backend/logs/vit/` |
| Comparação das quatro arquiteturas | `backend/results/arquiteturas/comparacao_arquiteturas.md` |
| Latência das quatro arquiteturas, mesma sessão | `backend/results/latencia_13out/cnns/`, `backend/results/vit/local_teste_original/` |
| Automação do Colab pela CLI | `colab/cli/` |
| Guia do treino do ViT no Colab | `docs/GUIA_TREINO_VIT_COLAB.md` |
| Artigo do TCC-II, v4 (13/10, com o ViT; a que o orientador comentou) | `docs/Entregas Finais/TCC2_Artigo_FitoVision_v4.docx` |
| Reanálise de 08/10 (fontes por partição, ganho por lote, confiança em campo) | `backend/results/ciclo4/reanalise_por_lote.json` (gerado por `backend/reanalise_teste_novo.py`) |
| Fotos de campo (gabarito) | `backend/campo/{healthy,anomalous,nao_folha}/` |
| Mapa de rotulagem auditável (proveniência pasta→rótulo) | `backend/data/label_map_audit.json` |
| Diagramas de fiabilidade (calibração) | `backend/results/calibration_*.png` |
| Matrizes de confusão | `backend/results/cm_*.png` |
| Grad-CAM dos erros de campo (inconclusivo, ver §8.3) | `backend/results/campo_anomalous_alface_0{1,5}_gradcam.png` |
| Investigação técnica completa do vazamento e ciclos 1-3 | `docs/CORRECOES_METODOLOGICAS.md` |
| Guia de metodologia de teste de campo | `docs/GUIA_TESTE_DE_CAMPO.md` |
| **Ficha técnica: números e método conferidos nos artefatos (13/10)** | `docs/FICHA_TECNICA_FITOVISION.md` |
| Auditoria de consistência: script e resultados | `backend/auditoria_consistencia.py`, `backend/results/auditoria/auditoria_13out.{json,md}` |
| Guia de preparação para a banca e roteiro da apresentação | `docs/GUIA_BANCA_TCC2.md` |
| Revisão da literatura na Web of Science (três rodadas) | `docs/REVISAO_LITERATURA_WOS.md` |
| Artigo do TCC-II, versão atual (substitui as linhas anteriores) | `docs/Entregas Finais/TCC2_Artigo_FitoVision_v11.docx` (27/10, 10 páginas a pedido do orientador; a v10 tem o mesmo conteúdo em 11); a v9 (27/10, conclusão reescrita) foi a entregue na "Postagem IV - parcial II"; as anteriores ficam ao lado; geradores em `docs/Entregas Finais/_gerador/build_v9.py`, `build_v10.py` e `build_v11.py` |
| Monografia, versão atual | `docs/Entregas Finais/TCC_FitoVision_ATUALIZADO_v6.docx` (27/10, seção 9.7); gerador `build_mono_v6.py` |
| Campo com fundo real, primeiro lote (27/10) | `backend/campo_novo/` (fotos e `metadados_campo.csv`), `backend/results/campo_novo/` (`analise_campo.md`, `calibracao_e_hortas.md`, `recortes/`); scripts `confere_fotos_campo.py`, `roda_campo.py`, `testa_recortes_campo.py`; leitura em `docs/CORRECOES_METODOLOGICAS.md` §15.13 |
| Leitura do texto completo da literatura (16/10) | `docs/REVISAO_LITERATURA_WOS.md`, Parte 4 |
| Ciclo 5: split, auditoria, pesos | `backend/data_c5/split_metadata.json`, `backend/results/ciclo5/leakage_report.json`, `backend/weights/ciclo5/` |
| Ciclo 5: avaliação por subconjunto e ciclo 4 × ciclo 5 | `backend/results/ciclo5/avaliacao_ciclo5.md`, `backend/results/ciclo5/ciclo4_x_ciclo5.md` (de `avalia_ciclo5.py` e `compara_c4_c5.py`) |
| Ciclo 5: treino no Colab (scripts, log, pacote de retorno) | `colab/cli_c5/`, `backend/logs/ciclo5/execucao_colab.log`, `colab_export_c5/retorno/c5_resultados.zip` |
| Campo histórico com os modelos do ciclo 5 | `backend/results/campo_historico_c5/` |
| Este documento (narrativa completa para a banca) | `docs/HISTORICO_COMPLETO_TCC.md` |

---

> Documento gerado em 29/07/2026, consolidando o trabalho de todas as sessões até
> aqui. Deve ser actualizado a cada novo ciclo ou achado relevante — é o documento
> de referência para a escrita do capítulo de metodologia/resultados e para a
> apresentação oral.
