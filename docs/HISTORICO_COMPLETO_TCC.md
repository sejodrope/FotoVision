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
9. [Estado atual — tudo num só lugar](#9-estado-atual--tudo-num-só-lugar)
10. [As lições metodológicas](#10-as-lições-metodológicas)
11. [Trabalho futuro / próximos passos](#11-trabalho-futuro--próximos-passos)
12. [Glossário rápido](#12-glossário-rápido)
13. [Índice de artefactos (onde está cada prova)](#13-índice-de-artefactos-onde-está-cada-prova)

---

## 1. O projeto em uma página

**FitoVision** é o TCC de José Pedro (Ciência da Computação, UNIVILLE): um sistema
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

---

## 3. Fase 0 — Construção do dataset

### O problema de partida

Não existe dataset público robusto de **hortaliças folhosas brasileiras**
especificamente (alface, rúcula, espinafre, acelga, couve). Decisão: usar
**PlantVillage** (54.306 imagens, 38 culturas, majoritariamente tomate/batata/milho)
como **proxy visual** — o modelo aprende a reconhecer *padrões visuais de doença em
folhas* de outras culturas e generaliza para as culturas-alvo — complementado por
datasets específicos de alface onde disponíveis.

### Datasets usados (dataset original, antes desta sessão)

| Fonte | Imagens | Papel |
|---|---|---|
| `abdallahalidev/plantvillage-dataset` (Kaggle) | 54.306 | Proxy visual — 15 pastas relevantes filtradas de 38 culturas |
| `ashishjstar/lettuce-diseases` (Kaggle) | 2.337 | **Única fonte específica de alface** até o ciclo 2 |
| `shuvokumarbasak2030/lettuce-disease-multi-transformation-dataset` (Kaggle) | 79.458 | Alface com augmentation pré-aplicado |
| `nirmalsankalana/plant-diseases-training-dataset` (Kaggle) | 116.147 | PlantVillage expandido, 70+ classes |
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

José reuniu 20 fotos reais de alface (horta, luz natural, terra visível). Triadas
visualmente: **13 usadas como gabarito** (5 healthy, 8 anomalous — as mesmas 13
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

Split refeito: 176.126 imagens, vazamento 0,04% (ainda melhor que o ciclo 1).

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

### 10.2 Tabela comparativa completa — os 4 ciclos, domínio de teste

| Modelo | Ciclo 1 | Ciclo 2 — rótulos errados | Ciclo 3 | **Ciclo 4 — activo** |
|---|---|---|---|---|
| EfficientNet-B0 | 0,9859 / 0,9835 | 0,9842 / 0,9803 | 0,9849 / 0,9844 | **0,9848 / 0,9844** |
| ResNet50 | 0,9789 / 0,9724 | 0,9761 / 0,9690 | 0,9773 / 0,9766 | **0,9782 / 0,9770** |
| MobileNetV2 | 0,9723 / 0,9662 | 0,9767 / 0,9708 | 0,9818 / 0,9805 | **0,9789 / 0,9783** |

*(bal.acc / F1 macro; o ciclo 4 medido nas MESMAS 23.967 imagens do ciclo 3, para a
comparação ser justa)*

### 10.2b Culturas novas — onde o ciclo 4 se justifica

657 imagens de espinafre e *downy mildew*, McNemar exacto pareado + Holm:

| Modelo | Ciclo 3 (bal.acc) | Ciclo 4 (bal.acc) | p (Holm) |
|---|---|---|---|
| EfficientNet-B0 | 0,811 | **0,903** | 7,1×10⁻¹⁴ |
| ResNet50 | 0,795 | **0,884** | 3,7×10⁻¹³ |
| MobileNetV2 | 0,758 | **0,901** | 3,5×10⁻¹¹ |

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
- **Acurácia honesta, domínio de teste**: ~98,5% (EfficientNet-B0, ciclos 3 e 4).
- **Ganho nas culturas novas (ciclo 4)**: +9 a +14 pontos de acurácia balanceada,
  p de Holm entre 10⁻¹¹ e 10⁻¹⁴ — o único ganho do projecto estatisticamente
  significativo, e o contraste com o p=1,00 das 13 fotos de campo é a melhor
  ilustração do que é poder estatístico.
- **Acurácia honesta, campo real**: 73-88% (N=13, intervalo estatístico amplo —
  não um número único fixo, e isso **é** o resultado, não uma imprecisão a esconder).
- **Bug de rotulagem corrigido**: ~3.972 imagens de alface saudável reclassificadas
  corretamente, mais 3.831 imagens de erva daninha removidas da classe `anomalous`.
- **Duração total de retreino, 4 ciclos**: ~17h (ciclo 1) + 17,5h (ciclo 2) + 22,2h
  (ciclo 3) + 22,8h (ciclo 4) ≈ **80 horas de GPU** nesta investigação.
- **Dataset final (ciclo 4)**: 175.123 imagens, 101.987 fotos distintas, 0,04% de
  vazamento auditado.

---

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

---

## 12. Trabalho futuro / próximos passos

Em ordem de prioridade real (não de facilidade):

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
| Fotos de campo (gabarito) | `backend/campo/{healthy,anomalous,nao_folha}/` |
| Mapa de rotulagem auditável (proveniência pasta→rótulo) | `backend/data/label_map_audit.json` |
| Diagramas de fiabilidade (calibração) | `backend/results/calibration_*.png` |
| Matrizes de confusão | `backend/results/cm_*.png` |
| Grad-CAM dos erros de campo (inconclusivo, ver §8.3) | `backend/results/campo_anomalous_alface_0{1,5}_gradcam.png` |
| Investigação técnica completa do vazamento e ciclos 1-3 | `docs/CORRECOES_METODOLOGICAS.md` |
| Guia de metodologia de teste de campo | `docs/GUIA_TESTE_DE_CAMPO.md` |
| Este documento (narrativa completa para a banca) | `docs/HISTORICO_COMPLETO_TCC.md` |

---

> Documento gerado em 29/07/2026, consolidando o trabalho de todas as sessões até
> aqui. Deve ser actualizado a cada novo ciclo ou achado relevante — é o documento
> de referência para a escrita do capítulo de metodologia/resultados e para a
> apresentação oral.
