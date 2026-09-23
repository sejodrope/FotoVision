# Correções Metodológicas — Registo da Investigação

### FitoVision · TCC · UNIVILLE · Julho 2026

> Este documento regista o processo que levou à **retratação dos resultados de
> 06/06/2026** (99,01% de acurácia) e à correcção do pipeline. Foi escrito para servir
> de base ao capítulo de metodologia e à defesa: descreve não só *o que* estava errado,
> mas *como* foi descoberto e *como* foi verificado que a correcção funciona.
>
> Nenhum número aqui é conjectura. O que foi medido está identificado como medido; o
> que ainda falta medir está identificado como pendente.

---

## 1. O sintoma

O sistema reportava **99,01% de acurácia** no conjunto de teste. Ao ser testado com
fotografias novas, tiradas fora do dataset, o desempenho não correspondeu de todo ao
prometido — as classificações eram erráticas e não representavam a realidade.

Esta discrepância — *excelente no papel, inútil na prática* — é um padrão clássico. Não
indica um modelo mal treinado. Indica que **o número está errado**, ou que **mede outra
coisa que não a capacidade de diagnóstico**.

O ponto de partida da investigação foi, portanto, inverter a pergunta:

> Não *"por que o modelo é mau?"*, mas **"por que o número é bom?"**

---

## 2. A investigação

### 2.1 Hipóteses consideradas

| # | Hipótese | Como foi testada | Veredicto |
|---|----------|------------------|-----------|
| H1 | Inconsistência de pré-processamento entre treino e inferência | Comparar `dataset.py::VAL_TRANSFORMS` com `app/ml/preprocessing.py` | **Parcialmente confirmada** — ver §3.5 |
| H2 | Pesos errados / modelo não carregado | Inspecção de `load_binary_model()` | Descartada — carrega correctamente |
| H3 | **Vazamento de dados entre treino e teste** | Análise da função de split + auditoria por hash perceptual | **✅ CONFIRMADA — causa principal** |
| H4 | Desbalanceamento de classes não tratado | Inspecção de `make_binary_folder_loaders()` | **Confirmada** — ver §3.3 |
| H5 | Excesso de confiança do softmax (calibração) | Medição do ECE | **Confirmada** — ver §3.4 |
| H6 | *Gap* de domínio (estúdio → campo) | Revisão da composição do dataset + literatura | **Confirmada — limitação de fundo, ver §6** |

### 2.2 O indício que estava à vista

A secção 7.1 do compilado original apresentava como **prova de qualidade**:

> *"A diferença entre val_acc e test_acc é mínima em todos os modelos
> (EfficientNet-B0: 98,98% → 99,01%). Isso confirma que os modelos generalizaram bem."*

**Esta inferência está invertida.** Validação e teste concordavam porque **ambos estavam
contaminados pelo treino**. A concordância entre dois conjuntos igualmente vazados não é
evidência de generalização — é um **sintoma** do vazamento.

Um modelo que atinge 99% num domínio visualmente difícil, com validação e teste a
coincidirem até à segunda casa decimal, é **suspeito, não excelente**. A *ausência* da
queda esperada entre validação e teste era, ela própria, o alarme.

> **Lição metodológica.** Métricas boas demais são uma hipótese a testar, não um
> resultado a celebrar. Perante um 99%, a primeira pergunta deve ser *"o que é que o
> conjunto de teste tem que não devia ter?"*.

---

## 3. As causas encontradas

### 3.1 Causa principal — vazamento de dados (*data leakage*)

O split era feito com um sorteio **ao nível do ficheiro**:

```python
# download_datasets.py — VERSÃO ANTIGA
images = sorted(f for f in src.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
rng.shuffle(images)                      # ← sorteio por FICHEIRO
n_train = n - n_val - n_test
splits = {
    "train": images[:n_train],
    "val":   images[n_train:n_train + n_val],
    "test":  images[n_train + n_val:],
}
```

O problema é que o pool de imagens contém **múltiplas cópias da mesma fotografia**:

| Fonte | Natureza da redundância |
|-------|-------------------------|
| `shuvokumarbasak2030/lettuce-disease-multi-transformation-dataset` | O nome é literal: são versões **rodadas, espelhadas e com brilho alterado** das mesmas folhas |
| `nirmalsankalana/plant-diseases-training-dataset` | É um **re-upload do PlantVillage** — as mesmas imagens, sob outros nomes de ficheiro |

Como os nomes dos ficheiros diferem, **nada no pipeline detectava a duplicação**. O
sorteio colocava a rotação de uma foto no treino e o espelhamento da **mesma foto** no
teste.

> **O modelo não estava a diagnosticar folhas. Estava a reconhecer fotografias que já
> tinha visto.** Os 99,01% mediam **memorização**, não capacidade de generalização.

### 3.2 A correcção — split agrupado por identidade visual

O split passou a ser feito ao nível de **grupos de identidade visual**, não de ficheiros:

1. Calcula-se um **hash perceptual** de cada imagem, invariante às 8 simetrias do
   quadrado (rotações de 90°/180°/270° e espelhamentos) e robusto a variações de
   exposição.
2. Agrupam-se os *near-duplicates* por *union-find*, com indexação LSH por bandas para
   manter o custo praticamente linear (comparar todos os pares em ~200k imagens seria
   O(n²), inviável).
3. Sorteiam-se **grupos inteiros** para treino/validação/teste.

**Consequência:** todas as variantes de uma mesma fotografia caem obrigatoriamente no
mesmo split. O vazamento torna-se estruturalmente impossível, não apenas improvável.

Implementação: `backend/imagehash_utils.py` + `backend/download_datasets.py::split_dataset()`.

#### Um erro cometido *durante* a correcção — e vale a pena registá-lo

A primeira implementação do hash colapsava as 8 orientações num único valor com `min()`.
Parecia correcto: como as 8 simetrias formam um grupo (D4), a órbita de uma imagem e a
da sua rotação são o **mesmo conjunto**, logo o mínimo coincide.

Mas `min()` é um **selector descontínuo**. Quando duas orientações têm valores de hash
próximos, basta **um bit perturbado** (um jitter de brilho, uma recompressão JPEG) para
mudar *qual* orientação atinge o mínimo — e o "hash canónico" salta para um padrão de
bits completamente diferente.

Medido: variantes com jitter de brilho davam distância de Hamming **~39 em 64**
(praticamente aleatória) em relação à foto de origem. Nenhuma tolerância de distância
resolvia o problema, porque o valor não estava *perto* — estava *noutro sítio*.

O erro só foi apanhado porque o teste sintético (§4) **falhou**: detectava 96 grupos onde
deviam existir 40. A correcção foi guardar a **órbita inteira** e considerar duas imagens
variantes se *alguma* orientação de uma estiver perto de *alguma* orientação da outra.

> Este episódio é, em si, um argumento a favor de testar as correcções — e não apenas
> confiar em que "faz sentido".

### 3.3 Desbalanceamento de classes não tratado

O compilado afirmava que o desbalanceamento (≈2:1 a favor de `anomalous`) era tratado
com `WeightedRandomSampler`. **Não era.**

O `WeightedRandomSampler` existia em `dataset.py`, mas só era usado pelo caminho
**multi-classe**. O pipeline **binário** — o que está em produção — usava `shuffle=True`
simples, sem qualquer compensação, e a `CrossEntropyLoss` não tinha pesos de classe.

Com `anomalous` a dominar 2:1, o modelo tinha um incentivo real a inclinar-se para
"doente" — consistente com fotos aleatórias serem classificadas como anómalas.

Agravante: o **melhor checkpoint era escolhido por accuracy**. Sob desbalanceamento, a
accuracy premia o modelo que prevê sempre a classe maioritária.

**Correcção:** amostragem balanceada + `CrossEntropyLoss(weight=...)` + selecção do
melhor checkpoint por **F1 macro**.

### 3.4 Excesso de confiança (calibração)

O softmax devolvia ~99% de confiança para **qualquer** imagem — inclusive para imagens
que não eram folhas. Redes profundas modernas são sistematicamente sobreconfiantes: o
softmax não tem como dizer *"não sei"*.

**Correcção — duas camadas:**

1. **Temperature scaling** (Guo et al., 2017): aprende-se um escalar `T` no conjunto de
   validação e dividem-se os logits por ele antes do softmax. Não altera a predição
   (o argmax é invariante), apenas achata as probabilidades para que "80% de confiança"
   signifique de facto "acerto 80% das vezes". Mede-se com o **ECE** (*Expected
   Calibration Error*).

2. **Política de abstenção**: abaixo de um limiar de confiança calibrada, o sistema
   devolve `inconclusive` em vez de arriscar um diagnóstico. E uma **guarda de domínio**
   pelo índice ExG (*Excess Green*) devolve `not_a_leaf` quando a imagem não contém
   vegetação.

> Abster-se é a resposta correcta quando o modelo não sabe. É o que distingue um sistema
> honesto de um que mente com convicção.

Implementação: `backend/calibrate.py` + `backend/app/ml/inference.py`.

### 3.5 Inconsistência geométrica entre treino e inferência

| | Antes | Depois |
|---|---|---|
| **Treino** | `RandomResizedCrop(224)` — recorta, **preserva a proporção** | igual |
| **Val / Teste / Inferência** | `Resize((224, 224))` — **esmaga a proporção** | `Resize(256) + CenterCrop(224)` |

O modelo treinava em folhas com geometria correcta e era servido, em produção, com
folhas **achatadas**. Numa foto de telemóvel (4:3 ou 16:9) a distorção é severa — uma
folha redonda chega ao modelo como uma elipse.

### 3.6 Augmentation destruía o sinal a detectar

O `ColorJitter` usava `saturation=0.3, hue=0.05`. Mas **clorose** (amarelecimento),
**míldio** e **oídio** (manchas) são definidos precisamente por **desvio de cor**.
Perturbar matiz e saturação com essa intensidade ensina o modelo a ignorar exactamente a
evidência de que precisa.

**Correcção:** `saturation=0.15, hue=0.02`. Brilho e contraste — que modelam variação de
*iluminação*, não de *doença* — foram mantidos generosos.

> **Princípio:** a augmentation deve perturbar as **variáveis de ruído** (pose, exposição,
> enquadramento) e nunca a **variável de decisão** (a cor e a textura da lesão).

### 3.7 Imagens corrompidas viravam quadrados pretos rotulados

```python
# dataset.py — VERSÃO ANTIGA
except (UnidentifiedImageError, OSError):
    img = Image.new("RGB", (224, 224), color=0)   # quadrado preto...
return img, label                                  # ...com o label ORIGINAL
```

Num dataset de ~200k imagens, os ficheiros ilegíveis não são raros. Cada um ensinava o
modelo a associar **"imagem preta" → um label real**. Ruído puro injectado na loss.

**Correcção:** imagens ilegíveis são **descartadas** no split; em tempo de execução,
reamostra-se outra imagem da **mesma classe** (preserva a distribuição de labels, não
inventa padrões).

### 3.8 Rotulagem incorrecta dos datasets Roboflow

```python
# VERSÃO ANTIGA
has_annotation = label_file.exists() and label_file.stat().st_size > 10
label = "anomalous" if has_annotation else "healthy"
```

Errado nos **dois** sentidos:

- Num dataset de detecção, uma imagem cuja *bounding box* marca uma folha **saudável**
  (classe `Healthy`, presente na maioria destes datasets) tem ficheiro de anotação — e
  era rotulada como **doente**.
- Uma imagem simplesmente não anotada era rotulada como **saudável**.

**Correcção:** lê-se o `data.yaml`, mapeia-se cada `class_id` para healthy/anomalous, e
decide-se pelo **conteúdo real** das anotações. Imagens ambíguas são descartadas.

### 3.9 Heurística de rótulos com termos genéricos

As palavras-chave incluíam `"good"`, `"normal"`, `"target"` (casavam com nomes de pasta
sem relação com sanidade) e `"weed"` — **erva daninha não é folha doente**; manter esse
termo ensinava o modelo a classificar uma *espécie de planta* como anomalia
fitossanitária.

**Correcção:** casamento por **token** (não substring solta), termos genéricos removidos,
e toda a atribuição registada em `data/label_map_audit.json` para conferência manual.
Pastas não reconhecidas são **reportadas e descartadas** — nenhum rótulo é adivinhado em
silêncio.

---

## 4. Verificação experimental das correcções

As correcções não foram assumidas como boas — foram **testadas**.

### 4.1 Teste do split agrupado

**Montagem.** Um dataset sintético que reproduz a estrutura do dataset real: 40
fotografias-base distintas, cada uma com 5 variantes por augmentation (rotação 90°,
espelhamento, rotação 180°, jitter de brilho, reescala/recompressão) mais um "re-upload"
sob outro nome — 7 cópias por foto, 280 ficheiros no total.

Como se conhece a origem verdadeira de cada ficheiro, é possível medir o vazamento de
forma exacta.

**Resultado (executando o código real de `download_datasets.py` e `audit_leakage.py`):**

| Métrica | Split antigo (por ficheiro) | Split novo (por grupo) |
|---------|----------------------------|------------------------|
| Vazamento no conjunto de teste | **100,0%** | **0,0%** |
| Fotos-base recuperadas | — | **40 / 40** |
| Grupos impuros (misturam fotos distintas) | — | **0** |

**Varredura do parâmetro de tolerância** (`max_distance`), para escolher um valor
defensável e não arbitrário:

| `max_distance` | Grupos detectados (verdade: 60) | Grupos impuros | Vazamento |
|---|---|---|---|
| 0 | 129 | 0 | 60,0% |
| 2 | **60** | 0 | **0,0%** |
| **4** *(default)* | **60** | **0** | **0,0%** |
| 6–10 | **60** | 0 | **0,0%** |
| 12 | 58 | 1 ⚠️ | 0,0% |

O valor **4** situa-se no meio de um patamar estável [2, 10]. A sobre-fusão só começa em
12.

> **Assimetria que orientou a escolha:** fundir duas fotos distintas no mesmo grupo é
> quase inofensivo (ficam apenas no mesmo split); *falhar* a fusão de duas variantes da
> mesma foto causa vazamento (fatal). Logo, na dúvida, deve-se **fundir**.

### 4.2 Teste ponta-a-ponta do pipeline real

Com um staging sintético de **600 ficheiros** correspondentes a **100 fotos distintas**
(83,3% de redundância), correndo o `split_dataset()` real seguido do `audit_leakage.py`
real:

```
[split] 600 imagens → 100 fotos distintas (83.3% de redundância, maior grupo = 6 cópias)
[split] Grupos por split: treino=68 | val=16 | teste=16

  ✅ SEM VAZAMENTO SIGNIFICATIVO
     Apenas 0.0% do teste tem duplicado no treino.

  fotos distintas detectadas : 100  (verdade: 100)
  vazamento no teste         : 0.0%
  veredicto do auditor       : OK
```

### 4.3 Teste da calibração

Simularam-se logits que reproduzem exactamente o sintoma observado — um modelo com **82%
de acerto real** que reporta **99,3% de confiança**:

| | Antes | Depois |
|---|---|---|
| Accuracy real | 82,2% | 82,2% *(inalterada)* |
| Confiança média reportada | **99,3%** | **80,8%** |
| ECE (*Expected Calibration Error*) | 0,1704 | **0,0398** *(−77%)* |
| Temperatura aprendida | — | T = 4,08 |
| Predições alteradas | — | **nenhuma** (argmax invariante) |

A confiança reportada passou a corresponder à accuracy real (diferença: 1,4 pp).

### 4.4 Teste da guarda de domínio (ExG)

| Imagem | Fracção de vegetação | Veredicto |
|--------|---------------------|-----------|
| Folha verde | 1,00 | ✅ aceite |
| Folha verde-escura | 1,00 | ✅ aceite |
| **Folha amarelada (clorose)** | 1,00 | ✅ **aceite** — crítico: clorose é uma doença-alvo |
| Pelo de gato / castanho | 0,00 | ❌ `not_a_leaf` |
| Parede cinza | 0,00 | ❌ `not_a_leaf` |
| Céu azul | 0,00 | ❌ `not_a_leaf` |

---

## 5. Resumo das alterações no código

### Ficheiros novos

| Ficheiro | Função |
|----------|--------|
| `backend/imagehash_utils.py` | Hash perceptual invariante a D4 + agrupamento de *near-duplicates* por union-find com LSH |
| `backend/audit_leakage.py` | Auditoria: quantifica o vazamento entre splits. Gera `results/leakage_report.json` |
| `backend/calibrate.py` | Temperature scaling + escolha do limiar de abstenção + diagrama de fiabilidade |

### Ficheiros alterados

| Ficheiro | Alteração |
|----------|-----------|
| `backend/download_datasets.py` | Split **agrupado**; rotulagem Roboflow corrigida; heurística de keywords endurecida; auditoria de rótulos |
| `backend/dataset.py` | Transforms geometricamente consistentes; augmentation que não destrói a cor; imagens corrompidas descartadas; sampler balanceado |
| `backend/train.py` | Loss com pesos de classe; selecção do melhor checkpoint por **F1 macro**; métricas de accuracy balanceada por época |
| `backend/evaluate.py` | Acrescenta **accuracy balanceada**, ROC-AUC e **ECE**; aplica a calibração |
| `backend/app/ml/inference.py` | Calibração + abstenção (`inconclusive`) + guarda de vegetação (`not_a_leaf`) |
| `backend/app/ml/preprocessing.py` | `Resize(256) + CenterCrop(224)` — idêntico ao de validação |
| `backend/app/config.py` | Limiar de confiança activado (era 0.0 = desligado); limiar de vegetação |
| `backend/run_pipeline.py` | Audita → treina → **calibra** → avalia |
| `frontend/` | Quatro estados de resultado; métricas inválidas removidas da interface |

---

## 6. A limitação de fundo — o *gap* de domínio

**Corrigir o vazamento torna as métricas verdadeiras. Não garante, por si só, que o
modelo acerte em fotografias de campo.**

Existe um segundo problema, **independente** do vazamento:

- O grosso do dataset é **PlantVillage**: folha única, fundo neutro de estúdio,
  iluminação controlada, enquadramento padronizado.
- Uma fotografia de telemóvel tirada numa horta tem terra, sombra, várias folhas
  sobrepostas, oclusão, luz natural variável.

Estas duas distribuições são **muito diferentes**. A literatura é consistente: modelos
treinados em PlantVillage, com ~99% no seu próprio conjunto de teste (medido
correctamente), caem para a faixa dos **30–60%** em fotografias reais de lavoura.

Acresce que o PlantVillage **não contém alface, rúcula nem espinafre** — as culturas-alvo
deste trabalho. Usam-se tomate, batata e milho como *proxy visual* de folha doente. O que
o modelo "sabe" sobre alface é **transferido, não observado**.

> Isto **não se resolve com mais épocas de treino**. Resolve-se com **imagens de campo**.

### Por que isto fortalece o TCC

São dois problemas distintos, e ambos merecem uma secção:

| Problema | O que explica |
|----------|---------------|
| **Vazamento de dados** | Por que a *métrica* era falsa |
| ***Gap* de domínio** | Por que a *abordagem* tem um limite real |

Um TCC que identifica, quantifica e discute ambos vale consideravelmente mais do que um
que reporta 99% e não sobrevive à primeira pergunta da banca sobre uma foto real.

---

## 7. Próximos passos

### Imediatos (antes de reportar qualquer número)

```bash
cd backend
python download_datasets.py --skip-download   # refaz o split, agora agrupado
python audit_leakage.py                       # quantifica o vazamento do split antigo
python run_pipeline.py                        # treina → calibra → avalia
```

Reportar a **acurácia balanceada** de `results/metrics_comparison.csv`, e citar
`results/leakage_report.json` como justificação da retratação.

> **Expectativa honesta:** as métricas **vão cair, e devem cair**. Um valor na casa dos
> 85–95% de acurácia balanceada, obtido sem vazamento, é infinitamente mais defensável do
> que 99% que não sobrevivem ao primeiro contacto com uma fotografia real.

### Para o TCC-II — a experiência que mais valor acrescenta

**Montar um conjunto de teste de campo:** 100–300 fotografias de telemóvel de hortas
reais, rotuladas por inspecção visual.

Reportar as duas métricas **lado a lado**:

| Conjunto | Acurácia balanceada |
|----------|--------------------|
| Test set (mesmo domínio do treino) | *a medir* |
| **Campo (fotos reais de telemóvel)** | *a medir* |

A diferença entre as duas **é um resultado do trabalho** — a quantificação do *gap* de
domínio para esta tarefa e estas culturas. Deixa de ser uma fragilidade escondida e passa
a ser uma contribuição.

---

## 8. Artefactos gerados para a defesa

| Artefacto | Onde | Para que serve |
|-----------|------|----------------|
| `results/leakage_report.json` | gerado por `audit_leakage.py` | **A prova** de que os 99% eram inválidos — número citável |
| `data/split_metadata.json` | gerado pelo split | Nº de **fotos distintas** (vs. nº de ficheiros), redundância, proveniência por split |
| `data/label_map_audit.json` | gerado pelo split | Mapa auditável pasta-de-origem → rótulo |
| `results/calibration_*.png` | gerado por `calibrate.py` | Diagrama de fiabilidade: confiança reportada vs. acerto real |
| `results/metrics_comparison.csv` | gerado por `evaluate.py` | Accuracy, **acurácia balanceada**, F1, AUC, **ECE**, latência |
| `docs/COMPILADO_FITOVISION.md` § 0 | este repositório | Errata formal, com o antes/depois de cada correcção |

---

## 9. As três lições

1. **Métricas boas demais são uma hipótese, não um resultado.** Perante um 99%, perguntar
   *"o que é que o conjunto de teste tem que não devia ter?"* antes de celebrar.

2. **Concordância entre validação e teste não prova generalização** — se ambos vierem da
   mesma fonte contaminada, prova apenas que a contaminação é uniforme. O sinal de
   overfitting que se procurava estava mascarado pelo próprio vazamento.

3. **Correcções também têm de ser testadas.** O primeiro hash perceptual escrito para
   resolver o vazamento tinha ele próprio um defeito (o selector `min()` descontínuo, §3.2)
   que só apareceu porque o teste sintético falhou. Sem esse teste, o "fix" teria sido
   publicado com o problema quase intacto — e com a falsa sensação de estar resolvido.

---

## 10. Ciclo 2 — o sintoma reapareceu: por que a abstenção sozinha não bastava

*(27/07/2026)*

Mesmo depois da correcção do vazamento e da introdução da guarda de vegetação (ExG) e
da abstenção `inconclusive` (§5, §6), o sintoma original — **qualquer fotografia recebe
um diagnóstico confiante** — voltou a ser reportado. A investigação encontrou uma
**segunda causa concreta**, distinta do *gap* de domínio já documentado no §6.

### 10.1 O bug: o limiar de abstenção nunca era atingível

Em `calibrate.py::pick_threshold()`, o algoritmo buscava o **menor** limiar τ, a partir
de **0,50**, cuja accuracy selectiva batesse 95%. Como o classificador binário já acerta
~98% no val set *mesmo em τ=0,50* (o piso matemático da confiança softmax numa
classificação de 2 classes — nunca é inferior a 0,5), o algoritmo parava sempre na
primeira iteração.

**Evidência (antes da correcção):**

```json
{"temperature": 0.550047, "threshold": 0.5, "coverage_at_threshold": 1.0,
 "selective_accuracy": 0.9802}
```

`coverage_at_threshold: 1.0` — o sistema **nunca** abstinha, mesmo tendo o mecanismo
implementado e a guarda de vegetação a funcionar correctamente para "isto não é uma
folha". Um dos dois pilares da honestidade do sistema (§5) estava, na prática,
desligado.

**Correcção**: piso mínimo `min_tau=0,85` e meta de accuracy selectiva mais exigente
(98%, antes 95%) em `pick_threshold()`. Recalibrados os 3 modelos:

| Modelo | τ antes | τ depois | Cobertura (decide) | Acc. quando decide |
|---|---|---|---|---|
| EfficientNet-B0 | 0,50 (nunca abstinha) | 0,85 | 98,8% | 98,6% |
| MobileNetV2 | 0,50 | 0,85 | 95,4% | 98,2% |
| ResNet50 | 0,50 | 0,85 | 96,7% | 98,3% |

### 10.2 Primeiro teste de campo real (N=13)

José reuniu 20 fotografias reais de alface (horta, luz natural, terra visível) em
`dataset/alface 27.07/`. Foram triadas visualmente: **13 usadas como gabarito** (5
healthy, 8 anomalous), **7 excluídas** por não serem representativas — uma vinha de um
banco de dados fitopatológico (ID de catálogo visível na imagem), uma tinha texto
sobreposto, e cinco tinham coloração ambígua (possível pigmentação natural da
variedade vs. sintoma) — nestes casos optou-se por não arriscar um gabarito errado
numa métrica citável, em vez de forçar um rótulo.

Executado `test_campo.py` (mesmo pipeline de produção: pré-processamento, guarda ExG,
calibração, abstenção):

| Conjunto | Acurácia balanceada | N |
|---|---|---|
| Teste (domínio PlantVillage, estúdio) | 98,6% | 21.520 |
| **Campo (horta real, telemóvel)** | **87,5%** | 13 |

**Leitura:** um gap de 11,1pp — bem menor que a faixa de 30–60% que a literatura
citada no §6 previa para PlantVillage→campo. A amostra é pequena demais (N=13, só 4
healthy) para ser conclusiva, mas é o primeiro dado real, não mais uma estimativa da
literatura. Dos 13 casos: 11 diagnósticos correctos, 1 erro (`alface_04.jpg`, healthy
classificado como anomalous com 98% de confiança — não investigado a fundo ainda,
candidato a Grad-CAM), e **1 abstenção correcta** — uma foto healthy ambígua (conf.
0,72) devolvida como `inconclusive` em vez de um veredicto errado. Este último caso é
evidência directa de que a correcção do §10.1 funciona na prática, não só na
matemática.

### 10.3 Expansão do dataset: cobrir a cultura-alvo, não só o proxy

O §6 já documentava que o PlantVillage não contém alface, rúcula, espinafre, acelga
nem couve — as culturas-alvo. Buscou-se no Kaggle datasets específicos de doenças em
alface (ainda não incorporados ao pipeline):

- `santoshshaha/lettuce-plant-disease-dataset` (2.813 imagens)
- `iqrapervez2000/lettuce-disease-dataset` (6.992 imagens)
- `ramadhanihsaniyulfa/tomato-and-lettuce-diseases-dataset` (271 imagens)

**Resultado negativo, também documentado por rigor:** não existe, à data, dataset
Kaggle de doenças foliares específico para rúcula, espinafre, acelga ou couve. A busca
confirma — não é falha de pesquisa — que esta é uma lacuna real de dados públicos, a
registar como limitação e como trabalho futuro (§10.5).

Com os 3 datasets novos, o staging cresceu de 170.152 para **179.957 imagens**. Split
agrupado (por identidade visual, hash perceptual) refeito; auditoria de vazamento do
split novo: **0,19%** (veredicto OK — consistente com o ciclo 1, ver `results/leakage_report.json`).

### 10.4 Retreino em curso

Os 3 modelos (EfficientNet-B0, ResNet50, MobileNetV2) estão a ser retreinados do zero
com o dataset expandido, seguidos de recalibração (já com o `pick_threshold` corrigido)
e avaliação. Processo desacoplado da sessão (`run_correction_pipeline.py`, PID em
`backend/logs/correction_pipeline_v2.pid.txt`).

| Modelo | Ciclo 1 (F1 macro / bal.acc, val) | Ciclo 2 (dataset expandido) |
|---|---|---|
| EfficientNet-B0 | 0,9774 / 0,9802 | **0,9811 / 0,9839** |
| ResNet50 | 0,9663 / — | **0,9720 / 0,9781** |
| MobileNetV2 | — | *em treino (27/07, ~19h45-20h ETA)* |

*(tabela a completar quando o retreino terminar — ver `results/metrics_comparison.csv`
e o comparativo salvo em `results/pre_lettuce_expansion/` para o "antes")*

Backup de segurança do estado pré-expansão (pesos + métricas + resultado de campo)
preservado em `weights/pre_lettuce_expansion/` e `results/pre_lettuce_expansion/`,
para permitir rollback caso o retreino piore os resultados.

### 10.5 Próximos passos para validar e melhorar ainda mais

Em ordem do mais barato ao mais caro (mesmo critério do `GUIA_TESTE_DE_CAMPO.md` §4):

1. **Concluir o retreino e comparar antes/depois** — `metrics_comparison.csv` (domínio
   de teste) e reexecutar `test_campo.py` nas **mesmas 13 fotos** de `campo/` (nunca
   trocar o conjunto de comparação no meio do experimento).
2. **Investigar o erro de campo já conhecido** (`alface_04.jpg`) via `generate_gradcam.py`
   — entender se o modelo reagiu à mão na foto, a uma mancha real, ou a artefacto de
   iluminação.
3. **Ampliar o conjunto de teste de campo**: 13 → 20-40 fotos (meta exploratória) → 100-300
   (meta do TCC-II). Fotos **novas**, nunca reaproveitando as 13 já usadas como teste
   nem misturando com o treino.
4. **Se o gap de campo persistir alto**: considerar data augmentation dirigida (fundos
   variados, oclusão parcial, iluminação forte) antes de colectar mais dados — é mais
   barato que fine-tuning com fotos novas.
5. **Fine-tuning com fotos de campo rotuladas** — a solução de fundo segundo o próprio
   §6, quando houver volume suficiente (50-100+ fotos de campo dedicadas a treino,
   **disjuntas** do conjunto de teste de campo).
6. **Registar como limitação formal do TCC**: ausência de datasets públicos para
   rúcula, espinafre, acelga e couve (§10.3) — é resultado do trabalho, não lacuna de
   esforço.

> **Regra de ouro, repetida do `GUIA_TESTE_DE_CAMPO.md`:** fotos de campo do TESTE
> nunca entram no TREINO. Violar isso reintroduz o mesmo tipo de vazamento que a
> correcção do ciclo 1 eliminou — desta vez entre "campo" e "campo", não entre
> "teste" e "treino", mas com o mesmo efeito: uma métrica que parece boa e não é.

### 10.6 O retreino terminou — e revelou um segundo bug, mais antigo e mais grave

O pipeline completo (17,5h) terminou em 27/07/2026. No domínio de teste (estúdio) os
números praticamente empataram com o ciclo 1:

| Modelo | Ciclo 1 (bal.acc / F1 macro, teste) | Ciclo 2 (bal.acc / F1 macro, teste) |
|---|---|---|
| EfficientNet-B0 | 0,9859 / 0,9835 | 0,9842 / 0,9803 |
| ResNet50 | 0,9789 / 0,9724 | 0,9761 / 0,9690 |
| MobileNetV2 | 0,9723 / 0,9662 | 0,9767 / 0,9708 |

Mas o **teste de campo** (as mesmas 13 fotos do §10.2, comparação justa) **piorou**:

| Conjunto | Ciclo 1 | Ciclo 2 |
|---|---|---|
| Campo (N=13) | 87,5% | **77,5%** |
| Abstenções | 1/13 | 0/13 |

Duas fotos de alface claramente doente (murcha, bordas necrosadas), que o ciclo 1
classificava correctamente, passaram a ser diagnosticadas como `healthy` com
confiança de 95-100%. Isto é o oposto do que se esperava de "mais dados de alface" —
motivou investigar a fundo em vez de descartar como ruído de amostra pequena.

**Causa raiz encontrada — e não é ruído.** `_collect_labeled_dirs()` em
`download_datasets.py` fazia BFS por pastas rotuláveis, mas parava assim que o
**nome do container** batia uma palavra-chave, sem verificar se havia subpastas
com rótulo próprio mais específico lá dentro. O dataset `ashishjstar/lettuce-diseases`
— a **única fonte específica de alface usada desde o início do projeto** — está
organizado como `Lettuce_disease_datasets/{Healthy, Bacterial, Downy_mildew, ...}/`.
O token "disease" no nome do container bate `ANOMALOUS_KEYWORDS`, então a pasta
inteira era rotulada `anomalous` **sem nunca descer até `Healthy/`**.

**Resultado: 1.123 imagens de alface saudável (48% deste dataset) estavam rotuladas
como doente desde o ciclo 1** — a única fonte de "como é uma alface saudável" que o
modelo tinha, estava sistematicamente errada. Os 2 novos datasets do ciclo 2
(`iqrapervez2000`) tinham o mesmo padrão de empacotamento e adicionaram mais 2.849
imagens saudáveis mal rotuladas. **Total: 3.972 imagens de alface saudável
mal-rotuladas como anomalous**, entre os dois ciclos.

Isto explica melhor o sintoma original relatado por José ("qualquer imagem recebe um
diagnóstico confiante") do que o *gap* de domínio sozinho (§6): o modelo nunca teve
uma fonte confiável de "alface saudável" para aprender — o pouco que existia estava
etiquetado ao contrário.

**Correcção**: `_collect_labeled_dirs()` agora só aceita o rótulo do nome do
container quando **nenhuma subpasta imediata resolve para um rótulo próprio**; caso
contrário desce e usa os rótulos mais específicos dos filhos. Testado directamente
contra `ashishjstar_lettuce-diseases` — `Healthy/` (1.123 imgs) e as pastas de
doença específicas (`Bacterial`, `Downy_mildew_on_lettuce`, etc.) agora são separadas
correctamente.

**Acção tomada**: os pesos do ciclo 2 (treinados com o rótulo ainda errado) foram
**revertidos** — a produção voltou a usar os pesos do ciclo 1
(`weights/pre_lettuce_expansion/`, backup preservado antes do ciclo 2, agora
restaurado como activo) até um ciclo 3 com a rotulagem corrigida ser treinado e
validado no mesmo conjunto de campo.

> **Lição adicional às três do §9**: uma métrica de domínio (teste limpo) que empata
> ou até melhora não garante nada sobre o domínio real — o teste de campo, mesmo
> pequeno (N=13), foi o único sinal que expôs o problema. E a causa nem sempre é o
> que se está a mexer no momento (os 3 datasets novos): pode ser um bug antigo,
> dormente desde o commit inicial, que só ficou visível porque o teste de campo
> finalmente existe.

### 10.7 Ciclo 3 — rótulos corrigidos, treino refeito, resultado de campo não melhorou

*(28-29/07/2026)*

Staging reconstruído do zero com `_collect_labeled_dirs()` corrigido. Confirmação
directa antes de retreinar: `staging/healthy` subiu de 72.252→76.224 (+3.972,
exactamente as imagens recuperadas), e `staging/anomalous` caiu de 107.705→99.902
(-7.803 — os 3.972 realocados **mais** 3.831 imagens de erva daninha
"Shepherd_purse_weeds" que o mesmo bug também deixava vazar para "anomalous" sem
passar pelo filtro `IGNORE_KEYWORDS`, porque a BFS parava cedo demais para chegar
lá). Novo split: 176.126 imagens, vazamento 0,04% (`results/leakage_report.json`,
veredicto OK).

**Resultado no domínio de teste** (17.813 imagens, `results/metrics_comparison.csv`):

| Modelo | Ciclo 1 | Ciclo 2 (rótulos errados) | Ciclo 3 (rótulos corrigidos) |
|---|---|---|---|
| EfficientNet-B0 | 0,9859 / 0,9835 | 0,9842 / 0,9803 | 0,9849 / 0,9844 |
| ResNet50 | 0,9789 / 0,9724 | 0,9761 / 0,9690 | 0,9773 / 0,9766 |
| MobileNetV2 | 0,9723 / 0,9662 | 0,9767 / 0,9708 | **0,9818 / 0,9805** |

*(formato: bal.acc / F1 macro)* — todos os três ciclos ficam dentro de uma faixa
estreita (97,2%-98,6%), como esperado: ~4 mil imagens corrigidas são uma fração
pequena de ~176 mil no total. O teste de domínio não é sensível o suficiente para
este tipo de correcção.

**Resultado no campo (mesmas 13 fotos, comparação justa):**

| Ciclo | Acurácia de campo | Abstenções |
|---|---|---|
| 1 (baseline) | 87,5% | 1/13 |
| 2 (rótulos errados) | 77,5% | 0/13 |
| 3 (rótulos corrigidos) | **73,3%** | 2/13 |

**Contra-intuitivo**: corrigir um bug real e bem documentado (quase 4 mil imagens de
alface saudável mal-rotuladas) **não melhorou** — e no ponto estimado, piorou — o
desempenho de campo. Em vez de descartar isso como "a correcção não funcionou", a
reacção correcta é desconfiar do instrumento de medição antes do modelo:

**Teste de significância**: intervalo de confiança de Wilson a 95% sobre a
proporção bruta de acertos **entre os diagnósticos emitidos** (abstenções
excluídas, uniformemente nos três ciclos):
- Ciclo 1 (11/12): **[64,6%, 98,5%]**
- Ciclo 2 (10/13): **[49,7%, 91,8%]**
- Ciclo 3 (8/11): **[43,4%, 90,3%]**

> **Correcção (31/07/2026)**: a primeira versão desta análise usava 11/13 para o
> ciclo 1 (abstenção contada como erro) e 8/11 para o ciclo 3 (abstenção
> excluída) — critérios **diferentes** para cada ciclo, o que tornava a
> comparação inválida. Recalculado uniformemente. A conclusão não muda.

Os intervalos **se sobrepõem na faixa 64,6%–90,3%**. Com esta amostra, os três ciclos
são **estatisticamente indistinguíveis** entre si — a diferença de 87,5% para 73,3%
(acurácia balanceada) pode ser inteiramente ruído de amostra pequena, não uma piora
real do modelo.
**Esta é a conclusão central desta rodada**: o teste de campo de N=13 já cumpriu o
papel de expor o bug de rotulagem (§10.6), mas **não tem poder estatístico para
decidir entre versões de modelo**. Ampliar a amostra (meta do
`GUIA_TESTE_DE_CAMPO.md`: 20-40, depois 100-300) deixou de ser "seria bom fazer" e
passou a ser **pré-requisito** para qualquer decisão de modelo daqui em diante.

**Padrão observado (não conclusivo, mas notável)**: os 2 erros novos do ciclo 3
(`alface_01.jpg`, `alface_05.png` — ambos anomalous→healthy, 93-96% de confiança)
são danos **estruturais** (murcha, furos de praga) em folhas que continuam
predominantemente verdes. Os acertos/abstenções entre as fotos anomalous
(`alface_03,04,06,07,08`) têm sinal de cor claro (manchas escuras, revestimento
esbranquiçado, bordas necrosadas marrons). Gerado Grad-CAM para os 2 erros
(`results/campo_anomalous_alface_01_gradcam.png`,
`results/campo_anomalous_alface_05_gradcam.png`) — **inconclusivo por limitação da
visualização** (o mapa de calor concentrou-se no *swatch* de legenda de cores, não
na folha), então isto fica registado como **hipótese para investigar**, não como
achado: o modelo pode estar a pesar mais pistas de cor (clorose, necrose escura)
do que pistas estruturais (furos, murcha) para decidir "anomalous". Se confirmado
com mais dados, sugere que o próximo ciclo de expansão deveria privilegiar exemplos
de dano por praga/murcha sobre mais exemplos de doença fúngica/bacteriana
(já bem representados).

**Decisão tomada**: manter o ciclo 3 em produção (é a versão metodologicamente mais
correcta — sem bugs conhecidos, melhor média no domínio de teste) em vez de reverter
de novo. Ao contrário do ciclo 2 (onde havia uma causa concreta e comprovada para
reverter — rótulos errados), não há aqui uma justificação equivalente para preferir
o ciclo 1 ou 2 sobre o 3: a única diferença mensurada (campo) não é estatisticamente
significativa.

### 10.8 Estado consolidado e próximos passos (29/07/2026)

**O que está em produção agora**: EfficientNet-B0 do ciclo 3 (bal.acc 0,9849 no
domínio de teste), calibrado (τ=0,85 em todos os modelos), pesos em `backend/weights/`.

**Registo dos 3 ciclos para a defesa** (todos os artefactos preservados):

| Artefacto | Ciclo 1 | Ciclo 2 | Ciclo 3 |
|---|---|---|---|
| Métricas de teste | `results/pre_lettuce_expansion/metrics_comparison.csv` | `results/ciclo2_labelbug/metrics_comparison.csv` | `results/metrics_comparison.csv` |
| Campo (CSV por foto) | `results/campo_resultados.csv` | `results/campo_resultados_v2.csv` | `results/campo_resultados_v3.csv` |
| Pesos | `results/pre_lettuce_expansion/` *(pth)* | sobrescritos (não preservados) | `backend/weights/*_binary.pth` (activo) |

**Próximos passos, em ordem de prioridade real** (revisto após o ciclo 3):

1. **Ampliar a amostra de campo é agora bloqueante, não opcional.** Com N=13 nada
   mais se decide com confiança. Meta imediata: 20-40 fotos novas (nunca
   reaproveitar as 13 já usadas como teste). Priorizar cobrir os dois padrões
   observados em §10.7: dano estrutural puro (murcha, furos) vs. dano com sinal de
   cor (manchas, clorose) em proporções equilibradas — para poder testar a hipótese
   levantada, não só aumentar N.
2. **Não lançar um ciclo 4 de retreino antes de ampliar o teste de campo.** Sem
   mais fotos, qualquer novo treino repete o mesmo problema: uma métrica de domínio
   estável e um veredicto de campo estatisticamente cego.
3. Investigar a hipótese cor-vs-estrutura com uma ferramenta de interpretabilidade
   melhor calibrada que o Grad-CAM actual (o *overlay* actual tem um artefacto
   visual que impede leitura — corrigir `generate_gradcam.py`/o pipeline de
   visualização antes de reusar para diagnóstico).
4. Registar o resultado do ciclo 3 como **contribuição metodológica do TCC**, não
   como fracasso: identificar e corrigir um bug de rotulagem real, medir o efeito
   no domínio de teste (neutro) e no campo (estatisticamente inconclusivo), e
   documentar honestamente que a correcção não bastou sozinha é exactamente o tipo
   de rigor que o `docs/CORRECOES_METODOLOGICAS.md` original defende no §9.
5. Manter a lacuna de rúcula/espinafre/acelga/couve (§10.3) documentada como
   trabalho futuro.

### 10.9 Ciclo 4 — espinafre + downy mildew, e a primeira melhoria que sobrevive a um teste estatístico

*(22-23/09/2026)*

**Por que o ciclo 4 foi lançado antes das fotos de campo novas** (revoga o item 2
do §10.8): as fotos de campo servem exclusivamente para **avaliar**, nunca para
treinar — regra fixada no plano de análise da Parcial I. Logo o treino não depende
delas. Treinar antes tem ainda uma vantagem metodológica: o modelo fica **congelado
antes de vermos as fotos novas**, o que torna a avaliação de campo verdadeiramente
cega e responde de antemão à crítica de ter ajustado o modelo olhando o conjunto de
teste. O que continua a depender das fotos é a **decisão de qual ciclo vai para
produção** — essa não foi tomada aqui.

#### O bug encontrado ao preparar o merge: 1.142 imagens, 49 fotos

O lote `lettuce_downy_v1`, aprovado na Fase 6, tinha o split feito por ficheiro. A
conferência dos nomes antes do merge mostrou que os 1.142 ficheiros são **recortes
de anotação** (`__annN`) e **variantes Roboflow** (`.rf.<hash>`) de apenas **49 fotos
originais** — e 45 dessas 49 tinham recortes em mais de um split (1.138 das 1.142
imagens afectadas).

O dedup por *phash* do §3.2 **não apanha este caso**, e a razão é instrutiva:
recortes diferentes da mesma fotografia não são quase-duplicados entre si — são
imagens genuinamente diferentes, de regiões diferentes. O critério de agrupamento
correcto aqui não é a semelhança visual, mas a **proveniência**: mesma foto de
origem ⇒ mesmo split. O split deste lote foi refeito agrupado pela foto original
(`backend/merge_ciclo4.py`, seed 42): 773/196/173 imagens vindas de 32/12/5 fotos.

> **Consequência para o texto do TCC**: este lote acrescenta **49 fotografias
> independentes** de *downy mildew*, não 1.142. O espinafre não tem este problema
> (0 grupos a cruzar splits).

**Lição, a somar às três do §9**: *o agrupamento tem de reflectir como os dados
foram produzidos, não só como se parecem.* Um dataset de detecção convertido em
classificação traz esta armadilha de origem.

#### O merge

+4.351 imagens (3.209 de espinafre *Malabar*, 1.142 de alface *downy mildew*),
**sem refazer o split existente** — os ficheiros novos foram acrescentados aos
splits do ciclo 3. Isso mantém o teste do ciclo 3 **contido** no teste do ciclo 4 e
permite uma comparação justa nesse subconjunto, em vez de comparar números medidos
em conjuntos diferentes (erro que o §10.7 já teve de corrigir uma vez).

Auditoria do split com o merge: **0,04% de vazamento no teste** (10/24.624),
veredicto OK — `results/ciclo4/leakage_report_ciclo4.json`. Dataset final: 175.123
imagens, 101.987 fotos distintas.

#### Resultados

**No domínio de teste ORIGINAL** (as mesmas 23.967 imagens do ciclo 3 — comparação
justa, `results/ciclo4/teste_original/`):

| Modelo | Ciclo 3 | Ciclo 4 |
|---|---|---|
| EfficientNet-B0 | 0,9849 / 0,9844 | 0,9848 / 0,9844 |
| ResNet50 | 0,9773 / 0,9766 | 0,9782 / 0,9770 |
| MobileNetV2 | 0,9818 / 0,9805 | 0,9789 / 0,9783 |

*(bal.acc / F1 macro)* — empate técnico, como esperado: aprender culturas novas não
custou desempenho nas antigas.

**No teste NOVO** (657 imagens de espinafre + downy mildew, `teste_novo/` e
`baseline_ciclo3_teste_novo/`), com **McNemar exacto pareado** foto a foto e
correcção de Holm (`compara_ciclos_teste_novo.py`):

| Modelo | Ciclo 3 (bal.acc, IC95) | Ciclo 4 (bal.acc, IC95) | só c3 / só c4 | p (Holm) |
|---|---|---|---|---|
| EfficientNet-B0 | 0,811 [0,780–0,841] | **0,903** [0,876–0,929] | 24 / 110 | 7,1×10⁻¹⁴ |
| ResNet50 | 0,795 [0,762–0,828] | **0,884** [0,854–0,912] | 16 / 89 | 3,7×10⁻¹³ |
| MobileNetV2 | 0,758 [0,720–0,796] | **0,901** [0,875–0,925] | 24 / 95 | 3,5×10⁻¹¹ |

**Este é o primeiro ganho do projecto que sobrevive a um teste de significância.**
Ganhos de 9 a 14 pontos de acurácia balanceada, ICs que não se sobrepõem, e nos três
modelos independentemente. Contraste deliberado com o §10.7, onde a diferença
aparente entre ciclos era ruído de N=13: aqui o N é 657 e o teste é pareado.

**Calibração** (temperature scaling no val, `weights/*_binary_calibration.json`):
ECE do EfficientNet-B0 de 0,051 → **0,0027** após ajuste; τ=0,85 agora é atingível
(cobertura 98,9%, acurácia selectiva 99,0%). No teste novo, o ECE caiu de **0,175
(ciclo 3) para 0,053 (ciclo 4)**: o ciclo 3 não só errava mais nessas culturas como
errava **com confiança alta** — exactamente o modo de falha que a abstenção existe
para conter.

**No campo (as mesmas 13 fotos históricas, `results/campo_resultados_v4.csv`)**:

| Ciclo | Bal. acc de campo | Bruta (IC95 Wilson) | Abstenções | ECE |
|---|---|---|---|---|
| 1 | 87,5% | 91,7% [64,6–98,5] | 1/13 | 0,066 |
| 2 | 77,5% | 76,9% [49,7–91,8] | 0/13 | 0,217 |
| 3 | 73,3% | 72,7% [43,4–90,3] | 2/13 | 0,274 |
| 4 | **83,8%** | 84,6% [57,8–95,7] | 0/13 | **0,146** |

O ponto estimado subiu 10 pontos face ao ciclo 3 e o ECE caiu para quase metade,
**mas o McNemar entre ciclos continua não-significativo (p de Holm = 1,00 em todos
os pares)** — os ICs ainda se sobrepõem todos. A conclusão do §10.7 mantém-se
intacta e é agora ilustrada pelo contraste directo: **os mesmos modelos, medidos com
N=657, dão p≈10⁻¹³; medidos com N=13, dão p=1,00.** Não é o modelo que mudou de
qualidade entre as duas medições — é o instrumento que tem ou não poder estatístico.
As 13 fotos ficam registadas como **conjunto histórico** (já orientaram decisões
anteriores), reportado à parte da estimativa final, conforme a regra da Parcial I.

#### Artefactos

| Artefacto | Localização |
|---|---|
| Pesos e calibração do ciclo 3 (preservados) | `backend/weights/ciclo3/` |
| Métricas e campo do ciclo 3 | `backend/results/ciclo3/`, `backend/logs/ciclo3/` |
| Teste completo do ciclo 4 | `backend/results/metrics_comparison.csv` |
| Teste original / novo / baseline do ciclo 3 | `backend/results/ciclo4/{teste_original,teste_novo,baseline_ciclo3_teste_novo}/` |
| Comparação pareada + McNemar | `backend/results/ciclo4/comparacao_pareada_teste_novo.{csv,json}` |
| Proveniência de cada imagem acrescentada | `backend/data/ciclo4_merge_manifest.csv` |
| Auditoria de vazamento | `backend/results/ciclo4/leakage_report_ciclo4.json` |
| Log completo do pipeline | `backend/logs/ciclo4.log` |

Scripts novos: `merge_ciclo4.py`, `run_ciclo4.py`, `compara_ciclos_teste_novo.py`,
`analise_campo.py` (toda a análise da Parcial I num comando).

#### O que continua em aberto

1. **A decisão de produção ainda não foi tomada.** O ciclo 4 é igual ao ciclo 3 no
   domínio antigo, muito melhor nas culturas novas e melhor calibrado — mas a
   escolha formal espera as fotos de campo novas, para não ser tomada apenas com
   dados de dataset.
2. **As fotos de campo continuam a ser o bloqueio real** (chuva/alagamentos em
   Joinville; janela prevista a partir de 05/10/2026, em hortas comunitárias).
   Meta e procedimento: `docs/ROTEIRO_CAMPO_TCC2_MULTIESPECIE.md`.
3. **Hipótese cor-vs-estrutura (§10.7) continua sem teste** — precisa da coluna
   `categoria_dano` da planilha `backend/campo/metadados_campo.csv`, preenchida na
   hora da coleta.
4. **Ressalva de espécie**: o espinafre é *Basella alba* (Malabar), não *Spinacia
   oleracea* — declarar no texto.
5. **ViT** fica como experimento exploratório do TCC-II (comparação de arquitecturas,
   Q1), treinado neste mesmo dataset final, depois do ciclo 4.
