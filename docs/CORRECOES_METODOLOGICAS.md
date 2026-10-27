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

> **Correcção (13/10/2026, §14.3 e §14.4):** as duas linhas precisam de ressalva. O
> `nirmalsankalana` contém re-uploads do PlantVillage, mas é uma coletânea: 37% dela são
> culturas que o PlantVillage não tem. E o `shuvokumarbasak2030` tem 17 transformações
> por foto, das quais o hash criado para resolver este problema só junta as mais leves
> (espelho, brilho, ruído); rotações de ângulo livre, recortes e filtros de bordas
> continuaram separados da foto original — e 99,6% das imagens de teste dessa fonte
> ficaram com outra variante da mesma foto no treino.

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

> **Correcção (13/10/2026, §14.2):** 176.126 é o que estava em *staging*. O split
> descartou 1 imagem corrompida e 5.353 de 167 grupos com rótulos contraditórios; nas
> partições ficaram **170.772** imagens (120.861 / 25.944 / 23.967). No ciclo 2 os
> grupos contraditórios eram 1.239 (32.027 imagens) e o split tinha 147.929: o defeito
> de rotulagem também excluía essas imagens.

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

> **Correcção (08/10/2026):** 173 destas 657 imagens são recortes de apenas 5 fotos
> de míldio e não são observações independentes, o que infla o p. A reanálise só
> com o espinafre (N = 484) mantém a conclusão com números mais modestos: ganho de
> 9 a 13 pontos, p de Holm entre 5,9×10⁻¹² e 1,9×10⁻⁷. **O valor a citar é o do
> §11.2 c.** A tabela acima fica como registo do que foi reportado na altura.

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

---

## 11. Revisão crítica (08/10/2026) — o teste é do mesmo domínio do treino

*(preparação do Seminário 2 do TCC-II)*

### 11.1 O que motivou

Ao rever a apresentação do Seminário 2, José formulou a leitura que faltava: **o
modelo está amarrado aos dados de treino, e treino, validação e teste são muito
parecidos entre si**. Os documentos anteriores já falavam em *gap* de domínio (§6),
mas continuavam a resumir o resultado como "em laboratório o sistema é confiável".
Essa frase é forte demais. Esta seção verifica a leitura nos artefatos já gerados,
sem nenhum treino novo, e corrige o que estava mal enquadrado.

Tudo abaixo sai de `backend/reanalise_teste_novo.py`, que relê os resultados
existentes e grava `backend/results/ciclo4/reanalise_por_lote.json`.

### 11.2 O que foi verificado

#### a) Treino, validação e teste são fatias das mesmas seis fontes

| Fonte | Treino | Validação | Teste |
|---|---|---|---|
| `nirmalsankalana` (re-upload do PlantVillage) | 60,2% | 62,2% | 60,9% |
| `plantvillage` | 9,9% | 10,0% | 11,1% |
| `shuvokumarbasak2030` (alface, multi-transformação) | 25,0% | 23,1% | 23,4% |
| `iqrapervez2000` (alface) | 2,6% | 2,4% | 2,3% |
| `santoshshaha` (alface) | 1,6% | 1,7% | 1,7% |
| `ashishjstar` (alface) | 0,8% | 0,6% | 0,6% |
| **Derivadas do PlantVillage (soma das duas primeiras)** | **70,1%** | **72,2%** | **72,0%** |

*(dataset do ciclo 3, antes do merge do ciclo 4 — `data/split_metadata.json`)*

> **Correcção (13/10/2026, §14.3):** a linha "Derivadas do PlantVillage" está errada. O
> `nirmalsankalana` não é um re-upload do PlantVillage: é uma coletânea, e 37% dela são
> culturas que o PlantVillage não tem (mandioca, arroz, rosa, cana, café, melancia),
> parte fotografada em lavoura. O derivado do PlantVillage é no máximo 47,7% do treino.
> O que a tabela mostra de certo é o que o título diz: as três partições são fatias das
> mesmas seis fontes, nas mesmas proporções.
>
> **E a frase seguinte precisa de uma ressalva (§14.4):** o hash garante que a mesma
> fotografia não aparece dos dois lados *quando as cópias se parecem*. Não vale para a
> fonte `shuvokumarbasak2030`, feita de rotações, recortes e filtros das mesmas fotos.

O split agrupado por hash (§3.2) garante que a **mesma fotografia** não aparece em
dois lados. Não garante, nem pretende garantir, que o teste seja **diferente** do
treino: as fotos de teste vêm das mesmas fontes, com o mesmo fundo, a mesma luz e
o mesmo enquadramento das de treino. Os 98,49% medem, portanto, o acerto **dentro
das fontes de treino**. A partir daqui esse número passa a ser chamado de
**acurácia intra-fonte**, e não de "desempenho de laboratório".

#### b) Numa fonte nunca vista, o modelo cai 23 pontos — mesmo sendo foto de estúdio

O ciclo 4 produziu, sem que isso tivesse sido lido assim na altura, uma avaliação
de **fonte retida**: o modelo do ciclo 3 nunca viu o dataset de espinafre-de-malabar
(Mendeley), e foi avaliado nas 484 imagens de teste desse lote.

| Modelo (ciclo 3) | Intra-fonte (N = 23.967) | Fonte nunca vista: espinafre (N = 484) | Queda |
|---|---|---|---|
| EfficientNet-B0 | 0,985 | **0,754** | 23,1 pp |
| ResNet50 | 0,977 | 0,776 | 20,1 pp |
| MobileNetV2 | 0,982 | 0,743 | 23,9 pp |

Dois pontos tornam este resultado mais informativo do que as 13 fotos de campo:
tem N suficiente (484 fotografias independentes), e **não é foto de campo** — o
relatório de validação do lote regista "folha isolada sobre fundo branco/papel".
Ou seja, basta mudar de fonte, mantendo o estilo de estúdio, para o desempenho
cair para a casa dos 75%.

**Ressalva:** mudam ao mesmo tempo a fonte e a espécie (*Basella alba* não existia
no treino). Os dois efeitos não são separáveis com este dado.

#### c) O ganho do ciclo 4 foi reportado com pseudo-replicação — corrigido

O §10.9 reportava o ganho do ciclo 4 sobre 657 imagens, com p de Holm ≈ 10⁻¹³.
Dessas 657, **173 são recortes de apenas 5 fotos** de míldio (o próprio §10.9
explica a origem). Recortes da mesma fotografia não são observações independentes,
e o teste de McNemar por imagem trata-os como se fossem. O p estava inflado.

Reanálise por lote (McNemar exato pareado, Holm dentro de cada lote):

| Lote | N | Modelo | Ciclo 3 | Ciclo 4 | p (Holm) |
|---|---|---|---|---|---|
| **Espinafre** (fotos independentes) | 484 | EfficientNet-B0 | 0,754 | **0,884** | 5,9×10⁻¹² |
| | | ResNet50 | 0,776 | **0,862** | 1,9×10⁻⁷ |
| | | MobileNetV2 | 0,743 | **0,874** | 1,9×10⁻⁷ |
| Míldio (5 fotos, 173 recortes) | 173 | EfficientNet-B0 | 0,948 | 1,000 | *indicativo* |
| | | ResNet50 | 0,850 | 1,000 | *indicativo* |
| | | MobileNetV2 | 0,884 | 0,988 | *indicativo* |

**A conclusão do §10.9 mantém-se, com números mais modestos:** o ganho é de 9 a 13
pontos e continua significativo nos três modelos usando só o espinafre. O valor a
citar passa a ser o do espinafre (N = 484), não o agregado de 657. O lote de míldio
fica como indicação: o seu teste tem cinco fotografias.

#### d) O ganho do ciclo 4 é intra-fonte

Lido junto com (b), o ganho do ciclo 4 diz: *quando a fonte entra no treino, o
teste dessa fonte sobe* (0,754 → 0,884). Isso confirma que dados da fonte certa
funcionam e é coerente com a leitura de que o modelo aprende a aparência das
fontes. **Não diz nada sobre horta.** A frase "primeiro ganho que sobrevive a um
teste de significância" continua verdadeira, mas o que foi ganho é desempenho numa
fonte que passou a ser de treino.

#### e) Em campo, a confiança não avisa quando o modelo erra

| Ciclo | Confiança média (13 fotos) | Abstenções | Erros emitidos | Confiança dos erros |
|---|---|---|---|---|
| 1 | 0,963 | 1 | 1 | 0,981 |
| 2 | 0,986 | 0 | 3 | 0,999 · 0,949 · 0,983 |
| 3 | 0,937 | 2 | 3 | 0,997 · 0,992 · 0,993 |
| 4 | 0,992 | 0 | 2 | 0,997 · 0,991 |

Nos quatro ciclos, os **9 erros emitidos em campo tiveram confiança entre 0,949 e
0,999**. No ciclo 4, todas as 13 fotos receberam confiança ≥ 0,967 e o sistema não
se absteve nenhuma vez. O ECE confirma: 0,0027 na validação, 0,146 em campo.

A causa é estrutural, não um bug: a temperatura e o limiar τ são ajustados na
**validação**, que é da mesma distribuição do treino (item a). A calibração vale
para essa distribuição. Fora dela, o mecanismo de abstenção — descrito no §3.4
como o que "distingue um sistema honesto de um que mente com convicção" — não
segurou nenhum dos erros que chegaram ao usuário. Com N = 13 isto é um indício
forte, não uma medida; mas é o indício que mais importa para o uso real.

#### f) Procedência das 13 fotos de campo — a confirmar

As 13 fotos de `backend/campo/` têm entre 516 e 739 px no lado maior; quatro são
PNG de 400 px de largura; uma traz EXIF de câmara compacta (Sony DSC-H70), e as
restantes não têm EXIF de aparelho. O §10.2 diz que José as "reuniu"; a monografia
diz "capturadas com telefone celular". **Antes da entrega final é preciso
registrar a origem de cada uma** (foto própria, recebida por mensagem, internet).
Não altera nenhum número, mas altera o que se pode chamar de "campo" — e é mais um
motivo para o conjunto novo trazer metadados desde a coleta
(`backend/campo/metadados_campo.csv`).

### 11.3 O que muda na leitura dos resultados

| Antes dizia-se | Passa a dizer-se |
|---|---|
| "Em laboratório o sistema é confiável (98,49%)" | "Nas fontes de treino o sistema acerta 98,49%. É uma acurácia intra-fonte." |
| "Teste de 23.967 imagens nunca vistas" | "23.967 imagens sem duplicata no treino, mas das mesmas fontes" |
| "Ganho de 9 a 14 pontos, p ≈ 10⁻¹³ (N = 657)" | "Ganho de 9 a 13 pontos no espinafre, p de Holm ≤ 1,9×10⁻⁷ (N = 484); míldio só indicativo" |
| "Calibrado, o sistema sabe quando se abster" | "Calibrado na validação. Em campo, os erros observados saíram com 95–99% de confiança" |
| "O gap de campo é de ~15 pontos" | "Fora das fontes de treino a queda medida vai de 15 pontos (horta, N = 13) a 23 pontos (fonte nova, N = 484)" |

### 11.4 O que ainda é hipótese: o fundo

> **Correcção (13/10/2026, §14.3):** "mais de 70% do treino é folha isolada sobre fundo
> homogêneo" está errado. É no máximo 47,7%, e 17,6% do treino já é foto de lavoura de
> outras culturas. A hipótese do fundo continua de pé, mas deixou de ser a única: mudam
> também a cultura, o enquadramento e a proporção de exemplos doentes.

A explicação proposta para a queda em horta é o **fundo**: mais de 70% do treino é
folha isolada sobre fundo homogêneo; a horta tem terra, sombra, folhas sobrepostas.
A literatura apoia a suspeita — Noyan (2022) obteve 49% de acurácia no PlantVillage
usando apenas 8 píxeis do fundo, sem ver a folha — e a análise de falsos negativos
do próprio projeto aponta no mesmo sentido (os erros no teste concentram-se em
fotos de contexto amplo).

**Mas o projeto ainda não mediu isto diretamente.** O resultado de (b) mostra
dependência da *fonte*; não isola o fundo, porque o espinafre também tem fundo
limpo. Fica registrado como hipótese, com o teste que a decide descrito abaixo.

### 11.5 Ajustes propostos (a alinhar com o orientador)

1. **Nomear as três medidas** em todo o texto: *intra-fonte* (teste atual),
   *fonte retida* (fonte pública nunca vista) e *campo*.
2. **Coleta de campo em dois conjuntos disjuntos, separados por horta** (não por
   foto): um de **avaliação**, que nunca treina, e um de **adaptação**. Separar por
   horta evita repetir, entre "campo" e "campo", o vazamento por proveniência do
   §10.9.
3. **Ordem fixa:** primeiro o ciclo 4 congelado é avaliado às cegas em todas as
   fotos novas (mede o gap). Só depois o conjunto de adaptação entra num novo
   ciclo, e a medição repete-se no conjunto de avaliação.
4. **Fundo real no treino:** conjunto de adaptação + bases públicas fotografadas em
   campo (ex.: PlantDoc, Singh et al., 2020) + *augmentation* de fundo.
5. **Teste direto da hipótese do fundo:** avaliar as mesmas imagens com o fundo
   removido e com a folha removida. Se o modelo continuar a acertar sem a folha,
   está a decidir pelo fundo.
6. **Calibração verificada no domínio de uso:** reportar ECE, cobertura e erros
   interceptados em campo, e não só na validação.

> **Isto altera uma regra da Parcial I.** A regra (ii) dizia "as fotografias de
> campo nunca são usadas para treinamento". A proposta é mantê-la **para o conjunto
> de avaliação** e criar o conjunto de adaptação à parte. Enquanto o orientador não
> validar, vale a regra antiga: nenhuma foto de campo treina.

**Lição, a somar às anteriores:** *um teste sem vazamento ainda é um teste da mesma
distribuição.* Tirar as duplicatas torna a métrica verdadeira; não a torna uma
medida de generalização. Para isso é preciso reter uma fonte inteira ou ir a campo.

### 11.6 Registro consolidado de erros, achados e ajustes

Todos os erros de método do projeto num só lugar, por ordem de descoberta.

| # | Erro ou achado | Como foi descoberto | Ajuste | Onde |
|---|---|---|---|---|
| 1 | Split por arquivo: 57,35% do teste repetia fotos do treino | 99,01% no papel, diagnósticos erráticos em fotos novas | Split por grupos de hash perceptual; vazamento reportado como métrica | §3.1–3.2 |
| 2 | Hash "canónico" com `min()` era descontínuo | Teste sintético achou 96 grupos onde havia 40 | Comparar a órbita D4 inteira | §3.2 |
| 3 | Desbalanceamento não tratado no caminho binário; checkpoint escolhido por *accuracy* | Leitura do código | Sampler balanceado, pesos na *loss*, seleção por F1 macro | §3.3 |
| 4 | Softmax sem calibração: ~99% de confiança para qualquer imagem | Fotos que não eram folha recebiam diagnóstico | *Temperature scaling*, abstenção e guarda ExG | §3.4 |
| 5 | Treino preservava a proporção, inferência esmagava a imagem | Comparação dos *transforms* | `Resize(256)+CenterCrop(224)` em val, teste e produção | §3.5 |
| 6 | *Augmentation* de cor apagava o sinal da doença | Revisão do `ColorJitter` | Saturação e matiz reduzidos | §3.6 |
| 7 | Imagens corrompidas viravam quadrados pretos rotulados | Leitura do `dataset.py` | Descartadas no split; reamostragem em execução | §3.7 |
| 8 | Roboflow rotulado pela existência de anotação | Leitura do código | Rótulo lido do `data.yaml` | §3.8 |
| 9 | Palavras-chave genéricas (`good`, `weed`) na rotulagem | Auditoria do mapa de rótulos | Casamento por *token*; mapa auditável | §3.9 |
| 10 | Abstenção nunca disparava (τ = 0,50, piso do softmax binário) | `coverage: 1.0` no arquivo de calibração | Piso de 0,85 e meta de 98% | §10.1 |
| 11 | 3.972 alfaces saudáveis rotuladas como doentes; 3.831 ervas daninhas na classe anômala | Campo piorou no ciclo 2 | BFS desce às subpastas antes de aceitar o rótulo do contêiner | §10.6 |
| 12 | Comparação de campo com critérios diferentes de abstenção entre ciclos | Revisão de 31/07 | Abstenções excluídas de forma uniforme | §10.7 |
| 13 | Grad-CAM na camada errada (EfficientNet) e *overlay* ilegível | Mapas concentrados numa célula / na legenda | Último bloco MBConv; *overlay* a corrigir | §10.7, monografia 4.5 |
| 14 | Critério do MVP "supera CNN do zero" declarado atendido sem ter sido executado | Revisão do TCC-I | Declarado pendente | monografia 4.8 |
| 15 | 13 fotos de campo reutilizadas para orientar decisões | Plano de análise da Parcial I | Viram "conjunto histórico", reportado à parte | Parcial I §3.4 |
| 16 | Split por recorte no míldio: 1.142 imagens de 49 fotos | Conferência dos nomes antes do merge | Split pela foto de origem | §10.9 |
| 17 | "Espinafre" é *Basella alba*, não *Spinacia oleracea* | Validação do dataset | Ressalva declarada no texto | §10.9 |
| 18 | Regra "não treinar o ciclo 4 antes do campo" | Reavaliação em 22/09 | Revogada: treinar antes deixa a avaliação cega | §10.9 |
| 19 | **Teste intra-fonte lido como prova de generalização** | Revisão do Seminário 2 | Renomear a métrica; avaliar por fonte retida e em campo | §11.2 a–b |
| 20 | **Pseudo-replicação no McNemar do teste novo** (173 recortes de 5 fotos) | Reanálise por lote | Reportar só o espinafre (N = 484); míldio indicativo | §11.2 c |
| 21 | **Calibração e limiar validados só intra-fonte** | 9 erros de campo com confiança ≥ 0,949 | Verificar ECE e abstenção em campo | §11.2 e |
| 22 | **Procedência das 13 fotos de campo não registrada** | Inspeção de resolução e EXIF | Registrar origem; metadados obrigatórios na coleta nova | §11.2 f |
| 23 | **Hipótese do fundo nunca testada diretamente** | Revisão do Seminário 2 | Teste de remoção de fundo e de folha | §11.4 |
| 24 | **"O ciclo 4 não custou nada no domínio antigo" valia só para um modelo** | McNemar ciclo 3 × ciclo 4 no teste original (12/10) | Dizer por modelo: EfficientNet-B0 e ResNet-50 empatam; MobileNetV2 caiu | §12.3 |
| 25 | **Erros interceptados (138 de 404) citados no artigo v2 eram de um ciclo antigo** | Recontagem a partir das previsões por imagem (12/10) | Usar 113 de 363 (ciclo 4), rastreável a `results/arquiteturas/` | §12.4 |
| 26 | **Latência medida uma vez, com variação de até 3× entre execuções** | Comparação dos CSVs dos ciclos 3 e 4 | Medir as quatro arquiteturas na mesma sessão antes de reportar | §12.6 |
| 27 | **Média do teste escondia onde o modelo erra** (86,3% em foto de lavoura, 99,7% em estúdio) | Reagrupamento das previsões por fonte e cultura (13/10) | Reportar por subconjunto | §14.1 |
| 28 | **Vazamento por proveniência na fonte de variantes de alface**: 23% do teste com outra variante da mesma foto no treino | Id da foto original no nome do arquivo | Split agrupado também por proveniência (ciclo 5) | §14.4 |
| 29 | **"70% do treino deriva do PlantVillage"**: a coletânea foi tratada como PlantVillage | Contagem por cultura nos nomes de arquivo | Máximo de 47,7%; 17,6% de foto de lavoura | §14.3 |
| 30 | **Cultura associada ao rótulo** (alface 87% saudável; culturas com um só rótulo) | Tabela fonte × classe do split | Amostragem equilibrada por cultura e rótulo (ciclo 5) | §14.3 |
| 31 | **Pastas de dano por praga fora do treino** (7.779 imagens) | Mapa de rótulos, lista de não reconhecidas | Rotular como anômalas (ciclo 5) | §14.5 |
| 32 | **Parâmetros citados eram os das arquiteturas de 1.000 classes** | Contagem nos modelos carregados com os pesos | 4,0 / 2,2 / 23,5 / 85,8 M; "21 vezes" | §14.6 |
| 33 | **"Nove erros de campo" são de quatro fotografias; uma tem rótulo duvidoso** | Tabela foto × ciclo | Dizer "em quatro fotografias"; conferir o rótulo | §14.7 |
| 34 | **Temperatura menor que 1 piora a calibração em campo** | ECE de campo com e sem a temperatura | Verificar e, se for o caso, reajustar no domínio de uso | §14.7 |
| 35 | **Três tamanhos de dataset citados como se fossem o mesmo** | `split_metadata.json` e logs | 176.126 reunidas, 170.772 na base, 175.123 no ciclo 4 | §14.2 |

### 11.7 Referências acrescentadas

- NOYAN, M. A. Uncovering bias in the PlantVillage dataset. arXiv:2206.04374, 2022.
- SINGH, D. et al. PlantDoc: a dataset for visual plant disease detection. In:
  ACM IKDD CoDS and COMAD, 7., 2020. p. 249-253. DOI: 10.1145/3371158.3371196.

---

## 12. Comparação pareada entre arquiteturas e protocolo do ViT (12/10/2026)

### 12.1 O que motivou

O artigo do TCC-II prometia, para a pergunta Q1, um teste de McNemar entre as
arquiteturas e intervalos por bootstrap, mas até aqui só havia o ponto estimado de
cada modelo (`metrics_comparison.csv`). Como o ViT-B/16 vai entrar na mesma
comparação, o cálculo foi feito de uma vez, de forma que sirva para as quatro.

`backend/compara_arquiteturas.py` grava um CSV por modelo com a previsão de cada
imagem do teste (`results/arquiteturas/predicoes/`) e compara todos os modelos
presentes: acurácia balanceada com IC de 95% (bootstrap estratificado, 10.000
reamostragens), falsos negativos, ECE, erros interceptados pelo limiar e McNemar
exato com correção de Holm. O teste é dividido em três recortes pelo manifesto do
merge: original (23.967), espinafre (484) e míldio (173, só descritivo — §11.2 c).

### 12.2 Resultado — CNNs do ciclo 4

**Teste original (N = 23.967, intra-fonte):**

| Modelo | Acur. bal. | IC 95% | FN | FP | ECE |
|---|---|---|---|---|---|
| EfficientNet-B0 | 0,9848 | [0,9832; 0,9864] | 209 | 154 | 0,0039 |
| MobileNetV2 | 0,9789 | [0,9770; 0,9807] | 294 | 211 | 0,0055 |
| ResNet-50 | 0,9782 | [0,9763; 0,9800] | 352 | 184 | 0,0061 |

| Par | Só o 1.º acerta | Só o 2.º acerta | p (Holm) |
|---|---|---|---|
| EfficientNet-B0 × MobileNetV2 | 265 | 123 | 9,0×10⁻¹³ |
| EfficientNet-B0 × ResNet-50 | 293 | 120 | 2,6×10⁻¹⁷ |
| MobileNetV2 × ResNet-50 | 210 | 179 | 0,13 |

O EfficientNet-B0 é melhor que os outros dois neste teste; MobileNetV2 e ResNet-50
não se distinguem. **Limite da conclusão:** cada modelo foi treinado uma vez, com uma
semente. O McNemar diz que os modelos acertam imagens diferentes neste teste, não
que a arquitetura seja melhor em geral.

**Espinafre (N = 484):** 0,884 / 0,874 / 0,862, com intervalos de cerca de ±3 pontos
e nenhum par significativo (p de Holm ≥ 0,58). Com essa amostra as três CNNs são
indistinguíveis.

### 12.3 Achado: o MobileNetV2 perdeu no domínio antigo entre os ciclos 3 e 4

A §10.9 concluiu que "aprender culturas novas não custou desempenho nas antigas" a
partir do EfficientNet-B0 (0,9849 → 0,9848). Refeita a comparação pareada para os
três modelos, nas mesmas 23.967 imagens
(`results/arquiteturas/ciclo3_vs_ciclo4_teste_original.json`):

| Modelo | Ciclo 3 | Ciclo 4 | Só o c3 acerta | Só o c4 acerta | p (Holm) |
|---|---|---|---|---|---|
| EfficientNet-B0 | 0,9849 | 0,9848 | 132 | 131 | 1,00 |
| ResNet-50 | 0,9773 | 0,9782 | 114 | 124 | 1,00 |
| **MobileNetV2** | **0,9818** | **0,9789** | 184 | 133 | **0,015** |

A frase vale para EfficientNet-B0 e ResNet-50. O MobileNetV2 caiu 0,29 ponto, com
significância. Pode ser efeito do merge ou variação entre execuções de treino (uma
semente por ciclo não separa as duas coisas). Como o modelo de produção é o
EfficientNet-B0, a decisão ciclo 3 × ciclo 4 não muda; muda o texto.

### 12.4 Abstenção no teste: quantos erros o limiar segura

EfficientNet-B0, ciclo 4, τ = 0,85, teste original: **113 de 363 erros (31,1%)** têm
confiança abaixo do limiar, com abstenção em 1,2% das imagens. A confiança separa
acertos de erros em média (0,990 contra 0,874; Mann-Whitney p < 0,001; AUROC 0,91),
mas mais de dois terços dos erros saem com confiança alta. No espinafre: 19 de 55.

O artigo v2 citava "138 de 404 (34,2%)". O número vem da monografia do TCC-I e é de
um ciclo anterior; não se reproduz com os pesos do ciclo 3 (133 de 362) nem do
ciclo 4. O artigo v3 usa a contagem acima.

### 12.5 Protocolo do ViT-B/16

Decisão: treinar o ViT-B/16 no Google Colab Pro, no dataset e no split do ciclo 4.
Guia: `docs/GUIA_TREINO_VIT_COLAB.md`. Orquestrador: `backend/run_vit.py`.

Igual às CNNs: dados, split, transformações, pesos iniciais do ImageNet-1k, 30
épocas, batch 64, paciência 10, entropia cruzada com label smoothing 0,1, amostragem
balanceada, seleção por F1 macro na validação, temperature scaling e τ ≥ 0,85.

Diferente, e a declarar no texto: LR 5×10⁻⁵ (CNNs: 10⁻³), weight decay 0,05 (CNNs:
10⁻⁴), duas épocas de aquecimento, clipping do gradiente em 1,0. É a receita usual
de fine-tuning de transformers.

Regras para não repetir erros anteriores:

1. Os hiperparâmetros foram fixados antes de qualquer execução. Se o treino falhar
   e for preciso mudar, a escolha é pela validação e a mudança é relatada.
2. O ViT não é avaliado no espinafre como "fonte nunca vista": o lote está no
   treino do ciclo 4. A medida de fonte retida (75,4%) continua sendo só a do ciclo 3.
3. Nas 13 fotos de campo o resultado entra como registro. Com N = 13 não distingue
   arquiteturas.

Alterações no código, sem mudar o comportamento das CNNs (os valores padrão são os
dos ciclos 1 a 4): `train.py` ganhou `--weight-decay`, `--warmup-epochs`,
`--grad-clip` e `--resume`; `app/config.py` ganhou `binary_model`, para o
`test_campo.py` avaliar outra arquitetura por variável de ambiente. O pipeline foi
testado de ponta a ponta com um mini-dataset de 160 imagens (treino, retomada,
calibração, avaliação, previsões e campo). O treino real foi executado em
12–13/10/2026; resultado na §13.

### 12.6 Latência: medir de novo antes de reportar

O `evaluate.py` mede a latência com 100 inferências de uma imagem. Entre as execuções
dos ciclos 3 e 4, a mesma arquitetura variou bastante (MobileNetV2: 5,1 ms e 18,2 ms;
EfficientNet-B0: 8,0 ms e 14,6 ms), o que sugere interferência de outros processos na
GPU. A tabela do artigo mantém os valores do ciclo 3. Antes da versão final, medir as
quatro arquiteturas na mesma sessão, com a GPU ociosa. A latência do ViT no Colab não
é comparável com a das CNNs na RTX 3050.

**Resolvido em 13/10/2026 (§13.4):** as quatro arquiteturas foram medidas na mesma
sessão e a tabela do artigo v4 usa esses valores.

### 12.7 Referência acrescentada

- DOSOVITSKIY, A. et al. An image is worth 16x16 words: transformers for image
  recognition at scale. In: INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS,
  9., 2021.

---

## 13. ViT-B/16 treinado e comparado (13/10/2026)

### 13.1 Execução

Google Colab Pro, GPU L4 (23 GB), 12 CPUs. Início em 12/10 às 19:27, fim em 13/10 às
01:43. Comando: o de `backend/run_vit.py` com os valores padrão (§12.5), sem nenhuma
alteração de hiperparâmetro depois de iniciado.

| Item | Valor |
|---|---|
| Épocas executadas | 30 de 30 (parada antecipada não disparou) |
| Melhor época / F1 macro na validação | 29 / 0,9845 |
| Tempo de treino | 6,1 h (~12 min por época, 181 imagens/s) |
| Temperatura / ECE na validação antes e depois | 0,620 / 0,0472 → 0,0022 |
| Cobertura e acurácia seletiva na validação (τ = 0,85) | 0,995 / 0,9872 |
| Conferência do dataset no Colab | 123.879 / 26.620 / 24.624 imagens, igual ao manifesto; zips com SHA-256 igual ao do PC |

As previsões por imagem foram refeitas na RTX 3050 pelo `compara_arquiteturas.py` e
reproduzem as do Colab (acurácia balanceada de 0,9848 no teste original nas duas
máquinas).

### 13.2 Comparação pareada, quatro arquiteturas

Fonte: `backend/results/arquiteturas/comparacao_arquiteturas.md` (a versão só com as
CNNs ficou em `comparacao_arquiteturas_3cnn_12out.md`).

Teste original (N = 23.967):

| Modelo | Acur. bal. | IC 95% | FN | FP | ECE | Erros | Interceptados | Cobertura |
|---|---|---|---|---|---|---|---|---|
| efficientnet_b0 | 0,9848 | [0,9832; 0,9864] | 209 | 154 | 0,0039 | 363 | 113 | 0,9882 |
| vit_b_16 | 0,9848 | [0,9832; 0,9864] | 137 | 204 | 0,0018 | 341 | 51 | 0,9950 |
| mobilenet_v2 | 0,9789 | [0,9770; 0,9807] | 294 | 211 | 0,0055 | 505 | 249 | 0,9739 |
| resnet50 | 0,9782 | [0,9763; 0,9800] | 352 | 184 | 0,0061 | 536 | 233 | 0,9755 |

| Par | Só o 1.º acerta | Só o 2.º acerta | p de Holm |
|---|---|---|---|
| efficientnet_b0 × vit_b_16 | 158 | 180 | 0,256 |
| efficientnet_b0 × mobilenet_v2 | 265 | 123 | 1,35×10⁻¹² |
| efficientnet_b0 × resnet50 | 293 | 120 | 4,26×10⁻¹⁷ |
| mobilenet_v2 × vit_b_16 | 152 | 316 | 1,07×10⁻¹³ |
| resnet50 × vit_b_16 | 161 | 356 | 3,49×10⁻¹⁷ |
| mobilenet_v2 × resnet50 | 210 | 179 | 0,256 |

**Atenção ao citar valores de p.** A correção de Holm depende do número de
comparações. Com três modelos (§12.2) eram três; com quatro são seis. Por isso os
valores entre as CNNs mudaram (9,0×10⁻¹³ → 1,35×10⁻¹²; 2,6×10⁻¹⁷ → 4,26×10⁻¹⁷;
0,13 → 0,256), sem que nenhum dado tenha mudado. O artigo v4 usa os de seis
comparações; não misturar com os da §12.

Espinafre (N = 484): ViT 0,8439 [0,8092; 0,8771], FN 26, FP 42, ECE 0,1105, 12 de 68
erros interceptados. Nenhum dos seis pares é significativo (ViT × EfficientNet-B0:
p de Holm = 0,513).

### 13.3 Leitura

1. **Q1.** ViT-B/16 e EfficientNet-B0 são indistinguíveis neste teste, e ambos
   superam os outros dois. Vale para uma execução por modelo e com a receita de
   otimização própria do ViT.
2. **Perfil de erro.** O ViT troca falsos negativos por falsos positivos em relação
   ao EfficientNet-B0. Não foi escolhido nenhum ponto de operação para isso: é o que
   sai do argmax.
3. **Q2.** O ViT tem o menor ECE no teste original e, ao mesmo tempo, a menor fração
   de erros interceptados (15,0%). Não é contradição: ECE baixo diz que a confiança
   média bate com a acurácia média; não diz que os erros ficam nas confianças
   baixas. No espinafre, o ECE do ViT é o pior dos quatro.
4. **Campo, 13 fotos.** 12 de 12 diagnósticos certos e uma abstenção (foto saudável,
   confiança 0,77); Wilson de 75,8% a 100%. Registro, conforme a regra 3 da §12.5.
5. **Produção.** Sem mudança: EfficientNet-B0.

### 13.4 Latência, quatro arquiteturas na mesma sessão

RTX 3050 (6 GB), 13/10/2026, `evaluate.py` no teste original, ViT primeiro e as três
CNNs em seguida, sem outro uso da GPU:

| Modelo | Latência |
|---|---|
| MobileNetV2 | 5,3 ms |
| EfficientNet-B0 | 7,2 ms |
| ResNet-50 | 8,6 ms |
| ViT-B/16 | 19,4 ms |

Arquivos: `backend/results/latencia_13out/cnns/metrics_comparison.csv` e
`backend/results/vit/local_teste_original/metrics_comparison.csv`. Substituem os
valores do ciclo 3 (8,0 / 5,1 / 5,8 ms) na Tabela 1 do artigo. A do ResNet-50 saiu
bem acima da medida anterior (5,8 ms); com 100 inferências por medida, a variação
entre sessões descrita na §12.6 continua valendo como ressalva.

### 13.5 Como o treino foi operado

O roteiro do guia (Drive + notebook) não foi usado. O treino foi automatizado pela
CLI oficial `google-colab-cli`, a partir do WSL: `colab/cli/orq.sh` cria o runtime,
envia os zips em pedaços de 32 MB, confere SHA-256, extrai, lança o `run_vit.py`,
baixa o pacote de resultados e desliga o runtime. Uma primeira tentativa foi perdida
porque o PC suspendeu durante o envio e o Colab recolheu o runtime; nenhum resultado
dessa tentativa foi aproveitado. Custo total: 12,7 unidades de computação.

---

## 14. Auditoria de consistência (13/10/2026) — os documentos contra os artefatos

*Tudo abaixo sai de `backend/auditoria_consistencia.py`, que não treina nem roda
inferência, e está em `backend/results/auditoria/auditoria_13out.{json,md}`. A versão
para consulta rápida é `docs/FICHA_TECNICA_FITOVISION.md`.*

O que motivou: três inconsistências entre documentos (tamanho do dataset, curso,
procedência das fotos de campo). Em vez de escolher um número, conferiu-se cada
afirmação no artefato de origem. Apareceram erros maiores do que os procurados.

### 14.1 Onde estão os erros do teste

As previsões por imagem das quatro arquiteturas (`results/arquiteturas/predicoes/`)
foram reagrupadas por fonte e por cultura. Teste original, N = 23.967:

| Subconjunto | N | EfficientNet-B0 | ViT-B/16 | MobileNetV2 | ResNet-50 |
|---|---|---|---|---|---|
| Coletânea, culturas fotografadas em lavoura (mandioca, arroz, cana, café) | 4.530 | 0,8626 [0,8447; 0,8803] · 295 erros | 0,8342 · 293 | 0,8658 · 328 | 0,8779 · 337 |
| Coletânea, demais culturas | 10.061 | 0,9973 [0,9961; 0,9983] · 29 erros | 0,9978 · 18 | 0,9954 · 53 | 0,9943 · 62 |
| Só mandioca | 3.150 | 0,8090 · 271 erros | 0,7570 · 280 | 0,8128 · 301 | 0,8275 · 315 |
| Fonte `plantvillage` | 2.659 | 0,9987 · 3 erros | 0,9992 · 1 | 0,9977 · 7 | 0,9970 · 8 |
| Variantes transformadas de alface | 5.616 | 0,9765 · 32 erros | 0,9888 · 28 | 0,9558 · 104 | 0,9458 · 102 |
| Sem as variantes transformadas | 18.351 | 0,9780 [0,9754; 0,9806] | 0,9760 | 0,9754 | 0,9750 |

- 295 dos 363 erros do EfficientNet-B0 (81,3%) estão nas culturas fotografadas em
  lavoura, que são 18,9% do teste. A mandioca sozinha tem 271.
- Dentro da mesma fonte, a distância entre lavoura e estúdio é de 13,5 pontos.
- Nas culturas de lavoura, só EfficientNet-B0 × ResNet-50 passa na correção de Holm
  (p = 0,043); o ViT-B/16 tem a menor acurácia balanceada. Na mandioca, a especificidade
  do EfficientNet-B0 é de 67% (124 falsos positivos em 377 saudáveis).
- Sem as variantes transformadas, Q1 mantém-se: EfficientNet-B0 × ViT, p = 0,32; os dois
  contra MobileNetV2 e ResNet-50, p de Holm entre 1,7×10⁻⁴ e 1,8×10⁻⁸.

Duas ressalvas. A separação entre "lavoura" e "demais" foi feita por inspeção visual de
amostras de cada cultura (o conjunto de mandioca é o Cassava Leaf Disease, de fotos de
campo). E o efeito mistura condição de captura com a qualidade dos rótulos dessas
fontes: o projeto não os separa.

### 14.2 Tamanho do dataset, etapa por etapa

| Etapa | Em staging | Nas partições | Treino / Val / Teste | Grupos contraditórios descartados |
|---|---|---|---|---|
| TCC-I | — | 210.832 | — / 31.625 / 31.625 | — |
| Ciclo 1 | 170.152 | 141.107 | 99.394 / 20.193 / 21.520 | — |
| Ciclo 2 | 179.957 | 147.929 | 103.355 / 23.307 / 21.267 | 1.239 (32.027 imagens) |
| Ciclo 3 | 176.126 | 170.772 | 120.861 / 25.944 / 23.967 | 167 (5.353 imagens) |
| Ciclo 4 | — | 175.123 | 123.879 / 26.620 / 24.624 | — |

Consequência para a leitura dos ciclos: os testes dos ciclos 1, 2 e 3 são conjuntos
diferentes. A tabela "os 4 ciclos, domínio de teste" compara números de testes
diferentes; só ciclo 3 e ciclo 4 foram medidos nas mesmas imagens.

### 14.3 Composição do treino-base (120.861 imagens)

| Fonte | `healthy` | `anomalous` | Total | % |
|---|---|---|---|---|
| `nirmalsankalana` (coletânea) | 18.230 | 54.501 | 72.731 | 60,2 |
| `shuvokumarbasak2030` (variantes de alface) | 27.508 | 2.648 | 30.156 | 25,0 |
| `plantvillage` (15 pastas) | 2.894 | 9.112 | 12.006 | 9,9 |
| `iqrapervez2000` | 2.095 | 988 | 3.083 | 2,6 |
| `santoshshaha` | 839 | 1.139 | 1.978 | 1,6 |
| `ashishjstar` | 827 | 80 | 907 | 0,8 |

- Dentro da coletânea: 45.645 imagens de culturas do PlantVillage e 27.086 de outras
  (mandioca 14.999, rosa 4.965, arroz 3.859, cana 1.380, café 1.087, melancia 796). Há
  ainda classes de batata e maçã que o PlantVillage não tem.
- Derivado do PlantVillage: 12.006 + 45.645 = 57.651, **47,7%** (limite superior).
- Fotografado em lavoura: 21.325, **17,6%**.
- Alface: 36.124 (29,9%), 86,6% `healthy`. Demais: 84.737, 24,9% `healthy`.
- 59,7% da classe `healthy` é alface; 7,1% da classe `anomalous` é alface.
- Culturas com um só rótulo: arroz, laranja, abóbora (`anomalous`); soja, mirtilo,
  framboesa (`healthy`).

O erro veio de `reanalise_teste_novo.py`, que definia
`FONTES_PLANTVILLAGE = {"plantvillage", "nirmalsankalana_..."}`.

### 14.4 Vazamento por proveniência

A fonte `shuvokumarbasak2030` tem 41.760 arquivos no split: 17 transformações de cada
uma de 1.231 fotos, em duas cópias idênticas (pastas `augmentation` e `dataset`). O nome
guarda a classe, o id da foto e a transformação. As contagens por classe (1.123 / 20 / 30
/ 18 / 19 / 15 / 6) são as do `ashishjstar`, e 100% das imagens do `ashishjstar` casam,
pelo hash, com uma variante pouco alterada dessa fonte: as fotos originais são as mesmas.

| Partição | Imagens da fonte | Com outra variante da mesma foto no treino |
|---|---|---|
| Validação | 5.988 | 5.876 (98,1%) |
| Teste | 5.616 | 5.592 (99,6%) |

Das 1.231 fotos originais, 144 ficaram em uma só partição, 605 em duas e 482 nas três.
Nas outras fontes de alface, ligadas às fotos originais pelo hash: `ashishjstar` 153 de
153 no teste, `iqrapervez2000` 450 de 550, `santoshshaha` pelo menos 141 de 398.

O hash não falhou: fez o que foi desenhado para fazer. Uma rotação de 20°, um recorte ou
um filtro `sobel` não ficam a 4 bits da foto original. O critério certo aqui é o do
§10.9: proveniência. Faltou aplicá-lo ao dataset inteiro.

### 14.5 Pastas que a rotulagem deixou de fora

| Fonte | Pasta | Imagens |
|---|---|---|
| `iqrapervez2000` | `Lettuce_disease_datasets\Pest damage` | 513 |
| `nirmalsankalana` | `Potato___pests` | 611 |
| `nirmalsankalana` | `Rose___slug_sawfly` | 4.979 |
| `nirmalsankalana` | `Tomato___spider_mites` | 1.676 |
| `nirmalsankalana` | `Sugercane___yellow_leaf` | 505 |
| `ramadhanihsaniyulfa` | `images` (estrutura não reconhecida) | 271 |

As quatro primeiras são dano por praga. O casamento por *token* não tem "pest", "pests",
"sawfly" nem "mites" (tem "mite", no singular). O projeto define anomalia como "doença,
praga ou deficiência", e a hipótese H2 diz que o modelo erra mais em dano estrutural.

### 14.6 Arquitetura e treino

- Parâmetros dos modelos como foram treinados (saída de duas classes): MobileNetV2
  2.226.434; EfficientNet-B0 4.010.110; ResNet-50 23.512.130; ViT-B/16 85.800.194. Os
  números citados até aqui (3,4 / 5,3 / 25,6 / 86,6 M) são das arquiteturas de 1.000
  classes. ViT ÷ EfficientNet-B0 = 21,4; ResNet-50 ÷ MobileNetV2 = 10,6.
- Lote de 64 nos ciclos 1 a 4 (`run_ciclo4.py`, `run_correction_pipeline.py`); o `README`
  mostrava 32, que é o padrão do `train.py`.
- Taxa de aprendizado de 10⁻³ até 10⁻⁵ por cosseno, decaimento de pesos de 10⁻⁴, AMP.
- A parada antecipada (paciência 10) não disparou nos ciclos 3 e 4. O ResNet-50 teve o
  melhor checkpoint na última época nos dois.
- Todas as temperaturas são menores que 1 (0,546 a 0,620).

### 14.7 As 13 fotos de campo

| Foto | Ciclo 1 | Ciclo 2 | Ciclo 3 | Ciclo 4 | ViT-B/16 |
|---|---|---|---|---|---|
| `healthy/alface_04.jpg` | erro (0,981) | erro (0,983) | erro (0,993) | erro (0,991) | certo |
| `anomalous/alface_01.jpg` | certo | erro (0,999) | erro (0,997) | erro (0,997) | certo |
| `anomalous/alface_02.jpg` | certo | erro (0,949) | abstém | certo | certo |
| `anomalous/alface_05.png` | certo | certo | erro (0,992) | certo | certo |
| `anomalous/alface_03.png` | certo | certo | abstém | certo | certo |
| `healthy/alface_02.jpg` | abstém | certo | certo | certo | abstém |
| As outras 7 | certo | certo | certo | certo | certo |

- Os 9 erros são de 4 fotografias. Tratá-los como nove observações é a pseudo-replicação
  do §11.2 c, aplicada ao próprio argumento.
- `healthy/alface_04.jpg` mostra uma folha com manchas escuras; o rótulo é duvidoso. Com
  ele trocado, o ciclo 4 teria 12 de 13 (IC de Wilson 66,7% a 98,6%).
- ECE de campo com e sem a temperatura ajustada na validação: ciclo 1, 0,066 e 0,033;
  ciclo 3, 0,274 e 0,243; ciclo 4, 0,146 e 0,094. A temperatura piora a calibração em
  campo. Sem ela os erros sairiam com 0,93 a 0,96, ainda acima de τ.
- Procedência: 13 JPEG de 516 a 739 px, sem EXIF de celular; um com EXIF de Sony DSC-H70,
  de 2014, e Adobe Photoshop CS6; um com faixas pretas de quadro de vídeo. Nenhuma é
  quase-duplicata de imagem do dataset (menor distância 7, limiar 4).

### 14.8 O que estas correções pedem ao ciclo 5

1. Split agrupado por proveniência: todas as variantes de uma foto do mesmo lado.
2. Teste só com fotografias: sem `sobel`, `laplacian` e `high_pass`.
3. Pastas de praga rotuladas como anômalas, por lista explícita e auditável.
4. Amostragem equilibrada por cultura e rótulo, para enfraquecer o atalho da cultura.
5. Resultado reportado por subconjunto, e não só pela média.
6. Temperatura e limiar verificados em dados de fora das fontes de treino.

**Lição:** *auditar os documentos contra os artefatos, não uns contra os outros.* Três
documentos concordavam em "70%" e em "16 vezes"; os dois estavam errados.

---

## 15. Ciclo 5 — o pipeline corrigido (preparado em 13/10/2026)

*Preparado em 13/10; treinado e avaliado nas quatro arquiteturas em 14/10 (§15.6–15.10);
no artigo v7. Nada do ciclo 4 é alterado:
dados em `backend/data_c5/` (hardlinks), pesos em `weights/ciclo5/`, resultados em
`results/ciclo5/`.*

### 15.1 O que cada correção faz

| Problema (§14) | Correção | Onde |
|---|---|---|
| Variantes da mesma foto de alface dos dois lados | Alface agrupada por **foto original**: o id do nome do arquivo junta as variantes, e o agrupador por hash liga as outras três fontes a elas. Fotos originais quase idênticas viram uma unidade só | `prepara_ciclo5.py` |
| Teste com imagens que não são fotografias | Saem `sobel`, `laplacian` e `high_pass`; saem as cópias idênticas da pasta `dataset/`. Validação e teste de alface ficam com **uma fotografia por foto original** | idem |
| Alface de origem desconhecida | Imagens sem foto original identificada, e a pasta `Pest damage`, ficam só no treino | idem |
| Pastas de praga fora do dataset | Entram como anômalas, por lista explícita; as da coletânea são repartidas por grupo de hash, seguindo a partição de qualquer quase-duplicata que já exista | idem; palavras-chave em `download_datasets.py` |
| Sem fonte retida para o modelo atual | O lote de espinafre `n56pn9fncw` (574 imagens) fica inteiro fora do treino e da validação | idem |
| Cultura associada ao rótulo | Amostragem que equilibra os rótulos **dentro de cada cultura** e mantém a fatia de cada cultura (`--amostragem cultura`) | `dataset.py`, `train.py` |
| Classe compensada duas vezes | Com a amostragem por cultura, a perda não leva pesos de classe (antes: amostragem equilibrada **e** perda ponderada) | `dataset.py` |
| Média que esconde os erros | Avaliação por subconjunto, com IC, ECE e erros interceptados em cada um, incluindo a fonte retida | `avalia_ciclo5.py` |
| Ciclos comparados em testes diferentes | A coletânea e o PlantVillage mantêm a partição do ciclo 4: nessas 17.250 imagens de teste, ciclo 4 e ciclo 5 são comparados foto a foto (McNemar exato) | idem |
| Fotos de campo sem procedência nem metadados | Conferência antes da inferência: EXIF, cópias, quase-duplicatas do treino, planilha com horta | `confere_fotos_campo.py` |
| Calibração só verificada nas fontes de treino | ECE de campo com e sem a temperatura, erros acima do limiar e resultado por horta, a cada rodada de campo | `roda_campo.py` |

O que **não** mudou, de propósito: arquiteturas, hiperparâmetros, aumento de dados,
número de épocas e a partição da coletânea e do PlantVillage. Mudar tudo de uma vez não
deixaria atribuir o efeito.

### 15.2 A alface, contada em fotografias

| | Treino | Validação | Teste |
|---|---|---|---|
| Unidades (fotos originais distintas) | 820 | 176 | 176 |
| das quais doentes | 66 | 14 | 15 |

São 1.231 fotos originais em 1.172 unidades, 95 doentes. **O teste honesto de alface tem
176 fotografias, 15 doentes** — contra as 6.717 imagens (961 "doentes") do teste do
ciclo 4, que eram quase todas variantes de fotos presentes no treino. Com esse N, o
intervalo de confiança da alface no ciclo 5 será largo; a medida da cultura-alvo continua
a depender das fotos de campo.

### 15.3 A amostragem por cultura, no treino

Cada cultura mantém a fatia que tem nos dados; dentro dela, saudável e anômala são
sorteadas meio a meio. A alface continua com cerca de 30% dos exemplos de cada época, mas
passa de 85% saudável para 50%. No total, 47,8% dos exemplos sorteados são saudáveis; o
maior fator de reamostragem é 4,3. Culturas com um só rótulo (arroz, laranja, abóbora,
soja, mirtilo, framboesa) ficam como estão: não há o que equilibrar.

### 15.4 Como rodar

```powershell
cd backend
.venv\Scripts\python.exe prepara_ciclo5.py --apply    # cria data_c5/ (já feito em 13/10)
.venv\Scripts\python.exe run_ciclo5.py                # EfficientNet-B0, cerca de 6 h
.venv\Scripts\python.exe run_ciclo5.py --modelos efficientnet_b0 mobilenet_v2 resnet50
```

O `run_ciclo5.py` é retomável e não troca o modelo servido. A comparação que decide é a
de campo: `roda_campo.py` roda o ciclo 4 e o ciclo 5 nas mesmas fotos.

### 15.5 O que o ciclo 5 não resolve

- A alface doente continua a ter cerca de cem fotografias originais.
- Rúcula, acelga e couve continuam sem dados.
- Culturas com um só rótulo continuam a permitir o atalho da cultura.
- A separação "lavoura × estúdio" continua a ser por inspeção visual.

### 15.6 Execução (14/10/2026)

**Auditoria do split.** O `audit_leakage.py` original agrupa todas as imagens por
similaridade, par a par em Python. Com as ~150 mil imagens do `data_c5`, que têm milhares
de folhas quase iguais sobre fundo branco, ele rodou mais de uma hora sem terminar (o
mesmo travamento visto no `prepara_ciclo5.py`). Foi acrescentado o modo `--rapido`, que
responde só o que decide o veredicto: para cada imagem de validação e teste, existe
alguma de treino a distância de órbita ≤ 4? O critério é o mesmo do `orbit_distance`
(qualquer das oito orientações contra qualquer das oito) e o cálculo é exato, por
princípio da casa dos pombos: duas hashes de 64 bits a distância ≤ 4 coincidem inteiras em
ao menos uma de 5 fatias, e só os pares que coincidem numa fatia têm a distância medida.
Foi conferido contra a comparação exaustiva em dados sintéticos, inclusive nos casos de
fronteira (4 e 5 bits), e deu o mesmo conjunto. Não calcula a redundância interna.

Resultado: **321 de 18.413 imagens de teste (1,7%) e 334 de 20.612 de validação (1,6%)**,
veredicto OK. No ciclo 4, 23% do teste. Os exemplos listados são, em boa parte, pares de
culturas diferentes (maçã sadia × tomate com mancha bacteriana; pimentão × espinafre):
colisões do dHash em imagens de pouca textura, e não a mesma fotografia. O 1,7% é,
portanto, um teto.

**Treino.** Google Colab Pro, GPU L4, pela CLI (`colab/cli_c5/`): as quatro arquiteturas
em sequência (EfficientNet-B0, ViT-B/16, MobileNetV2, ResNet-50), cada uma calibrada logo
após o treino, e a avaliação por subconjunto ao final. Mesma receita do ciclo 4 para as
CNNs e da rodada de 12–13/10 para o ViT. A auditoria fica no PC (`--sem-auditoria` no
Colab), e o orquestrador só lança o treino com o veredicto OK no relatório local. Os zips
(9,46 GB) chegaram ao runtime com SHA-256 igual ao do PC. Na primeira tentativa, o Colab
recolheu o runtime no meio do envio, sem suspensão do PC; na segunda, uma célula de
manutenção (`keep.py`) ficou rodando durante o envio. Treino lançado em 14/10 às 06:06.

### 15.7 Resultado do EfficientNet-B0 (14/10/2026)

Treino: 30 épocas em 109 min na L4; melhor F1 macro na validação 0,9720 na época 30
(acurácia balanceada 0,9780). Calibração: T = 0,655, ECE de validação 0,048 → 0,004.
Avaliação feita no PC (`avalia_ciclo5.py`, `results/ciclo5/avaliacao_ciclo5.md`).

| Subconjunto do teste do ciclo 5 | N | Acur. bal. [IC 95%] |
|---|---|---|
| Teste inteiro | 18.413 | 97,4% [97,1; 97,7] |
| Coletânea, culturas fotografadas em lavoura | 4.530 | 88,8% [87,2; 90,3] |
| Coletânea, demais culturas | 10.061 | 99,7% [99,6; 99,8] |
| PlantVillage | 2.659 | 99,8% [99,5; 100,0] |
| Alface, uma fotografia por foto original | 176 (15 doentes) | 96,7% [90,0; 100,0] |
| Espinafre | 400 | 86,6% [83,1; 90,0] |
| **Espinafre, fonte retida (`n56pn9fncw`)** | **574** | **87,9% [84,4; 91,2]** |

**Ciclo 4 × ciclo 5 nas mesmas imagens** (as que não mudaram de partição; pesos do ciclo 4
congelados em `weights/`):

| Subconjunto | N | Ciclo 4: bal. / espec. / sens. | Ciclo 5: bal. / espec. / sens. | Só c4 acerta / só c5 |
|---|---|---|---|---|
| Lavoura | 4.530 | 86,3 / 76,6 / 96,0 | **88,8 / 83,4** / 94,2 | 126 / 94 |
| Demais da coletânea | 10.061 | 99,7 / 99,8 / 99,7 | 99,7 / 99,7 / 99,7 | 14 / 12 |
| PlantVillage | 2.659 | 99,9 | 99,8 | 1 / 0 |
| Espinafre | 400 | 87,0 / 86,8 / 87,1 | 86,6 / 79,6 / 93,6 | 17 / 20 |
| Todas as comuns | 17.839 | 97,3 / 96,2 / 98,4 | 97,4 / 96,7 / 98,0 | 158 / 126 |

Leitura. A amostragem por cultura fez o que se esperava dela no subconjunto que concentrava
81% dos erros: a especificidade nas culturas de lavoura subiu 6,8 pontos, e a acurácia
balanceada 2,5, ao custo de 1,8 ponto de sensibilidade. No resto, nada muda. Contado
imagem a imagem, o ciclo 4 acerta um pouco mais (McNemar exato nas 17.250 do
`avalia_ciclo5.py`: 141 × 106, p = 0,03), porque as anômalas são maioria e cada acerto
pesa igual; a acurácia balanceada pesa as duas classes por igual. **Os dois números têm
de ser citados juntos**: o ciclo 5 troca sensibilidade por especificidade, não é melhor em
tudo. A fonte retida (87,9%) não tem par no ciclo 4, que treinou com esse lote.

Campo histórico (13 fotos de `backend/campo/`, `results/campo_historico_c5/`): ciclo 5
11 de 12 emitidos, uma abstenção (ciclo 4: 11 de 13); pareado, 1 × 1, p = 1. ECE de campo
0,091 com T e 0,125 sem — o contrário do ciclo 4 (0,146 com, 0,094 sem). Com N = 13,
inconclusivo; quem decide são as fotos novas.

### 15.8 Resultado do ViT-B/16 (14/10/2026)

Treino: melhor F1 macro na validação 0,9721 na época 23 (328 min na L4) — empata com o
EfficientNet-B0 na validação, como no ciclo 4. T = 0,676; ECE de validação 0,047 → 0,003.

| Subconjunto do teste do ciclo 5 | N | ViT-B/16 | EfficientNet-B0 |
|---|---|---|---|
| Teste inteiro | 18.413 | 96,9% [96,6; 97,2] | 97,4% [97,1; 97,7] |
| Lavoura | 4.530 | 84,5% [82,6; 86,3] | 88,8% [87,2; 90,3] |
| Demais da coletânea | 10.061 | 99,6% | 99,7% |
| PlantVillage | 2.659 | 99,9% | 99,8% |
| Alface, uma foto por original | 176 | 100% (sem erros) | 96,7% (1 FN) |
| Espinafre, fonte retida | 574 | 85,8% [82,0; 89,5] | 87,9% [84,4; 91,2] |

ViT ciclo 4 × ViT ciclo 5 nas mesmas imagens: lavoura 83,4% → 84,5% (especificidade
69,8% → 72,1%; McNemar 92 × 99, p = 0,66); demais da coletânea 99,8% → 99,6%
(22 × 9, p = 0,03); total das 17.839 comuns 96,9% → 96,8% (p = 0,95). **A amostragem por
cultura quase não mexeu no ViT**, enquanto no EfficientNet-B0 deu +6,8 pontos de
especificidade na lavoura. Com o mesmo F1 de validação, o EfficientNet-B0 do ciclo 5 é
melhor justamente no subconjunto que concentra os erros — mais um sinal de que a
validação intra-fonte não separa os modelos onde importa (§14).

### 15.9 Resultado do MobileNetV2 (14/10/2026)

Treino: F1 macro de validação 0,9597 na época 24 (106 min); T = 0,647; ECE de validação
→ 0,008. Teste inteiro 96,9% [96,6; 97,2]; lavoura **90,1% [88,8; 91,3]**, o melhor dos
modelos do ciclo 5 até aqui; alface uma por foto 100% (176); espinafre 79,5% [75,6; 83,4];
fonte retida 88,4% [84,9; 91,7].

Mesmas imagens (`results/ciclo5/ciclo4_x_ciclo5.md`, de `compara_c4_c5.py`):

| Subconjunto | N | Ciclo 4: bal. / espec. / sens. | Ciclo 5 | Só c4 / só c5 | p |
|---|---|---|---|---|---|
| Lavoura | 4.530 | 86,6 / 78,3 / 94,9 | 90,1 / **90,2 / 90,0** | 218 / 94 | 1,7×10⁻¹² |
| Espinafre | 400 | 86,0 / 90,4 / 81,5 | 79,5 / 65,9 / 93,1 | 43 / 29 | 0,13 |
| Todas as comuns | 17.839 | 97,1 / 96,5 / 97,8 | 96,8 / 97,0 / 96,6 | 292 / 156 | 1,3×10⁻¹⁰ |

É a troca mais nítida: na lavoura, o modelo passa de "quase tudo é doente" (sensibilidade
95%, especificidade 78%) para equilibrado (90% e 90%). Como as anômalas são 3 em 4 nesse
subconjunto, o acerto contado por imagem cai (McNemar favorece o ciclo 4), enquanto a
acurácia balanceada sobe 3,5 pontos. No espinafre a troca vai no sentido contrário
(especificidade 90 → 66). **Qual dos dois é "melhor" depende do custo de cada erro na
horta** — um falso alarme custa uma inspeção; uma doença perdida pode custar o canteiro —,
e essa escolha deve ser dita no texto, não escondida numa métrica só.

### 15.10 Ciclo 5 completo — as quatro arquiteturas (14/10/2026)

ResNet-50: F1 macro de validação 0,9607 (época 28, 151 min); T = 0,637; ECE de validação
0,046 → 0,008. Rodada inteira no Colab de 06:06 a 17:52, cerca de 22 unidades de
computação; pacote em `colab_export_c5/retorno/c5_resultados.zip` (SHA-256 conferido).
As previsões refeitas no PC reproduzem as do Colab (pesos idênticos; 2 de 18.987 previsões
do EfficientNet-B0 diferem, em casos de fronteira, por arredondamento da GPU). Relatórios:
`results/ciclo5/avaliacao_ciclo5.md` e `results/ciclo5/ciclo4_x_ciclo5.md`.

Acurácia balanceada (%) no teste do ciclo 5, IC 95% por bootstrap em §15.7–15.9 e nos
relatórios:

| Modelo | F1 val. | Teste inteiro | Lavoura (espec. / sens.) | Lavoura no ciclo 4 | Alface 1/foto (176) | Espinafre | Fonte retida (574) |
|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | 0,972 | **97,4** | 88,8 (83,4 / 94,2) | 86,3 (76,6 / 96,0) | 96,7 | **86,6** | 87,9 |
| ViT-B/16 | 0,972 | 96,9 | 84,5 (72,1 / 96,9) | 83,4 (69,8 / 97,0) | 100 | 84,5 | 85,8 |
| MobileNetV2 | 0,960 | 96,9 | **90,1** (90,2 / 90,0) | 86,6 (78,3 / 94,9) | 100 | 79,5 | 88,4 |
| ResNet-50 | 0,961 | 96,9 | 89,6 (89,0 / 90,2) | 87,8 (81,4 / 94,2) | 100 | 83,3 | **90,1** |

O que o ciclo 5 mostra:

1. **A correção funcionou onde o problema estava.** Nas culturas fotografadas em lavoura,
   as três CNNs ganham de 1,8 a 3,5 pontos de acurácia balanceada e de 6,8 a 11,9 de
   especificidade. A amostragem por cultura desfez boa parte do atalho "lavoura → doente".
2. **Não é um ganho de graça.** A sensibilidade na lavoura cai de 1,8 a 4,9 pontos, e,
   contado por imagem, o ciclo 4 acerta mais nas CNNs (McNemar: EfficientNet p = 0,03;
   MobileNetV2 e ResNet-50 p < 10⁻⁸), porque as anômalas são maioria. No espinafre, a
   especificidade cai em todos os modelos.
3. **O ViT quase não muda** (lavoura 83,4 → 84,5, p = 0,66). Com o mesmo F1 de validação
   do EfficientNet-B0, é o pior modelo na lavoura nos dois ciclos.
4. **A validação não ordena os modelos pelo que importa.** O F1 de validação põe
   EfficientNet-B0 e ViT na frente (0,972) e MobileNetV2 e ResNet-50 atrás (0,960); na
   lavoura e na fonte retida a ordem se inverte.
5. **A alface honesta é fácil para todos** (96,7–100% em 176 fotografias), mas são 15
   doentes: o IC inferior chega a 90% e a pergunta da cultura-alvo continua em aberto até
   as fotos de campo.
6. **Fonte retida de espinafre: 85,8–90,1%** para um lote que nenhum modelo viu, contra
   ~98–99% nas fontes vistas. É a medida de mudança de fonte mais limpa que o trabalho
   tem; a queda de cerca de 10 pontos é o tamanho do problema de domínio dentro da mesma
   cultura.

Para o artigo (decisão do José com o orientador): o modelo servido pode continuar o
EfficientNet-B0 do ciclo 4 até o resultado de campo; se a horta mostrar o mesmo padrão da
lavoura (falsos alarmes em folhas sadias), o EfficientNet-B0 do ciclo 5 é o candidato,
com a troca de sensibilidade dita explicitamente.

### 15.11 No artigo (v7, 14/10/2026)

O ciclo 5 entrou no artigo v7 (`docs/Entregas Finais/TCC2_Artigo_FitoVision_v7.docx`):
um parágrafo de método na Seção 3.1, a linha 5 na Tabela 2 (campo histórico), a Tabela 3
(as quatro arquiteturas: teste, lavoura c4 → c5 com especificidade e sensibilidade,
alface uma por foto, lote retido), um parágrafo de resultados, uma frase na conclusão e
no resumo/abstract, e o lote retido no Quadro 1. O artigo continua com 11 páginas
(contadas no LibreOffice): para caber, parágrafos das Seções 2 a 4 foram enxugados sem
perda de conteúdo e as células do Quadro 2, encurtadas. Gerador:
`docs/Entregas Finais/_gerador/build_v7.py`, que parte do v6 descompactado.

Duas correções de número feitas ao escrever: as imagens de praga que entram são **8.266**
(8.284 nas pastas, 18 removidas por rótulo contraditório com uma quase-duplicata; a ficha
dizia 8.284), e uma das pastas (folha amarela da cana) é de doença, não de praga — daí
"quase todas de dano por praga".

### 15.12 Leitura do texto completo da literatura; artigo v8 e monografia v5 (16/10/2026)

Os trabalhos relacionados tinham sido lidos pelos resumos da Web of Science. Em 16/10,
sete foram lidos inteiros (`docs/REVISAO_LITERATURA_WOS.md`, Parte 4). Duas leituras
estavam erradas, e o padrão é o mesmo dos erros de número desta seção: um documento
repetia o outro sem voltar à fonte.

| # | O que estava | O que é | Onde foi corrigido |
|---|---|---|---|
| 1 | A revisão de Shafay et al. (2025) "aponta transformers como mais robustos" e o ViT-B/16 daqui "diverge" | Aponta os **híbridos** (Swin) como mais robustos e o **ViT puro como mais sensível ao ambiente**; os "88% × 53%" são do Swin no Cropped PlantDoc, treinado e testado nele. O ViT-B/16, pior na lavoura, **converge** | Artigo v8 (§2.3 e discussão), guia da banca (tabela, "confirma/contradiz", pergunta 19), monografia v5, REVISAO Parte 1 |
| 2 | Gerz e Jelali (2026): "98,7% → 32,8%" | **mAP50 de detecção** (YOLOv8n), e o artigo deles se contradiz: 32,77% no resumo, 30,64% na tabela | Guia da banca (3 trechos), monografia v5 |
| 3 | Ergün et al. (2026): ECE externo 0,287 → 0,038 citado sem ressalva | A confiança média cai para 0,55 (binário), sem acurácia seletiva; IC de ±0,7 ponto com N = 500 | Artigo v8 (discussão), guia, monografia v5 |
| 4 | Abbasi et al. (2023): "94,13% no estado de saúde" | Medido **na validação**, com imagens da web e aumento aparentemente antes da divisão | Guia, monografia v5 |

O artigo v8 (`_gerador/build_v8.py`, a partir do v7) mexe em quatro frases (as três da
literatura e, na Seção 3.2, a regra das fotografias ambíguas: marcadas e reportadas com e
sem elas, em vez de excluídas) e continua com 11 páginas no LibreOffice. O guia de campo e
o LEIA-ME de `backend/campo_novo/` seguem a mesma regra. A monografia v5 (`_gerador/build_mono_v5.py`, a partir da v4)
ganha a nota de 16/10, a seção 9.6, seis referências e o sumário com a nota de 14/10 e o
capítulo 9, que a v4 não tinha acrescentado.

### 15.13 Primeiro lote de campo com fundo real; artigo v9 e v10 (27/10/2026)

**O que foi fotografado.** Em 27/10, às 16h35, 10 fotografias numa horta (`horta_27out`
em `backend/campo_novo/metadados_campo.csv`): canteiros de alface crespa, todos os pés
sadios segundo o José, tiradas de cima, com vários pés por quadro (a horta tinha poucos
pés e eles estavam encostados, então não deu para isolar um). Fundo real: solo, mato e
trevo. Distância "ampla" em nove e "média" em uma. O lote de casa de 18/10
(`casa_colhida_18out`: um pé colhido na chuva, fotografado sobre toalha branca, 3 folhas
sadias e 7 anômalas) continua na pasta e é reportado à parte, por não ser de campo.

**Conferência** (`confere_fotos_campo.py`): nenhum problema que impeça a medição. As 10
fotos novas vêm **sem EXIF de câmera** (as de 18/10 têm: Xiaomi POCO X6 Pro 5G), embora
estejam em resolução cheia (4624×3472, 6–10 MB). Origem confirmada pelo José em 27/10:
celular → Google Drive → download no notebook → descompactação. A resolução é a mesma do
lote de 18/10 e o campo `Software` ainda diz "MediaTek Camera Application", o mesmo das
fotos do POCO X6 Pro; o envio apagou fabricante, modelo, data e exposição, mas não há
sinal de recompressão nem de edição. Duas têm um furo pequeno numa folha, anotado
como "conferir" na planilha **antes** da inferência; o rótulo continua `healthy`.

**Resultado às cegas** (`roda_campo.py`, `results/campo_novo/`):

| Versão | Horta 27/10 (10 sadias) | Casa 18/10 (3 sadias, 7 anômalas) |
|---|---|---|
| EfficientNet-B0, ciclo 4 | 10 de 10, confiança ≥ 0,996 | 7 de 10 (as 3 sadias dadas como anômalas) |
| EfficientNet-B0, ciclo 5 | 10 de 10, ≥ 0,988 | 7 de 10 |
| ViT-B/16, ciclo 4 | 10 de 10, ≥ 0,994 | 7 de 10 |
| ViT-B/16, ciclo 5 | 10 de 10, ≥ 0,986 | 7 de 9 (1 inconclusiva) |

Na horta: IC de Wilson 72,2% a 100% para a especificidade, sem abstenção. Nenhuma
diferença entre versões (McNemar, todas as discordâncias são zero).

**Recortes** (`testa_recortes_campo.py`, `results/campo_novo/recortes/`): recortes centrais
com metade e um quarto do lado de cada foto. Na horta, 10 de 10 nas duas escalas e nas
quatro versões: a resposta não depende de haver vários pés no quadro. No lote de casa, os
erros continuam nos recortes, ou seja, não vêm da toalha nas bordas.

**Como ler.** 

1. *O que o lote mostra:* fundo real (solo, mato, vários pés) não provoca falso alarme
   em alface sadia, nas quatro versões e em três enquadramentos. Era o risco mais óbvio
   do *domain gap* e não apareceu.
2. *O que o lote não mostra:* não há folha doente em campo, então a **sensibilidade em
   campo, a acurácia balanceada de campo e o *domain gap* continuam sem medida**. O
   "88,5% de acurácia balanceada" que `analise_campo.md` imprime para as 20 fotos juntas
   **não deve ser citado**: todas as anômalas são do lote de casa e quase todas as sadias
   são da horta, então a métrica mistura classe e cena.
3. *O efeito da cena não se separa do da folha.* Os modelos dizem "sadia" para tudo na
   horta e "anômala" para tudo em casa (com confiança ≥ 0,97). Pode ser a folha, pode ser
   a cena (luz artificial, folha molhada e machucada da colheita). Só fotos de folhas
   doentes **na horta** desfazem essa confusão.
4. *H3 (distância):* o Cochran-Armitage de `analise_campo.md` dá p = 0,03, mas é artefato:
   todas as fotos "amplas" são da horta e sadias, e as "close" são de casa. Não citar.
5. *Rótulo a conferir (observação posterior à inferência, não altera o gabarito):*
   `healthy/IMG_20261018_192328.jpg` mostra pontos escuros e a base amarronzada, e
   `IMG_20261018_192548.jpg` é uma folha interna quase branca. Os modelos chamam as duas
   de anômalas. Pelo protocolo, o rótulo da coleta vale; a observação vai para o José.
6. *Calibração:* os 3 erros de cada versão no lote de casa têm confiança acima do
   limiar τ, como os nove erros do conjunto histórico (Tabela 2 do artigo): fora da
   distribuição de treino, o limiar não intercepta os erros. Com 20 fotos de duas cenas,
   o ECE (0,09 a 0,14) não é estimativa útil.
7. *Recalibração:* **não foi feita**. Pelo protocolo, a adaptação usa uma horta separada
   da de avaliação, e com uma horta e uma só classe não há o que ajustar.

**No artigo.** A v9 (27/10, entrega "Postagem IV - parcial II - 5. Conclusão",
`_gerador/build_v9.py`) reescreveu a Seção 5 como conclusão completa, antes das fotos. A
v10 (`_gerador/build_v10.py`, a partir da v9) traz o lote: uma frase no resumo e no
abstract, uma na Seção 3.2, um parágrafo no fim da Seção 4 e a conclusão atualizada (Q3
respondida só em parte). Continua com 11 páginas no LibreOffice; para caber, as frases da
Seção 4 foram enxugadas sem mudar números.

**O que falta para Q3:** folhas doentes fotografadas na horta (mínimo de 15 por
cultura), uma segunda horta para a adaptação, fotos sem folha para a guarda de
vegetação, e outra cultura (couve) para H1.
