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
