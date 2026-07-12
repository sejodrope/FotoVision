# FitoVision — Compilado Técnico Completo
### Seminário III (TCC-I) · UNIVILLE · 2026

> Gerado automaticamente em 06/06/2026. Use como base para a apresentação e para os capítulos de metodologia e resultados do TCC.

---

## 0. ERRATA — Invalidação dos resultados de 06/06/2026

> **Todas as métricas deste documento (99,01%, 98,81%, 98,71%) estão RETRATADAS.**
> Foram medidas sobre um conjunto de teste contaminado. As secções 3, 4, 7.1 e 7.2
> não devem ser citadas até que o experimento seja refeito.
>
> 📌 **O registo completo da investigação — hipóteses testadas, causa, as 9 correcções e
> a verificação experimental de cada uma — está em
> [`CORRECOES_METODOLOGICAS.md`](CORRECOES_METODOLOGICAS.md).**
> Esta secção 0 é o resumo; aquele documento é a fonte para o capítulo de metodologia.

### 0.1 O que aconteceu

Ao testar o sistema com fotos novas, o desempenho não correspondeu de todo aos 99%
reportados. A investigação encontrou **vazamento de dados (*data leakage*)** entre os
conjuntos de treino e teste.

O split era feito com um sorteio **ao nível do ficheiro**:

```python
images = sorted(f for f in src.iterdir() ...)
rng.shuffle(images)                      # ← sorteio por FICHEIRO
train, val, test = images[:a], images[a:b], images[b:]
```

Acontece que o pool de imagens contém **várias cópias da mesma fotografia**:

| Fonte | Natureza da redundância |
|-------|-------------------------|
| `lettuce-disease-multi-transformation-dataset` | o nome diz tudo: são versões **rodadas, espelhadas e com brilho alterado** da mesma folha |
| `plant-diseases-training-dataset` | é um **re-upload do PlantVillage** — as mesmas imagens, sob outros nomes de ficheiro |

O sorteio por ficheiro colocava, portanto, a **rotação** de uma foto no treino e o
**espelhamento da mesma foto** no teste. Como os nomes dos ficheiros eram diferentes,
nada detectava a duplicação.

**O modelo não estava a diagnosticar folhas. Estava a reconhecer fotografias que já
tinha visto.** Os 99,01% mediam **memorização**, não capacidade de generalização.

### 0.2 O indício que foi mal interpretado

A secção 7.1 apresentava como prova de qualidade o facto de `val_acc ≈ test_acc`:

> *"A diferença entre val_acc e test_acc é mínima (…) Isso confirma que os modelos
> generalizaram bem e não sofreram overfitting."*

**Esta inferência está invertida.** Validação e teste concordavam porque **ambos
estavam contaminados pelo treino**. A concordância entre dois conjuntos igualmente
vazados não é evidência de generalização — é um *sintoma* do vazamento. Um teste
honesto teria mostrado uma queda, e foi a ausência dessa queda que deveria ter
levantado suspeitas.

Vale registar isto no TCC: é precisamente o tipo de raciocínio que a validação
experimental existe para apanhar.

### 0.3 A correcção

| # | Problema | Correcção | Ficheiro |
|---|----------|-----------|----------|
| 1 | Split sorteava ficheiros; variantes da mesma foto caíam em splits diferentes | **Split agrupado por identidade visual** — hash perceptual invariante a rotação/espelho/brilho agrupa as variantes, e sorteiam-se **grupos inteiros** | `imagehash_utils.py`, `download_datasets.py` |
| 2 | Sem forma de detectar o vazamento | **Auditoria** que quantifica quantas imagens de teste têm duplicado no treino | `audit_leakage.py` |
| 3 | Softmax devolvia ~99% de confiança para qualquer imagem, inclusive não-folhas | **Temperature scaling** + **limiar de abstenção** (`inconclusive`) + **guarda de vegetação** (`not_a_leaf`) | `calibrate.py`, `app/ml/inference.py` |
| 4 | Classes desbalanceadas; melhor checkpoint escolhido por *accuracy* | **Loss com pesos de classe**, amostragem balanceada, selecção por **F1 macro** | `train.py`, `dataset.py` |
| 5 | Treino recortava a folha; inferência **esmagava** a proporção (`Resize((224,224))`) | `Resize(256) + CenterCrop(224)` em ambos | `dataset.py`, `app/ml/preprocessing.py` |
| 6 | Imagem corrompida virava **quadrado preto com o label original** | Imagem ilegível é **descartada** (reamostra-se outra da mesma classe) | `dataset.py` |
| 7 | Roboflow: *qualquer* imagem anotada → `anomalous` (mesmo quando a caixa marcava uma folha **saudável**) | Lê as classes do `data.yaml` e mapeia cada `class_id` | `download_datasets.py` |
| 8 | Rótulos por palavra-chave incluíam `weed`, `good`, `normal`, `target` | Casamento por *token*; termos genéricos removidos; mapa auditável em `data/label_map_audit.json` | `download_datasets.py` |

A correcção do item 1 foi **verificada experimentalmente**: num conjunto sintético com
40 fotos-base e 7 cópias transformadas de cada, o split antigo vazava **100%** do teste;
o novo vaza **0%**, recuperando exactamente as 40 fotos de origem.

### 0.4 Como refazer o experimento

```bash
cd backend
python download_datasets.py --skip-download   # refaz o split, agora agrupado
python audit_leakage.py                       # quantifica o vazamento do split antigo
python run_pipeline.py                        # treina → calibra → avalia
```

**Expectativa honesta:** as métricas vão **cair**, e devem cair. Um valor na casa dos
85–95% de acurácia balanceada, obtido sem vazamento, vale infinitamente mais — e é
defensável numa banca — do que 99% que não sobrevivem ao primeiro contacto com uma
foto real.

Reportar a **acurácia balanceada** (`results/metrics_comparison.csv`) e citar
`results/leakage_report.json` como justificação da retratação.

---

## 1. Visão Geral do Sistema

### Problema
Doenças fitossanitárias em hortaliças folhosas (alface, rúcula, espinafre, acelga, couve) causam perdas significativas na produção agrícola. A identificação visual precoce exige experiência agronômica e tempo. Pequenos produtores frequentemente não têm acesso a esse diagnóstico.

### Solução — FitoVision
Sistema de visão computacional que recebe uma **foto de folha** e classifica automaticamente em:

- **healthy** (saudável) — folha sem anomalias visíveis
- **anomalous** (anômala) — folha com sintomas de doença, praga ou estresse

A inferência ocorre em menos de 25 ms na GPU. O sistema é acessado via interface web (upload de imagem → resultado instantâneo).

### Culturas-alvo
Hortaliças folhosas: alface, rúcula, espinafre, acelga, couve.  
O dataset inclui também imagens proxy de outras culturas (PlantVillage) para enriquecer a representação de padrões de doença.

### Arquitetura Geral

```
[Usuário — browser]
        │  upload JPG/PNG/WebP
        ▼
[Frontend React]          ← localhost:5173
        │  POST /api/predict/ (multipart)
        ▼
[FastAPI Backend]         ← localhost:8000
        │
        ├─ preprocess_image()  → Resize 224×224 + normalização ImageNet
        │
        ├─ load_binary_model() → carrega EfficientNet-B0 (CUDA/CPU)
        │
        └─ predict_binary()    → { label, confidence, healthy_prob, anomalous_prob }
        │
        ▼
[Frontend] exibe resultado:
  - Classificação (Saudável / Anômala)
  - Gauge circular de confiança (SVG)
  - Barras de probabilidade por classe
  - Badge: modelo + acurácia
```

### Stack Tecnológica Completa

| Camada | Tecnologia | Versão |
|--------|-----------|--------|
| **Backend ML** | PyTorch | 2.5.1+cu121 |
| **GPU** | NVIDIA RTX 3050 Laptop | 6 GB VRAM, CUDA 12.3 |
| **API** | FastAPI + Uvicorn | — |
| **ORM / DB** | SQLAlchemy async + SQLite (aiosqlite) | — |
| **Validação** | Pydantic v2 + pydantic-settings | — |
| **Frontend** | React 19 + TypeScript | Vite 6, TSC 5.7 |
| **Estilo** | Tailwind CSS 3 + Inter font | — |
| **HTTP client** | Axios | — |
| **State / Cache** | TanStack React Query v5 | — |
| **Ícones** | Lucide React | — |
| **OS treino** | Windows 11 Education | — |
| **Python** | 3.12 | venv em `backend/.venv` |

---

## 2. Dataset

### 2.1 Fontes de Dados

| Fonte | Plataforma | Imagens úteis | Papel |
|-------|-----------|---------------|-------|
| `abdallahalidev/plantvillage-dataset` | Kaggle | ~54.000 (subconjunto selecionado) | Proxy visual — padrões de doença em folhas diversas |
| `ashishjstar/lettuce-diseases` | Kaggle | 2.337 | Dataset primário — alface (Bacterial, Downy Mildew, Healthy) |
| `shuvokumarbasak2030/lettuce-disease-multi-transformation-dataset` | Kaggle | 79.458 | Dataset primário — alface com 14 tipos de augmentation |
| `nirmalsankalana/plant-diseases-training-dataset` | Kaggle | 116.147 | Diversidade adicional — 70+ classes de doença em 20+ culturas |
| Roboflow (20 datasets) | Roboflow | **0** | Todos de detecção de objetos (bounding box); inutilizáveis para classificação |

> **Nota metodológica:** Os datasets do Roboflow foram descartados porque eram de detecção de objetos (YOLOv8), não de classificação. A conversão foi tentada e retornou 0 imagens úteis — resultado documentado como decisão baseada em evidência.

### 2.2 Distribuição Final por Classe

| Classe | Imagens | % do total |
|--------|---------|-----------|
| healthy | 71.052 | 33,5% |
| anomalous | 139.780 | 66,5% |
| **TOTAL** | **210.832** | 100% |

**Razão de desbalanceamento:** 1:1,97 (≈ 2:1)

> ⚠️ **Correcção.** Este documento afirmava que o desbalanceamento era tratado com
> `WeightedRandomSampler`. **Não era** — no pipeline binário. O `WeightedRandomSampler`
> existia em `dataset.py`, mas só era usado pelo caminho multi-classe; o binário
> (`make_binary_folder_loaders`) usava `shuffle=True` simples, sem qualquer
> compensação, e a loss não tinha pesos de classe.
>
> Com `anomalous` a dominar 2:1, o modelo tinha um incentivo real a inclinar-se para
> "anomalous" — o que é consistente com fotos aleatórias serem classificadas como
> doentes. **Corrigido:** o pipeline binário passou a usar amostragem balanceada
> **e** `CrossEntropyLoss(weight=...)`.

### 2.3 Distribuição por Split (70/15/15 estratificado) — ❌ INVÁLIDA

> ⚠️ **Estes números descrevem o split contaminado** (ver Secção 0 — Errata).
>
> Duas ressalvas sobre a tabela abaixo:
>
> 1. **O total conta cópias, não fotos.** As 210.832 "imagens" incluem múltiplas
>    variantes transformadas da mesma fotografia. O número de **fotos distintas** é
>    substancialmente menor e só é conhecido após o agrupamento — fica registado em
>    `data/split_metadata.json` (campo `n_distinct_photos`).
>
> 2. **A afirmação "nunca usado durante otimização" é enganosa.** É verdade que os
>    ficheiros de teste não foram lidos durante o treino — mas *variantes das mesmas
>    fotos* foram. Para efeitos de generalização, o modelo tinha visto o conteúdo do
>    teste.

| Split | healthy | anomalous | Total |
|-------|---------|-----------|-------|
| ~~train~~ | ~~49.736~~ | ~~97.846~~ | ~~147.582~~ |
| ~~val~~ | ~~10.658~~ | ~~20.967~~ | ~~31.625~~ |
| ~~test~~ | ~~10.658~~ | ~~20.967~~ | ~~31.625~~ |

**Split corrigido (group-aware):** sorteiam-se **grupos de identidade visual**, não
ficheiros. Todas as variantes de uma foto caem no mesmo split, por construção. Números
reais em `data/split_metadata.json` após correr `download_datasets.py --skip-download`.

### 2.4 Data Augmentation

**Treino** (torchvision transforms) — *versão corrigida*:
```python
RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.333))
RandomHorizontalFlip()
RandomVerticalFlip()
RandomRotation(30)
ColorJitter(brightness=0.3, contrast=0.3,
            saturation=0.15, hue=0.02)       # ← saturação/matiz REDUZIDAS
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225])
RandomErasing(p=0.25, scale=(0.02, 0.15))    # ← oclusão aleatória (novo)
```

**Val / Test / Inferência** — *versão corrigida*:
```python
Resize(256)          # ← preserva a proporção
CenterCrop(224)
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

> ⚠️ **Duas correcções, ambas relevantes para o desempenho em fotos reais.**
>
> **(a) A geometria de val/test estava inconsistente com a do treino.** A versão
> anterior usava `Resize((224, 224))` — um resize para um alvo quadrado, que **esmaga a
> proporção** da imagem. O treino, porém, usava `RandomResizedCrop`, que *recorta*
> preservando a proporção. O modelo treinava em folhas com geometria correcta e era
> servido, em produção, com folhas achatadas. Numa foto de telemóvel (4:3 ou 16:9) a
> distorção é severa — uma folha redonda chega ao modelo como uma elipse.
> `Resize(256) + CenterCrop(224)` (convenção ImageNet) elimina a discrepância, e a
> mesma transformação passou a ser usada em `app/ml/preprocessing.py`.
>
> **(b) O `ColorJitter` estava a apagar o sinal a detectar.** Com `saturation=0.3` e
> `hue=0.05`, o augmentation perturbava agressivamente matiz e saturação — mas clorose
> (amarelecimento), míldio e oídio (manchas) **são definidos precisamente por desvio de
> cor**. Perturbar cor com essa intensidade ensina o modelo a ignorar exactamente a
> evidência de que precisa. Brilho e contraste, que modelam variação de *iluminação* e
> não de *doença*, foram mantidos generosos.

**Justificativa científica:** Augmentation é regularização — o modelo aprende features
robustas a variações de ângulo, iluminação e escala que ocorrem naturalmente em campo.
Mas a augmentation tem de perturbar as **variáveis de ruído** (pose, exposição,
enquadramento), nunca a **variável de decisão** (a cor e a textura da lesão).

---

## 3. Modelos Treinados

> ## ⚠️ Métricas de qualidade desta secção: RETRATADAS
> Todos os valores de **accuracy, F1, precision e recall** apresentados nas subsecções
> 3.1–3.3 foram medidos sobre o split contaminado (ver **Secção 0 — Errata**) e **não
> devem ser citados**.
>
> Permanecem válidos: arquitecturas, contagens de parâmetros, hiperparâmetros, tempos
> de treino e latências — nenhum depende do split.

### Configuração Comum de Treino

| Hiperparâmetro | Valor | Justificativa |
|----------------|-------|---------------|
| Epochs | 30 | Suficiente para convergência; early stopping interrompe antes se necessário |
| Otimizador | AdamW | Adam com L2 correto (Loshchilov & Hutter, 2019) |
| Learning Rate | 1e-3 | Padrão para fine-tuning de modelos ImageNet |
| Weight Decay | 1e-4 | Regularização L2 |
| Scheduler | CosineAnnealingLR | Decaimento suave: eta_min = lr × 0.01 |
| Loss | CrossEntropyLoss | label_smoothing=0.1 — melhora calibração das probabilidades |
| Early Stopping | patience=10 | Interrompe se val_acc não melhora por 10 épocas |
| Precisão | AMP (float16) | ~2× mais rápido na GPU sem perda de acurácia |
| Workers | 2 | DataLoader paralelo |
| GPU | RTX 3050 6GB + CUDA 12.3 | |

---

### 3.1 MobileNetV2

**Arquitetura:**
- Parâmetros: **3,4 milhões**
- Inovação: bottleneck invertido + depthwise separable convolutions
- Projetado para dispositivos móveis — baixa latência
- Publicação: Sandler et al. (2018), CVPR

**Por que escolhido:** Baseline ideal — menor modelo, mais rápido, referência de eficiência.

**Configuração:** batch_size=32 (padrão), epochs=30

**Curva de Aprendizado (épocas selecionadas):**

| Época | Train Loss | Train Acc | Val Loss | Val Acc | Melhor? |
|-------|-----------|-----------|----------|---------|---------|
| 1 | 0,3089 | 93,69% | 0,2715 | 95,83% | ✅ |
| 2 | 0,2746 | 95,61% | 0,2559 | 96,69% | ✅ |
| 3 | 0,2624 | 96,33% | 0,2443 | 97,31% | ✅ |
| 5 | 0,2513 | 96,92% | 0,2346 | 97,97% | ✅ |
| 10 | ~0,237 | ~97,5% | ~0,234 | ~98,0% | vários |
| 24 | — | — | — | **98,86%** | ✅ (melhor) |
| 25–30 | fine-tuning | — | — | ~98,6–98,8% | — |

**Modelo salvo:** `weights/mobilenet_v2_binary.pth` (val_acc = **98,86%** @ época 24)  
**Tempo total:** 55.393 s = **15,4 horas** (inclui overhead da época 1 com cold cache e throttle noturno do laptop)

**Métricas no Test Set (31.625 imagens):**

| Métrica | Geral | healthy | anomalous |
|---------|-------|---------|-----------|
| Accuracy | **98,81%** | — | — |
| F1 (macro) | **98,67%** | 98,23% | 99,10% |
| Precision (macro) | **98,69%** | 98,32% | 99,06% |
| Recall (macro) | **98,65%** | 98,15% | 99,15% |
| Latência | **14,3 ms** | — | — |

**Matriz de Confusão:**

|  | Pred: healthy | Pred: anomalous |
|--|--------------|----------------|
| **Real: healthy** (10.658) | 10.461 ✅ | 197 ❌ |
| **Real: anomalous** (20.967) | 179 ❌ | 20.788 ✅ |

Falsos negativos (anomalous → healthy): **179** — mais crítico para uso agronômico (doença não detectada).

---

### 3.2 EfficientNet-B0

**Arquitetura:**
- Parâmetros: **5,3 milhões**
- Inovação: compound scaling — profundidade, largura e resolução escaladas simultaneamente com um coeficiente único
- Melhor acurácia/parâmetro no ImageNet (2019)
- Publicação: Tan & Le (2019), ICML — "EfficientNet: Rethinking Model Scaling for CNNs"

**Por que escolhido:** Representante da eficiência moderna — mais parâmetros que MobileNetV2 mas compound scaling superior.

**Configuração:** batch_size não registrado (estimativa: 32 padrão), epochs=30

**Curva de Aprendizado (épocas selecionadas):**

| Época | Train Loss | Train Acc | Val Loss | Val Acc | Melhor? |
|-------|-----------|-----------|----------|---------|---------|
| 1 | 0,2725 | 95,83% | 0,2532 | 96,92% | ✅ |
| 5 | 0,2320 | 98,01% | 0,2252 | 98,42% | ✅ |
| 9 | 0,2247 | 98,42% | 0,2201 | 98,65% | ✅ |
| 13 | 0,2186 | 98,76% | 0,2178 | 98,79% | ✅ |
| 18 | 0,2139 | 99,06% | 0,2164 | 98,90% | ✅ |
| 20 | 0,2118 | 99,20% | 0,2163 | **98,98%** | ✅ (melhor) |
| 25 | 0,2083 | 99,42% | 0,2161 | 98,94% | — |
| 30 | 0,2065 | 99,53% | 0,2172 | 98,94% | — |

Observação: atingiu plateau a partir da época 20. Train_acc continua subindo mas val_acc estabiliza → comportamento esperado de fine-tuning bem regularizado.

**Modelo salvo:** `weights/efficientnet_b0_binary.pth` (val_acc = **98,98%** @ época 20)  
**Tempo total:** 23.244 s = **6,46 horas**

**Métricas no Test Set (31.625 imagens):**

| Métrica | Geral | healthy | anomalous |
|---------|-------|---------|-----------|
| Accuracy | **99,01%** | — | — |
| F1 (macro) | **98,89%** | 98,53% | 99,25% |
| Precision (macro) | **98,93%** | 98,70% | 99,17% |
| Recall (macro) | **98,85%** | 98,36% | 99,34% |
| Latência | **24,1 ms** | — | — |

**Matriz de Confusão:**

|  | Pred: healthy | Pred: anomalous |
|--|--------------|----------------|
| **Real: healthy** (10.658) | 10.483 ✅ | 175 ❌ |
| **Real: anomalous** (20.967) | 138 ❌ | 20.829 ✅ |

Falsos negativos: **138** — o menor dos 3 modelos (melhor detecção de doenças).

---

### 3.3 ResNet50

**Arquitetura:**
- Parâmetros: **25,6 milhões**
- Inovação: conexões residuais (skip connections) — resolvem o vanishing gradient em redes profundas (>50 camadas)
- Ganhou o ILSVRC 2015 (ImageNet Large Scale Visual Recognition Challenge)
- Publicação: He et al. (2016), CVPR — "Deep Residual Learning for Image Recognition"

**Por que escolhido:** Referência clássica de alto desempenho — permite avaliar se mais parâmetros (25,6M vs 5,3M) se traduz em melhor acurácia no domínio folhosas.

**Configuração:** batch_size=64, epochs=30, via `run_pipeline.py`

**Curva de Aprendizado (épocas selecionadas):**

| Época | Train Loss | Train Acc | Val Loss | Val Acc | Melhor? |
|-------|-----------|-----------|----------|---------|---------|
| 1 | 0,3459 | 91,47% | 0,2905 | 95,03% | ✅ |
| 5 | 0,2647 | 96,20% | 0,2503 | 96,99% | ✅ |
| 10 | 0,2439 | 97,36% | 0,2369 | 97,65% | ✅ |
| 15 | 0,2333 | 97,94% | 0,2270 | 98,31% | ✅ |
| 20 | 0,2247 | 98,43% | 0,2216 | 98,57% | ✅ |
| 25 | 0,2189 | 98,75% | 0,2200 | 98,68% | ✅ |
| 30 | 0,2161 | 98,93% | 0,2193 | **98,71%** | ✅ (melhor) |

Observação crítica: **ResNet50 ainda estava melhorando na época 30 (sem plateau)**. Convergência mais lenta que EfficientNet, confirmando que mais parâmetros não implicam convergência mais rápida.

**Modelo salvo:** `weights/resnet50_binary.pth` (val_acc = **98,71%** @ época 30)  
**Tempo total:** 39.721 s = **11,03 horas**

**Métricas no Test Set (31.625 imagens):**

| Métrica | Geral | healthy | anomalous |
|---------|-------|---------|-----------|
| Accuracy | **98,71%** | — | — |
| F1 (macro) | **98,56%** | 98,09% | 99,03% |
| Precision (macro) | **98,54%** | 97,99% | 99,08% |
| Recall (macro) | **98,59%** | 98,20% | 98,97% |
| Latência | **16,7 ms** | — | — |

**Matriz de Confusão:**

|  | Pred: healthy | Pred: anomalous |
|--|--------------|----------------|
| **Real: healthy** (10.658) | 10.466 ✅ | 192 ❌ |
| **Real: anomalous** (20.967) | 215 ❌ | 20.752 ✅ |

Falsos negativos: **215** — o pior dos 3 modelos na detecção de doenças.

---

## 4. Comparação e Modelo de Produção

> ## ⚠️ SECÇÃO RETRATADA
> As métricas abaixo foram medidas sobre um test set contaminado por duplicados do
> treino (ver **Secção 0 — Errata**). **Não citar.** A afirmação "nunca visto durante
> treino ou validação" é falsa: uma fracção substancial destas 31.625 imagens são
> variantes por *augmentation* de fotos presentes no treino.
>
> Mantidas apenas como registo histórico do que foi corrigido.

### 4.1 Tabela Comparativa Final (`metrics_comparison.csv`) — ❌ INVÁLIDA

> ~~Avaliação realizada em 06/06/2026 com `evaluate.py --binary` no test set de 31.625 imagens (nunca visto durante treino ou validação).~~

| Modelo | Parâmetros | ~~Accuracy~~ | ~~F1 (macro)~~ | ~~Precision~~ | ~~Recall~~ | Latência (ms) | Treino (h) |
|--------|-----------|----------|------------|-----------|--------|--------------|-----------|
| **EfficientNet-B0** | **5,3M** | ~~99,01%~~ | ~~98,89%~~ | ~~98,93%~~ | ~~98,85%~~ | 24,1 | 6,46 |
| MobileNetV2 | 3,4M | ~~98,81%~~ | ~~98,67%~~ | ~~98,69%~~ | ~~98,65%~~ | 14,3 | 15,4* |
| ResNet50 | 25,6M | ~~98,71%~~ | ~~98,56%~~ | ~~98,54%~~ | ~~98,59%~~ | 16,7 | 11,03 |

*MobileNetV2 incluiu overhead de época 1 com laptop throttle; tempo real de treino estável ~6–7h.

> **Nota.** A latência e a contagem de parâmetros **permanecem válidas** — não dependem
> do split. Só as métricas de qualidade (accuracy, F1, precision, recall) foram
> invalidadas pelo vazamento.
>
> Como as três arquitecturas foram avaliadas sobre o *mesmo* test set contaminado, é
> plausível que o **ranking relativo** se mantenha; mas isso é uma conjectura, não um
> resultado — o vazamento pode favorecer arquitecturas com maior capacidade de
> memorização (ResNet50 tem 4,8× mais parâmetros). O ranking tem de ser re-medido.

### 4.2 Análise por Classe (Falsos Negativos — mais crítico)

| Modelo | FN (doença não detectada) | FP (falso alarme) |
|--------|--------------------------|-------------------|
| **EfficientNet-B0** | **138** ✅ melhor | 175 |
| MobileNetV2 | 179 | 197 |
| ResNet50 | 215 ❌ pior | 192 |

**EfficientNet-B0 tem o menor número de falsos negativos** — o erro mais crítico no contexto agronômico (deixar uma planta doente ser classificada como saudável).

### 4.3 Modelo de Produção: EfficientNet-B0

**Justificativa:**
1. **Melhor accuracy** em todas as métricas (+0,20 pp sobre MobileNetV2, +0,30 pp sobre ResNet50)
2. **Melhor F1 macro** — importante com desbalanceamento 2:1 entre classes
3. **Menor taxa de falsos negativos** (138 vs 179 vs 215)
4. **Eficiência paramétrica** — 5,3M parâmetros vs 25,6M do ResNet50, com acurácia superior em todas as métricas
5. **Latência aceitável** — 24,1 ms por imagem é imperceptível para o usuário (< 30 fps)
6. **Tempo de treino inferior ao ResNet50** — 6,46h vs 11,03h

**Resultado empírico:** ResNet50, apesar de ter 4,8× mais parâmetros que EfficientNet-B0, ficou em último lugar. Isso **confirma empiricamente a tese de Tan & Le (2019)** de que compound scaling é superior ao simples aumento de profundidade/parâmetros.

---

## 5. Pipeline Completo

### 5.1 Sequência de Scripts

```
1. download_datasets.py     ← baixa Kaggle/Roboflow; organiza em data/train|val|test/
                                 healthy/ e anomalous/
        ↓
2. check_dataset.py         ← verifica estrutura, conta imagens, detecta desbalanceamento
        ↓
3. train.py --binary        ← treina modelo(s); salva pesos em weights/ e histórico em logs/
        ↓
4. evaluate.py --binary     ← avalia no test set; gera CSV, JSON, PNGs de confusion matrix
        ↓
5. run.py                   ← inicia servidor FastAPI (backend)
        +
   npm run dev (frontend)   ← inicia dev server React
```

Pipeline automatizado para EfficientNet → ResNet50:
```
run_pipeline.py   ← aguarda EfficientNet, avalia, treina ResNet50, avalia final
```

### 5.2 Scripts Principais do Backend

| Arquivo | Função |
|---------|--------|
| `download_datasets.py` | Download via Kaggle API; copia para `staging/`; organiza em `data/train\|val\|test/healthy\|anomalous/` via BFS recursiva |
| `check_dataset.py` | Diagnóstico do dataset: conta imagens por split/label, detecta desbalanceamento, gera grid visual 4×4 |
| `prepare_data.py` | DataLoaders com augmentation albumentations (alternativa ao torchvision) |
| `dataset.py` | Classes `BinaryFolderDataset`, `PlantDataset`; `WeightedRandomSampler`; transforms |
| `train.py` | Fine-tuning com AdamW + CosineAnnealingLR + AMP + early stopping |
| `evaluate.py` | Accuracy, F1, Precision, Recall, Latência; confusion matrix; gráficos |
| `run_pipeline.py` | Orquestrador sequencial: aguarda → avalia → treina → avalia |
| `run.py` | `uvicorn app.main:app` (porta 8000) |
| `app/main.py` | FastAPI app, CORS, routers |
| `app/config.py` | Settings (pydantic), CLASS_NAMES, AVAILABLE_MODELS |
| `app/ml/inference.py` | `load_binary_model()` (EfficientNet-B0 em produção) + `predict_binary()` |
| `app/ml/preprocessing.py` | `preprocess_image()`: bytes → tensor 1×3×224×224 normalizado |
| `app/api/routes/predict.py` | `POST /api/predict/`: recebe imagem, retorna `{label, confidence, healthy_prob, anomalous_prob}` |
| `app/db/models.py` | SQLAlchemy models (histórico de diagnósticos) |

### 5.3 Como Rodar do Zero

```bash
# ── 1. Clonar e preparar ambiente ──────────────────────────────────────────
git clone <repo>
cd FitoVision/backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install torch==2.5.1+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn[standard] sqlalchemy aiosqlite pydantic-settings \
            python-dotenv pillow pandas scikit-learn matplotlib seaborn tqdm \
            kaggle albumentations

# ── 2. Configurar credenciais Kaggle ───────────────────────────────────────
# Criar backend/.env com:
# KAGGLE_USERNAME=seu_usuario
# KAGGLE_KEY=sua_chave

# ── 3. Dataset ─────────────────────────────────────────────────────────────
python download_datasets.py          # ~2–3h (download ~5 GB)
python check_dataset.py              # verificar distribuição

# ── 4. Treinar ─────────────────────────────────────────────────────────────
python train.py --data ./data --binary --model efficientnet_b0 \
                --epochs 30 --batch-size 32 --workers 2

# ── 5. Avaliar ─────────────────────────────────────────────────────────────
python evaluate.py --test-split ./logs/test_split_binary.json --binary

# ── 6. Iniciar servidor ────────────────────────────────────────────────────
python run.py

# ── 7. Frontend ────────────────────────────────────────────────────────────
cd ../frontend
npm install
npm run dev
# Aceder em http://localhost:5173
```

---

## 6. Frontend / MVP

### 6.1 Visão Geral

Interface web single-page desenvolvida em React 19 + TypeScript + Tailwind CSS 3, com design minimalista usando paleta **verde sage** (muted forest green, `#267350`).

**Tecnologias:**
- React 19, TypeScript 5.7, Vite 6
- TailwindCSS 3 com cores customizadas
- TanStack React Query v5 (estado assíncrono)
- Axios (HTTP)
- Lucide React (ícones)
- Inter font (Google Fonts)

### 6.2 Tela Principal (HomePage)

**Seção Hero:**
- Logo folha verde + título "Diagnóstico Fitossanitário"
- Subtítulo: "Detecção automática de anomalias em folhosas · Transfer Learning · Classificação binária"
- ~~Strip de 3 estatísticas: Acurácia (99,01%), Total de imagens (210.832)~~ → **removidas
  da interface**: eram os números inválidos da Secção 0. O strip mostra agora
  "em reavaliação" até que o novo treino produza métricas honestas.

**Coluna esquerda — Upload:**
- `ImageUploader`: drag-and-drop ou click para selecionar imagem (JPG, PNG, WebP, BMP, máx. 10 MB)
- Botão "Analisar folha" (desabilitado sem imagem ou durante processamento)
- Banner de erro com ícones contextuais (servidor offline, arquivo inválido, etc.)

**Coluna direita — Resultado:**
- *Placeholder* enquanto sem imagem
- *Loading* com spinner animado enquanto processa
- `BinaryResultCard`, agora com **quatro** resultados possíveis (antes eram dois):

| Resultado | Quando | Aparência |
|-----------|--------|-----------|
| **Saudável** | confiança calibrada ≥ limiar | gauge verde |
| **Anômala** | confiança calibrada ≥ limiar | gauge vermelho |
| **Inconclusivo** | confiança abaixo do limiar | ícone âmbar + instrução ao utilizador |
| **Não é uma folha** | a imagem não contém vegetação (ExG) | ícone cinza + instrução |

> Os dois últimos são **novos e deliberados**. A versão anterior devolvia sempre um
> veredicto binário, com confiança alta, mesmo para imagens que não eram folhas.
> **Abster-se é a resposta correcta quando o modelo não sabe** — e é isso que distingue
> um sistema honesto de um que mente com convicção.

### 6.3 Como a Inferência Funciona

```
[Usuário seleciona imagem]
        ↓
[Frontend: runPredict(file)]        → axios POST /api/predict/ (multipart)
        ↓
[Backend: predict.py router]
        → valida content-type (JPEG/PNG/WebP/BMP)
        → valida tamanho (máx. 10 MB)
        → preprocess_image(bytes):
             PIL.Image → RGB → Resize(256) → CenterCrop(224)   ← preserva a proporção
             → ToTensor() → Normalize(ImageNet) → tensor[1,3,224,224]
        → predict_binary(tensor, image):
             (1) GUARDA DE DOMÍNIO — índice ExG (Excess Green)
                 vegetação < 10%  →  'not_a_leaf'  (não é folha: não arrisca diagnóstico)
             (2) model(tensor) → logits[1,2]
             (3) CALIBRAÇÃO — softmax(logits / T), com T aprendido em calibrate.py
                 (T > 1 achata as probabilidades: corrige o excesso de confiança)
             (4) ABSTENÇÃO — confiança < limiar  →  'inconclusive'
             (5) caso contrário → label = argmax
        → retorna JSON: { label, confidence, healthy_prob, anomalous_prob,
                          calibrated, vegetation_fraction, message }
        ↓
[Frontend: BinaryResultCard] exibe um dos 4 estados
```

### 6.4 Outros Componentes Existentes

| Componente | Localização | Função |
|------------|-------------|--------|
| `Navbar` | `components/layout/Navbar.tsx` | Barra de navegação com logo FitoVision |
| `ModelInfoModal` | `components/layout/ModelInfoModal.tsx` | Modal com métricas do modelo ativo |
| `ImageUploader` | `components/diagnosis/ImageUploader.tsx` | Drag-and-drop de imagem |
| `DiagnosisCard` | `components/diagnosis/DiagnosisCard.tsx` | Card para diagnóstico multi-classe (legado) |
| `ConfidenceBar` | `components/diagnosis/ConfidenceBar.tsx` | Barra de confiança |
| `GradCamViewer` | `components/diagnosis/GradCamViewer.tsx` | Visualização Grad-CAM |
| `ModelComparison` | `components/dashboard/ModelComparison.tsx` | Gráfico comparativo de modelos |
| `MetricsCard` | `components/dashboard/MetricsCard.tsx` | Card de métricas individuais |

---

## 7. Resultados e Análise

### 7.1 Significado dos Resultados para o TCC — ❌ RETRATADA

> **Esta secção continha um erro de raciocínio, e vale a pena preservá-lo como lição.**

O texto original argumentava:

> ~~"Os três modelos superaram 98,7% de acurácia no test set de 31.625 imagens.~~
> ~~A diferença entre val_acc e test_acc é mínima (EfficientNet-B0: 98,98% → 99,01%).~~
> ~~Isso confirma que os modelos generalizaram bem e não sofreram overfitting."~~

**A conclusão está invertida.** A validação e o teste concordavam porque **ambos
continham duplicados do treino**. A concordância entre dois conjuntos igualmente
contaminados não demonstra generalização: é um *sintoma* do vazamento.

O sinal de alarme estava à vista e foi lido ao contrário. Um modelo que atinge 99% num
domínio visualmente difícil, com val e test a coincidirem até à segunda casa decimal, é
**suspeito**, não excelente. A ausência da queda esperada entre validação e teste era a
prova de que o teste não era independente.

**Lição metodológica para o TCC:** métricas boas demais são uma hipótese a testar, não
um resultado a celebrar. A primeira pergunta perante um 99% deve ser *"o que é que o
conjunto de teste tem que não devia ter?"* — e a resposta chega-se com uma auditoria de
duplicados (`audit_leakage.py`), não com mais uma época de treino.

### 7.2 Comparação com a Literatura — ❌ RETRATADA

> As linhas do FitoVision nesta tabela **não são comparáveis** com a literatura: os
> valores publicados por Mohanty, Too e Atila foram obtidos em splits sem vazamento; os
> do FitoVision, não. Comparar os dois é comparar generalização com memorização.

| Referência | Tarefa | Dataset | Accuracy | F1 |
|------------|--------|---------|---------|-----|
| Mohanty et al. (2016) | 26 classes PlantVillage | PlantVillage | 99,35% | — |
| Too et al. (2019) | Multi-classe Plant Disease | PlantVillage | 98,8% | — |
| Atila et al. (2021) | Plant Disease (5 classes) | PlantVillage | 97,4% | — |
| ~~FitoVision — EfficientNet-B0~~ | ~~binário (folhosas)~~ | ~~multi-fonte~~ | ~~99,01%~~ | ~~98,89%~~ |
| ~~FitoVision — MobileNetV2~~ | ~~binário (folhosas)~~ | ~~multi-fonte~~ | ~~98,81%~~ | ~~98,67%~~ |
| ~~FitoVision — ResNet50~~ | ~~binário (folhosas)~~ | ~~multi-fonte~~ | ~~98,71%~~ | ~~98,56%~~ |

**Contexto importante para a discussão do TCC.** Mesmo os trabalhos acima, medidos
correctamente, são conhecidos por **não se transferirem para condições de campo**: a
literatura reporta que modelos treinados em PlantVillage, com ~99% no seu próprio test
set, caem para a faixa dos **30–60%** em fotografias reais de lavoura (fundo de terra,
iluminação natural, múltiplas folhas, oclusão). O *gap de domínio* é um resultado
estabelecido — e é a razão de fundo pela qual um sistema treinado em fotos de estúdio
decepciona numa foto de telemóvel, **mesmo depois de eliminado o vazamento**.

Isto dá ao TCC uma discussão muito mais rica do que "atingimos 99%":
o vazamento explica a *fraude métrica*; o gap de domínio explica o *limite real da
abordagem*. São dois problemas distintos, e ambos merecem uma secção.

**Critério original do TCC:** F1 macro ≥ 0,85.
**Estado:** por re-medir. O critério só é significativo sobre o split corrigido.

### 7.3 Limitações Identificadas

| Limitação | Descrição | Impacto |
|-----------|-----------|---------|
| **Gap de domínio** ⚠️ | Treino em fotos de estúdio (fundo neutro, folha isolada); uso real em fotos de telemóvel na horta | **A limitação mais séria.** A literatura reporta quedas para 30–60% nesta transição. Não se resolve com mais épocas — exige imagens de campo |
| **Dataset proxy** | PlantVillage não tem alface/rúcula; usa-se tomate/batata/milho como proxy visual | O que o modelo "sabe" sobre alface é transferido, não observado |
| **Rótulos heurísticos** | O label binário vem de palavras-chave no nome da pasta de origem | Ruído de rotulagem; mapa auditável em `data/label_map_audit.json` |
| **Redundância do dataset** | O total de 210.832 imagens conta cópias transformadas da mesma foto como imagens distintas | O nº de **fotos distintas** é significativamente menor — é esse que deve ser reportado (`data/split_metadata.json`) |
| **Classes binárias apenas** | Sistema não informa qual doença específica | Limita diagnóstico detalhado |
| **Dados brasileiros** | Sem imagens de hortaliças folhosas do contexto agrícola brasileiro | Variabilidade regional não representada |
| **ViT-B/16 não treinado** | Não incluído por limitações de tempo/VRAM | Comparação incompleta |
| **Latência em CPU** | Medições em GPU; em CPU pode ser 10–50× maior | Deployment sem GPU seria mais lento |

### 7.4 O que Fica para o TCC-II

**Prioridade máxima (decorre da errata):**
- [ ] **Refazer treino e avaliação sobre o split agrupado** e reportar acurácia balanceada
- [ ] **Montar um conjunto de teste de campo** — 100–300 fotos de telemóvel de hortas reais,
      rotuladas por inspecção. É o único número que responde à pergunta *"isto funciona?"*.
      Reportar as duas métricas lado a lado (test set vs. campo) torna o gap de domínio
      um **resultado do trabalho**, e não uma fragilidade escondida
- [ ] Reportar a **calibração** (ECE, diagrama de fiabilidade) e a taxa de abstenção

**Restante:**
- [ ] Implementar classificação multi-classe (tipo específico de doença)
- [ ] Treinar ViT-B/16 e incluir na comparação
- [ ] Implementar Grad-CAM para explicabilidade (código já existe em `gradcam.py`)
- [ ] Avaliar performance em CPU (dispositivos móveis sem GPU)
- [ ] Estudar deployment mobile (ONNX / TorchLite)

---

## 8. Estrutura de Arquivos

```
FitoVision/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/
│   │   │       ├── diagnosis.py       ← POST /api/diagnosis/ (multi-classe, legado)
│   │   │       ├── history.py         ← GET/DELETE /api/history/
│   │   │       ├── models_info.py     ← GET /api/models/
│   │   │       └── predict.py         ← POST /api/predict/ (binário, em produção)
│   │   ├── db/
│   │   │   ├── database.py            ← engine SQLite async + init_db()
│   │   │   └── models.py             ← SQLAlchemy ORM (tabela diagnósticos)
│   │   ├── ml/
│   │   │   ├── gradcam.py            ← Grad-CAM (implementado, não em produção)
│   │   │   ├── inference.py          ← load_binary_model() + predict_binary()
│   │   │   └── preprocessing.py      ← bytes → tensor normalizado 1×3×224×224
│   │   ├── schemas/
│   │   │   └── diagnosis.py          ← Pydantic schemas
│   │   ├── config.py                 ← Settings, CLASS_NAMES, AVAILABLE_MODELS
│   │   └── main.py                   ← FastAPI app + CORS + routers
│   │
│   ├── logs/
│   │   ├── mobilenet_v2_history.json      ← 30 épocas, best 98,86%
│   │   ├── efficientnet_b0_history.json   ← 30 épocas, best 98,98%
│   │   ├── resnet50_history.json          ← 30 épocas, best 98,71%
│   │   └── test_split_binary.json        ← caminhos do test set (31.625 amostras)
│   │
│   ├── results/
│   │   ├── cm_mobilenet_v2.png           ← Confusion matrix MobileNetV2
│   │   ├── cm_efficientnet_b0.png        ← Confusion matrix EfficientNet-B0
│   │   ├── cm_resnet50.png               ← Confusion matrix ResNet50
│   │   ├── metrics_comparison.csv        ← Tabela comparativa (5 métricas × 3 modelos)
│   │   ├── metrics_full.json             ← Métricas detalhadas por classe + confusion matrices
│   │   └── model_comparison.png          ← Gráfico de barras: accuracy, F1, latência
│   │
│   ├── weights/
│   │   ├── mobilenet_v2_binary.pth       ← Pesos MobileNetV2 (val_acc 98,86%)
│   │   ├── efficientnet_b0_binary.pth    ← Pesos EfficientNet-B0 (val_acc 98,98%) ★ produção
│   │   └── resnet50_binary.pth           ← Pesos ResNet50 (val_acc 98,71%)
│   │
│   ├── check_dataset.py      ← diagnóstico do dataset (contagem + alertas + grid visual)
│   ├── dataset.py            ← BinaryFolderDataset, PlantDataset, WeightedRandomSampler
│   ├── download_datasets.py  ← download Kaggle + organização binária (BFS recursiva)
│   ├── evaluate.py           ← avaliação completa: métricas + confusion matrix + plots
│   ├── prepare_data.py       ← DataLoaders com albumentations (augmentation avançado)
│   ├── run.py                ← uvicorn startup (port 8000)
│   ├── run_pipeline.py       ← pipeline sequencial automatizado (EfficientNet→ResNet50)
│   └── train.py              ← fine-tuning: AdamW + CosineAnnealingLR + AMP + early stopping
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── dashboard/
│   │   │   │   ├── MetricsCard.tsx        ← card de métrica individual
│   │   │   │   └── ModelComparison.tsx    ← gráfico comparativo
│   │   │   ├── diagnosis/
│   │   │   │   ├── ConfidenceBar.tsx      ← barra de confiança
│   │   │   │   ├── DiagnosisCard.tsx      ← card diagnóstico multi-classe (legado)
│   │   │   │   ├── GradCamViewer.tsx      ← visualizador Grad-CAM
│   │   │   │   └── ImageUploader.tsx      ← drag-and-drop upload
│   │   │   └── layout/
│   │   │       ├── ModelInfoModal.tsx     ← modal com métricas do modelo ativo
│   │   │       └── Navbar.tsx             ← barra de navegação
│   │   ├── pages/
│   │   │   └── HomePage.tsx              ← página principal: hero + upload + resultado binário
│   │   ├── services/
│   │   │   └── api.ts                    ← funções axios (runPredict, getHistory, etc.)
│   │   ├── types/
│   │   │   └── index.ts                  ← TypeScript types (PredictResult, DiagnosisResult, etc.)
│   │   └── index.css                     ← Inter font + Tailwind base
│   ├── index.html
│   ├── package.json                      ← deps: React 19, Vite 6, Tailwind 3, etc.
│   ├── tailwind.config.js                ← paleta primary: verde sage (#267350 = primary-600)
│   └── vite.config.ts
│
└── docs/
    ├── base color/                       ← referências de cor para o design
    └── COMPILADO_FITOVISION.md           ← este documento
```

---

## 9. Arquivos de Métricas, Logs e Pesos — Caminhos Completos

### Pesos (modelos treinados)
| Arquivo | Caminho | Tamanho aprox. | Uso |
|---------|---------|----------------|-----|
| `mobilenet_v2_binary.pth` | `backend/weights/mobilenet_v2_binary.pth` | ~14 MB | Comparação |
| `efficientnet_b0_binary.pth` | `backend/weights/efficientnet_b0_binary.pth` | ~21 MB | **Produção** |
| `resnet50_binary.pth` | `backend/weights/resnet50_binary.pth` | ~98 MB | Comparação |

### Logs de Treino
| Arquivo | Caminho | Conteúdo |
|---------|---------|---------|
| `mobilenet_v2_history.json` | `backend/logs/mobilenet_v2_history.json` | 30 épocas, best_val_acc=0,9886 |
| `efficientnet_b0_history.json` | `backend/logs/efficientnet_b0_history.json` | 30 épocas, best_val_acc=0,9898 |
| `resnet50_history.json` | `backend/logs/resnet50_history.json` | 30 épocas, best_val_acc=0,9871 |
| `test_split_binary.json` | `backend/logs/test_split_binary.json` | 31.625 pares (caminho, label) |

### Resultados de Avaliação
| Arquivo | Caminho | Conteúdo |
|---------|---------|---------|
| `metrics_comparison.csv` | `backend/results/metrics_comparison.csv` | Tabela 3 modelos × 5 métricas |
| `metrics_full.json` | `backend/results/metrics_full.json` | Métricas por classe + confusion matrices |
| `cm_mobilenet_v2.png` | `backend/results/cm_mobilenet_v2.png` | Confusion matrix MobileNetV2 |
| `cm_efficientnet_b0.png` | `backend/results/cm_efficientnet_b0.png` | Confusion matrix EfficientNet-B0 |
| `cm_resnet50.png` | `backend/results/cm_resnet50.png` | Confusion matrix ResNet50 |
| `model_comparison.png` | `backend/results/model_comparison.png` | Gráfico barras: accuracy + F1 + latência |

---

## 10. O que Não Foi Extraído Automaticamente

Os itens abaixo precisam ser **completados manualmente** na apresentação ou no TCC:

1. **Screenshots da interface web** — o compilado descreve o frontend mas não capturou prints. Recomendo tirar screenshots de:
   - Tela inicial (sem imagem carregada)
   - Upload de imagem em andamento (spinner)
   - Resultado "Saudável" com gauge verde
   - Resultado "Anômala" com gauge vermelho

2. **Imagens dos gráficos gerados** — os arquivos `.png` em `backend/results/` precisam ser inseridos na apresentação manualmente:
   - `model_comparison.png` → slide de comparação de modelos
   - `cm_efficientnet_b0.png` → slide da confusion matrix do modelo vencedor

3. **Curva de aprendizado visual** — os dados JSON estão completos mas não foi gerado um gráfico `loss_curves.png`. Se quiser, pode gerar com:
   ```python
   import json, matplotlib.pyplot as plt
   with open("backend/logs/efficientnet_b0_history.json") as f:
       h = json.load(f)["history"]
   epochs = [e["epoch"] for e in h]
   plt.plot(epochs, [e["val_acc"] for e in h], label="val_acc")
   plt.plot(epochs, [e["train_acc"] for e in h], label="train_acc")
   plt.legend(); plt.savefig("learning_curve.png")
   ```

4. **Tempo de treino do MobileNetV2 real** — 15,4h registradas incluem overhead noturno (laptop throttle). O tempo real de treino estável é estimado em ~6–7h mas não foi medido isoladamente.

5. **Referências bibliográficas completas** (para o capítulo de referências do TCC):
   - Sandler et al. (2018) — MobileNetV2 — CVPR 2018
   - Tan & Le (2019) — EfficientNet — ICML 2019
   - He et al. (2016) — Deep Residual Learning — CVPR 2016
   - Mohanty et al. (2016) — Plant Disease Detection — Frontiers in Plant Science
   - Loshchilov & Hutter (2019) — AdamW — ICLR 2019
   - Müller et al. (2019) — Label Smoothing — NeurIPS 2019
   - Dosovitskiy et al. (2020) — ViT — ICLR 2021
