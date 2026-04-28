# FitoVision — Guia Completo do TCC

> **Objetivo:** Treinar e comparar 4 redes neurais (MobileNetV2, ResNet-50, EfficientNet-B0, ViT-B/16)
> para detectar 6 condições fitossanitárias em hortaliças folhosas, com foco em cultivos da região de Joinville/SC.

---

## Estado actual do projecto

| Componente | Estado |
|---|---|
| API FastAPI (diagnóstico, histórico, modelos) | ✅ Pronto |
| 4 arquitecturas de rede neural (transfer learning) | ✅ Pronto |
| Grad-CAM para explicabilidade visual | ✅ Pronto |
| Pipeline de treino com early stopping e AMP | ✅ Pronto |
| Pipeline de avaliação com métricas e gráficos | ✅ Pronto |
| Ferramenta de consolidação de datasets | ✅ Pronto |
| Frontend React (diagnóstico, histórico, dashboard) | ✅ Pronto |
| **Dataset real com imagens locais** | ⏳ A fazer |
| **Modelos treinados (pesos .pth)** | ⏳ A fazer |
| **Capítulos experimentais do TCC** | ⏳ A fazer |

---

## As 6 Classes do Sistema

| ID interno | Nome para exibição | O que é |
|---|---|---|
| `saudavel` | Saudável | Folha sem anomalias visíveis |
| `mildio` | Míldio | Fungo; manchas amareladas/acinzentadas na face inferior |
| `oidio` | Oídio | Fungo; pó branco na superfície da folha |
| `clorose_nitrogenio` | Clorose por Nitrogênio | Amarelamento generalizado por deficiência de N |
| `danos_pragas` | Danos por Pragas | Furos, bordas roídas, minas foliares |
| `estresse_hidrico` | Estresse Hídrico | Murcha, enrolamento, bordas secas/queimadas |

---

## FASE 1 — Recolha de Imagens em Joinville

Esta é a fase mais importante para o TCC. Imagens locais garantem maior relevância para a pesquisa e originalidade científica que datasets genéricos não oferecem.

### 1.1 Onde fotografar em Joinville/SC

**Produtores e locais recomendados:**
- EPAGRI de Joinville — parceiros de pesquisa com acesso a cultivos e especialistas agronômicos
- Sítios e hortas na região de Pirabeiraba, Quiriri e Estrada Blumenau
- CEASA Joinville (SC-108) — mercado atacadista com grande variedade de hortaliças
- Hortas comunitárias urbanas (horta do Parque Joinville, bairro Boa Vista)
- Laboratório de Agronomia da UNIVILLE (se disponível)

**Hortaliças folhosas prioritárias para a região:**
Concentre-se nessas — são as mais cultivadas em Joinville e têm melhor representação das 6 classes:
1. **Alface** (Lactuca sativa) — fácil acesso, míldio e oídio comuns
2. **Couve** (Brassica oleracea) — danos por pragas (lagartas) muito frequentes
3. **Espinafre / Taioba** — clorose e estresse hídrico visíveis
4. **Rúcula** — danos por pulgões e mosca-minadora
5. **Repolho** — oídio e pragas características
6. **Acelga / Beterraba** — clorose frequente em solos pobres

### 1.2 Protocolo de fotografia (para máxima qualidade do dataset)

**Equipamento:**
- Smartphone com câmera ≥ 12 MP é suficiente (iPhone, Samsung Galaxy, etc.)
- Evitar câmera traseira com flash — gera reflexo nas folhas

**Configurações:**
- Fotografar sempre com o modo automático (sem filtros, sem HDR forçado)
- Resolução máxima disponível
- Salvar em JPEG ou PNG — evitar HEIC (converter antes)

**Técnica de captura:**

| O que fazer | Por quê |
|---|---|
| Fotografar a folha com fundo neutro (terra, bancada branca) | Evita que o modelo aprenda o fundo |
| Manter distância de 20–40 cm da folha | Boa resolução dos sintomas |
| Incluir a folha inteira no enquadramento | O modelo precisa ver o padrão global |
| Fotografar também recortes próximos dos sintomas | Aumenta diversidade do dataset |
| Fotografar em luz natural difusa (sombra ou dia nublado) | Evita sombras duras e reflexos |
| Múltiplos ângulos da mesma folha doente | Mais variação de perspectiva |
| Fotografar folhas em vários estágios da doença | Leve, moderado e severo |

**O que NÃO fazer:**
- Não usar zoom digital (perde resolução)
- Não fotografar com luz artificial fluorescente direto na folha
- Não usar filtros ou edição (Lightroom, Instagram, etc.)
- Não fotografar só sintomas avançados — inclua casos leves

**Meta mínima por classe:**

| Prioridade | Imagens mínimas | Imagens ideais |
|---|---|---|
| Saudável | 300 | 800+ |
| Míldio | 200 | 500+ |
| Oídio | 200 | 500+ |
| Clorose por Nitrogênio | 150 | 400+ |
| Danos por Pragas | 200 | 500+ |
| Estresse Hídrico | 150 | 400+ |

> **Dica:** Fotografe sempre mais do que o mínimo — imagens repetidas ou de má qualidade serão descartadas depois.

### 1.3 Organização das fotos no computador

Organize exactamente assim (os nomes das pastas devem ser idênticos):

```
backend/
  fotos_joinville/
    saudavel/          ← folhas sem problemas
    mildio/            ← manchas cinzentas/amarelas face inferior
    oidio/             ← pó branco superficial
    clorose_nitrogenio/← amarelamento generalizado
    danos_pragas/      ← furos, bordas roídas
    estresse_hidrico/  ← murcha, bordas secas
```

### 1.4 Complementar com datasets públicos (opcional mas recomendado)

Para aumentar volume, pode combinar as suas fotos com datasets públicos:

**PlantVillage** (mais fácil de obter)
- Download: https://www.kaggle.com/datasets/emmarex/plantdisease
- Já tem o `plantvillage_map.json` configurado no projecto

**PlantDoc** (mais realista, imagens de campo)
- Download: https://github.com/pratikkayal/PlantDoc-Dataset
- Já tem o `plantdoc_map.json` configurado no projecto

---

## FASE 2 — Preparação e Verificação do Dataset

### 2.1 Verificar o que tem antes de consolidar

```bash
cd backend

# Ver contagem por classe SEM copiar nada
python prepare_data.py --dry-run \
    --source fotos_joinville

# Se quiser combinar com PlantVillage também:
python prepare_data.py --dry-run \
    --source fotos_joinville \
    --source caminho/plantvillage:plantvillage_map.json
```

Analise o output — classes com `<< INSUFICIENTE` precisam de mais fotos antes de continuar.

### 2.2 Consolidar tudo na pasta `data/`

```bash
# Apenas fotos locais (recomendado para TCC com foco regional)
python prepare_data.py \
    --source fotos_joinville \
    --output ./data

# Com PlantVillage como complemento:
python prepare_data.py \
    --source fotos_joinville \
    --source caminho/plantvillage:plantvillage_map.json \
    --output ./data \
    --max-per-class 2000
```

### 2.3 Verificar a estrutura gerada

```
backend/data/
  saudavel/          → X imagens
  mildio/            → X imagens
  oidio/             → X imagens
  clorose_nitrogenio/→ X imagens
  danos_pragas/      → X imagens
  estresse_hidrico/  → X imagens
```

---

## FASE 3 — Treino dos Modelos

### 3.1 Onde treinar

| Opção | Velocidade | Custo | Recomendação |
|---|---|---|---|
| Google Colab (GPU T4) | ~30–45 min/modelo | Gratuito | **Recomendado** |
| PC com GPU NVIDIA | Mais rápido | Hardware próprio | Se tiver |
| CPU local | Muito lento (horas) | Nenhum | Apenas para testes |

### 3.2 Google Colab — passo a passo

1. Aceder a https://colab.research.google.com
2. Criar um novo notebook
3. Em **Ambiente de execução → Alterar tipo de ambiente de execução → GPU T4**
4. Copiar a pasta `backend/` para o Google Drive
5. Montar o Drive no Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

6. Instalar dependências:

```bash
pip install torch torchvision tqdm scikit-learn matplotlib seaborn pandas
```

7. Navegar para a pasta e treinar:

```bash
cd /content/drive/MyDrive/FitoVision/backend
python train.py --data ./data --model all --epochs 50 --batch-size 64 --workers 2
```

### 3.3 Localmente (com venv activado)

```bash
cd backend
.venv\Scripts\Activate.ps1  # Windows PowerShell

# Treinar todos os modelos
python train.py --data ./data --model all --epochs 30 --batch-size 32

# Ou um de cada vez (se quiser acompanhar individualmente)
python train.py --data ./data --model mobilenet_v2 --epochs 30
python train.py --data ./data --model resnet50      --epochs 30
python train.py --data ./data --model efficientnet_b0 --epochs 30
python train.py --data ./data --model vit_b_16     --epochs 50
```

### 3.4 O que o treino produz

```
backend/
  weights/
    mobilenet_v2.pth      ← pesos do melhor epoch por val_accuracy
    resnet50.pth
    efficientnet_b0.pth
    vit_b_16.pth
  logs/
    mobilenet_v2_history.json   ← curvas de loss e accuracy por epoch
    resnet50_history.json
    efficientnet_b0_history.json
    vit_b_16_history.json
    test_split.json             ← índice do conjunto de teste (usar no evaluate.py)
```

### 3.5 Verificar se o treino convergiu

Abrir o ficheiro `logs/<modelo>_history.json` e verificar:
- `best_val_acc` > 0.70 é um bom resultado para começar
- Se `best_val_acc` ≈ 0.16 (1/6), o modelo não aprendeu — mais dados ou ajuste de hiperparâmetros necessário

---

## FASE 4 — Avaliação e Geração de Métricas para o TCC

### 4.1 Avaliar todos os modelos com o mesmo test set

```bash
cd backend
python evaluate.py --test-split ./logs/test_split.json
```

### 4.2 Ficheiros gerados em `backend/results/`

| Ficheiro | O que contém | Usar no TCC |
|---|---|---|
| `metrics_comparison.csv` | Accuracy, F1, Precision, Recall, Latência | Tabela principal do capítulo de resultados |
| `metrics_full.json` | Métricas detalhadas por classe | Análise aprofundada por anomalia |
| `model_comparison.png` | Gráfico comparativo dos 4 modelos | Figura do capítulo de resultados |
| `f1_per_class.png` | F1 por classe para cada modelo | Figura de análise por condição |
| `cm_mobilenet_v2.png` | Matriz de confusão | Figura por modelo |
| `cm_resnet50.png` | Matriz de confusão | Figura por modelo |
| `cm_efficientnet_b0.png` | Matriz de confusão | Figura por modelo |
| `cm_vit_b_16.png` | Matriz de confusão | Figura por modelo |

### 4.3 O que analisar nos resultados

**Matrizes de confusão:** identificar quais pares de classes o modelo mais confunde.
Confusões esperadas: `mildio ↔ clorose_nitrogenio` (ambas amarelam a folha), `oidio ↔ saudavel` (sintomas iniciais subtis).

**F1 por classe:** qual condição o modelo detecta com mais dificuldade?
Classes com poucos exemplos no treino terão F1 mais baixo — documentar isso no TCC.

**Latência:** MobileNetV2 e EfficientNet-B0 devem ser significativamente mais rápidos que ViT-B/16.
Isso é relevante para argumentar qual modelo seria mais adequado para uso em campo (smartphone/edge).

---

## FASE 5 — Activar os Modelos no Sistema

### 5.1 Se treinou no Colab

Fazer download dos `.pth` do Google Drive e copiar para `backend/weights/`.

### 5.2 Mudar para modo produção

Editar `backend/.env`:

```env
DEMO_MODE=false
WEIGHTS_DIR=./weights
```

### 5.3 Reiniciar o backend

```bash
python run.py
```

O dashboard em http://localhost:5173/dashboard deve mostrar **"4/4 modelos calibrados"**.

---

## FASE 6 — Estrutura Sugerida para o TCC

### Capítulo de Metodologia

```
4.1 Caracterização do Dataset
    - Fontes utilizadas (fotos locais Joinville + dataset X se usado)
    - Protocolo de captura das imagens locais
    - Distribuição por classe (tabela do prepare_data.py --dry-run)
    - Técnicas de Data Augmentation aplicadas

4.2 Arquitecturas de Redes Neurais Avaliadas
    - MobileNetV2: depthwise separable convolutions, foco em eficiência
    - ResNet-50: skip connections, 50 camadas, arquitectura consolidada
    - EfficientNet-B0: compound scaling, melhor acurácia/parâmetro
    - ViT-B/16: Vision Transformer, patches 16×16, atenção global

4.3 Protocolo de Transfer Learning
    - Pesos iniciais: ImageNet (ILSVRC 2012)
    - Estratégia: fine-tuning completo da rede
    - Substituição da camada final por 6 neurónios (softmax)

4.4 Protocolo de Treino
    - Divisão estratificada: 70% treino / 15% validação / 15% teste
    - Optimizador: AdamW (lr=1e-3, weight_decay=1e-4)
    - Scheduler: Cosine Annealing (T_max=epochs, eta_min=lr×0.01)
    - Label smoothing: 0.1 (melhora calibração das probabilidades)
    - Early stopping: patience=10 épocas
    - Balanceamento de classes: WeightedRandomSampler
```

### Capítulo de Resultados

```
5.1 Comparação Geral dos Modelos
    → Inserir tabela de metrics_comparison.csv
    → Inserir figura model_comparison.png

5.2 Análise por Condição Fitossanitária
    → Inserir figura f1_per_class.png
    → Discutir classes com F1 mais baixo e justificativa

5.3 Matrizes de Confusão
    → Inserir cm_*.png (uma por modelo)
    → Identificar padrões de erro (quais classes são confundidas)

5.4 Análise de Explicabilidade — Grad-CAM
    → Screenshots do sistema com mapas de calor
    → Verificar se o modelo foca nas regiões correctas (manchas, descoloração)
    → Comparar o Grad-CAM entre modelos para o mesmo caso

5.5 Análise de Latência e Viabilidade Prática
    → Tabela de latência (ms/imagem) por modelo e hardware
    → Discussão sobre uso em campo (smartphone, edge device)
```

### Capítulo de Discussão

```
6.1 Qual modelo apresentou melhor resultado geral e por quê?
6.2 Existe trade-off entre acurácia e latência? Qual modelo é mais adequado para uso real?
6.3 Quais condições foram mais difíceis de classificar?
6.4 Como as imagens locais de Joinville influenciaram os resultados vs. datasets genéricos?
6.5 Limitações: tamanho do dataset, variabilidade de condições de luz, generalização
6.6 Trabalhos futuros: detecção em tempo real, app mobile, integração com sensores
```

---

## Referências Bibliográficas (para o TCC)

```
DOSOVITSKIY, A. et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR, 2021.

HE, K. et al. Deep Residual Learning for Image Recognition. CVPR, 2016.

HOWARD, A. G. et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv, 2017.

HUGHES, D.; SALATHÉ, M. An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv, 2015. [PlantVillage]

MÜLLER, R. et al. When Does Label Smoothing Help? NeurIPS, 2019.

SELVARAJU, R. R. et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV, 2017.

SINGH, D. et al. PlantDoc: A Dataset for Visual Plant Disease Detection. CoDS-COMAD, 2020.

TAN, M.; LE, Q. V. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML, 2019.
```

---

## Referência Rápida de Comandos

```bash
# Ver distribuição antes de consolidar
python prepare_data.py --dry-run --source fotos_joinville

# Consolidar dataset
python prepare_data.py --source fotos_joinville --output ./data

# Treinar todos os modelos
python train.py --data ./data --model all --epochs 30

# Avaliar (usar o mesmo test split do treino)
python evaluate.py --test-split ./logs/test_split.json

# Iniciar backend
python run.py

# Iniciar frontend (terminal separado)
cd ../frontend && npm run dev
```

---

## Problemas Comuns e Soluções

| Problema | Causa | Solução |
|---|---|---|
| `[ERRO] Classes sem imagens` | Pastas das classes ausentes ou vazias | Verificar estrutura dentro de `data/` |
| Modelos a prever ~16% de confiança | Não convergiu ou `DEMO_MODE=true` | Verificar `logs/<modelo>_history.json`; definir `DEMO_MODE=false` |
| GradCAM não aparece | Falha silenciada (ver log do backend) | Procurar `WARNING fitovision.diagnosis` no terminal do backend |
| `CUDA out of memory` | Batch size demasiado grande | Reduzir `--batch-size` para 16 ou 8 |
| Frontend não conecta | Backend não iniciado | Confirmar `python run.py` a correr no outro terminal |
| Imagem rejeitada (415) | Formato não suportado | Converter para JPEG ou PNG |
| Imagem rejeitada (413) | Ficheiro > 10 MB | Reduzir resolução ou definir `MAX_UPLOAD_BYTES` maior no `.env` |
