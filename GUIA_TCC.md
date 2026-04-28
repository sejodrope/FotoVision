# FitoVision — Guia de Desenvolvimento do TCC

> **Estado actual (Abril 2026):** Toda a infraestrutura está implementada e funcional.
> O que falta é o trabalho de dados e treino — que é exactamente o conteúdo central do TCC.

---

## O que já está pronto

| Componente | Estado |
|---|---|
| API FastAPI (diagnóstico, histórico, modelos) | ✅ Completo |
| Pipeline de treino (`train.py`) | ✅ Completo |
| Pipeline de avaliação (`evaluate.py`) | ✅ Completo |
| Ferramenta de preparação de dados (`prepare_data.py`) | ✅ Completo |
| Dataset com augmentation e balanceamento (`dataset.py`) | ✅ Completo |
| 4 arquitecturas de rede (MobileNetV2, ResNet-50, EfficientNet-B0, ViT-B/16) | ✅ Completo |
| Grad-CAM para explicabilidade | ✅ Completo |
| Frontend React (diagnóstico, histórico, dashboard) | ✅ Completo |
| Base de dados SQLite para histórico | ✅ Completo |
| Mapeamentos PlantVillage e PlantDoc | ✅ Completo |

---

## Passo a Passo — O que falta

### FASE 1 — Recolha e Preparação do Dataset

#### 1.1 Fontes de dados disponíveis

O FitoVision já inclui mapeamentos para duas fontes públicas:

**PlantVillage** (`plantvillage_map.json`)
- Download: [Kaggle — PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Contém hortaliças (pimento, tomate, batata) com classes de doença bem documentadas
- Usar o `plantvillage_map.json` já incluído para mapear para as 6 classes do FitoVision

**PlantDoc** (`plantdoc_map.json`)
- Download: [GitHub — CVIT PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset)
- Dataset mais diverso, imagens de campo (mais realista)
- Usar o `plantdoc_map.json` já incluído

**Fotos próprias / agricultores** (recomendado para o TCC)
- Organizar em pastas com os nomes das classes:
  ```
  fotos_agricultores/
    saudavel/
    mildio/
    oidio/
    clorose_nitrogenio/
    danos_pragas/
    estresse_hidrico/
  ```
- Mínimo recomendado: 300 imagens por classe (500+ para resultados sólidos)

#### 1.2 Verificar distribuição antes de consolidar

```bash
cd backend
python prepare_data.py --dry-run \
    --source caminho/plantvillage:plantvillage_map.json \
    --source caminho/plantdoc:plantdoc_map.json \
    --source caminho/fotos_agricultores
```

Analise o output — classes com `<< INSUFICIENTE` precisam de mais imagens.

#### 1.3 Consolidar o dataset

```bash
python prepare_data.py \
    --source caminho/plantvillage:plantvillage_map.json \
    --source caminho/plantdoc:plantdoc_map.json \
    --source caminho/fotos_agricultores \
    --output ./data \
    --max-per-class 2000
```

O resultado fica em `backend/data/` com uma pasta por classe.

---

### FASE 2 — Treino dos Modelos

#### 2.1 Ambiente recomendado

- **Localmente com GPU NVIDIA:** funciona directamente, o script detecta CUDA automaticamente.
- **Google Colab (recomendado para TCC sem GPU):** GPU gratuita, 15–30 min por modelo.
- **CPU:** funciona, mas cada modelo leva horas. Apenas para testes rápidos.

#### 2.2 Preparar o ambiente

```bash
cd backend
pip install -r requirements.txt
```

#### 2.3 Treinar todos os modelos

```bash
# Treino completo — todos os 4 modelos
python train.py \
    --data ./data \
    --model all \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-3 \
    --patience 10

# No Google Colab (mais workers, batch maior):
python train.py \
    --data ./data \
    --model all \
    --epochs 50 \
    --batch-size 64 \
    --workers 2
```

Saídas geradas:
- `weights/mobilenet_v2.pth`
- `weights/resnet50.pth`
- `weights/efficientnet_b0.pth`
- `weights/vit_b_16.pth`
- `logs/<modelo>_history.json` — curvas de aprendizagem
- `logs/test_split.json` — índice do teste (necessário para o evaluate.py)

#### 2.4 Treinar modelos individualmente (se um falhar)

```bash
python train.py --data ./data --model mobilenet_v2 --epochs 30
python train.py --data ./data --model resnet50 --epochs 30
python train.py --data ./data --model efficientnet_b0 --epochs 30
python train.py --data ./data --model vit_b_16 --epochs 50
```

> **Nota ViT:** O ViT-B/16 é o modelo mais pesado. No Colab com GPU T4, ~45 min para 50 épocas.

---

### FASE 3 — Avaliação e Geração de Métricas

Após treinar, usar o mesmo test split para comparar todos os modelos de forma justa:

```bash
cd backend
python evaluate.py --test-split ./logs/test_split.json
```

Saídas em `backend/results/`:
- `metrics_comparison.csv` — tabela pronta para o TCC (accuracy, F1, precision, recall, latência)
- `metrics_full.json` — métricas detalhadas por classe (para análise aprofundada)
- `cm_<modelo>.png` — matriz de confusão por modelo
- `model_comparison.png` — gráfico comparativo (accuracy, F1, latência)
- `f1_per_class.png` — F1 por classe para cada modelo (útil para o capítulo de análise)

---

### FASE 4 — Activar os Modelos na API

#### 4.1 Copiar os pesos para o backend

Se treinou no Colab, faça download dos `.pth` e coloque em `backend/weights/`.

#### 4.2 Desactivar o modo demo

No ficheiro `backend/.env` (criar a partir do `.env.example`):

```env
DEMO_MODE=false
WEIGHTS_DIR=./weights
```

#### 4.3 Iniciar o backend

```bash
cd backend
python run.py
```

#### 4.4 Iniciar o frontend

```bash
cd frontend
npm install
npm run dev
```

Aceder em: http://localhost:5173

---

### FASE 5 — Testes e Validação

#### 5.1 Testar o Grad-CAM

Na interface web, após fazer um diagnóstico com Grad-CAM activado, verifique se o mapa de calor incide nas regiões correctas da folha (manchas, descoloração, etc.).

#### 5.2 Testar com imagens externas

Use imagens que **não** estavam no dataset de treino para testar a generalização.

#### 5.3 Verificar o dashboard

Aceder a http://localhost:5173/dashboard — deve mostrar "4/4" modelos calibrados quando os `.pth` estiverem carregados.

---

### FASE 6 — Escrita do TCC

#### Estrutura sugerida para o capítulo experimental

```
4. METODOLOGIA
   4.1 Dataset
       - Fontes utilizadas
       - Distribuição por classe (tabela do prepare_data.py)
       - Augmentation aplicada
   4.2 Arquitecturas avaliadas
       - MobileNetV2, ResNet-50, EfficientNet-B0, ViT-B/16
       - Transfer learning com ImageNet
   4.3 Protocolo de treino
       - Split estratificado 70/15/15
       - AdamW + Cosine Annealing + Label Smoothing
       - Early stopping (patience=10)

5. RESULTADOS
   5.1 Comparação de modelos (tabela de metrics_comparison.csv)
   5.2 Análise por classe (f1_per_class.png)
   5.3 Matrizes de confusão (cm_*.png)
   5.4 Análise de latência (model_comparison.png)
   5.5 Grad-CAM — Interpretabilidade visual

6. DISCUSSÃO
   - Qual modelo teve melhor accuracy vs. latência?
   - Quais classes foram mais difíceis de separar? (ver matrizes de confusão)
   - Limitações: tamanho do dataset, generalização para condições de campo
```

---

## Referências para o TCC

- **Transfer Learning:** Tan & Le (2019) — EfficientNet; He et al. (2016) — ResNet
- **Vision Transformer:** Dosovitskiy et al. (2021) — "An Image is Worth 16x16 Words"
- **Grad-CAM:** Selvaraju et al. (2017) — "Grad-CAM: Visual Explanations from Deep Networks"
- **PlantVillage:** Hughes & Salathé (2015) — "An Open Access Repository of Images on Plant Health"
- **PlantDoc:** Singh et al. (2020) — "PlantDoc: A Dataset for Visual Plant Disease Detection"
- **Label Smoothing:** Müller et al. (2019) — "When Does Label Smoothing Help?"

---

## Comandos de Referência Rápida

```bash
# 1. Ver distribuição do dataset antes de consolidar
python prepare_data.py --dry-run --source PASTA[:MAP.JSON] ...

# 2. Consolidar o dataset
python prepare_data.py --source PASTA[:MAP.JSON] ... --output ./data

# 3. Treinar
python train.py --data ./data --model all --epochs 30

# 4. Avaliar
python evaluate.py --test-split ./logs/test_split.json

# 5. Iniciar API
python run.py

# 6. Iniciar frontend (outra janela)
cd ../frontend && npm run dev
```

---

## Problemas Comuns

| Problema | Causa | Solução |
|---|---|---|
| `[ERRO] Classes sem imagens` | Pastas das classes não encontradas | Verificar estrutura em `data/` |
| Modelos a prever aleatoriamente | `DEMO_MODE=true` ainda activo | Definir `DEMO_MODE=false` no `.env` |
| GradCAM não aparece | Erro silenciado (ver logs do backend) | Verificar log `WARNING fitovision.diagnosis` |
| `CUDA out of memory` | Batch size demasiado grande | Reduzir `--batch-size` para 16 |
| Confiança sempre ~16% (1/6) | Modelo não convergiu ou pesos demo | Verificar `logs/<modelo>_history.json` |
| Frontend não conecta ao backend | CORS ou backend não iniciado | Confirmar que `python run.py` está em execução |
