# FitoVision

Sistema de diagnóstico fitossanitário automatizado para hortaliças folhosas, desenvolvido como TCC no curso de Engenharia de Software da UNIVILLE (2026).

Recebe uma foto de folha e classifica em **saudável** ou **anômala** em menos de 25 ms, via Transfer Learning com PyTorch.

---

## Resultados

Três arquiteturas treinadas e avaliadas em **31.625 imagens** nunca vistas durante o treino:

| Modelo | Accuracy | F1 macro | Latência | Parâmetros |
|--------|---------|----------|---------|-----------|
| **EfficientNet-B0** ★ | **99,01%** | **98,89%** | 24 ms | 5,3 M |
| MobileNetV2 | 98,81% | 98,67% | 14 ms | 3,4 M |
| ResNet50 | 98,71% | 98,56% | 17 ms | 25,6 M |

★ modelo em produção

Dataset: **210.832 imagens** binárias (healthy / anomalous) agregadas de múltiplas fontes Kaggle.

---

## Demo rápido

```
backend/  →  FastAPI + PyTorch   →  localhost:8000
frontend/ →  React + TypeScript  →  localhost:5173
```

1. Abrir `http://localhost:5173`
2. Arrastar ou clicar para enviar foto de folha (JPG / PNG / WebP)
3. Ver classificação + gauge de confiança + barras de probabilidade

---

## Instalação

### Pré-requisitos
- Python 3.12
- Node.js 20+
- NVIDIA GPU com CUDA 12.x (opcional; funciona em CPU com latência maior)

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install torch==2.5.1+cu121 torchvision \
    --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn[standard] sqlalchemy aiosqlite \
    pydantic-settings python-dotenv pillow pandas \
    scikit-learn matplotlib seaborn tqdm albumentations

python run.py                   # inicia em http://localhost:8000
```

> Sem GPU: instale `torch` sem sufixo `+cu121` — o sistema usa CPU automaticamente.

### Frontend

```bash
cd frontend
npm install
npm run dev                     # inicia em http://localhost:5173
```

---

## Reproduzir o Treino

```bash
cd backend

# 1. Configurar credenciais Kaggle em backend/.env
#    KAGGLE_USERNAME=...
#    KAGGLE_KEY=...

# 2. Baixar e organizar dataset (~5 GB, ~2-3h)
python download_datasets.py

# 3. Verificar dataset
python check_dataset.py

# 4. Treinar EfficientNet-B0
python train.py --data ./data --binary --model efficientnet_b0 \
                --epochs 30 --batch-size 32 --workers 2

# 5. Avaliar
python evaluate.py --test-split ./logs/test_split_binary.json --binary

# Ou rodar pipeline automatizado (EfficientNet → ResNet50)
python run_pipeline.py
```

Resultados salvos em `backend/results/` (CSV, JSON, confusion matrices, gráfico comparativo).

---

## Estrutura

```
FitoVision/
├── backend/
│   ├── app/
│   │   ├── api/routes/predict.py   ← POST /api/predict/
│   │   ├── ml/inference.py         ← load_binary_model() + predict_binary()
│   │   ├── ml/preprocessing.py     ← bytes → tensor normalizado
│   │   └── config.py
│   ├── logs/                       ← histórico de treino (JSON por modelo)
│   ├── results/                    ← métricas + confusion matrices + gráficos
│   ├── weights/                    ← pesos .pth (não versionados — ver .gitignore)
│   ├── train.py                    ← fine-tuning com AdamW + CosineAnnealingLR + AMP
│   ├── evaluate.py                 ← avaliação completa com métricas e plots
│   ├── download_datasets.py        ← pipeline de download Kaggle
│   ├── run_pipeline.py             ← orquestrador sequencial automatizado
│   └── run.py                      ← uvicorn startup
├── frontend/
│   ├── src/
│   │   ├── pages/HomePage.tsx      ← página principal
│   │   ├── components/             ← ImageUploader, gauge SVG, cards
│   │   ├── services/api.ts         ← chamadas HTTP
│   │   └── types/index.ts          ← TypeScript types
│   └── tailwind.config.js          ← paleta verde sage (primary)
└── docs/
    ├── COMPILADO_FITOVISION.md     ← compilado técnico completo (metodologia + resultados)
    └── GUIA_TCC.md                 ← guia de desenvolvimento do TCC
```

---

## Stack

| Camada | Tecnologia |
|--------|-----------|
| ML | PyTorch 2.5.1 + CUDA 12.1 |
| API | FastAPI + Uvicorn |
| DB | SQLite + SQLAlchemy async |
| Frontend | React 19 + TypeScript + Vite 6 |
| Estilo | Tailwind CSS 3 + Inter |
| Dados | 210.832 imagens · Kaggle (PlantVillage + lettuce datasets) |

---

## Documentação técnica

Ver [`docs/COMPILADO_FITOVISION.md`](docs/COMPILADO_FITOVISION.md) para:
- Metodologia completa (dataset, augmentation, hiperparâmetros)
- Curvas de aprendizado dos 3 modelos
- Análise de resultados vs literatura
- Matrizes de confusão detalhadas por classe
- Comandos para reproduzir o experimento

---

## TCC · UNIVILLE 2026

Autor: **José Pedro Vieira Silva**  
Curso: Engenharia de Software — UNIVILLE, Joinville/SC  
Orientação: TCC-I (Seminário III)
