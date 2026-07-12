# FitoVision

Sistema de diagnóstico fitossanitário automatizado para hortaliças folhosas, desenvolvido como TCC no curso de Engenharia de Software da UNIVILLE (2026).

Recebe uma foto de folha e classifica em **saudável** ou **anômala** via Transfer Learning com PyTorch — ou **abstém-se**, quando não tem confiança suficiente ou a imagem nem sequer é uma folha.

---

## Resultados

> ### ⚠️ Resultados retirados — em reavaliação
>
> As métricas anteriormente publicadas aqui (**99,01% de acurácia**) **eram inválidas**
> e foram removidas. Não mediam generalização.
>
> **A causa.** O conjunto de teste era construído com um *shuffle ao nível do ficheiro*
> sobre um pool que contém múltiplas cópias transformadas da mesma foto — o dataset
> `lettuce-disease-multi-transformation-dataset` é, literalmente, versões rodadas,
> espelhadas e com brilho alterado das mesmas folhas, e o
> `plant-diseases-training-dataset` é um re-upload do PlantVillage. O sorteio colocava
> uma rotação de uma foto no treino e o espelhamento da **mesma foto** no teste.
>
> O modelo não estava a diagnosticar folhas: estava a reconhecer fotos que já tinha
> visto. Os 99% mediam **memorização**, não capacidade de diagnóstico — e é por isso
> que não se reproduziam em fotos novas.
>
> **A correcção.** O split passou a ser feito ao nível de *grupos de identidade
> visual*: todas as variantes de uma mesma foto caem obrigatoriamente no mesmo split
> (`imagehash_utils.py`). Acrescentou-se ainda calibração de confiança, balanceamento
> de classes e abstenção. Ver [`docs/COMPILADO_FITOVISION.md`](docs/COMPILADO_FITOVISION.md).
>
> **Para reproduzir os números honestos:**
> ```bash
> cd backend
> python download_datasets.py --skip-download   # refaz o split (group-aware)
> python audit_leakage.py                       # quantifica o vazamento antigo
> python run_pipeline.py                        # treina + calibra + avalia
> ```
> Reportar a **acurácia balanceada** de `results/metrics_comparison.csv`.
> Espere um valor **substancialmente inferior a 99% — e verdadeiro.**

---

## Demo rápido

```
backend/  →  FastAPI + PyTorch   →  localhost:8000
frontend/ →  React + TypeScript  →  localhost:5173
```

1. Abrir `http://localhost:5173`
2. Arrastar ou clicar para enviar foto de folha (JPG / PNG / WebP)
3. Ver o resultado — um de quatro:

| Resultado | Quando |
|-----------|--------|
| **Saudável** / **Anômala** | confiança calibrada acima do limiar |
| **Inconclusivo** | o modelo não tem confiança suficiente — pede outra foto |
| **Não é uma folha** | a imagem não contém vegetação (índice ExG) |

> Os dois últimos estados são deliberados. A versão anterior devolvia sempre um
> veredicto — mesmo para fotos que não eram folhas, e sempre com confiança alta.

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
#    O split é AGRUPADO por identidade visual: todas as variantes (rotação, flip,
#    brilho) de uma mesma foto vão obrigatoriamente para o mesmo split.
python download_datasets.py

# 3. Auditar vazamento — confirma que nenhuma foto de teste tem duplicado no treino
python audit_leakage.py

# 4. Verificar dataset
python check_dataset.py

# 5. Treinar (loss com pesos de classe + selecção por F1 macro)
python train.py --data ./data --binary --model efficientnet_b0 \
                --epochs 30 --batch-size 32 --workers 2

# 6. Calibrar a confiança (temperature scaling + limiar de abstenção)
python calibrate.py --data ./data --binary --model efficientnet_b0

# 7. Avaliar (accuracy balanceada, F1, AUC, ECE)
python evaluate.py --test-split ./logs/test_split_binary.json --binary

# Ou rodar tudo automatizado (audita → treina → calibra → avalia)
python run_pipeline.py
```

Resultados salvos em `backend/results/`:

| Ficheiro | Conteúdo |
|----------|----------|
| `leakage_report.json` | quantifica o vazamento — a prova de que os 99% eram inválidos |
| `metrics_comparison.csv` | accuracy, **acurácia balanceada**, F1, AUC, ECE, latência |
| `calibration_*.png` | diagrama de fiabilidade (confiança reportada vs. acerto real) |
| `cm_*.png` | matrizes de confusão |

> **Reportar no TCC a acurácia balanceada**, não a accuracy simples: sob
> desbalanceamento de classes, prever sempre a classe maioritária já dá uma accuracy
> alta sem qualquer mérito.

---

## Estrutura

```
FitoVision/
├── backend/
│   ├── app/
│   │   ├── api/routes/predict.py   ← POST /api/predict/
│   │   ├── ml/inference.py         ← predict_binary() + calibração + abstenção
│   │   ├── ml/preprocessing.py     ← bytes → tensor normalizado
│   │   └── config.py
│   ├── logs/                       ← histórico de treino (JSON por modelo)
│   ├── results/                    ← métricas + confusion matrices + gráficos
│   ├── weights/                    ← pesos .pth (não versionados — ver .gitignore)
│   ├── imagehash_utils.py          ← hash perceptual D4 + agrupamento de duplicados
│   ├── audit_leakage.py            ← quantifica vazamento entre splits
│   ├── calibrate.py                ← temperature scaling + limiar de abstenção
│   ├── train.py                    ← fine-tuning; loss ponderada; selecção por F1
│   ├── evaluate.py                 ← accuracy balanceada, F1, AUC, ECE, plots
│   ├── download_datasets.py        ← download + split AGRUPADO (group-aware)
│   ├── run_pipeline.py             ← orquestrador: audita → treina → calibra → avalia
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
| Dados | Kaggle (PlantVillage + lettuce datasets) · nº de fotos distintas a apurar |

---

## Limitações conhecidas

Documentadas por honestidade metodológica — e porque um TCC que as discute vale mais
que um que as esconde:

1. **Domínio de treino ≠ domínio de uso.** O grosso do dataset é PlantVillage: folha
   única, fundo uniforme de estúdio, iluminação controlada. Uma foto de telemóvel
   feita na horta (terra, sombra, várias folhas, mão no enquadramento) está fora dessa
   distribuição. A literatura reporta quedas grandes de desempenho nessa transição —
   é o principal factor a limitar o uso real, e não se resolve só com mais épocas.

2. **Proxy de espécie.** PlantVillage não tem alface, rúcula ou espinafre. Treina-se
   em tomate/batata/milho como *proxy visual* de folha doente. O que o modelo aprende
   sobre alface é, portanto, transferido — não observado.

3. **Rótulos inferidos por heurística.** O label binário vem de palavras-chave no nome
   da pasta de origem. O mapeamento é registado em `data/label_map_audit.json`
   justamente para poder ser conferido à mão.

4. **Binário, não diagnóstico.** O sistema diz *"há algo de errado"*, não *qual* doença.
   As 6 classes de `config.py` pertencem ao pipeline multi-classe, ainda sem dados
   suficientes.

---

## Documentação técnica

📌 **[`docs/CORRECOES_METODOLOGICAS.md`](docs/CORRECOES_METODOLOGICAS.md)** — **começar por aqui.**
Registo completo da investigação: o sintoma, as hipóteses, a causa (vazamento de dados),
as 9 correcções, a **verificação experimental** de cada uma, e o que fica para o TCC-II.
É a base do capítulo de metodologia e da defesa.

[`docs/COMPILADO_FITOVISION.md`](docs/COMPILADO_FITOVISION.md) — compilado técnico:
- Secção 0: **Errata** — retratação formal dos resultados de 06/06/2026
- Metodologia (dataset, augmentation, hiperparâmetros)
- Arquitecturas, curvas de aprendizado, matrizes de confusão
- Comandos para reproduzir o experimento

---

## TCC · UNIVILLE 2026

Autor: **José Pedro Vieira Silva**  
Curso: Engenharia de Software — UNIVILLE, Joinville/SC  
Orientação: TCC-I (Seminário III)
