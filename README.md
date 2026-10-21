# FitoVision

Sistema de diagnóstico fitossanitário automatizado para hortaliças folhosas, desenvolvido como TCC no curso de Engenharia de Software da UNIVILLE (2026).

Recebe uma foto de folha e classifica em **saudável** ou **anômala** via Transfer Learning com PyTorch — ou **abstém-se**, quando não tem confiança suficiente ou a imagem nem sequer é uma folha.

---

## Resultados

Estado em 13/10/2026 (ciclo 4, depois da auditoria de consistência; o ciclo 5, de 14/10,
vem logo abaixo). Classificação
binária saudável × anômalo, com calibração por temperature scaling e abstenção abaixo de
τ = 0,85. Os números e de onde cada um sai estão em
[`docs/FICHA_TECNICA_FITOVISION.md`](docs/FICHA_TECNICA_FITOVISION.md).

| Onde foi medido | Resultado (EfficientNet-B0) | O que significa |
|---|---|---|
| Teste das mesmas seis fontes do treino (N = 23.967) | 98,48% de acurácia balanceada (IC 95%: 98,32 a 98,64) | Média entre fotos fáceis e difíceis das fontes de treino |
| ↳ culturas fotografadas em estúdio, na coletânea (N = 10.061) | 99,7% | Folha isolada, fundo liso |
| ↳ culturas fotografadas em lavoura, na mesma coletânea (N = 4.530) | **86,3%** (IC 95%: 84,5 a 88,0) | Onde estão 81% dos erros do teste |
| Fonte nunca vista no treino — espinafre, ciclo 3 (N = 484) | 75,4% | Queda de 23 pontos ao mudar de fonte |
| Mesma fonte, depois de entrar no treino — ciclo 4 | 88,4% | Ganho real, mas de novo intra-fonte |
| Horta real (13 fotos de alface) | 11 acertos em 13 (IC 95%: 57,8% a 95,7%) | Amostra pequena demais para concluir |

Entre as quatro arquiteturas comparadas, EfficientNet-B0 e ViT-B/16 empatam no teste
(98,48%; McNemar exato com correção de Holm, p = 0,26) e superam MobileNetV2 e ResNet-50
(p < 10⁻¹¹). O ViT-B/16, treinado no Google Colab em 13/10/2026, tem 21 vezes mais
parâmetros (85,8 milhões contra 4,0 milhões nos modelos treinados) e é 2,7 vezes mais
lento, de modo que o EfficientNet-B0 segue como modelo servido. Nas culturas fotografadas
em lavoura, as quatro arquiteturas ficam entre 83,4% e 87,8%. O limiar de abstenção segura
31% dos erros do EfficientNet-B0 no teste (15% no ViT-B/16); em campo, os nove erros
registrados (de quatro fotografias) saíram com confiança acima de 0,94.

**O que estes números não dizem:** quanto o sistema acerta em uso real com hortaliças
folhosas. Isso depende da coleta de fotos de campo.

**Uma ressalva sobre o teste.** O particionamento agrupa quase-duplicatas (vazamento
residual de 0,04% por esse critério), mas não reconhece variantes muito alteradas da
mesma foto. Uma das fontes de alface é feita de rotações, recortes e filtros das mesmas
1.231 fotografias: 23% do teste são variantes com outra variante no treino. A correção
(split por proveniência) entrou no ciclo 5.

### Ciclo 5 (14/10/2026) — pipeline corrigido

Alface separada por fotografia original (uma foto por original no teste), 8.266 imagens
de pastas de praga rotuladas como anômalas, amostragem que equilibra os rótulos dentro de
cada cultura e um lote de espinafre fora do treino. 152.057 imagens; 1,7% do teste com
quase-duplicata no treino. As quatro arquiteturas foram treinadas no Google Colab (GPU L4).
Acurácia balanceada (%):

| Modelo | Teste do ciclo 5 | Lavoura: ciclo 4 → ciclo 5 | Especificidade na lavoura | Alface, 1 foto por original (176) | Lote retido (574) |
|---|---|---|---|---|---|
| EfficientNet-B0 | 97,4 | 86,3 → 88,8 | 76,6 → 83,4 | 96,7 | 87,9 |
| MobileNetV2 | 96,9 | 86,6 → 90,1 | 78,3 → 90,2 | 100 | 88,4 |
| ResNet-50 | 96,9 | 87,8 → 89,6 | 81,4 → 89,0 | 100 | 90,1 |
| ViT-B/16 | 96,9 | 83,4 → 84,5 | 69,8 → 72,1 | 100 | 85,8 |

O ciclo 5 reduz os falsos alarmes nas culturas fotografadas em lavoura, em troca de 2 a 5
pontos de sensibilidade; contado imagem a imagem, o ciclo 4 ainda acerta mais. O modelo
servido continua o EfficientNet-B0 do ciclo 4 até a validação em campo. Detalhe em
[`docs/CORRECOES_METODOLOGICAS.md`](docs/CORRECOES_METODOLOGICAS.md) §15 e
`backend/results/ciclo5/`.

> **Retratação (julho de 2026).** A primeira versão deste repositório reportava 99,01%
> de acurácia. O número era inválido: 57,35% do conjunto de teste eram cópias
> transformadas de fotos do treino. O particionamento passou a ser feito por grupos de
> identidade visual (`imagehash_utils.py`), com vazamento residual de 0,04%. A
> investigação completa está em
> [`docs/CORRECOES_METODOLOGICAS.md`](docs/CORRECOES_METODOLOGICAS.md).

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
#    (lote de 64, como nos ciclos 1 a 4; o padrão do train.py é 32)
python train.py --data ./data --binary --model efficientnet_b0 \
                --epochs 30 --batch-size 64 --workers 2

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
│   ├── auditoria_consistencia.py   ← confere dataset, proveniência e erros por subconjunto (13/10)
│   ├── prepara_ciclo5.py           ← split corrigido: proveniência, pragas, fonte retida → data_c5/
│   ├── run_ciclo5.py               ← ciclo 5: treina, calibra e avalia sem tocar no ciclo 4
│   ├── avalia_ciclo5.py            ← avaliação por subconjunto e na fonte retida
│   ├── compara_c4_c5.py            ← ciclo 4 × ciclo 5 por subconjunto, nas mesmas imagens
│   ├── confere_fotos_campo.py      ← confere as fotos de campo antes de rodar o modelo
│   ├── roda_campo.py               ← validação em campo num comando (confere → infere → analisa)
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
| Dados | Seis fontes do Kaggle + espinafre (Mendeley) + míldio (Roboflow): 175.123 imagens, 101.987 fotos distintas pelo hash |

---

## Limitações conhecidas

Documentadas por honestidade metodológica — e porque um TCC que as discute vale mais
que um que as esconde:

1. **Domínio de treino ≠ domínio de uso.** Cerca de metade do treino (no máximo 47,7%)
   deriva do PlantVillage: folha única, fundo uniforme de estúdio, iluminação
   controlada. Outros 17,6% são fotos de lavoura, mas de mandioca, arroz, cana e café.
   Uma foto de horta de hortaliça folhosa (terra, sombra, planta inteira, mão no
   enquadramento) está fora dessa distribuição. A queda aparece dentro do próprio teste
   (86,3% em foto de lavoura contra 99,7% em estúdio) e não se resolve só com mais épocas.

2. **Poucas fotos da cultura-alvo, e quase todas saudáveis.** A alface é 30% do treino,
   mas vem de 1.231 fotografias originais (108 doentes), repetidas em cópias e
   transformações; 87% das imagens de alface são saudáveis. Rúcula, acelga e couve não
   têm dataset público; o espinafre é *Basella alba*. O resto é aprendido de outras
   culturas, como *proxy visual*.

3. **Rótulos inferidos por heurística.** O label binário vem de palavras-chave no nome
   da pasta de origem. O mapeamento é registado em `data/label_map_audit.json`
   justamente para poder ser conferido à mão. A conferência de 13/10 mostrou que pastas
   de dano por praga não foram reconhecidas e ficaram fora; no ciclo 5 elas entram como
   anômalas, por lista explícita.

4. **Binário, não diagnóstico.** O sistema diz *"há algo de errado"*, não *qual* doença.
   As 6 classes de `config.py` pertencem ao pipeline multi-classe, ainda sem dados
   suficientes.

---

## Documentação técnica

| Para saber... | Ler |
|---|---|
| Onde o projeto está e o que fazer a seguir | [`PROXIMOS_PASSOS_TCC2.md`](PROXIMOS_PASSOS_TCC2.md) |
| Um número ou uma afirmação de método, com a prova | [`docs/FICHA_TECNICA_FITOVISION.md`](docs/FICHA_TECNICA_FITOVISION.md) (gerada por `backend/auditoria_consistencia.py`) |
| Preparação para a banca e roteiro da apresentação | [`docs/GUIA_BANCA_TCC2.md`](docs/GUIA_BANCA_TCC2.md) |
| Trabalhos relacionados (Web of Science) | [`docs/REVISAO_LITERATURA_WOS.md`](docs/REVISAO_LITERATURA_WOS.md) |
| A história inteira, em ordem | [`docs/HISTORICO_COMPLETO_TCC.md`](docs/HISTORICO_COMPLETO_TCC.md) |
| Cada erro encontrado, como foi achado e como foi corrigido | [`docs/CORRECOES_METODOLOGICAS.md`](docs/CORRECOES_METODOLOGICAS.md) |
| Como treinar o ViT-B/16 no Google Colab | [`docs/GUIA_TREINO_VIT_COLAB.md`](docs/GUIA_TREINO_VIT_COLAB.md); automação pela CLI em [`colab/cli/`](colab/cli/) (ViT) e [`colab/cli_c5/`](colab/cli_c5/) (ciclo 5, quatro arquiteturas) |
| Ciclo 5 por subconjunto e contra o ciclo 4 | [`backend/results/ciclo5/avaliacao_ciclo5.md`](backend/results/ciclo5/avaliacao_ciclo5.md), [`backend/results/ciclo5/ciclo4_x_ciclo5.md`](backend/results/ciclo5/ciclo4_x_ciclo5.md) |
| Comparação das quatro arquiteturas | [`backend/results/arquiteturas/comparacao_arquiteturas.md`](backend/results/arquiteturas/comparacao_arquiteturas.md) |
| Como coletar e analisar fotos de campo | [`docs/GUIA_CAMPO_HORTAS_COMUNITARIAS.md`](docs/GUIA_CAMPO_HORTAS_COMUNITARIAS.md), [`docs/GUIA_TESTE_DE_CAMPO.md`](docs/GUIA_TESTE_DE_CAMPO.md) |
| Artigo do TCC-II e monografia | [`docs/Entregas Finais/`](docs/Entregas%20Finais/) — versão atual do artigo: `TCC2_Artigo_FitoVision_v8.docx` (ciclo 5 e literatura lida na íntegra); monografia: `TCC_FitoVision_ATUALIZADO_v5.docx` |

`docs/COMPILADO_FITOVISION.md` e `docs/GUIA_TCC.md` são do TCC-I e ficam como registro;
os números deles foram superados pelos documentos acima.

---

## TCC · UNIVILLE 2026

Autor: **José Pedro Vieira Silva**  
Curso: Engenharia de Software — UNIVILLE, Joinville/SC  
Orientação: Prof. Leanderson André · TCC-II
