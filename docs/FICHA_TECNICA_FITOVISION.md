# FitoVision — Ficha técnica (fatos verificados)

### Auditoria de consistência de 13/10/2026 · TCC-II · Univille

> **Para que serve.** É a fonte única dos números e das afirmações de método do projeto.
> Cada linha foi conferida em um artefato (split em disco, logs, código de treino, pesos,
> previsões por imagem), e não em outro documento. Quando um documento do projeto
> discordar desta ficha, vale a ficha; a Seção 11 lista o que estava errado em cada um.
>
> **Como reproduzir.** `cd backend` e
> `.venv\Scripts\python.exe auditoria_consistencia.py --parametros --hash`. O script não
> treina nem roda inferência; grava `backend/results/auditoria/auditoria_13out.json` e
> `.md`.

---

## 1. Identificação

| Item | Valor |
|---|---|
| Título | Detecção automática de anomalias fitossanitárias em hortaliças folhosas utilizando visão computacional e aprendizado de máquina |
| Autor | José Pedro Vieira Silva — josepedro@univille.br |
| Curso | **Bacharelado em Engenharia de Software**, Univille, Joinville/SC |
| Orientador | Prof. Leanderson André — leandersonandre@univille.br |
| Banca | 08/12/2026, fala de no máximo 20 minutos |
| Artigo atual | `docs/Entregas Finais/TCC2_Artigo_FitoVision_v8.docx` (16/10: v7 + correções da leitura completa da literatura; 11 páginas no LibreOffice) |
| Monografia atual | `docs/Entregas Finais/TCC_FitoVision_ATUALIZADO_v5.docx` (16/10: v4 + nota de 16/10, seção 9.6 sobre a literatura lida na íntegra, sumário completado) |

## 2. Sistema

- **Tarefa:** triagem binária, `healthy` × `anomalous` (doença, praga ou deficiência).
- **Quatro saídas:** saudável, anômalo, inconclusivo (confiança calibrada < τ) e não é
  folha (menos de 10% dos pixels com ExG = 2g − r − b acima de 0,05).
- **Modelo servido:** EfficientNet-B0 do ciclo 4, τ = 0,85 (`backend/app/config.py`,
  `backend/weights/efficientnet_b0_binary_calibration.json`).
- **Stack:** PyTorch 2.5.1 + CUDA 12.1; FastAPI + Uvicorn; SQLite; React 19 +
  TypeScript + Vite 6. GPU local RTX 3050 de 6 GB.

## 3. Dataset: quantas imagens, em cada etapa

Os três números que apareciam nos documentos são de etapas diferentes:
**176.126** é o que foi reunido (staging), **170.772** é o que entrou nas partições da
base e **175.123** é a base mais os dois lotes do ciclo 4.

| Etapa | Reunidas | Nas partições | Treino / Validação / Teste | Fotos distintas¹ | Vazamento no teste¹ |
|---|---|---|---|---|---|
| TCC-I (split por arquivo) | — | 210.832 | — / 31.625 / 31.625 | 106.403 | 57,35% (18.136 de 31.625) |
| Ciclo 1 | 170.152² | 141.107 | 99.394 / 20.193 / 21.520 | 96.463 | 0,18% (39 de 21.520) |
| Ciclo 2 (rótulos com defeito) | 179.957 | 147.929 | 103.355 / 23.307 / 21.267 | 97.978 | 0,19% (41 de 21.267) |
| **Ciclo 3 — base de seis fontes** | **176.126** | **170.772** | **120.861 / 25.944 / 23.967** | 97.626 | 0,04% (10 de 23.967) |
| **Ciclo 4 — base + espinafre + míldio** | — | **175.123** | **123.879 / 26.620 / 24.624** | 101.987 | 0,04% (10 de 24.624) |

¹ Pelo hash perceptual (Seção 4). Não enxerga variantes muito alteradas da mesma foto
(Seção 5). ² Número dos documentos; o log disponível do ciclo 1 começa depois do split.

- **De 176.126 para 170.772:** o split descarta 1 imagem corrompida e 5.353 imagens de
  167 grupos em que a mesma foto aparecia com os dois rótulos.
- **O defeito de rotulagem custou mais do que os 3.972 rótulos trocados.** No ciclo 2, os
  grupos com rótulos contraditórios eram 1.239 (32.027 imagens descartadas); corrigida a
  rotulagem, caíram para 167 (5.353). Foi assim que a partição passou de 147.929 para
  170.772 imagens.
- **Os testes dos ciclos 1, 2 e 3 são conjuntos diferentes** (21.520, 21.267 e 23.967
  imagens). Só ciclo 3 e ciclo 4 são comparáveis foto a foto.
- **Ciclo 4:** +4.351 imagens, sem refazer o split da base. Espinafre-de-malabar
  (*Basella alba*): 3.209 (2.245 / 480 / 484). Alface com míldio: 1.142 recortes de 49
  fotografias (773 / 196 / 173), particionados pela foto de origem.
- **Classes na base:** 72.601 `healthy` e 98.171 `anomalous` (1,35 para 1). No teste,
  9.908 e 14.059 (58,66% anômalas).

### 3.1 As seis fontes (staging do ciclo 3)

| Fonte (Kaggle) | O que é | `healthy` | `anomalous` | Total |
|---|---|---|---|---|
| `nirmalsankalana/plant-diseases-training-dataset` | **Coletânea** de vários conjuntos: PlantVillage e outros (mandioca, arroz, rosa, cana, café, melancia) | 28.487 | 79.889 | 108.376 |
| `shuvokumarbasak2030/lettuce-disease-multi-transformation-dataset` | 17 transformações de cada foto de alface do `ashishjstar`, em duas cópias | 38.182 | 3.672 | 41.854 |
| `abdallahalidev/plantvillage-dataset` | PlantVillage: 54.306 imagens, **38 classes de 14 culturas**; usadas 15 pastas | 4.383 | 13.202 | 17.585 |
| `iqrapervez2000/lettuce-disease-dataset` | Alface; 86% das imagens derivam das fotos do `ashishjstar` | 2.849 | 1.418 | 4.267 |
| `santoshshaha/lettuce-plant-disease-dataset` | Alface, com imagens já aumentadas | 1.200 | 1.613 | 2.813 |
| `ashishjstar/lettuce-diseases` | Alface: 1.123 fotos saudáveis e 108 doentes | 1.123 | 108 | 1.231 |
| **Total** | | **76.224** | **99.902** | **176.126** |

- **Ficaram fora, por não serem reconhecidas pela rotulagem automática (8.555 imagens):**
  `Pest damage` de alface (513), `Potato___pests` (611), `Rose___slug_sawfly` (4.979),
  `Tomato___spider_mites` (1.676), `Sugercane___yellow_leaf` (505) e o conjunto
  `ramadhanihsaniyulfa` inteiro (271). **Quatro dessas pastas são dano por praga**, o tipo
  de sintoma estrutural em que o modelo mais erra em campo.
- Vinte conjuntos do Roboflow foram tentados e nenhum aproveitado no TCC-I (eram de
  detecção); no ciclo 4, seis de alface foram avaliados e um aprovado (o lote de míldio).

### 3.2 Do que é feito o treino-base (120.861 imagens)

| Recorte | Imagens | % do treino | O que se observa |
|---|---|---|---|
| Coletânea `nirmalsankalana` | 72.731 | 60,2% | 25% `healthy` |
| `shuvokumarbasak2030` (variantes de alface) | 30.156 | 25,0% | 91% `healthy` |
| `plantvillage` | 12.006 | 9,9% | 24% `healthy` |
| `iqrapervez2000`, `santoshshaha`, `ashishjstar` | 5.968 | 4,9% | — |
| **Alface (quatro fontes)** | **36.124** | **29,9%** | **86,6% `healthy`** |
| **Demais culturas** | **84.737** | **70,1%** | **24,9% `healthy`** |
| PlantVillage + culturas do PlantVillage dentro da coletânea | 57.651 | **47,7%** | Estúdio: folha isolada, fundo liso |
| Culturas da coletânea fotografadas em lavoura (mandioca, arroz, cana, café) | 21.325 | **17,6%** | Planta no solo, fundo natural |

- **A afirmação "70% do treino deriva do PlantVillage" estava errada.** Os 70,1% são a
  soma de duas *fontes*, e uma delas é uma coletânea em que 37,2% das imagens (27.086) são
  de culturas que o PlantVillage não tem. O derivado do PlantVillage é **no máximo 47,7%**.
- **O treino já contém fotos de lavoura: 17,6%**, de mandioca (14.999, do conjunto Cassava
  Leaf Disease), arroz, cana e café. Não contém foto de lavoura de hortaliça folhosa.
- **Cultura e rótulo estão associados.** 59,7% da classe `healthy` é alface, e só 7,1% da
  classe `anomalous` é alface. Arroz, laranja e abóbora só têm o rótulo `anomalous`; soja,
  mirtilo e framboesa só têm `healthy`. Uma regra "alface → saudável, o resto → anômalo"
  acertaria 78,5% do treino sem olhar sintoma algum.
- **A alface do treino vem de 1.231 fotografias originais**, 1.123 saudáveis e 108 doentes
  (bacteriana 20, míldio 30, oídio 18, septoriose 19, viral 15, murcha 6). O resto são
  cópias e transformações. O ciclo 4 acrescentou 49 fotos de míldio.

## 4. Deduplicação e particionamento

| Item | Como é feito | Onde |
|---|---|---|
| Hash | dHash de 64 bits, depois de converter para cinza e normalizar o contraste | `imagehash_utils.py` |
| Simetrias | Órbita das oito transformações do grupo diedral D4 (rotações de 90° e espelhos) | idem |
| Quase-duplicata | Distância de Hamming ≤ 4 entre órbitas | idem |
| Busca | LSH com 5 bandas (garante achar todo par até distância 4) e união-busca | idem |
| Split | Por **grupo**, 70/15/15, estratificado por rótulo, semente 42 | `download_datasets.py` |
| Descartes | Imagens corrompidas; grupos com os dois rótulos | idem |
| Auditoria | Fração do teste com quase-duplicata no treino, reportada como métrica | `audit_leakage.py` |

Como o split é por grupo, a proporção em imagens é 70,8 / 15,2 / 14,0.

## 5. O que o hash não vê: vazamento por proveniência

O hash junta espelhos, rotações de 90° e ajustes leves de brilho. **Não junta** uma
rotação de ângulo livre, um recorte, uma translação ou um filtro de bordas à foto de onde
vieram. A fonte `shuvokumarbasak2030` é feita exatamente disso: 17 transformações de cada
uma de 1.231 fotos (`brightness_adjusted`, `contrast_adjusted`, `cropped`,
`flipped_horizontal`, `flipped_vertical`, `gaussian_noise`, `high_pass`, `hist_equalized`,
`jittered`, `laplacian`, `poisson_noise`, `rotated`, `salt_pepper_noise`,
`saturation_adjusted`, `sobel`, `translated`, `unsharp_mask`), com o id da foto original
no nome do arquivo.

| Fonte | Imagens no teste | Com outra variante da mesma foto no treino | Como foi medido |
|---|---|---|---|
| `shuvokumarbasak2030` | 5.616 | **5.592 (99,6%)** | Id da foto original no nome do arquivo |
| `ashishjstar` | 153 | 153 (100%) | Ligação por hash às variantes pouco alteradas |
| `iqrapervez2000` | 550 | 450 (81,8%) | idem |
| `santoshshaha` | 398 | 141 (35,4%), no mínimo | idem; 257 sem foto original identificada |
| **Alface, total** | **6.717** | **6.336 (94,3%)** | |

- **23,3% do teste (5.592 de 23.967)** são variantes transformadas de fotos de alface que
  têm outra variante no treino; somando as outras três fontes, **26,4%**.
- **O "0,04% de vazamento" continua correto para o que mede** (quase-duplicatas) e não
  mede isto. É o mesmo tipo de problema do lote de míldio (recortes da mesma foto),
  corrigido no ciclo 4 só para aquele lote.
- Três das transformações (`sobel`, `laplacian`, `high_pass`) não são fotografias: são
  mapas de bordas.

## 6. Arquiteturas e parâmetros

Todos os modelos saem do `torchvision` com pesos `IMAGENET1K_V1`, e a camada final é
trocada por uma camada linear de duas saídas (`train.py`, `build_model`).

| Modelo | Arquitetura original (1.000 classes) | **Modelo treinado (2 classes)** | Arquivo de pesos |
|---|---|---|---|
| MobileNetV2 | 3.504.872 | **2.226.434 (2,2 M)** | 9 MB |
| EfficientNet-B0 | 5.288.548 | **4.010.110 (4,0 M)** | 16 MB |
| ResNet-50 | 25.557.032 | **23.512.130 (23,5 M)** | 94 MB |
| ViT-B/16 | 86.567.656 | **85.800.194 (85,8 M)** | 343 MB |

- ViT-B/16 ÷ EfficientNet-B0: **21,4 vezes** nos modelos treinados (16,4 nas arquiteturas
  originais). ResNet-50 ÷ MobileNetV2: **10,6 vezes** (7,3). ResNet-50 ÷ EfficientNet-B0:
  5,9 vezes (4,8).
- Os documentos citavam 5,3 M, 3,4 M, 25,6 M e 86,6 M, que são os tamanhos publicados das
  arquiteturas para 1.000 classes, e "16 vezes". Na revisão da literatura do artigo esses
  números continuam, identificados como da versão original; na Tabela 1 e nas razões
  passam a valer os dos modelos treinados.

## 7. Treino

| Item | CNNs (ciclos 1 a 4) | ViT-B/16 |
|---|---|---|
| Ajuste | Fino completo, todas as camadas | idem |
| Épocas | Até 30, parada antecipada com paciência 10 | idem |
| Lote | **64** | 64 |
| Otimizador | AdamW, decaimento de pesos 10⁻⁴ | AdamW, decaimento de pesos 0,05 |
| Taxa de aprendizado | 10⁻³, cosseno até 10⁻⁵ | 5×10⁻⁵, 2 épocas de aquecimento linear, cosseno até 5×10⁻⁷ |
| Gradiente | Sem limitação | Norma limitada a 1,0 |
| Perda | Entropia cruzada, *label smoothing* 0,1, pesos de classe | idem |
| Amostragem | Ponderada, para igualar as duas classes | idem |
| Precisão | Mista (AMP) | idem |
| Seleção | Melhor F1 macro na validação | idem |
| Entrada | 224 × 224 | idem |

- **Aumento de dados no treino:** `RandomResizedCrop(224, escala 0,6–1,0)`, espelho
  horizontal e vertical, rotação de ±30°, `ColorJitter` (brilho 0,3, contraste 0,3,
  saturação 0,15, matiz 0,02), `RandomErasing` (p = 0,25). Validação, teste e produção:
  `Resize(256)` + `CenterCrop(224)`.
- **Uma execução e uma semente por modelo.**
- **A parada antecipada não disparou nos ciclos 3 e 4** (só no MobileNetV2 do ciclo 1, na
  época 19). O melhor checkpoint do ciclo 4 foi na época 28 (EfficientNet-B0), 24
  (MobileNetV2) e **30, a última** (ResNet-50); o do ViT, na época 29.

| Execução | Tempo de treino | Tempo do pipeline inteiro |
|---|---|---|
| Ciclo 1, três CNNs | 12,7 h | 14,5 h (execução retomada) |
| Ciclo 2 | — | 17,5 h |
| Ciclo 3 | 18,1 h | 22,2 h |
| Ciclo 4 | 20,7 h (6,4 + 6,4 + 8,0) | 22,8 h |
| ViT-B/16, GPU L4 no Colab | 6,1 h | 6,2 h |

Os "cerca de 80 horas de GPU" dos documentos são a soma dos pipelines inteiros (hash,
split, treino, calibração e avaliação): 77 h, mais 6 h do ViT.

## 8. Calibração e abstenção

Temperatura ajustada por L-BFGS sobre a log-verossimilhança negativa na validação; ECE
com 15 faixas de igual largura; τ é o menor valor a partir de 0,85, em passos de 0,01,
com acurácia seletiva de pelo menos 98% na validação (`calibrate.py`).

| Modelo (ciclo 4) | *T* | ECE antes → depois (validação) | τ | Cobertura | Acurácia seletiva |
|---|---|---|---|---|---|
| EfficientNet-B0 | 0,561 | 0,0509 → 0,0027 | 0,85 | 98,9% | 99,0% |
| MobileNetV2 | 0,546 | 0,0511 → 0,0047 | 0,85 | 97,6% | 99,1% |
| ResNet-50 | 0,566 | 0,0484 → 0,0050 | 0,85 | 97,2% | 98,9% |
| ViT-B/16 | 0,620 | 0,0472 → 0,0022 | 0,85 | 99,5% | 98,7% |

**Todas as temperaturas são menores que 1: a calibração aumenta a confiança.** É o efeito
esperado do *label smoothing*, que deixa a rede pouco confiante nas fontes de treino.

## 9. Resultados verificados

### 9.1 Teste original (N = 23.967), ciclo 4

| Modelo | Acur. bal. | IC 95% | F1 macro | AUC | ECE | FN | FP | Erros | Interceptados (conf < τ) | Abstenção | Latência |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | 98,48% | 98,32–98,64 | 98,44% | 0,9982 | 0,0039 | 209 | 154 | 363 | 113 (31,1%) | 1,2% | 7,2 ms |
| ViT-B/16 | 98,48% | 98,32–98,64 | 98,53% | 0,9976 | 0,0018 | 137 | 204 | 341 | 51 (15,0%) | 0,5% | 19,4 ms |
| MobileNetV2 | 97,89% | 97,70–98,07 | 97,83% | 0,9976 | 0,0055 | 294 | 211 | 505 | 249 (49,3%) | 2,6% | 5,3 ms |
| ResNet-50 | 97,82% | 97,63–98,00 | 97,70% | 0,9966 | 0,0061 | 352 | 184 | 536 | 233 (43,5%) | 2,5% | 8,6 ms |

- IC por *bootstrap* estratificado por classe, 10.000 reamostragens, semente 42. McNemar
  exato (binomial sobre as discordâncias) com correção de Holm sobre os seis pares.
- EfficientNet-B0 × ViT-B/16: 158 contra 180 discordâncias, p de Holm = 0,26. Ambos
  superam MobileNetV2 e ResNet-50 (p de Holm de 1,4×10⁻¹² a 3,5×10⁻¹⁷). MobileNetV2 ×
  ResNet-50: p = 0,26.
- Confiança calibrada média do EfficientNet-B0: 0,990 nos acertos e 0,874 nos erros
  (Mann-Whitney, p < 10⁻¹⁶⁰).
- Latência medida em 13/10 na RTX 3050, os quatro modelos na mesma sessão.

### 9.2 O mesmo teste, por subconjunto — onde estão os erros

| Subconjunto do teste | N | EfficientNet-B0 | ViT-B/16 | MobileNetV2 | ResNet-50 |
|---|---|---|---|---|---|
| Coletânea, **culturas fotografadas em lavoura** (mandioca, arroz, cana, café) | 4.530 | **86,3%** [84,5–88,0], 295 erros | 83,4% [81,5–85,3] | 86,6% [84,8–88,3] | 87,8% [86,2–89,4] |
| Coletânea, **demais culturas** | 10.061 | **99,7%** [99,6–99,8], 29 erros | 99,8% | 99,5% | 99,4% |
| Só mandioca | 3.150 | 80,9% [78,5–83,3], 271 erros | 75,7% [73,1–78,2] | 81,3% | 82,8% |
| Fonte PlantVillage | 2.659 | 99,9%, 3 erros | 99,9% | 99,8% | 99,7% |
| Variantes transformadas de alface | 5.616 | 97,7% [96,7–98,5], 32 erros | 98,9% | 95,6% | 94,6% |
| Teste sem as variantes transformadas | 18.351 | 97,8% [97,5–98,1] | 97,6% | 97,5% | 97,5% |

- **81% dos erros do EfficientNet-B0 (295 de 363) estão em 19% do teste**: as culturas
  fotografadas em lavoura. Só a mandioca responde por 271.
- **Dentro da mesma fonte, a diferença entre lavoura e estúdio é de 13 pontos** (86,3%
  contra 99,7%), com N de milhares. É a medida de *domain gap* mais sólida do projeto até
  aqui, e sai do próprio teste.
- **Nas culturas de lavoura as arquiteturas quase não se distinguem** (só EfficientNet-B0
  × ResNet-50 passa na correção de Holm, p = 0,043), e o ViT-B/16 tem a menor acurácia
  balanceada. A vantagem de EfficientNet-B0 e ViT sobre os outros dois vem das imagens de
  estúdio e das variantes de alface.
- **Sem as variantes transformadas (N = 18.351), a conclusão de Q1 se mantém**, com
  evidência menor: EfficientNet-B0 × ViT, p = 0,32; os dois contra MobileNetV2 e ResNet-50,
  p de Holm de 1,7×10⁻⁴ a 1,8×10⁻⁸.
- **Ressalva:** a classificação "fotografada em lavoura" foi feita por inspeção visual de
  amostras de cada cultura. O efeito mistura condição de captura e qualidade dos rótulos
  dessas fontes; o projeto não separa os dois.

### 9.3 Fonte nunca vista e ciclo 3 × ciclo 4

| Medida | EfficientNet-B0 | ResNet-50 | MobileNetV2 | ViT-B/16 |
|---|---|---|---|---|
| Espinafre, modelo do ciclo 3 (nunca viu a fonte), N = 484 | **75,4%** | 77,6% | 74,3% | — |
| Espinafre, ciclo 4 (fonte no treino) | 88,4% [85,3–91,2] | 86,2% | 87,4% | 84,4% [80,9–87,7] |
| p de Holm, ciclo 3 × ciclo 4 no espinafre | 5,9×10⁻¹² | 1,9×10⁻⁷ | 1,9×10⁻⁷ | — |
| Teste original, ciclo 3 → ciclo 4 | 98,49 → 98,48 (p = 1,00) | 97,73 → 97,82 (p = 1,00) | 98,18 → 97,89 (p de Holm = 0,015) | — |

### 9.4 Campo: 13 fotografias de alface

| Versão | Diagnósticos | Acertos | Proporção | IC 95% (Wilson) | Acur. bal. | ECE |
|---|---|---|---|---|---|---|
| Ciclo 1 | 12 de 13 | 11 | 91,7% | 64,6–98,5 | 87,5% | 0,066 |
| Ciclo 2 | 13 de 13 | 10 | 76,9% | 49,7–91,8 | 77,5% | 0,217 |
| Ciclo 3 | 11 de 13 | 8 | 72,7% | 43,4–90,3 | 73,3% | 0,274 |
| Ciclo 4 | 13 de 13 | 11 | 84,6% | 57,8–95,7 | 83,8% | 0,146 |
| ViT-B/16 (dados do ciclo 4) | 12 de 13 | 12 | 100% | 75,8–100 | — | — |

- **Os 9 erros dos quatro ciclos são de 4 fotografias**: `healthy/alface_04.jpg` (4
  vezes), `anomalous/alface_01.jpg` (3), `anomalous/alface_02.jpg` (1) e
  `anomalous/alface_05.png` (1). Confiança entre 0,949 e 0,999.
- **`healthy/alface_04.jpg` tem rótulo duvidoso.** A foto mostra uma folha com manchas
  escuras sendo segurada, e os quatro ciclos a dão como anômala. Se o rótulo estiver
  errado, o ciclo 4 passa a 12 de 13 (92,3%; 66,7–98,6); se a foto for excluída, 11 de 12
  (91,7%; 64,6–98,5). A conclusão não muda: os intervalos continuam sobrepostos.
- **A temperatura ajustada na validação piora a calibração em campo.** No ciclo 4, o ECE
  de campo é 0,146 com a temperatura e 0,094 sem ela (ciclo 3: 0,274 e 0,243; ciclo 1:
  0,066 e 0,033). Mesmo sem temperatura, os erros sairiam com 0,93 a 0,96 e não seriam
  interceptados por τ = 0,85. Com 13 fotos, é indício.
- **Procedência.** São 13 arquivos JPEG (quatro com extensão `.png`), de 516 a 739 px no
  lado maior. Nenhum tem EXIF de celular. `anomalous/alface_01.jpg` traz EXIF de câmera
  Sony DSC-H70, data de 07/03/2014 e Adobe Photoshop CS6. `healthy/alface_01_bom1.jpg` tem
  faixas pretas de quadro de vídeo. `healthy/alface_04.jpg` e `anomalous/alface_08.jpg`
  têm a mesma resolução e o mesmo tipo de cena. Não há metadados registrados
  (`backend/campo/metadados_campo.csv` só tem as linhas de exemplo) e a pasta `nao_folha/`
  está vazia.
- **Nenhuma das 13 é quase-duplicata de imagem do dataset** (menor distância de Hamming
  7, limiar 4, contra 462.950 hashes), nem de outra das 13.

## 10. O que a auditoria achou, em ordem de importância

| # | Achado | Efeito |
|---|---|---|
| A1 | 81% dos erros do teste estão nas culturas fotografadas em lavoura; nelas a acurácia balanceada é de 86,3%, contra 99,7% no resto da mesma fonte | **Novo resultado.** Mede o *domain gap* dentro do próprio teste, com N = 4.530 |
| A2 | 23% do teste (26% com as outras fontes de alface) são variantes de fotos de alface com outra variante no treino | O vazamento de 0,04% não cobre proveniência; o 98,48% está, nessa parte, inflado |
| A3 | "70% do treino deriva do PlantVillage" está errado: é no máximo 47,7%, e 17,6% já é foto de lavoura | Corrige a explicação da queda em campo |
| A4 | Cultura e rótulo estão associados: 87% da alface do treino é saudável; várias culturas têm um só rótulo | Atalho disponível para a rede; coerente com alface doente dada como saudável em campo |
| A5 | A alface do treino vem de 1.231 fotos originais, 108 delas doentes | "Só a alface tem dados próprios" precisa dessa ressalva |
| A6 | Quatro pastas de dano por praga (8.284 imagens com o `yellow_leaf`) ficaram fora do treino | Reforça a hipótese H2 (sintomas estruturais) |
| A7 | Os modelos treinados têm 4,0 / 2,2 / 23,5 / 85,8 milhões de parâmetros; "16 vezes" é 21 | Corrige a Tabela 1 e três frases |
| A8 | Os 9 erros de campo são de 4 fotos, e uma tem rótulo duvidoso | Corrige a leitura de "nove erros" |
| A9 | A temperatura menor que 1 piora a calibração em campo | Hipótese levantada no guia da banca, agora com indício |
| A10 | 176.126, 170.772 e 175.123 são três etapas do mesmo dataset | Resolve a inconsistência de tamanho |
| A11 | O curso é Engenharia de Software | Corrigido no histórico e na memória do projeto |
| A12 | O PlantVillage tem 38 classes de 14 culturas, não "38 culturas" | Corrigido no histórico |

## 11. Errata por documento

| Documento | O que estava | O que passa a valer |
|---|---|---|
| Artigo v5 | "A versão atual contém 170.772 imagens" | v6: 176.126 reunidas, 170.772 na base, 175.123 com o ciclo 4 |
| Artigo v5 | Tabela 1 com 5,3 / 3,4 / 25,6 / 86,6 M; "16 vezes"; "7,5 vezes" | v6: 4,0 / 2,2 / 23,5 / 85,8 M; "21 vezes"; "dez vezes" |
| Artigo v5 | "70% das imagens de treino derivam do PlantVillage" | v6: frase trocada pelo resultado por subconjunto (A1) |
| Artigo v5 | "vazamento residual de 0,04%" sem ressalva | v6: "por esse critério", e a frase sobre as variantes de alface |
| Artigo v5 | "Os nove erros emitidos em campo" | v6: "concentrados em quatro fotografias" |
| Artigo v5 | Too et al. (2019) citado uma vez | v6: citação e referência retiradas, para caber em 11 páginas |
| `HISTORICO_COMPLETO_TCC.md` | Ciência da Computação; "38 culturas"; "Split refeito: 176.126"; "70% derivado do PlantVillage"; "16 vezes" | Corrigido, com a §9-E nova |
| `CORRECOES_METODOLOGICAS.md` | Tabela da §11.2 a ("re-upload do PlantVillage", 70,1%); "Novo split: 176.126" | Notas de correção nos trechos e §14 nova |
| `PROXIMOS_PASSOS_TCC2.md` | Linha "Fontes derivadas do PlantVillage: 70,1% / 72,0%"; artigo v4 | Corrigido; seção nova de 13/10 |
| `README.md` | "16 vezes"; "o grosso do dataset é PlantVillage"; "o que o modelo aprende sobre alface é transferido"; lote 32 | Corrigido |
| `GUIA_BANCA_TCC2.md` | Três inconsistências em aberto; 70%; 16 vezes; parâmetros | Resolvidas; guia revisto com os achados |
| `REVISAO_LITERATURA_WOS.md` | Causa 1 do diagnóstico ("70% derivam do PlantVillage") | Corrigido |
| `backend/reanalise_teste_novo.py` | Comentário tratava a coletânea inteira como PlantVillage | Comentário corrigido; o cálculo certo está em `auditoria_consistencia.py` |
| Artigo v7 | "O empate entre ViT-B/16 e EfficientNet-B0 diverge da robustez atribuída aos *transformers* (Shafay et al., 2025)"; "*transformers* apontados como mais robustos" | v8 (16/10): a revisão aponta os **híbridos** como mais robustos e o **ViT puro como mais sensível ao ambiente**, o que converge com o ViT-B/16, o pior na lavoura. Também: Nguyen (a perda vem de atribuir a folha à cultura errada) e Ergün (ECE 0,038 com confiança média de 0,55) |
| `GUIA_BANCA_TCC2.md` | "98,7% → 32,8%" (Gerz; Jelali) como se fosse acurácia | mAP50 de detecção, de 98,7% para cerca de 31% (o artigo deles dá 32,77% no resumo e 30,64% na tabela) |
| Monografia v3 (`TCC_FitoVision_ATUALIZADO_v3.docx`) | "Foram capturadas 20 fotografias [...] com telefone celular"; 5,3 M e 25,6 M; "nove erros" | **Corrigida na v4 (14/10)**, por errata na nota de atualização (o texto dos capítulos 1 a 8 fica como registro); a procedência das fotos continua pendente |
| `COMPILADO_FITOVISION.md`, `GUIA_TCC.md` | Documentos do TCC-I | Nota no topo apontando para esta ficha |

## 12. O que depende de você ou do orientador

1. **De onde vieram as 13 fotos de campo.** Os arquivos não sustentam "capturadas com
   telefone celular". Se foram reunidas de mensagens, vídeos ou da internet, o texto
   precisa dizer isso, e elas continuam valendo como conjunto histórico.
2. **O rótulo de `healthy/alface_04.jpg`.** Conferir e decidir: manter, trocar ou excluir.
3. **Levar o resultado por subconjunto (A1) ao orientador.** Está no artigo (v6 a v8) em uma
   frase; é resultado novo e pode merecer mais espaço.
4. ~~**Lançar o treino do ciclo 5.**~~ Lançado no Colab em 14/10 às 06:06, antes das fotos
   de campo (Seção 13). Falta trazer os resultados e preencher esta ficha.
5. **Monografia.** Atualizar depois da coleta, com esta ficha ao lado.
6. ~~**As 10 fotos de 27/10 vieram sem EXIF de câmera.**~~ Origem confirmada: celular →
   Google Drive → notebook (zip). Mesma resolução e mesmo `Software` do POCO X6 Pro; o envio
   apagou modelo e data. Próximas coletas: copiar por cabo USB para manter o EXIF.
7. **Rótulo de `healthy/IMG_20261018_192328.jpg`** (lote de casa): a foto mostra pontos
   escuros e a base amarronzada. O gabarito da coleta vale; a dúvida fica registrada.

## 13. Ciclo 5 — pipeline corrigido (preparado em 13/10; treino lançado no Colab em 14/10)

Detalhe em `docs/CORRECOES_METODOLOGICAS.md` §15. Nada do ciclo 4 é alterado.

| Split do ciclo 5 | Saudável | Anômala | Total |
|---|---|---|---|
| Treino | 36.753 | 75.705 | 112.458 |
| Validação | 4.403 | 16.209 | 20.612 |
| Teste | 4.480 | 13.933 | 18.413 |
| Fonte retida (espinafre `n56pn9fncw`) | 135 | 439 | 574 |
| **Total** | | | **152.057** |

Auditoria do split (`results/ciclo5/leakage_report.json`, modo `--rapido`): **1,7%** do
teste e **1,6%** da validação com alguma imagem de treino a distância de órbita ≤ 4 —
veredicto OK (ciclo 4: 23%). É um teto: parte dos pares são colisões do dHash entre
culturas diferentes.

**EfficientNet-B0 do ciclo 5** (detalhe e tabela ciclo 4 × ciclo 5 em CORRECOES §15.7):

| Medida | Valor |
|---|---|
| Validação: F1 macro / acur. bal. | 0,9720 / 0,9780 (época 30, 109 min na L4) |
| Temperatura; ECE validação antes → depois | 0,655; 0,048 → 0,004 |
| Teste inteiro (18.413) | 97,4% [97,1; 97,7] |
| Lavoura (4.530) — ciclo 4 nas mesmas imagens | 88,8% [87,2; 90,3] — 86,3%; especificidade 83,4% × 76,6% |
| Alface, uma foto por original (176, 15 doentes) | 96,7% [90,0; 100,0] |
| Espinafre, fonte retida (574) | 87,9% [84,4; 91,2] |
| Mesmas 17.250 imagens, acerto por imagem | só c4 acerta 141, só c5 106 (McNemar p = 0,03) |
| Campo histórico (13) | 11 de 12 emitidos, 1 abstenção; × ciclo 4: 1 a 1, p = 1 |

**ViT-B/16 do ciclo 5** (CORRECOES §15.8): validação F1 0,9721 (época 23, 328 min);
T = 0,676; teste 96,9% [96,6; 97,2]; lavoura 84,5% [82,6; 86,3] (ViT do ciclo 4 nas
mesmas imagens: 83,4%, p = 0,66); alface uma por foto 100% (176); fonte retida 85,8%
[82,0; 89,5]. A amostragem por cultura quase não mudou o ViT.

**MobileNetV2 do ciclo 5** (CORRECOES §15.9): validação F1 0,9597 (época 24, 106 min);
T = 0,647; teste 96,9%; lavoura 90,1% [88,8; 91,3] com especificidade 90,2% e
sensibilidade 90,0% (ciclo 4: 78,3% e 94,9%); espinafre 79,5%; fonte retida 88,4%.
Por imagem, o ciclo 4 acerta mais (292 × 156, p = 1,3×10⁻¹⁰): troca de sensibilidade por
especificidade. Tabela completa dos três modelos: `backend/results/ciclo5/ciclo4_x_ciclo5.md`.

**ResNet-50 do ciclo 5**: validação F1 0,9607 (época 28, 151 min); T = 0,637; teste
96,9%; lavoura 89,6% (espec. 89,0%, sens. 90,2%; ciclo 4: 87,8%); fonte retida 90,1%
[86,7; 93,1], a melhor das quatro.

**Ciclo 5, resumo das quatro arquiteturas** (acurácia balanceada, %; CORRECOES §15.10):

| Modelo | F1 val. | Teste | Lavoura (c4 → c5) | Alface 1/foto | Fonte retida |
|---|---|---|---|---|---|
| EfficientNet-B0 | 0,972 | 97,4 | 86,3 → 88,8 | 96,7 | 87,9 |
| ViT-B/16 | 0,972 | 96,9 | 83,4 → 84,5 | 100 | 85,8 |
| MobileNetV2 | 0,960 | 96,9 | 86,6 → 90,1 | 100 | 88,4 |
| ResNet-50 | 0,961 | 96,9 | 87,8 → 89,6 | 100 | 90,1 |

Treino no Colab (L4) em 14/10, 06:06–17:52, cerca de 22 unidades.

| Achado | Correção no ciclo 5 |
|---|---|
| A2 — variantes de alface dos dois lados | Split por foto original; teste de alface com uma fotografia por foto (176 fotos, 15 doentes) |
| A2 — imagens que não são fotografias | Fora `sobel`, `laplacian`, `high_pass` e as cópias idênticas |
| A4 — cultura associada ao rótulo | Amostragem equilibrada por rótulo dentro de cada cultura; perda sem pesos de classe |
| A6 — pastas de praga fora do treino | 8.266 imagens entram como anômalas (8.284 nas pastas; 18 saem por rótulo contraditório com uma quase-duplicata) |
| A1 — média que esconde os erros | Avaliação por subconjunto (`avalia_ciclo5.py`) |
| Sem fonte retida para o modelo atual | Lote de espinafre `n56pn9fncw` (574 imagens) fora do treino |
| A8, A9 — campo | `confere_fotos_campo.py` antes da inferência; `roda_campo.py` com ECE com e sem temperatura e resultado por horta |

Mantidos de propósito: arquiteturas, hiperparâmetros, aumento de dados e a partição da
coletânea e do PlantVillage, para que ciclo 4 e ciclo 5 sejam comparáveis foto a foto
nessas 17.250 imagens de teste.

## 14. Campo com fundo real — primeiro lote (27/10/2026)

Detalhe e leitura em `docs/CORRECOES_METODOLOGICAS.md` §15.13; saídas em
`backend/results/campo_novo/` (`analise_campo.md`, `calibracao_e_hortas.md`,
`recortes/recortes.md`).

| Lote | Fotos | Ciclo 4 Eff | Ciclo 5 Eff | Ciclo 4 ViT | Ciclo 5 ViT |
|---|---|---|---|---|---|
| `horta_27out` (campo, fundo real) | 10 sadias | 10 de 10 | 10 de 10 | 10 de 10 | 10 de 10 |
| `casa_colhida_18out` (não é campo) | 3 sadias, 7 anômalas | 7 de 10 | 7 de 10 | 7 de 10 | 7 de 9 + 1 inconcl. |

- Horta: especificidade 10/10, IC de Wilson **72,2% a 100%**; confiança ≥ 0,986 em todas as
  versões; recortes centrais (½ e ¼ do lado): 10 de 10 nas duas escalas, nas quatro versões.
- Casa: as 3 sadias saem anômalas em todas as versões, com confiança ≥ 0,97; os erros
  continuam nos recortes.
- **Não citar** a acurácia balanceada das 20 fotos juntas (88,5%) nem o teste de H3
  (p = 0,03): classe, cena e distância estão confundidas entre os dois lotes.
- Sensibilidade em campo e *domain gap*: **sem medida** (nenhuma folha doente na horta).

