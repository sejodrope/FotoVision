# Revisão da literatura — Web of Science (13/10/2026)

Base: Web of Science Core Collection, acesso CAPES, busca avançada por tópico (`TS=`), sem filtro de ano.
Os números e conclusões abaixo vêm dos **resumos** indexados na WoS; o texto completo dos artigos ainda não foi lido.

## Buscas

| # | Tema | Consulta | Resultados |
|---|------|----------|-----------:|
| 1 | Generalização laboratório → campo | `TS=(("plant disease*" OR "leaf disease*" OR "crop disease*") AND ("deep learning" OR "convolutional neural network*" OR "transfer learning") AND ("field condition*" OR "in-the-wild" OR "real-world" OR "in-field") AND (generaliz* OR "domain shift" OR "domain gap" OR "cross-dataset" OR "domain adaptation"))` | 167 |
| 2 | Vazamento e viés de dataset | `TS=(("plant disease*" OR "leaf disease*" OR "crop disease*" OR PlantVillage) AND ("data leakage" OR "near-duplicate*" OR "dataset bias" OR "background bias" OR "shortcut learning" OR "spurious correlation*" OR "duplicate images"))` | 20 |
| 3 | Calibração e abstenção | `TS=(("plant disease*" OR "leaf disease*" OR "crop disease*") AND ("deep learning" OR "neural network*") AND ("temperature scaling" OR "confidence calibration" OR "selective classification" OR "expected calibration error" OR "reject option" OR abstention OR "calibrated" OR "selective prediction"))` | 24 |
| 4 | Hortaliças folhosas | `TS=((lettuce OR spinach OR "leafy vegetable*" OR arugula OR kale OR chard OR "leafy green*") AND (disease* OR "nutrient deficienc*" OR anomal* OR stress) AND ("deep learning" OR "convolutional neural network*" OR "vision transformer*" OR "transfer learning") AND (classif* OR detect* OR diagnos*) AND image*)` | 46 |
| 5 | ViT × CNN fora do domínio | `TS=(("plant disease*" OR "leaf disease*") AND ("vision transformer*" OR ViT) AND (CNN* OR convolutional) AND ("comparative" OR comparison OR benchmark*) AND ("cross-dataset" OR "cross-domain" OR PlantDoc OR "out-of-distribution" OR "unseen"))` | 7 |

Triagem por título e resumo: busca 1, os 50 primeiros; buscas 2 a 5, até os 35 primeiros (todos nas buscas 2, 3 e 5).

## Trabalhos incluídos no artigo (v5, v6 e v7)

| Trabalho | Periódico | O que faz | Resultado que interessa | Relação com o FitoVision |
|---|---|---|---|---|
| Workneh et al. (2026), DOI 10.1016/j.compag.2026.112464 | Comput. Electron. Agric. v. 256 (fascículo jan/2027, on-line set/2026) | Amêndoa, 1.717 imagens de campo, 8 classes; validação agrupada, temperature scaling, predição seletiva, explicabilidade | F1 macro 0,9941 (hold-out aleatório) → 0,9797 (5-fold agrupado) → 0,5587 em fontes disjuntas (N = 46, IC [0,44; 0,60], tratado como exploratório); ECE 0,036 → 0,025; 99,7% de acurácia a ~90% de cobertura | **O mais parecido**: mesmo protocolo (vazamento, calibração, abstenção, fonte disjunta) e mesma conclusão — a perda só aparece quando a fonte é separada |
| Ergün et al. (2026), DOI 10.3389/fpls.2026.1839652 | Front. Plant Sci. v. 17 | Videira; EfficientNet-B0 e MobileNetV3-Large; terceiro conjunto reservado para validação externa sem vazamento | EfficientNet-B0: 73,8% de acurácia externa; ECE bruto 0,287 → 0,038 com temperature scaling | Mesma arquitetura, queda externa da mesma ordem (75,4% aqui); difere na calibração (ECE de campo 0,146 aqui) |
| Nguyen et al. (2026), DOI 10.1016/j.atech.2026.102531 | Smart Agric. Technol. v. 15 | PlantVillage → PlantDoc; DINOv2 congelado × treino completo; McNemar exato por imagem | >99% no domínio → 26–30% no PlantDoc; DINOv2 congelado 51,1%; ajuste fino completo derruba de 50,6% para 29,4% | Sustenta "o limite está nos dados, não na arquitetura"; usa McNemar como aqui. Sugere experimento futuro: backbone congelado |
| Shafay et al. (2025), DOI 10.1186/s13007-025-01450-0 | Plant Methods v. 21 | Revisão sistemática, 11 datasets | Laboratório 95–99% × campo 70–85%; SWIN 88% × CNNs 53% em dados reais | Faixa de referência do domain gap. ~~**Diverge** do empate ViT × EfficientNet~~ — corrigido na Parte 4 (16/10): a revisão aponta o ViT puro como mais sensível ao ambiente, o que **converge** com este trabalho |
| Kabya et al. (2026), DOI 10.1371/journal.pone.0340989 | PLOS ONE v. 21 | Espinafre-de-malabar, 3 classes; SimSiam + CBAM-ResNet-50; ViT e SwinV2 | 97,31% no teste interno; SwinV2-Base 97,98% | Mesma cultura do lote em que o modelo do ciclo 3 fez 75,4% sem tê-la visto |
| Abbasi, Martinez e Ahmad (2023), DOI 10.1016/j.aiia.2023.09.001 | Artif. Intell. Agric. v. 10 | Alface, manjericão, espinafre e salsa em aquaponia; 2.640 imagens | 95,83% (cultura), 94,13% (estado de saúde), 82,13% (detecção) | Único com alface + espinafre e saída saudável × doente |

## Lidos e não incluídos (por espaço) — úteis para a monografia e a defesa

- Salman, Muhammad e Han (2025), Front. Plant Sci. v. 16, DOI 10.3389/fpls.2025.1522985 — ViT + mistura de especialistas; 68% em PlantVillage → PlantDoc.
- Shrivastava et al. (2026), Sci. Rep. v. 16, DOI 10.1038/s41598-026-55107-6 — ViT hierárquico com MC Dropout e calibração; 97,8% controlado e 93,6% em campo (PlantDoc), ECE 0,032/0,041. Resultado bem mais otimista que os demais; vale ler o protocolo antes de citar.
- Zhan et al. (2025), Eng. Appl. Artif. Intell. v. 156, DOI 10.1016/j.engappai.2025.111168 — benchmarks públicos de generalização de domínio laboratório → campo (CNNs, transformers e modelos de fundação).
- Chansanam et al. (2026), Appl. Sci. v. 16(14) — transfer learning "leakage-aware" em manga (MangoLeafBD).
- Gui et al. (2021), Comput. Electron. Agric. v. 191 — reconhecimento em campo.
- Outros de hortaliças folhosas: Xu et al. (2024, espinafre), Nasra et al. (2025, alface), Hamidon e Ahamed (2022, tip-burn em alface), Bose, Roy e Sutradhar (2023, espinafre).

## O que a literatura confirma e o que contradiz

- **Confirma**: desempenho intra-fonte não prevê desempenho fora da fonte (Workneh, Ergün, Nguyen, Shafay); calibração precisa ser verificada no domínio de uso (Ergün); trocar a arquitetura com ajuste fino completo não fecha o gap (Nguyen).
- ~~**Contradiz ou tensiona**: a revisão de Shafay et al. aponta transformers como mais robustos em campo; aqui o ViT-B/16 empata com o EfficientNet-B0 no teste e não o supera no lote de espinafre (84,4% × 88,4%, p = 0,51).~~
  **Corrigido na Parte 4 (16/10):** a revisão aponta os híbridos (Swin) como mais robustos e o ViT puro como mais sensível ao ambiente, o que **converge** com o ViT-B/16 daqui (empata no teste, é o pior na lavoura).
- **Lacuna**: nenhum resultado das cinco buscas reúne, para hortaliças folhosas, auditoria de vazamento, calibração com abstenção e avaliação fora da fonte de treino.

---

# Parte 2 — Por que o modelo não se sustenta fora do laboratório (13/10/2026)

Segunda rodada na WoS, voltada às **causas** da queda em campo e ao que a literatura mostra que funciona. Como na Parte 1, os números dos artigos vêm dos resumos, salvo indicação.

## Buscas adicionais

| # | Tema | Consulta (resumida) | Resultados |
|---|------|---------------------|-----------:|
| 6 | Viés de fundo e de dataset | doença de plantas/PlantVillage + background/dataset bias/laboratory conditions + background removal/segment*/bias + generaliz*/field images/cross-dataset | 35 |
| 7 | Culturas não vistas e diversidade de dados | doença de plantas + unseen crop/cross-crop/cross-species/data diversity/factors influencing | 69 |
| 8 | Trabalhos clássicos, por título | Barbedo (2018, 2019), Ferentinos (2018), PlantDoc, Toda e Okura (2019), Brahimi et al. (2018) | 7 |

Uma busca sobre adaptação com poucas imagens de campo (few-shot/fine-tuning/domain adaptation) não retornou resultados na interface e não foi refeita; o tema está coberto por Yang et al. (2024), vindo da busca 1.

## Diagnóstico: o que deu errado, em ordem de peso

> **Atualização de 13/10 (noite).** A auditoria de consistência corrigiu a causa 1 e acrescentou evidência do próprio projeto: 86,3% nas culturas fotografadas em lavoura contra 99,7% nas de estúdio, dentro do mesmo teste; 23% do teste com vazamento por proveniência; alface com 1.231 fotos originais, 87% saudável. A versão revista do diagnóstico está em `docs/GUIA_BANCA_TCC2.md` §5; os números, em `docs/FICHA_TECNICA_FITOVISION.md`.

> **Atualização de 14/10 — o ciclo 5 testou duas das causas.** Com a alface separada por fotografia original e a amostragem equilibrando os rótulos dentro de cada cultura (causas 1 e 2, do lado dos dados), as três CNNs ganham de 6,8 a 11,9 pontos de especificidade nas culturas fotografadas em lavoura e perdem de 1,8 a 4,9 de sensibilidade; o ViT quase não muda. É evidência direta de *shortcut learning* pela associação cultura–rótulo (Geirhos et al., 2020): tirar o atalho muda o comportamento do modelo onde ele estava. Num lote de espinafre retido do treino (com outro lote da espécie no treino), os quatro modelos ficam entre 85,8% e 90,1%, cerca de dez pontos abaixo das fontes vistas — a ordem de grandeza da queda por fonte disjunta de Workneh et al. (2026), mais branda por ser só troca de lote. Detalhe em `docs/CORRECOES_METODOLOGICAS.md` §15.10.

O treino em si não é o problema. Quatro arquiteturas, de 2,2 a 85,8 milhões de parâmetros nos modelos treinados, chegam todas a 97,8–98,5% no teste, e trocar CNN por transformer não muda nada. O que falha é a **relação entre os dados de treino e o uso real**. E o desempenho em campo ainda não está medido: 83,8% em 13 fotos, com intervalo de 58% a 96%, não permite dizer nem que funciona nem que não funciona.

| # | Causa | Evidência no FitoVision | O que a literatura diz |
|---|-------|-------------------------|------------------------|
| 1 | **Pouca diversidade de condições de captura na cultura-alvo** | ~~70,1% do treino deriva do PlantVillage~~ — corrigido em 13/10: é no máximo 47,7%, e 17,6% do treino já é foto de lavoura, mas de mandioca, arroz, cana e café. No próprio teste, as culturas fotografadas em lavoura dão 86,3% contra 99,7% das de estúdio (ficha técnica, §9.2) | É a causa mais documentada. Classificadores treinados no PlantVillage passam de 99% e caem para 26–30% no PlantDoc (Nguyen et al., 2026). O principal problema é a falta de bases que representem a variedade de condições e sintomas da prática, e o aumento de dados não reproduz essa diversidade (Barbedo, 2019). CNNs treinadas em laboratório vão mal em campo por diferenças de resolução, iluminação, apresentação do sintoma e fundo (Aarizou; Merah; Mahiout, 2026a) |
| 2 | **O teste se parece com o treino** | As 6 fontes aparecem nas mesmas proporções no treino e no teste; 98,5% é acurácia intra-fonte | A validação agrupada quase não muda o F1 (0,994 → 0,980); só a separação por fonte revela a perda (0,56) (Workneh et al., 2026). Resultados positivos escondem problemas raramente considerados nos experimentos (Barbedo, 2018b) |
| 3 | **Espécies-proxy no lugar das espécies-alvo** | Só a alface tem dados próprios; o espinafre nunca visto caiu para 75,4% mesmo sendo foto de estúdio — ou seja, 23 pontos se perdem só por mudar de fonte e de espécie, sem mudar o tipo de fundo | Os sintomas ficam acoplados à aparência da cultura, o que derruba o desempenho em culturas ausentes do treino (Wu et al., 2026; Chai et al., 2024). Rotular pela doença, e não pelo par cultura-doença, melhora a generalização para espécies não vistas (Aarizou; Merah; Mahiout, 2026b) |
| 4 | **Classe "anômalo" heterogênea, com sintomas estruturais pouco representados** | Os erros de campo do TCC-I concentram-se em murcha e perfurações (hipótese H2) | As CNNs capturam cor e textura das lesões (Toda; Okura, 2019), o que é coerente com acertar manchas e errar murcha. Trabalhar com lesões individuais, em vez da folha inteira, elevou a acurácia em 12 pontos em média (Barbedo, 2019) |
| 5 | **Calibração feita só no domínio de treino** | Os 9 erros de campo tiveram confiança de 0,949 a 0,999; ECE de 0,003 na validação e 0,146 em campo; a abstenção intercepta 31% (EfficientNet-B0) e 15% (ViT) dos erros | Sob mudança de distribuição a confiança deixa de refletir o acerto: ECE bruto de 0,287 em conjunto externo (Ergün et al., 2026) |
| 6 | **Defeitos de protocolo já corrigidos** | Vazamento de 57,35% no teste do TCC-I; 3.972 imagens de alface saudável rotuladas como doentes | Protocolo "leakage-aware" virou tema próprio (Workneh et al., 2026; Chansanam et al., 2026) |

### O que **não** é a causa

- **Arquitetura e capacidade.** ViT-B/16 empata com EfficientNet-B0; o ajuste fino completo de um modelo maior chega a piorar a transferência (de 50,6% para 29,4% em Nguyen et al., 2026).
- **Quantidade de imagens.** São 175 mil; o que falta é diversidade de condições de captura (Barbedo, 2018a). A alface, 30% do treino, vem de 1.231 fotos originais.

### O que ainda não foi testado (dizer isso na defesa)

- Dependência do fundo: o teste com fundo removido / folha removida está planejado e não foi rodado. A queda no espinafre de estúdio mostra que o fundo não explica tudo.
- Uma semente por modelo: não há estimativa de variância entre treinos.
- Se o ajuste fino completo prejudicou a transferência. Nguyen et al. (2026) mostram que um backbone auto-supervisionado **congelado** com classificador linear transfere melhor (51,1% contra 27–30%). É um experimento barato para um próximo ciclo (o ciclo 5, de 14/10, corrigiu os dados e manteve o ajuste fino completo, para atribuir o efeito).

## O que a literatura mostra que funciona

| Medida | Resultado reportado | Como aplicar aqui |
|--------|---------------------|-------------------|
| Misturar imagens de campo às de laboratório no treino | 77,5–80,3% de generalização quando o PlantVillage é combinado com imagens de campo (Ahmad; Gamal; Saraswat, 2023); PlantDoc eleva a acurácia em até 31% (Singh et al., 2020) | Conjunto de adaptação separado por horta + PlantDoc no ciclo 5 |
| Classe de fundo e recortes de imagens de campo | +11% em avaliação entre domínios e até +33,8% em dados de campo novos (Aarizou; Merah; Mahiout, 2026a) | As fotos "sem folha" (solo, mão, parede) já previstas na coleta podem alimentar uma classe de fundo |
| Remover o fundo | 81,6% com imagens RGBA sem fundo (Ahmad; Gamal; Saraswat, 2023) | É o próprio teste de dependência do fundo planejado |
| Backbone de fundação congelado | 51,1% contra 27–30% do treino completo no PlantDoc (Nguyen et al., 2026) | Sonda linear sobre DINOv2 como quinta "arquitetura" |
| Poucos exemplos do domínio-alvo | A acurácia sobe conforme aumenta a semelhança entre domínio de origem e de destino (Yang et al., 2024) | Separar por horta, para que a semelhança entre adaptação e avaliação não infle o resultado |
| Recalibrar no domínio de uso | ECE de 0,287 para 0,038 (Ergün et al., 2026) | Reajustar a temperatura e o limiar τ no conjunto de adaptação |

## Referências desta parte (além das da Parte 1)

- AARIZOU, A.; MERAH, M.; MAHIOUT, D. A data-centric deep learning framework for Esca disease detection on visible grapevine leaf surfaces under cross-dataset field conditions. Engineering Applications of Artificial Intelligence, v. 176, 114771, 2026a. DOI: 10.1016/j.engappai.2026.114771.
- AARIZOU, A.; MERAH, M.; MAHIOUT, D. Enhancing cross-crop plant disease detection on complex field images through dataset merging, transfer learning and ensemble averaging of convolutional neural networks. Journal of Phytopathology, v. 174, n. 2, e70264, 2026b. DOI: 10.1111/jph.70264.
- ABADE, A.; FERREIRA, P. A.; VIDAL, F. B. Plant diseases recognition on images using convolutional neural networks: a systematic review. Computers and Electronics in Agriculture, v. 185, 106125, 2021. DOI: 10.1016/j.compag.2021.106125. (Revisão de 121 artigos, autores brasileiros.)
- BARBEDO, J. G. A. Factors influencing the use of deep learning for plant disease recognition. Biosystems Engineering, v. 172, p. 84-91, 2018b. DOI: 10.1016/j.biosystemseng.2018.05.013. (2018a é o que já está no artigo.)
- BARBEDO, J. G. A. Plant disease identification from individual lesions and spots using deep learning. Biosystems Engineering, v. 180, p. 96-107, 2019. DOI: 10.1016/j.biosystemseng.2019.02.002.
- CHAI, A. Y. H. et al. Beyond supervision: harnessing self-supervised learning in unseen plant disease recognition. Neurocomputing, v. 610, 128608, 2024. DOI: 10.1016/j.neucom.2024.128608.
- FERENTINOS, K. P. Deep learning models for plant disease detection and diagnosis. Computers and Electronics in Agriculture, v. 145, p. 311-318, 2018. DOI: 10.1016/j.compag.2018.01.009. (99,53% em 58 classes; o resumo não traz a queda laboratório → campo — conferir no texto antes de citar esse ponto.)
- SINGH, D. et al. PlantDoc: a dataset for visual plant disease detection. In: ACM IKDD CoDS and COMAD, 7., 2020. Proceedings [...]. p. 249-253. DOI: 10.1145/3371158.3371196.
- TODA, Y.; OKURA, F. How convolutional neural networks diagnose plant disease. Plant Phenomics, v. 2019, 9237136, 2019. DOI: 10.34133/2019/9237136.
- WU, Z. et al. Unseen-crop plant disease classification via disentangled representation learning. Electronics, v. 15, n. 8, 1553, 2026. DOI: 10.3390/electronics15081553.
- YANG, S. et al. From laboratory to field: cross-domain few-shot learning for crop disease identification in the field. Frontiers in Plant Science, v. 15, 1434222, 2024. DOI: 10.3389/fpls.2024.1434222.

---

# Parte 3 — Relevância, adoção e reprodutibilidade (13/10/2026)

Terceira rodada (buscas 9 a 11 e uma busca por título). As consultas, os trabalhos e o uso de cada um estão em `docs/GUIA_BANCA_TCC2.md`, Seções 6, 7 e 15. Os dois achados mais úteis: Tian et al. (2024), com laboratório 98,2% → campo 71,6% e variação mínima entre arquiteturas, e Gerz e Jelali (2026), que refazem um estudo com vazamento e veem 98,7% cair para 32,8%.


---

# Parte 4 — Leitura do texto completo (16/10/2026)

Nas Partes 1 a 3, os números vieram dos resumos. Aqui estão os artigos lidos **inteiros**:
sete, todos de acesso aberto, com o texto obtido nas páginas dos editores (Frontiers, PLOS,
Springer/BMC, ScienceDirect). Workneh e Tian ainda dependem do PDF (ver o fim desta parte).

## Ergün et al. (2026) — videira, *Front. Plant Sci.* 17

**O que fazem.** EfficientNet-B0 e MobileNetV3-Large, ajuste fino completo, AdamW, 5-fold
estratificado. Treino e validação em dois conjuntos públicos **já fotografados em campo**
com celular (Dataset1: 2.726 imagens; Dataset2: 3.103). Teste externo (Dataset3): 500 folhas
(300 sadias, 200 doentes) de **quatro vinhedos** na Turquia, fotografadas com iPhone, 30
videiras por vinhedo e 5 folhas por videira, rótulo dado no campo. Comparam inferência
crua, temperature scaling, MC Dropout e as duas juntas.

**Resultados.** Dataset1 (binário): 99,63% de acurácia, ECE 0,0055. Dataset3, mesmos
modelos (os do Dataset1): 72,4% cru e 73,8% com MC Dropout + temperatura (MobileNetV3:
70,6% e 71,8%); linha de base da classe majoritária, 60%. ECE cru 0,287 → 0,038 com
temperatura; T médio **8,34**.

**Como se compara ao FitoVision.**

- **Reforça a tese, e com mais força do que parecia.** O treino deles já é de campo, e
  mesmo assim cai 27 pontos ao trocar de vinhedo, país e celular. A queda não é só "fundo
  de estúdio × fundo de horta": é **troca de fonte**. Bate com a fonte retida do ciclo 5
  (cerca de 10 pontos abaixo, mesma espécie, outro lote) e com o espinafre do ciclo 3 (23
  pontos).
- **A temperatura vai no sentido oposto.** Lá T = 8,34 (o modelo era confiante demais);
  aqui T = 0,55–0,68 (o *label smoothing* deixa a rede pouco confiante nas fontes de treino
  e a temperatura sobe a confiança). Os dois chegam ao mesmo ponto: a calibração da
  validação não serve fora dela.
- **Três ressalvas ao método deles (citar com cuidado):**
  1. *A calibração externa achata a confiança.* Depois da temperatura, a confiança média no
     Dataset3 é **0,551** num problema binário — quase o acaso. O ECE de 0,038 é baixo
     porque todas as previsões ficam perto de 55%, não porque o modelo sabe quando erra. Não
     reportam acurácia seletiva nem cobertura. Além disso, um ECE de validação de 0,0055
     deveria dar T perto de 1; um T de 8,34 sugere que a temperatura do resultado externo
     não saiu só da validação, embora o texto diga que sim.
  2. *O intervalo de confiança é estreito demais.* 73,8% [73,1; 74,5] com N = 500 é ±0,7
     ponto; pela binomial seria cerca de ±3,9. O *bootstrap* parece ter sido sobre as
     médias dos cinco *folds*, não sobre as imagens. Citar 73,8%, não o intervalo.
  3. *Exclusão das folhas ambíguas.* O Dataset3 manteve só sintomas "claros e bem
     definidos"; o resultado externo tende a ser otimista.
- **Serve de modelo para a coleta de campo:** vários locais, amostragem fixa por planta e
  por posição, rótulo anotado no campo antes da inferência, linha de base da classe
  majoritária no relatório. A diferença que vale manter aqui: **não descartar as folhas
  ambíguas**, e sim marcá-las como duvidosas no `metadados_campo.csv`.
- Declaram uso de IA generativa na redação.

## Kabya et al. (2026) — espinafre-de-malabar, *PLOS ONE* 21

**O que fazem.** Pouco mais de **700 fotos originais** de *Basella alba* (3 classes),
juntando um acervo privado da Habiganj Agricultural University e um conjunto público; um
único split 70/15/15 (teste de cerca de 105 imagens); aumento de dados só no treino;
SimSiam + CBAM-ResNet-50, CNNs, ViT e SwinV2. O conjunto está no Hugging Face
(`saifullah03/malabar_spinach_leaf_disease_dataset`) e não é o lote Mendeley do FitoVision.

**Resultados.** SimSiam-CBAM 97,31%; SwinV2-Base pré-treinado 97,98% (98,99% com TTA); ViT
treinado do zero 90,7%; EfficientNet-B0 pré-treinado 98%. Temperatura ajustada na
validação: ECE de 3,3% a 6,4%.

**Como se compara.** É o exemplo do que o FitoVision critica: tudo intra-fonte, sem
verificação de quase-duplicatas entre as duas fontes que juntaram, sem teste externo, e com
um teste de cerca de 105 imagens (intervalo de ±3 pontos ou mais). Os 97–98% deles e os
75,4% do ciclo 3 no espinafre de outra fonte não se contradizem: medem coisas diferentes.

## Shafay et al. (2025) — revisão sistemática, *Plant Methods* 21

**O que fazem.** Revisão de mais de 150 estudos (RGB e hiperespectral) e um *benchmark* em
11 conjuntos, misturando resultados da literatura com uma análise exploratória própria.

**Resultados que interessam.** Laboratório 95–99% × campo 70–85%; queda de 15 a 30%;
ResNet-50 de 98,18% (PlantVillage) para 41,79% (PlantDoc); ConvNeXt de 99,47% para 71,76%.
Recomendam mirar 80–87% em campo, e não os 97–100% de laboratório.

**Correção ao que a Parte 1 dizia.** A Parte 1 registrou que esta revisão "diverge" do
empate ViT × EfficientNet. Lida inteira, a afirmação é outra:

- Os 88,07% do Swin contra 52,74% das CNNs são no **Cropped PlantDoc**, com treino e teste
  no mesmo conjunto de campo; não é transferência laboratório → campo, e os números vêm de
  estudos com protocolos diferentes.
- Sobre o ViT puro, a revisão diz que mecanismos de atenção pura **mostram maior
  sensibilidade ao ambiente, apesar da vantagem em laboratório**. Isso **converge** com o
  FitoVision (o ViT-B/16 é o pior nas culturas de lavoura nos dois ciclos). O que fica sem
  teste aqui é o Swin, uma arquitetura híbrida.

Redação sugerida para o artigo e a banca: "a revisão de Shafay et al. (2025) aponta
arquiteturas híbridas (Swin) como mais robustas e o ViT puro como mais sensível ao
ambiente; o resultado do ViT-B/16 aqui é coerente com a segunda parte; a primeira não foi
testada".

## Yang et al. (2024) — *few-shot* entre domínios, *Front. Plant Sci.* 15

**O que fazem.** Aprendizado com poucos exemplos (1 e 5 por classe) a partir de seis
domínios de origem, avaliado num conjunto de doenças fotografadas em campo (9 culturas, 43
doenças). Medem a semelhança entre domínios pela similaridade de cosseno das médias de
*features*.

**Resultados.** A acurácia sobe com a semelhança entre origem e destino. O melhor domínio de
origem é o de **pragas fotografadas em campo**, e não o PlantVillage: 80,13% (5 exemplos) e
63,95% (1 exemplo) com o CDFSL-BDC. Quando a distância é grande, treinar com vários domínios
transfere melhor que com um só.

**Como se compara.** Método diferente (*few-shot*), então não entra como comparação de
número. Serve de argumento: uma fonte com cena de campo vale mais que uma de estúdio, o que
apoia a hipótese do fundo e o conjunto de adaptação separado por horta.

## Nguyen et al. (2026) — PlantVillage → PlantDoc, *Smart Agric. Technol.* 15

(Lido inteiro no ScienceDirect, 16/10.)

**O que fazem.** Treinam só no PlantVillage, restrito às 27 classes que ele divide com o
PlantDoc, e testam no PlantDoc (236 imagens de campo) e no Plant Pathology 2021 (11.310
fotos de macieira em pomar). Comparam CNNs treinadas por inteiro (EfficientNet-B0,
MobileNetV3, MobileViT) com um DINOv2 ViT-B/14 **congelado** e um classificador linear.
Três a sete sementes por modelo, McNemar exato por imagem, *bootstrap* no conjunto de teste.

**Resultados.**

- PlantDoc: EfficientNet-B0 27,7%; DINOv2 congelado 51,1% [45,3; 58,1]. A vantagem não é
  resolução: a 224 px, 48,3% contra 27,7%.
- Ajuste fino completo do DINOv2: 99,8% no PlantVillage e **29,4%** no PlantDoc (congelado:
  98,7% e 50,6%). Quanto mais o modelo se adapta ao laboratório, pior ele transfere.
- **A explicação é a cultura, não a doença.** Decompõem a acurácia em "manter a folha na
  cultura certa" × "acertar a doença dentro da cultura". As CNNs mandam 66–79% das maçãs
  de pomar para outra cultura; o DINOv2 mantém 92%. Dentro da cultura certa, todos acertam
  a doença em 55–75% (o DINOv2 é o pior nessa parte).
- Remover fundo, aumento de dados pesado e adaptação no teste mudam poucos pontos (citam
  a literatura).
- Concluem que ~51% em campo **não basta para uso** e que um sistema real precisa de
  rejeição de imagens fora da distribuição.

**Como se compara ao FitoVision.**

- **É o mecanismo do achado A4 da auditoria visto por outro ângulo.** Lá, a CNN erra a
  cultura; aqui, a cultura decide o rótulo (alface → saudável, lavoura → doente). Nos dois
  casos a rede aprende "que planta é" em vez de "que sintoma tem". A amostragem por cultura
  do ciclo 5 ataca exatamente isso, e o ganho de especificidade na lavoura é coerente.
- **A recomendação final deles é o que o FitoVision já tem:** abstenção e o veredicto
  "não é folha". O que eles pedem como trabalho futuro é a medida que a coleta de campo vai
  produzir.
- **Experimento barato e direto para um ciclo 6:** DINOv2 ViT-B/14 congelado + regressão
  logística, sem ajuste fino. Cabe na RTX 3050 (só extração de *features*) e testaria a
  conclusão "a representação importa mais que a arquitetura" no próprio problema binário.
- Ressalvas: tarefa de 27 classes (não binária), PlantDoc com 236 imagens, uma fonte de
  laboratório. Declaram uso de ChatGPT só para revisão de texto.

## Gerz e Jelali (2026) — vazamento por aumento de dados, *Smart Agric. Technol.* 15

(Lido inteiro no ScienceDirect, 16/10.)

**O que fazem.** Reproduzem dois estudos de **detecção de objetos** (YOLOv8n) que aplicaram
aumento de dados antes de dividir o conjunto, e refazem com divisão primeiro.

**Resultados.**

- Tomate (PlantDoc, Joshi et al.): mAP50 de 98,70% (aumentar e depois dividir) para
  **30,64%** (dividir e depois aumentar). O resumo do próprio artigo diz 32,77%, e a Seção 4
  fala em "89,7% → 30,64%": **o artigo é inconsistente consigo mesmo**. Citar como "queda de
  cerca de 66 pontos de mAP50", e não como acurácia.
- Pragas de arroz (IP102): 99,5% → 89,1% depois de deduplicar (pHash + SSIM + união-busca +
  conferência manual: 81 de 451 imagens removidas, 18%) e dividir antes de aumentar.
- Separam dois tipos de vazamento: **induzido pelo processo** (aumentar antes de dividir) e
  **inerente aos dados** (o conjunto público já traz duplicatas ou versões aumentadas).
  Citam conjuntos publicados com a versão aumentada, que passa o vazamento para quem usa.

**Como se compara ao FitoVision.**

- **O A2 da auditoria é exatamente o vazamento "inerente aos dados" deles.** A fonte
  `shuvokumarbasak2030` é uma versão aumentada (17 transformações) publicada como conjunto
  próprio, e o hash não liga as variantes à foto original. O ciclo 5 corrigiu pelo id da
  foto original no nome do arquivo.
- O pipeline de deduplicação deles é o mesmo desenho do FitoVision (hash perceptual,
  união-busca), com duas diferenças: somam SSIM e conferem os grupos à mão. O FitoVision
  usa a órbita D4 (rotações e espelhos), que eles não usam.
- O caso do gado (Wang et al., citado por eles: 0,94–0,95 com divisão aleatória contra
  0,72–0,82 deixando o animal inteiro fora) é a mesma lógica da separação por horta.

## Abbasi, Martinez e Ahmad (2023) — folhosas em aquaponia, *Artif. Intell. Agric.* 10

(Lido inteiro no ScienceDirect, 16/10.)

**O que fazem.** 2.000 imagens de alface, manjericão, espinafre e salsa, parte de uma
instalação de aquaponia da Universidade de Alberta e parte **coletada do Google e do
Ecosia**; aumento de dados para 2.640; ResNet-50 em três fases (cultura → cultura×saúde →
detecção com YOLOv5s).

**Resultados.** 95,83% (cultura), 94,13% (cultura × saudável/doente, 8 classes), mAP50
82,13% (detecção). Os números de classificação são **da validação**; não há teste separado
para essas fases.

**Como se compara.** Citado no artigo como o único trabalho com alface e espinafre e saída
saudável × doente, e continua sendo. Mas: o conjunto final de 2.640 é contado depois do
aumento, e os conjuntos de validação vêm desse total, o que indica aumento antes da divisão
(o vazamento de Gerz e Jelali); imagens da web sem verificação de duplicatas; sem teste
externo. Na banca, vale dizer que os 94% são intra-fonte e possivelmente inflados.

## O que muda na leitura do FitoVision

1. A queda fora da fonte aparece **mesmo quando o treino é de campo** (Ergün). A explicação
   não é só o fundo de estúdio; é a fonte. Isso fortalece a fonte retida como medida.
2. **O atalho pela cultura tem mecanismo medido na literatura** (Nguyen): a CNN aprende a
   cultura antes do sintoma. É o A4 da auditoria e a razão da amostragem por cultura do
   ciclo 5.
3. A calibração na validação falha fora dela nos dois sentidos: confiante demais lá
   (T = 8,34), pouco confiante aqui (T < 1). Nenhum dos trabalhos lidos mostra a abstenção
   interceptando erros fora da fonte, e Nguyen pede rejeição como trabalho futuro; essa
   medida é contribuição do FitoVision, assim que houver fotos de campo.
4. O ViT puro converge com a literatura (Shafay); o Swin e um ViT auto-supervisionado
   congelado (Nguyen) ficam como trabalho futuro. Ajuste fino completo pode piorar a
   transferência, e o FitoVision usa ajuste fino completo em tudo.
5. O vazamento por proveniência (A2) tem nome na literatura: inerente aos dados (Gerz e
   Jelali).
6. Um ECE baixo pode esconder confiança achatada (Ergün, confiança média 0,551). Reportar
   ECE junto com acurácia seletiva e cobertura, como o FitoVision já faz.
7. Dos trabalhos com folhosas, nenhum tem teste fora da fonte (Kabya, Abbasi).

## Ainda falta ler

| Trabalho | Acesso | Situação |
|---|---|---|
| Workneh et al. (2026), *Comput. Electron. Agric.* | Pede acesso institucional | Precisa do login CAPES (CAFe) ou do PDF |
| Tian et al. (2024), *Pest Manag. Sci.* | Fechado | A Wiley pede verificação anti-robô; precisa do PDF |
