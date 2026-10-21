# FitoVision — Guia de preparação para a banca

### TCC-II · Engenharia de Software · Univille · Banca em 08/12/2026, 20 minutos de fala · Versão 4, de 16/10/2026 (com a leitura do texto completo da literatura e o artigo v8)

> **O que é este documento.** O guia de estudo e o roteiro da apresentação. Ele junta a
> história do projeto, os conceitos que você precisa dominar, a posição do trabalho na
> literatura, a relevância, as limitações, as perguntas prováveis da banca e um plano de
> treino. É um documento vivo: os trechos marcados com **[ATUALIZAR]** dependem da coleta
> de campo e das conversas com o Prof. Leanderson.
>
> **De onde vêm os números.** Do projeto: **`docs/FICHA_TECNICA_FITOVISION.md`**, que é a
> fonte única desde a auditoria de 13/10 (cada número conferido no split, nos logs, no
> código e nas previsões por imagem), mais `docs/HISTORICO_COMPLETO_TCC.md`,
> `docs/CORRECOES_METODOLOGICAS.md` e o artigo v8. Da literatura: `docs/REVISAO_LITERATURA_WOS.md`
> (Partes 1 e 2) e a Parte 3 deste guia. Os números dos artigos externos foram lidos nos
> **resumos** da Web of Science; antes da banca, leia o texto completo dos sete marcados
> com ★ na Seção 6.

---

## Sumário

1. [A tese em 30 segundos, 2 minutos e 5 minutos](#1-a-tese-em-30-segundos-2-minutos-e-5-minutos)
2. [A história em sete atos](#2-a-história-em-sete-atos)
3. [Cartão de números](#3-cartão-de-números)
4. [Conceitos que você precisa dominar](#4-conceitos-que-você-precisa-dominar)
5. [Por que não funciona bem no mundo real](#5-por-que-não-funciona-bem-no-mundo-real)
6. [Onde o trabalho está na literatura](#6-onde-o-trabalho-está-na-literatura)
7. [Relevância: acadêmica, profissional e social](#7-relevância-acadêmica-profissional-e-social)
8. [Limitações — diga antes que perguntem](#8-limitações--diga-antes-que-perguntem)
9. [Melhorias, trabalhos futuros e áreas de estudo](#9-melhorias-trabalhos-futuros-e-áreas-de-estudo)
10. [Roteiro da apresentação](#10-roteiro-da-apresentação)
11. [Perguntas prováveis da banca](#11-perguntas-prováveis-da-banca)
12. [Demonstração do sistema](#12-demonstração-do-sistema)
13. [Plano de treino até 08/12](#13-plano-de-treino-até-0812)
14. [O que atualizar neste guia](#14-o-que-atualizar-neste-guia)
15. [Referências novas (Parte 3 da revisão)](#15-referências-novas-parte-3-da-revisão)

---

## 1. A tese em 30 segundos, 2 minutos e 5 minutos

### 30 segundos

> "Treinei quatro redes neurais para dizer se uma folha de hortaliça está saudável ou
> não. No teste elas acertam 98,5%. Só que esse número é uma média: 99,7% em foto de
> estúdio e 86% em foto de lavoura, dentro do mesmo teste. Numa fonte que o modelo nunca
> viu, cai para 75%, e em horta os erros saem com 99% de confiança. O trabalho mostra,
> com teste estatístico, que o limite está nos dados e na forma de avaliar, não na
> arquitetura — e propõe um protocolo para medir isso direito."

### 2 minutos

1. **Problema.** Diagnóstico de doença em folha por foto é um dos usos mais estudados de
   visão computacional na agricultura. Os artigos reportam 99%. Em campo, a literatura
   registra 70% a 85%.
2. **O que fiz.** Um sistema de triagem (saudável × anômalo) para hortaliças folhosas, com
   três CNNs e um Vision Transformer, calibração de confiança, abstenção e uma guarda que
   rejeita fotos sem folha.
3. **O que descobri no caminho.** O meu próprio 99,01% do TCC-I era inválido: 57% do
   teste eram fotos quase idênticas às do treino. Corrigi, e o número honesto é 98,48%.
4. **O segundo achado.** Mesmo sem duplicatas, o teste se parece com o treino. Em uma
   fonte nunca vista, 75,4%. Em 13 fotos de horta, 83,8%, com intervalo de 58% a 96%.
5. **O terceiro achado.** Dentro do próprio teste, 81% dos erros estão nas culturas
   fotografadas em lavoura: 86,3% ali, contra 99,7% em estúdio, na mesma fonte.
6. **O quarto achado.** Trocar a CNN por um transformer 21 vezes maior dá exatamente o
   mesmo resultado. E a confiança calibrada não avisa quando o modelo erra fora do domínio.
7. **Conclusão.** O valor do trabalho é o protocolo: auditoria de vazamento (por
   semelhança e por proveniência), avaliação em níveis (por subconjunto, fonte retida,
   campo) e estatística adequada a amostras pequenas. **[ATUALIZAR com o resultado da
   coleta de campo.]**

### 5 minutos

É o resumo de 2 minutos mais três números de apoio em cada ponto. Use a Seção 2 como
esqueleto e o cartão da Seção 3 como cola.

### A frase que resume a postura

> "Este trabalho não entrega um modelo que funciona em campo. Entrega a medida honesta de
> quanto falta, a explicação de por que falta e o caminho para chegar lá."

Não peça desculpas pelo resultado. Um 99% inválido retratado, um 98,5% explicado e um
75% medido formam um trabalho mais forte do que um 99% sem auditoria — e a literatura
recente diz exatamente isso (Seção 7).

---

## 2. A história em sete atos

Conte como história. A banca lembra de narrativa, não de tabela.

### Ato 1 — O 99% (TCC-I)

Dataset de 210.832 imagens de fontes públicas (PlantVillage e derivados, mais datasets de
alface). Três CNNs com transfer learning. Melhor resultado: **EfficientNet-B0, 99,01%**.

Sintoma: em fotos de fora do dataset o diagnóstico era errático. A pergunta que destravou
tudo foi **"por que o número é bom demais?"**, e não "por que o modelo é ruim?".

### Ato 2 — O vazamento

O split tinha sido feito por arquivo. Vários datasets redistribuem as mesmas fotos do
PlantVillage com recorte, compressão ou rotação diferentes. Auditoria por *hash*
perceptual:

- 210.832 arquivos, mas só 106.403 fotografias distintas (49,5% de redundância);
- **57,35% do teste** já tinha sido visto no treino sob outro nome de arquivo.

Correção: agrupar quase-duplicatas e manter cada grupo inteiro em uma única partição.
Vazamento residual: 0,18% no ciclo 1 e 0,04% do ciclo 3 em diante. O 99,01% foi
**retratado**; o número sem vazamento ficou em 98,5%.

### Ato 3 — Os bugs que só o campo mostrou

- **A abstenção nunca disparava.** O limiar era escolhido a partir de 0,50, que é o piso
  matemático da confiança em classificação binária. Cobertura de 100%: o sistema nunca se
  abstinha. Correção: piso de 0,85 e meta de 98% de acurácia seletiva.
- **Alface saudável rotulada como doente.** A pasta `Lettuce_disease_datasets/` continha
  a subpasta `Healthy/`, mas a rotulagem automática parava no nome do contêiner ("disease")
  e marcava tudo como anômalo. **3.972 imagens** de alface saudável estavam com o rótulo
  trocado desde o primeiro ciclo, e 3.831 imagens de erva daninha estavam na classe
  anômala.
- **Recortes da mesma foto em treino e teste.** Um lote de míldio tinha 1.142 arquivos que
  eram recortes de apenas 49 fotografias. O *hash* não pega isso, porque dois recortes da
  mesma foto não são parecidos — o critério certo é a proveniência.

Nenhum desses defeitos aparecia nas métricas do teste. O de rotulagem só apareceu porque
13 fotos de horta se comportaram de forma estranha.

### Ato 4 — A lição de estatística

Em pontos estimados, o campo "piorou" a cada correção: 91,7% → 76,9% → 72,7% de acertos.
A tentação era reverter. O intervalo de Wilson mostrou que os três se sobrepõem entre
64,6% e 90,3%: **com 13 fotos, as versões são indistinguíveis.**

O ciclo 4 fechou o argumento: os mesmos modelos, comparados no espinafre (N = 484), dão
p de 10⁻¹²; nas 13 fotos de campo, p = 1,00. O modelo não mudou de qualidade entre as duas
medições. O que mudou foi o poder do instrumento.

### Ato 5 — O teste se parece com o treino (revisão de 08/10)

Três medidas, três significados:

| Medida | Acurácia balanceada | N | O que mede |
|---|---|---|---|
| **Intra-fonte** | 98,48% | 23.967 | Acerto em fotos das mesmas seis fontes do treino |
| **Fonte retida** (espinafre, ciclo 3) | 75,4% | 484 | Acerto em uma fonte nunca vista, ainda em estúdio |
| **Campo** (13 fotos, ciclo 4) | 83,8% [58–96] | 13 | Indício do uso real, sem poder estatístico |

As seis fontes aparecem nas mesmas proporções no treino e no teste. Tirar as duplicatas
tornou o 98,5% mais honesto, mas não o tornou uma medida de generalização. (Não diga mais
"70% do treino deriva do PlantVillage": a auditoria mostrou que é no máximo 47,7%.)

E a confiança não avisa: os 9 erros de campo dos quatro ciclos saíram com confiança entre
0,949 e 0,999.

### Ato 6 — Não é a arquitetura (ViT, 13/10)

ViT-B/16, 85,8 milhões de parâmetros no modelo treinado, feito no Colab. Resultado no
teste: **98,48%, idêntico ao EfficientNet-B0** (McNemar, p de Holm = 0,26). É 21 vezes
maior e 2,7 vezes mais lento. No espinafre, 84,4% contra 88,4% (p = 0,51). A abstenção
intercepta só 15% dos seus erros.

> "Um transformer com 21 vezes mais parâmetros chega exatamente ao mesmo lugar. A
> arquitetura não é o gargalo; os dados são."

### Ato 7 — A auditoria: os números contra os artefatos (13/10)

Antes da banca, cada número foi conferido no artefato de origem. Três coisas mudaram.

**O teste escondia o resultado que eu procurava em campo.** Reagrupando as previsões por
tipo de foto:

| Subconjunto do teste | N | EfficientNet-B0 |
|---|---|---|
| Culturas fotografadas em estúdio, na coletânea | 10.061 | 99,7% |
| Culturas fotografadas em lavoura, na mesma coletânea (mandioca, arroz, cana, café) | 4.530 | **86,3%** [84,5–88,0] |
| Só mandioca | 3.150 | 80,9% |

81% dos erros estão em 19% do teste. Nas culturas de lavoura, as quatro arquiteturas
ficam entre 83,4% e 87,8%, e o ViT é o pior.

**O detector de duplicatas não via as variantes de alface.** A maior fonte de alface são
17 transformações de cada uma de 1.231 fotos. Rotação livre, recorte e filtro de bordas
não ficam "parecidos" com a foto original para o hash. Resultado: 23% do teste são
variantes com outra variante no treino. É o mesmo problema do lote de míldio, que eu tinha
corrigido só para aquele lote.

**A composição do treino não era a que eu dizia.** No máximo 47,7% deriva do PlantVillage
(não 70%); 17,6% já é foto de lavoura, de outras culturas; a alface é 30% do treino, mas
vem de 1.231 fotos originais, 108 delas doentes, e 87% das imagens de alface são
saudáveis.

> "Quando reagrupei o teste por tipo de foto, o resultado que eu procurava em campo já
> estava lá: 99,7% em estúdio e 86% em lavoura, na mesma fonte. O 98,5% era a média."

### Ato 7b — Corrigir o atalho tem custo (ciclo 5, 14/10)

Tudo o que a auditoria apontou foi corrigido num split novo: alface separada por
fotografia original (uma foto por original no teste — 176 fotos, 15 doentes), 8.266
imagens de praga rotuladas, amostragem que equilibra os rótulos dentro de cada cultura e
um lote de espinafre fora do treino. A auditoria do split deu 1,7% (era 23%). As quatro
arquiteturas foram treinadas de novo no Colab.

| Lavoura (mesmas 4.530 imagens) | Ciclo 4 | Ciclo 5 |
|---|---|---|
| EfficientNet-B0: acur. bal. / espec. / sens. | 86,3 / 76,6 / 96,0 | **88,8 / 83,4** / 94,2 |
| MobileNetV2 | 86,6 / 78,3 / 94,9 | **90,1 / 90,2** / 90,0 |
| ResNet-50 | 87,8 / 81,4 / 94,2 | **89,6 / 89,0** / 90,2 |
| ViT-B/16 | 83,4 / 69,8 / 97,0 | 84,5 / 72,1 / 96,9 |

Os falsos alarmes caíram nas CNNs (especificidade +7 a +12); as doenças perdidas
subiram (sensibilidade −2 a −5). O ViT quase não mudou. No lote de espinafre que nenhum
modelo viu: 85,8% a 90,1%.

> "Corrigir o atalho não deu um modelo melhor em tudo: deu outro ponto de operação. Ele
> erra menos em folha sadia e mais em folha doente, e qual dos dois serve depende do
> custo de cada erro na horta. Por isso eu reporto especificidade e sensibilidade, e não
> só uma acurácia."

### Ato 8 — O campo **[ATUALIZAR depois da coleta]**

Espaço reservado: N por cultura, acurácia balanceada com intervalo, taxa de abstenção, ECE
em campo, resultado de H1, H2 e H3, e a comparação, nas mesmas fotos, entre o ciclo 4 e
o ciclo 5 — os dois congelados antes da coleta. Um ciclo 6 adaptado com fotos de horta só
se o orientador aprovar o conjunto de adaptação.

---

## 3. Cartão de números

Decore os da primeira tabela. Os da segunda, saiba onde achar.

### Para saber de cor

| Número | O que é |
|---|---|
| **99,01%** | Acurácia do TCC-I, retratada |
| **57,35%** | Fração do teste antigo contaminada por vazamento |
| **0,04%** | Vazamento residual no split atual |
| **98,48%** | Acurácia balanceada intra-fonte, EfficientNet-B0 e ViT-B/16 (IC 95%: 98,32–98,64) |
| **p = 0,26** | McNemar EfficientNet-B0 × ViT-B/16: empate |
| **75,4%** | Fonte nunca vista (espinafre, N = 484) — queda de 23 pontos |
| **88,4%** | Mesmo teste de espinafre depois de a fonte entrar no treino |
| **83,8% [58–96]** | Campo, 13 fotos, ciclo 4 (acurácia balanceada) |
| **11 de 13 = 84,6% [57,8–95,7]** | Mesmo campo, em proporção de acertos com Wilson |
| **31% e 15%** | Erros interceptados pela abstenção: EfficientNet-B0 e ViT |
| **0,949 a 0,999** | Confiança dos 9 erros de campo |
| **0,003 → 0,146** | ECE na validação e em campo |
| **86,3% × 99,7%** | Mesmo teste, mesma fonte: culturas fotografadas em lavoura (N = 4.530) × em estúdio (N = 10.061) |
| **81%** | Fração dos erros do teste que está nas culturas de lavoura (295 de 363) |
| **23%** | Fração do teste que são variantes de fotos de alface com outra variante no treino |
| **47,7%** | Máximo do treino que deriva do PlantVillage (não diga 70%) |
| **1.231** | Fotos originais de alface no treino; 108 doentes |
| **21 vezes** | ViT-B/16 ÷ EfficientNet-B0 em parâmetros, nos modelos treinados (não diga 16) |
| **3.972** | Alfaces saudáveis que estavam rotuladas como doentes |
| **1,7%** | Teste do ciclo 5 com quase-duplicata no treino (era 23%) |
| **86,3 → 88,8%** | Lavoura, EfficientNet-B0, ciclo 4 → ciclo 5 (especificidade 76,6 → 83,4; sensibilidade 96,0 → 94,2) |
| **+7 a +12 / −2 a −5** | Ciclo 5 nas três CNNs, na lavoura: pontos de especificidade ganhos / de sensibilidade perdidos |
| **85,8–90,1%** | Lote de espinafre que nenhum modelo do ciclo 5 viu |
| **176 (15 doentes)** | Fotos de alface no teste honesto do ciclo 5; os modelos acertam de 96,7% a 100% |
| **13** | Fotos de campo até agora **[ATUALIZAR]** |

### Para consultar

| Item | Valor |
|---|---|
| Dataset, etapa por etapa | 176.126 reunidas (staging) → 170.772 nas partições da base (saem 1 corrompida e 5.353 de grupos com rótulos contraditórios) → 175.123 no ciclo 4 (+4.351); 101.987 fotografias distintas pelo hash |
| Partições da base | 120.861 treino / 25.944 validação / 23.967 teste |
| Desbalanceamento | cerca de 1,35 anômala para 1 saudável |
| Parâmetros dos modelos treinados (saída de 2 classes) | MobileNetV2 2,2 M · EfficientNet-B0 4,0 M · ResNet-50 23,5 M · ViT-B/16 85,8 M. As arquiteturas originais, de 1.000 classes, têm 3,5 / 5,3 / 25,6 / 86,6 M |
| Latência (RTX 3050) | 5,3 · 7,2 · 8,6 · 19,4 ms |
| Falsos negativos / positivos | EfficientNet-B0 209 / 154 · ViT 137 / 204 · MobileNetV2 294 / 211 · ResNet-50 352 / 184 |
| Discordâncias EfficientNet × ViT | 158 imagens que só o primeiro acerta, 180 que só o segundo |
| Demais comparações | EfficientNet-B0 e ViT acima de MobileNetV2 e ResNet-50, p de Holm ≤ 1,4×10⁻¹² |
| Temperatura *T* | EfficientNet-B0 0,561 · MobileNetV2 0,546 · ResNet-50 0,566 · ViT 0,620 |
| ECE antes → depois (validação) | EfficientNet-B0 0,0509 → 0,0027 · ViT 0,0472 → 0,0022 |
| Limiar τ | 0,85 nos quatro; cobertura de 98,9% (EfficientNet-B0) e 99,5% (ViT) |
| Ganho do ciclo 4 no espinafre | 0,754 → 0,884 (p de Holm = 5,9×10⁻¹²); ResNet-50 0,776 → 0,862; MobileNetV2 0,743 → 0,874 |
| MobileNetV2 no teste antigo | caiu de 98,18% para 97,89% entre os ciclos 3 e 4 (p de Holm = 0,015) |
| Custo | 77 h de pipeline nos quatro ciclos (20,7 h de treino no ciclo 4); ViT 6,1 h em GPU L4 |
| Treino | Lote 64; AdamW; taxa de 10⁻³ com cosseno; até 30 épocas (a parada antecipada não disparou nos ciclos 3 e 4) |
| Composição do treino-base | Coletânea 60,2% · variantes de alface 25,0% · PlantVillage 9,9% · outras de alface 4,9%. Alface: 29,9% do treino, 86,6% saudável. Foto de lavoura (mandioca, arroz, cana, café): 17,6% |
| Por subconjunto, quatro arquiteturas | Lavoura: 86,3 / 83,4 (ViT) / 86,6 / 87,8%. Só mandioca: 80,9 / 75,7 / 81,3 / 82,8% |
| Erros de campo | 9 erros em 4 fotos: `healthy/alface_04` (4), `anomalous/alface_01` (3), `alface_02` (1), `alface_05` (1) |
| ECE de campo, ciclo 4 | 0,146 com a temperatura ajustada na validação; 0,094 sem ela |
| Campo por ciclo (acertos / diagnósticos) | 11/12 · 10/13 · 8/11 · 11/13 · ciclo 5 11/12 · ViT 12/12 · ViT do ciclo 5 12/13 |
| Ciclo 5, dataset | 152.057 imagens: 112.458 / 20.612 / 18.413 + 574 retidas; alface em 1.172 unidades de foto original |
| Ciclo 5, validação (F1 macro) | EfficientNet-B0 0,972 · ViT 0,972 · ResNet-50 0,961 · MobileNetV2 0,960 |
| Ciclo 5, teste inteiro | 97,4 · 96,9 · 96,9 · 96,9 (EfficientNet-B0, ViT, ResNet-50, MobileNetV2); não comparável ao 98,48 — o teste é outro |
| Ciclo 5, McNemar por imagem na lavoura | Ciclo 4 acerta mais: EfficientNet-B0 126 × 94 (p = 0,04); MobileNetV2 218 × 94 e ResNet-50 189 × 74 (p < 10⁻¹¹); ViT 92 × 99 (p = 0,66) |
| Ciclo 5, custo | Colab L4, 06:06–17:52 de 14/10, cerca de 22 unidades |

### Inconsistências resolvidas em 13/10

1. **Tamanho do dataset.** São três etapas, não três versões: 176.126 reunidas, 170.772
   nas partições da base, 175.123 com o ciclo 4. No artigo, a Tabela 1 é medida nas
   23.967 imagens de teste da base.
2. **Curso.** Engenharia de Software. Corrigido no histórico.
3. **Parâmetros.** Os da Tabela 1 passaram a ser os dos modelos treinados.

### O que ainda depende de você

1. **Procedência das 13 fotos de campo.** Nenhuma tem EXIF de celular; uma é de câmera
   Sony, de 2014, editada no Photoshop; uma tem faixas pretas de quadro de vídeo. A
   monografia diz "capturadas com telefone celular". Diga de onde vieram. Se foram
   reunidas de mensagens, vídeos ou da internet, o texto precisa dizer, e a banca pode
   perguntar. A auditoria confirmou que nenhuma é duplicata de imagem do treino.
2. **O rótulo de `healthy/alface_04.jpg`.** A foto mostra uma folha com manchas escuras,
   e os quatro ciclos a dão como anômala. É responsável por 4 dos 9 "erros de campo".

---

## 4. Conceitos que você precisa dominar

Para cada conceito: o que é, por que você usou, como dizer em uma frase e o que a banca
pode perguntar. Estude nesta ordem.

### 4.1 Transfer learning e ajuste fino

- **O que é.** Começar de uma rede já treinada no ImageNet (1,2 milhão de imagens, 1.000
  classes) e continuar o treino no seu problema. As primeiras camadas já sabem detectar
  bordas, texturas e formas; você reaproveita isso.
- **O que você fez.** Ajuste fino **completo**: todas as camadas são atualizadas, por até
  30 épocas, AdamW, taxa de aprendizado de 10⁻³, escalonador de cosseno, parada antecipada
  com paciência 10, *label smoothing* de 0,1 e amostragem ponderada por classe.
- **Uma frase.** "Em vez de ensinar a rede a enxergar do zero, eu ensino uma rede que já
  enxerga a olhar para folhas."
- **Pergunta provável.** "Por que ajuste fino completo e não congelar o backbone?" —
  Resposta honesta: era a prática usual e maximiza o acerto no domínio de treino. Um
  estudo de 2026 mostra que, para generalizar, congelar um backbone auto-supervisionado
  transfere melhor (51% contra 27–30% no PlantDoc; Nguyen et al., 2026): ajustado por
  inteiro, o mesmo modelo cai de 50,6% para 29,4%. É o primeiro experimento da lista de
  trabalhos futuros, e cabe na RTX 3050 (só extração de *features*).

### 4.2 As quatro arquiteturas

| Rede | Ideia central | Por que está no trabalho |
|---|---|---|
| **ResNet-50** | Conexões residuais: cada bloco aprende uma correção sobre a entrada, o que permite redes profundas | Referência clássica, a mais pesada das CNNs |
| **MobileNetV2** | Convoluções separáveis em profundidade e blocos residuais invertidos | Pensada para celular; a mais rápida |
| **EfficientNet-B0** | Escalonamento composto: profundidade, largura e resolução crescem juntas | Melhor relação acurácia/custo; é a que está em produção |
| **ViT-B/16** | Corta a imagem em blocos de 16×16 pixels e aplica autoatenção, sem convolução | Arquitetura de contraste: testa se o limite é a família de modelos |

- **Parâmetros.** Os modelos treinados têm a camada final trocada por uma de duas saídas:
  2,2 M (MobileNetV2), 4,0 M (EfficientNet-B0), 23,5 M (ResNet-50) e 85,8 M (ViT-B/16). Os
  números que todo mundo conhece (3,5 / 5,3 / 25,6 / 86,6 M) são os das arquiteturas de
  1.000 classes. Se a banca estranhar o "4,0 M" da Tabela 1, é essa a explicação.
- **Viés indutivo.** A convolução assume que pixels vizinhos se relacionam e que um padrão
  vale em qualquer posição. O ViT não assume isso: aprende dos dados, e por isso precisa
  de mais dados ou de pré-treino forte.
- **Pergunta provável.** "A comparação é justa se o ViT usou outros hiperparâmetros?" —
  Dados, perda, seleção de modelo e calibração são os mesmos. Mudam taxa de aprendizado
  (5×10⁻⁵), decaimento de pesos (0,05), duas épocas de aquecimento e limitação da norma do
  gradiente, que é a receita usual de transformers. Comparo cada arquitetura com uma
  receita adequada a ela. A ressalva está escrita no artigo: um treino e uma semente por
  modelo, então a conclusão vale para este teste, não para as arquiteturas em geral.

### 4.3 Vazamento de dados e o *hash* perceptual

- **O que é vazamento.** Informação do teste que chega ao treino. Em imagens, a forma mais
  comum é a quase-duplicata: a mesma foto com outro nome, recorte ou compressão.
- **Hash perceptual.** Um resumo curto da aparência da imagem. Imagens parecidas geram
  resumos parecidos; a diferença entre dois resumos é medida em bits (distância de
  Hamming). Você considera duplicata quando a distância é ≤ 4.
- **Grupo diedral D4.** As oito combinações de rotação de 90° e espelhamento. O *hash* é
  comparado nas oito orientações, para que uma foto girada não escape.
- **União-busca (*union-find*).** Estrutura que junta em um mesmo grupo todas as imagens
  ligadas por semelhança, mesmo indiretamente (A parece B, B parece C → A, B e C no mesmo
  grupo).
- **LSH.** Técnica para não comparar cada imagem com todas as outras, o que seria inviável
  com 210 mil.
- **Split agrupado.** Cada grupo vai inteiro para treino, validação ou teste.
- **Uma frase.** "Dividi por fotografia, não por arquivo."
- **Os dois tipos de vazamento.** Por **semelhança** (quase-duplicatas, pegas pelo *hash*)
  e por **proveniência** (variantes da mesma foto que o *hash* não vê como parecidas). A
  literatura separa o vazamento **induzido pelo processo** (aumentar os dados antes de
  dividir) do **inerente aos dados** (o conjunto público já traz duplicatas ou a versão
  aumentada) (Gerz; Jelali, 2026). A fonte `shuvokumarbasak2030` é o segundo caso: uma
  versão aumentada publicada como conjunto próprio.
- **Onde a proveniência escapou.** No lote de míldio (1.142 recortes de 49 fotos),
  corrigido no ciclo 4. E na maior fonte de alface, descoberta só na auditoria de 13/10:
  17 transformações de cada uma de 1.231 fotos, com o id da foto no nome do arquivo.
  99,6% das imagens de teste dessa fonte têm outra variante da mesma foto no treino, o
  que dá 23% do teste. **O 0,04% mede semelhança; não mede isso.**
- **Como dizer.** "Tirei as duplicatas que se parecem. Descobri depois que uma fonte era
  feita de rotações, recortes e filtros das mesmas fotos, e que o meu detector não as
  ligava. Medi o tamanho do problema e refiz a comparação sem essa fonte: a conclusão
  sobre as arquiteturas se mantém."
- **Pergunta provável.** "Por que distância 4?" — É um limiar conservador, validado com
  teste sintético; e o vazamento residual é reportado como métrica (0,04%), em vez de
  assumir que é zero.

### 4.4 Acurácia balanceada

- **O que é.** A média do acerto em cada classe: ½ · (sensibilidade + especificidade).
- **Por que.** Com 1,35 anômala para cada saudável, um classificador que sempre responde
  "anômalo" teria 58,66% de acurácia comum e 50% de acurácia balanceada. A balanceada não
  premia quem chuta a classe maior.
- **Falso negativo é o erro caro.** Planta doente declarada saudável: o produtor não age.
  Por isso você reporta os falsos negativos à parte.

### 4.5 Bootstrap e intervalo de confiança

- **O que é.** Sortear, com reposição, um novo "teste" do mesmo tamanho a partir do teste
  original, recalcular a métrica, repetir 10.000 vezes e olhar onde caem 95% dos valores.
- **Estratificado.** O sorteio mantém a proporção das classes.
- **Uma frase.** "O intervalo diz quanto a métrica mudaria se o teste fosse outro sorteio
  da mesma população."
- **Cuidado ao falar.** O intervalo de 98,32% a 98,64% é estreito porque N é grande. Ele
  mede a incerteza **dentro** da distribuição do teste. Não diz nada sobre outra
  distribuição — esse é o ponto do Ato 5.

### 4.6 Teste de McNemar

- **O problema.** Comparar dois modelos avaliados nas **mesmas** imagens. As previsões são
  pareadas; um teste de proporções independentes estaria errado.
- **A ideia.** Só importam as imagens em que os modelos discordam. Se os dois são
  equivalentes, cada discordância é uma moeda honesta: tanto pode favorecer um quanto o
  outro.
- **Com os seus números.** EfficientNet-B0 × ViT: 158 imagens que só o primeiro acerta,
  180 que só o segundo. Em 338 discordâncias, 158 contra 180 é compatível com moeda
  honesta (p = 0,26). Empate.
- **Forma exata.** Usa a distribuição binomial em vez da aproximação qui-quadrado; é a
  indicada para amostras pequenas, como o campo.
- **Referência.** É o teste recomendado para comparar classificadores em um mesmo conjunto
  de teste (Dietterich, 1998).

### 4.7 Correção de Holm

- **O problema.** Com seis comparações entre quatro modelos, a chance de algum p < 0,05
  aparecer por acaso cresce.
- **A ideia.** Ordenar os valores de p do menor para o maior e exigir mais do menor: o
  primeiro é multiplicado por 6, o segundo por 5, e assim por diante.
- **Por que Holm e não Bonferroni.** Controla o mesmo erro e é menos conservador.
- **Detalhe que pode cair.** Quando o ViT entrou, as comparações passaram de três para
  seis e os valores de p entre as CNNs mudaram um pouco. Nenhuma conclusão mudou.

### 4.8 Calibração, ECE e *temperature scaling*

- **Calibração.** Um modelo é calibrado se, entre as previsões feitas com 90% de
  confiança, 90% estão certas.
- **ECE.** Agrupa as previsões em 15 faixas de confiança e mede, em cada uma, a distância
  entre a confiança média e o acerto real; depois faz a média ponderada. Zero é perfeito.
- **Temperature scaling.** Divide os *logits* por um escalar *T* antes do softmax. Não
  muda a classe prevista, só a confiança. *T* é ajustado na validação.
- **O detalhe que quase ninguém percebe e que você deve saber.** Os seus quatro modelos
  têm ***T* menor que 1** (0,55 a 0,62). *T* < 1 **aumenta** a confiança. Isso acontece
  porque o *label smoothing* de 0,1 deixa a rede pouco confiante no domínio de treino, e a
  calibração corrige "para cima". Consequência: fora do domínio, onde a rede erra mais, os
  *logits* também são ampliados em cerca de 1,8 vez, e os erros saem muito confiantes.
  **A auditoria mediu o efeito:** no ciclo 4, o ECE de campo é 0,146 com a temperatura e
  0,094 sem ela; nos ciclos 1 e 3, o mesmo sentido. A temperatura ajustada nas fontes de
  treino piora a calibração em campo. Não explica sozinha os erros confiantes: sem ela
  eles sairiam com 0,93 a 0,96, ainda acima de τ. Com 13 fotos é indício, não medida.
- **O que a literatura diz.** A calibração pós-treino falha sob mudança de distribuição
  (Ovadia et al., 2019). Em um conjunto externo de folhas de videira, o ECE sem calibração
  chega a 0,287 (Ergün et al., 2026). Lá a temperatura é 8,34 (rede confiante demais);
  aqui é 0,55–0,68 (pouco confiante, efeito do *label smoothing*). Sentidos opostos, mesma
  conclusão. Cuidado ao citar o 0,038 depois da calibração deles: a confiança média cai
  para 0,55 num problema binário, ou seja, o ECE baixo vem de achatar a confiança, e eles
  não reportam acurácia seletiva.
- **Uma frase.** "A calibração vale para a distribuição em que foi ajustada."

### 4.9 Abstenção e classificação seletiva

- **O que é.** Abaixo de um limiar τ de confiança, o sistema responde "inconclusivo".
- **Medidas.** Cobertura (fração que recebe diagnóstico), acurácia seletiva (acerto entre
  os diagnosticados) e erros interceptados (erros que ficaram abaixo de τ).
- **Curva risco–cobertura.** Mostra o que se ganha em acerto ao aceitar diagnosticar menos.
- **O seu resultado.** τ = 0,85 intercepta 113 dos 363 erros do EfficientNet-B0 (31,1%),
  abstendo-se em 1,2% das imagens; no ViT, 51 de 341 (15,0%). **Mais de dois terços dos
  erros saem com confiança alta, mesmo dentro do domínio de treino.**
- **A frase que mostra maturidade.** "O ViT tem o menor ECE e a menor fração de erros
  interceptados. ECE baixo em média não garante que os erros estejam nas confianças
  baixas."

### 4.10 Intervalo de Wilson

- **O problema.** Com 13 fotos, a aproximação normal para proporções dá intervalos errados
  (às vezes passando de 100%).
- **O que é.** Um intervalo para proporções que funciona com N pequeno.
- **Com os seus números.** 11 acertos em 13: 84,6%, intervalo de 57,8% a 95,7%. São quase
  40 pontos de largura.
- **Planejamento de amostra.** Para uma proporção de 80%: mais de 40 pontos de largura com
  13 fotos, 15,5 com 100 e 9,0 com 300. É daí que sai a meta de 100 a 300 fotografias.

### 4.11 Pseudo-replicação

- **O que é.** Tratar como independentes observações que vêm da mesma origem.
- **Onde apareceu.** 173 recortes de 5 fotos de míldio entraram no McNemar como 173
  observações. O p de 10⁻¹³ estava inflado. Refeito só com o espinafre (484 fotos
  independentes), o ganho continua significativo, com p mais modesto.
- **Por que importa para o campo.** Várias fotos da mesma folha, do mesmo canteiro ou da
  mesma horta não são independentes. Por isso a separação por horta.

### 4.12 Domain gap e *shortcut learning*

- **Domain gap.** A diferença de desempenho entre a distribuição de treino e a de uso.
- **Shortcut learning.** A rede aprende qualquer regularidade que reduza a perda, mesmo
  que não tenha relação com o problema — o fundo, a iluminação, o estilo da câmera. Se o
  teste tem a mesma regularidade, o atalho passa despercebido (Geirhos et al., 2020).
- **Evidência forte da literatura.** É possível obter 49% de acurácia no PlantVillage
  usando apenas 8 pixels do fundo, sem ver a folha (Noyan, 2022).
- **O atalho da cultura.** No treino, 87% das imagens de alface são saudáveis e 75% das
  demais são anômalas; arroz, laranja e abóbora só têm o rótulo anômalo, e soja, mirtilo e
  framboesa só o saudável. "É alface, então é saudável" acerta 78,5% do treino sem olhar
  sintoma. É coerente com o erro mais comum em campo (alface doente dada como saudável) e
  com o que a literatura descreve como sintoma acoplado à aparência da cultura (Wu et al.,
  2026). É hipótese: não foi testada.
- **O que é seu e o que é hipótese.** Você **mediu** dependência da fonte (espinafre) e a
  distância entre foto de lavoura e foto de estúdio dentro do teste (86,3% contra 99,7%).
  A dependência do fundo e o atalho da cultura são **hipóteses**.

### 4.13 Guarda de vegetação (ExG)

- **O que é.** O índice *Excess Green*, ExG = 2g − r − b, calculado sobre os canais de cor
  normalizados. Se menos de 10% dos pixels têm vegetação, o sistema responde "não é folha".
- **Por que.** Uma rede binária sempre responde uma das duas classes. Sem a guarda, uma
  foto de parede recebe diagnóstico.
- **Limitação.** É um filtro de cor: uma parede verde passa.

### 4.14 Grad-CAM

- **O que é.** Um mapa de calor que mostra quais regiões da imagem mais pesaram para a
  classe prevista, a partir dos gradientes na última camada de características.
- **Estado no projeto.** A primeira versão estava na camada errada no EfficientNet e com a
  sobreposição ilegível. Não apresente mapas antigos. **[ATUALIZAR quando os mapas dos
  erros de campo novos estiverem prontos.]**
- **Para que serve na defesa.** Mostrar para onde o modelo olha quando erra em campo. Se
  olhar para a terra e não para a folha, é evidência direta de atalho.

### 4.15 Os testes das hipóteses de campo

| Hipótese | Pergunta | Teste | Por quê |
|---|---|---|---|
| **H1** | O desempenho difere entre culturas? | Fisher generalizado (Freeman-Halton) | Tabela com mais de duas categorias e contagens pequenas |
| **H2** | Erra mais em sintomas estruturais (murcha, furos) do que em cromáticos (manchas, clorose)? | Fisher exato, tabela 2×2 | Duas categorias, contagens pequenas |
| **H3** | O desempenho cai com a distância de captura? | Freeman-Halton e tendência de Cochran-Armitage | As distâncias são ordenadas; o teste de tendência usa essa ordem |
| Tamanho de efeito | De quanto é a diferença? | Intervalo de Newcombe para diferença de proporções | Um p sozinho não diz o tamanho |
| Confiança de acertos × erros | A confiança separa os dois? | Mann-Whitney | Compara duas distribuições sem supor normalidade |

Diga sempre que essas análises são **exploratórias**: os subgrupos serão pequenos.

---

## 5. Por que não funciona bem no mundo real

Resumo do diagnóstico, já com a auditoria de 13/10 (detalhe na ficha técnica e na Parte 2
de `REVISAO_LITERATURA_WOS.md`).

**O treino não é o problema.** Quatro arquiteturas de 2,2 a 85,8 milhões de parâmetros
ficam entre 97,8% e 98,5% no teste e entre 83% e 88% nas fotos de lavoura do mesmo teste.
O que falha é a relação entre os dados de treino e o uso real. E o campo com hortaliça
folhosa ainda não está medido: 13 fotos não permitem dizer que funciona nem que não
funciona.

| # | Causa | Sua evidência | Literatura |
|---|---|---|---|
| 1 | **Foto de lavoura é mais difícil, mesmo quando está no treino** | No próprio teste: 86,3% nas culturas fotografadas em lavoura contra 99,7% em estúdio, na mesma fonte; 81% dos erros | Laboratório 98,2% → misto 91,8% → campo 71,6%, com variação mínima entre arquiteturas (Tian et al., 2024); 99% → 26–30% no PlantDoc (Nguyen et al., 2026) |
| 2 | **Quase nada do treino é foto de lavoura de hortaliça folhosa** | 17,6% do treino é lavoura, mas de mandioca, arroz, cana e café; a alface é close-up da folha | A falta de bases que representem as condições da prática é o problema principal (Barbedo, 2019) |
| 3 | **Poucas fotos da cultura-alvo, e quase todas saudáveis** | Alface: 1.231 fotos originais, 108 doentes, repetidas em cópias e transformações; 87% das imagens de alface são saudáveis | O sintoma fica acoplado à aparência da cultura (Wu et al., 2026) |
| 4 | **O teste se parece com o treino** | Seis fontes nas mesmas proporções; 23% do teste são variantes de fotos que estão no treino | Só a separação por fonte revela a perda: F1 de 0,98 para 0,56 (Workneh et al., 2026) |
| 5 | **Espécies-*proxy*** | Espinafre nunca visto: 75,4%, mesmo em estúdio | Culturas ausentes do treino derrubam o desempenho (Chai et al., 2024) |
| 6 | **Sintomas estruturais fora do treino** | 7.779 imagens de dano por praga não entraram, por falha da rotulagem; erros de campo em murcha e perfurações | CNNs capturam cor e textura das lesões (Toda; Okura, 2019) |
| 7 | **Calibração ajustada só nas fontes de treino** | Erros de campo com 0,949 a 0,999; a temperatura piora o ECE de campo (0,146 contra 0,094) | Calibração pós-treino falha sob mudança de distribuição (Ovadia et al., 2019) |
| 8 | **Defeitos de protocolo** | Vazamento de 57% (corrigido); 3.972 rótulos trocados (corrigido); vazamento por proveniência (medido, a corrigir no ciclo 5) | mAP50 de 98,7% para cerca de 31% ao refazer um estudo de detecção com a divisão antes do aumento (Gerz; Jelali, 2026) |

**O argumento mais forte.** O contraste aparece **dentro do mesmo teste e da mesma
fonte**: 99,7% em estúdio e 86,3% em lavoura, com N de milhares. Não depende das 13 fotos.

**O segundo argumento.** A queda no espinafre aconteceu **sem mudar o tipo de fundo**.
Vinte e três pontos se perdem só por trocar de fonte e de espécie. Quem disser "é só o
fundo" está simplificando.

**O que não é a causa.** Arquitetura, capacidade e quantidade de imagens. Nas fotos de
lavoura do teste, as quatro arquiteturas ficam entre 83,4% e 87,8%, e o ViT é o pior. Um
estudo independente com cinco famílias de redes encontrou a mesma coisa e concluiu que
enriquecer os dados importa mais do que trocar o modelo (Tian et al., 2024).

**O que ainda não foi testado.** A dependência do fundo; o atalho da cultura; a variância
entre treinos (uma semente por modelo); se o ajuste fino completo prejudicou a
transferência; e quanto dos 86% em lavoura é condição de captura e quanto é ruído de
rótulo das fontes.

**O que não dizer mais.** "70% do treino é PlantVillage" (é no máximo 47,7%); "o treino é
todo de estúdio" (17,6% é lavoura); "o split não tem vazamento" (não tem quase-duplicata;
tem variantes da mesma foto em 23% do teste).

---

## 6. Onde o trabalho está na literatura

### Os trabalhos que você precisa conhecer (★ = ler o texto completo)

| Trabalho | O que diz | Como usar na banca |
|---|---|---|
| ★ **Workneh et al. (2026)** — amêndoa | Validação agrupada quase não muda o F1; fontes disjuntas derrubam para 0,56 | "É o estudo mais parecido com o meu, publicado este ano em uma revista de primeira linha, com a mesma conclusão" |
| ★ **Tian et al. (2024)** — maçã, batata, tomate | Laboratório 98,2%, misto 91,8%, campo 71,6%; variação mínima entre arquiteturas | "A queda e a irrelevância da arquitetura já foram medidas de forma independente" |
| ★ **Ergün et al. (2026)** — videira (lido inteiro) | Treino já de campo e, mesmo assim, 99,6% → 72–74% em outro vinhedo; ECE 0,287 → 0,038, mas com confiança média de 0,55; T = 8,34 | "Mesma arquitetura, queda da mesma ordem — e a queda vem da troca de fonte, não só do fundo" |
| ★ **Nguyen et al. (2026)** (lido inteiro) | 99% → 26–30%; as CNNs mandam a folha para a cultura errada (só 21–34% ficam na cultura certa, contra 92% com DINOv2 congelado); ajuste fino completo piora | "É o mecanismo do meu atalho cultura → rótulo, e justifica o próximo experimento" |
| ★ **Gerz e Jelali (2026)** (lido inteiro) | Detecção (YOLOv8n): mAP50 98,7% → 30,6% ao dividir antes de aumentar; dois tipos de vazamento | "O que aconteceu com o meu 99,01% é um problema reconhecido da área". Citar como mAP50, não acurácia; o próprio artigo dá 32,77% no resumo e 30,64% na tabela |
| ★ **Kapoor e Narayanan (2023)** | Vazamento encontrado em 17 áreas, 294 artigos; propõem fichas de verificação | "Vazamento é uma crise de reprodutibilidade, não um descuido isolado" |
| ★ **Ahmad, Gamal e Saraswat (2023)** — milho | Misturar campo e laboratório leva a 77,5–80,3% | "É a medida que a literatura aponta e que o conjunto de adaptação implementa" |
| Shafay et al. (2025) — revisão (lida inteira) | Laboratório 95–99%, campo 70–85%; híbridos (Swin) mais robustos, **ViT puro mais sensível ao ambiente** | Faixa de referência; **concorda** com o seu ViT, o pior na lavoura |
| Ramcharan et al. (2019) — mandioca | App em campo na Tanzânia: F1 cai 32% | "Mesmo o grupo do PlantVillage mediu queda quando foi a campo" |
| Mohanty, Hughes e Salathé (2016) | 99,35% no PlantVillage | A origem do otimismo da área |
| Barbedo (2018, 2019) | A falta de diversidade dos dados é o problema principal | Autor brasileiro, da Embrapa; cite com segurança |
| Noyan (2022) | 49% no PlantVillage com 8 pixels de fundo | Evidência de atalho no dataset-base |
| Kabya et al. (2026) — espinafre-de-malabar (lido inteiro) | 97,3% num teste de ~105 imagens, de ~700 fotos originais, sem teste externo | Mesma cultura em que você mediu 75,4% sem tê-la visto: protocolos diferentes |
| Abbasi, Martinez e Ahmad (2023) — folhosas (lido inteiro) | 94,1% (cultura × saúde) **na validação**; parte das imagens do Google; aumento de dados aparentemente antes da divisão | Único com alface e espinafre; resultado intra-fonte e possivelmente inflado |
| Ovadia et al. (2019) | Calibração pós-treino falha sob mudança de distribuição | Fundamenta o achado da confiança em campo |
| Recht et al. (2019) | Queda de 11–14% no ImageNet com um teste novo feito pelo mesmo processo | "Acontece até no ImageNet" |
| Guo et al. (2017) | *Temperature scaling* | Base da calibração |
| Geirhos et al. (2020) | *Shortcut learning* | Base conceitual |

### Confirma, contradiz, lacuna

- **Confirma.** O desempenho intra-fonte não prevê o desempenho fora da fonte. A
  arquitetura pesa pouco. A calibração precisa ser verificada no domínio de uso.
- **Converge (correção de 16/10, depois da leitura completa).** Pelos resumos, parecia
  que a revisão de Shafay et al. (2025) contradizia o seu ViT. Lida inteira, ela diz que
  arquiteturas **híbridas** (Swin) são mais robustas e que o **ViT puro é mais sensível ao
  ambiente**, apesar da vantagem em laboratório. O seu ViT-B/16 empata no teste e é o pior
  nas culturas de lavoura: concorda com a segunda parte. Os "88% contra 53%" são do Swin
  no Cropped PlantDoc, com treino e teste no mesmo conjunto de campo, e reúnem estudos com
  protocolos diferentes. O que fica sem teste aqui é o Swin.
- **Tensiona.** A sua calibração vai no sentido oposto à de Ergün et al. (2026): lá a
  temperatura reduz a confiança (T = 8,34), aqui aumenta (T < 1). Os dois mostram que a
  calibração da validação não serve fora dela.
- **Lacuna.** Nas buscas feitas, nenhum estudo reúne, para hortaliças folhosas, auditoria
  de vazamento, calibração com abstenção e avaliação fora da fonte. Diga "não foi
  localizado", nunca "não existe".

### Como descrever a revisão, se perguntarem

"Busquei na Web of Science, Coleção Principal, em outubro de 2026, com doze consultas em
três rodadas: generalização laboratório–campo, vazamento, calibração, hortaliças folhosas,
ViT contra CNN, viés de fundo, culturas não vistas, e relevância e adoção. Foram cerca de
450 registros, triados por título e resumo. As consultas estão registradas."

Não chame de revisão sistemática: não houve protocolo registrado nem dupla triagem.

---

## 7. Relevância: acadêmica, profissional e social

A banca vai perguntar "e daí?". Tenha uma resposta de três partes.

### 7.1 Acadêmica

1. **Um caso documentado de vazamento, com retratação.** Vazamento já foi encontrado em
   17 áreas e 294 artigos, com conclusões "amplamente superotimistas" (Kapoor; Narayanan,
   2023). Em visão computacional agrícola, refazer um estudo com o protocolo correto
   derrubou o mAP50 de 98,7% para cerca de 31% (Gerz; Jelali, 2026). O seu trabalho é um
   caso concreto, com a prova nos arquivos, do mesmo fenômeno — e com os dois caminhos de
   vazamento que essa literatura descreve.
2. **Um protocolo de avaliação em três níveis.** Intra-fonte, fonte retida e campo. A
   fonte retida é uma forma barata de medir generalização sem ir a campo, e pode ser
   adotada por qualquer TCC que use datasets públicos.
3. **Evidência para uma cultura pouco estudada.** Hortaliças folhosas têm poucos
   trabalhos, quase todos avaliados em partição interna.
4. **Um resultado negativo útil.** O empate entre ViT e EfficientNet-B0, e a constatação
   de que ECE baixo não implica boa interceptação de erros.
5. **Uma medida do *domain gap* sem sair do teste.** Separar o teste por tipo de foto
   (lavoura × estúdio) custa minutos e mostra 13 pontos de diferença. É replicável por
   qualquer trabalho que use coletâneas públicas.
6. **Material didático.** Autores da área defendem ensinar pelos erros (*pitfall-based
   learning*; Gerz; Jelali, 2026). O registro dos 35 erros e ajustes do projeto serve
   exatamente a isso.

### 7.2 Profissional (Engenharia de Software e ML em produção)

1. **Qualidade de dados é engenharia.** Os defeitos mais graves — vazamento, rotulagem
   trocada, recortes e variantes da mesma foto, pastas inteiras não rotuladas — estavam
   nos dados, não no modelo. Nenhum apareceu nas métricas. O que os tornou visíveis foi instrumentação: mapa de rótulos
   auditável, relatório de vazamento como critério de aceitação, previsões por imagem
   salvas.
2. **Reprodutibilidade.** Pipeline por script, semente fixa, registro de parâmetros,
   pesos de cada ciclo preservados, treino no Colab automatizado por linha de comando com
   verificação de integridade.
3. **Segurança do sistema em uso.** Quatro saídas (saudável, anômalo, inconclusivo, não é
   folha) em vez de duas. Em produto real, saber quando não responder vale tanto quanto
   acertar.
4. **Decisão baseada em evidência.** Reverter o ciclo 2 (causa concreta: rótulos errados)
   e não reverter o ciclo 3 (só um número dentro do ruído) é o tipo de decisão que um
   time de ML toma toda semana.
5. **Custo.** O modelo em produção tem 4,0 milhões de parâmetros e 7,2 ms de latência; o
   que tem 21 vezes mais parâmetros não acerta mais.

### 7.3 Social e para o setor

1. **O tamanho do problema.** Patógenos e pragas causam perdas estimadas de 17% a 30% nas
   cinco principais culturas alimentares do mundo (Savary et al., 2019). Atenção: esse
   dado é de trigo, arroz, milho, batata e soja, não de hortaliças — use como contexto.
2. **Quem é o público.** 77% dos estabelecimentos rurais brasileiros são de agricultura
   familiar (IBGE, 2019). Em um levantamento no Rio Grande do Sul, as barreiras econômicas
   foram as mais severas para adotar tecnologias digitais, com destaque para a falta de
   soluções acessíveis (da Silveira et al., 2026). Um modelo leve, que roda em celular
   comum, responde a essa barreira.
3. **Há demanda real.** Um aplicativo de diagnóstico de mandioca teve 45% de adoção entre
   152 agricultores e técnicos na Costa do Marfim, usado sobretudo para acompanhar o
   estado da lavoura (Adjéi et al., 2024).
4. **Por que a honestidade importa aqui.** Um aplicativo que diz "saudável" com 99% de
   confiança para uma planta doente causa prejuízo. O achado de que a confiança não avisa
   em campo é diretamente relevante para quem pretende distribuir esse tipo de ferramenta.

### 7.4 O que você **não** pode afirmar

- Que o sistema ajuda produtores. Não houve estudo com usuários.
- Que funciona em campo. **[ATUALIZAR depois da coleta.]**
- Que o protocolo é novo. Workneh et al. (2026) fazem algo muito parecido. A novidade está
  na aplicação a hortaliças folhosas e na documentação completa do processo.

### 7.5 Vale colocar isso no artigo?

Sim, com parcimônia. O artigo está no limite de 11 páginas e as Seções 4 e 5 serão
reescritas com os dados de campo. A sugestão é entrar, nessa reescrita, com **duas
frases na Conclusão**: uma de contribuição (o protocolo em três níveis e o caso
documentado de vazamento) e uma de trabalhos futuros (backbone congelado, adaptação com
fotos de campo, recalibração). O desenvolvimento completo cabe na monografia e na
apresentação.

---

## 8. Limitações — diga antes que perguntem

Apresente as limitações você mesmo, em um slide. Banca que encontra uma limitação
escondida desconta; banca que a ouve do candidato pergunta sobre trabalhos futuros.

1. **Campo com N = 13**, de uma só cultura, com procedência não registrada.
   **[ATUALIZAR.]**
2. **Teste intra-fonte, com vazamento por proveniência.** O 98,48% não mede
   generalização, e 23% do teste são variantes de fotos de alface que estão no treino.
   Sem essa fonte, o teste dá 97,8%.
3. **Um treino e uma semente por modelo.** Não há variância entre execuções.
4. **Espécies-*proxy*.** Só a alface tem dados próprios (1.231 fotos originais, 108
   doentes); o espinafre é *Basella alba*, não
   *Spinacia oleracea*; rúcula, acelga e couve não têm dataset público.
5. **Classificação binária.** O sistema não diz qual é a doença.
6. **Rótulos de fontes públicas**, sem validação por fitopatologista. As fotos de campo
   são rotuladas por você.
7. **Hipótese do fundo não testada.**
8. **Calibração só verificada intra-fonte.**
9. **Sem estudo com usuários.**
10. **Hiperparâmetros próprios para o ViT.**
11. **ExG é um filtro de cor**, não um detector de folha.
12. **Rotulagem por nome de pasta.** Quatro pastas de dano por praga ficaram fora do
    treino; cultura e rótulo ficaram associados.
13. **"Lavoura" e "estúdio" foram separados por inspeção visual** de amostras de cada
    cultura, e o efeito mistura condição de captura com ruído de rótulo.

---

## 9. Melhorias, trabalhos futuros e áreas de estudo

### Curto prazo — cabe antes da banca

| Ação | Custo | O que responde |
|---|---|---|
| Coleta de campo separada por horta (avaliação e adaptação), com os arquivos originais e os metadados | Dias | Q3; é o que falta |
| Refazer o split por proveniência e sem as variantes que não são fotografias | Horas | Tira o vazamento que o hash não vê |
| Rotular as pastas de dano por praga; equilibrar a amostragem por cultura e rótulo | Horas + retreino | Sintomas estruturais; atalho da cultura |
| Avaliar o ciclo 4 congelado, às cegas, em todas as fotos novas | Horas | Tamanho do *domain gap* |
| ECE, cobertura e erros interceptados em campo | Horas | Q2 no domínio de uso |
| ECE de campo com *T* = 1 e com *T* ajustado | Minutos | Se a calibração está inflando a confiança fora do domínio |
| Recalibrar *T* e τ no conjunto de adaptação | Minutos | Se a abstenção volta a proteger |
| Teste de fundo: mesma imagem com fundo removido e com folha removida | Horas | Hipótese do fundo |
| Grad-CAM dos erros de campo, com a camada corrigida | Horas | Para onde o modelo olha quando erra |

> **14/10/2026:** o ciclo 5 com as correções de split e de amostragem já foi treinado
> (quatro arquiteturas; CORRECOES §15.10). As ações abaixo seriam um ciclo 6.

### Médio prazo — ciclo 6, se houver tempo e aval do orientador

| Ação | Base na literatura |
|---|---|
| Treinar com conjunto de adaptação + PlantDoc | Ahmad, Gamal e Saraswat (2023); Singh et al. (2020) |
| Classe ou exemplos de fundo, usando as fotos "sem folha" | Aarizou, Merah e Mahiout (2026) |
| Sonda linear sobre backbone auto-supervisionado congelado (DINOv2) | Nguyen et al. (2026) |
| Três a cinco sementes por modelo | Dá variância e responde a crítica de "um treino só" |

### Longo prazo — trabalhos futuros para a Conclusão

1. **Dataset público de hortaliças folhosas brasileiras em campo**, com procedência por
   horta. É a lacuna de dados mais clara que o trabalho identifica.
2. **Multiclasse**: identificar a doença, não só a presença de anomalia.
3. **Localização do sintoma** (detecção ou segmentação), que amplia a variabilidade dos
   dados sem novas fotos (Barbedo, 2019).
4. **Generalização de domínio e adaptação com poucas imagens** (Yang et al., 2024; Zhan et
   al., 2025).
5. **Incerteza além do softmax**: *ensembles* e métodos que se saem melhor sob mudança de
   distribuição (Ovadia et al., 2019).
6. **Inferência no dispositivo**, com quantização, para uso sem internet.
7. **Estudo com produtores e técnicos**: usabilidade, confiança e efeito na decisão.
8. **Validação dos rótulos por especialista.**

### Áreas de estudo, se quiser seguir na pesquisa

Mudança de distribuição e robustez · calibração e quantificação de incerteza ·
aprendizado centrado em dados · modelos de fundação e aprendizado auto-supervisionado ·
avaliação e reprodutibilidade em aprendizado de máquina · MLOps · interação
humano–IA em apoio à decisão.

---

## 10. Roteiro da apresentação

> **Confirme com o orientador** o tempo de fala e o de arguição. O roteiro abaixo é para
> **20 minutos**; os slides marcados com ◇ saem na versão de 15. Regra prática: um slide
> por minuto e meio, uma ideia por slide.
>
> O deck do Seminário 2 (nove slides: capa, problema, sistema, objetivo, erros, domínio,
> resultados, conclusão, próximos) é a base. O roteiro abaixo o amplia.
>
> Com o slide 10b (ciclo 5), a soma completa passa de 20 minutos; sem os dois ◇ (7 e 12)
> fica em 20:00 exatos. Se o orientador preferir manter o slide 7, junte o 10 e o 10b.

| # | Slide | Tempo | Mensagem única |
|---|---|---|---|
| 1 | Capa | 0:30 | Título, você, orientador |
| 2 | O problema | 1:30 | 99% no papel, 70–85% no campo |
| 3 | Questão e objetivos | 1:30 | Quanto vale em uso real? Q1, Q2, Q3 |
| 4 | O sistema | 1:30 | Quatro saídas, não duas |
| 5 | Dados e método | 1:30 | 175 mil imagens, quatro arquiteturas, estatística pareada |
| 6 | O 99,01% e o vazamento | 2:00 | 57% do teste já tinha sido visto |
| 7 ◇ | Os bugs que só o campo mostrou | 1:30 | Abstenção desligada; 3.972 rótulos trocados |
| 8 | Q1 — arquiteturas | 1:30 | EfficientNet-B0 e ViT empatam; arquitetura não é o gargalo |
| 9 | Q2 — calibração e abstenção | 1:30 | Dois terços dos erros saem confiantes |
| 10 | O teste por dentro e por fora | 2:00 | 99,7% em estúdio · 86% em lavoura · 75% em fonte nova |
| 10b | Corrigir o atalho (ciclo 5) | 1:30 | Menos falsos alarmes, mais doenças perdidas |
| 11 | Q3 — campo | 2:00 | **[ATUALIZAR]** |
| 12 ◇ | O que a literatura diz | 1:00 | O mesmo padrão, em outras culturas |
| 13 | Limitações | 1:00 | Ditas por você |
| 14 | Conclusão e contribuições | 1:30 | O limite está nos dados; o protocolo é a entrega |
| 15 | Trabalhos futuros | 0:30 | Três itens |

### Fala, slide a slide

**1. Capa.**
"Boa tarde. Sou José Pedro, e este é o meu trabalho de conclusão, orientado pelo professor
Leanderson André: detecção automática de anomalias fitossanitárias em hortaliças folhosas."

**2. O problema.**
"Diagnosticar doença de planta por foto é uma das aplicações mais estudadas de visão
computacional na agricultura. O artigo que abriu a área, em 2016, reportou mais de 99%.
Só que essas fotos são de estúdio: folha isolada, fundo liso, luz controlada. Quando os
mesmos modelos vão para a lavoura, a literatura registra entre 70% e 85%. E isso importa
porque 77% dos estabelecimentos rurais do Brasil são de agricultura familiar, onde o
diagnóstico ainda é feito no olho."
*Transição:* "A minha pergunta parte daí."

**3. Questão e objetivos.**
"Em que medida um sistema desse tipo, treinado com imagens públicas, mantém desempenho
confiável em fotos reais de campo? Dividi em três perguntas. Q1: qual arquitetura é
melhor, com significância estatística. Q2: a confiança do modelo avisa quando ele erra.
Q3: de quanto é a queda em campo e do que ela depende."

**4. O sistema.**
"O sistema recebe uma foto e devolve uma de quatro respostas: saudável, anômalo,
inconclusivo ou não é folha. As duas últimas existem porque uma rede binária sempre
responde alguma coisa, mesmo para a foto de uma parede."
*Se houver demonstração, é aqui (Seção 12).*

**5. Dados e método.**
"São 175 mil imagens de fontes públicas, quatro arquiteturas — três redes convolucionais e
um Vision Transformer — e, para comparar modelos, teste de McNemar com correção para
comparações múltiplas. Todas as métricas têm intervalo de confiança."

**6. O 99,01% e o vazamento.**
"No TCC-I cheguei a 99,01%. Em fotos novas, o sistema errava de forma errática. A pergunta
que mudou o trabalho foi: por que o número é bom demais? A resposta: o conjunto de dados
tinha 210 mil arquivos, mas só 106 mil fotografias distintas. Cinquenta e sete por cento
do teste eram fotos que o modelo já tinha visto no treino, com outro nome. Refiz a divisão
agrupando as fotos quase idênticas, e o vazamento caiu para 0,04%. O 99% foi retratado."
*Pausa aqui. É o momento mais forte da apresentação.*

**7. ◇ Os bugs que só o campo mostrou.**
"Com o teste limpo, apareceram dois defeitos que as métricas não mostravam. O limiar de
abstenção nunca disparava. E quase quatro mil fotos de alface saudável estavam rotuladas
como doentes, por causa do nome de uma pasta. Esse segundo só apareceu porque treze fotos
de horta se comportaram de forma estranha."

**8. Q1 — arquiteturas.**
"No teste, EfficientNet-B0 e ViT empatam em 98,48%, e os dois superam MobileNetV2 e
ResNet-50. O ViT tem vinte e uma vezes mais parâmetros e é quase três vezes mais lento.
Chega exatamente ao mesmo lugar. A arquitetura não é o gargalo."

**9. Q2 — calibração e abstenção.**
"A calibração funciona: o erro de calibração cai de 0,05 para 0,003. Mas o limiar de
abstenção intercepta só 31% dos erros do EfficientNet e 15% dos erros do ViT. Mais de dois
terços dos erros saem com confiança alta."

**10. O teste por dentro e por fora.**
"E aqui está o resultado central. O 98,5% é uma média. Quando separo o teste pelo tipo de
foto, dentro da mesma fonte, as culturas fotografadas em estúdio dão 99,7% e as
fotografadas em lavoura dão 86%. Oitenta e um por cento dos erros estão ali. E quando
testei em uma fonte que o modelo nunca tinha visto — espinafre, também em estúdio — a
acurácia foi de 75%. Um teste sem duplicatas continua sendo um teste da mesma
distribuição."
*Gráfico: três barras com intervalo de confiança — estúdio, lavoura, fonte nova.*

**10b. Corrigir o atalho (ciclo 5).**
"Uma auditoria mostrou por que a lavoura errava: a cultura estava associada ao rótulo, e
variantes da mesma foto de alface estavam dos dois lados do split. Corrigi as duas coisas
e treinei as quatro arquiteturas de novo. Na lavoura, as redes convolucionais passaram a
errar bem menos em folha sadia — de sete a doze pontos de especificidade — e um pouco mais
em folha doente. Não é um modelo melhor em tudo: é outro ponto de operação. E numa fonte
de espinafre que nenhum modelo viu, todos ficam entre 86% e 90%."
*Gráfico: para cada CNN, duas barras (ciclo 4, ciclo 5) de especificidade e de
sensibilidade na lavoura.*

**11. Q3 — campo. [ATUALIZAR]**
*Versão atual, com 13 fotos:* "Em treze fotos de horta, 83,8%, com intervalo de 58% a 96%.
Com essa amostra, as versões do modelo são estatisticamente indistinguíveis. E os erros de
campo saíram com confiança entre 95% e 99%."
*Versão final:* N por cultura, intervalo, abstenção, ECE em campo, H1 a H3.

**12. ◇ O que a literatura diz.**
"Esse padrão não é só meu. Em amêndoa, a acurácia cai de 0,98 para 0,56 quando as fontes
são separadas. Em videira, a mesma EfficientNet faz 73,8% em um conjunto externo. E um
estudo com cinco famílias de redes mediu 98% em laboratório, 92% em dados mistos e 72% em
campo, com diferença mínima entre arquiteturas. Os meus 99,7 e 86 contam a mesma história."

**13. Limitações.**
"Quatro limitações principais. O campo ainda é pequeno. Há um treino por modelo, então
não tenho variância entre execuções. Só a alface tem dados próprios, e são poucas fotos
originais — no teste honesto do ciclo 5 são 176 fotos, só 15 doentes. E a correção do
ciclo 5 troca sensibilidade por especificidade: qual ponto de operação servir depende do
campo."

**14. Conclusão e contribuições.**
"Respondendo à pergunta: o sistema é confiável nas fontes em que foi treinado e não
mantém esse desempenho fora delas. O limite está nos dados, não na arquitetura: corrigir
os dados mudou o comportamento das redes, e trocar de arquitetura não mudou. O que o
trabalho entrega é o protocolo: auditoria de vazamento, três níveis de avaliação e
estatística adequada a amostras pequenas."

**15. Trabalhos futuros.**
"Os próximos passos são treinar com fotos de fundo real, testar um modelo de fundação
congelado e recalibrar a confiança no domínio de uso. Obrigado."

### Regras de palco

- **Não leia o slide.** Pouco texto, um número grande por slide.
- **Um gráfico por resultado.** Barras com intervalo de confiança para Q1; três barras
  para o slide 10.
- **Diga os intervalos em voz alta.** "83,8%, com intervalo de 58 a 96" soa mais
  competente do que "83,8%".
- **Não use "laboratório"** para o 98,5%. Use "intra-fonte" ou "nas fontes de treino".
- **Pare no slide 10.** É o que a banca vai lembrar.
- **Tenha slides de apoio** depois do último: tabela completa das quatro arquiteturas,
  tabela de campo por ciclo, os 35 erros e ajustes, diagrama do *pipeline*, curva
  risco–cobertura, diagrama de fiabilidade.

---

## 11. Perguntas prováveis da banca

Treine as respostas em voz alta. Cada uma cabe em 30 a 45 segundos.

### Sobre o resultado

**1. "Então o seu sistema não funciona?"**
Funciona nas fontes em que foi treinado e ainda não foi medido com amostra suficiente em
campo. O que o trabalho mostra é que o número que a literatura costuma reportar não mede
uso real — e mede o quanto falta. **[ATUALIZAR com o resultado de campo.]**

**2. "Qual é a contribuição, se o resultado é negativo?"**
Três: um caso documentado de vazamento com retratação; um protocolo de avaliação em três
níveis; e a evidência, com teste pareado, de que a arquitetura não é o gargalo.

**3. "Por que não treinou melhor?"**
Quatro arquiteturas chegam ao mesmo teto e um transformer vinte e uma vezes maior não
melhora. Um estudo independente achou a mesma coisa (Tian et al., 2024). Mais treino no
mesmo dado não resolve; o que falta é dado de outra distribuição.

**4. "99,01% para 98,48%: meio ponto. O vazamento importava tanto assim?"**
A queda é pequena porque as duplicatas eram fáceis e as fotos restantes são da mesma
distribuição. O vazamento importa por outro motivo: ele invalidava o número. E a medida
de generalização veio de outro lado — 86% nas fotos de lavoura do próprio teste e 75% em
uma fonte nunca vista.

**5. "Como você sabe que não há mais vazamento?"**
Sei que não é zero, porque medi. Por semelhança, o vazamento residual é de 0,04%. Por
proveniência, uma auditoria mostrou que uma fonte de alface é feita de rotações, recortes
e filtros das mesmas fotos, e que 23% do teste têm outra variante no treino; o *hash* não
liga uma rotação de ângulo livre à foto original. Refiz a comparação sem essa fonte: o
teste dá 97,8% e a conclusão sobre as arquiteturas se mantém. A correção definitiva é
agrupar por foto de origem — e foi o que fiz no ciclo 5: alface separada por fotografia
original, uma foto por original no teste. A auditoria desse split deu 1,7% do teste com
alguma imagem parecida no treino, contra 23% antes, e boa parte desses pares são colisões
entre culturas diferentes, não a mesma foto. Nas 176 fotos de alface honestas, os quatro
modelos acertam de 96,7% a 100%; com só 15 doentes, o intervalo é largo, e por isso a
resposta sobre a alface vem do campo.

**5b. "E o ciclo 5 ficou melhor?"** (14/10/2026; detalhe em CORRECOES §15.10)
Melhor onde o erro estava, com um custo. Nas culturas fotografadas em lavoura, que tinham
81% dos erros, as CNNs ganham de 2 a 3,5 pontos de acurácia balanceada e de 7 a 12 de
especificidade: o atalho "foto de lavoura = doente" diminuiu. Em troca, a sensibilidade
cai de 2 a 5 pontos e, contado imagem a imagem, o ciclo 4 acerta mais, porque as doentes
são maioria. Não escondo isso: digo qual erro cada modelo comete e deixo claro que a
escolha depende do custo de cada erro na horta. O ViT quase não muda. E a fonte de
espinafre que nenhum modelo viu dá 86–90%, cerca de 10 pontos abaixo das fontes vistas —
é o tamanho do problema de domínio dentro da mesma cultura.

### Sobre o método

**6. "Por que classificação binária?"**
É triagem: dizer ao produtor se precisa olhar com atenção. E os datasets públicos não têm
as mesmas doenças para as cinco culturas, o que inviabilizava classes consistentes.
Multiclasse está nos trabalhos futuros.

**7. "Por que usar PlantVillage se ele não tem as suas culturas?"**
Porque não existe dataset público de doenças para rúcula, acelga ou couve. Usei como
*proxy* visual e declaro isso como limitação. O resultado do espinafre mostra o custo
dessa escolha. E vale precisar: o PlantVillage e as culturas dele são no máximo 48% do
treino; há 30% de alface e 18% de fotos de lavoura de outras culturas.

**8. "Por que McNemar e não um teste t?"**
Os modelos são avaliados nas mesmas imagens, então as previsões são pareadas e binárias. O
McNemar olha só as discordâncias. É o recomendado para esse caso (Dietterich, 1998).

**9. "Por que o intervalo de Wilson?"**
Com 13 observações, a aproximação normal dá intervalos errados. O de Wilson mantém a
cobertura com amostra pequena.

**10. "Por que 0,85 de limiar?"**
É o menor valor, a partir de 0,85, que garante 98% de acurácia seletiva na validação. O
piso existe porque, em classificação binária, a confiança nunca é menor que 0,5, e sem
piso o sistema nunca se abstinha. Reconheço que o limiar foi ajustado na validação e não
protege fora dela.

**11. "A comparação com o ViT é justa?"**
Mesmos dados, mesma perda, mesmo critério de seleção e mesma calibração. Mudam os
hiperparâmetros de otimização, que são os usuais para transformers. A ressalva está no
texto: um treino por modelo.

**12. "Uma semente só. E se o resultado for sorte?"**
É uma limitação real. O intervalo por *bootstrap* cobre a incerteza do teste, não a do
treino. O que me dá alguma segurança é a consistência: os quatro ciclos de treino do
EfficientNet-B0, em dados ligeiramente diferentes, ficaram entre 98,42% e 98,59%.

**13. "Temperatura de 0,56. O que isso significa?"**
Que a calibração **aumenta** a confiança. O *label smoothing* deixa a rede pouco confiante
no domínio de treino e a temperatura corrige para cima. Fora do domínio, isso pode estar
inflando a confiança dos erros. É uma hipótese que identifiquei e que se testa
recalculando o ECE de campo com temperatura igual a 1.

**14. "Como garante que os rótulos estão certos?"**
Não garanto por completo. Os rótulos vêm das fontes públicas, e já encontrei quase quatro
mil trocados por um defeito meu de rotulagem automática. Hoje há um mapa auditável de
pasta para rótulo. Validação por fitopatologista está nos trabalhos futuros.

### Sobre o campo

**15. "Por que só 13 fotos?"** **[ATUALIZAR]**
Era a amostra do TCC-I, e ela cumpriu um papel: revelou o defeito de rotulagem. Mas não
tem poder para comparar modelos, e o trabalho demonstra isso com o intervalo de Wilson.

**16. "Por que a acurácia de campo caiu quando você corrigiu os bugs?"**
Não caiu de forma detectável. Os intervalos dos ciclos se sobrepõem entre 64,6% e 90,3%.
Com 13 fotos, uma foto a mais ou a menos muda o ponto estimado em quase oito pontos.

**17. "A culpa é do fundo?"**
É uma das hipóteses, e eu não a testei diretamente. O que medi: o espinafre tem fundo
limpo e mesmo assim caiu 23 pontos, então o fundo não explica tudo; e as fotos de lavoura
que já estão no treino dão 86%, então ter fundo real no treino ajuda mas não resolve. Há
outras diferenças: a alface do treino é close-up da folha e quase toda saudável.

**18. "Se usar fotos de campo no treino, não está trapaceando?"**
Estaria, se fossem as mesmas da avaliação. Por isso a divisão é por horta: um conjunto que
nunca treina e outro de adaptação. E a ordem é fixa: primeiro o modelo congelado é medido
às cegas em tudo; só depois há adaptação.

### Sobre a literatura

**19. "A literatura diz que transformers generalizam melhor. O seu não. Por quê?"**
A literatura não diz isso do ViT puro. A revisão de Shafay et al. (2025) aponta as
arquiteturas híbridas, como o Swin, como mais robustas, e o ViT puro como mais sensível ao
ambiente — que é o que eu vi: o meu ViT empata no teste e é o pior nas fotos de lavoura.
Além disso, ele recebeu ajuste fino completo, que prejudica a transferência (Nguyen et al.,
2026), e é um treino só. Não afirmo que transformers não ajudam; afirmo que, neste teste,
este não ajudou, e o Swin fica como trabalho futuro.

**20. "O que o seu trabalho tem de novo em relação a Workneh et al.?"**
O protocolo é parecido, e reconheço isso. A diferença é a cultura (hortaliças folhosas,
com espécies-*proxy*), a comparação com um transformer, o registro completo dos erros do
processo e, com a coleta, a avaliação em campo real.

**21. "Como fez a revisão?"**
Web of Science, Coleção Principal, doze consultas registradas, triagem por título e
resumo. Não é uma revisão sistemática.

### Sobre relevância e engenharia

**22. "Qual a relevância para a Engenharia de Software?"**
Os defeitos mais graves estavam nos dados e nenhum aparecia nas métricas. O trabalho é
sobre garantir a qualidade de um sistema de aprendizado de máquina: auditoria,
rastreabilidade, reprodutibilidade e comportamento seguro quando o modelo não sabe.

**23. "Um produtor poderia usar isso hoje?"**
Como triagem com supervisão, com a ressalva explícita de que a confiança mostrada não é
confiável fora do domínio de treino. Como diagnóstico autônomo, não.

**24. "O que faria diferente se começasse hoje?"**
Três coisas: separaria uma fonte inteira para teste desde o primeiro dia; coletaria fotos
de campo antes de treinar qualquer coisa; e registraria a procedência de cada imagem.

**25. "Qual foi o maior aprendizado?"**
Que uma métrica boa demais é uma hipótese, não um resultado.

### Perguntas difíceis

**26. "Por que o MobileNetV2 piorou no ciclo 4?"**
Caiu de 98,18% para 97,89% no teste antigo, com significância. Os outros dois não
mudaram. Não investiguei a causa; a hipótese é que, sendo o menor modelo, teve menos
capacidade para acomodar as culturas novas. Não afeta a produção, que usa o
EfficientNet-B0.

**27. "O ViT acertou 12 de 12 em campo. Não é o melhor?"**
Com intervalo de 75,8% a 100%, que se sobrepõe ao de todos os outros. Doze fotos não
distinguem arquiteturas.

**28. "O ganho no espinafre prova que o modelo generaliza?"**
Não. Prova que, quando a fonte entra no treino, o teste dessa fonte sobe. É um ganho
intra-fonte.

**29. "Por que não usou validação cruzada?"**
Com 175 mil imagens e 20 horas por ciclo de treino, o custo era proibitivo; e o problema
principal não era a variância da partição, era a distribuição do teste. A fonte retida
responde melhor a isso do que mais dobras da mesma fonte.

**30. "Esse 75,4% é de uma versão antiga. Qual é o número da versão atual?"**
A versão atual viu o espinafre no treino e faz 88,4%, mas esse já é um número intra-fonte.
O 75,4% é o único com fonte genuinamente retida, e por isso é o que cito. Uma nova fonte
retida para o ciclo 4 seria o conjunto de campo.

**31. "Os 98,48% então estão errados?"**
Não estão errados; estão incompletos. São a média de 99,7% em foto de estúdio, 97,7% nas
variantes de alface e 86% em foto de lavoura. O número que descreve o uso real está mais
perto do último.

**32. "Por que 81% na mandioca, se ela está no treino?"**
Porque é foto de lavoura: planta no solo, fundo natural, sintoma pequeno no quadro. A
mandioca tem as duas classes, então a rede não pode decidir pela cultura. A especificidade
ali é de 67%. Parte pode ser ruído de rótulo da própria fonte; não separei os dois efeitos.

**33. "Por que tantas culturas se o trabalho é sobre hortaliças folhosas?"**
Por falta de dados públicos das culturas-alvo. A consequência é que cultura e rótulo
ficaram associados no treino: 87% da alface é saudável, e várias culturas têm um só
rótulo. É um atalho que a rede pode ter aprendido, e é uma das correções do próximo ciclo.

**34. "Você mesmo encontrou esses erros. Não é um trabalho com muitos erros?"**
É um trabalho com muitos erros *encontrados e medidos*. A literatura mostra que eles são
comuns e raramente reportados: vazamento em 294 artigos de 17 áreas (Kapoor; Narayanan,
2023). A diferença está em ter os artefatos para conferir e em reportar o que foi achado.

**35. "De onde vieram as 13 fotos de campo?"** **[RESPONDER]**
Diga a verdade sobre a origem. O que a auditoria garante: nenhuma é duplicata de imagem
do treino, e elas são tratadas como conjunto histórico, à parte da estimativa final.

### Se não souber responder

"Não medi isso. O que tenho de mais próximo é … e o experimento que responderia seria …"
É uma resposta completa. Inventar é o único erro grave.

---

## 12. Demonstração do sistema

Só faça ao vivo se treinou três vezes sem falha. Senão, vídeo de 40 segundos.

- **Roteiro (60 a 90 segundos).** Quatro imagens, uma para cada saída: folha saudável,
  folha com sintoma, uma imagem que caia em "inconclusivo" e uma parede ("não é folha").
- **A quinta imagem, se houver tempo.** Um erro de campo com confiança de 99%. É honesto e
  ilustra o achado principal melhor do que qualquer gráfico.
- **Plano B.** Capturas de tela nos slides de apoio. Backend e frontend rodando antes de
  entrar na sala; sem depender de internet.
- **Confira antes.** Qual ciclo está em `backend/weights/` no dia.

---

## 13. Plano de treino até 08/12

Oito semanas a partir de 13/10.

| Semana | Foco | Entrega |
|---|---|---|
| 13–19/10 | ~~Decidir o ciclo 5~~ (treinado em 14/10; artigo v7). Coleta de campo (adiada de 14/10); enviar o artigo v8 ao orientador | Fotos com metadados por horta |
| 20–26/10 | Avaliação cega dos ciclos 4 e 5; ECE e abstenção em campo; teste de *T* | Tabela de campo |
| 27/10–02/11 | Ler os sete artigos ★; teste de fundo; Grad-CAM corrigido | Fichamento de uma página por artigo |
| 03–09/11 | Ciclo 6 com fotos de adaptação, se aprovado; reescrever as Seções 4 e 5 do artigo | Artigo com Q3 respondida |
| 10–16/11 | Atualizar este guia; montar os slides | Deck completo |
| 17–23/11 | Ensaio 1, sozinho e cronometrado; estudar a Seção 4 | Gravação para rever |
| 24–30/11 | Ensaio 2, com o orientador ou colegas; treinar a Seção 11 | Lista de ajustes |
| 01–07/12 | Ensaio 3, em condições reais; testar a demonstração | Versão final |

### Como treinar

1. **Seção 4 primeiro.** Explique cada conceito em voz alta, sem olhar, como se fosse para
   um colega de outra área. Se travar, volte ao texto.
2. **Cartão de números.** Cubra a coluna da direita e recite.
3. **Perguntas em sorteio.** Peça a alguém para sortear dez da Seção 11.
4. **Cronometre sempre.** Estourar o tempo é o erro mais comum e o mais fácil de evitar.
5. **Grave-se uma vez.** Vícios de fala e slides lidos só aparecem na gravação.

---

## 14. O que atualizar neste guia

### Depois da coleta de campo

- [ ] Seção 1: último item da tese de 2 minutos
- [ ] Seção 2: Ato 8
- [ ] Seção 3: N de campo e números novos
- [ ] Seção 5: o que foi testado (fundo, *T*)
- [ ] Seção 8: limitação 1
- [ ] Seção 10: slide 11 e a fala correspondente
- [ ] Seção 11: perguntas 1, 15, 17 e 30
- [ ] Seção 4.14: Grad-CAM

### Depois de falar com o orientador

- [ ] Tempo de fala e de arguição; formato da banca
- [ ] Aval para o conjunto de adaptação (muda a regra "fotos de campo nunca treinam")
- [ ] Aval para manter o ViT como quarta arquitetura
- [ ] Se a demonstração ao vivo é bem-vinda
- [ ] Palavras-chave e demais pendências do artigo
- [ ] Composição da banca, para ajustar a ênfase (mais estatística, mais engenharia ou
      mais agronomia)

### Pendências já conhecidas

- [x] Inconsistências de tamanho do dataset, curso e parâmetros (auditoria de 13/10)
- [ ] Dizer de onde vieram as 13 fotos de campo e decidir o rótulo de `healthy/alface_04.jpg`
- [ ] Levar ao orientador o resultado por subconjunto (lavoura 86,3% × estúdio 99,7%)
- [x] Atualizar a resposta 5 quando o ciclo 5 estiver treinado (14/10; nova resposta 5b)
- [x] Ciclo 5 no guia: Ato 7b, cartão de números, slide 10b, falas 13 e 14 (14/10; o ciclo 5 entrou no artigo v7)
- [ ] Atualizar a limitação 2 da Seção 8 se o orientador tirar a Tabela 3 do artigo
- [ ] Ler o texto completo dos sete artigos ★
- [ ] Levar ao orientador o efeito da temperatura menor que 1 (ECE de campo 0,146 × 0,094)

---

## 15. Referências novas (Parte 3 da revisão)

Terceira rodada na Web of Science, em 13/10/2026, sobre relevância, adoção e
reprodutibilidade.

| # | Tema | Resultados |
|---|---|---:|
| 9 | Aplicativos de diagnóstico e adoção por produtores | 22 |
| 10 | Agricultura familiar e tecnologias digitais no Brasil | 13 |
| 11 | Vazamento, reprodutibilidade e protocolo de avaliação em aprendizado de máquina agrícola | 34 |
| — | Trabalhos de referência, por título | 9 |

- ADJÉI, E. A. et al. Perception and adoption by cassava farmers of the PlantVillage Nuru
  application disseminated in the agricultural environment of Côte d'Ivoire. Frontiers in
  Agronomy, v. 6, 1433204, 2024. DOI: 10.3389/fagro.2024.1433204.
- DA SILVEIRA, F. et al. Dynamics of Agriculture 4.0 technology adoption in the agri-food
  system: insights from an exploratory study in Rio Grande do Sul-Brazil. Foods, v. 15,
  n. 11, 1892, 2026. DOI: 10.3390/foods15111892.
- GERZ, F.; JELALI, M. Towards reliable data augmentation in machine learning: practices
  to prevent data leakage. Smart Agricultural Technology, v. 15, 102462, 2026. DOI:
  10.1016/j.atech.2026.102462.
- HASAN, M. et al. Comprehensive smart smartphone image dataset for plant leaf disease
  detection and freshness assessment from Bangladesh vegetable fields. Data in Brief,
  v. 56, 110775, 2024. DOI: 10.1016/j.dib.2024.110775. (12.643 imagens de campo de
  hortaliças; possível fonte retida.)
- KAPOOR, S.; NARAYANAN, A. Leakage and the reproducibility crisis in
  machine-learning-based science. Patterns, v. 4, n. 9, 100804, 2023. DOI:
  10.1016/j.patter.2023.100804.
- OVADIA, Y. et al. Can you trust your model's uncertainty? Evaluating predictive
  uncertainty under dataset shift. In: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS,
  32., 2019.
- RAMCHARAN, A. et al. A mobile-based deep learning model for cassava disease diagnosis.
  Frontiers in Plant Science, v. 10, 272, 2019. DOI: 10.3389/fpls.2019.00272.
- RECHT, B. et al. Do ImageNet classifiers generalize to ImageNet? In: INTERNATIONAL
  CONFERENCE ON MACHINE LEARNING, 36., 2019. Proceedings [...]. PMLR, v. 97, 2019.
- SAVARY, S. et al. The global burden of pathogens and pests on major food crops. Nature
  Ecology & Evolution, v. 3, n. 3, p. 430-439, 2019. DOI: 10.1038/s41559-018-0793-y.
- TIAN, Q. et al. Enhancing practicality of deep learning for crop disease identification
  under field conditions: insights from model evaluation and crop-specific approaches.
  Pest Management Science, v. 80, n. 11, p. 5864-5875, 2024. DOI: 10.1002/ps.8317.

As demais referências citadas neste guia estão no artigo v8 e em
`docs/REVISAO_LITERATURA_WOS.md`. NOYAN (2022) é um *preprint* do arXiv (2206.04374), já
registrado em `docs/CORRECOES_METODOLOGICAS.md` §11.7.
