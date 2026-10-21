# FitoVision — Próximos Passos (TCC-II)

### Última atualização: 16/10/2026 · literatura lida na íntegra (7 de 9 artigos); artigo v8 e monografia v5; coleta de campo a partir de 16/10

> **Leia este arquivo primeiro** quando voltar a trabalhar no projeto — ele
> resume onde paramos e o que fazer a seguir. Para a **coleta de fotos em hortas
> comunitárias**, o passo a passo prático está em
> [`docs/GUIA_CAMPO_HORTAS_COMUNITARIAS.md`](docs/GUIA_CAMPO_HORTAS_COMUNITARIAS.md). Os detalhes completos de cada
> etapa estão nos relatórios linkados abaixo.

---

## Estado em uma frase

**O ciclo 4 está treinado, calibrado e avaliado. No teste intra-fonte o
EfficientNet-B0 é o melhor dos três, com significância (98,48%); numa fonte nunca
vista a versão anterior caía para 75,4%, e em campo há só 13 fotos. O ViT-B/16 foi
treinado no Colab em 12–13/10 e empata com o EfficientNet-B0 no teste intra-fonte. O
orientador comentou o artigo v4; as correções dele e as da auditoria de 13/10 estão no
v6. Em 14/10 o ciclo 5 (pipeline corrigido) foi treinado nas quatro arquiteturas e
entrou no artigo v7. Em 16/10 a literatura foi lida na íntegra e corrigiu duas
leituras (artigo v8, monografia v5). O que responde à questão central continua pendente:
coletar fotos de campo com fundo real (a partir de 16/10).**

> A frase que estava aqui até 08/10 ("+9 a +14 pontos, p ≈ 10⁻¹³, sem perder nada no
> domínio antigo") foi corrigida duas vezes: o ganho é de +9 a +13 pontos só no
> espinafre (N = 484), e "sem perder nada" vale para EfficientNet-B0 e ResNet-50,
> não para o MobileNetV2 (seção de 12/10 abaixo).

> **Acréscimo de 13/10 (noite), depois da auditoria:** dentro do próprio teste, **81%
> dos erros estão nas culturas fotografadas em lavoura**, onde a acurácia balanceada é
> de 86,3%, contra 99,7% no resto da mesma fonte. **23% do teste são variantes
> transformadas de fotos de alface com outra variante no treino.** E "70% do treino
> deriva do PlantVillage" estava errado (é no máximo 47,7%). O artigo atual é o **v6**;
> os fatos conferidos estão em `docs/FICHA_TECNICA_FITOVISION.md`.

---

## 📚 16/10/2026 — literatura lida na íntegra; artigo v8; monografia v5

Sete dos nove trabalhos centrais foram lidos inteiros (Ergün, Kabya, Shafay, Yang,
Nguyen, Gerz e Jelali, Abbasi). Análise completa em `docs/REVISAO_LITERATURA_WOS.md`,
**Parte 4**. Faltam **Workneh** (precisa do login CAPES) e **Tian** (a Wiley pede
verificação anti-robô): baixar os PDFs para `docs/literatura/` (não versionar).

O que mudou:
- **Shafay et al. (2025) converge, não diverge:** a revisão aponta os híbridos (Swin)
  como mais robustos e o **ViT puro como mais sensível ao ambiente**; o ViT-B/16 daqui é
  o pior na lavoura. Corrigido no artigo, no guia e na monografia.
- **Gerz e Jelali (2026):** o "98,7% → 32,8%" é **mAP50 de detecção**, e o artigo deles se
  contradiz (32,77% no resumo, 30,64% na tabela). Citar "cerca de 31%".
- **Nguyen et al. (2026):** as CNNs erram a **cultura** antes do sintoma; é o mecanismo
  do atalho cultura → rótulo do A4 e o que o ciclo 5 atacou. Ajuste fino completo piora a
  transferência (50,6% → 29,4%); DINOv2 congelado é o experimento barato de um ciclo 6.
- **Ergün et al. (2026):** treino já de campo e queda de 27 pontos em outro vinhedo; o ECE
  de 0,038 vem com confiança média de 0,55; T = 8,34 (aqui é < 1).
- **Kabya e Abbasi:** resultados intra-fonte (Abbasi, na validação e com imagens da web).
- **Para a coleta:** marcar as folhas ambíguas como duvidosas, sem descartar (o descarte
  deixou o resultado de Ergün otimista). Já está no artigo v8 (Seção 3.2), no guia de
  campo e no LEIA-ME de `backend/campo_novo/`.

Arquivos: `docs/Entregas Finais/TCC2_Artigo_FitoVision_v8.docx` (gerador `build_v8.py`,
parte do v7 descompactado; continua com 11 páginas) e `TCC_FitoVision_ATUALIZADO_v5.docx`
(gerador `build_mono_v5.py`: nota de 16/10, seção 9.6, 6 referências, sumário completado
com a nota de 14/10 e o capítulo 9, que a v4 não tinha posto).

---

## ✅ 14/10/2026 (noite) — ciclo 5 completo, as quatro arquiteturas

Pesos em `backend/weights/ciclo5/`, relatórios em `backend/results/ciclo5/`
(`avaliacao_ciclo5.md`, `ciclo4_x_ciclo5.md`). Detalhe em CORRECOES §15.10.

| Modelo | Teste | Lavoura (c4 → c5) | Fonte retida |
|---|---|---|---|
| EfficientNet-B0 | 97,4 | 86,3 → 88,8 | 87,9 |
| ViT-B/16 | 96,9 | 83,4 → 84,5 | 85,8 |
| MobileNetV2 | 96,9 | 86,6 → 90,1 | 88,4 |
| ResNet-50 | 96,9 | 87,8 → 89,6 | 90,1 |

Em uma frase: a amostragem por cultura reduziu os falsos alarmes nas culturas de lavoura
(especificidade +7 a +12 pontos nas CNNs) em troca de alguma sensibilidade, e a fonte
retida de espinafre mostra cerca de 10 pontos de queda quando a fonte muda.

**Feito em 14/10 (noite), depois do treino:**
- **Artigo v7** (`docs/Entregas Finais/TCC2_Artigo_FitoVision_v7.docx` e `.pdf`, 11
  páginas no LibreOffice): parágrafo do ciclo 5 na Seção 3.1, linha 5 na Tabela 2, nova
  Tabela 3 (as quatro arquiteturas: teste, lavoura c4 → c5 com especificidade e
  sensibilidade, alface uma por foto, lote retido), parágrafo de resultados, frase na
  conclusão e no resumo/abstract, Quadro 1 com o lote retido. Para caber, vários
  parágrafos foram enxugados sem perder conteúdo (2.1–2.3, 3.1–3.3, Q1, Q2, discussão),
  e as células do Quadro 2, encurtadas. Gerador: `_gerador/build_v7.py <v6 descompactado>
  <saída>`, depois `pack2.py`.
- Documentos atualizados com o ciclo 5: README, HISTORICO (§9-F, lições 17–18), FICHA
  (§13), CORRECOES (§15.6–15.10), guia da banca, monografia v4, revisão da literatura,
  guia de campo.

**Próximo:**
1. **Coleta de campo** (adiada de 14/10) → `confere_fotos_campo.py` → `roda_campo.py`
   (roda às cegas ciclo 4, ViT, EfficientNet-B0 do ciclo 5 e ViT do ciclo 5).
2. **Enviar o v7 ao orientador** e discutir: (a) se a Tabela 3 fica no artigo ou vai para
   a monografia; (b) qual modelo servir — o ciclo 4 até o campo; o ciclo 5 se a horta
   mostrar falsos alarmes em folha sadia.
3. Conferir no Word se o v7 continua em 11 páginas (a contagem foi feita no LibreOffice).

## 🚀 14/10/2026 (madrugada) — ciclo 5 em treino no Colab

- **Auditoria do split do ciclo 5: veredicto OK.** 1,7% do teste (321 de 18.413) e 1,6% da
  validação (334 de 20.612) têm alguma imagem de treino a distância de órbita ≤ 4, contra
  23% no ciclo 4. Os pares listados são, em boa parte, de culturas diferentes (maçã sadia ×
  tomate com mancha): colisões do dHash em imagens de pouca textura, não a mesma foto. O
  número é um teto. Relatório: `backend/results/ciclo5/leakage_report.json`.
- A auditoria completa não termina com ~150 mil imagens; foi feita com
  `audit_leakage.py --rapido` (mesmo critério, exato, em ~10 min; ver CORRECOES §15.6).
- **Treino das quatro arquiteturas** (EfficientNet-B0 primeiro, depois ViT-B/16,
  MobileNetV2 e ResNet-50) lançado em 14/10 às 06:06, GPU L4, pela CLI (`colab/cli_c5/orq.sh`).
  Ao terminar, o pacote volta para `colab_export_c5/retorno/c5_resultados.zip`.
- As fotos de campo de hoje continuam valendo às cegas: o ciclo 5 foi congelado (lançado)
  antes de elas existirem.
- **EfficientNet-B0 do ciclo 5 pronto (08:00)** e já no PC (`backend/weights/ciclo5/`):
  teste 97,4%; lavoura 88,8% (ciclo 4 nas mesmas imagens: 86,3%; especificidade +6,8
  pontos, sensibilidade −1,8); alface uma por foto 96,7% [90; 100]; fonte retida de
  espinafre 87,9%. Contado por imagem, o ciclo 4 acerta um pouco mais (p = 0,03): o ciclo 5
  troca sensibilidade por especificidade. Detalhe em CORRECOES §15.7. O `roda_campo.py` já
  o inclui nas fotos de hoje.

## 🛠 Atualização de 13/10/2026 (noite) — correções de método prontas: ciclo 5 e validação em campo

Depois da auditoria (seção abaixo), o pipeline foi corrigido. **Nada do ciclo 4 foi
alterado** e **nenhum treino foi lançado ainda**. Detalhe em
`docs/CORRECOES_METODOLOGICAS.md` §15.

### O que mudou no código

| Arquivo | O que faz |
|---|---|
| `backend/prepara_ciclo5.py` | Refaz o split em `data_c5/` (hardlinks): alface agrupada por foto original, teste de alface com uma fotografia por foto, sem as variantes que não são fotografias, pastas de praga rotuladas, um lote de espinafre como fonte retida |
| `backend/dataset.py`, `train.py` | `--amostragem cultura`: equilibra os rótulos dentro de cada cultura; sem pesos de classe na perda |
| `backend/run_ciclo5.py` | Confere o split, audita, treina, calibra e avalia, tudo em pastas `ciclo5/` |
| `backend/avalia_ciclo5.py` | Resultado por subconjunto, na fonte retida, e ciclo 4 × ciclo 5 foto a foto no que é comparável |
| `backend/confere_fotos_campo.py` | Confere as fotos de campo antes de rodar o modelo (EXIF, cópias, duplicatas do treino, planilha) |
| `backend/roda_campo.py` | Validação em campo num comando, com todos os modelos disponíveis |
| `backend/download_datasets.py` | Palavras-chave de praga na rotulagem automática |

### Coleta de campo, na ordem (era para 14/10; adiada — o roteiro continua valendo)

1. **Na horta:** seguir `backend/campo_novo/LEIA-ME.md`. Arquivo original do celular,
   horta anotada, planta inteira e de perto, dano estrutural também.
2. **Em casa:** copiar as fotos para `backend/campo_novo/<cultura>/<healthy|anomalous>/` e
   preencher `backend/campo_novo/metadados_campo.csv`.
3. `cd backend` e `.venv\Scripts\python.exe confere_fotos_campo.py` — corrige o que ele
   apontar.
4. `.venv\Scripts\python.exe roda_campo.py` — roda às cegas o ciclo 4, o ViT, o
   EfficientNet-B0 do ciclo 5 e o ViT do ciclo 5 (todos já treinados). Relatórios em
   `backend/results/campo_novo/`.
5. Só depois: separar por horta o conjunto de avaliação e o de adaptação, e decidir com o
   orientador se o de adaptação entra em treino.

### Ciclo 5: lançar antes ou depois das fotos?

**Antes é melhor**, pelo mesmo motivo do ciclo 4: um modelo congelado antes de as fotos
existirem é avaliado às cegas. Treinado depois, deixa de ser. O comando é
`.venv\Scripts\python.exe run_ciclo5.py` (EfficientNet-B0, cerca de 6 horas na RTX 3050;
o computador não pode suspender, e o script bloqueia a suspensão).

---

## 🧾 Atualização de 13/10/2026 (noite) — auditoria de consistência, artigo v6, guia da banca

### O que foi feito

Cada número e cada afirmação de método foi conferido no artefato de origem (split em
disco, logs, código de treino, pesos, previsões por imagem), com
`backend/auditoria_consistencia.py`. Nenhum treino, nenhuma inferência nova.

| Achado | Número | Onde está |
|---|---|---|
| Erros do teste por tipo de foto | 86,3% [84,5–88,0] nas culturas fotografadas em lavoura (N = 4.530, 81% dos erros) contra 99,7% nas demais da mesma fonte (N = 10.061) | Ficha §9.2 |
| Vazamento por proveniência | 5.592 de 5.616 imagens de teste da fonte de variantes de alface (99,6%) têm outra variante da mesma foto no treino; 23% do teste | Ficha §5 |
| Composição do treino | No máximo 47,7% derivado do PlantVillage (não 70%); 17,6% de foto de lavoura; alface 29,9%, 86,6% dela saudável | Ficha §3.2 |
| Alface | 1.231 fotos originais, 108 doentes; o resto são cópias e transformações | Ficha §3.2 |
| Pastas fora do treino | 7.779 imagens de dano por praga não reconhecidas pela rotulagem | Ficha §3.1 |
| Parâmetros dos modelos treinados | 4,0 / 2,2 / 23,5 / 85,8 M; ViT ÷ EfficientNet-B0 = 21 (não 16) | Ficha §6 |
| Campo | Os 9 erros são de 4 fotos; `healthy/alface_04.jpg` tem rótulo duvidoso; a temperatura piora o ECE de campo (0,146 contra 0,094) | Ficha §9.4 |
| Tamanho do dataset | 176.126 reunidas → 170.772 na base → 175.123 no ciclo 4 | Ficha §3 |

### Documentos novos e atualizados

- **Novos:** `docs/FICHA_TECNICA_FITOVISION.md` (fonte única dos números),
  `docs/GUIA_BANCA_TCC2.md` (preparação e roteiro da apresentação de 08/12),
  `docs/REVISAO_LITERATURA_WOS.md` (três rodadas na Web of Science),
  `backend/auditoria_consistencia.py` e `backend/results/auditoria/`.
- **Artigo v6** (`docs/Entregas Finais/TCC2_Artigo_FitoVision_v6.docx`, 11 páginas):
  correções do orientador (da v5), trabalhos relacionados da Web of Science e as
  correções da auditoria. O gerador ficou em `docs/Entregas Finais/_gerador/`
  (`build_v6.py <pasta do v4 descompactado> <pasta de saída>`, depois `pack2.py`).
- **Corrigidos:** `HISTORICO_COMPLETO_TCC.md` (§9-E nova), `CORRECOES_METODOLOGICAS.md`
  (§14 nova), `README.md` e este arquivo.

### O que mudou do v4 para o v6

| Onde | Mudança |
|---|---|
| Cabeçalho | E-mails no formato `{josepedro, leandersonandre}@univille.br` |
| 1 Introdução | Reescrita em funil (IA → aplicações → agricultura → trabalhos relacionados → problema → questão → objetivos), sem subseções |
| 2 Revisão | Parágrafo de abertura; citações no fim das frases; 2.3 Trabalhos relacionados com o Quadro 1 |
| 3 Metodologia | De nove subseções para três; tamanho do dataset por etapa; frase sobre as variantes de alface |
| 4 Resultados | Tabela 1 com os parâmetros dos modelos treinados; "21 vezes"; comparação com a literatura; resultado por subconjunto (lavoura 86,3% × 99,7%); "nove erros em quatro fotografias" |
| Referências | Entraram sete da Web of Science; saiu Too et al. (2019) |

### Pendências que são suas

1. **Procedência das 13 fotos de campo.** Os arquivos não sustentam "capturadas com
   telefone celular" (monografia). Dizer de onde vieram.
2. **Rótulo de `healthy/alface_04.jpg`**: manter, trocar ou excluir.
3. ~~**Enviar o artigo v6 ao orientador**~~ — superado em 14/10: enviar o **v7**, que já tem o ciclo 5, e apontar o resultado por subconjunto.
4. **Palavras-chave** e paginação no Word (pendências da v5 que continuam).

---

## 📌 Atualização de 13/10/2026 — ViT-B/16 treinado no Colab e comparado

### Como foi o treino

Colab Pro, GPU L4, de 12/10 19:27 a 13/10 01:43. Receita do guia, sem alterações:
30 épocas, batch 64, LR 5e-5, weight decay 0,05, 2 épocas de aquecimento, clipping
1,0. Rodou as 30 épocas (a parada antecipada não disparou); melhor F1 macro na
validação 0,9845, na época 29; 6,1 h de treino, ~12 min por época. Temperatura
T = 0,620, ECE na validação de 0,047 para 0,002.

Custo: 12,7 unidades de computação no total (saldo de 299,99 para 287,29), das quais
1,9 perdidas numa primeira tentativa em que o PC suspendeu durante o envio dos dados.

### Resultado (tabelas completas em `backend/results/arquiteturas/comparacao_arquiteturas.md`)

| Recorte | ViT-B/16 | EfficientNet-B0 | McNemar ViT × EffNet (Holm) |
|---|---|---|---|
| Teste original (N = 23.967) | 98,48% [98,32; 98,64], FN 137, FP 204, ECE 0,0018 | 98,48% [98,32; 98,64], FN 209, FP 154, ECE 0,0039 | p = 0,256, não significativo |
| Espinafre (N = 484) | 84,39% [80,92; 87,71], FN 26, FP 42, ECE 0,1105 | 88,36% [85,28; 91,22], FN 31, FP 24, ECE 0,0746 | p = 0,513, não significativo |
| Teste completo (N = 24.624) | 98,22%, FN 163, FP 246 | 98,29%, FN 240, FP 178 | p = 0,684, não significativo |

- **Q1:** o ViT empata com o EfficientNet-B0 e supera MobileNetV2 e ResNet-50 com
  significância (p de Holm ≤ 1e-12 no teste original). Um transformer não muda o
  quadro no teste intra-fonte.
- **Perfil de erro diferente:** no teste original o ViT tem menos falsos negativos
  (137 contra 209) e mais falsos positivos (204 contra 154) que o EfficientNet-B0.
- **Q2 (abstenção):** o ViT é o mais confiante dos quatro, e isso joga contra ele.
  Cobertura de 99,5%, mas só 51 dos 341 erros ficam abaixo de τ (o EfficientNet-B0
  intercepta 113 de 363). No espinafre, o ECE do ViT é o pior dos quatro (0,11) e
  só 12 de 68 erros são interceptados.
- **Campo (13 fotos):** 12 de 12 certas e 1 abstenção (uma foto saudável, conf 0,77).
  N = 13: registro, não evidência.
- **Latência na RTX 3050, quatro modelos na mesma sessão (13/10):** MobileNetV2
  5,3 ms; EfficientNet-B0 7,2 ms; ResNet-50 8,6 ms; ViT-B/16 19,4 ms. Substitui as
  medições de sessões diferentes citadas abaixo
  (`results/latencia_13out/cnns/` e `results/vit/local_teste_original/`).

Leitura com os cuidados do guia: uma execução, uma semente; hiperparâmetros de
otimização próprios do ViT; espinafre está no treino do ciclo 4.

### Artigo v4 (13/10)

> **Superado em 13/10 (noite):** a versão atual é a **v6** (seção acima). O v4 é o que
> o orientador comentou e fica como registro.

**Versão enviada ao orientador:** `docs/Entregas Finais/TCC2_Artigo_FitoVision_v4.docx`
(**11 páginas** no LibreOffice, a última com 12 linhas; PDF ao lado). O v3 fica como
registro. O José liberou em 13/10 passar de 10 para 11 páginas, sem ir além disso.

| Onde | Mudança do v3 para o v4 |
|---|---|
| Resumo / abstract | ViT-B/16 comparado, não mais "será acrescentado"; empate com o EfficientNet-B0 (p = 0,26); 31% e 15% dos erros interceptados; "p < 10⁻¹¹" (com seis comparações o maior p de Holm é 1,4×10⁻¹²) |
| 4, abertura | Q1 e Q2 respondidas para as quatro arquiteturas |
| Tabela 1 | Linha do ViT-B/16; latências das quatro arquiteturas medidas na mesma sessão (7,2 / 5,3 / 8,6 / 19,4 ms) |
| 4, Q1 | Empate EfficientNet-B0 × ViT, ambos acima dos outros dois; perfil de erro distinto (FN 137 × 209, FP 204 × 154). Os p de Holm mudaram porque agora são seis comparações |
| 4, Q2 | ViT: 51 de 341 erros interceptados (15,0%), abstenção de 0,5% |
| 4, espinafre | ViT: 84,4%, sem diferença significativa (p = 0,51), pior ECE (0,11) |
| 3.3 | Uma frase dizendo onde cada modelo foi treinado (ViT em GPU L4 no Colab; CNNs na RTX 3050) |
| 4, campo | ViT nas 13 fotos como linha "4*" da Tabela 2 (12 de 12, Wilson 75,8% a 100%) e no texto; colunas da tabela realinhadas, sem quebra nos cabeçalhos |
| 5 | Q1 e Q2 para as quatro arquiteturas; "trocar a CNN por um transformer não altera esse quadro"; próxima etapa é só o campo |

**Nenhum corte no texto do v3.** Uma primeira versão do v4 cabia em 10 páginas à
custa de cortes de redação em 3.1, 3.2, 3.5 e 3.5.4; com o limite em 11 páginas eles
foram desfeitos. Há folga de umas 30 linhas na página 11 para o que vier do campo; se
o Word paginar diferente e passar para 12, cortar primeiro a frase final da Q2 (ECE
baixo × erros interceptados) e a frase da 3.3 sobre as GPUs.

**Mensagem sugerida para o MS Loop (substitui a de 12/10):**

> Bom dia, professor! Segue a versão mais nova do artigo do TCC-II: [link].
>
> **Prontas para correção:** 1 Introdução; 2 Revisão da literatura; 3.1 a 3.3 e 3.5
> (metodologia e plano de análise); 4, na parte de laboratório, agora com as quatro
> arquiteturas (o ViT-B/16 foi treinado e empata com o EfficientNet-B0).
>
> **Ainda vão mudar:** 3.4 e as análises de campo da Seção 4, que dependem da coleta
> nas hortas (nesta semana); 5 Conclusão, que acompanha esses resultados.
>
> **Duas decisões em que preciso do seu aval:** (1) manter o ViT-B/16 como quarta
> arquitetura, com hiperparâmetros de otimização próprios (Seção 3.3); (2) separar
> as fotos de campo por horta em um conjunto de avaliação e um de adaptação, o que
> muda a regra "fotos de campo nunca treinam" da Parcial I.

### Onde ficou cada coisa

- Pesos e calibração: `backend/weights/vit/`; resultados: `backend/results/vit/`;
  histórico e log do treino: `backend/logs/vit/`.
- Pacote bruto vindo do Colab: `colab_export/retorno/vit_resultados.zip`.
- Comparação só com as três CNNs, de 12/10, preservada como
  `comparacao_arquiteturas_3cnn_12out.md/.json`.

### Como repetir um treino no Colab (para o retreino com fotos de campo)

Não foi pelo notebook nem pelo Drive: foi pela CLI oficial `google-colab-cli`,
instalada no WSL Ubuntu (`~/.local/bin/colab`, login já feito). Os scripts estão em
`colab/cli/` (cópia de `~/fv/` no WSL); `orq.sh` faz tudo: cria o runtime L4, envia
os zips de `colab_export/` em pedaços de 32 MB, confere SHA-256, extrai, lança o
`run_vit.py`, baixa o pacote de resultados e desliga o runtime.

Três lições desta rodada:

1. **O PC não pode suspender.** O Windows suspende após 30 min sem uso; se isso
   acontece, o Colab recolhe o runtime e tudo o que foi enviado se perde. O `orq.sh`
   foi lançado com um bloqueio de suspensão (`SetThreadExecutionState`).
2. `colab upload` carrega o arquivo inteiro na memória (6,7 GB de RAM para 1,4 GB) e
   falha acima de ~100 MB por requisição; por isso o envio em pedaços.
3. O envio dos 11,4 GB levou 46 min e ~1,1 unidade. O runtime caiu em Singapura.

## 📌 Atualização de 12/10/2026 — artigo v3, Q1 respondida para as CNNs, ViT preparado

> **Registro histórico.** Superado pela seção de 13/10 em quatro pontos: o artigo
> atual é o v4; os p de Holm entre as CNNs mudaram com a entrada do ViT (seis
> comparações); a pendência de latência foi resolvida; e o treino do ViT foi feito
> pela CLI, não pelo roteiro Drive + notebook descrito abaixo.

### Entrega ao orientador

- **Artigo para correção:** `docs/Entregas Finais/TCC2_Artigo_FitoVision_v3.docx`
  (10 páginas; há um PDF ao lado para conferência). O v2 fica como registro.
- O Prof. Leanderson pediu, no MS Loop, o link do documento e a lista das seções
  prontas. Texto sugerido para a mensagem: ver o fim desta seção.
- A monografia (`TCC_FitoVision_ATUALIZADO_v3.docx`) não foi alterada em 12/10. (Em 14/10 saiu a v4, em `docs/Entregas Finais/`.)

### O que mudou do v2 para o v3

| Onde | Mudança |
|---|---|
| Resumo / abstract | Passam a relatar o resultado do Q1 e do Q2, e que o ViT será acrescentado |
| 1.2 e 2.1 | ViT-B/16 como arquitetura de contraste; referência Dosovitskiy et al. (2021) |
| 3.3 | Receita do ViT e o que difere das CNNs (LR, weight decay, aquecimento, clipping) |
| 3.5.3 | A Tabela 1 (amplitude do IC de Wilson por N) virou uma frase, para caber em 10 páginas; as outras tabelas foram renumeradas |
| 4 | Tabela 1 com os modelos do **ciclo 4**; IC de 95%, McNemar entre arquiteturas, Mann-Whitney e erros interceptados com números novos; ciclo 3 × ciclo 4 por modelo |
| 5 | Conclusão diz o que já tem resposta (Q1 e Q2 no teste) e o que continua aberto (campo) |

**Conferir no Word:** o arquivo fecha em 10 páginas no LibreOffice, com 3 linhas de
folga. O Word pode quebrar linhas de forma um pouco diferente; se passar para 11,
há frases marcadas como candidatas a corte no fim desta seção.

### Resultados novos (detalhe em `docs/CORRECOES_METODOLOGICAS.md` §12)

| O quê | Número |
|---|---|
| EfficientNet-B0, teste original | 98,48% (IC 95%: 98,32 a 98,64), 209 falsos negativos |
| EfficientNet-B0 × MobileNetV2 / × ResNet-50 | p de Holm 9,0×10⁻¹³ / 2,6×10⁻¹⁷ |
| MobileNetV2 × ResNet-50 | p = 0,13 (não diferem) |
| Espinafre (N = 484), três CNNs | 0,884 / 0,874 / 0,862, nenhum par significativo |
| Erros interceptados por τ = 0,85 (EfficientNet-B0, teste original) | 113 de 363 (31,1%), abstenção de 1,2% |
| **MobileNetV2, ciclo 3 → ciclo 4, teste original** | **0,9818 → 0,9789, p de Holm = 0,015** |

**Parar de escrever:** "o ciclo 4 não custou nada no domínio antigo" sem dizer que
isso vale para EfficientNet-B0 e ResNet-50; "138 de 404 erros interceptados" (número
de um ciclo antigo, vindo da monografia do TCC-I).

**Pendência de medição:** a latência variou até 3× entre execuções (MobileNetV2:
5,1 e 18,2 ms). Medir as quatro arquiteturas na mesma sessão antes da versão final.

### ViT-B/16 no Colab Pro — como fazer

Guia completo: [`docs/GUIA_TREINO_VIT_COLAB.md`](docs/GUIA_TREINO_VIT_COLAB.md).
Resumo:

1. `cd backend` e `.venv\Scripts\python.exe empacota_colab.py` → gera `colab_export/`
   (4 zips, 11,4 GB, mais o notebook). **Já rodado em 12/10.**
2. Subir o conteúdo de `colab_export/` para a pasta `FitoVision_Colab` do Google
   Drive (conferir antes se há ~13 GB livres).
3. Abrir `FitoVision_ViT_Colab.ipynb` no Colab, escolher GPU e RAM alta, **Executar
   tudo**. Se a sessão cair, executar tudo de novo: o treino retoma.
4. Baixar `vit_resultados.zip`, extrair em `backend/` e rodar
   `compara_arquiteturas.py` e o `evaluate.py` da latência (comandos no guia).
5. Acrescentar a linha do ViT na Tabela 1 do artigo e o resultado do McNemar.

O pipeline foi testado de ponta a ponta com um mini-dataset; o treino real ainda
não foi executado e o tempo no Colab é estimativa (2 a 8 horas, conforme a GPU).

### Mensagem sugerida para o MS Loop

> Bom dia, professor! Segue a versão mais nova do artigo do TCC-II: [link].
>
> **Prontas para correção:** 1 Introdução (questão de pesquisa e objetivos);
> 2 Revisão da literatura; 3.1 a 3.3 e 3.5 (metodologia e plano de análise);
> 4, na parte de laboratório (comparação das três CNNs com McNemar, calibração e
> abstenção, fonte nunca vista).
>
> **Ainda vão mudar:** a linha do ViT-B/16 na Tabela 1 (treino no Colab nesta
> semana); 3.4 e as análises de campo da Seção 4, que dependem da coleta nas
> hortas; 5 Conclusão, que acompanha esses resultados.
>
> **Duas decisões em que preciso do seu aval:** (1) incluir o ViT-B/16 como quarta
> arquitetura, com hiperparâmetros de otimização próprios (Seção 3.3); (2) separar
> as fotos de campo por horta em um conjunto de avaliação e um de adaptação, o que
> muda a regra "fotos de campo nunca treinam" da Parcial I.

### Se o artigo passar de 10 páginas no Word

Candidatas a corte, em ordem: a frase do classificador trivial (fim da 3.5.1); a
última frase do parágrafo das 13 fotos ("A amostra atual foi suficiente...");
"embora as imagens também sejam de folha isolada sobre fundo claro" (Seção 4).

---

## 🔎 Revisão crítica de 08/10/2026 — ler antes de qualquer texto novo

Ao rever o Seminário 2, a leitura do projeto mudou. **O modelo está amarrado aos
dados de treino, e o teste se parece com o treino.** Verificado nos artefatos que
já existiam (`backend/reanalise_teste_novo.py` → `results/ciclo4/reanalise_por_lote.json`):

| O quê | Número |
|---|---|
| ~~Fontes derivadas do PlantVillage no treino / no teste~~ | ~~70,1% / 72,0%~~ — **corrigido em 13/10:** esses números somam duas fontes, e uma é uma coletânea com outras culturas. O derivado do PlantVillage é no máximo 47,7% do treino. O que vale: as 6 fontes aparecem nas mesmas proporções nas três partições |
| Acurácia **intra-fonte** (o antigo "laboratório") | 98,5% (N = 23.967) |
| **Fonte nunca vista** — espinafre no ciclo 3, foto de estúdio | **75,4%** (N = 484), queda de 23 pontos |
| Ganho do ciclo 4, só espinafre (corrige o N = 657) | 0,754 → 0,884; p de Holm 5,9×10⁻¹² a 1,9×10⁻⁷ |
| Confiança dos 9 erros de campo, nos 4 ciclos | 0,949 a 0,999 — abstenção não segurou nenhum |

**O que parar de escrever:** "em laboratório o sistema é confiável"; "p ≈ 10⁻¹³
com N = 657" (173 dessas imagens são recortes de 5 fotos); "o sistema sabe quando
se abster" sem dizer que isso só foi medido na validação.

**O que passou a ser o plano** (itens 2 e 3 dependem do aval do Prof. Leanderson,
porque mudam a regra "fotos de campo nunca treinam" da Parcial I):

1. Coletar fotos **com fundo real**, anotando a horta de cada uma.
2. Separar **por horta** em dois conjuntos: avaliação (nunca treina) e adaptação.
3. Rodar o ciclo 4 congelado em tudo, às cegas. Só depois treinar um ciclo 5 com
   o conjunto de adaptação + bases públicas de campo (ex.: PlantDoc), e medir de
   novo no conjunto de avaliação.
4. Teste de dependência do fundo: mesmas imagens com o fundo removido e com a
   folha removida.
5. Reportar ECE e abstenção em campo, não só na validação.

**Pendência sua:** registrar de onde veio cada uma das 13 fotos de `backend/campo/`
(4 são PNG de 400 px, uma tem EXIF de câmera Sony DSC-H70). A monografia diz
"capturadas com celular"; se alguma veio de outra fonte, o texto precisa dizer.

Detalhe: `docs/CORRECOES_METODOLOGICAS.md` §11 (inclui a tabela com os 23 erros e
ajustes do projeto) e `docs/HISTORICO_COMPLETO_TCC.md` §9-B.

Arquivos atualizados em 08/10: deck do Seminário 2 (claude.ai, 9 slides),
`docs/TCC2_Artigo_FitoVision_v2.docx` e `docs/TCC_FitoVision_ATUALIZADO_v3.docx`.
Os `.docx` e o PDF do TCC-I estão agora em `docs/`, não mais na raiz.

---

## ⚡ Atualização 22/09/2026 — ciclo 4 lançado ANTES das fotos de campo

**Decisão revista:** as fotos de campo servem só para avaliação (regra da
Parcial I) e nunca entram no treino, então o ciclo 4 não depende delas.
Treinar agora tem outra vantagem: o modelo fica **congelado antes de vermos as
fotos novas**, e a avaliação de campo passa a ser cega de verdade. Isto
substitui o item 2 do §10.8 do `CORRECOES_METODOLOGICAS.md` ("não lançar
ciclo 4 antes do campo"). O que continua valendo: **a decisão de qual ciclo vai
para produção (3 ou 4) só é tomada com as fotos novas.**

- Merge feito com `backend/merge_ciclo4.py` (+4.351 imagens), **sem refazer o
  split existente**. O teste do ciclo 3 está contido no do ciclo 4 e a
  comparação é justa nesse subconjunto.
- **Bug achado e corrigido:** o `lettuce_downy_v1` são 1.142 recortes de só
  **49 fotos originais**, e o split da Fase 6 vazava entre treino e teste.
  O split foi refeito agrupado por foto (ver nota no fim do `PHASE6_REPORT.md`).
- Backup do ciclo 3: `backend/weights/ciclo3/`, `results/ciclo3/`, `logs/ciclo3/`.
- Pipeline: `backend/run_ciclo4.py` (retomável). Log em `backend/logs/ciclo4.log`,
  PID em `logs/ciclo4.pid.txt`. Resultados em `results/ciclo4/`
  (`baseline_ciclo3_teste_novo`, `teste_original`, `teste_novo`).
- ⚠ Durante e depois do treino, `backend/weights/*.pth` são os do **ciclo 4**.
  Para voltar a demo ao ciclo 3: copiar `weights/ciclo3/*` para `weights/`.
- **ViT**: fica como experimento exploratório **depois** do ciclo 4, no mesmo
  dataset final, para comparar arquiteturas (entraria no Q1). José vai
  alinhar com o Prof. Leanderson (que já tinha gostado da ideia). ~~Não precisa
  de Colab Pro: ViT-S/DeiT-S cabe na RTX 3050 com AMP.~~ **Revisto em 12/10:** vai
  ser o ViT-B/16 no Colab Pro, com batch 64 como nas CNNs (seção de 12/10 acima).
- **Feito em 22/09**: `backend/analise_campo.py` (toda a análise da Parcial I num
  comando) e `backend/campo/metadados_campo.csv` (planilha para preencher NA HORA
  da coleta — `categoria_dano` e `distancia` não dá para recuperar depois).
  O `test_campo.py` agora varre `campo/<cultura>/<label>/` e escreve a coluna
  `cultura`.
- Falta até ~05/10: adiantar o texto do TCC-II (metodologia do ciclo 4, ressalva
  do Malabar Spinach, trabalhos relacionados).

---

### ✅ Ciclo 4 concluído em 23/09/2026 — resultados

**Teste original** (mesmas 23.967 imagens do ciclo 3 — comparação justa): empate.
EfficientNet-B0 0,9848 vs 0,9849 de bal. acc. Aprender culturas novas não custou
desempenho nas antigas.

> ⚠ **Corrigido em 08/10:** a tabela abaixo usa N = 657, com 173 recortes de 5
> fotos de míldio. Citar os números só de espinafre da seção de revisão acima.

**Teste novo** (657 imagens de espinafre + downy), McNemar pareado + Holm:

| Modelo | Ciclo 3 | Ciclo 4 | p (Holm) |
|---|---|---|---|
| EfficientNet-B0 | 0,811 | **0,903** | 7,1×10⁻¹⁴ |
| ResNet50 | 0,795 | **0,884** | 3,7×10⁻¹³ |
| MobileNetV2 | 0,758 | **0,901** | 3,5×10⁻¹¹ |

**Calibração melhor**: ECE do EfficientNet-B0 no teste novo caiu de 0,175 para
0,053; no val, 0,0027 depois do ajuste de temperatura, com τ=0,85 atingível
(cobertura 98,9%, acurácia seletiva 99,0%).

**Campo (13 fotos históricas)**: bal. acc subiu de 73,3% (ciclo 3) para 83,8%, ECE
de 0,274 para 0,146 — **mas o McNemar dá p=1,00**, como esperado com N=13. Esse
contraste (p≈10⁻¹³ com N=657, p=1,00 com N=13, mesmos modelos) virou um dos
melhores argumentos do TCC sobre poder estatístico. Detalhe completo: §10.9 do
`docs/CORRECOES_METODOLOGICAS.md`.

⚠ **Decisão pendente de produção**: `backend/weights/` tem hoje os pesos do
**ciclo 4**. Ele é igual ao ciclo 3 no domínio antigo, muito melhor nas culturas
novas e melhor calibrado — mas a escolha formal espera as fotos novas. Para voltar
ao ciclo 3: copiar `weights/ciclo3/*` para `weights/`.

---

## Lotes preparados na Fase 6 (mergeados em 22/09 — ver acima)

| Lote | Imagens finais | Localização | Relatório |
|---|---|---|---|
| Espinafre (Malabar Spinach, Mendeley) | 3.209 (2.245 treino / 480 val / 484 teste) | `dataset/_work/final/spinach/` | [`dataset/_reports/SPINACH_VALIDATION_REPORT.md`](dataset/_reports/SPINACH_VALIDATION_REPORT.md) |
| Alface — downy mildew (Roboflow) | 1.142 (799 / 171 / 172) | `dataset/_work/final/lettuce_downy_v1/` | [`dataset/_reports/DATASET_VALIDATION_REPORT.md`](dataset/_reports/DATASET_VALIDATION_REPORT.md) |

Manifestos com proveniência + phash: `dataset/_reports/final_manifest_spinach.csv`
e `final_manifest_lettuce_downy_v1.csv`. Detalhe completo do que foi feito
(dedup, split, transforms): [`dataset/_reports/PHASE6_REPORT.md`](dataset/_reports/PHASE6_REPORT.md).

**Ressalva pendente de decisão:** o espinafre é *Malabar Spinach* (Basella
alba), não o espinafre-verdadeiro (Spinacia oleracea) do escopo original —
precisa declarar isso no texto do TCC-II quando for usar esse dado.

**6 datasets Roboflow de alface foram descartados** (85–94% duplicata do
que já existia, sem fotos de campo real) — detalhes no relatório acima, não
precisa reabrir essa investigação.

---

## O bloqueio real: fotos de campo

Roteiro detalhado já pronto:
[`docs/ROTEIRO_CAMPO_TCC2_MULTIESPECIE.md`](docs/ROTEIRO_CAMPO_TCC2_MULTIESPECIE.md).

Resumo: sair de **"1 cultura (alface), N=13"** para **"pelo menos 2-3
culturas, N=15-25 cada"** — rúcula, espinafre, acelga ou couve, o que for
mais acessível. José está esperando o tempo melhorar em Joinville para
sair e fotografar.

**Quando as fotos chegarem, os passos são:**
1. Organizar em `campo/<cultura>/<healthy|anomalous>/` (ver
   `docs/GUIA_TESTE_DE_CAMPO.md` §1 e o roteiro acima §4) + preencher
   `backend/campo/metadados_campo.csv` na hora da coleta.
2. `python test_campo.py --dir ./campo --out ./results/campo_resultados_v5.csv`
   (já varre todas as culturas de uma vez e grava a coluna `cultura`), depois
   `python analise_campo.py --csv ... --metadados campo/metadados_campo.csv`.
3. Calcular acurácia balanceada de campo por cultura, comparar com o
   domínio de teste e com o N=13 anterior (agora com mais poder
   estatístico).
4. **Só então** decidir sobre o merge dos dados preparados (próxima seção).

---

## Decisão pendente: qual ciclo vai para produção

O merge e o ciclo 4 **já foram feitos** (22-23/09) — a seção acima tem os
resultados. O que continua pendente é só a escolha entre ciclo 3 e ciclo 4 para
produção, e ela **espera as fotos de campo**, para não ser tomada apenas com
dados de dataset.

O que já se sabe hoje, sem as fotos:

| Critério | Vencedor |
|---|---|
| Domínio de teste antigo | empate (0,9848 vs 0,9849) |
| Culturas novas (espinafre, downy) | **ciclo 4**, com folga e significância |
| Calibração (ECE) | **ciclo 4** (0,053 vs 0,175 no teste novo) |
| Campo (13 fotos históricas) | ciclo 4 no ponto estimado, mas p=1,00 |

`backend/weights/` tem hoje o **ciclo 4**. Para voltar ao ciclo 3, copiar
`backend/weights/ciclo3/*` para `backend/weights/`.

---

## Mapa de documentos (o que ler para quê)

| Preciso saber... | Ler |
|---|---|
| **Qual é o artigo atual do TCC-II** | **`docs/Entregas Finais/TCC2_Artigo_FitoVision_v8.docx`** (16/10: v7 + correções da literatura; as anteriores ficam como registro) |
| **Ciclo 5: resultados por subconjunto e contra o ciclo 4** | **`backend/results/ciclo5/avaliacao_ciclo5.md`**, **`backend/results/ciclo5/ciclo4_x_ciclo5.md`**; CORRECOES §15 |
| **Um número ou uma afirmação de método, com a prova** | **`docs/FICHA_TECNICA_FITOVISION.md`** |
| **Preparação para a banca de 08/12 e roteiro da apresentação** | **`docs/GUIA_BANCA_TCC2.md`** |
| Trabalhos relacionados e por que o modelo cai em campo, segundo a literatura | `docs/REVISAO_LITERATURA_WOS.md` (Parte 4: leitura do texto completo, 16/10) |
| **Como treinar no Colab (CLI, scripts)** | **`colab/cli/orq.sh`** (ViT) e **`colab/cli_c5/orq.sh`** (ciclo 5, quatro arquiteturas, com trava da auditoria e keepalive); `docs/GUIA_TREINO_VIT_COLAB.md` |
| Comparação pareada entre arquiteturas (números e tabelas) | `backend/results/arquiteturas/comparacao_arquiteturas.md` |
| Todo o histórico do projeto, para a banca | `docs/HISTORICO_COMPLETO_TCC.md` |
| Detalhe metodológico de cada ciclo de correção | `docs/CORRECOES_METODOLOGICAS.md` |
| **Coleta em hortas comunitárias — passo a passo** | **`docs/GUIA_CAMPO_HORTAS_COMUNITARIAS.md`** |
| Como fotografar/testar/analisar campo (referência detalhada) | `docs/GUIA_TESTE_DE_CAMPO.md` |
| Metas específicas da próxima campanha multi-espécie | `docs/ROTEIRO_CAMPO_TCC2_MULTIESPECIE.md` |
| Por que os 6 datasets Roboflow de alface foram rejeitados | `dataset/_reports/DATASET_VALIDATION_REPORT.md` |
| Por que o espinafre foi aprovado | `dataset/_reports/SPINACH_VALIDATION_REPORT.md` |
| O que a Fase 6 fez exatamente (dedup/split/manifesto) | `dataset/_reports/PHASE6_REPORT.md` |

---

## Checklist rápido

- [x] 6 datasets Roboflow de alface avaliados — 4 descartados, 1 aprovado (`lettuce_downy_v1`)
- [x] 2 datasets de espinafre (Mendeley) avaliados e aprovados
- [x] Fase 6 rodada (dedup + split 70/15/15 + manifesto) para os 2 lotes aprovados
- [x] Roteiro de campo multi-espécie entregue
- [ ] **Fotos de campo tiradas** (bloqueante — esperando tempo bom)
- [ ] Teste de campo por cultura rodado e analisado
- [x] Decisão de merge tomada (22/09 — merge feito, split do downy corrigido)
- [x] Ciclo 4 de treino executado, calibrado e avaliado (22-23/09) — ganho significativo nas culturas novas
- [x] Scripts de análise estatística prontos e testados (`backend/analise_campo.py`, 22/09 — reproduz os ICs do §10.7)
- [x] Revisão crítica: teste intra-fonte, reanálise do ciclo 4 sem pseudo-replicação, confiança em campo (08/10)
- [x] Comparação pareada entre as três CNNs: IC por bootstrap, McNemar + Holm, abstenção (12/10)
- [x] Artigo v3 gerado, em 10 páginas (12/10)
- [x] Pipeline do ViT preparado e testado com mini-dataset; zips do Colab gerados (12/10)
- [x] Treinar o ViT-B/16 no Colab (12–13/10, pela CLI; ver seção de 13/10)
- [x] Trazer os pesos do ViT e rodar `compara_arquiteturas.py` (13/10)
- [x] Medir a latência das quatro arquiteturas na mesma sessão (13/10)
- [x] Levar o ViT ao artigo: v4 gerada em 13/10, 11 páginas (limite liberado pelo José)
- [x] Artigo v4 comentado pelo orientador (13/10); correções aplicadas na v5 e, com a auditoria, na v6
- [x] Revisão da literatura na Web of Science, três rodadas (13/10)
- [x] Guia de preparação para a banca (13/10)
- [x] Auditoria de consistência: dataset, arquitetura, método e resultados contra os artefatos (13/10)
- [x] Ciclo 5 preparado (13/10), auditado (1,7%) e treinado no Colab nas quatro arquiteturas (14/10)
- [x] Artigo v7 com o ciclo 5, 11 páginas no LibreOffice (14/10)
- [x] Literatura lida na íntegra, 7 de 9 (16/10); artigo v8 e monografia v5
- [ ] Baixar os PDFs de Workneh (login CAPES) e Tian (Wiley) para `docs/literatura/` e terminar a leitura
- [ ] **Enviar o artigo v8 ao orientador pelo MS Loop** (substitui o envio do v7)
- [ ] Conferir no Word se o artigo v8 continua em 11 páginas (não passar disso)
- [ ] Dizer de onde vieram as 13 fotos de campo e decidir o rótulo de `healthy/alface_04.jpg`
- [ ] **Coleta de campo** (prevista para 14/10, adiada; nova data a definir), depois recalibrar e validar; apresentação em 08/12
- [ ] Alinhar com o orientador a divisão avaliação/adaptação por horta (muda regra da Parcial I)
- [ ] Registrar a procedência das 13 fotos históricas de campo
- [ ] Teste de dependência do fundo (fundo removido / folha removida)
- [ ] Decisão do modelo servido — ciclo 4 ou EfficientNet-B0 do ciclo 5 (só com as fotos novas)
- [ ] Ressalva de espécie do Malabar Spinach registrada no texto do TCC-II
