# Guia da Coleta em Hortas Comunitárias — FitoVision

### TCC-II · UNIVILLE · Escrito em 23/09/2026, depois do ciclo 4

> **Para que serve este guia:** é o documento que você abre quando voltar ao
> projeto. Explica, em ordem, **o que já está pronto**, **o que falta**, **o que
> fazer na horta** e **quais comandos rodar ao voltar**. Os outros documentos são
> mais detalhados; este é o que dá para ler no ônibus a caminho da horta.

---

## Parte 1 — Onde o projeto está (resumo honesto)

### O que já está pronto

O modelo passou por **4 ciclos de correção e retreino**. O ciclo 4 terminou em
23/09/2026 e é o primeiro com um ganho que **passa num teste estatístico**:

| O quê | Número |
|---|---|
| Acurácia no domínio de teste (EfficientNet-B0) | **98,5%** |
| Acurácia nas culturas novas (espinafre + downy mildew) | **90,3%** — era 81,1% no ciclo 3 |
| Significância desse ganho | p ≈ 10⁻¹⁴ (McNemar pareado + Holm) |
| Calibração (ECE, quanto a confiança "mente") | **0,0027** no val; 0,053 nas culturas novas |
| Vazamento de dados auditado | 0,04% |
| Dataset | 175.123 imagens, 101.987 fotos distintas |

O sistema também sabe **se calar**: quando a confiança fica abaixo de 0,85 ele
responde `inconclusive` em vez de chutar (cobre 98,9% dos casos e acerta 99,0%
dos que decide).

### O que está travado, e é só isto

**Falta medir o modelo em fotos de campo de verdade, em mais de uma cultura.**

Hoje o projeto tem **13 fotos de campo, todas de alface**. Com N=13 não dá para
decidir nada: os mesmos três modelos que dão p ≈ 10⁻¹³ quando comparados em 657
imagens dão **p = 1,00** quando comparados nessas 13 fotos. Não é o modelo que
muda — é a amostra que é pequena demais para enxergar qualquer diferença.

> **Esse contraste virou um dos melhores argumentos do seu TCC.** Você não vai
> apresentar "o modelo é bom"; vai apresentar "eis como se descobre se um
> resultado é real ou ruído" — e mostrar os dois casos medidos por você.

### O que a sua coleta destrava

1. **A decisão de produção**: ciclo 3 ou ciclo 4? Hoje `backend/weights/` tem o
   ciclo 4, mas a escolha formal espera as suas fotos.
2. **A resposta às três hipóteses** prometidas na Parcial I (H1 cultura,
   H2 cor vs estrutura, H3 distância).
3. **O número central do TCC-II**: a acurácia de campo, por cultura, com
   intervalo de confiança.

---

## Parte 2 — Antes de sair de casa

### Combinar o acesso

Hortas comunitárias costumam ter um responsável ou associação. **Peça autorização
antes**, explique em uma frase ("sou aluno da Univille, estou testando um sistema
que identifica doença em folha por foto, preciso fotografar folhas — não vou
colher nem mexer nas plantas"). Se puder, combine de voltar outro dia: duas
visitas curtas rendem mais que uma longa.

### Levar

- Celular com bateria e **espaço livre** (200 fotos ocupam bastante).
- A planilha de metadados aberta no celular — `backend/campo/metadados_campo.csv`
  (ou o bloco de notas, e depois você transcreve).
- Se for usar, um cartão branco ou folha de papel para referência de cor.

### Regra de ouro, sem exceção

**Estas fotos são para MEDIR, nunca para treinar.** Se um dia fizermos
fine-tuning com fotos de campo, vai ser preciso um **segundo lote** de fotos
novas. Misturar as duas coisas invalidaria o resultado inteiro — é a mesma
família de erro do vazamento de dados que já custou os 99,01% retratados.

---

## Parte 3 — Na horta: o que fotografar

### Meta de quantidade

| | Mínimo | Ideal |
|---|---|---|
| Por cultura | 15 fotos | 25 fotos |
| Culturas diferentes | 2 | 3 ou mais |
| Total | ~30 | ~75 |

**Menos culturas bem cobertas vale mais do que muitas mal cobertas.** Cinco
culturas com 6 fotos cada não permitem calcular nada com confiança.

Prioridade: **uma cultura nova além da alface** (rúcula, espinafre, acelga ou
couve — a que estiver acessível). Sair de "1 cultura testada" para "2 culturas
testadas" é o maior salto que essa coleta pode dar ao trabalho.

### Equilíbrio saudável vs doente

Tente **metade e metade** em cada cultura. Se a horta estiver saudável demais
(ótimo para eles, ruim para o teste), anote quantas conseguiu de cada tipo e
compense na próxima cultura. Não invente doença nem force o enquadramento para
"parecer" doente.

### Varie de propósito

Cada variação abaixo testa uma coisa diferente. Não precisa ser em todas as
fotos, mas tente cobrir cada uma pelo menos algumas vezes por cultura:

| Varie | Por quê |
|---|---|
| **Tipo de sintoma**: mancha, clorose (amarelão), míldio/oídio (pó branco), furo de inseto, murcha, ponta seca | É o que testa a hipótese "o modelo pesa mais cor do que estrutura" |
| **Distância**: close (folha preenche a tela), média (folha + caule/vaso), ampla (planta inteira + canteiro) | Testa se o sistema aguenta a condição difícil. Tente as 3 na mesma folha em 20-30% dos casos |
| **Luz**: manhã, meio-dia nublado, fim de tarde | Luz é o fator que mais muda entre foto de estúdio e foto real |
| **Fundo**: terra, outras plantas, mão segurando | O dataset de treino tem fundo neutro; a realidade não |

### Regras técnicas de cada foto

- Foco nítido na folha (toque na tela antes de disparar).
- Luz natural, **sem flash**, evitando sol direto forte e sombra dura.
- Uma folha em destaque (pode ter outras ao fundo).
- Não corte a folha ao meio no enquadramento.

### As "pega-ratão"

2 ou 3 fotos por cultura que **não são folha**: terra, sua mão, parede, calçada.
Servem para confirmar que a guarda de vegetação responde `not_a_leaf` em vez de
inventar um diagnóstico. Guarde em `nao_folha/`.

### Anote NA HORA — isto não dá para recuperar depois

Para cada foto, na planilha `backend/campo/metadados_campo.csv`:

```
ficheiro, cultura, verdade, tipo_sintoma, categoria_dano, distancia, hora, notas
```

Duas colunas são as que destravam as hipóteses do artigo:

- **`categoria_dano`** — `cromatico` (o sintoma é **cor**: mancha, clorose,
  míldio) ou `estrutural` (o sintoma é **forma**: furo, murcha, ponta seca).
  Use `misto` se tiver os dois. **Esta coluna responde H2.**
- **`distancia`** — `close`, `media` ou `ampla`. **Esta responde H3.**

E o mais importante: **se ficar em dúvida se a folha está doente, anote a
dúvida** ("sintoma leve, conferir") em vez de decidir de memória à noite. Uma
foto com gabarito duvidoso contamina a medição inteira — é melhor marcar como
incerta e decidirmos juntos depois.

---

## Parte 4 — Ao voltar para casa

### 1. Organize as fotos

```
backend/campo/
  alface/
    healthy/IMG_alface_001.jpg
    anomalous/IMG_alface_010.jpg
  rucula/
    healthy/...
    anomalous/...
  espinafre/
    healthy/...
    anomalous/...
  nao_folha/
    IMG_020.jpg
```

⚠ As 13 fotos antigas já estão em `backend/campo/{healthy,anomalous}/`. **Não as
apague** — elas são o "conjunto histórico" e são reportadas à parte. As fotos
novas vão nas pastas por cultura, ao lado delas.

### 2. Rode o teste (2 comandos)

```powershell
cd backend
.venv\Scripts\python.exe test_campo.py --dir ./campo --out ./results/campo_resultados_v5.csv
.venv\Scripts\python.exe analise_campo.py --csv results/campo_resultados_v4.csv:ciclo4 results/campo_resultados_v5.csv:campo_novo --metadados campo/metadados_campo.csv
```

O primeiro roda o modelo em todas as fotos e gera a planilha por foto. O segundo
calcula **tudo o que a Parcial I promete**: acurácia balanceada com intervalo por
bootstrap, IC de Wilson, McNemar com correção de Holm, ECE, curva risco-cobertura
e as três hipóteses. Sai um relatório em `results/analise_campo.md`, pronto para
colar no TCC.

### 3. O que olhar no resultado

| Pergunta | Onde ela é respondida |
|---|---|
| O modelo generaliza igual em todas as culturas? | tabela "por cultura" + H1 |
| Ele erra mais em dano estrutural que em dano de cor? | H2 |
| A acurácia cai quando a foto é mais ampla? | H3 |
| A confiança dele é honesta fora do estúdio? | ECE + curva risco-cobertura |
| O ciclo 4 é mesmo melhor que o ciclo 3? | McNemar pareado |

---

## Parte 5 — Depois da coleta, o que ainda falta no TCC-II

1. **Decidir produção**: ciclo 3 ou ciclo 4, agora com dado de campo na mesa.
   Para voltar ao ciclo 3: copiar `backend/weights/ciclo3/*` para
   `backend/weights/`.
2. **ViT como experimento exploratório** — comparação de arquiteturas (entra no
   Q1), treinado no mesmo dataset final. Alinhar com o Prof. Leanderson.
   Não precisa de Colab Pro: ViT-S/DeiT-S cabe na RTX 3050 com AMP, e o Kaggle dá
   GPU grátis.
3. **Texto**: metodologia do ciclo 4 (§10.9 já escrito, é só adaptar), a
   ressalva de que o espinafre é *Basella alba* (Malabar) e não *Spinacia
   oleracea*, e o downy mildew contado como **49 fotos originais**, não 1.142.
4. **Grad-CAM**: a visualização atual tem um defeito (o mapa de calor foca na
   legenda de cores, não na folha). Consertar antes de usar como evidência.

---

## Se algo der errado

| Sintoma | O que fazer |
|---|---|
| Muitas fotos voltam `not_a_leaf` | Reenquadre mais perto; se persistir, baixar `min_vegetation_fraction` |
| Muitas voltam `inconclusive` | É comportamento honesto, não bug — anote e siga; a taxa de abstenção é um resultado |
| Acurácia de campo baixa (50-70%) | **É o resultado esperado** na literatura para esse salto de domínio. Não é fracasso, é a medição |
| Não consegue 15 fotos de uma cultura | Registre quantas conseguiu; menos culturas bem cobertas > mais mal cobertas |

---

## Documentos relacionados

| Preciso de... | Ler |
|---|---|
| Detalhe técnico de cada ciclo, inclusive o 4 | `docs/CORRECOES_METODOLOGICAS.md` §10.9 |
| A história completa, para a banca | `docs/HISTORICO_COMPLETO_TCC.md` |
| Detalhe de como fotografar/anotar/refinar | `docs/GUIA_TESTE_DE_CAMPO.md` |
| Metas da campanha multi-espécie | `docs/ROTEIRO_CAMPO_TCC2_MULTIESPECIE.md` |
| Estado do projeto e checklist | `PROXIMOS_PASSOS_TCC2.md` |
| Como preencher a planilha | `backend/campo/LEIA-ME_metadados.md` |
