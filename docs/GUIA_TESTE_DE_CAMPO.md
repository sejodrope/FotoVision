# Guia de Teste de Campo — FitoVision

### TCC · UNIVILLE · Como fotografar, testar, anotar e refinar o modelo

> Este guia serve para produzir o **conjunto de teste de campo** previsto na §7 do
> `CORRECOES_METODOLOGICAS.md` — o experimento que mais valor acrescenta ao TCC-II.
> O objetivo não é "provar que o modelo é bom". É **medir honestamente** o quanto ele
> cai fora do domínio de treino (fotos de estúdio → fotos de quintal) e transformar
> esse número numa contribuição do trabalho.

---

## 0. Antes de começar — o que esperar

O modelo foi treinado quase só com **PlantVillage**: folha única, fundo neutro, luz
controlada. Uma foto de quintal tem terra, sombra, várias folhas, luz natural. São
distribuições diferentes. A literatura aponta **30–60% de acerto** nesse salto de
domínio. **Um resultado ruim aqui é o resultado esperado e é um dado, não um bug.**

Três respostas possíveis do sistema, e o que cada uma significa:

| Resposta | O que quer dizer | É erro? |
|----------|------------------|---------|
| `healthy` / `anomalous` | Diagnóstico com confiança acima do limiar | Pode acertar ou errar — anote |
| `inconclusive` | Confiança abaixo do limiar → o sistema abstém-se | **Não. É o comportamento honesto novo.** |
| `not_a_leaf` | A guarda de vegetação (ExG) não viu folha suficiente | Não — reenquadre mais perto |

---

## 1. Como tirar as fotos

Objetivo: cobrir **casos fáceis e difíceis de propósito**, para o teste medir o
limite real e não só o melhor caso.

### Regras básicas (para toda foto)
- **Foco nítido** na folha. Toque na tela para focar antes de disparar.
- **Luz natural difusa** de dia. Evite sol direto batendo na folha (estoura o branco)
  e evite sombra escura demais.
- **Uma folha em destaque** por foto, mesmo que haja outras ao fundo.
- **Sem flash** — distorce a cor, e cor é exatamente o sinal da doença.
- Formato paisagem ou retrato, tanto faz; o sistema redimensiona.

### Varie deliberadamente (para o teste ter valor)
Tire fotos em **3 distâncias** de algumas folhas:
1. **Close** — folha preenche o quadro (condição mais parecida com o treino)
2. **Média** — folha + um pouco de contexto (haste, vaso)
3. **Ampla** — a planta inteira com fundo de quintal (condição mais difícil)

E cubra estes cenários:
- Folhas **claramente saudáveis** (verdes, íntegras)
- Folhas com **problema visível** (manchas, amarelão/clorose, furos, míldio, oídio,
  pontas secas)
- **2–3 fotos "pega-ratão"** de propósito: uma foto que **não é folha** (parede, chão,
  sua mão) para confirmar que o `not_a_leaf` dispara.

### Meta de quantidade
Para o TCC-II o alvo é **100–300 fotos**. Para um primeiro teste exploratório,
**20–40 fotos** já dão um sinal claro. Tente equilibrar healthy vs anomalous
(ex.: 15 de cada) — senão a acurácia fica enganosa.

### Organização dos ficheiros (importante para medir depois)
Antes de testar, **você já sabe a verdade de cada foto** (viu a folha). Guarde essa
verdade no nome ou na pasta:

```
campo/
  healthy/
    IMG_001.jpg
    IMG_002.jpg
  anomalous/
    IMG_010.jpg
  nao_folha/
    IMG_020.jpg   ← parede, mão, etc.
```

Essa é a sua "gabarito" (ground truth). Sem ela não dá para calcular acurácia.

---

## 2. Como fazer os testes

Duas formas — escolha uma.

### Opção A — pela interface (rápido, visual)
1. Abra **http://localhost:5173**
2. Arraste uma foto de cada vez
3. Para cada uma, **anote na planilha** (seção 3): o que o sistema respondeu, a
   confiança, e se bateu com a verdade que você já conhecia.

### Opção B — em lote, por script (recomendado para 20+ fotos)
Peça para eu criar um script `backend/test_campo.py` que:
- percorre a pasta `campo/<label>/`,
- chama o modelo em cada foto,
- e gera automaticamente `results/campo_resultados.csv` + a **matriz de confusão**
  e a **acurácia balanceada de campo**.

Isso elimina a digitação manual e produz direto os números para o TCC. **Quando
chegar a hora, me peça "cria o script de teste de campo".**

### Cuidado técnico
- Confirme que a API responde: abrir **http://127.0.0.1:8000/docs** deve carregar.
- Se o servidor estiver em CPU (durante o treino), cada foto leva ~1–2s. Normal.

---

## 3. Como anotar (planilha de resultados)

Uma linha por foto. Copie esta tabela para uma planilha (Excel/Sheets):

| # | Ficheiro | Verdade (você) | Resposta do sistema | Confiança | Acertou? | Distância | Observações |
|---|----------|----------------|---------------------|-----------|----------|-----------|-------------|
| 1 | IMG_001  | healthy        | healthy             | 0,82      | ✅       | close     | folha verde limpa |
| 2 | IMG_010  | anomalous      | inconclusive        | 0,61      | —        | média     | absteve-se |
| 3 | IMG_020  | nao_folha      | not_a_leaf          | —         | ✅       | ampla     | foto do chão |

**Colunas que importam depois:**
- **Verdade** — o que você sabe que é (o gabarito).
- **Resposta + Confiança** — o que o modelo disse.
- **Acertou?** — ✅ / ❌ / — (— para `inconclusive`/`not_a_leaf`, que não são
  acerto nem erro de classe).
- **Distância** — para descobrir se close acerta mais que ampla (quase certo que sim).

**Números para calcular no fim:**
- **Acurácia de campo** = acertos ÷ (fotos que receberam diagnóstico healthy/anomalous)
- **Taxa de abstenção** = `inconclusive` ÷ total (quanto o sistema "teve a humildade" de não cravar)
- **Taxa de rejeição** = `not_a_leaf` ÷ total
- Comparar com a **acurácia balanceada de teste** (~97–98% no domínio PlantVillage).
  **A diferença entre as duas É o resultado do experimento.**

---

## 4. Como corrigir / refinar o modelo

Depois de ver os erros, o refino segue uma ordem — do mais barato ao mais caro.
**Não pule para o retreino sem antes olhar os erros um a um.**

### Passo 1 — Diagnosticar o tipo de erro (grátis, só olhar)
Separe os erros em categorias:
- **Erro de enquadramento** (`not_a_leaf` numa folha real) → problema da guarda ExG,
  não do modelo.
- **Abstenção excessiva** (muitos `inconclusive` em fotos óbvias) → limiar alto demais.
- **Erro de classe com confiança alta** (diz `healthy` numa folha doente) → é o gap
  de domínio de verdade; só se resolve com dados de campo no treino.

### Passo 2 — Ajustes baratos (minutos, sem retreinar)
- **Limiar de abstenção** (`confidence_threshold` em `app/config.py`, hoje 0,70):
  se o sistema se abstém demais, baixar; se crava erros demais, subir.
- **Guarda de vegetação** (`min_vegetation_fraction`, hoje 0,10): se rejeita folhas
  reais, baixar; se aceita paredes, subir.
- Re-testar as mesmas fotos e ver se melhora. **Anote o antes/depois.**

### Passo 3 — Recalibração (rápido, se a confiança estiver mentindo)
Se as confianças não batem com a realidade (ex.: erra muito a 90%), rodar de novo:
```
python calibrate.py --data ./data --binary --model efficientnet_b0
```
(Isto usa o val do PlantVillage; ajuda pouco no gap de campo, mas mantém a confiança honesta.)

### Passo 4 — Data augmentation dirigida ao campo (médio esforço)
Adicionar ao treino augmentations que **imitam** condições de campo: fundos variados,
oclusão parcial, variação forte de iluminação. Ajuda a fechar parte do gap **sem**
coletar dados novos. Requer retreino (~4h por modelo).

### Passo 5 — Fine-tuning com fotos de campo (o que realmente resolve)
A solução de fundo, segundo o próprio doc de correções: **incorporar fotos de campo
rotuladas ao treino**. Mesmo 100–300 fotos suas, adicionadas ao dataset, deslocam o
modelo na direção do domínio real. É o núcleo do TCC-II.

> Regra de ouro do refino: **as fotos de campo do TESTE nunca entram no TREINO.**
> Se você retreinar com elas, precisa de fotos de campo NOVAS para testar de novo —
> senão volta o vazamento que acabámos de corrigir.

---

## 5. Como registar os resultados (para o TCC)

Guarde, em `docs/` ou numa pasta `experimentos/`:
1. A **planilha** preenchida (seção 3).
2. A **matriz de confusão de campo** (o script gera; ou monte à mão).
3. Uma tabela-resumo lado a lado — **este é o quadro que vai para a defesa**:

| Conjunto | Acurácia balanceada | Abstenção | Nº fotos |
|----------|--------------------|-----------|---------|
| Teste (domínio PlantVillage) | ~97–98% *(a confirmar no passo 9)* | — | 21.520 |
| **Campo (quintal, telemóvel)** | *a medir* | *a medir* | *20–300* |

4. Um parágrafo de **análise dos erros**: que tipo de foto falha mais (distância?
   tipo de doença? luz?). Isso vira a seção de discussão do TCC.
5. Antes/depois de cada ajuste que você fizer (limiar, augmentation, fine-tuning),
   com o número que mudou.

> **A narrativa vencedora do TCC:** "corrigimos o vazamento (57,3% → 0,18%), o que
> baixou a métrica de 99,01% falso para ~97% honesto no domínio; depois medimos o gap
> de domínio em campo (X%) e mostramos como o fine-tuning com N fotos reais o reduziu
> para Y%." Isso é ciência. Um 99% que morre na primeira foto real, não.

---

## 6. Checklist rápido

- [ ] Pipeline de treino terminou (os 3 modelos + calibração + avaliação)
- [ ] `results/metrics_comparison.csv` conferido (acurácia balanceada de teste)
- [ ] uvicorn reiniciado **na GPU** (sem `CUDA_VISIBLE_DEVICES=-1`) para inferência rápida
- [ ] 20–40 fotos tiradas, variando distância e sanidade, com 2–3 "pega-ratão"
- [ ] Fotos organizadas em `campo/<label>/` (o gabarito)
- [ ] Script de teste de campo criado (me pedir) OU testes manuais pela UI
- [ ] Planilha preenchida
- [ ] Acurácia de campo vs teste calculada e registada
- [ ] Erros analisados por tipo
- [ ] Ajustes testados com antes/depois anotado
