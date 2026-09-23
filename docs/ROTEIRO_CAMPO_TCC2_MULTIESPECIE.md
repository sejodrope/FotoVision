# Roteiro de Campo — Semana Multi-Espécie (TCC-II)

### Complementa `GUIA_TESTE_DE_CAMPO.md` (que explica o "como fotografar/testar/anotar")
### Este documento é o "plano de batalha" para a semana de bom tempo

---

## 0. Por que isto importa mais do que parece

Hoje o seu teste de campo tem **13 fotos, todas de alface**. Isso já foi
suficiente pra achar um bug real (§10.6 do `CORRECOES_METODOLOGICAS.md`), mas
é estatisticamente pequeno demais pra comparar versões de modelo (IC95% de
Wilson se sobrepõem entre os 3 ciclos) e **não diz nada sobre as outras 4
culturas do seu escopo** (rúcula, espinafre, acelga, couve) — o sistema nunca
foi testado em campo nelas.

Você validou nesta sessão que **datasets prontos não resolvem isso** — 6
datasets de alface do Roboflow eram 85-94% redundantes com o que você já
tinha, e mesmo o melhor achado extra (Malabar Spinach, Mendeley) é foto de
folha isolada em fundo branco, não campo real. **A única forma de fechar o
domain gap é você fotografar.** Esta semana de bom tempo vale mais para o
TCC do que qualquer busca de dataset.

---

> **23/09/2026**: o passo a passo prático desta campanha (com o estado do
> projeto depois do ciclo 4 e os comandos a rodar) está em
> [`GUIA_CAMPO_HORTAS_COMUNITARIAS.md`](GUIA_CAMPO_HORTAS_COMUNITARIAS.md).
> Este roteiro continua a ser a referência das metas por cultura.

---

## 1. Meta por cultura

| Cultura | Mínimo aceitável | Meta ideal | Se sobrar tempo |
|---|---|---|---|
| Alface | +7 novas (total ≥20) | +27 novas (total ≥40) | seguir até 60+ |
| Rúcula | 15 | 25 | 40 |
| Espinafre | 15 | 25 | 40 |
| Acelga | 15 | 25 | 40 |
| Couve | 15 | 25 | 40 |

**Por quê esses números:** 15 por cultura nova é o piso pra sair de "zero
dado" pra "primeiro sinal com algum valor estatístico" (ainda pouco, mas
já é infinitamente mais do que zero). 25 por cultura é a meta — dá pra
calcular um IC de Wilson menos largo que o de N=13 atual. Alface começa
"na frente" porque já tem 13; passar de 20 no total já muda a análise
atual (que está travada em N=13 há 3 ciclos).

**Se o tempo for curto, priorize NESTA ordem:**
1. **Uma cultura nova além de alface** (qualquer uma — rúcula, espinafre,
   acelga OU couve) até bater 15-25 fotos. Sair de "1 cultura testada" pra
   "2 culturas testadas" é o salto de maior valor pro TCC.
2. **Alface até 20+** (reforça o N que já está no meio do caminho).
3. Só depois, expandir para a 3ª, 4ª, 5ª cultura.

Não tente cobrir as 5 culturas "de raspão" (5-8 fotos cada) — um N tão
baixo por cultura não dá pra calcular nada com confiança. **Menos culturas
bem cobertas > mais culturas mal cobertas.**

---

## 2. Balanceamento saudável vs. anômala (por cultura)

Tente ~50/50 em cada cultura — um desbalanceamento forte infla a acurácia
aparente. Se estiver difícil achar folha doente de uma cultura específica
(o que é bom sinal pra sua horta, ruim pro teste), não force — anote quantas
consegue de cada tipo e ajuste a proporção da próxima cultura pra compensar.

Dentro de "anômala", varie o **tipo de sintoma** quando possível — mancha,
clorose (amarelão), oídio/míldio (pó branco), furo de inseto, murcha,
ponta seca. O histórico do projeto tem uma hipótese aberta ("o modelo pode
pesar mais cor que estrutura que furo/murcha") — quanto mais variado o
tipo de dano fotografado, mais essa hipótese pode ser testada de verdade.

---

## 3. Checklist técnico (resumo — detalhes completos no `GUIA_TESTE_DE_CAMPO.md` §1)

Para cada foto:
- [ ] Foco nítido na folha (toque na tela antes de disparar)
- [ ] Luz natural difusa, sem sol direto nem sombra forte
- [ ] Sem flash
- [ ] Uma folha em destaque (pode ter outras ao fundo)

Para cada folha "de interesse" (ex.: uma com sintoma claro), tire em
**3 distâncias**: close (folha preenche o quadro), média (folha + caule/vaso),
ampla (planta inteira + fundo de canteiro/quintal). Isso não precisa ser 3x
em TODAS as fotos — mas tente em pelo menos 20-30% de cada cultura, porque
é o que testa se o sistema aguenta a condição mais difícil (ampla).

**2-3 fotos "pega-ratão" por cultura** (não-folha: terra, mão, parede) —
confirma que o `not_a_leaf` continua disparando certo em todas as culturas,
não só em alface.

---

## 4. Organização de arquivos — nomenclatura por cultura

Adapte a estrutura do guia atual para incluir a cultura:

```
campo/
  alface/
    healthy/IMG_alface_001.jpg
    anomalous/IMG_alface_010.jpg
  rucula/
    healthy/IMG_rucula_001.jpg
    anomalous/IMG_rucula_010.jpg
  espinafre/
    healthy/...
    anomalous/...
  acelga/
    healthy/...
    anomalous/...
  couve/
    healthy/...
    anomalous/...
  nao_folha/
    IMG_020.jpg   ← pode misturar culturas, ou marcar qual é qual no nome
```

**Anote a verdade na hora**, não depois — no calor da coleta é fácil
esquecer se aquela folha específica tinha sintoma leve ou era só uma
sombra. Se tiver dúvida na hora, tire a foto mas marque num caderno/notas
do celular "IMG_alface_007 — incerto, conferir depois" em vez de decidir
tudo de memória à noite.

---

## 4.1 Planilha de metadados — preencha NA HORA (22/09/2026)

Modelo pronto em `backend/campo/metadados_campo.csv` (instruções em
`backend/campo/LEIA-ME_metadados.md`). Uma linha por foto:

`ficheiro, cultura, verdade, tipo_sintoma, categoria_dano, distancia, hora, notas`

Duas colunas fazem toda a diferença e **não dá para recuperar depois**:

- **`categoria_dano`**: `cromatico` (mancha, clorose, míldio — o sintoma é cor)
  ou `estrutural` (furo, murcha, ponta seca — o sintoma é forma). É o que testa
  a hipótese do §10.7 ("o modelo pode pesar mais cor que estrutura"). Para ela
  render, fotografe os dois tipos em quantidade parecida.
- **`distancia`**: `close`, `media` ou `ampla`. É o que testa se o sistema
  aguenta a condição mais difícil.

Sem essa planilha, o `analise_campo.py` calcula acurácia, ICs e comparação entre
versões, mas **pula H2 e H3** — e essas duas hipóteses estão prometidas na
Parcial I.

---

## 5. Logística da semana (se tiver ~5-7 dias de bom tempo)

Não deixe tudo pro último dia — clima muda, e se chover de novo você quer
ter algo capturado. Sugestão de ritmo (ajuste à sua disponibilidade real):

| Dia | Foco |
|---|---|
| Dia 1 | Cultura nova prioritária (a que for mais acessível) — bater a meta mínima (15) |
| Dia 2 | Continuar a mesma cultura até a meta ideal (25) + começar 2ª cultura nova |
| Dia 3 | 2ª cultura nova até a meta |
| Dia 4 | Alface — completar até 20-40 |
| Dia 5 | 3ª/4ª cultura nova, se acessível |
| Dias 6-7 (reserva) | Cobrir o que faltou, revisar fotos tiradas (nitidez, enquadramento), descartar as ruins e retirar se sobrar tempo |

Se você só tiver acesso físico a 2-3 das 5 culturas (nem toda horta tem
todas), não force as outras — registre no relatório final quais culturas
ficaram sem cobertura de campo, isso também é uma limitação honesta pro
TCC (assim como já foi feito com a ausência de datasets Kaggle pra
rúcula/acelga/couve).

## 6. Ao voltar — o que eu faço com o material

Quando você trouxer as fotos:
1. Eu confirmo a organização das pastas (`campo/<cultura>/<label>/`).
2. Rodo (ou adapto) o `backend/test_campo.py` pra cada cultura separadamente
   — isso já existe e funciona (usado nos 3 ciclos anteriores).
3. Calculo acurácia balanceada de campo **por cultura**, não só geral —
   é isso que vai mostrar se o modelo generaliza igual pra todas ou se
   alface (única treinada com foco) está bem acima das outras.
4. Comparo com o domínio de teste (~98%) e com o N=13 anterior — agora com
   mais poder estatístico (IC de Wilson mais estreito).
5. Analiso erros por tipo de dano (cor vs. estrutura) e por distância
   (close vs. ampla) — testa as duas hipóteses já abertas no histórico.
   Tudo isso já está automatizado desde 22/09/2026 em
   `backend/analise_campo.py` (um comando; acurácia balanceada com IC por
   bootstrap, IC de Wilson, McNemar exacto + Holm entre versões, ECE,
   curva risco-cobertura, H1 Fisher/Freeman-Halton, H2 Fisher 2x2,
   H3 Cochran-Armitage):

   ```
   python test_campo.py --dir ./campo --out ./results/campo_resultados_v4.csv
   python analise_campo.py --csv results/campo_resultados_v3.csv:ciclo3 results/campo_resultados_v4.csv:ciclo4 --metadados campo/metadados_campo.csv
   ```
6. **Regra de ouro, sem exceção**: as fotos de campo do TESTE nunca entram
   no TREINO. Se decidirmos fazer fine-tuning com fotos de campo depois
   (§10.5 do guia), vamos precisar de um SEGUNDO lote de fotos novas pra
   isso — as desta semana ficam reservadas para medir, não para treinar.
7. Atualizamos juntos `GUIA_TESTE_DE_CAMPO.md` §6 (checklist) e o histórico
   do TCC com os números novos.

---

## 7. Resumo de uma linha

**Prioridade: sair de "1 cultura, N=13" para "pelo menos 2-3 culturas,
N=20-40 cada" — isso vale mais pro TCC do que qualquer dataset que
encontrarmos prontos.**
