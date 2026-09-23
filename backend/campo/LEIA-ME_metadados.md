# Planilha de metadados das fotos de campo

Preencha **na hora da coleta** (ver `docs/ROTEIRO_CAMPO_TCC2_MULTIESPECIE.md`).
Sem ela, o `analise_campo.py` calcula tudo menos as hipóteses H2 e H3.

| Coluna | O que pôr | Obrigatório para |
|---|---|---|
| `ficheiro` | nome do arquivo, só o nome (`IMG_alface_010.jpg`) | tudo |
| `cultura` | alface, rucula, espinafre, acelga, couve | H1 (ou vem da pasta) |
| `verdade` | healthy, anomalous, nao_folha | conferência |
| `tipo_sintoma` | mancha, clorose, oídio/míldio, furo de inseto, murcha, ponta seca | descrição |
| `categoria_dano` | **cromatico** (cor: mancha, clorose, míldio) ou **estrutural** (forma: furo, murcha, ponta seca); use `misto` se tiver os dois | **H2** |
| `distancia` | **close**, **media** ou **ampla** | **H3** |
| `hora` | hora aproximada (luz do dia varia) | contexto |
| `notas` | qualquer dúvida do momento ("sintoma leve, conferir") | honestidade |

`categoria_dano` é o que testa a hipótese aberta do §10.7 do
`CORRECOES_METODOLOGICAS.md` — o modelo parecia pesar mais cor do que estrutura.
Para ela render, tente fotografar os dois tipos em quantidade parecida.
Fotos `healthy` deixam `tipo_sintoma` e `categoria_dano` em branco.

As 3 linhas de exemplo em `metadados_campo.csv` são para apagar.
