# Fotos de campo novas — onde pôr e o que anotar

As 13 fotos do TCC-I continuam em `backend/campo/` e são o conjunto histórico. **O lote
novo fica aqui**, separado, para não se misturar com elas.

## Onde pôr cada foto

```
backend/campo_novo/
  alface/healthy/      alface/anomalous/
  rucula/healthy/      rucula/anomalous/        (crie a pasta da cultura que fotografar)
  nao_folha/                                    (terra, mão, parede: 2 ou 3 por horta)
  metadados_campo.csv
```

A pasta é o gabarito: decida `healthy` ou `anomalous` **na horta**, olhando a planta, e
não depois, olhando a foto.

## Cinco regras que as 13 fotos antigas não cumpriram

1. **Arquivo original do celular.** Copie por cabo ou pela nuvem do próprio celular.
   Não mande por WhatsApp, não tire captura de tela, não edite. O conferidor avisa quando
   a foto vem sem EXIF de câmera. Se o celular grava em HEIC, exporte em JPG.
2. **Nome único para cada foto.** O nome que o celular dá (`IMG_20261014_093012.jpg`)
   serve. O mesmo nome não pode aparecer em duas pastas.
3. **Uma linha na planilha para cada foto**, com a **horta**. A separação entre o
   conjunto de avaliação e o de adaptação é por horta: sem ela não dá para separar.
4. **Em dúvida, escreva a dúvida** na coluna `notas` ("sintoma leve, conferir"). Foto de
   gabarito duvidoso **não é descartada**: o conferidor lista todas, e o resultado é
   reportado com e sem elas (descartar as ambíguas deixa o resultado otimista, como no
   teste externo de Ergün et al., 2026).
5. **A mesma folha duas vezes conta uma vez.** Pode e deve fotografar a mesma planta de
   perto, de média distância e de longe (hipótese H3), mas anote na coluna `notas` que
   são da mesma planta.

## A planilha

| Coluna | O que pôr |
|---|---|
| `ficheiro` | nome do arquivo, só o nome |
| `horta` | um nome curto e fixo para cada horta (`horta_a`, `bom_retiro`...) |
| `cultura` | alface, rucula, espinafre, acelga, couve |
| `verdade` | healthy, anomalous ou nao_folha (tem de bater com a pasta) |
| `tipo_sintoma` | mancha, clorose, míldio, oídio, furo de inseto, murcha, ponta seca |
| `categoria_dano` | **cromatico** (cor), **estrutural** (forma) ou **misto** — só para as anômalas |
| `distancia` | **close** (a folha enche a tela), **media** (folha e caule) ou **ampla** (planta inteira e canteiro) |
| `hora` | hora aproximada |
| `notas` | dúvidas, "mesma planta da foto X", o que chamar a atenção |

As linhas que começam com `EXEMPLO_` são para apagar.

## O que fotografar, em uma frase por item

- **Meta:** pelo menos 15 fotos por cultura, o ideal é 25; metade saudável, metade anômala.
- **Duas hortas ou mais**, se der.
- **Dano estrutural também:** furo de inseto, murcha, ponta seca. O treino quase não tem.
- **Planta inteira e de perto.** O treino quase só tem close-up de folha de alface.
- **Sem flash**, foco na folha, luz natural.

## Depois, dois comandos

```powershell
cd backend
.venv\Scripts\python.exe confere_fotos_campo.py     # confere as fotos e a planilha
.venv\Scripts\python.exe roda_campo.py              # roda os modelos às cegas e analisa
```

O relatório sai em `backend/results/campo_novo/`. Nenhuma dessas fotos entra em treino
antes de o conjunto de avaliação estar separado e medido.
