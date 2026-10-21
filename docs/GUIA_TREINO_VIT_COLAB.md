# Guia — treino do ViT-B/16 no Google Colab Pro

### Criado em 12/10/2026 · experimento de comparação de arquiteturas (Q1 do TCC-II)

> **Situação em 13/10/2026: o treino foi feito.** Resultados em
> `PROXIMOS_PASSOS_TCC2.md` (seção de 13/10) e
> `backend/results/arquiteturas/comparacao_arquiteturas.md`.
>
> **O caminho usado não foi o deste guia.** Em vez de subir os zips para o Drive e
> rodar o notebook, tudo foi automatizado pela CLI oficial do Colab
> (`google-colab-cli`, no WSL Ubuntu): `colab/cli/orq.sh` cria o runtime L4, envia os
> zips direto do PC, confere SHA-256, treina, baixa os resultados e desliga a
> máquina. Dispensa os 13 GB livres no Drive. Para repetir, no PowerShell, com o PC
> impedido de suspender:
>
> ```powershell
> wsl.exe -d Ubuntu -- bash /home/sejodrope/fv/orq.sh
> ```
>
> O roteiro abaixo (Drive + notebook) continua válido como alternativa manual.
>
> **Tempos e custos medidos (L4):** envio dos 11,4 GB em 46 min; ~12 min por época,
> 6,1 h para as 30 épocas; 1,54 unidade de computação por hora, ~11 unidades na
> rodada completa. A estimativa de "3 a 6 min por época na L4" mais abaixo estava
> otimista.

O ViT-B/16 entra como quarta arquitetura, treinado no **mesmo dataset e no mesmo
split** das três CNNs do ciclo 4. O objetivo é responder se um transformer muda o
quadro no teste intra-fonte, no lote de espinafre e nas 13 fotos de campo.

Tudo o que o guia pede já está no repositório:

| Arquivo | Para quê |
|---|---|
| `backend/empacota_colab.py` | Junta dataset e código em 4 zips |
| `colab/FitoVision_ViT_Colab.ipynb` | Notebook que roda no Colab |
| `backend/run_vit.py` | Treino → calibração → avaliação → campo, retomável |
| `backend/compara_arquiteturas.py` | McNemar pareado entre as quatro arquiteturas |

---

## O que é igual às CNNs e o que muda

| | CNNs (ciclo 4) | ViT-B/16 |
|---|---|---|
| Dados, split, transformações | — | idênticos |
| Pesos iniciais | ImageNet-1k | ImageNet-1k |
| Épocas / batch / paciência | 30 / 64 / 10 | 30 / 64 / 10 |
| Loss, amostragem, critério de seleção | entropia cruzada com label smoothing 0,1; amostragem balanceada; F1 macro na validação | idênticos |
| Calibração e limiar de abstenção | temperature scaling; τ ≥ 0,85 | idênticos |
| **Taxa de aprendizado** | 1e-3 | **5e-5** |
| **Weight decay** | 1e-4 | **0,05** |
| **Aquecimento do LR** | não | **2 épocas** |
| **Clipping do gradiente** | não | **1,0** |

As quatro linhas em negrito precisam ser declaradas no artigo. A receita acima é a
usual para fine-tuning de ViT (LR baixo, aquecimento, weight decay maior); a das
CNNs, com LR de 1e-3, não é adequada a um transformer. Isso significa que a comparação é entre "cada arquitetura com
uma receita razoável para ela", e não entre arquiteturas com hiperparâmetros
idênticos.

---

## Antes de começar

- **Colab Pro** ativo.
- **~13 GB livres no Google Drive**: 11,4 GB dos zips, mais ~1,4 GB durante o treino
  (o melhor modelo e o checkpoint de retomada). A conta gratuita do Drive tem 15 GB
  no total, somando Gmail e Fotos; se não couber, veja "Se o Drive não tiver espaço"
  no fim do guia. Confira
  em <https://drive.google.com/settings/storage> antes de subir.
- **~12 GB livres no PC** para gerar os zips (já gerados em 12/10/2026).

---

## Passo 1 — Empacotar (no PC)

```powershell
cd backend
.venv\Scripts\python.exe empacota_colab.py
```

Gera a pasta `colab_export/` na raiz do projeto:

```
fitovision_dados_train.zip   8,25 GB
fitovision_dados_val.zip     1,70 GB
fitovision_dados_test.zip    1,46 GB
fitovision_codigo.zip        < 1 MB
FitoVision_ViT_Colab.ipynb
```

Leva cerca de 30 minutos. **Já foi rodado em 12/10/2026**: a pasta está pronta.

As imagens vão como estão, sem redimensionar. Se depois você mexer em algum script,
basta `empacota_colab.py --only codigo` e subir de novo só o zip pequeno.

## Passo 2 — Subir para o Drive

Crie a pasta **`FitoVision_Colab`** na raiz do "Meu Drive" e envie os cinco arquivos
para dentro dela. É a etapa mais demorada e depende só da sua internet (11,4 GB a
20 Mbps de upload levam cerca de 1h20). O Google Drive para computador retoma o
envio se a conexão cair; pelo navegador também funciona, com a aba aberta.

## Passo 3 — Abrir o notebook

1. No Drive, dê dois cliques em `FitoVision_ViT_Colab.ipynb` → **Abrir com Google
   Colaboratory**.
2. **Ambiente de execução → Alterar o tipo de ambiente de execução**: acelerador
   **GPU** (L4 ou A100 se houver; T4 serve) e **RAM alta**. A RAM alta importa mais
   pelas CPUs extras, que decodificam as imagens.
3. Confira o saldo em **Recursos** (ícone de RAM/disco no canto superior direito).

## Passo 4 — Rodar

**Ambiente de execução → Executar tudo.** O notebook pede acesso ao Drive, copia e
extrai os zips (5 a 15 min), e chama o `run_vit.py`.

O que você vai ver na célula 5:

```
0/5  Conferência do dataset        ← tem que terminar com "Confere com o manifesto"
1/5  Treino vit_b_16 (30 epochs...)
  Epoch   1 | loss=... acc=... | val_loss=... acc=... bal_acc=... f1=... *
  Epoch   2 | ...
```

Cada linha `Epoch` é uma época concluída e `*` marca um novo melhor modelo.

**Quanto demora.** Não medi no Colab; a ordem de grandeza esperada para 124 mil
imagens por época é de 10 a 15 min por época na T4 e de 3 a 6 min na L4/A100, ou
seja, de 2 a 8 horas no total (menos se a parada antecipada disparar). A barra de
progresso da primeira época dá a estimativa real em um minuto. Se ela indicar mais
de 20 min por época, confira se a RAM alta está ligada.

**Sinais de que algo está errado:**

- `bal_acc` da validação parado em ~0,50 depois da época 3: o modelo não está
  aprendendo. Não deixe rodar as 30 épocas; me mostre o log.
- `ABORTADO — o dataset não confere com o manifesto`: algum zip subiu incompleto.
  Compare o tamanho dos arquivos no Drive com os de `colab_export/` e reenvie.
- `CUDA out of memory`: não deve acontecer com 16 GB ou mais. Se acontecer, não
  reduza o batch por conta própria (muda a comparação); reinicie o ambiente e rode
  de novo.

## Passo 5 — Se a sessão cair

Reconecte e rode **Executar tudo** de novo. Os zips já extraídos são refeitos (o
disco local do Colab é apagado quando a sessão termina), mas o treino continua da
última época concluída: os pesos, o checkpoint e os logs ficam no Drive, em
`FitoVision_Colab/vit/`. Aparece a linha `Retomando da época N`.

Para reduzir o risco: deixe a aba do Colab aberta e o computador sem suspender.

## Passo 6 — Trazer de volta para o PC

A última célula gera `FitoVision_Colab/vit_resultados.zip` (~350 MB). Baixe e
extraia **dentro de `backend/`**. Ficam:

```
backend/weights/vit/vit_b_16_binary.pth
backend/weights/vit/vit_b_16_binary_calibration.json
backend/results/vit/...
backend/logs/vit/...
```

Depois, no PC:

```powershell
cd backend

# 1. McNemar pareado entre as quatro arquiteturas (prevê o ViT na RTX 3050 e compara)
.venv\Scripts\python.exe compara_arquiteturas.py

# 2. Latência do ViT na mesma GPU das CNNs (a do Colab não é comparável)
.venv\Scripts\python.exe evaluate.py --test-split logs\test_split_binary_original.json --binary --weights-dir weights\vit --results-dir results\vit\local_teste_original --workers 2
```

O primeiro comando escreve `results/arquiteturas/comparacao_arquiteturas.md`, com as
tabelas prontas. Na RTX 3050 ele leva cerca de 20 minutos para o ViT; não deixe o PC
suspender no meio. Desligue o ambiente do Colab ao terminar (**Ambiente de execução →
Desconectar e excluir**), para não gastar unidades com a GPU parada.

---

## O que vai para o artigo

| Número | Onde está | Onde entra |
|---|---|---|
| Acurácia balanceada, IC 95%, FN, ECE no teste original | `comparacao_arquiteturas.md`, recorte `original` | Tabela 2 (linha do ViT) |
| McNemar ViT × cada CNN, com Holm | idem | Seção 4, parágrafo do Q1 |
| Acurácia balanceada no espinafre (N = 484) | recorte `espinafre` | Seção 4, parágrafo da fonte nova |
| Erros interceptados pela abstenção | coluna "Interceptados" | Seção 4, parágrafo do Q2 |
| 13 fotos de campo | `results/vit/campo_resultados_vit.csv` | Tabela 3, como linha à parte |
| Latência | `results/vit/local_teste_original/metrics_comparison.csv` | Tabela 2 |
| Tempo de treino e melhor época | `logs/vit/vit_b_16_binary_history.json` | Seção 3.3 |

### Como ler o resultado (e o que não concluir)

- **O teste é intra-fonte.** Se o ViT passar de 98,5% lá, isso diz que ele também
  aprende as seis fontes; não diz nada sobre o campo. O número que interessa mais
  é o do espinafre, mas atenção: no ciclo 4 o espinafre **está no treino**. A
  comparação "fonte nunca vista" (75,4%) foi feita com pesos do ciclo 3 e não se
  repete com este ViT.
- **Uma execução, uma semente.** O McNemar diz se dois modelos acertam imagens
  diferentes neste teste. Não diz que a arquitetura é melhor em geral: para isso
  seriam precisas várias sementes. Escreva "neste conjunto de teste".
- **Hiperparâmetros diferentes.** Declare a tabela da primeira seção.
- **Campo, N = 13.** Qualquer diferença entre ViT e CNN nessas fotos tem p alto.
  Entra na Tabela 3 só como registro.
- **O ViT fica congelado** como os outros modelos do ciclo 4: nada de ajustar
  hiperparâmetros olhando o teste. Se o treino falhar e for preciso mudar a
  receita, a escolha é feita pela validação e a mudança é relatada.

---

## Se o Drive não tiver espaço

Depois de extraídos no Colab, os zips de dados não são mais lidos naquela sessão,
mas são necessários de novo se a sessão cair. Opções, da mais simples para a mais
trabalhosa: liberar espaço no Drive; usar outra conta Google só para isto (mude
`DRIVE` na célula 2 se a pasta tiver outro nome); ou assinar o plano de 100 GB do
Google One pelo mês do experimento.

## Alternativa sem Colab

O mesmo comando roda no PC (`.venv\Scripts\python.exe run_vit.py`). O ViT-B/16 com
batch 64 pode não caber nos 6 GB da RTX 3050, e reduzir o batch muda a receita;
por isso o Colab é o caminho recomendado.
