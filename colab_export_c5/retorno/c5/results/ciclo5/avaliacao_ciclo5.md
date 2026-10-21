# Ciclo 5 — avaliação por subconjunto

Gerado por `backend/avalia_ciclo5.py`. Acurácia balanceada, IC de 95% por bootstrap
estratificado.

## efficientnet_b0 (T = 0.655, τ = 0.85)

| Subconjunto | N | Acur. bal. | IC 95% | FN | FP | ECE | Interceptados | Abstenção |
|---|---|---|---|---|---|---|---|---|
| teste_inteiro | 18413 | 0.9742 | [0.9714; 0.9770] | 276 | 142 | 0.0059 | 115 de 418 | 1.4% |
| coletanea_lavoura | 4530 | 0.8883 | [0.8725; 0.9034] | 229 | 96 | 0.0451 | 84 de 325 | 3.9% |
| coletanea_demais | 10061 | 0.9968 | [0.9955; 0.9980] | 21 | 10 | 0.0098 | 9 de 31 | 0.2% |
| plantvillage | 2659 | 0.9980 | [0.9954; 0.9998] | 2 | 2 | 0.0101 | 1 de 4 | 0.0% |
| alface_uma_por_foto | 176 | 0.9667 | [0.9000; 1.0000] | 1 | 0 | 0.0159 | 1 de 1 | 0.6% |
| espinafre | 400 | 0.8660 | [0.8309; 0.8998] | 15 | 34 | 0.0651 | 18 de 49 | 12.0% |
| antes_nao_reconhecidas | 414 | 0.9807 | — | 8 | 0 | 0.0076 | 2 de 8 | 1.2% |
| mildio_recortes_descritivo | 173 | 1.0000 | — | 0 | 0 | 0.0107 | 0 de 0 | 0.0% |
| espinafre_fonte_retida | 574 | 0.8792 | [0.8439; 0.9120] | 28 | 24 | 0.0438 | 26 de 52 | 10.5% |
| comparavel_com_ciclo4 | 17250 | 0.9774 | [0.9747; 0.9801] | 252 | 108 | 0.0052 | 94 de 360 | 1.2% |

Ciclo 4 × ciclo 5 nas 17250 imagens de teste que não mudaram de partição: só o ciclo 4 acerta 139, só o ciclo 5 acerta 106; McNemar exato, p = 0.0407.

## vit_b_16 (T = 0.676, τ = 0.85)

| Subconjunto | N | Acur. bal. | IC 95% | FN | FP | ECE | Interceptados | Abstenção |
|---|---|---|---|---|---|---|---|---|
| teste_inteiro | 18413 | 0.9691 | [0.9658; 0.9723] | 146 | 230 | 0.0047 | 45 de 376 | 0.6% |
| coletanea_lavoura | 4530 | 0.8446 | [0.8261; 0.8628] | 124 | 162 | 0.0416 | 35 de 286 | 1.9% |
| coletanea_demais | 10061 | 0.9955 | [0.9938; 0.9971] | 8 | 23 | 0.0115 | 4 de 31 | 0.1% |
| plantvillage | 2659 | 0.9992 | [0.9977; 1.0000] | 0 | 1 | 0.0125 | 0 de 1 | 0.0% |
| alface_uma_por_foto | 176 | 1.0000 | [1.0000; 1.0000] | 0 | 0 | 0.0131 | 0 de 0 | 0.0% |
| espinafre | 400 | 0.8447 | [0.8078; 0.8801] | 11 | 44 | 0.1027 | 6 de 55 | 4.8% |
| antes_nao_reconhecidas | 414 | 0.9928 | — | 3 | 0 | 0.0100 | 0 de 3 | 0.0% |
| mildio_recortes_descritivo | 173 | 1.0000 | — | 0 | 0 | 0.0125 | 0 de 0 | 0.0% |
| espinafre_fonte_retida | 574 | 0.8584 | [0.8200; 0.8949] | 17 | 33 | 0.0612 | 7 de 50 | 2.6% |
| comparavel_com_ciclo4 | 17250 | 0.9726 | [0.9692; 0.9758] | 132 | 186 | 0.0031 | 39 de 318 | 0.5% |

Ciclo 4 × ciclo 5 nas 17250 imagens de teste que não mudaram de partição: só o ciclo 4 acerta 115, só o ciclo 5 acerta 109; McNemar exato, p = 0.738.

## mobilenet_v2 (T = 0.647, τ = 0.85)

| Subconjunto | N | Acur. bal. | IC 95% | FN | FP | ECE | Interceptados | Abstenção |
|---|---|---|---|---|---|---|---|---|
| teste_inteiro | 18413 | 0.9690 | [0.9661; 0.9718] | 466 | 128 | 0.0098 | 260 de 594 | 3.1% |
| coletanea_lavoura | 4530 | 0.9009 | [0.8877; 0.9133] | 395 | 57 | 0.0537 | 190 de 452 | 9.0% |
| coletanea_demais | 10061 | 0.9946 | [0.9931; 0.9961] | 45 | 13 | 0.0096 | 23 de 58 | 0.6% |
| plantvillage | 2659 | 0.9992 | [0.9977; 1.0000] | 0 | 1 | 0.0100 | 0 de 1 | 0.0% |
| alface_uma_por_foto | 176 | 1.0000 | [1.0000; 1.0000] | 0 | 0 | 0.0104 | 0 de 0 | 0.0% |
| espinafre | 400 | 0.7950 | [0.7557; 0.8340] | 16 | 57 | 0.0927 | 40 de 73 | 21.2% |
| antes_nao_reconhecidas | 414 | 0.9783 | — | 9 | 0 | 0.0139 | 6 de 9 | 3.6% |
| mildio_recortes_descritivo | 173 | 0.9942 | — | 1 | 0 | 0.0141 | 1 de 1 | 0.6% |
| espinafre_fonte_retida | 574 | 0.8843 | [0.8487; 0.9174] | 17 | 26 | 0.0165 | 24 de 43 | 14.5% |
| comparavel_com_ciclo4 | 17250 | 0.9747 | [0.9721; 0.9771] | 440 | 71 | 0.0106 | 213 de 511 | 2.7% |

Ciclo 4 × ciclo 5 nas 17250 imagens de teste que não mudaram de partição: só o ciclo 4 acerta 248, só o ciclo 5 acerta 125; McNemar exato, p = 1.84e-10.

## resnet50 (T = 0.637, τ = 0.85)

| Subconjunto | N | Acur. bal. | IC 95% | FN | FP | ECE | Interceptados | Abstenção |
|---|---|---|---|---|---|---|---|---|
| teste_inteiro | 18413 | 0.9691 | [0.9663; 0.9720] | 463 | 128 | 0.0107 | 251 de 591 | 3.0% |
| coletanea_lavoura | 4530 | 0.8957 | [0.8820; 0.9090] | 388 | 64 | 0.0545 | 188 de 452 | 9.0% |
| coletanea_demais | 10061 | 0.9937 | [0.9920; 0.9954] | 43 | 19 | 0.0089 | 27 de 62 | 0.6% |
| plantvillage | 2659 | 0.9977 | [0.9951; 0.9998] | 3 | 2 | 0.0089 | 1 de 5 | 0.1% |
| alface_uma_por_foto | 176 | 1.0000 | [1.0000; 1.0000] | 0 | 0 | 0.0097 | 0 de 0 | 0.0% |
| espinafre | 400 | 0.8326 | [0.7950; 0.8682] | 18 | 43 | 0.0722 | 31 de 61 | 17.8% |
| antes_nao_reconhecidas | 414 | 0.9734 | — | 11 | 0 | 0.0121 | 4 de 11 | 1.9% |
| mildio_recortes_descritivo | 173 | 1.0000 | — | 0 | 0 | 0.0096 | 0 de 0 | 0.0% |
| espinafre_fonte_retida | 574 | 0.9006 | [0.8672; 0.9314] | 19 | 21 | 0.0306 | 21 de 40 | 9.8% |
| comparavel_com_ciclo4 | 17250 | 0.9732 | [0.9705; 0.9758] | 434 | 85 | 0.0098 | 216 de 519 | 2.7% |

Ciclo 4 × ciclo 5 nas 17250 imagens de teste que não mudaram de partição: só o ciclo 4 acerta 213, só o ciclo 5 acerta 101; McNemar exato, p = 2.44e-10.
