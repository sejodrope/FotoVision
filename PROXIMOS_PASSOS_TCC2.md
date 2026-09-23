# FitoVision — Próximos Passos (TCC-II)

### Última atualização: 23/09/2026 · Ciclo 4 CONCLUÍDO; fotos de campo a partir de ~05/10

> **Leia este arquivo primeiro** quando voltar a trabalhar no projeto — ele
> resume onde paramos e o que fazer a seguir. Para a **coleta de fotos em hortas
> comunitárias**, o passo a passo prático está em
> [`docs/GUIA_CAMPO_HORTAS_COMUNITARIAS.md`](docs/GUIA_CAMPO_HORTAS_COMUNITARIAS.md). Os detalhes completos de cada
> etapa estão nos relatórios linkados abaixo.

---

## Estado em uma frase

**O ciclo 4 está treinado, calibrado e avaliado (23/09/2026): +9 a +14 pontos de
acurácia balanceada nas culturas novas, com p de Holm ≈ 10⁻¹³ — o primeiro ganho
do projeto que sobrevive a um teste de significância — e sem perder nada no
domínio antigo. O que falta agora é só campo: fotos em hortas comunitárias, a
partir de ~05/10, para decidir qual ciclo vai para produção.**

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
  alinhar com o Prof. Leanderson (que já tinha gostado da ideia). Não precisa
  de Colab Pro: ViT-S/DeiT-S cabe na RTX 3050 com AMP, e o Kaggle dá GPU grátis.
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
- [ ] Decisão ciclo 3 vs 4 em produção (só com as fotos novas)
- [ ] ViT exploratório (depois do ciclo 4; alinhar com orientador)
- [ ] Ressalva de espécie do Malabar Spinach registrada no texto do TCC-II
