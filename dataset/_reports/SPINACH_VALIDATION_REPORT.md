# Validação dos datasets de Malabar Spinach (Mendeley) — TCC-II FitoVision

Data: 31/08/2026
Origem: 2 datasets baixados do Mendeley Data (licença CC BY 4.0):
- `n56pn9fncw` v2 — "Malabar Spinach dataset for diseases classification"
  (DOI 10.17632/n56pn9fncw.2), só a versão "Original Dataset.zip" (sem
  aumentada) — 603 fotos.
- `sy69db2nz5` v2 — "IDDMSLD: Malabar Spinach Disease Detection Dataset"
  (DOI 10.17632/sy69db2nz5.2) — 3.006 fotos.

Nada foi apagado/sobrescrito. Trabalho em `dataset/_work/spinach_extracted/`
(zips extraídos) e `dataset/_work/spinach_classified/` (organizado em
saudavel/anomala). **Nenhum treinamento executado.**

---

## Resumo executivo

Diferente dos 6 datasets Roboflow de alface (sessão anterior), estes **dois
datasets de espinafre são úteis**: já vêm em ImageFolder nativo (sem bbox
pra recortar), resolução excelente (0% de imagens <224px, contra 79% do
pior caso anterior), praticamente sem sobreposição com o que você já tem
(2 colisões com o test — ambas confirmadas como **falso positivo** de hash)
e os dois datasets são **fontes independentes entre si** (só 5 grupos
cruzam os dois, de ~3.600 imagens). A ressalva: apesar da descrição dizer
"coletado em campo real", a fotografia em si é de **folha isolada sobre
fundo branco/papel** — não fecha o domain gap da forma que uma foto sua de
quintal fecharia, mas ainda adiciona uma espécie-alvo (espinafre) que hoje
tem zero representação e zero teste de campo no seu projeto.

**Recomendação: GO**, com uma ressalva de nomenclatura científica (ver
Fase 3) e ciente de que isso é dado-proxy de melhor qualidade, não
substituto de foto de campo.

---

## Fase 1 — Formato

Ambos já são **classificação nativa (ImageFolder)** — uma pasta por classe,
sem bounding box, sem conversão necessária.

| Dataset | Estrutura | Classes |
|---|---|---|
| `spinach_n56pn9fncw` | `Original Dataset/<classe>/` | `healthy`, `anthracnose_leaf_spot`, `straw_mite` |
| `spinach_sy69db2nz5` | `Malabar_Dataset/<classe(N)>/` | `Healthy-Leaf`, `Anthracnose`, `Bacterial-Spot`, `Downy-Mildew`, `Pest-Damage` |

---

## Fase 2 — Integridade e estatísticas

| Dataset | OK | Corrompidos | <224px | Resolução (mediana W×H) | Formato |
|---|---|---|---|---|---|
| `spinach_n56pn9fncw` | 603 | 3 | **0%** | 4640×3472 (altíssima) | JPEG |
| `spinach_sy69db2nz5` | 3.006 | 0 | **0%** | 900×1200 | JPEG (2.998) + MPO (8) |

Os "3 corrompidos" de `n56pn9fncw` são só `desktop.ini` (arquivo de sistema
do Windows que veio dentro do zip, não é imagem) — zero problema real de
integridade. Todas RGB. Resolução muito acima do necessário para 224×224
do ViT — nenhum problema de upscaling aqui, ao contrário do caso anterior.

**Balanceamento:**

| Dataset | saudavel | anomala | proporção |
|---|---|---|---|
| `spinach_n56pn9fncw` | 150 | 453 | 25% : 75% |
| `spinach_sy69db2nz5` | 1.399 | 1.607 | 47% : 53% |
| **Combinado** | **1.549** | **2.060** | **43% : 57%** |

O combinado é bem mais balanceado que os dados de alface do Roboflow
(24%:76%) e até mais balanceado que seu dataset atual (34%:66%).

---

## Fase 3 — Mapeamento de classes

| Dataset | Categoria original | Classe FitoVision |
|---|---|---|
| `n56pn9fncw` | `healthy` | `saudavel` |
| `n56pn9fncw` | `anthracnose_leaf_spot`, `straw_mite` | `anomala` |
| `sy69db2nz5` | `Healthy-Leaf` | `saudavel` |
| `sy69db2nz5` | `Anthracnose`, `Bacterial-Spot`, `Downy-Mildew`, `Pest-Damage` | `anomala` |

**Ressalva de espécie (importante para o texto do TCC):** Malabar Spinach
(*Basella alba*) é uma hortaliça diferente do espinafre-verdadeiro
(*Spinacia oleracea*) que está no seu escopo declarado. É popularmente
vendida/consumida como "espinafre" em vários países (daí o nome), tem porte
e folha semelhantes, mas é botanicamente outra família (Basellaceae vs
Amaranthaceae) — as doenças listadas (Anthracnose, Downy Mildew, Bacterial
Spot, dano de praga) são gênericas o bastante para se aplicarem a folhosas
em geral, mas não é tecnicamente a mesma espécie. **Recomendo declarar isso
explicitamente no capítulo de dados do TCC-II** — é o tipo de nuance que
uma banca vai notar se você não mencionar primeiro.

---

## Fase 4 — Duplicatas e colisões

Mesmo método (D4-orbit dHash, limiar ≤4) cruzando os 3.609 crops novos
contra os 174.381 combinados (3.609 novos + 170.772 já existentes em
`backend/data/`).

| Métrica | Valor |
|---|---|
| Duplicata interna (só entre os dados novos) | 137 grupos, 304 imagens redundantes (**8,4%** — contra 83% do caso Roboflow) |
| Grupos que cruzam os 2 datasets de espinafre | 5 de 137 (confirma que são fontes majoritariamente independentes) |
| Colisão com TEST existente | 2 grupos — **ambos confirmados como falso positivo** (comparação visual: espinafre vs. folha de maçã, espinafre vs. folha de laranja — sem relação nenhuma) |
| Colisão com TRAIN/VAL existente | 22 grupos (a checar se também são falsos positivos antes de decidir se preocupa) |

**Resultado: nenhuma colisão real de vazamento contra o test set.** Isso é
esperado — é a primeira vez que espinafre (ou Malabar spinach) entra no
projeto, então não há como ter sido "a mesma foto" de antes.

---

## Fase 5 — Caracterização de domínio

Revisei as grades de amostra (`dataset/_reports/sample_grids/grid_spinach_*.jpg`).
**Não é foto de campo natural.** É folha isolada, colhida, fotografada sobre
fundo branco/papel/mesa — mesmo estilo controlado do PlantVillage e da
maioria do que você já tem. A descrição "coletado em campos reais" nos
papers provavelmente se refere a onde a planta foi cultivada (não em
laboratório/estufa), não a como a foto foi tirada. **Não conta como dado
que fecha o domain gap** — mas também não piora nada, e adiciona uma
espécie nova ao repertório visual do modelo.

---

## Recomendação go/no-go

**GO**, com as ressalvas registadas acima (nomenclatura de espécie
+ não é "campo" no sentido do TCC). Motivos:
- Resolução excelente, sem necessidade de upscaling.
- Redundância interna baixa (8,4%, ordem de grandeza normal, não uma bandeira vermelha como no caso anterior).
- Zero vazamento real confirmado contra o test set.
- Preenche uma lacuna real: hoje o projeto tem zero dado de espinafre/similar.
- Balanceamento saudável, ajuda o desbalanceamento geral do dataset.

**O que falta antes de eu integrar de verdade (Fase 6, que ainda não rodei):**
- [ ] Sua aprovação explícita — quer que eu prossiga com os 3.609 crops
      (menos os ~304 redundantes internos, ficando 1 representante por
      grupo)?
- [ ] Decidir se quer registrar no TCC como "espinafre (Malabar
      spinach/*Basella alba*, proxy botânico)" ou se prefere não incluir
      por rigor de escopo — é uma decisão sua, não técnica.
- [ ] Conferir rapidamente os 22 grupos de colisão com train/val (provável
      que sejam falso positivo como os 2 de test, mas não confirmei
      individualmente).

Se aprovar, entra junto com o `lettuce_downy_v1` (do relatório anterior) na
próxima rodada da Fase 6.
