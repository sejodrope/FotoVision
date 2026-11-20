# -*- coding: utf-8 -*-
"""Artigo v14 (19/11/2026): correções do Prof. Leanderson (entrega 1) + Guia de
Trabalhos Acadêmicos da Univille (2019).

Base: docs/Entregas Finais/TCC2_Artigo_FitoVision.docx (a versão que o orientador
revisou, = v13 com o título curto do José). Uso:
    python build_v14.py <base.docx> <saida.docx>

Marcação nos textos: **negrito**, _itálico_ (alternam).
"""
import copy, re, sys, zipfile, shutil, tempfile, os
from lxml import etree

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
W14 = 'http://schemas.microsoft.com/office/word/2010/wordml'
def q(t): return '{%s}%s' % (W, t)

# ---------------------------------------------------------------- textos
SIS = ('um sistema de triagem fitossanitária para hortaliças folhosas, baseado em '
       '_transfer learning_, treinado com imagens públicas auditadas contra vazamento '
       'de dados e dotado de calibração e abstenção')
OBJ = 'avaliar em que medida %s mantém desempenho confiável em fotografias de campo' % SIS

ABSTRACT = ('_**Abstract.** Plant disease diagnosis from photographs is usually evaluated on '
 'laboratory images. The objective of this work was to evaluate to what extent a phytosanitary '
 'screening system for leafy vegetables, based on transfer learning, trained on public images '
 'audited for data leakage and equipped with calibration and abstention, remains reliable on '
 'field photographs. Three convolutional networks (EfficientNet-B0, MobileNetV2 and ResNet-50) and a '
 'Vision Transformer (ViT-B/16) were compared with '
 'paired McNemar tests on a test set drawn from the training sources, on a held-out source and '
 'on field photographs. EfficientNet-B0 and ViT-B/16 tied at 98.48% balanced accuracy and '
 'outperformed the other two, but the abstention threshold intercepted only 31% and 15% of '
 'their errors. On a source never seen in training, an earlier version scored 75.4%. On 13 field '
 'photographs the versions could not be distinguished, and on 10 photographs of healthy lettuce '
 'with real background none raised a false alarm. Performance depended more on the evaluation '
 'protocol than on the architecture, and field reliability could not be fully measured, for '
 'lack of photographs of diseased leaves._')
KEYWORDS = '_**Keywords:** Computer vision. Plant diseases. Domain gap._'
RESUMO = ('_**Resumo.** O diagnóstico de doenças de plantas por fotografia costuma ser avaliado '
 'em imagens de laboratório. O objetivo deste trabalho foi ' + OBJ.replace('_', '') + '. Três '
 'redes convolucionais (EfficientNet-B0, MobileNetV2 e ResNet-50) e um Vision Transformer (ViT-B/16) '
 'foram comparados pelo teste de McNemar pareado em '
 'um teste extraído das mesmas fontes do treino, em uma fonte retida e em fotografias de campo. '
 'EfficientNet-B0 e ViT-B/16 empataram em 98,48% de acurácia balanceada e superaram os outros '
 'dois, mas o limiar de abstenção interceptou apenas 31% e 15% dos seus erros. Em uma fonte nunca '
 'vista no treino, uma versão anterior obteve 75,4%. Em 13 fotografias de campo as versões não se '
 'distinguiram e, em 10 de alface sadia com fundo real, nenhuma emitiu falso alarme. Concluiu-se '
 'que o desempenho dependeu mais do protocolo de avaliação do que da arquitetura e que a '
 'confiabilidade em campo não pôde ser medida por completo, por faltarem fotografias de folhas '
 'doentes._')
PALAVRAS = '_**Palavras-chave:** Visão computacional. Doenças de plantas. Domain gap._'

NEW = {
 9: ('O problema deste trabalho nasceu dessa lacuna e de um resultado próprio. Na primeira etapa do '
     'Trabalho de Conclusão de Curso (TCC-I), uma auditoria por _hash_ perceptual revelou que 57,35% do conjunto de teste era '
     'composto por fotografias quase idênticas a imagens de treino, o que invalidou a acurácia de '
     '99,01% inicialmente obtida. Após a reconstrução do particionamento, o melhor modelo atingiu '
     '98,49% de acurácia balanceada no domínio de teste, mas apenas 73,3% em uma validação '
     'exploratória com 13 fotografias de alface. Como um sistema que erra com aparência de '
     'certeza transfere confiança indevida ao produtor, a calibração e a abstenção fizeram parte '
     'do objeto de estudo.'),
 10: ('Diante do exposto, a questão de pesquisa foi: em que medida %s mantém desempenho confiável '
      'em fotografias de campo? Assim, o objetivo deste trabalho foi avaliar em que medida esse '
      'sistema mantém desempenho confiável em fotografias de campo. Para isso, o sistema foi '
      'avaliado em um teste extraído das mesmas fontes do treino, em uma fonte nunca vista e em '
      'campo, verificando-se também se a abstenção interceptou os erros.') % SIS,
 20: ('A busca foi feita na Web of Science, em outubro de 2026 (doença de plantas e aprendizado '
      'profundo com generalização em campo, vazamento, calibração e hortaliças folhosas); o Quadro 1 '
      'resume os estudos mais próximos. Classificadores que passam de 99% no PlantVillage caem para '
      '26% a 30% no conjunto de campo PlantDoc, sobretudo por atribuírem a folha à cultura errada, e '
      'o ajuste fino completo de um modelo de fundação piora a transferência (Nguyen et al., 2026); '
      'uma revisão aponta arquiteturas híbridas como mais robustas e o ViT puro como mais sensível '
      'ao ambiente (Shafay et al., 2025). Dois estudos têm protocolo semelhante: com amêndoa, a '
      'validação agrupada quase não altera o F1 macro, mas um teste com fontes disjuntas o reduz a '
      '0,56 (Workneh et al., 2026); com videira, a EfficientNet-B0 obtém 73,8% em um conjunto '
      'externo (Ergün et al., 2026).'),
 25: ('Para atingir o objetivo de avaliar em que medida o sistema mantém desempenho confiável em fotografias de campo, a pesquisa foi quantitativa, exploratória e descritiva quanto '
      'aos objetivos e experimental quanto aos procedimentos, pois arquitetura, calibração, limiar '
      'de abstenção e domínio de avaliação foram manipulados sob condições controladas e '
      'reprodutíveis (Gil, 2010). O sistema foi avaliado em três situações: um teste extraído das '
      'mesmas fontes do treino, uma fonte retida e fotografias de campo. Todo o _pipeline_ rodou por '
      '_script_, com semente fixa (42) e registro de cada execução.'),
 27: ('O _dataset_ binário (saudável × anômalo) reuniu seis fontes públicas do Kaggle, entre elas o '
      'PlantVillage, como _proxy_ visual, e _datasets_ de alface. A rotulagem foi automática, pela '
      'árvore de diretórios, com cada decisão registrada para auditoria. Contra o vazamento, cada '
      'imagem foi reduzida a um _hash_ perceptual invariante às oito transformações do grupo '
      'diedral D4; imagens a distância de Hamming ≤ 4 foram agrupadas por união-busca, e cada grupo '
      'foi inteiro para uma partição (≈70/15/15, estratificada por classe), descartados os grupos '
      'com rótulos contraditórios. Das 176.126 imagens reunidas restaram 170.772 (120.861 de treino, '
      '25.944 de validação e 23.967 de teste), com vazamento residual de 0,04% no teste. Como as '
      'partições provêm das mesmas seis fontes, o teste mede desempenho intra-fonte. O critério não '
      'alcançou variantes muito alteradas: 23% do teste eram variantes de fotografias de alface com '
      'outra variante no treino.'),
 28: ('Em um quarto ciclo foram incorporados dois lotes preparados com o mesmo protocolo: 3.209 '
      'imagens de espinafre-de-malabar (_Basella alba_, e não _Spinacia oleracea_) e 1.142 recortes '
      'de 49 fotografias de alface com míldio, particionados pela fotografia de origem, o que elevou '
      'o conjunto a 175.123 imagens.'),
 29: ('As três CNNs foram inicializadas com pesos do ImageNet e ajustadas integralmente por até 30 '
      'épocas, com imagens de 224×224 pixels (AdamW, taxa de aprendizado de 10⁻³ com escalonador de '
      'cosseno, parada antecipada com paciência 10, entropia cruzada com _label smoothing_ de 0,1 e '
      'amostragem ponderada pelo inverso da frequência de classe). O ViT-B/16 seguiu o mesmo '
      'protocolo, com taxa de 5×10⁻⁵, decaimento de pesos de 0,05, duas épocas de aquecimento e '
      'limitação da norma do gradiente. Cada modelo foi calibrado por _temperature scaling_ na '
      'validação, e τ foi o menor valor, a partir de 0,85, que assegurou acurácia seletiva de ao '
      'menos 98% na validação. Em produção, o sistema devolve um de quatro veredictos: saudável, '
      'anômalo, inconclusivo (confiança < τ) ou não é folha (menos de 10% de pixels vegetativos '
      'por ExG).'),
 30: ('Um quinto ciclo corrigiu três defeitos achados na auditoria: a alface passou a ser '
      'particionada pela fotografia original, com uma por original na validação e no teste (176 no '
      'teste, 15 doentes); 8.266 imagens de pastas que a rotulagem deixava de fora, quase todas de '
      'dano por praga, entraram como anômalas; e a amostragem passou a equilibrar os rótulos dentro '
      'de cada cultura, sem pesos na perda, porque cultura e rótulo estavam associados (87% da '
      'alface de treino era saudável). Um lote de espinafre-de-malabar (574 imagens) ficou fora do '
      'treino. O conjunto teve 152.057 imagens, com 1,7% do teste a distância ≤ 4 do treino, e as '
      'demais fontes mantiveram a partição, o que permitiu comparar os ciclos imagem a imagem.'),
 32: ('O protocolo de coleta previu fotografias de hortas da região de Joinville (SC), feitas com '
      'celular, sob luz natural e sem preparação da cena: de 100 a 300 fotografias, com mínimo de 15 '
      'e meta de 25 por cultura, cerca de metade de folhas anômalas e variedade de sintomas '
      '(manchas, clorose, míldio, furos, murcha). Previram-se também fotografias da mesma folha em três distâncias e '
      'imagens sem folha (solo, mão), para testar a guarda de vegetação.'),
 33: ('Quatro regras garantiram a validade do conjunto: (i) o rótulo foi registrado na coleta, antes '
      'de qualquer inferência, e fotografias ambíguas seriam marcadas e reportadas com e sem elas; '
      '(ii) fotografias de campo nunca entraram no treino; (iii) versões do modelo foram comparadas '
      'sobre as mesmas fotografias; e (iv) as 13 fotografias do TCC-I formaram um conjunto '
      'histórico, reportado à parte. A coleta começou em 27/10/2026, com 10 fotografias de canteiros '
      'de alface sadia, tiradas de cima, com vários pés por quadro; um lote anterior, de um pé '
      'colhido e fotografado em casa sobre toalha branca (3 folhas sadias e 7 anômalas), também foi '
      'reportado à parte, por não ser de campo.'),
 35: ('A análise acompanhou as três situações de avaliação, com α = 0,05 e a correção sequencial de '
      'Holm nas comparações múltiplas sobre os mesmos dados.'),
 36: ('Para o desempenho no domínio de teste, considerando a classe anômala como positiva, foram '
      'calculados, a partir da matriz de confusão, a acurácia global, a precisão, a revocação, o '
      'a medida F1 por classe e macro, a área sob a curva ROC (_receiver operating characteristic_), ou AUC, e a latência média de inferência. A '
      'métrica principal foi a acurácia balanceada (Equação 1), média das revocações das duas '
      'classes, que não é inflada pelo desbalanceamento de aproximadamente 1,35:1 do _dataset_ '
      '(Brodersen et al., 2010):'),
 38: ('Na Equação 1, VP, VN, FP e FN são os verdadeiros e falsos positivos e negativos. Os falsos negativos — plantas doentes declaradas saudáveis — foram reportados à parte, por '
      'serem o erro de maior custo agronômico. Os intervalos de confiança (IC) de 95% vieram de '
      '_bootstrap_ estratificado por classe, com 10.000 reamostragens, e, como as arquiteturas foram '
      'avaliadas sobre as mesmas 23.967 imagens, pares de modelos foram comparados pelo teste de '
      'McNemar, que considera só as imagens em que os dois discordam (Dietterich, 1998).'),
 39: ('A calibração foi avaliada pelo ECE com _M_ = 15 faixas de igual largura (Equação 2), onde '
      '_B_ₘ é o conjunto de predições com confiança na faixa _m_, acc e conf são a acurácia e a confiança '
      'média nessa faixa e _n_ é o total de predições, antes e depois do _temperature scaling_:'),
 41: ('O mecanismo de abstenção foi avaliado pela cobertura (fração de imagens diagnosticadas), pela '
      'acurácia seletiva (acurácia entre elas) e pela fração de erros interceptados (confiança < τ); '
      'a curva risco–cobertura, com τ de 0,50 a 0,99, orientou o ponto de operação, e o teste de '
      'Mann-Whitney comparou a confiança de acertos e erros.'),
 42: ('Em campo, as abstenções tiveram taxa própria e as métricas de acerto foram calculadas sobre '
      'os diagnósticos emitidos. '
      'Como as amostras são pequenas, toda proporção recebeu intervalo de Wilson a 95%, e a acurácia '
      'balanceada, intervalo por _bootstrap_ estratificado. O _domain gap_ foi definido como a '
      'diferença entre a acurácia balanceada no teste e em campo, e versões do modelo foram '
      'comparadas sobre as mesmas fotografias pelo teste de McNemar exato. Para uma proporção de '
      '80%, o intervalo de Wilson tem mais de 40 pontos com 13 fotografias, 15,5 com 100 e 9,0 com '
      '300, o que definiu a meta de coleta.'),
 43: ('A desagregação por cultura, tipo de sintoma e distância de captura, prevista no protocolo, '
      'exige folhas doentes fotografadas em campo e não pôde ser feita.'),
 48: ('Na Tabela 1, EfficientNet-B0 e ViT-B/16 empataram, com IC 95% de 98,32% a 98,64% e sem '
      'diferença significativa (p de Holm = 0,26). Ambos superaram o MobileNetV2 (97,70% a 98,07%) '
      'e o ResNet-50 (97,63% a 98,00%), com p ≤ 1,4×10⁻¹², e estes não diferiram entre si '
      '(p = 0,26). O ViT-B/16 teve menos falsos negativos (137 contra 209) e mais falsos positivos '
      '(204 contra 154), com 21 vezes mais parâmetros e 2,7 vezes a latência.'),
 49: ('Quanto à calibração, o _temperature scaling_ reduziu o ECE de validação a 0,0027 no '
      'EfficientNet-B0 e a 0,0022 no ViT-B/16. No teste, a confiança calibrada do EfficientNet-B0 '
      'separou acertos de erros (0,990 contra 0,874; p < 0,001), mas τ = 0,85 interceptou apenas 113 '
      'dos 363 erros (31,1%); no ViT-B/16, 51 dos 341 (15,0%).'
      ' Assim, o menor ECE, o do ViT-B/16, não '
      'resultou em mais erros interceptados.'),
 50: ('A Tabela 1 tem uma ressalva: o teste veio das mesmas seis fontes do treino, e os 98,48% são '
      'uma acurácia intra-fonte. No lote de espinafre-de-malabar que o modelo do terceiro ciclo '
      'nunca viu, o EfficientNet-B0 teve 75,4% (N = 484), 23 pontos abaixo.'
      ' Com o lote no treino, subiu para 88,4%, mas o '
      'ganho é intra-fonte e não informa sobre o campo. A Tabela 2 reúne o desempenho de cada ciclo '
      'nas 13 fotografias de campo do TCC-I.'),
 54: ('Na Tabela 2, os intervalos de Wilson se sobrepuseram e, com N = 13, versões e arquiteturas '
      'não se distinguiram. A amostra revelou um defeito de rotulagem, mas não permitiu decidir '
      'entre modelos. Os nove erros dos quatro primeiros ciclos tiveram confiança calibrada entre '
      '0,949 e 0,999, e o ECE passou de 0,003 na validação para 0,146 em campo, valor só indicativo '
      'com N = 13. A Tabela 3 compara o '
      'quinto ciclo ao quarto, no teste e nos subconjuntos em que os erros se concentravam.'),
 58: ('Na Tabela 3, a correção agiu onde estavam os erros: nas culturas de lavoura, as três CNNs '
      'ganharam de 1,8 a 3,5 pontos de acurácia balanceada e de 6,8 a 11,9 de especificidade, isto '
      'é, menos falsos alarmes; o ViT-B/16 quase não mudou (p = 0,66). O custo foi a sensibilidade, '
      'que caiu de 1,8 a 4,9 pontos: por imagem, o quarto ciclo acertou mais (p ≤ 0,04), pois três '
      'em cada quatro dessas imagens são anômalas. No lote retido, os quatro ficaram entre 85,8% e '
      '90,1%, dez pontos abaixo das fontes de treino.'),
 59: ('Em relação ao Quadro 1, a queda para 75,4% em uma fonte retida, de espinafre-de-malabar, foi da ordem dos 73,8% da '
      'EfficientNet-B0 em videira (Ergün et al., 2026) e, como na amêndoa, só apareceu com a fonte '
      'separada (Workneh et al., 2026); diante dos 97,3% da mesma cultura em partição interna '
      '(Kabya et al., 2026), a diferença parece ser mais de protocolo do que de cultura. O próprio teste '
      'reproduziu a perda: as culturas de lavoura tiveram 86,3% no ciclo 4 e 81% dos erros, contra '
      '99,7% nas demais.'),
 60: ('No lote de campo, as quatro versões (EfficientNet-B0 e ViT-B/16, ciclos 4 e 5) deram como '
      'sadias as 10 fotografias da horta, sem abstenção, com confiança acima de 0,98 (IC 95% de Wilson: 72,2% '
      'a 100%), também em recortes centrais. Solo, mato e vários pés '
      'no quadro não provocaram falso alarme. No lote de casa, todas as fotografias diagnosticadas '
      'saíram anômalas, com confiança acima de 0,97, inclusive as das três folhas sadias, que foram falsos '
      'alarmes; como cada lote estava quase todo em uma classe, folha e cena não puderam ser separadas.'),
 61: ('**Limitações do estudo.** Há cinco ressalvas. (i) A evidência de campo foi pequena: 23 '
      'fotografias, só de alface, e o lote com fundo real não teve folhas doentes; por isso, a '
      'sensibilidade e o _domain gap_ em campo seguem sem medida. (ii) Treino e teste vieram das '
      'mesmas fontes públicas, distantes das condições reais de uso (Xu et al., 2024), e parte das '
      'culturas entrou só como _proxy_ visual. '
      '(iii) A calibração foi ajustada fora do campo: do laboratório para a lavoura, o _temperature '
      'scaling_ reduz o ECE, mas a confiança continua bem acima da acurácia (Xiang; Shi; Zhu, 2026), '
      'como no ECE de 0,146 obtido aqui. (iv) A tarefa foi binária e não nomeia a doença, e sintomas '
      'parecidos podem ter causas diferentes (Barbedo, 2016). (v) Cada modelo foi treinado uma única '
      'vez, e o desempenho de uma rede varia entre sementes (Coakley; Gundersen, 2026): o empate '
      'entre EfficientNet-B0 e ViT-B/16 pode não se repetir.'),
}

LIT_INTRO = ('Esta seção apresenta os modelos de aprendizado profundo e o _transfer learning_ (2.1), os '
 'fatores que comprometem a validade da avaliação (2.2) e os trabalhos mais próximos deste (2.3).')
QUADRO_COMENT = ('Pelo Quadro 1, as quedas mais fortes surgiram com fonte ou conjunto externo ao treino, e '
 'o único estudo com hortaliça folhosa avaliou só em partição interna. Não se localizou estudo que '
 'reunisse, nessas culturas, auditoria de vazamento, calibração, abstenção e avaliação fora da fonte.')
TAB1_INTRO = ('A Tabela 1 mostra o desempenho das arquiteturas no teste do quarto ciclo.')

CONCLUSAO = [
 ('O diagnóstico de doenças de plantas por fotografia passa de 99% de acurácia em laboratório, mas '
  'perde desempenho em campo, diferença pouco medida em hortaliças folhosas. Partindo dessa lacuna e '
  'dos 99,01% do TCC-I, invalidados por vazamento de dados, este trabalho teve como objetivo ' + OBJ + '.'),
 ('Os resultados indicam que o desempenho dependeu mais do protocolo de '
  'avaliação do que da arquitetura: 98,48% de acurácia balanceada no teste intra-fonte, 75,4% em uma '
  'fonte nunca vista, 84,6% de acertos em 13 fotografias de campo e nenhum falso alarme em 10 com '
  'fundo real. EfficientNet-B0 e ViT-B/16 empataram no teste, e o EfficientNet-B0, 21 vezes menor e '
  'melhor na lavoura, mostrou-se a opção mais adequada neste estudo. A calibração foi boa na '
  'distribuição de treino, mas o limiar de abstenção deixou passar mais de dois terços dos erros, e em '
  'campo os erros vieram com confiança acima de 0,94, de modo que a abstenção sozinha não '
  'protegeu o produtor.'),
 ('A resposta à questão de pesquisa é, portanto, parcial: o sistema não gerou falso alarme em alface '
  'sadia na horta, mas a sensibilidade e o _domain gap_ em campo não puderam ser medidos, por falta de '
  'folhas doentes. A principal contribuição foi um protocolo reprodutível, com teste intra-fonte, fonte '
  'retida, campo com rótulo prévio à inferência e testes pareados, que expôs o vazamento do TCC-I e os '
  'limites da calibração fora do treino.'),
 ('Como trabalhos futuros, sugere-se completar a coleta com folhas doentes e outras culturas; '
  'recalibrar o modelo com fotografias de campo; testar extratores congelados de modelos de fundação '
  '(Nguyen et al., 2026); repetir o treino com várias sementes; e definir, com produtores, o custo de '
  'cada tipo de erro, para escolher o limiar de abstenção.'),
]

NOTAS = {  # índice original -> (nota, fonte)
 23: (None, 'Fonte: elaboração própria (2026), a partir dos trabalhos citados.'),
 47: ('Nota: Parâm. = parâmetros do modelo treinado, com saída de duas classes; Acur. bal. = acurácia '
      'balanceada; FN = falsos negativos; latência em placa gráfica (GPU) RTX 3050 (6 GB), medida na mesma sessão para os quatro modelos.',
      'Fonte: elaboração própria (2026).'),
 53: ('Nota: ciclos 1 a 5 com EfficientNet-B0; * ViT-B/16 com os dados do ciclo 4.',
      'Fonte: elaboração própria (2026).'),
 57: ('Nota: teste do ciclo 5 com 18.413 imagens. Lavoura: mandioca, arroz, cana e café '
      '(N = 4.530), também em Espec. (especificidade) e Sens. (sensibilidade); ciclo 4 → ciclo 5. Alface: N = 176. Retido: '
      'espinafre, N = 574.', 'Fonte: elaboração própria (2026).'),
}

AC = 'Acesso em: 19 nov. 2026.'
REF_STYLE = os.environ.get('REF_STYLE', 'doi')
REF_GAP = int(os.environ.get('REF_GAP', '240'))
def doi(d):
    if REF_STYLE == 'url': return 'Disponível em: https://doi.org/%s. %s' % (d, AC)
    return 'DOI: %s.' % d
REFS = [
 'ABBASI, R.; MARTINEZ, P.; AHMAD, R. Crop diagnostic system: a robust disease detection and management system for leafy green crops grown in an aquaponics facility. **Artificial Intelligence in Agriculture**, v. 10, p. 1-12, 2023. ' + doi('10.1016/j.aiia.2023.09.001'),
 'AHMAD, A.; GAMAL, A.; SARASWAT, D. Toward generalization of deep learning-based plant disease identification under controlled and field conditions. **IEEE Access**, v. 11, p. 9042-9057, 2023. ' + doi('10.1109/ACCESS.2023.3240100'),
 'BARBEDO, J. G. A. A review on the main challenges in automatic plant disease identification based on visible range images. **Biosystems Engineering**, v. 144, p. 52-60, 2016. ' + doi('10.1016/j.biosystemseng.2016.01.017'),
 'BARBEDO, J. G. A. Impact of dataset size and variety on the effectiveness of deep learning and transfer learning for plant disease classification. **Computers and Electronics in Agriculture**, v. 153, p. 46-53, 2018. ' + doi('10.1016/j.compag.2018.08.013'),
 'BRODERSEN, K. H. et al. The balanced accuracy and its posterior distribution. In: INTERNATIONAL CONFERENCE ON PATTERN RECOGNITION, 20., 2010, Istambul. **Proceedings** [...]. Los Alamitos: IEEE, 2010. p. 3121-3124. ' + doi('10.1109/ICPR.2010.764'),
 'COAKLEY, K. L.; GUNDERSEN, O. E. Measuring deep learning performance: an empirical study of performance distributions across architectures and tasks. **Scientific Reports**, v. 16, n. 1, art. 20474, 2026. ' + doi('10.1038/s41598-026-49656-z'),
 'DIETTERICH, T. G. Approximate statistical tests for comparing supervised classification learning algorithms. **Neural Computation**, v. 10, n. 7, p. 1895-1923, 1998. ' + doi('10.1162/089976698300017197'),
 'DOSOVITSKIY, A. et al. An image is worth 16x16 words: transformers for image recognition at scale. In: INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS, 9., 2021, [S. l.]. **Proceedings** [...]. [S. l.: s. n.], 2021. Disponível em: https://openreview.net/forum?id=YicbFdNTTy. ' + AC,
 'ERGÜN, E. et al. Confidence and uncertainty aware deep learning for reliable grape leaf disease diagnosis under real world field conditions. **Frontiers in Plant Science**, v. 17, art. 1839652, 2026. ' + doi('10.3389/fpls.2026.1839652'),
 'FACELI, K. et al. **Inteligência artificial**: uma abordagem de aprendizado de máquina. 2. ed. Rio de Janeiro: LTC, 2023.',
 'GEIFMAN, Y.; EL-YANIV, R. Selective classification for deep neural networks. In: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS, 30., 2017, Long Beach. **Proceedings** [...]. Red Hook: Curran Associates, 2017.',
 'GEIRHOS, R. et al. Shortcut learning in deep neural networks. **Nature Machine Intelligence**, v. 2, n. 11, p. 665-673, 2020. ' + doi('10.1038/s42256-020-00257-z'),
 'GIL, A. C. **Como elaborar projetos de pesquisa**. 5. ed. São Paulo: Atlas, 2010.',
 'GUO, C. et al. On calibration of modern neural networks. In: INTERNATIONAL CONFERENCE ON MACHINE LEARNING, 34., 2017, Sydney. **Proceedings** [...]. [S. l.]: PMLR, 2017. p. 1321-1330. Disponível em: https://proceedings.mlr.press/v70/guo17a.html. ' + AC,
 'HE, K. et al. Deep residual learning for image recognition. In: IEEE CONFERENCE ON COMPUTER VISION AND PATTERN RECOGNITION, 2016, Las Vegas. **Proceedings** [...]. Piscataway: IEEE, 2016. p. 770-778. ' + doi('10.1109/CVPR.2016.90'),
 'INSTITUTO BRASILEIRO DE GEOGRAFIA E ESTATÍSTICA – IBGE. **Censo Agropecuário 2017**: resultados definitivos. Rio de Janeiro: IBGE, 2019.',
 'KABYA, N. D. et al. Towards practical AI for agriculture: a self-supervised attention framework for spinach leaf disease detection. **PLOS ONE**, v. 21, n. 1, e0340989, 2026. ' + doi('10.1371/journal.pone.0340989'),
 'KAUFMAN, S. et al. Leakage in data mining: formulation, detection, and avoidance. **ACM Transactions on Knowledge Discovery from Data**, v. 6, n. 4, p. 1-21, 2012. ' + doi('10.1145/2382577.2382579'),
 'MOHANTY, S. P.; HUGHES, D. P.; SALATHÉ, M. Using deep learning for image-based plant disease detection. **Frontiers in Plant Science**, v. 7, art. 1419, 2016. ' + doi('10.3389/fpls.2016.01419'),
 'NGUYEN, T. A. et al. From laboratory to field: frozen foundation-model features toward robust plant disease recognition. **Smart Agricultural Technology**, v. 15, art. 102531, 2026. ' + doi('10.1016/j.atech.2026.102531'),
 'SANDLER, M. et al. MobileNetV2: inverted residuals and linear bottlenecks. In: IEEE CONFERENCE ON COMPUTER VISION AND PATTERN RECOGNITION, 2018, Salt Lake City. **Proceedings** [...]. Piscataway: IEEE, 2018. p. 4510-4520. ' + doi('10.1109/CVPR.2018.00474'),
 'SHAFAY, M. et al. Recent advances in plant disease detection: challenges and opportunities. **Plant Methods**, v. 21, n. 1, art. 140, 2025. ' + doi('10.1186/s13007-025-01450-0'),
 'TAN, M.; LE, Q. V. EfficientNet: rethinking model scaling for convolutional neural networks. In: INTERNATIONAL CONFERENCE ON MACHINE LEARNING, 36., 2019, Long Beach. **Proceedings** [...]. [S. l.]: PMLR, 2019. p. 6105-6114. Disponível em: https://proceedings.mlr.press/v97/tan19a.html. ' + AC,
 'UPADHYAY, A. et al. Deep learning and computer vision in plant disease detection: a comprehensive review of techniques, models, and trends in precision agriculture. **Artificial Intelligence Review**, v. 58, n. 3, art. 92, 2025. ' + doi('10.1007/s10462-024-11100-x'),
 'WOEBBECKE, D. M. et al. Color indices for weed identification under various soil, residue, and lighting conditions. **Transactions of the ASAE**, v. 38, n. 1, p. 259-269, 1995. ' + doi('10.13031/2013.27838'),
 'WORKNEH, A. D. et al. Leakage-aware, calibrated, and explainable deep learning for robust almond disease classification. **Computers and Electronics in Agriculture**, v. 256, art. 112464, 2026. ' + doi('10.1016/j.compag.2026.112464'),
 'XIANG, K.; SHI, D.; ZHU, X. Quantifying the reliability gap in cross-domain plant disease classification: benchmarking the limited efficacy of standard mitigation techniques under controlled-to-field shift. **Frontiers in Plant Science**, v. 17, art. 1826962, 2026. ' + doi('10.3389/fpls.2026.1826962'),
 'XU, M. et al. Plant disease recognition datasets in the age of deep learning: challenges and opportunities. **Frontiers in Plant Science**, v. 15, art. 1452551, 2024. ' + doi('10.3389/fpls.2024.1452551'),
]
# Os underscores de URL não existem nas refs acima (verificado em check_refs()).

# ------------------------------------------------- citações ABNT (NBR 10520)
NAME = r"[A-ZÀ-Ý][\wÀ-ÿ'\-]*(?: [A-ZÀ-Ý][\wÀ-ÿ'\-]*)*"
def _cit(m):
    inner = m.group(1)
    parts = inner.split('; ')
    if not re.search(r', \d{4}[a-z]?$', parts[-1]):
        return m.group(0)
    out = []
    for p in parts:
        mm = re.fullmatch(r'(%s)( et al\.)?(, \d{4}[a-z]?)?' % NAME, p)
        if not mm:
            return m.group(0)
        out.append(mm.group(1).upper() + (mm.group(2) or '') + (mm.group(3) or ''))
    return '(' + '; '.join(out) + ')'
def abnt(s): return re.sub(r'\(([^()]+)\)', _cit, s)

# ---------------------------------------------------------------- XML utils
def set_val(rpr, tag, on):
    e = rpr.find(q(tag))
    if e is None:
        e = etree.SubElement(rpr, q(tag))
    e.set(q('val'), '1' if on else '0')

def segments(markup):
    toks = re.split(r'(\*\*|_)', markup)
    b = i = False
    for t in toks:
        if t == '**': b = not b
        elif t == '_': i = not i
        elif t: yield t, b, i

def fill(p, markup, base_rpr=None):
    if base_rpr is None:
        r0 = p.find('.//' + q('r') + '/' + q('rPr'))
        base_rpr = copy.deepcopy(r0) if r0 is not None else etree.Element(q('rPr'))
    for ch in list(p):
        if ch.tag != q('pPr'):
            p.remove(ch)
    for text, b, i in segments(markup):
        r = etree.SubElement(p, q('r'))
        rpr = copy.deepcopy(base_rpr)
        for t in ('b', 'bCs'): set_val(rpr, t, b)
        for t in ('i', 'iCs'): set_val(rpr, t, i)
        r.append(rpr)
        t_ = etree.SubElement(r, q('t'))
        t_.text = text
        t_.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')

def strip_ids(el):
    for e in el.iter():
        for a in list(e.attrib):
            if a.startswith('{%s}' % W14):
                del e.attrib[a]

def set_ppr(p, before=None, after=None, first=None, left=None, jc=None, hanging_off=False):
    ppr = p.find(q('pPr'))
    if ppr is None:
        ppr = etree.Element(q('pPr')); p.insert(0, ppr)
    rpr = ppr.find(q('rPr'))
    def put(tag):
        e = ppr.find(q(tag))
        if e is None:
            e = etree.Element(q(tag))
            if rpr is not None: rpr.addprevious(e)
            else: ppr.append(e)
        return e
    if before is not None or after is not None:
        sp = put('spacing')
        if before is not None: sp.set(q('before'), str(before))
        if after is not None: sp.set(q('after'), str(after))
        sp.set(q('line'), '240'); sp.set(q('lineRule'), 'auto')
    if first is not None or left is not None or hanging_off:
        ind = put('ind')
        if hanging_off and q('hanging') in ind.attrib: del ind.attrib[q('hanging')]
        if left is not None: ind.set(q('left'), str(left)); ind.set(q('right'), '0')
        if first is not None: ind.set(q('firstLine'), str(first))
    if jc is not None:
        put('jc').set(q('val'), jc)
    # ordem do schema: spacing, ind, jc antes de rPr; garantir spacing antes de ind
    sp, ind, j = ppr.find(q('spacing')), ppr.find(q('ind')), ppr.find(q('jc'))
    for e in (sp, ind, j):
        if e is not None:
            ppr.remove(e)
            if rpr is not None: rpr.addprevious(e)
            else: ppr.append(e)

def clone_after(anchor, model, markup, base_rpr=None):
    p = copy.deepcopy(model); strip_ids(p)
    fill(p, markup, base_rpr)
    anchor.addnext(p)
    return p

# ---------------------------------------------------------------- build
def build(src, dst):
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(src) as z: z.extractall(tmp)
    dpath = os.path.join(tmp, 'word', 'document.xml')
    spath = os.path.join(tmp, 'word', 'styles.xml')

    st = etree.parse(spath)
    pd = st.getroot().find('.//' + q('pPrDefault') + '/' + q('pPr'))
    sp = pd.find(q('spacing'))
    sp.set(q('before'), '0'); sp.set(q('after'), '0'); sp.set(q('line'), '240'); sp.set(q('lineRule'), 'auto')
    HEAD_BEFORE = os.environ.get('HEAD_BEFORE', '120')
    for sty in st.getroot().iter(q('style')):
        if sty.get(q('styleId')) in ('Heading1', 'Heading2'):
            hs = sty.find(q('pPr') + '/' + q('spacing'))
            hs.set(q('before'), HEAD_BEFORE); hs.set(q('after'), '0')
    st.write(spath, xml_declaration=True, encoding='UTF-8', standalone=True)

    t = etree.parse(dpath)
    body = t.getroot().find(q('body'))
    o = list(body)
    body_model = o[7]
    body_rpr = copy.deepcopy(o[7].find('.//' + q('rPr')))
    body_rpr = copy.deepcopy(o[7].find(q('r')).find(q('rPr')))

    BODY = [7, 8, 9, 10, 14, 16, 17, 18, 20, 25, 27, 28, 29, 30, 32, 33, 35, 36, 38, 39, 41,
            42, 43, 48, 49, 50, 54, 58, 59, 60, 61]

    # resumo / abstract + palavras-chave
    fill(o[4], ABSTRACT); set_ppr(o[4], before=240, after=0)
    kw = clone_after(o[4], o[4], KEYWORDS); set_ppr(kw, before=0, after=240)
    fill(o[5], RESUMO); set_ppr(o[5], before=0, after=0)
    pk = clone_after(o[5], o[5], PALAVRAS); set_ppr(pk, before=0, after=0)

    # textos do corpo
    import base_markup
    for i in BODY:
        txt = NEW.get(i, base_markup.TXT.get(i))
        fill(o[i], abnt(txt), body_rpr)
        set_ppr(o[i], before=0, after=0, first=709, jc='both')

    body.remove(o[11])  # objetivo geral e específicos (TCC-I)

    # termos estrangeiros em itálico também nos títulos (Guia Univille, 3.8)
    for i, mk in ((13, '2.1 Modelos de aprendizado profundo e _transfer learning_'),
                  (15, '2.2 Validade da avaliação: vazamento, _domain gap_ e calibração')):
        hr = copy.deepcopy(o[i].find('.//' + q('r')).find(q('rPr')))
        fill(o[i], '**' + mk + '**', hr)

    # Seção 2: texto introdutório
    li = clone_after(o[12], body_model, abnt(LIT_INTRO), body_rpr)
    set_ppr(li, before=0, after=0, first=709, jc='both')

    # legendas: "Quadro 1 – ...", alinhadas à esquerda; tabelas centralizadas
    for i in (21, 45, 51, 55):
        for tt in o[i].iter(q('t')):
            tt.text = re.sub(r'^((?:Quadro|Tabela) \d)\. ', r'\1 – ', tt.text or '')
        set_ppr(o[i], before=120, after=60, left=0, first=0, jc='left')
    # Quadro 1 mais compacto: autor e ano numa linha só
    q1 = o[22]
    widths = ['2150', '1150', '2300', '2904']
    for grid in q1.iter(q('tblGrid')):
        for gc, w in zip(grid.findall(q('gridCol')), widths): gc.set(q('w'), w)
    for tr in q1.findall(q('tr')):
        for tc, w in zip(tr.findall(q('tc')), widths):
            tw = tc.find(q('tcPr') + '/' + q('tcW'))
            if tw is not None: tw.set(q('w'), w); tw.set(q('type'), 'dxa')
    for tbl_ in body.iter(q('tbl')):
        for sp in tbl_.iter(q('spacing')):
            sp.set(q('before'), '0'); sp.set(q('after'), '0')
    # tabela inteira na mesma página: todas as linhas, menos a última, "manter com a próxima"
    QCEL = {'26% a 30%': 'Acurácia: 26% a 30%; 51% com DINOv2 congelado',
            '73,8%; ECE': 'Acurácia: 73,8%; ECE de 0,287 para 0,038',
            '97,3% (teste interno)': 'Acurácia: 97,3% (teste interno)',
            '75,4% (fonte retida, ciclo 3)': 'Acurácia balanceada: 75,4% (fonte retida); 85,8% a 90,1% (ciclo 5)'}
    for tc in o[22].iter(q('tc')):
        txt = ''.join(t.text or '' for t in tc.iter(q('t')))
        for k, v in QCEL.items():
            if txt.startswith(k):
                ts = list(tc.iter(q('t')))
                ts[0].text = v
                for t in ts[1:]: t.text = ''
    for tbl in body.iter(q('tbl')):
        for tr in tbl.findall(q('tr')):
            for p in tr.iter(q('p')):
                ppr = p.find(q('pPr'))
                kn = ppr.find(q('keepNext')) if ppr is not None else None
                if kn is None and ppr is not None:
                    kn = etree.Element(q('keepNext')); ppr.insert(0, kn)
                if kn is not None: kn.set(q('val'), '1')
    for tbl in body.iter(q('tbl')):
        tp = tbl.find(q('tblPr')); j = tp.find(q('jc'))
        if j is None: j = etree.SubElement(tp, q('jc'))
        j.set(q('val'), 'center')
        # Tabelas (dados estatísticos) abertas nas laterais, com linhas só no topo, sob o cabeçalho e
        # no fim (IBGE, 1993; Guia Univille 2.4.2). O Quadro 1 continua fechado.
        if tbl is o[22]:
            continue
        tb = tp.find(q('tblBorders'))
        for side in ('left', 'right', 'insideH', 'insideV'):
            e = tb.find(q(side))
            for a in list(e.attrib): del e.attrib[a]
            e.set(q('val'), 'nil')
        for tc in tbl.find(q('tr')).findall(q('tc')):
            tcpr = tc.find(q('tcPr'))
            bd = etree.Element(q('tcBorders'))
            b = etree.SubElement(bd, q('bottom'))
            for k, v in (('val', 'single'), ('sz', '4'), ('space', '0'), ('color', '000000')): b.set(q(k), v)
            shd = tcpr.find(q('shd'))
            if shd is not None: shd.addprevious(bd)
            else: tcpr.append(bd)

    # notas e fontes abaixo das tabelas, alinhadas à esquerda
    note_rpr = copy.deepcopy(o[47].find('.//' + q('r')).find(q('rPr')))
    for i, (nota, fonte) in NOTAS.items():
        base = copy.deepcopy(note_rpr)
        if nota:
            fill(o[i], nota, base); set_ppr(o[i], before=40, after=0, left=0, first=0, jc='both')
            o[i].find(q('pPr') + '/' + q('keepNext')).set(q('val'), '1')
            f = clone_after(o[i], o[i], fonte, base); set_ppr(f, before=0, after=240, left=0, first=0, jc='left')
            f.find(q('pPr') + '/' + q('keepNext')).set(q('val'), '0')
        else:
            fill(o[i], fonte, base); set_ppr(o[i], before=40, after=240, left=0, first=0, jc='left')
            qc = clone_after(o[i], body_model, abnt(QUADRO_COMENT), body_rpr)
            set_ppr(qc, before=0, after=0, first=709, jc='both')

    # Seção 4: frase que apresenta a Tabela 1
    t1 = clone_after(o[44], body_model, TAB1_INTRO, body_rpr)
    set_ppr(t1, before=0, after=0, first=709, jc='both')

    # Conclusão
    fill(o[63], abnt(CONCLUSAO[0]), body_rpr); set_ppr(o[63], before=0, after=0, first=709, jc='both')
    prev = o[63]
    for txt in CONCLUSAO[1:]:
        prev = clone_after(prev, o[63], abnt(txt), body_rpr)
    body.remove(o[64]); body.remove(o[65])

    # Referências (ABNT: alinhadas à esquerda, simples, uma linha em branco entre elas)
    model = o[67]; rbase = copy.deepcopy(model.find(q('r')).find(q('rPr')))
    for i in range(67, 96):
        body.remove(o[i])
    prev = o[66]
    for ref in REFS:
        prev = clone_after(prev, model, ref, rbase)
        set_ppr(prev, before=0, after=REF_GAP, left=0, first=0, jc='left', hanging_off=True)

    # ordem do schema OOXML (o Word acusa arquivo corrompido se ela não for seguida)
    from valida_ooxml import ORDEM
    for nome, seq in ORDEM.items():
        for el in t.getroot().iter(q(nome)):
            filhos = list(el)
            filhos.sort(key=lambda c: seq.index(etree.QName(c).localname) if etree.QName(c).localname in seq else len(seq))
            for c in filhos: el.append(c)
    t.write(dpath, xml_declaration=True, encoding='UTF-8', standalone=True)
    if os.path.exists(dst): os.remove(dst)
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            zout.write(os.path.join(tmp, item.filename), item.filename)
    shutil.rmtree(tmp)

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    build(sys.argv[1], sys.argv[2])
    print('ok ->', sys.argv[2])
