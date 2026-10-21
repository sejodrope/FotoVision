# Monta a v7 do artigo a partir do document.xml da v6: acrescenta o quinto ciclo (split de
# alface por fotografia original, pragas rotuladas, amostragem por cultura, lote retido).
# Uso: python build_v7.py <dir v6 descompactado> <dir de saida descompactado>
import re, sys, shutil, os
from xml.sax.saxutils import escape

src, dst = sys.argv[1], sys.argv[2]
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
path = os.path.join(dst, 'word', 'document.xml')
d = open(path, encoding='utf8').read()
b = d.find('<w:body>') + len('<w:body>')
head = d[:b]
ps = re.findall(r'<w:p[ >].*?</w:p>|<w:tbl>.*?</w:tbl>|<w:sectPr.*?</w:sectPr>', d[b:], re.S)
assert len(ps) == 102, len(ps)


def texto(p):
    return ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p))


def runs(text):
    """*italico*, **negrito**"""
    out = []
    for part in re.split(r'(\*\*.+?\*\*|\*.+?\*)', text):
        if not part:
            continue
        rpr = ''
        if part.startswith('**'):
            part, rpr = part[2:-2], '<w:rPr><w:b/></w:rPr>'
        elif part.startswith('*'):
            part, rpr = part[1:-1], '<w:rPr><w:i/></w:rPr>'
        out.append(f'<w:r>{rpr}<w:t xml:space="preserve">{escape(part)}</w:t></w:r>')
    return ''.join(out)


def P(text, style=None):
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else '<w:pPr/>'
    return f'<w:p>{ppr}{runs(text)}</w:p>'


def repl(i, text, inicio):
    """Troca o texto do parágrafo i mantendo o pPr (estilo) original."""
    assert texto(ps[i]).startswith(inicio), (i, texto(ps[i])[:60])
    ppr = re.match(r'<w:p[^>]*>(<w:pPr>.*?</w:pPr>|<w:pPr/>)?', ps[i], re.S).group(1) or '<w:pPr/>'
    return f'<w:p>{ppr}{runs(text)}</w:p>'


def sub(i, old, new):
    assert old in ps[i], (i, old)
    return ps[i].replace(old, new)


def TABLE(widths, rows):
    grid = ''.join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    trs = []
    for ri, row in enumerate(rows):
        tcs = []
        for ci, cell in enumerate(row):
            shd = '<w:shd w:val="clear" w:color="auto" w:fill="E7E6E6"/>' if ri == 0 else ''
            txt = f'**{cell}**' if ri == 0 else cell
            keep = '<w:keepNext/>' if ri < len(rows) - 1 else ''
            tcs.append(f'<w:tc><w:tcPr><w:tcW w:type="dxa" w:w="{widths[ci]}"/>{shd}</w:tcPr>'
                       f'<w:p><w:pPr><w:pStyle w:val="CelulaTabela"/>{keep}<w:jc w:val="left"/></w:pPr>{runs(txt)}</w:p></w:tc>')
        trs.append('<w:tr><w:trPr><w:cantSplit/></w:trPr>' + ''.join(tcs) + '</w:tr>')
    return ('<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:type="auto" w:w="0"/><w:jc w:val="center"/>'
            '<w:tblLayout w:type="fixed"/><w:tblLook w:firstColumn="1" w:firstRow="1" w:lastColumn="0" w:lastRow="0" '
            'w:noHBand="0" w:noVBand="1" w:val="04A0"/></w:tblPr><w:tblGrid>' + grid + '</w:tblGrid>' + ''.join(trs) + '</w:tbl>')


def linha_nova(tabela, modelo_idx, celulas, depois_idx):
    """Copia a linha `modelo_idx` da tabela, troca o texto das células e insere após `depois_idx`."""
    trs = re.findall(r'<w:tr[ >].*?</w:tr>', tabela, re.S)
    nova = trs[modelo_idx]
    tcs = re.findall(r'<w:tc>.*?</w:tc>', nova, re.S)
    assert len(tcs) == len(celulas), (len(tcs), len(celulas))
    for tc, cel in zip(tcs, celulas):
        ts = re.findall(r'<w:t[^>]*>[^<]*</w:t>', tc)
        novo_tc = tc.replace(ts[0], f'<w:t xml:space="preserve">{escape(cel)}</w:t>', 1)
        for t in ts[1:]:
            novo_tc = novo_tc.replace(t, '<w:t></w:t>', 1)
        nova = nova.replace(tc, novo_tc, 1)
    return tabela.replace(trs[depois_idx], trs[depois_idx] + nova, 1)


body = ps[0:4]

# ---------------------------------------------------------------- Abstract / Resumo
body.append(sub(4,
    'On a source never seen in training, the previous model version scored 75.4% (N = 484); on 13 field photographs the current one scores 83.8%, a sample too small to distinguish model versions.',
    'On a source never seen in training, an earlier version scored 75.4% (N = 484). A fifth cycle that splits lettuce by '
    'original photograph and balances labels within each crop cuts false alarms on field-photographed crops, at a cost in '
    'sensitivity. On 13 field photographs, versions are indistinguishable.'))
body.append(sub(5,
    'Em uma fonte nunca vista no treino, a versão anterior do modelo obteve 75,4% (N = 484); em 13 fotografias de campo, a atual obtém 83,8%, amostra insuficiente para distinguir versões do modelo.',
    'Em uma fonte nunca vista no treino, uma versão anterior obteve 75,4% (N = 484). Um quinto ciclo, que separa a alface '
    'pela fotografia original e equilibra os rótulos dentro de cada cultura, reduz os falsos alarmes em culturas fotografadas '
    'em lavoura, com perda de sensibilidade. Em 13 fotografias de campo, as versões são indistinguíveis.'))

body += ps[6:14]
body.append(repl(14,
    'Esta seção apresenta os modelos, os conceitos que sustentam a validade da avaliação e os trabalhos relacionados, '
    'retomados na Seção 4.', 'Esta seção'))
body.append(ps[15])
body.append(repl(16,
    'As CNNs extraem hierarquias de características visuais por camadas convolucionais e dominam a classificação de imagens '
    '(Russell; Norvig, 2022). Na versão original de 1.000 classes, a ResNet-50, que introduziu as conexões residuais, tem 25,6 '
    'milhões de parâmetros (He et al., 2016); a MobileNetV2, com convoluções separáveis e blocos residuais invertidos, 3,4 '
    'milhões (Sandler et al., 2018); e a EfficientNet-B0, que equilibra profundidade, largura e resolução por escalonamento '
    'composto, 5,3 milhões (Tan; Le, 2019). Os *Vision Transformers* substituem a convolução por autoatenção sobre blocos de '
    '16×16 pixels; o ViT-B/16 tem 86,6 milhões de parâmetros e menos viés indutivo espacial que as CNNs (Dosovitskiy et al., '
    '2021). O *transfer learning* reutiliza pesos aprendidos no ImageNet como ponto de partida (Faceli et al., 2023), com '
    'eficácia demonstrada no diagnóstico foliar (Mohanty; Hughes; Salathé, 2016), mas ampliar o volume de dados tem retorno '
    'decrescente se a diversidade das condições de captura não aumenta (Barbedo, 2018).', 'As CNNs extraem'))
body.append(ps[17])
body.append(repl(18,
    'Vazamento de dados designa a presença, no conjunto de avaliação, de informação indisponível na predição real, '
    'tipicamente por contaminação a partir do treino (Kaufman et al., 2012). Em imagens, a forma mais comum é a duplicação '
    'quase exata — a mesma fotografia com recorte, compressão ou rotação diferentes — distribuída entre treino e teste por um '
    'particionamento aleatório por arquivo. É um caso de *shortcut learning*: redes profundas exploram qualquer regularidade '
    'que reduza a perda, e um teste que compartilhe essa regularidade não a detecta (Geirhos et al., 2020).', 'Vazamento de dados'))
body.append(repl(19,
    'O *domain gap* é a discrepância entre o desempenho em condições controladas e em condições de uso: modelos treinados em '
    'imagens de estúdio podem cair para 31% a 72% de acurácia em fotografias de lavoura, com solo, sombra, sobreposição de '
    'folhas e iluminação variável (Ahmad; Gamal; Saraswat, 2023; Upadhyay et al., 2025). Aqui o salto é duplo, de estúdio '
    'para campo e de espécies-*proxy* para espécies-alvo, pois o PlantVillage não contém as hortaliças-alvo.', 'O domain gap'))
body.append(repl(20,
    'As probabilidades de redes profundas modernas são sistematicamente mal calibradas, o que se corrige com o *temperature '
    'scaling*, que divide os *logits* por um escalar *T* aprendido na validação sem alterar a classe predita (Guo et al., '
    '2017). A calibração é medida pelo Erro de Calibração Esperado (ECE), que compara confiança declarada e acurácia observada '
    'por faixas de confiança (Naeini; Cooper; Hauskrecht, 2015). Sobre um modelo calibrado constrói-se um classificador '
    'seletivo, que se abstém abaixo de um limiar τ, trocando cobertura por acurácia seletiva (Geifman; El-Yaniv, 2017), e o '
    'índice *Excess Green* (ExG = 2*g* − *r* − *b*) permite rejeitar antes da inferência imagens sem vegetação (Woebbecke et '
    'al., 1995).', 'As probabilidades'))
body.append(ps[21])
body.append(repl(22,
    'A busca por trabalhos relacionados foi feita na Web of Science, em outubro de 2026, combinando doença de plantas e '
    'aprendizado profundo com generalização em campo, vazamento de dados, calibração e hortaliças folhosas; o Quadro 1 resume '
    'os estudos mais próximos. Classificadores que passam de 99% no PlantVillage caem para 26% a 30% no conjunto de campo '
    'PlantDoc, e o ajuste fino completo de um modelo de fundação piora a transferência (Nguyen et al., 2026), embora '
    'arquiteturas com *transformers* sejam apontadas como mais robustas (Shafay et al., 2025). Dois estudos adotam protocolo '
    'semelhante ao deste: com amêndoa, a validação agrupada quase não altera o F1 macro, mas um teste com fontes disjuntas o '
    'reduz a 0,56 (Workneh et al., 2026); com videira, a EfficientNet-B0 obtém 73,8% em um conjunto externo (Ergün et al., '
    '2026). Não foi localizado estudo que reúna, para hortaliças folhosas, auditoria de vazamento, calibração com abstenção e '
    'avaliação fora da fonte de treino.', 'A busca por trabalhos'))
body.append(ps[23])

# Quadro 1: linha deste trabalho
q1 = sub(24, '98,48% para 75,4% (fonte retida)', '75,4% (fonte retida, ciclo 3); 85,8% a 90,1% (lote retido, ciclo 5)')
body += [q1, ps[25], ps[26], repl(27,
    'A pesquisa é quantitativa, exploratória e descritiva quanto aos objetivos e experimental quanto aos procedimentos, pois '
    'arquitetura, calibração, limiar de abstenção e domínio de avaliação são manipulados sob condições controladas e '
    'reprodutíveis (Gil, 2010); todo o *pipeline* roda por *script*, com semente fixa (42) e registro dos parâmetros e '
    'métricas de cada execução.', 'A pesquisa é quantitativa'), ps[28], repl(29,
    'O dataset binário (saudável × anômalo) reúne seis fontes públicas do Kaggle, entre elas o PlantVillage, como *proxy* '
    'visual, e datasets de alface. A rotulagem é automática, por busca em largura na árvore de diretórios, com cada decisão '
    'pasta-rótulo registrada em um mapa de auditoria. Contra o vazamento, cada imagem é reduzida a um *hash* perceptual '
    'invariante às oito transformações do grupo diedral D4; imagens a distância de Hamming ≤ 4 são agrupadas por união-busca, '
    'e cada grupo vai inteiro para uma partição (≈70/15/15, estratificada por classe), descartados os grupos com rótulos '
    'contraditórios. Das 176.126 imagens reunidas restam 170.772 (120.861 de treino, 25.944 de validação e 23.967 de teste), '
    'com vazamento residual de 0,04% no teste por esse critério. As partições provêm das mesmas seis fontes, e o teste mede '
    'desempenho intra-fonte. O critério não alcança variantes muito alteradas da mesma fotografia: 23% do teste são '
    'rotações, recortes e filtros de fotografias de alface com outra variante no treino.', 'O dataset binário'), ps[30], repl(31,
    'As três CNNs são inicializadas com pesos do ImageNet e ajustadas integralmente por até 30 épocas, com imagens de 224×224 '
    'pixels (AdamW, taxa de aprendizado de 10⁻³ com escalonador de cosseno, parada antecipada com paciência 10, entropia '
    'cruzada com *label smoothing* de 0,1 e amostragem ponderada pelo inverso da frequência de classe). O ViT-B/16 segue o '
    'mesmo protocolo de dados, perda, seleção e calibração, com hiperparâmetros de otimização usuais no ajuste fino de '
    '*transformers*: taxa de 5×10⁻⁵, decaimento de pesos de 0,05, duas épocas de aquecimento e limitação da norma do '
    'gradiente. Cada modelo é calibrado por *temperature scaling* na validação, e τ é o menor valor, a partir de 0,85, que '
    'assegura acurácia seletiva de ao menos 98% na validação. Em produção, o sistema devolve um de quatro veredictos: '
    'saudável, anômalo, inconclusivo (confiança < τ) ou não é folha (menos de 10% de pixels vegetativos por ExG).',
    'As três CNNs')]

# 3.1: o quinto ciclo
body.append(P(
    'Um quinto ciclo corrige três defeitos achados em auditoria posterior: a alface passa a ser particionada pela fotografia '
    'original, com uma por original na validação e no teste (176 no teste, 15 doentes); 8.266 imagens de pastas que a '
    'rotulagem deixava de fora, quase todas de dano por praga, entram como anômalas; e a amostragem equilibra os rótulos '
    'dentro de cada cultura, sem pesos na perda, porque cultura e rótulo estavam associados (87% da alface de treino era '
    'saudável). Um lote de espinafre-de-malabar (574 imagens) fica fora do treino, onde permanece outro lote da espécie. São '
    '152.057 imagens, 1,7% do teste a distância ≤ 4 do treino, e as demais fontes mantêm a partição, o que permite comparar '
    'os ciclos imagem a imagem.'))

body += ps[32:33]
body.append(repl(33,
    'Os dados de campo serão coletados em hortas da região de Joinville (SC), com telefone celular, sob luz natural e sem '
    'preparação da cena: de 100 a 300 fotografias, com mínimo de 15 e meta de 25 por cultura, proporção próxima de 50/50 entre '
    'folhas saudáveis e anômalas e variedade de sintomas (manchas, clorose, oídio ou míldio, perfurações, murcha). Para 20% a '
    '30% das folhas há três fotografias em distâncias diferentes (aproximada, média e ampla), e cada cultura inclui duas ou '
    'três imagens sem folha (solo, mão, parede), para verificar a guarda de vegetação.', 'Os dados de campo'))
body.append(repl(34,
    'Quatro regras garantem a validade do conjunto: (i) o rótulo é registrado na coleta, antes de qualquer inferência, e '
    'fotografias ambíguas são excluídas com o motivo anotado; (ii) fotografias de campo nunca entram no treino; (iii) versões '
    'do modelo são comparadas sobre as mesmas fotografias; e (iv) as 13 fotografias do TCC-I formam um conjunto histórico, '
    'reportado à parte.', 'Quatro regras'))
body.append(ps[35])
body.append(repl(36,
    'A análise responde às perguntas Q1 a Q3, e o Quadro 2 relaciona cada objetivo específico aos dados e às técnicas. '
    'Adota-se α = 0,05, com a correção sequencial de Holm (1979) nas comparações múltiplas sobre os mesmos dados.', 'A análise responde'))
body += ps[37:39]
body.append(repl(39,
    'Os falsos negativos — plantas doentes declaradas saudáveis — são reportados à parte, por serem o erro de maior custo '
    'agronômico. Os intervalos de confiança de 95% vêm de *bootstrap* estratificado por classe, com 10.000 reamostragens '
    '(Efron; Tibshirani, 1993), e, como as arquiteturas são avaliadas sobre as mesmas 23.967 imagens, pares de modelos são '
    'comparados pelo teste de McNemar, que considera só as imagens em que os dois discordam (McNemar, 1947; Dietterich, 1998).',
    'Os falsos negativos'))
body += ps[40:42]
body.append(repl(42,
    'O mecanismo de abstenção é avaliado pela cobertura (fração de imagens que recebem diagnóstico), pela acurácia seletiva '
    '(acurácia entre as diagnosticadas) e pela fração de erros interceptados (confiança inferior a τ); a curva '
    'risco–cobertura, com τ de 0,50 a 0,99, orienta o ponto de operação, e o teste de Mann-Whitney compara a confiança de '
    'acertos e erros.', 'O mecanismo de abstenção'))
body.append(repl(43,
    'Para Q3, cada fotografia de campo tem três desfechos: acerto, erro ou abstenção. As abstenções são reportadas como taxa '
    'própria, e as métricas de acerto, calculadas sobre os diagnósticos emitidos; as imagens sem folha medem a rejeição '
    'correta da guarda de vegetação.', 'Para Q3'))
body.append(repl(44,
    'Como as amostras de campo são pequenas, toda proporção leva intervalo de confiança de Wilson a 95% (Wilson, 1927), e a '
    'acurácia balanceada, intervalo por *bootstrap* estratificado. O *domain gap* é a diferença entre a acurácia balanceada no '
    'teste e em campo, e versões do modelo são comparadas sobre as mesmas fotografias pelo teste de McNemar exato. Para uma '
    'proporção de 80%, o intervalo de Wilson tem mais de 40 pontos de amplitude com 13 fotografias, 15,5 com 100 e 9,0 com '
    '300, o que define a meta de coleta.', 'Como as amostras'))
# 3.3, H1 a H3, mais curto
body.append(repl(45,
    'Ainda em Q3, a acurácia de campo é desagregada por três hipóteses exploratórias: o desempenho difere entre culturas, '
    'pois só a alface tem dados de treino específicos (H1); é menor em sintomas estruturais, como murcha e perfurações, do '
    'que em cromáticos, como manchas e clorose (H2); e diminui com a distância de captura (H3). H1 e H3 são testadas pelo '
    'teste exato de Freeman-Halton, H3 também pelo de Cochran-Armitage, e H2 pelo teste exato de Fisher (Agresti, 2013); as '
    'diferenças de proporções são reportadas com intervalo de Newcombe (1998), e os resultados, interpretados como '
    'exploratórios. Cada erro de campo é ainda examinado com mapas Grad-CAM (Selvaraju et al., 2017) e classificado quanto à '
    'causa provável: sintoma sutil, enquadramento amplo, fundo com vegetação, iluminação ou rótulo duvidoso.', 'Ainda em Q3'))
body.append(ps[47])
q2 = ps[48]
for a, b_ in (('Acurácia balanceada, F1 macro, AUC, falsos negativos; IC por bootstrap; McNemar com correção de Holm',
               'Acurácia balanceada, F1 macro, AUC, FN; bootstrap; McNemar e Holm'),
              ('ECE e diagrama de fiabilidade; cobertura, acurácia seletiva, erros interceptados; curva risco–cobertura; Mann-Whitney',
               'ECE; cobertura, acurácia seletiva, erros interceptados; Mann-Whitney'),
              ('Estatística descritiva por cultura, classe, sintoma e distância; registro de exclusões',
               'Descrição por cultura, classe, sintoma e distância'),
              ('Diferença de acurácia balanceada; IC de Wilson e bootstrap; McNemar exato entre versões',
               'Diferença de acurácia balanceada; Wilson; McNemar exato'),
              ('Fisher / Freeman-Halton; Cochran-Armitage; IC de Newcombe; Grad-CAM dos erros',
               'Freeman-Halton; Cochran-Armitage; Newcombe; Grad-CAM')):
    assert a in q2, a
    q2 = q2.replace(a, b_)
body += [q2, ps[49], ps[50]]

# ---------------------------------------------------------------- 4. Resultados
body.append(sub(51, 'Os resultados vão até o quarto ciclo de treinamento;',
                'As Tabelas 1 e 2 vão até o quarto ciclo de treinamento, e a Tabela 3 apresenta o quinto;'))
body += ps[52:55]
body.append(repl(55,
    'EfficientNet-B0 e ViT-B/16 empatam: mesma acurácia balanceada e mesmo IC 95% (98,32% a 98,64%), sem diferença pelo '
    'McNemar exato com correção de Holm (p = 0,26). Ambos superam o MobileNetV2 (97,70% a 98,07%) e o ResNet-50 (97,63% a '
    '98,00%), com p ≤ 1,4×10⁻¹², e estes não diferem entre si (p = 0,26), embora o segundo tenha dez vezes mais parâmetros. '
    'O ViT-B/16 comete menos falsos negativos (137 contra 209) e mais falsos positivos (204 contra 154), com 21 vezes mais '
    'parâmetros que o EfficientNet-B0 e latência 2,7 vezes maior. Isso responde a Q1 neste teste; com um único treino por '
    'modelo, a conclusão não se estende às arquiteturas em geral.', 'EfficientNet-B0 e ViT-B/16 empatam'))
body.append(repl(56,
    'Quanto a Q2, o *temperature scaling* reduziu o ECE na validação de 0,0509 para 0,0027 no EfficientNet-B0 e de 0,0472 '
    'para 0,0022 no ViT-B/16. No teste, a confiança calibrada do EfficientNet-B0 separa acertos de erros (0,990 contra 0,874; '
    'Mann-Whitney, p < 0,001), mas τ = 0,85 intercepta apenas 113 dos 363 erros (31,1%), com abstenção em 1,2%; no ViT-B/16, '
    '51 dos 341 (15,0%), com abstenção em 0,5%. O ViT-B/16 tem o menor ECE da Tabela 1 e a menor fração de erros '
    'interceptados: boa calibração média não garante que os erros fiquem nas confianças baixas.', 'Quanto a Q2'))
body.append(repl(57,
    'A Tabela 1 deve ser lida com uma ressalva: o teste é extraído das mesmas seis fontes do treino, de modo que os 98,48% '
    'são uma acurácia intra-fonte. No lote de espinafre-de-malabar que o modelo do terceiro ciclo nunca viu, a acurácia '
    'balanceada do EfficientNet-B0 foi de 75,4% (N = 484), 23 pontos abaixo, embora as imagens também sejam de folha isolada '
    'sobre fundo claro. Incorporado o lote ao treino, o resultado nele subiu para 88,4% (IC 95%: 85,3% a 91,2%; p de Holm = '
    '5,9×10⁻¹²), mas o ganho é intra-fonte: o modelo aprende a aparência das fontes de treino, e isso não informa sobre o '
    'campo.', 'A Tabela 1 deve'))
body.append(ps[58])
t2 = linha_nova(ps[59], 4, ['5', 'Alface por foto; amostragem', '12 de 13', '11', '91,7%', '[64,6%; 98,5%]'], 4)
assert len(re.findall(r'<w:tr[ >]', t2)) == len(re.findall(r'<w:tr[ >]', ps[59])) + 1
assert '91,7%' in t2 and 'Alface por foto' in t2
body.append(t2)
body.append(sub(60, 'Ciclos 1 a 4: EfficientNet-B0.', 'Ciclos 1 a 5: EfficientNet-B0.'))
body.append(repl(61,
    'A Tabela 2 mostra por que a análise estatística é central. Em pontos estimados, o desempenho de campo teria piorado nos '
    'três primeiros ciclos, mesmo com a correção de defeitos reais, como 3.972 imagens de alface saudável rotuladas como '
    'doentes; mas os intervalos de Wilson de todos os ciclos se sobrepõem, e com N = 13 versões e arquiteturas são '
    'indistinguíveis. A amostra revelou o defeito de rotulagem, mas não tem poder para decidir entre modelos. Os nove erros '
    'emitidos nos quatro primeiros ciclos, concentrados em quatro fotografias, tiveram confiança calibrada entre 0,949 e '
    '0,999: o limiar τ é ajustado na validação, que compartilha a distribuição do treino, e fora dela o ECE passou de 0,003 '
    'para 0,146.', 'A Tabela 2 mostra'))

# Tabela 3 e o resultado do quinto ciclo
body.append(P('Tabela 3. Quinto ciclo: acurácia balanceada (%) no teste e por subconjunto', 'LegendaArtigo'))
body.append(TABLE([1640, 760, 1380, 1380, 1380, 860, 1104], [
    ['Modelo', 'Teste', 'Lavoura', 'Espec.', 'Sens.', 'Alface', 'Retido'],
    ['EfficientNet-B0', '97,4', '86,3 → 88,8', '76,6 → 83,4', '96,0 → 94,2', '96,7', '87,9'],
    ['MobileNetV2', '96,9', '86,6 → 90,1', '78,3 → 90,2', '94,9 → 90,0', '100', '88,4'],
    ['ResNet-50', '96,9', '87,8 → 89,6', '81,4 → 89,0', '94,2 → 90,2', '100', '90,1'],
    ['ViT-B/16', '96,9', '83,4 → 84,5', '69,8 → 72,1', '97,0 → 96,9', '100', '85,8'],
]))
body.append(P(
    'Teste do ciclo 5: 18.413 imagens, sem variantes de alface. Lavoura: mandioca, arroz, cana e café (N = 4.530), com '
    'Espec. e Sens. no mesmo subconjunto; ciclo 4 → ciclo 5, mesmas imagens. Alface: N = 176. Retido: lote de espinafre, N = 574. Fonte: '
    'elaboração própria (2026).', 'FonteTabela'))
body.append(P(
    'Na Tabela 3, a correção agiu onde estavam os erros: nas culturas de lavoura, as três CNNs ganham de 1,8 a 3,5 pontos '
    'de acurácia balanceada e de 6,8 a 11,9 de especificidade, ou seja, emitem menos falsos alarmes em folhas sadias; o '
    'ViT-B/16 quase não muda (p = 0,66). O custo é a sensibilidade, que cai de 1,8 a 4,9 pontos: por imagem, o quarto ciclo '
    'acerta mais (p ≤ 0,04), pois três em cada quatro dessas imagens são anômalas, e o perfil preferível '
    'depende do custo de cada erro na horta. No lote retido, os modelos ficam entre 85,8% e 90,1%, cerca de dez pontos '
    'abaixo das fontes de treino; na alface sem variantes, entre 96,7% e 100%, com só 15 doentes. A validação, que põe '
    'EfficientNet-B0 e ViT-B/16 à frente (F1 macro de 0,972 contra 0,96), não ordena os modelos como os dados de fora.'))

body.append(repl(63,
    'Esses resultados são coerentes com o Quadro 1. A queda de 98,48% para 75,4% em uma fonte retida tem a ordem de grandeza '
    'dos 73,8% da EfficientNet-B0 em um conjunto externo de videira (Ergün et al., 2026) e, como no estudo com amêndoa, só '
    'aparece com a fonte separada (Workneh et al., 2026); o contraste com os 97,3% do espinafre-de-malabar em partição '
    'interna (Kabya et al., 2026) indica que a diferença está no protocolo de avaliação, e não na cultura. O empate entre '
    'ViT-B/16 e EfficientNet-B0 diverge da robustez atribuída aos *transformers* (Shafay et al., 2025), mas concorda com a '
    'evidência de que o ajuste fino completo não reduz o *domain gap* (Nguyen et al., 2026). A literatura atribui a perda à '
    'pouca diversidade das condições de captura (Barbedo, 2018), e o próprio teste a reproduz: as culturas fotografadas em '
    'lavoura têm 86,3% (IC 95%: 84,5% a 88,0%) no EfficientNet-B0 do quarto ciclo e concentram 81% dos seus erros, contra '
    '99,7% nas demais. Na calibração, o estudo de videira reduz o ECE do conjunto externo de 0,287 para 0,038 (Ergün et al., '
    '2026), enquanto aqui o ECE de campo fica em 0,146 após a calibração na validação, o que recomenda reservar fotografias de '
    'campo para recalibração.', 'Esses resultados'))

# ---------------------------------------------------------------- 5. Conclusao
body.append(ps[64])
p65 = sub(65, 'A questão central — quanto um sistema de triagem fitossanitária treinado em imagens públicas vale em condições reais de campo — permanece aberta:',
           'A questão central permanece aberta:')
p65 = p65.replace('No domínio de teste, Q1 e Q2 têm resposta para as quatro arquiteturas:', 'No domínio de teste,')
assert 'No domínio de teste, EfficientNet' in texto(p65), texto(p65)[:80]
ps[65] = p65
body.append(sub(65, 'com intervalo de 57,8% a 95,7%.',
                'com intervalo de 57,8% a 95,7%. O quinto ciclo mostra que parte do erro vinha de atalhos nos dados e que '
                'corrigi-los reduz os falsos alarmes em fotografias de lavoura, com alguma perda de sensibilidade.'))
body.append(repl(66,
    'A próxima etapa é a coleta de campo, com fotografias de fundo real, que responde a Q3 (Seção 3.3): os modelos do quarto '
    'e do quinto ciclos, congelados antes dela, serão comparados sobre as mesmas fotografias, separadas por horta em '
    'conjuntos de avaliação e de adaptação.', 'A próxima etapa'))
body += ps[67:]

open(path, 'w', encoding='utf8').write(head + ''.join(body) + '</w:body></w:document>')
print('paragrafos:', len(body))
