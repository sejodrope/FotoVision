# Monta a v5 do artigo a partir do document.xml da v4.
# Uso: python build_v5.py <dir v4 descompactado> <dir de saida descompactado>
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
assert len(ps) == 109, len(ps)


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


def H1(t): return P(t, 'Heading1')
def H2(t): return P(t, 'Heading2')
def REF(t): return P(t, 'Reference')


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


def sub(i, old, new):
    assert old in ps[i], (i, old)
    return ps[i].replace(old, new)


body = []
body += ps[0:3]
body.append(sub(3, 'vieirasilvajosepedro@gmail.com / leandersonandre@univille.br', '{josepedro, leandersonandre}@univille.br'))
body += ps[4:6]               # abstract / resumo

# ---------------------------------------------------------------- 1. Introducao
body.append(ps[6])
body.append(P(
    'A inteligência artificial (IA) consolidou-se como uma das áreas mais ativas da computação, e o aprendizado profundo '
    'tornou-se a abordagem predominante em tarefas de percepção, como a classificação de imagens (Russell; Norvig, 2022). Redes '
    'neurais sustentam hoje aplicações em diagnóstico médico por imagem, inspeção industrial e agricultura de precisão, e o '
    '*transfer learning* tornou essas técnicas acessíveis a problemas com poucos dados rotulados, ao reaproveitar modelos '
    'pré-treinados (Faceli et al., 2023).'))
body.append(P(
    'Na agricultura, uma das aplicações mais estudadas é o diagnóstico de doenças de plantas a partir de fotografias de folhas. '
    'O tema é relevante no Brasil, onde 77% dos estabelecimentos rurais pertencem à agricultura familiar (IBGE, 2019), segmento '
    'no qual o diagnóstico fitossanitário ainda depende predominantemente de inspeção visual manual. Hortaliças folhosas como alface, rúcula, espinafre, acelga e couve '
    'são culturas de ciclo curto e alta perecibilidade, nas quais o atraso na identificação de doenças, pragas ou deficiências '
    'nutricionais se traduz rapidamente em perda de produção.'))
body.append(P(
    'Os trabalhos relacionados mostram um contraste entre laboratório e campo. Redes neurais convolucionais (CNNs) treinadas no '
    'dataset PlantVillage superam 99% de acurácia (Mohanty; Hughes; Salathé, 2016), mas sobre imagens de estúdio — folha isolada, '
    'fundo homogêneo, iluminação controlada. Em fotografias de lavoura a acurácia cai, fenômeno denominado *domain gap* '
    '(Barbedo, 2018), e uma revisão sistemática recente situa o campo entre 70% e 85% (Shafay et al., 2025). Estudos recentes passaram a combinar validação sem vazamento de '
    'dados, calibração da confiança e abstenção (Ergün et al., 2026; Workneh et al., 2026), mas em culturas como videira e '
    'amêndoa; para hortaliças folhosas os trabalhos são menos numerosos e avaliados em partições do próprio conjunto de dados '
    '(Abbasi; Martinez; Ahmad, 2023; Kabya et al., 2026).'))
body.append(P(
    'O problema deste trabalho nasce dessa lacuna e de um resultado próprio. Na primeira fase (TCC-I), uma auditoria por *hash* '
    'perceptual revelou que 57,35% do conjunto de teste era composto por fotografias quase idênticas a imagens de treino, o que '
    'invalidou a acurácia de 99,01% inicialmente obtida. Após a reconstrução do particionamento, o melhor modelo atingiu 98,49% '
    'de acurácia balanceada no domínio de teste, mas apenas 73,3% em uma validação exploratória com 13 fotografias de horta real. '
    'A questão passa a ser quanto um sistema desse tipo vale em uso real; e, como um sistema que erra com aparência de certeza transfere confiança indevida ao produtor, a calibração da confiança e a '
    'capacidade de abster-se fazem parte do objeto de estudo.'))
body.append(P(
    'Diante do exposto, a questão de pesquisa é: em que medida um sistema de triagem fitossanitária para hortaliças folhosas, '
    'baseado em *transfer learning*, treinado com imagens públicas auditadas contra vazamento de dados e dotado de calibração de '
    'confiança e mecanismo de abstenção, mantém desempenho confiável em fotografias capturadas em condições reais de campo? Ela se '
    'desdobra em três perguntas operacionais: (Q1) qual arquitetura apresenta o melhor desempenho no domínio de teste, com '
    'significância estatística; (Q2) em que medida as probabilidades calibradas e o limiar de abstenção interceptam os erros do '
    'modelo; e (Q3) qual é a magnitude do *domain gap* entre laboratório e campo e se ele varia conforme a cultura, o tipo de '
    'manifestação do sintoma e a distância de captura.'))
body.append(P(
    'O objetivo geral é avaliar o desempenho e a confiabilidade de um sistema de detecção automática de anomalias fitossanitárias '
    'em hortaliças folhosas, baseado em *transfer learning* com as redes convolucionais EfficientNet-B0, MobileNetV2 e ResNet-50 '
    'e, como arquitetura de contraste, o *Vision Transformer* ViT-B/16, quantificando com rigor estatístico a diferença entre o '
    'desempenho em laboratório e em condições reais de campo. Os objetivos específicos são: (a) consolidar o conjunto de dados de '
    'treinamento a partir de fontes públicas, com deduplicação e particionamento agrupado, reportando o vazamento residual; (b) comparar as arquiteturas no conjunto de teste, com métricas robustas ao '
    'desbalanceamento de classes e teste estatístico pareado; (c) avaliar a calibração das probabilidades e a eficácia da '
    'abstenção em interceptar erros; (d) construir um conjunto de validação de campo com gabarito, abrangendo mais de uma cultura, '
    'diferentes tipos de sintoma e distâncias de captura; (e) quantificar o *domain gap* entre laboratório e campo, com intervalos '
    'de confiança e testes adequados a amostras pequenas; e (f) investigar se o desempenho em campo varia conforme a cultura, o '
    'tipo de manifestação do sintoma (cromática ou estrutural) e a distância de captura.'))

# ---------------------------------------------------------------- 2. Revisao
body.append(ps[22])
body.append(P(
    'Esta seção apresenta os modelos utilizados e o *transfer learning*, os conceitos que sustentam a validade da avaliação e '
    'os trabalhos relacionados, com os quais os resultados são comparados na Seção 4.'))
body.append(H2('2.1 Modelos de aprendizado profundo e transfer learning'))
body.append(P(
    'As CNNs extraem hierarquias de características visuais por camadas convolucionais e dominam a classificação de imagens '
    '(Russell; Norvig, 2022). A ResNet-50 introduziu conexões residuais e tem, na versão original de 1.000 classes, 25,6 milhões de parâmetros (He et al., 2016); a '
    'MobileNetV2 usa convoluções separáveis e blocos residuais invertidos, com 3,4 milhões '
    '(Sandler et al., 2018); e a EfficientNet-B0 equilibra profundidade, largura e resolução por escalonamento composto, com 5,3 '
    'milhões (Tan; Le, 2019). Os *Vision Transformers* substituem a convolução por autoatenção sobre blocos de 16×16 pixels; o '
    'ViT-B/16 tem 86,6 milhões de parâmetros e menos viés indutivo espacial que as CNNs (Dosovitskiy et al., 2021). O *transfer '
    'learning* reutiliza pesos aprendidos no ImageNet como ponto de partida para a tarefa-alvo (Faceli et al., 2023), com eficácia '
    'demonstrada no diagnóstico foliar (Mohanty; Hughes; Salathé, 2016). Ampliar o volume de dados, contudo, '
    'tem retorno decrescente se a diversidade das condições de captura não aumenta (Barbedo, 2018).'))
body.append(H2('2.2 Validade da avaliação: vazamento, domain gap e calibração'))
body.append(P(
    'Vazamento de dados designa a presença, no conjunto de avaliação, de informação indisponível no momento da predição real, '
    'tipicamente por contaminação a partir do treino (Kaufman et al., 2012). Em imagens, a forma mais comum é a duplicação quase '
    'exata: a mesma fotografia com recorte, compressão ou rotação diferentes, distribuída entre treino e teste por um '
    'particionamento aleatório por arquivo. O fenômeno enquadra-se no conceito mais amplo de *shortcut learning*: redes profundas '
    'exploram qualquer regularidade que reduza a perda, e um teste que compartilhe essa regularidade não a detecta (Geirhos et '
    'al., 2020).'))
body.append(P(
    'O *domain gap* é a discrepância entre o desempenho em condições controladas e em condições de uso. Modelos treinados em '
    'imagens de estúdio podem cair para a faixa de 31% a 72% de acurácia em fotografias de lavoura, que contêm solo, sombra, '
    'sobreposição de folhas e iluminação variável (Ahmad; Gamal; Saraswat, 2023; Upadhyay et al., 2025). Neste trabalho o salto é '
    'duplo: de estúdio para campo e de espécies-*proxy* para espécies-alvo, pois o PlantVillage não contém as hortaliças-alvo.'))
body.append(P(
    'As probabilidades produzidas por redes profundas modernas são sistematicamente mal calibradas, o que se corrige com o '
    '*temperature scaling*, que divide os *logits* por um escalar *T* aprendido na validação sem alterar a classe predita (Guo et '
    'al., 2017). A calibração é medida pelo Erro de Calibração Esperado (ECE), que compara confiança declarada e acurácia '
    'observada em faixas de confiança (Naeini; Cooper; Hauskrecht, 2015). Sobre um modelo calibrado constrói-se um classificador '
    'seletivo, que se abstém abaixo de um limiar τ, trocando cobertura por acurácia seletiva (Geifman; El-Yaniv, 2017). '
    'Complementarmente, o índice *Excess Green* (ExG = 2*g* − *r* − *b*) permite rejeitar imagens sem vegetação antes da '
    'inferência (Woebbecke et al., 1995).'))
body.append(H2('2.3 Trabalhos relacionados'))
body.append(P(
    'A busca por trabalhos relacionados foi feita na Web of Science, em outubro de 2026, com termos de doença de plantas e '
    'aprendizado profundo combinados a generalização em campo, vazamento de dados, calibração e hortaliças folhosas; o Quadro 1 '
    'resume os estudos mais próximos. Classificadores que passam de 99% no PlantVillage caem para 26% a 30% no conjunto de '
    'campo PlantDoc, e o ajuste fino completo de um modelo de fundação piora a transferência (Nguyen et al., 2026); arquiteturas com *transformers* são, ainda assim, apontadas como mais robustas (Shafay et al., 2025). Dois estudos adotam protocolo semelhante ao deste trabalho: com '
    'amêndoa, a validação agrupada quase não altera o F1 macro, mas um teste com fontes disjuntas o reduz a 0,56 (Workneh et al., '
    '2026); com videira, a EfficientNet-B0 obtém 73,8% em um conjunto externo (Ergün et al., 2026). Não foi localizado estudo que '
    'reúna, para hortaliças folhosas, auditoria de vazamento, calibração com abstenção e avaliação fora da fonte de treino.'))
body.append(P('Quadro 1. Trabalhos relacionados e comparação com este trabalho', 'LegendaArtigo'))
body.append(TABLE([1701, 1304, 2600, 2899], [
    ['Trabalho', 'Cultura', 'Protocolo de avaliação', 'Resultado fora da origem'],
    ['Nguyen et al. (2026)', 'Várias', 'PlantVillage → PlantDoc; McNemar', '26% a 30%; 51% com DINOv2 congelado'],
    ['Ergün et al. (2026)', 'Videira', 'Conjunto externo; calibração', '73,8%; ECE de 0,287 para 0,038'],
    ['Workneh et al. (2026)', 'Amêndoa', 'Validação agrupada; calibração; abstenção', 'F1 macro de 0,98 para 0,56'],
    ['Kabya et al. (2026)', 'Espinafre-de-malabar', 'Partição do próprio conjunto', '97,3% (teste interno)'],
    ['Este trabalho', 'Hortaliças folhosas', 'Deduplicação; calibração; abstenção',
     '98,48% para 75,4% (fonte retida)'],
]))
body.append(P('Fonte: elaboração própria (2026), a partir dos trabalhos citados.', 'FonteTabela'))

# ---------------------------------------------------------------- 3. Metodologia
body.append(ps[30])
body.append(P(
    'A pesquisa é quantitativa quanto à abordagem, exploratória e descritiva quanto aos objetivos e experimental quanto aos '
    'procedimentos técnicos, pois arquitetura, calibração, limiar de abstenção e domínio de avaliação são manipulados sob '
    'condições controladas e reprodutíveis (Gil, 2010). Todo o *pipeline* é executado por *script*, com semente fixa (42) e '
    'registro dos parâmetros e métricas de cada execução.'))
body.append(H2('3.1 Dados, modelos e treinamento'))
p34 = sub(34, ' (≈70/15/15, estratificada por classe). A versão atual contém 170.772 imagens — 120.861 de treino, 25.944 de validação e 23.967 de teste — com vazamento residual de 0,04% no teste, métrica usada como critério de aceitação do particionamento. As três',
          ' (≈70/15/15, estratificada por classe), descartando-se os grupos com rótulos contraditórios. Das 176.126 imagens reunidas restam 170.772 — 120.861 de treino, 25.944 de validação e 23.967 de teste —, com vazamento residual de 0,04% no teste por esse critério. As três')
p34 = p34.replace('o teste mede, portanto, desempenho intra-fonte.',
                  'o teste mede, portanto, desempenho intra-fonte. O critério não alcança, porém, variantes muito alteradas da mesma fotografia: 23% do teste são rotações, recortes e filtros de fotografias de alface com outra variante no treino.')
assert 'com outra variante no treino' in p34
p35 = sub(35, 'particionados pela fotografia de origem.', 'particionados pela fotografia de origem, o que eleva o conjunto a 175.123 imagens.')
body += [p34, p35, sub(37, ' O ViT-B/16 foi treinado em GPU L4, no Google Colab; as CNNs, em GPU RTX 3050.', '').replace(' Compara-se, portanto, cada arquitetura com uma receita adequada a ela, e não com hiperparâmetros idênticos.', '')]
body.append(H2('3.2 Coleta dos dados de campo'))
body.append(sub(39, ' — priorizando cobrir bem ao menos duas culturas além da alface, em vez de cobrir superficialmente as cinco', ''))
body.append(P(
    'Quatro regras garantem a validade do conjunto: (i) o rótulo é registrado no momento da coleta, antes de qualquer '
    'inferência, e fotografias ambíguas são excluídas com o motivo anotado; (ii) as fotografias de campo nunca são usadas para '
    'treinamento; (iii) versões do modelo são comparadas sobre exatamente as mesmas fotografias; e (iv) as 13 fotografias já '
    'utilizadas no TCC-I formam um conjunto histórico, reportado à parte.'))
body.append(H2('3.3 Análise dos dados'))
body.append(sub(42, 'A análise é organizada em quatro blocos, que respondem às perguntas operacionais Q1 a Q3. O Quadro 1',
                'A análise responde, em sequência, às perguntas operacionais Q1 a Q3. O Quadro 2').replace(' As análises são implementadas em Python.', ''))
body.append(sub(44, 'Considerando a classe anômala como positiva,',
                'Para o desempenho no domínio de teste (Q1), considerando a classe anômala como positiva,'))
body.append(ps[45])
body.append(P(
    'Os falsos negativos — plantas doentes declaradas saudáveis — são reportados separadamente, por constituírem o erro de maior '
    'custo agronômico. Os intervalos de confiança de 95% são estimados por *bootstrap* estratificado por classe, com 10.000 '
    'reamostragens (Efron; Tibshirani, 1993). Como todas as arquiteturas são avaliadas sobre as mesmas 23.967 imagens, pares de '
    'modelos são comparados pelo teste de McNemar, que considera apenas as imagens em que os dois modelos discordam (McNemar, '
    '1947; Dietterich, 1998).'))
body.append(sub(50, 'A calibração é avaliada pelo ECE', 'Para Q2, a calibração é avaliada pelo ECE').replace('(Equação 3)', '(Equação 2)'))
body.append(sub(51, '(3)', '(2)'))
body.append(P(
    'O mecanismo de abstenção é avaliado pela cobertura (fração de imagens que recebem diagnóstico), pela acurácia seletiva '
    '(acurácia entre as imagens diagnosticadas) e pela fração de erros interceptados (erros com confiança inferior a τ). A curva '
    'risco–cobertura, com τ entre 0,50 e 0,99, permite discutir o ponto de operação, e o teste de Mann-Whitney compara a '
    'confiança calibrada de acertos e erros.'))
body.append(sub(54, 'Em campo, cada fotografia', 'Para Q3, em campo, cada fotografia').replace(', de forma uniforme em todas as versões do modelo', ''))
body.append(P(
    'Como as amostras de campo são pequenas, toda proporção é acompanhada do intervalo de confiança de Wilson a 95% (Wilson, 1927); para a acurácia balanceada, o intervalo é estimado por *bootstrap* '
    'estratificado. O *domain gap* é definido como a diferença entre a acurácia balanceada no teste e em campo, e versões do '
    'modelo são comparadas sobre as mesmas fotografias pelo teste de McNemar exato. O tamanho de amostra pretendido decorre da '
    'amplitude do intervalo de Wilson para uma proporção de 80%: mais de 40 pontos percentuais com as 13 fotografias atuais, 15,5 '
    'com 100 e 9,0 com 300.'))
body.append(P(
    'Ainda em Q3, a acurácia de campo é desagregada por três fatores, cada um com uma hipótese exploratória: (H1) o desempenho '
    'difere entre culturas, já que apenas a alface possui dados de treino específicos; (H2) o modelo erra mais em sintomas '
    'estruturais (murcha, perfurações) do que em cromáticos (manchas, clorose, revestimentos); e (H3) o desempenho diminui com a '
    'distância de captura. H1 e H3 são testadas pelo teste exato de Fisher generalizado (Freeman-Halton), H3 também pelo teste de '
    'tendência de Cochran-Armitage, e H2 pelo teste exato de Fisher em tabela 2×2 (Agresti, 2013). Como os subgrupos serão '
    'pequenos, reportam-se as diferenças de proporções com intervalo de confiança (Newcombe, 1998), e os resultados são '
    'interpretados como exploratórios.'))
body += [sub(59, ', gerados sobre a última camada de características de cada arquitetura,', ''), sub(60, 'Quadro 1.', 'Quadro 2.'), ps[61], ps[62]]

# ---------------------------------------------------------------- 4. Resultados
body.append(ps[63])
body.append(sub(64, 'Os resultados vão até o quarto ciclo de treinamento. Q1 e Q2 já têm resposta no domínio de teste para as quatro arquiteturas; as análises de campo (Q3) dependem da coleta descrita na Seção 3.4.',
                'Os resultados vão até o quarto ciclo de treinamento; as análises de campo (Q3) dependem da coleta descrita na Seção 3.2.'))
t1 = ps[66]
for a, b in (('5,3 M', '4,0 M'), ('3,4 M', '2,2 M'), ('25,6 M', '23,5 M'), ('86,6 M', '85,8 M')):
    assert t1.count('>' + a + '<') == 1, a
    t1 = t1.replace('>' + a + '<', '>' + b + '<')
p67 = sub(67, 'FN = falsos negativos.', 'Parâm. = modelo treinado, com saída de duas classes. FN = falsos negativos.')
p68 = sub(68, 'embora o segundo tenha 7,5 vezes mais parâmetros', 'embora o segundo tenha dez vezes mais parâmetros')
p68 = p68.replace('com 16 vezes mais parâmetros que o EfficientNet-B0', 'com 21 vezes mais parâmetros que o EfficientNet-B0')
assert '21 vezes' in p68
p75 = sub(75, 'Os nove erros emitidos em campo nos quatro ciclos tiveram', 'Os nove erros emitidos em campo nos quatro ciclos, concentrados em quatro fotografias, tiveram')
p74 = sub(74, 'A amostra atual foi suficiente para revelar o defeito de rotulagem, invisível às métricas de laboratório, mas não tem poder para decidir entre modelos, o que justifica a coleta planejada.',
          'A amostra revelou o defeito de rotulagem, mas não tem poder para decidir entre modelos, o que justifica a coleta planejada.')
body += [ps[65], t1, p67, p68] + ps[69:74] + [p74, p75]
body.append(P(
    'Esses resultados são coerentes com os trabalhos do Quadro 1. A queda de 98,48% para 75,4% em uma fonte retida tem a mesma '
    'ordem de grandeza dos 73,8% da EfficientNet-B0 em um conjunto externo de videira (Ergün et al., 2026) e, como no estudo com '
    'amêndoa, só aparece quando a fonte é separada (Workneh et al., 2026); o contraste com os 97,3% reportados para o '
    'espinafre-de-malabar em partição interna (Kabya et al., 2026) indica que a diferença está no protocolo de avaliação, e não '
    'na cultura. O empate entre ViT-B/16 e EfficientNet-B0 diverge da robustez atribuída aos *transformers* (Shafay et al., '
    '2025), mas concorda com a evidência de que o ajuste fino completo não reduz o *domain gap* (Nguyen et al., 2026). A literatura atribui a perda à pouca diversidade das condições de captura no treino (Barbedo, 2018), e o próprio teste a reproduz: dentro de uma mesma fonte, as culturas fotografadas em lavoura (mandioca, arroz, cana-de-açúcar e café; N = 4.530) têm acurácia balanceada de 86,3% (IC 95%: 84,5% a 88,0%) no EfficientNet-B0 e concentram 81% dos seus erros, contra 99,7% nas demais culturas (N = 10.061); nas de lavoura, as quatro arquiteturas ficam entre 83,4% e 87,8%. Na '
    'calibração, o estudo de videira reduz o ECE do conjunto externo de 0,287 para 0,038 (Ergün et al., 2026), ao passo que aqui '
    'o ECE de campo permanece em 0,146 após a calibração feita na validação, o que recomenda reservar fotografias de campo para '
    'recalibração.'))

# ---------------------------------------------------------------- 5. Conclusao
body.append(ps[76])
body.append(sub(77, 'um transformer com 16 vezes mais parâmetros', 'um transformer com 21 vezes mais parâmetros'))
body.append(sub(78, 'pré-requisito para as análises das Seções 3.5.3 e 3.5.4', 'pré-requisito para as análises de Q3 (Seção 3.3)'))

# ---------------------------------------------------------------- Referencias
body.append(ps[79])
refs = [p for p in ps[80:108] if 'TOO, E. C.' not in p]
assert len(refs) == 27
new_refs = [
    REF('ABBASI, R.; MARTINEZ, P.; AHMAD, R. Crop diagnostic system: a robust disease detection and management system for leafy '
        'green crops grown in an aquaponics facility. **Artificial Intelligence in Agriculture**, v. 10, p. 1-12, 2023. DOI: '
        '10.1016/j.aiia.2023.09.001.'),
    REF('ERGÜN, E. et al. Confidence and uncertainty aware deep learning for reliable grape leaf disease diagnosis under real '
        'world field conditions. **Frontiers in Plant Science**, v. 17, 1839652, 2026. DOI: 10.3389/fpls.2026.1839652.'),
    REF('KABYA, N. D. et al. Towards practical AI for agriculture: a self-supervised attention framework for spinach leaf disease '
        'detection. **PLOS ONE**, v. 21, n. 1, e0340989, 2026. DOI: 10.1371/journal.pone.0340989.'),
    REF('NGUYEN, T. A. et al. From laboratory to field: frozen foundation-model features toward robust plant disease recognition. '
        '**Smart Agricultural Technology**, v. 15, 102531, 2026. DOI: 10.1016/j.atech.2026.102531.'),
    REF('SHAFAY, M. et al. Recent advances in plant disease detection: challenges and opportunities. **Plant Methods**, v. 21, '
        '140, 2025. DOI: 10.1186/s13007-025-01450-0.'),
    REF('WORKNEH, A. D. et al. Leakage-aware, calibrated, and explainable deep learning for robust almond disease classification. '
        '**Computers and Electronics in Agriculture**, v. 256, 112464, 2026. DOI: 10.1016/j.compag.2026.112464.'),
]
refs += new_refs


def key(p):
    return ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p)).upper()


refs.sort(key=key)
body += refs
body.append(ps[108])

open(path, 'w', encoding='utf8').write(head + ''.join(body) + '</w:body></w:document>')
print('paragrafos:', len(body))
