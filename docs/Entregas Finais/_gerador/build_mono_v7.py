# Monta a v7 da monografia (documento de estudo) a partir da v6: nota de 12/11/2026 e seção 9.8, com as
# limitações do estudo, as quatro fontes novas e o material da defesa (artigo v13, slides, roteiro).
# Localiza os pontos de inserção pelo texto, não por índice.
# Uso: python build_mono_v7.py <dir v6 descompactado> <dir de saida descompactado>
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


def texto(p):
    return ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p))


def acha(cond, ini=0):
    return next(i for i in range(ini, len(ps)) if cond(texto(ps[i])))


QUEBRA = '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'
NOTA27 = 'NOTA DE ATUALIZAÇÃO — 27 DE OUTUBRO DE 2026'
i_h97 = acha(lambda t: t.startswith('9.7') and 'campo' in t and len(t) < 80, 300)
i_ref = acha(lambda t: t == 'REFERÊNCIAS', i_h97)
i_quebra_refs = max(i for i in range(i_h97, i_ref) if ps[i] == QUEBRA)
i_nota27 = acha(lambda t: t == NOTA27)
fim_nota27 = next(i for i in range(i_nota27, len(ps)) if ps[i] == QUEBRA) + 1

PPR_TEXTO = re.match(r'<w:p><w:pPr>.*?</w:pPr>', ps[i_h97 + 1]).group(0)[5:]
RPR_TEXTO = '<w:rPr><w:b w:val="0"/><w:i w:val="0"/></w:rPr>'
i_bullet = next(i for i in range(i_h97, i_ref) if texto(ps[i]).startswith('O que o lote mostra.'))
PPR_BULLET = re.match(r'<w:p><w:pPr>.*?</w:pPr>', ps[i_bullet]).group(0)[5:]
H1_NOTA, H2_97 = ps[i_nota27], ps[i_h97]


def runs(text):
    out = []
    for part in re.split(r'(\*\*.+?\*\*|\*.+?\*)', text):
        if not part:
            continue
        extra = ''
        if part.startswith('**'):
            part, extra = part[2:-2], '<w:b/>'
        elif part.startswith('*'):
            part, extra = part[1:-1], '<w:i/>'
        rpr = f'<w:rPr>{extra}</w:rPr>' if extra else RPR_TEXTO
        out.append(f'<w:r>{rpr}<w:t xml:space="preserve">{escape(part)}</w:t></w:r>')
    return ''.join(out)


def P(text):
    return f'<w:p>{PPR_TEXTO}{runs(text)}</w:p>'


def BULLET(lead, text):
    return (f'<w:p>{PPR_BULLET}<w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">{escape(lead)}</w:t></w:r>'
            f'<w:r><w:t xml:space="preserve"> {escape(text)}</w:t></w:r></w:p>')


def clona(modelo, novo):
    t = re.findall(r'<w:t[^>]*>[^<]*</w:t>', modelo)
    assert t
    m = modelo.replace(t[0], f'<w:t xml:space="preserve">{escape(novo)}</w:t>', 1)
    for x in t[1:]:
        m = m.replace(x, '<w:t></w:t>', 1)
    return m


# ------------------------------------------------------------- nota de 12/11
nota = [
    clona(H1_NOTA, 'NOTA DE ATUALIZAÇÃO — 12 DE NOVEMBRO DE 2026'),
    P('Esta sétima versão é um documento de estudo para a defesa de 8 de dezembro. Acrescenta a seção 9.8, com as '
      'limitações do estudo que passaram a constar do artigo, as quatro fontes que as sustentam e o material da '
      'apresentação. O artigo está na versão 13, com 10 páginas e no template de 2026: ganhou o parágrafo “Limitações do '
      'estudo” no fim da Seção 4 e teve a lista de referências reduzida de 38 para 30. Não houve nova coleta de campo: os '
      'resultados são os da seção 9.7.'),
    QUEBRA,
]

# ------------------------------------------------------------- seção 9.8
secao = [
    clona(H2_97, '9.8  Limitações do estudo e material da defesa'),
    P('O artigo passou a declarar cinco limitações, logo antes da conclusão. Cada uma delimita até onde os resultados '
      'valem, e quatro delas se apoiam em trabalhos publicados em revistas indexadas.'),
    BULLET('Evidência de campo pequena.',
           'São 23 fotografias, todas de alface: 13 do conjunto histórico e 10 com fundo real, estas todas de plantas '
           'sadias. Sem folhas doentes na horta, a sensibilidade e o domain gap em campo não foram medidos, e os intervalos '
           'de Wilson das 13 fotografias não separam versões.'),
    BULLET('Mesmas fontes públicas no treino e no teste.',
           'Os 98,48% são uma acurácia intra-fonte. Xu et al. (2024) argumentam que os conjuntos públicos de doenças de '
           'plantas foram montados para servir aos modelos e estão longe das condições reais de uso. Além disso, parte das '
           'culturas entra apenas como aproximação visual das hortaliças-alvo.'),
    BULLET('Calibração ajustada fora do campo.',
           'A temperatura é aprendida na validação, que tem a distribuição do treino. Xiang, Shi e Zhu (2026) treinaram uma '
           'ResNet-50 no PlantVillage e a testaram em fotografias de campo: a acurácia caiu de 99,73% para 32,05%, com '
           'confiança média de 79,76%, e o temperature scaling reduziu o ECE de 0,476 para 0,365, ainda alto. É o mesmo '
           'padrão do ECE de 0,146 medido aqui em campo, contra 0,003 na validação.'),
    BULLET('Tarefa binária.',
           'O sistema diz se há anomalia, não qual é a doença. Barbedo (2016) lista, entre os desafios do diagnóstico por '
           'imagem, sintomas que variam e doenças diferentes com sintomas parecidos.'),
    BULLET('Um treino por modelo.',
           'Cada arquitetura foi treinada uma vez, com semente 42. Coakley e Gundersen (2026) treinaram cada configuração '
           '100 vezes, com sementes diferentes, e mostraram que o desempenho de redes como ResNet e ViT é uma distribuição, '
           'e que a média não basta para escolher modelo. O teste de McNemar compara os dois modelos treinados; o empate '
           'entre EfficientNet-B0 e ViT-B/16 pode não se repetir com outras sementes.'),
    P('Para caber em 10 páginas, saíram do artigo oito referências: os trabalhos originais de métodos estatísticos '
      'clássicos (McNemar, 1947; Holm, 1979; Wilson, 1927; Efron e Tibshirani, 1993; Newcombe, 1998; Agresti, 2013), '
      'Naeini, Cooper e Hauskrecht (2015) e Russell e Norvig (2022). Os métodos continuam descritos e citados pelo nome; o '
      'teste de McNemar segue apoiado em Dietterich (1998), e o ECE, em Guo et al. (2017). As quatro fontes novas foram '
      'localizadas em novembro de 2026, em base aberta, e são de revistas indexadas na Web of Science (*Frontiers in '
      'Plant Science*, *Scientific Reports* e *Biosystems Engineering*).'),
    P('**Material da defesa.** A apresentação segue o modelo de slides de 2026 da Univille e tem 16 slides: capa, '
      'problema, questão de pesquisa, sistema, método, o vazamento do TCC-I, Q1, Q2, o resultado central (99,7% em '
      'estúdio, 86,3% em lavoura e 75,4% em fonte nunca vista), o quinto ciclo, Q3, a literatura, as limitações, a '
      'conclusão, os trabalhos futuros e o agradecimento. O roteiro de falas soma 19 minutos e 15 segundos em ritmo '
      'calmo e está em docs/apresentacao_banca/ROTEIRO_FALAS_20MIN.md; o guia de estudo, com os conceitos, o cartão de '
      'números e 39 perguntas prováveis, está em docs/GUIA_BANCA_TCC2.md.'),
    P('**A tese em uma frase.** O desempenho depende mais do protocolo de avaliação do que da arquitetura: 98,48% nas '
      'fontes de treino, 75,4% em uma fonte nunca vista e 84,6% em 13 fotografias de campo, com intervalo de 57,8% a '
      '95,7%. O trabalho não entrega um modelo pronto para o campo; entrega a medida de quanto falta e o protocolo para '
      'medi-la.'),
]

# ------------------------------------------------------------- sumário manual
sum_ini = acha(lambda t: t == 'SUMÁRIO')
i_s27 = acha(lambda t: t == NOTA27, sum_ini)
i_s97 = acha(lambda t: t.startswith('9.7'), sum_ini)
assert fim_nota27 <= sum_ini < i_s27 < i_s97 < i_h97  # as notas vêm antes do sumário
ps[i_s27] = ps[i_s27] + clona(ps[i_s27], 'NOTA DE ATUALIZAÇÃO — 12 DE NOVEMBRO DE 2026')
ps[i_s97] = ps[i_s97] + clona(ps[i_s97], '9.8  Limitações do estudo e material da defesa')

out = (ps[:fim_nota27] + nota + ps[fim_nota27:i_quebra_refs] + secao + ps[i_quebra_refs:])
corpo = d[b:]
fim = corpo[corpo.rfind(ps[-1]) + len(ps[-1]):]
open(path, 'w', encoding='utf8').write(head + ''.join(out) + fim)
print('ok', len(out))
