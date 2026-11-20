# Monta a apresentação da defesa a partir do modelo da Univille (TCC_ModeloSlides_2026.pptx).
# Uso: python build_slides.py <modelo.pptx> <saida.pptx> <raiz do projeto>
# As falas vêm de roteiro_data.py e vão para as anotações do orador de cada slide.
import copy, os, sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn
from roteiro_data import SLIDES

MODELO, SAIDA, RAIZ = sys.argv[1], sys.argv[2], sys.argv[3]
DG, G, LG, OR, LOR = '1F5130', '92D050', 'E6F4D7', 'C8600A', 'FBE9D7'
TX, MU, WH, GR = '333333', '666666', 'FFFFFF', 'D9D9D9'

prs = Presentation(MODELO)
capa, base, _metod, fim = list(prs.slides)
LAYOUT = base.slide_layout


def rgb(h):
    return RGBColor.from_string(h)


def novo_slide(secao, rodape):
    s = prs.slides.add_slide(LAYOUT)
    for sh in base.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip() == 'Texto':
            continue
        el = copy.deepcopy(sh._element)
        if sh.shape_type == 13:  # imagem: recria a relação com a mesma parte
            blip = el.find('.//' + qn('a:blip'))
            parte = base.part.related_part(blip.get(qn('r:embed')))
            blip.set(qn('r:embed'), s.part.relate_to(parte, RT.IMAGE))
        s.shapes._spTree.append(el)
    for sh in s.shapes:
        if not sh.has_text_frame:
            continue
        t = sh.text_frame.text
        if t.strip() == 'Introdução':
            troca(sh, secao)
        elif 'Slide/P' in t:
            troca(sh, rodape)
            sh.text_frame.paragraphs[0].alignment = PP_ALIGN.RIGHT
    return s


def troca(sh, texto, size=None):
    """Troca o texto de uma forma mantendo a formatação do primeiro run de cada parágrafo."""
    linhas = texto.split('\n')
    pars = sh.text_frame.paragraphs
    for i, p in enumerate(pars):
        runs = p.runs
        if i < len(linhas) and runs:
            runs[0].text = linhas[i]
            if size:
                runs[0].font.size = Pt(size)
            for r in runs[1:]:
                r.text = ''
        else:
            for r in runs:
                r.text = ''


def tb(s, x, y, w, h, linhas, size=32, bold=False, color=TX, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
       italic=False, espaco=0, ls=None):
    """Caixa de texto. `linhas`: str, ou lista de str, ou lista de listas de (texto, {opções})."""
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    if isinstance(linhas, str):
        linhas = [linhas]
    for i, ln in enumerate(linhas):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if espaco and i:
            p.space_before = Pt(espaco)
        if ls:
            p.line_spacing = ls
        partes = ln if isinstance(ln, list) else [(ln, {})]
        for txt, o in partes:
            r = p.add_run()
            r.text = txt
            f = r.font
            f.size = Pt(o.get('size', size))
            f.bold = o.get('bold', bold)
            f.italic = o.get('italic', italic)
            f.color.rgb = rgb(o.get('color', color))
    return box


def caixa(s, x, y, w, h, fill=WH, raio=0.06, forma=MSO_SHAPE.ROUNDED_RECTANGLE):
    sh = s.shapes.add_shape(forma, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(fill)
    sh.line.fill.background()
    sh.shadow.inherit = False
    if forma == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = raio
    return sh


def bola(s, x, y, d, texto, fill=DG, color=WH, size=30):
    sh = caixa(s, x, y, d, d, fill, forma=MSO_SHAPE.OVAL)
    tf = sh.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = texto
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.color.rgb = rgb(color)
    return sh


def seta(s, x, y, w=0.9, h=0.7, fill=G):
    return caixa(s, x, y, w, h, fill, forma=MSO_SHAPE.RIGHT_ARROW)


def titulo(s, texto):
    tb(s, X0, 2.55, 21.6, 1.2, texto, size=54, bold=True, color=DG)


def numero(s, x, y, w, h, num, rotulo, cor=DG, fill=WH, nsize=96, rsize=28, rcor=TX):
    caixa(s, x, y, w, h, fill)
    tb(s, x + 0.45, y + 0.35, w - 0.9, h * 0.5, num, size=nsize, bold=True, color=cor)
    tb(s, x + 0.45, y + h * 0.56, w - 0.9, h * 0.4, rotulo, size=rsize, color=rcor)


def faixa(s, y, texto, fill=LG, cor=DG, h=1.15, size=32):
    caixa(s, X0, y, CW, h, fill, raio=0.2)
    tb(s, X0 + 0.5, y, CW - 1.0, h, texto, size=size, bold=True, color=cor, anchor=MSO_ANCHOR.MIDDLE)


def tabela(s, x, y, w, linhas, larguras, destaque=(), size=26, h_linha=0.85):
    n, m = len(linhas), len(linhas[0])
    gf = s.shapes.add_table(n, m, Inches(x), Inches(y), Inches(w), Inches(h_linha * n))
    t = gf.table
    tot = sum(larguras)
    for j, lw in enumerate(larguras):
        t.columns[j].width = Inches(w * lw / tot)
    for i, ln in enumerate(linhas):
        t.rows[i].height = Inches(h_linha)
        for j, v in enumerate(ln):
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(DG if i == 0 else (LG if i in destaque else WH))
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_left = c.margin_right = Inches(0.2)
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = v
            r.font.size = Pt(size)
            r.font.bold = (i == 0) or (i in destaque)
            r.font.color.rgb = rgb(WH if i == 0 else TX)
    tblPr = gf._element.find('.//' + qn('a:tblPr'))
    for filho in list(tblPr):
        if filho.tag == qn('a:tableStyleId'):
            tblPr.remove(filho)
    return t


def grafico(s, x, y, w, h, cats, series, cores, vmin, vmax, legenda=True, por_ponto=None, fsize=22, gap=60):
    cd = CategoryChartData()
    cd.categories = cats
    for nome, vals in series:
        cd.add_series(nome, vals)
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    ch.font.size = Pt(fsize)
    ch.font.color.rgb = rgb(TX)
    ch.has_legend = legenda
    if legenda:
        ch.legend.position = XL_LEGEND_POSITION.TOP
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(fsize)
    va, ca = ch.value_axis, ch.category_axis
    va.minimum_scale, va.maximum_scale = vmin, vmax
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb(GR)
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(fsize)
    va.tick_labels.number_format = '0'
    va.tick_labels.number_format_is_linked = False
    ca.tick_labels.font.size = Pt(fsize)
    ca.format.line.color.rgb = rgb(GR)
    pl = ch.plots[0]
    pl.gap_width = gap
    pl.overlap = -10
    pl.has_data_labels = True
    dl = pl.data_labels
    dl.font.size = Pt(fsize)
    dl.font.bold = True
    dl.number_format = '0.0'
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    for ser, cor in zip(pl.series, cores):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = rgb(cor)
    if por_ponto:
        for i, cor in enumerate(por_ponto):
            pt = pl.series[0].points[i]
            pt.format.fill.solid()
            pt.format.fill.fore_color.rgb = rgb(cor)
    return ch


def foto(s, caminho, x, y, w=None, h=None):
    return s.shapes.add_picture(caminho, Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


X0, CW = 1.64, 22.4  # margem esquerda e largura útil (até as barras verdes da direita)
D = {d['id']: d for d in SLIDES}
N = len(SLIDES) + 1
feitos = []


def abre(id_):
    d = D[id_]
    i = [x['id'] for x in SLIDES].index(id_) + 1
    s = novo_slide(d['secao'], f"{i} / {N}  ·  {d['pagina']} do artigo")
    titulo(s, d['titulo'])
    s.notes_slide.notes_text_frame.text = d['fala'] + ('\n\n[Transição] ' + d['transicao'] if d['transicao'] else '')
    feitos.append(s)
    return s


# ------------------------------------------------------------------ capa e encerramento
TIT = 'Detecção de anomalias fitossanitárias em hortaliças utilizando visão computacional e aprendizado de máquina'
for sl in (capa, fim):
    for sh in sl.shapes:
        if not sh.has_text_frame:
            continue
        t = sh.text_frame.text
        if t.startswith('Título'):
            troca(sh, TIT, size=46)
            sh.width = Inches(17.6)
            sh.text_frame.word_wrap = True
        elif t.startswith('Aluno'):
            troca(sh, 'José Pedro Vieira Silva\nOrientador: Prof. Leanderson André', size=44)
            sh.top = Inches(6.6)
        elif t.startswith('Defesa'):
            troca(sh, 'Defesa do Trabalho de Conclusão de Curso\nBacharelado em Engenharia de Software', size=40)
        elif t.startswith('Julho'):
            troca(sh, 'Dezembro / 2026')
capa.notes_slide.notes_text_frame.text = D['capa']['fala'] + '\n\n[Transição] ' + D['capa']['transicao']
fim.notes_slide.notes_text_frame.text = ('Agradeça e fique neste slide durante as perguntas. Slides de apoio: não há; '
                                         'para números, volte ao slide do resultado correspondente.')

# ------------------------------------------------------------------ 2 problema
s = abre('problema')
cw = 6.9
for k, (num, rot, cor, fill) in enumerate([
        ('> 99%', 'de acurácia em fotos de estúdio:\nfolha isolada, fundo liso', DG, WH),
        ('70–85%', 'em fotos de lavoura:\nsolo, sombra, várias plantas', OR, WH),
        ('77%', 'dos estabelecimentos rurais do Brasil\nsão de agricultura familiar', TX, WH)]):
    numero(s, X0 + k * (cw + 0.85), 4.5, cw, 5.0, num, rot, cor=cor, fill=fill, nsize=110, rsize=30)
faixa(s, 10.3, 'O risco não é só errar: é errar com aparência de certeza.', fill=LOR, cor=OR, h=1.3, size=36)
tb(s, X0, 12.1, 16, 0.6, 'Fontes: Mohanty, Hughes e Salathé (2016); Shafay et al. (2025); IBGE (2019).', size=22, color=MU)

# ------------------------------------------------------------------ 3 questão
s = abre('questao')
tb(s, X0, 4.2, CW, 0.7, 'Questão de pesquisa', size=30, bold=True, color=DG)
caixa(s, X0, 5.0, CW, 3.4, DG)
tb(s, X0 + 0.7, 5.0, CW - 1.4, 3.4,
   'Em que medida um sistema de triagem fitossanitária para hortaliças folhosas, baseado em transfer learning, '
   'treinado com imagens públicas auditadas contra vazamento de dados e dotado de calibração e abstenção, '
   'mantém desempenho confiável em fotografias de campo?',
   size=34, bold=True, color=WH, anchor=MSO_ANCHOR.MIDDLE, ls=1.1)
tb(s, X0, 8.85, CW, 0.7, 'Objetivo', size=30, bold=True, color=DG)
caixa(s, X0, 9.65, CW, 1.6, LG)
tb(s, X0 + 0.7, 9.65, CW - 1.4, 1.6,
   'Avaliar em que medida esse sistema mantém desempenho confiável em fotografias de campo.',
   size=34, bold=True, color=DG, anchor=MSO_ANCHOR.MIDDLE)
tb(s, X0, 11.6, CW, 1.2,
   'Avaliado em três situações, cada vez mais longe do treino: teste das mesmas fontes · fonte nunca vista · '
   'fotos de horta. Em todas: a abstenção intercepta os erros?',
   size=26, color=MU)

# ------------------------------------------------------------------ 4 sistema
s = abre('sistema')
for k, (nome, desc, fill, cor) in enumerate([
        ('Saudável', 'nenhum sinal de anomalia', G, DG),
        ('Anômalo', 'doença, praga ou deficiência', OR, WH),
        ('Inconclusivo', 'confiança calibrada abaixo de 0,85', '8C8C8C', WH),
        ('Não é folha', 'menos de 10% de pixels de vegetação', '404040', WH)]):
    y = 4.5 + k * 1.95
    caixa(s, X0, y, 4.6, 1.55, fill, raio=0.3)
    tb(s, X0, y, 4.6, 1.55, nome, size=34, bold=True, color=cor, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    tb(s, X0 + 5.0, y, 5.2, 1.55, desc, size=28, anchor=MSO_ANCHOR.MIDDLE)
img = os.path.join(RAIZ, 'docs', 'seminarios', 'Interface FotoVision.png')
caixa(s, 12.4, 4.5, 11.64, 6.02, WH, raio=0.02)
foto(s, img, 12.6, 4.7, w=11.24)
tb(s, 12.4, 10.75, 11.64, 1.6, ['Triagem: diz se há algo errado com a folha,', 'não qual é a doença.'],
   size=32, bold=True, color=DG)

# ------------------------------------------------------------------ 5 método
s = abre('metodo')
cw, gap = 4.75, 1.13
passos = [('6 fontes públicas', '175.123 imagens\nsaudável × anômalo'),
          ('Deduplicação', 'hash perceptual\nvazamento residual de 0,04%'),
          ('4 arquiteturas', 'EfficientNet-B0, MobileNetV2,\nResNet-50 e ViT-B/16'),
          ('Confiança', 'temperature scaling\nabstenção abaixo de 0,85')]
for k, (cab, txt) in enumerate(passos):
    x = X0 + k * (cw + gap)
    caixa(s, x, 4.4, cw, 3.5)
    tb(s, x + 0.35, 4.7, cw - 0.7, 0.9, cab, size=34, bold=True, color=DG)
    tb(s, x + 0.35, 5.75, cw - 0.7, 2.0, txt.split('\n'), size=25)
    if k < 3:
        seta(s, x + cw + 0.14, 5.8, w=0.85, h=0.7)
tb(s, X0, 8.5, CW, 0.8, 'Avaliação em três níveis', size=36, bold=True, color=DG)
cw = 6.9
for k, (n, cab, txt) in enumerate([('1', 'Teste intra-fonte', '23.967 imagens das mesmas fontes do treino'),
                                   ('2', 'Fonte nunca vista', '484 imagens de espinafre-de-malabar'),
                                   ('3', 'Campo', '13 + 10 fotos, rótulo anotado antes da inferência')]):
    x = X0 + k * (cw + 0.85)
    caixa(s, x, 9.5, cw, 2.5, LG)
    bola(s, x + 0.35, 9.85, 1.0, n, size=30)
    tb(s, x + 1.6, 9.85, cw - 1.9, 1.0, cab, size=32, bold=True, color=DG, anchor=MSO_ANCHOR.MIDDLE)
    tb(s, x + 0.35, 10.95, cw - 0.7, 1.0, txt, size=25)
tb(s, X0, 12.35, CW, 0.7,
   'Acurácia balanceada  ·  IC 95% por bootstrap  ·  teste de McNemar pareado  ·  intervalo de Wilson no campo',
   size=26, color=MU)

# ------------------------------------------------------------------ 6 vazamento
s = abre('vazamento')
cw = 6.3
blocos = [('99,01%', 'acurácia do TCC-I:\nboa demais para ser verdade', MU, WH),
          ('57,35%', 'do teste eram fotos quase\nidênticas às do treino', OR, WH),
          ('0,04%', 'de vazamento residual depois\nde agrupar por hash perceptual', DG, WH)]
for k, (num, rot, cor, fill) in enumerate(blocos):
    x = X0 + k * (cw + 1.75)
    numero(s, x, 4.5, cw, 4.6, num, rot, cor=cor, fill=fill, nsize=100, rsize=28)
    if k < 2:
        seta(s, x + cw + 0.42, 6.4, w=0.95, h=0.8)
# risca o 99,01%
for sh in s.shapes:
    if sh.has_text_frame and sh.text_frame.text == '99,01%':
        sh.text_frame.paragraphs[0].runs[0]._r.get_or_add_rPr().set('strike', 'sngStrike')
tb(s, X0, 9.65, CW, 0.8, 'Com o teste limpo, apareceram dois defeitos que a métrica escondia', size=34, bold=True, color=DG)
for k, (cab, txt) in enumerate([('Abstenção desligada', 'o limiar nunca disparava'),
                                ('3.972 rótulos trocados', 'alfaces sadias marcadas como doentes, pelo nome de uma pasta')]):
    x = X0 + k * (CW / 2 + 0.2)
    caixa(s, x, 10.65, CW / 2 - 0.2, 2.2, LOR)
    tb(s, x + 0.45, 10.85, CW / 2 - 1.1, 0.8, cab, size=32, bold=True, color=OR)
    tb(s, x + 0.45, 11.75, CW / 2 - 1.1, 0.9, txt, size=26)

# ------------------------------------------------------------------ 7 Q1
s = abre('q1')
tabela(s, X0, 4.5, 13.9, [
    ['Modelo', 'Parâmetros', 'Acur. bal.', 'Falsos neg.', 'Latência'],
    ['EfficientNet-B0', '4,0 M', '98,48%', '209', '7,2 ms'],
    ['ViT-B/16', '85,8 M', '98,48%', '137', '19,4 ms'],
    ['MobileNetV2', '2,2 M', '97,89%', '294', '5,3 ms'],
    ['ResNet-50', '23,5 M', '97,82%', '352', '8,6 ms']],
    [3.4, 2.4, 2.4, 2.4, 2.2], destaque=(1, 2), size=28, h_linha=1.0)
tb(s, X0, 9.8, 13.9, 0.6, 'Teste com 23.967 imagens, quarto ciclo. IC 95% do 98,48%: 98,32 a 98,64.', size=22, color=MU)
numero(s, 16.3, 4.5, 7.74, 3.55, 'p = 0,26', 'EfficientNet-B0 × ViT-B/16:\nempate (teste de McNemar)', nsize=72, rsize=26)
numero(s, 16.3, 8.4, 7.74, 3.55, '21×', 'mais parâmetros no ViT,\npara o mesmo resultado', cor=OR, nsize=88, rsize=26)
faixa(s, 12.25, 'Neste estudo, o EfficientNet-B0 foi o mais adequado: mesmo desempenho, muito menos custo.', h=1.1, size=30)
# sobe a nota para não colidir
for sh in s.shapes:
    if sh.has_text_frame and sh.text_frame.text.startswith('Teste com 23.967'):
        sh.top = Inches(9.75)

# ------------------------------------------------------------------ 8 Q2
s = abre('q2')
numero(s, X0, 4.5, 7.6, 4.9, '0,003', 'ECE na validação: quando o modelo\ndiz 90%, acerta perto de 90%', nsize=110, rsize=27)
caixa(s, 9.9, 4.5, 14.14, 4.9)
tb(s, 10.4, 4.8, 13.2, 0.8, 'Erros interceptados pela abstenção (τ = 0,85)', size=32, bold=True, color=DG)
for k, (nome, v, det) in enumerate([('EfficientNet-B0', 31.1, '113 de 363 erros'), ('ViT-B/16', 15.0, '51 de 341 erros')]):
    y = 6.0 + k * 1.65
    tb(s, 10.4, y, 4.0, 1.0, nome, size=28, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    caixa(s, 14.5, y + 0.15, 6.2, 0.7, GR, raio=0.5)
    caixa(s, 14.5, y + 0.15, 6.2 * v / 100, 0.7, OR, raio=0.5)
    tb(s, 21.0, y, 2.8, 1.0, f'{v:.0f}%', size=44, bold=True, color=OR, anchor=MSO_ANCHOR.MIDDLE)
    tb(s, 14.5, y + 0.9, 6.2, 0.5, det, size=20, color=MU)
numero(s, X0, 9.8, 10.95, 2.9, '0,95 a 0,999', 'confiança dos erros nas 13 fotos de campo', cor=OR, fill=LOR, nsize=64, rsize=26)
numero(s, 13.09, 9.8, 10.95, 2.9, '0,003 → 0,146', 'ECE: da validação para o campo' + chr(10) + '(13 fotos: valor indicativo)', cor=OR, fill=LOR, nsize=64, rsize=26)

# ------------------------------------------------------------------ 9 central
s = abre('central')
caixa(s, X0, 4.4, 14.4, 8.5)
grafico(s, X0 + 0.3, 4.6, 13.8, 8.1,
        ['Estúdio\n(dentro do teste)', 'Lavoura\n(dentro do teste)', 'Fonte nunca vista\n(espinafre)'],
        [('Acurácia balanceada (%)', (99.7, 86.3, 75.4))], [DG], 50, 100, legenda=False,
        por_ponto=[DG, G, OR], fsize=24, gap=70)
numero(s, 16.5, 4.4, 7.54, 2.65, '98,48%', 'é a média do teste', cor=MU, nsize=64, rsize=26)
numero(s, 16.5, 7.33, 7.54, 2.65, '81%', 'dos erros estão na lavoura', cor=DG, nsize=64, rsize=26)
numero(s, 16.5, 10.25, 7.54, 2.65, '−23 pts', 'em uma fonte nunca vista', cor=OR, fill=LOR, nsize=64, rsize=26)
tb(s, X0, 13.0, 15, 0.5, 'EfficientNet-B0. Estúdio e lavoura: ciclo 4. Fonte nunca vista: ciclo 3, N = 484.', size=20, color=MU)

# ------------------------------------------------------------------ 10 ciclo 5
s = abre('ciclo5')
mods = ['EfficientNet-B0', 'MobileNetV2', 'ResNet-50', 'ViT-B/16']
for k, (cab, c4, c5, sub) in enumerate([
        ('Especificidade na lavoura (%)', (76.6, 78.3, 81.4, 69.8), (83.4, 90.2, 89.0, 72.1), 'menos falsos alarmes'),
        ('Sensibilidade na lavoura (%)', (96.0, 94.9, 94.2, 97.0), (94.2, 90.0, 90.2, 96.9), 'mais doenças perdidas')]):
    x = X0 + k * (CW / 2 + 0.2)
    w = CW / 2 - 0.2
    caixa(s, x, 4.4, w, 7.0)
    tb(s, x + 0.4, 4.6, w - 0.8, 0.7, [[(cab, {}), ('   ' + sub, {'bold': False, 'color': MU, 'size': 24})]],
       size=28, bold=True, color=DG)
    grafico(s, x + 0.2, 5.3, w - 0.4, 6.0, mods, [('Ciclo 4', c4), ('Ciclo 5', c5)], ['A6A6A6', DG], 60, 100,
            fsize=22, gap=40)
faixa(s, 11.75, 'Não é um modelo melhor em tudo: é outro ponto de operação.', h=1.2, size=34)

# ------------------------------------------------------------------ 11 campo
s = abre('campo')
w1 = 9.2
caixa(s, X0, 4.4, w1, 6.5)
tb(s, X0 + 0.45, 4.7, w1 - 0.9, 0.8, '13 fotos históricas (TCC-I)', size=30, bold=True, color=DG)
tb(s, X0 + 0.45, 5.6, w1 - 0.9, 1.8, '84,6%', size=110, bold=True, color=TX)
tb(s, X0 + 0.45, 7.75, w1 - 0.9, 3.0,
   ['11 de 13 acertos (ciclo 4)', 'IC 95%: 57,8% a 95,7%', 'As versões são indistinguíveis.'], size=28, espaco=8)
x2, w2 = X0 + w1 + 0.4, CW - w1 - 0.4
caixa(s, x2, 4.4, w2, 6.5)
tb(s, x2 + 0.45, 4.7, w2 - 0.9, 0.8, '10 fotos em horta, com fundo real (27/10/2026)', size=30, bold=True, color=DG)
pasta = os.path.join(RAIZ, 'backend', 'campo_novo', 'alface', 'healthy')
fotos = sorted(f for f in os.listdir(pasta) if f.startswith('IMG_20261027'))
from PIL import Image, ImageOps
tmp = os.path.join(os.path.dirname(SAIDA), '_tmp_fotos')
os.makedirs(tmp, exist_ok=True)
for k, f in enumerate([fotos[0], fotos[4]]):
    im = ImageOps.exif_transpose(Image.open(os.path.join(pasta, f))).convert('RGB')
    lado = min(im.size)
    im = ImageOps.fit(im, (900, 760), centering=(0.5, 0.5))
    dest = os.path.join(tmp, f'horta{k}.jpg')
    im.save(dest, quality=85)
    foto(s, dest, x2 + 0.45 + k * 3.55, 5.65, w=3.35)
tb(s, x2 + 7.6, 5.55, w2 - 8.0, 1.6, '10 de 10', size=80, bold=True, color=DG)
tb(s, x2 + 7.6, 7.2, w2 - 8.0, 1.8, ['sadias, nas 4 versões', 'confiança > 0,98'], size=26, espaco=6)
tb(s, x2 + 0.45, 8.75, w2 - 0.9, 2.0,
   ['Solo, mato e vários pés no quadro', 'não provocaram falso alarme.'], size=28)
faixa(s, 11.3, 'A horta estava toda sadia: sensibilidade em campo e domain gap ainda sem medida.',
      fill=LOR, cor=OR, h=1.3, size=32)

# ------------------------------------------------------------------ 12 literatura
s = abre('literatura')
linhas = [('Videira', 'Ergün et al. (2026)', '—', '73,8%', 'conjunto externo, EfficientNet-B0'),
          ('Amêndoa', 'Workneh et al. (2026)', 'F1 0,98', 'F1 0,56', 'fontes disjuntas'),
          ('Várias culturas', 'Xiang; Shi; Zhu (2026)', '99,7%', '32,1%', 'PlantVillage → fotos de campo'),
          ('Hortaliças folhosas', 'Este trabalho', '98,48%', '75,4%', 'fonte nunca vista')]
for k, (cult, ref, a, b, obs) in enumerate(linhas):
    y = 4.4 + k * 1.85
    meu = k == 3
    caixa(s, X0, y, CW, 1.6, DG if meu else WH)
    c1, c2 = (WH, WH) if meu else (DG, MU)
    tb(s, X0 + 0.5, y + 0.15, 6.6, 0.75, cult, size=32, bold=True, color=c1)
    tb(s, X0 + 0.5, y + 0.9, 6.6, 0.6, ref, size=22, color=c2)
    tb(s, X0 + 7.4, y, 3.6, 1.6, a, size=44, bold=True, color=WH if meu else TX, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    seta(s, X0 + 11.2, y + 0.5, w=0.9, h=0.6, fill=G)
    tb(s, X0 + 12.3, y, 3.6, 1.6, b, size=44, bold=True, color=WH if meu else OR, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    tb(s, X0 + 16.3, y, 5.8, 1.6, obs, size=24, color=WH if meu else TX, anchor=MSO_ANCHOR.MIDDLE)
tb(s, X0, 11.95, CW, 1.2,
   'Lacuna: nenhum estudo localizado reúne, em hortaliças folhosas, auditoria de vazamento, calibração, abstenção e avaliação fora da fonte.',
   size=28, bold=True, color=DG)

# ------------------------------------------------------------------ 13 limitações
s = abre('limitacoes')
itens = [('Campo pequeno', '23 fotos, só de alface, sem folhas doentes no lote com fundo real', ''),
         ('Mesmas fontes públicas', 'treino e teste da mesma origem; parte das culturas é só aproximação visual', 'Xu et al. (2024)'),
         ('Calibração fora do campo', 'na passagem para a lavoura, a confiança segue acima da acurácia', 'Xiang; Shi; Zhu (2026)'),
         ('Tarefa binária', 'não nomeia a doença; sintomas parecidos têm causas diferentes', 'Barbedo (2016)'),
         ('Uma semente por modelo', 'o desempenho varia entre treinos; o empate pode não se repetir', 'Coakley; Gundersen (2026)')]
for k, (cab, txt, ref) in enumerate(itens):
    y = 4.35 + k * 1.72
    caixa(s, X0, y, CW, 1.5)
    bola(s, X0 + 0.3, y + 0.25, 1.0, str(k + 1), size=30)
    tb(s, X0 + 1.7, y, 6.3, 1.5, cab, size=30, bold=True, color=DG, anchor=MSO_ANCHOR.MIDDLE)
    tb(s, X0 + 8.1, y, 9.9, 1.5, txt, size=26, anchor=MSO_ANCHOR.MIDDLE)
    tb(s, X0 + 17.9, y, 4.3, 1.5, ref, size=20, color=MU, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)

# ------------------------------------------------------------------ 14 conclusão
s = abre('conclusao')
caixa(s, X0, 4.4, 9.6, 8.5, DG)
tb(s, X0 + 0.6, 4.8, 8.4, 4.4, 'O desempenho dependeu mais do protocolo de avaliação do que da arquitetura.',
   size=42, bold=True, color=WH, ls=1.05)
tb(s, X0 + 0.6, 9.6, 8.4, 3.0,
   [[('98,48%', {'bold': True, 'color': WH}), ('  nas fontes de treino', {'color': 'D5EBC0'})],
    [('75,4%', {'bold': True, 'color': WH}), ('  em fonte nunca vista', {'color': 'D5EBC0'})],
    [('84,6%', {'bold': True, 'color': WH}), ('  em 13 fotos de campo', {'color': 'D5EBC0'})]], size=30, espaco=8)
xr, wr = X0 + 10.0, CW - 10.0
for k, (q, txt) in enumerate([('1', 'Teste: EfficientNet-B0 e ViT-B/16 empataram; o EfficientNet-B0 foi o mais adequado.'),
                              ('2', 'Confiança: bem calibrada no treino; a abstenção deixou passar mais de 2/3 dos erros.'),
                              ('3', 'Campo: nenhum falso alarme em horta sadia; sensibilidade ainda sem medida.')]):
    y = 4.4 + k * 1.95
    caixa(s, xr, y, wr, 1.7)
    bola(s, xr + 0.3, y + 0.3, 1.1, q, size=28)
    tb(s, xr + 1.75, y, wr - 2.1, 1.7, txt, size=26, anchor=MSO_ANCHOR.MIDDLE)
caixa(s, xr, 10.3, wr, 2.6, LG)
tb(s, xr + 0.45, 10.5, wr - 0.9, 0.7, 'Contribuição: um protocolo reprodutível', size=30, bold=True, color=DG)
tb(s, xr + 0.45, 11.3, wr - 0.9, 1.5,
   'Auditoria de vazamento · três situações de avaliação (intra-fonte, fonte retida, campo) · testes pareados',
   size=25)

# ------------------------------------------------------------------ 15 futuros
s = abre('futuros')
cw, gp = 4.0, 0.6
for k, (cab, txt) in enumerate([('Completar a coleta', 'folhas doentes e outras culturas'),
                                ('Recalibrar', 'com fotografias de campo'),
                                ('Modelos de fundação', 'extratores congelados'),
                                ('Várias sementes', 'repetir o treino e medir a variação'),
                                ('Custo do erro', 'definir com os produtores')]):
    x = X0 + k * (cw + gp)
    caixa(s, x, 4.6, cw, 5.6)
    bola(s, x + 0.4, 5.0, 1.2, str(k + 1), size=34)
    tb(s, x + 0.4, 6.6, cw - 0.8, 1.7, cab, size=32, bold=True, color=DG)
    tb(s, x + 0.4, 8.3, cw - 0.8, 1.7, txt, size=26)
faixa(s, 10.9, 'Não é um modelo pronto para o campo: é a medida honesta de quanto falta.', h=1.3, size=34)

# ------------------------------------------------------------------ ordem final e limpeza
lst = prs.slides._sldIdLst
ids = list(lst)
por_slide = {prs.slides[i].slide_id: ids[i] for i in range(len(ids))}
ordem = [capa] + feitos + [fim]
els_ordem = [por_slide[sl.slide_id] for sl in ordem]
els_fora = [por_slide[sl.slide_id] for sl in (base, _metod)]
for el in ids:
    lst.remove(el)
for el in els_ordem:
    lst.append(el)
# tira do pacote os dois slides de exemplo do modelo (Introdução e Metodologia)
for el in els_fora:
    prs.part.drop_rel(el.rId)
prs.save(SAIDA)
print('ok', len(ordem), 'slides')
