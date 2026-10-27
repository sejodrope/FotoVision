# Monta a v10 do artigo a partir do document.xml da v9: entra o primeiro lote de campo com
# fundo real (27/10/2026, horta_27out: 10 fotografias de alface sadia, vista de cima, vários pés
# por quadro), avaliado às cegas pelas quatro versões com backend/roda_campo.py, e o lote de casa
# de 18/10 (reportado à parte, não é campo). Resumo, Seção 3.2, Seção 4 e conclusão.
# Uso: python build_v10.py <dir v9 descompactado> <dir de saida descompactado>
import re, sys, shutil, os
from xml.sax.saxutils import escape

src, dst = sys.argv[1], sys.argv[2]
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
path = os.path.join(dst, 'word', 'document.xml')
d = open(path, encoding='utf8').read()


def texto(p):
    return ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p))


def marcado(p):
    """Texto do parágrafo com *itálico* marcado (só itálico ocorre nestes parágrafos)."""
    out = []
    for r in re.findall(r'<w:r[ >].*?</w:r>', p, re.S):
        t = ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', r))
        if not t:
            continue
        t = t.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        out.append(f'*{t}*' if '<w:i/>' in r else t)
    return ''.join(out).replace('**', '')


def runs(text):
    out = []
    for part in re.split(r'(\*.+?\*)', text):
        if not part:
            continue
        rpr = ''
        if part.startswith('*'):
            part, rpr = part[1:-1], '<w:rPr><w:i/></w:rPr>'
        out.append(f'<w:r>{rpr}<w:t xml:space="preserve">{escape(part)}</w:t></w:r>')
    return ''.join(out)


def acha(inicio):
    ps = [p for p in re.findall(r'<w:p[ >].*?</w:p>', d, re.S) if texto(p).startswith(inicio)]
    assert len(ps) == 1, (inicio, len(ps))
    return ps[0]


def troca(inicio, pares):
    global d
    p = acha(inicio)
    m = marcado(p)
    for old, new in pares:
        assert m.count(old) == 1, (inicio, old)
        m = m.replace(old, new)
    ppr = re.match(r'<w:p[^>]*>(<w:pPr>.*?</w:pPr>|<w:pPr/>)?', p, re.S).group(1) or ''
    d = d.replace(p, f'<w:p>{ppr}{runs(m)}</w:p>', 1)


def substitui(inicio, novos):
    """Troca um parágrafo por vários, com a formatação de parágrafo dele; [] apaga."""
    global d
    p = acha(inicio)
    ppr = re.match(r'<w:p[^>]*>(<w:pPr>.*?</w:pPr>|<w:pPr/>)?', p, re.S).group(1) or ''
    d = d.replace(p, ''.join(f'<w:p>{ppr}{runs(t)}</w:p>' for t in novos), 1)


def insere_depois(inicio, novos):
    """Acrescenta parágrafos logo depois de um, com a formatação de parágrafo dele."""
    global d
    p = acha(inicio)
    ppr = re.match(r'<w:p[^>]*>(<w:pPr>.*?</w:pPr>|<w:pPr/>)?', p, re.S).group(1) or ''
    d = d.replace(p, p + ''.join(f'<w:p>{ppr}{runs(t)}</w:p>' for t in novos), 1)


def troca_xml(old, new):
    """Troca texto direto no XML (para parágrafos com negrito, como o resumo)."""
    global d
    assert d.count(old) == 1, old
    d = d.replace(old, new)


# ---------------------------------------------------------------- resumo e abstract
troca_xml('On 13 field photographs, versions are indistinguishable.',
          'On 13 field photographs, versions are indistinguishable, and on the first 10 photographs '
          'with real background, all of healthy lettuce, no version raised a false alarm.')
troca_xml('Em 13 fotografias de campo, as versões são indistinguíveis.',
          'Em 13 fotografias de campo, as versões são indistinguíveis e, nas 10 primeiras com fundo '
          'real, todas de alface sadia, nenhuma emitiu falso alarme.')

# ---------------------------------------------------------------- 3.2 Coleta
troca('Os dados de campo serão coletados', [(
    'Os dados de campo serão coletados', 'Os dados de campo são coletados')])
troca('Quatro regras garantem a validade do conjunto', [(
    'formam um conjunto histórico, reportado à parte.',
    'formam um conjunto histórico, reportado à parte. A coleta começou em 27/10/2026, com 10 '
    'fotografias de canteiros de alface sadia, tiradas de cima, com vários pés por quadro; um lote '
    'anterior, de um pé colhido e fotografado em casa sobre toalha branca (3 folhas sadias e 7 '
    'anômalas), também é reportado à parte, por não ser de campo.')])

# ---------------------------------------------------------------- 4. Análise
substitui('As Tabelas 1 e 2 vão até o quarto ciclo', [
    'As Tabelas 1 e 2 vão até o quarto ciclo de treinamento, a Tabela 3 apresenta o quinto, e o '
    'primeiro lote de campo fecha a seção.'])
insere_depois('Os resultados são coerentes com o Quadro 1', [
    'No lote de campo, as quatro versões (EfficientNet-B0 e ViT-B/16, ciclos 4 e 5) deram como '
    'sadias as 10 fotografias da horta, sem abstenção, com confiança acima de 0,98 (Wilson: 72,2% '
    'a 100%), também em recortes centrais com metade e um quarto do lado. Solo, mato e vários pés '
    'no quadro não provocaram falso alarme, mas, sem folhas doentes, a sensibilidade em campo e o '
    '*domain gap* seguem sem medida. No lote de casa, todas as diagnosticadas saíram anômalas, '
    'inclusive as três sadias, com confiança acima de 0,97: com cada lote quase todo em uma '
    'classe, folha e cena não se separam.'])

# ---------------------------------------------------------------- cortes na Seção 4 para caber
troca('A Tabela 2 mostra por que', [
    ('a análise estatística é central', 'a estatística é central'), (
    'Os nove erros dos quatro primeiros ciclos, em quatro fotografias, tiveram confiança calibrada '
    'entre 0,949 e 0,999: o limiar τ é ajustado na validação, da mesma distribuição do treino, e '
    'fora dela o ECE passou de 0,003 para 0,146.',
    'Os nove erros dos quatro primeiros ciclos tiveram confiança calibrada entre 0,949 e 0,999, '
    'e o ECE passou de 0,003 na validação para 0,146 em campo.')])
troca('EfficientNet-B0 e ViT-B/16 empatam,', [
    ('e estes não diferem entre si (p = 0,26), embora o segundo tenha dez vezes mais parâmetros.',
     'e estes não diferem (p = 0,26).'),
    ('empatam, com o mesmo IC 95% (98,32% a 98,64%) e sem diferença pelo McNemar exato com '
     'correção de Holm (p = 0,26).',
     'empatam, com IC 95% de 98,32% a 98,64% e sem diferença significativa (p de Holm = 0,26).'),
    ('O ViT-B/16 comete menos', 'O ViT-B/16 tem menos'),
    ('com 21 vezes mais parâmetros e latência 2,7 vezes maior.',
     'com 21 vezes mais parâmetros e 2,7 vezes a latência.')])
troca('Os resultados são coerentes com o Quadro 1', [(
    'Na calibração, o estudo de videira reduz o ECE do conjunto externo de 0,287 para 0,038, com '
    'confiança média de 0,55 (Ergün et al., 2026), enquanto aqui o ECE de campo é de 0,146, o que '
    'recomenda recalibrar em campo.',
    'Em videira, o ECE externo chega a 0,038, com confiança média de 0,55 (Ergün et al., 2026); '
    'aqui, o de campo é de 0,146.'),
    ('Os resultados são coerentes com o Quadro 1. A queda de 98,48% para 75,4%',
     'Quanto ao Quadro 1, a queda para 75,4%'),
    ('a diferença está no protocolo de avaliação, e não na cultura',
     'a diferença está no protocolo, e não na cultura'),
    ('e concentram 81% dos seus erros', 'e concentram 81% dos erros')])

# ---------------------------------------------------------------- 5. Conclusão
troca('Este trabalho avaliou em que medida', [(
    ' e 84,6% de acertos em 13 fotografias de campo (IC 95%: 57,8% a 95,7%).',
    ', 84,6% de acertos em 13 fotografias de campo (IC 95%: 57,8% a 95,7%) e nenhum falso alarme '
    'em 10 com fundo real.'),
    ('a resposta até aqui é que o desempenho medido depende', 'a resposta é que o desempenho depende'),
    ('em uma fonte nunca vista', 'numa fonte nunca vista')])
troca('Q3 e os objetivos (d) a (f) dependem', [(
    'Q3 e os objetivos (d) a (f) dependem da coleta de campo com fundo real, ainda em andamento, '
    'mas o sentido da resposta já aparece:',
    'A coleta com fundo real responde a Q3 só em parte: nenhuma versão deu falso '
    'alarme em 10 fotografias de alface sadia, mas, sem folhas doentes, a sensibilidade e '
    'o *domain gap* em campo seguem sem medida, e os objetivos (d) a (f), abertos. Vê-se que'),
    ('e se concentra nas culturas de lavoura, e corrigir atalhos nos dados (quinto ciclo) reduz os '
     'falsos alarmes', 'e se concentra na lavoura, e corrigir atalhos nos dados (quinto ciclo) '
     'reduz falsos alarmes')])
substitui('A principal contribuição é um protocolo', [])
troca('As conclusões têm limites', [
    ('As conclusões têm limites:',
     'A principal contribuição é um protocolo reprodutível (teste intra-fonte, fonte retida e '
     'campo com rótulo prévio à inferência, e versões comparadas por testes pareados), que expôs '
     'o vazamento dos 99,01% do TCC-I e os limites da calibração fora do treino. Há limites:'),
    ('que decide entre os perfis dos dois ciclos', 'que decide entre os dois ciclos'),
    ('e a evidência de campo se restringe a 13 fotografias de alface.',
     'e a evidência de campo se restringe à alface: 13 fotografias históricas e 10 com fundo real, '
     'todas de plantas sadias.'),
    ('além da coleta, propõem-se', 'além de completar a coleta com folhas doentes e outras culturas, '
     'propõem-se')])

# ---------------------------------------------------------------- últimos cortes (págs. 8 e 9)
troca('A Tabela 2 mostra por que', [
    ('A Tabela 2 mostra por que a estatística é central: os intervalos', 'Na Tabela 2, os intervalos'),
    ('A amostra revelou um defeito real, 3.972 imagens de alface saudável rotuladas como doentes, '
     'mas não decide entre modelos.',
     'A amostra revelou um defeito de rotulagem, mas não decide entre modelos.')])
troca('Na Tabela 3, a correção agiu', [
    ('ou seja, emitem menos falsos alarmes em folhas sadias', 'isto é, menos falsos alarmes'),
    ('pois três em cada quatro dessas imagens são anômalas, e o perfil preferível depende do custo '
     'de cada erro na horta.', 'pois três em cada quatro dessas imagens são anômalas.'),
    ('No lote retido, os modelos ficam entre 85,8% e 90,1%, cerca de dez pontos abaixo',
     'No lote retido, os quatro ficam entre 85,8% e 90,1%, dez pontos abaixo')])

open(path, 'w', encoding='utf8').write(d)
print('ok')
