# Monta a v8 do artigo a partir do document.xml da v7: corrige a leitura da literatura
# depois da leitura do texto completo (docs/REVISAO_LITERATURA_WOS.md, Parte 4, 16/10/2026).
# - Shafay et al. (2025): o ViT puro é apontado como mais sensível ao ambiente (converge com
#   o ViT-B/16 daqui); a robustez é das arquiteturas híbridas (Swin).
# - Nguyen et al. (2026): a perda no PlantDoc vem sobretudo de atribuir a folha à cultura errada.
# - Ergün et al. (2026): o ECE externo de 0,038 vem com confiança média de 0,55; e, ao contrário
#   do teste externo deles, as fotografias ambíguas da coleta são marcadas, não excluídas.
# Uso: python build_v8.py <dir v7 descompactado> <dir de saida descompactado>
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


def troca(inicio, pares):
    global d
    ps = [p for p in re.findall(r'<w:p[ >].*?</w:p>', d, re.S) if texto(p).startswith(inicio)]
    assert len(ps) == 1, (inicio, len(ps))
    p = ps[0]
    m = marcado(p)
    for old, new in pares:
        assert m.count(old) == 1, (inicio, old)
        m = m.replace(old, new)
    ppr = re.match(r'<w:p[^>]*>(<w:pPr>.*?</w:pPr>|<w:pPr/>)?', p, re.S).group(1) or ''
    d = d.replace(p, f'<w:p>{ppr}{runs(m)}</w:p>', 1)


troca('A busca por trabalhos relacionados', [(
    'caem para 26% a 30% no conjunto de campo PlantDoc, e o ajuste fino completo de um modelo '
    'de fundação piora a transferência (Nguyen et al., 2026), embora arquiteturas com '
    '*transformers* sejam apontadas como mais robustas (Shafay et al., 2025).',
    'caem para 26% a 30% no conjunto de campo PlantDoc, sobretudo por atribuírem a folha à '
    'cultura errada, e o ajuste fino completo de um modelo de fundação piora a transferência '
    '(Nguyen et al., 2026); uma revisão aponta arquiteturas híbridas como mais robustas e o ViT '
    'puro como mais sensível ao ambiente (Shafay et al., 2025).')])

troca('Esses resultados são coerentes com o Quadro 1', [
    ('O empate entre ViT-B/16 e EfficientNet-B0 diverge da robustez atribuída aos '
     '*transformers* (Shafay et al., 2025), mas concorda com a evidência de que o ajuste fino '
     'completo não reduz o *domain gap* (Nguyen et al., 2026).',
     'O ViT-B/16, que empata no teste e é o pior nas culturas de lavoura, confirma a '
     'sensibilidade do ViT puro ao ambiente (Shafay et al., 2025), e o ajuste fino completo '
     'não reduz o *domain gap* (Nguyen et al., 2026).'),
    ('reduz o ECE do conjunto externo de 0,287 para 0,038 (Ergün et al., 2026)',
     'reduz o ECE do conjunto externo de 0,287 para 0,038, com confiança média de 0,55 '
     '(Ergün et al., 2026)'),
])

troca('Quatro regras garantem a validade do conjunto', [(
    'e fotografias ambíguas são excluídas com o motivo anotado;',
    'e fotografias ambíguas são marcadas, e o resultado é reportado com e sem elas;')])

open(path, 'w', encoding='utf8').write(d)
print('ok')
