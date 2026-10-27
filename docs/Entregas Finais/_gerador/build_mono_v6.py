# Monta a v6 da monografia (documento para o orientador) a partir da v5: nota de 27/10/2026 e
# seção 9.7 com o primeiro lote de campo com fundo real (docs/CORRECOES_METODOLOGICAS.md §15.13).
# Localiza os pontos de inserção pelo texto, não por índice.
# Uso: python build_mono_v6.py <dir v5 descompactado> <dir de saida descompactado>
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
i_h96 = acha(lambda t: t.startswith('9.6') and 'literatura' in t and len(t) < 80, 300)
i_ref = acha(lambda t: t == 'REFERÊNCIAS', i_h96)
i_quebra_refs = max(i for i in range(i_h96, i_ref) if ps[i] == QUEBRA)
i_nota16 = acha(lambda t: t == 'NOTA DE ATUALIZAÇÃO — 16 DE OUTUBRO DE 2026')
fim_nota16 = next(i for i in range(i_nota16, len(ps)) if ps[i] == QUEBRA) + 1

PPR_TEXTO = re.match(r'<w:p><w:pPr>.*?</w:pPr>', ps[i_h96 + 1]).group(0)[5:]
RPR_TEXTO = '<w:rPr><w:b w:val="0"/><w:i w:val="0"/></w:rPr>'
PPR_BULLET = re.match(r'<w:p><w:pPr>.*?</w:pPr>', ps[i_h96 + 2]).group(0)[5:]
H1_NOTA, H2_96 = ps[i_nota16], ps[i_h96]


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


# ------------------------------------------------------------- nota de 27/10
nota = [
    clona(H1_NOTA, 'NOTA DE ATUALIZAÇÃO — 27 DE OUTUBRO DE 2026'),
    P('Esta sexta versão acrescenta a seção 9.7, com o primeiro lote de fotografias de campo com fundo real: 10 '
      'fotografias de canteiros de alface sadia, tiradas em uma horta em 27 de outubro. As quatro versões avaliadas (o '
      'EfficientNet-B0 e o ViT-B/16 do quarto e do quinto ciclos) classificaram as 10 como sadias, e solo, mato e vários '
      'pés no quadro não provocaram falso alarme. Como não houve folhas doentes nesse lote, a sensibilidade em campo e o '
      '*domain gap* ainda não puderam ser medidos. O artigo passou à versão 10, e a conclusão da versão 9 foi a entregue '
      'na disciplina em 27 de outubro.'),
    QUEBRA,
]

# ------------------------------------------------------------- seção 9.7
secao = [
    clona(H2_96, '9.7  Primeiro lote de campo com fundo real'),
    P('Em 27 de outubro de 2026, às 16h35, foram tiradas 10 fotografias numa horta, todas de alface crespa sadia segundo '
      'a avaliação feita no local. A horta tinha poucos pés, encostados uns nos outros, e as fotografias foram tiradas de '
      'cima, com vários pés por quadro, sobre solo com mato e trevo. O rótulo foi registrado antes de qualquer inferência; '
      'em duas fotografias, um furo pequeno numa folha foi anotado como dúvida, sem mudar o rótulo. Um lote anterior, de '
      '18 de outubro, com um pé colhido na chuva e fotografado em casa sobre toalha branca (3 folhas sadias e 7 '
      'anômalas), não é de campo e é reportado à parte.'),
    P('Os modelos rodaram às cegas, sem nenhum ajuste. As quatro versões deram como sadias as 10 fotografias da horta, '
      'sem abstenção e com confiança acima de 0,98; a proporção de acertos é de 10 em 10, com intervalo de Wilson de '
      '72,2% a 100%. Para verificar se o resultado dependia de haver vários pés no quadro, cada fotografia foi recortada '
      'no centro, com metade e com um quarto do lado: os recortes também saíram sadios, nas quatro versões. No lote de '
      'casa, todas as fotografias diagnosticadas saíram anômalas, inclusive as três rotuladas como sadias, com confiança '
      'acima de 0,97, e os erros continuaram nos recortes, que excluem a toalha das bordas.'),
    BULLET('O que o lote mostra.',
           'Fundo real não gera falso alarme em alface sadia. Era o risco mais direto do domain gap — o modelo foi treinado '
           'sobretudo com folhas isoladas — e não apareceu em nenhuma versão nem em nenhum enquadramento.'),
    BULLET('O que o lote não mostra.',
           'Sem folhas doentes na horta, não há sensibilidade em campo, acurácia balanceada de campo nem domain gap. A '
           'acurácia balanceada que o relatório calcula para as 20 fotografias juntas não deve ser usada: todas as anômalas '
           'são do lote de casa e quase todas as sadias são da horta, de modo que a métrica mistura a classe com a cena. Pelo '
           'mesmo motivo, a tendência com a distância (H3) que o teste aponta é um artefato.'),
    BULLET('Folha ou cena.',
           'Os modelos dizem sadia para tudo na horta e anômala para tudo em casa. Com cada lote quase todo em uma classe, '
           'não se separa o efeito da folha do efeito da cena (luz, folha molhada e machucada pela colheita). Uma das três '
           'folhas de casa rotuladas como sadias mostra, na revisão posterior, pontos escuros e a base amarronzada; o '
           'rótulo da coleta é mantido, e a dúvida fica registrada.'),
    BULLET('Calibração.',
           'Os erros do lote de casa têm confiança acima do limiar de abstenção, como os do conjunto histórico: fora da '
           'distribuição de treino, o limiar não os intercepta. A recalibração com dados de campo não foi feita, porque '
           'exige uma horta de adaptação separada da de avaliação e as duas classes.'),
    P('Para responder a Q3, falta fotografar folhas doentes na horta (no mínimo 15), uma segunda horta para a adaptação, '
      'imagens sem folha para a guarda de vegetação e uma segunda cultura, como a couve, para a hipótese H1.'),
]

# ------------------------------------------------------------- sumário manual
sum_ini = acha(lambda t: t == 'SUMÁRIO')
i_s16 = acha(lambda t: t == 'NOTA DE ATUALIZAÇÃO — 16 DE OUTUBRO DE 2026', sum_ini)
i_s96 = acha(lambda t: t.startswith('9.6'), sum_ini)
assert fim_nota16 <= sum_ini < i_s16 < i_s96 < i_h96  # as notas vêm antes do sumário
ps[i_s16] = ps[i_s16] + clona(ps[i_s16], 'NOTA DE ATUALIZAÇÃO — 27 DE OUTUBRO DE 2026')
ps[i_s96] = ps[i_s96] + clona(ps[i_s96], '9.7  Primeiro lote de campo com fundo real')

out = (ps[:fim_nota16] + nota + ps[fim_nota16:i_quebra_refs] + secao + ps[i_quebra_refs:])
corpo = d[b:]
fim = corpo[corpo.rfind(ps[-1]) + len(ps[-1]):]
open(path, 'w', encoding='utf8').write(head + ''.join(out) + fim)
print('ok', len(out))
