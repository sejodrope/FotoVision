# Monta a v5 da monografia (documento para o orientador) a partir da v4: nota de 16/10/2026,
# seção 9.6 com a leitura do texto completo da literatura (docs/REVISAO_LITERATURA_WOS.md,
# Parte 4) e as referências novas, em ordem alfabética.
# Uso: python build_mono_v5.py <dir v4 descompactado> <dir de saida descompactado>
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
assert len(ps) == 542, len(ps)


def texto(p):
    return ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p))


assert texto(ps[47]) == 'NOTA DE ATUALIZAÇÃO — 14 DE OUTUBRO DE 2026'
assert texto(ps[484]).startswith('9.5')
assert ps[487] == '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'
assert texto(ps[489]) == 'REFERÊNCIAS'
assert texto(ps[534]).startswith('WORKNEH')

PPR_TEXTO = re.match(r'<w:p><w:pPr>.*?</w:pPr>', ps[485]).group(0)[5:]
RPR_TEXTO = '<w:rPr><w:b w:val="0"/><w:i w:val="0"/></w:rPr>'
PPR_BULLET = re.match(r'<w:p><w:pPr>.*?</w:pPr>', ps[50]).group(0)[5:]
PPR_REF = re.match(r'<w:p><w:pPr>.*?</w:pPr>', ps[490]).group(0)[5:]


def runs(text, base=''):
    out = []
    for part in re.split(r'(\*\*.+?\*\*|\*.+?\*)', text):
        if not part:
            continue
        extra = ''
        if part.startswith('**'):
            part, extra = part[2:-2], '<w:b/>'
        elif part.startswith('*'):
            part, extra = part[1:-1], '<w:i/>'
        rpr = f'<w:rPr>{extra}{base}</w:rPr>' if (extra or base) else RPR_TEXTO
        out.append(f'<w:r>{rpr}<w:t xml:space="preserve">{escape(part)}</w:t></w:r>')
    return ''.join(out)


def P(text):
    return f'<w:p>{PPR_TEXTO}{runs(text)}</w:p>'


def H1(t):
    return f'<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t xml:space="preserve">{escape(t)}</w:t></w:r></w:p>'


def H2(t):
    return f'<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t xml:space="preserve">{escape(t)}</w:t></w:r></w:p>'


def BULLET(lead, text):
    return (f'<w:p>{PPR_BULLET}<w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">{escape(lead)}</w:t></w:r>'
            f'<w:r><w:t xml:space="preserve"> {escape(text)}</w:t></w:r></w:p>')


def REF(text):
    return f'<w:p>{PPR_REF}{runs(text, "<w:sz w:val=\"22\"/>")}</w:p>'


QUEBRA = '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'

# ------------------------------------------------------------- nota de 16/10
nota = [
    H1('NOTA DE ATUALIZAÇÃO — 16 DE OUTUBRO DE 2026'),
    P('Esta quinta versão acrescenta a seção 9.6. Os trabalhos relacionados tinham sido lidos pelos resumos indexados na '
      'Web of Science; em 16 de outubro, sete deles foram lidos na íntegra, e a leitura corrigiu duas afirmações. A revisão '
      'de Shafay et al. (2025) não diverge do resultado do ViT-B/16: ela aponta as arquiteturas híbridas como mais robustas '
      'e o ViT puro como mais sensível ao ambiente, o que concorda com o desempenho do ViT-B/16 nas fotografias de lavoura. '
      'E a queda de 98,7% para cerca de 31% atribuída a Gerz e Jelali (2026) é de mAP50 em detecção de objetos, e não de '
      'acurácia. O artigo passou à versão 8 com essas correções.'),
    QUEBRA,
]

# ------------------------------------------------------------- seção 9.6
secao = [
    H2('9.6  O que a literatura lida na íntegra acrescenta'),
    P('Sete dos trabalhos relacionados foram lidos na íntegra em 16 de outubro de 2026 (os de Workneh et al., 2026, e Tian '
      'et al., 2024, aguardam acesso ao texto completo). A leitura reforça a interpretação deste trabalho em quatro pontos e '
      'acrescenta ressalvas a três resultados citados.'),
    BULLET('A queda é de fonte, não só de fundo.',
           'No estudo com videira, o treino já é de fotografias de campo, e mesmo assim a acurácia cai de 99,6% para 72% a '
           '74% em folhas de outro vinhedo, outro país e outro celular (Ergün et al., 2026). Isso fortalece o lote retido como '
           'medida de generalização: a mudança de fonte, sozinha, já custa pontos.'),
    BULLET('O atalho pela cultura tem mecanismo descrito.',
           'Treinadas no PlantVillage e testadas em fotografias de campo, as CNNs atribuem a folha à cultura errada antes de '
           'olhar o sintoma: só 21% a 34% das folhas de macieira ficam na cultura certa, contra 92% com um modelo '
           'auto-supervisionado congelado (Nguyen et al., 2026). É o mesmo atalho encontrado pela auditoria, em que a cultura '
           'decide o rótulo, e a razão da amostragem por cultura do quinto ciclo.'),
    BULLET('O ViT puro converge com a literatura.',
           'A revisão de Shafay et al. (2025) aponta arquiteturas híbridas como mais robustas em campo e o ViT puro como mais '
           'sensível ao ambiente, apesar da vantagem em laboratório. O ViT-B/16 deste trabalho empata no teste e é o pior nas '
           'fotografias de lavoura nos dois ciclos. A arquitetura híbrida (Swin) não foi testada aqui. O ajuste fino '
           'completo, usado em todos os modelos deste trabalho, pode piorar a transferência: com o backbone congelado, um '
           'modelo de fundação chega a 50,6% no PlantDoc, e ajustado por inteiro cai para 29,4% (Nguyen et al., 2026).'),
    BULLET('O vazamento por proveniência tem nome.',
           'A literatura separa o vazamento induzido pelo processo, quando os dados são aumentados antes da divisão, do '
           'inerente aos dados, quando o conjunto público já traz duplicatas ou a versão aumentada (Gerz; Jelali, 2026). A '
           'maior fonte de alface deste trabalho é o segundo caso: um conjunto publicado com 17 transformações de cada foto.'),
    P('As ressalvas. No estudo com videira, a calibração reduz o ECE externo de 0,287 para 0,038, mas a confiança média '
      'cai para 0,55 num problema binário, e não se reporta acurácia seletiva; o intervalo de confiança da acurácia externa '
      '(±0,7 ponto com 500 imagens) parece calculado sobre as médias dos cinco *folds*, e as folhas ambíguas foram excluídas '
      '(Ergün et al., 2026). A temperatura vai no sentido oposto ao deste trabalho (8,34 lá, de 0,55 a 0,68 aqui), e as duas '
      'situações mostram que a calibração ajustada na validação não vale fora dela. Nos dois trabalhos com hortaliças '
      'folhosas, os resultados são intra-fonte: 97,3% num teste de cerca de 105 imagens, sem teste externo (Kabya et al., '
      '2026), e 94,1% na validação, com imagens coletadas da web e aumento de dados aparentemente anterior à divisão '
      '(Abbasi; Martinez; Ahmad, 2023). E a queda de 98,7% para cerca de 31% em Gerz e Jelali (2026) é de mAP50 em '
      'detecção de objetos; o próprio artigo traz 32,77% no resumo e 30,64% na tabela.'),
    P('Duas consequências para a coleta de campo. Como no estudo com videira, as fotografias serão separadas por local e '
      'rotuladas no campo antes de qualquer inferência; diferentemente dele, as folhas ambíguas serão marcadas, e não '
      'descartadas, para que o resultado não fique otimista. E nenhum dos trabalhos lidos mede se a abstenção intercepta '
      'os erros fora da fonte; Nguyen et al. (2026) apontam a rejeição de imagens fora da distribuição como trabalho '
      'futuro. Essa medida, com as fotografias de campo, é uma contribuição própria deste trabalho.'),
]

# ------------------------------------------------------------- referências novas
novas = {
    'ABBASI': 'ABBASI, R.; MARTINEZ, P.; AHMAD, R. Crop diagnostic system: a robust disease detection and management system '
              'for leafy green crops grown in an aquaponics facility. **Artificial Intelligence in Agriculture**, v. 10, p. '
              '1-12, 2023. DOI: 10.1016/j.aiia.2023.09.001.',
    'ERGÜN': 'ERGÜN, E. et al. Confidence and uncertainty aware deep learning for reliable grape leaf disease diagnosis under '
             'real world field conditions. **Frontiers in Plant Science**, v. 17, 1839652, 2026. DOI: '
             '10.3389/fpls.2026.1839652.',
    'GERZ': 'GERZ, F.; JELALI, M. Towards reliable data augmentation in machine learning: practices to prevent data leakage. '
            '**Smart Agricultural Technology**, v. 15, 102462, 2026. DOI: 10.1016/j.atech.2026.102462.',
    'KABYA': 'KABYA, N. D. et al. Towards practical AI for agriculture: a self-supervised attention framework for spinach '
             'leaf disease detection. **PLOS ONE**, v. 21, n. 1, e0340989, 2026. DOI: 10.1371/journal.pone.0340989.',
    'NGUYEN': 'NGUYEN, T. A. et al. From laboratory to field: frozen foundation-model features toward robust plant disease '
              'recognition. **Smart Agricultural Technology**, v. 15, 102531, 2026. DOI: 10.1016/j.atech.2026.102531.',
    'SHAFAY': 'SHAFAY, M. et al. Recent advances in plant disease detection: challenges and opportunities. **Plant '
              'Methods**, v. 21, 140, 2025. DOI: 10.1186/s13007-025-01450-0.',
}

refs_idx = [i for i in range(490, 535)]
existentes = {texto(ps[i]).split(',')[0].split(' ')[0] for i in refs_idx}
for k in novas:
    assert k not in existentes, k

# chave de ordenação sem acento
def chave(t):
    import unicodedata
    return unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode().upper()

refs = [(texto(ps[i]), ps[i]) for i in refs_idx] + [(t, REF(t)) for t in novas.values()]
refs.sort(key=lambda x: chave(x[0]))

# ------------------------------------------------------------- sumário manual
# A v4 não acrescentou ao sumário a nota de 14/10 nem o capítulo 9; entram agora.
sum_ini = next(i for i in range(60, 200) if texto(ps[i]) == 'SUMÁRIO')
i_nota = next(i for i in range(sum_ini, sum_ini + 10) if texto(ps[i]) == 'NOTA DE ATUALIZAÇÃO — OUTUBRO DE 2026')
i_86 = next(i for i in range(sum_ini, sum_ini + 80) if texto(ps[i]).startswith('8.6'))
i_cap8 = next(i for i in range(sum_ini, sum_ini + 80) if texto(ps[i]).startswith('8  ATUALIZA'))


def clona(modelo, novo):
    t = re.findall(r'<w:t[^>]*>[^<]*</w:t>', modelo)
    assert t
    m = modelo.replace(t[0], f'<w:t xml:space="preserve">{escape(novo)}</w:t>', 1)
    for x in t[1:]:
        m = m.replace(x, '<w:t></w:t>', 1)
    return m


ps[i_nota] = ps[i_nota] + clona(ps[i_nota], 'NOTA DE ATUALIZAÇÃO — 14 DE OUTUBRO DE 2026') +     clona(ps[i_nota], 'NOTA DE ATUALIZAÇÃO — 16 DE OUTUBRO DE 2026')
ps[i_86] = ps[i_86] + clona(ps[i_cap8], '9  ATUALIZAÇÃO DE OUTUBRO DE 2026: QUATRO ARQUITETURAS, AUDITORIA E QUINTO CICLO') +     ''.join(clona(ps[i_86], t) for t in [
        '9.1  Quatro arquiteturas', '9.2  Auditoria de consistência', '9.3  Quinto ciclo: o que foi corrigido',
        '9.4  Quinto ciclo: resultados', '9.5  O que muda', '9.6  O que a literatura lida na íntegra acrescenta'])

# a nota de 16/10 entra depois da de 14/10 (que termina numa quebra de página)
fim_nota14 = next(i for i in range(48, 487) if ps[i] == QUEBRA) + 1
out = (ps[:fim_nota14] + nota + ps[fim_nota14:487] + secao + ps[487:490]
       + [r for _, r in refs] + ps[535:])
# reconstrói o corpo preservando o que vier depois dos blocos (</w:body></w:document>)
corpo = d[b:]
fim = corpo[corpo.rfind(ps[-1]) + len(ps[-1]):]
open(path, 'w', encoding='utf8').write(head + ''.join(out) + fim)
print('ok', len(out))
