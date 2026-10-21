# Monta a v4 da monografia (documento para o orientador) a partir da v3: nota de
# atualização de 14/10/2026 com errata, capítulo 9 (quatro arquiteturas, auditoria de
# consistência e quinto ciclo), uma referência nova e linhas novas no Apêndice A.
# Os capítulos 1 a 8 ficam como registro, como nas versões anteriores.
# Uso: python build_mono_v4.py <dir v3 descompactado> <dir de saida descompactado>
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
assert len(ps) == 498, len(ps)


def texto(p):
    return ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p))


assert texto(ps[43]).startswith('NOTA DE ATUALIZAÇÃO — OUTUBRO DE 2026')
assert texto(ps[408]).startswith('8  ATUALIZAÇÃO DO TCC-II')
assert ps[445] == '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'
assert texto(ps[446]) == 'REFERÊNCIAS'
assert texto(ps[490]).startswith('WOEBBECKE')
assert texto(ps[494]).startswith('EvidênciaArtefato')

PPR_TEXTO = re.match(r'<w:p><w:pPr>.*?</w:pPr>', ps[419]).group(0)[5:]
RPR_TEXTO = '<w:rPr><w:b w:val="0"/><w:i w:val="0"/></w:rPr>'
PPR_BULLET = re.match(r'<w:p><w:pPr>.*?</w:pPr>', ps[433]).group(0)[5:]
PPR_LEGENDA = re.match(r'<w:p><w:pPr>.*?</w:pPr>', ps[418]).group(0)[5:]
PPR_REF = re.match(r'<w:p><w:pPr>.*?</w:pPr>', ps[490]).group(0)[5:]


def runs(text, base=''):
    """*italico*, **negrito**; `base` é o rPr interno aplicado a tudo (ex.: tamanho)."""
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


def LEGENDA(text):
    return (f'<w:p>{PPR_LEGENDA}<w:r><w:rPr><w:sz w:val="19"/></w:rPr><w:t xml:space="preserve">{escape(text)} </w:t></w:r>'
            f'<w:r><w:rPr><w:i/><w:sz w:val="19"/></w:rPr><w:t>Fonte: elaboração própria (2026).</w:t></w:r></w:p>')


def REF(text):
    return f'<w:p>{PPR_REF}{runs(text, "<w:sz w:val=\"22\"/>")}</w:p>'


# Tabela no mesmo formato das do capítulo 8 (cabeçalho com fundo verde-escuro)
_t = ps[417]
_TBLPR = re.search(r'<w:tblPr>.*?</w:tblPr>', _t, re.S).group(0)
_trs = re.findall(r'<w:tr[ >].*?</w:tr>', _t, re.S)
_TC_HEAD = re.findall(r'<w:tc>.*?</w:tc>', _trs[0], re.S)[0]
_TC_BODY = re.findall(r'<w:tc>.*?</w:tc>', _trs[1], re.S)[1]   # centralizada


def _celula(modelo, largura, txt, alinhar=None):
    c = re.sub(r'<w:tcW w:type="dxa" w:w="\d+"/>', f'<w:tcW w:type="dxa" w:w="{largura}"/>', modelo)
    if alinhar:
        c = re.sub(r'<w:jc w:val="\w+"/>', f'<w:jc w:val="{alinhar}"/>', c)
    # runs simples, como nas tabelas originais (o cabeçalho herda a formatação da célula)
    return re.sub(r'<w:r>.*</w:r>', f'<w:r><w:t xml:space="preserve">{escape(txt)}</w:t></w:r>', c, flags=re.S)


def TABLE(widths, rows):
    grid = '<w:tblGrid>' + ''.join(f'<w:gridCol w:w="{w}"/>' for w in widths) + '</w:tblGrid>'
    trs = []
    for ri, row in enumerate(rows):
        tcs = [_celula(_TC_HEAD if ri == 0 else _TC_BODY, widths[ci], cel,
                       'left' if (ri > 0 and ci == 0) else None) for ci, cel in enumerate(row)]
        trs.append('<w:tr>' + ''.join(tcs) + '</w:tr>')
    return '<w:tbl>' + _TBLPR + grid + ''.join(trs) + '</w:tbl>'


QUEBRA = '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'

# ------------------------------------------------------------- nota de 14/10
nota = [
    H1('NOTA DE ATUALIZAÇÃO — 14 DE OUTUBRO DE 2026'),
    P('Esta quarta versão acrescenta o capítulo 9, com o que foi feito entre 12 e 14 de outubro de 2026: a comparação entre '
      'quatro arquiteturas, com a inclusão de um *Vision Transformer* (ViT-B/16); uma auditoria de consistência, que conferiu '
      'cada número do trabalho contra os artefatos de origem (particionamento em disco, registros de treino e previsões por '
      'imagem); e um quinto ciclo de treinamento, que corrige os defeitos encontrados pela auditoria. Os capítulos 1 a 8 '
      'continuam como registro dos estados de julho e de 8 de outubro.'),
    P('A auditoria corrigiu números que aparecem nos capítulos anteriores. Para não reescrever o registro, as correções ficam '
      'listadas aqui; o capítulo 9 dá os detalhes, e o arquivo docs/FICHA_TECNICA_FITOVISION.md é a fonte única dos números '
      'conferidos.'),
    BULLET('Composição do treino (seções 4.5, 8.2 e 8.5):',
           'onde se lê que cerca de 70% do treino deriva do PlantVillage ou é folha isolada sobre fundo homogêneo, leia-se: no '
           'máximo 47,7% deriva do PlantVillage. A fonte descrita como reedição do PlantVillage é uma coletânea que inclui '
           'mandioca, arroz, cana-de-açúcar e café fotografados em lavoura (17,6% do treino).'),
    BULLET('Número de parâmetros (resumo e tabela da seção 4.3):',
           '5,3, 3,4 e 25,6 milhões são as arquiteturas originais, de 1.000 classes. Os modelos treinados, com saída de duas '
           'classes, têm 4,0 (EfficientNet-B0), 2,2 (MobileNetV2) e 23,5 milhões (ResNet-50); o ViT-B/16 treinado tem 85,8 '
           'milhões, 21 vezes o EfficientNet-B0.'),
    BULLET('Tamanho do dataset:',
           'são três etapas, e não versões divergentes: 176.126 imagens reunidas, 170.772 nas partições da base (após o '
           'descarte de grupos com rótulos contraditórios) e 175.123 com os lotes do quarto ciclo.'),
    BULLET('Vazamento residual de 0,04%:',
           'vale para o critério de semelhança visual. O critério não alcança variantes muito alteradas da mesma fotografia: '
           '23% do teste da base são rotações, recortes e filtros de fotografias de alface com outra variante no treino '
           '(seção 9.2). O quinto ciclo corrige isso.'),
    BULLET('Fotografias de campo (seção 4.6):',
           'nenhuma das 13 fotografias do conjunto histórico tem metadados EXIF de telefone celular, e uma delas é de câmera '
           'digital de 2014, editada; a procedência de cada uma ainda será registrada. A auditoria confirmou que nenhuma é '
           'duplicata de imagem de treino.'),
    QUEBRA,
]

# ------------------------------------------------------------- capítulo 9
cap9 = [
    H1('9  ATUALIZAÇÃO DE OUTUBRO DE 2026: QUATRO ARQUITETURAS, AUDITORIA E QUINTO CICLO'),
    P('Este capítulo registra três etapas executadas entre 12 e 14 de outubro de 2026. A primeira completa a resposta à '
      'pergunta Q1 com uma quarta arquitetura. A segunda é uma auditoria que conferiu cada número do trabalho contra os '
      'artefatos de origem e encontrou, dentro do próprio conjunto de teste, onde estão os erros do modelo. A terceira é o '
      'quinto ciclo de treinamento, que corrige os defeitos encontrados e mostra o que muda quando um atalho dos dados é '
      'removido.'),

    H2('9.1  Quatro arquiteturas'),
    P('O ViT-B/16 foi treinado em GPU L4, no Google Colab, com os mesmos dados, a mesma perda, o mesmo critério de seleção e '
      'a mesma calibração das CNNs; mudaram apenas os hiperparâmetros de otimização, ajustados ao que é usual no ajuste fino '
      'de *transformers* (taxa de aprendizado de 5×10⁻⁵, decaimento de pesos de 0,05, duas épocas de aquecimento e limitação '
      'da norma do gradiente). A Tabela 13 reúne as quatro arquiteturas no conjunto de teste da base, com o modelo do quarto '
      'ciclo.'),
    TABLE([1950, 950, 1150, 1100, 1000, 1050, 700, 1170], [
        ['Modelo', 'Parâm.', 'Acur. bal.', 'F1 macro', 'AUC', 'ECE', 'FN', 'Latência'],
        ['EfficientNet-B0', '4,0 M', '98,48%', '98,44%', '0,9982', '0,0039', '209', '7,2 ms'],
        ['MobileNetV2', '2,2 M', '97,89%', '97,83%', '0,9976', '0,0055', '294', '5,3 ms'],
        ['ResNet-50', '23,5 M', '97,82%', '97,70%', '0,9966', '0,0061', '352', '8,6 ms'],
        ['ViT-B/16', '85,8 M', '98,48%', '98,53%', '0,9976', '0,0018', '137', '19,4 ms'],
    ]),
    LEGENDA('Tabela 13 — Desempenho das quatro arquiteturas no conjunto de teste da base (23.967 imagens), quarto ciclo. '
            'Parâmetros dos modelos treinados, com saída de duas classes; FN = falsos negativos; latência em GPU RTX 3050, '
            'medida na mesma sessão.'),
    P('EfficientNet-B0 e ViT-B/16 empatam: mesma acurácia balanceada, mesmo intervalo de confiança de 95% (98,32% a 98,64%) '
      'e nenhuma diferença pelo teste de McNemar exato com correção de Holm (p = 0,26). Ambos superam MobileNetV2 e ResNet-50 '
      '(p ≤ 1,4×10⁻¹²). O ViT-B/16 comete menos falsos negativos e mais falsos positivos, tem 21 vezes mais parâmetros e é '
      '2,7 vezes mais lento. Na abstenção, intercepta só 15% dos seus erros, contra 31% do EfficientNet-B0, embora tenha o '
      'menor ECE da tabela: boa calibração média não garante que os erros fiquem nas confianças baixas. Um transformer muito '
      'maior chega ao mesmo lugar, o que reforça a leitura de que o limite do sistema está nos dados (Dosovitskiy et al., '
      '2021).'),

    H2('9.2  Auditoria de consistência'),
    P('Antes da escrita final, cada número citado nos documentos do projeto foi conferido no artefato que o produz, com um '
      '*script* próprio (backend/auditoria_consistencia.py). A auditoria trouxe três achados de método, além das correções '
      'listadas na nota de atualização.'),
    P('**Onde estão os erros do teste.** As previsões por imagem, agrupadas por fonte e por cultura, mostram que a média de '
      '98,48% junta situações muito diferentes (Tabela 14). Dentro da mesma coletânea, as culturas fotografadas em lavoura '
      '— mandioca, arroz, cana-de-açúcar e café — têm acurácia balanceada de 86,3% e concentram 81% dos erros do '
      'EfficientNet-B0 (295 de 363), enquanto as demais culturas, fotografadas em estúdio, chegam a 99,7%. A mandioca sozinha '
      'fica em 80,9%. Nas culturas de lavoura, as quatro arquiteturas ficam entre 83,4% e 87,8%, e o ViT-B/16 é o pior.'),
    TABLE([4571, 1500, 3000], [
        ['Subconjunto do teste da base', 'N', 'Acurácia balanceada'],
        ['Coletânea, culturas fotografadas em estúdio', '10.061', '99,7%'],
        ['Coletânea, culturas fotografadas em lavoura', '4.530', '86,3% (84,5% a 88,0%)'],
        ['Só mandioca', '3.150', '80,9%'],
        ['Teste inteiro', '23.967', '98,48% (98,32% a 98,64%)'],
    ]),
    LEGENDA('Tabela 14 — Acurácia balanceada do EfficientNet-B0 (quarto ciclo) por subconjunto do conjunto de teste da base, '
            'com intervalo de confiança de 95% por bootstrap.'),
    P('**Vazamento por proveniência.** A maior fonte de alface é feita de 17 transformações (rotações de ângulo livre, '
      'recortes e filtros de borda) de cada uma de 1.231 fotografias. Essas variantes não ficam parecidas com a original para '
      'o *hash* perceptual, de modo que o particionamento por semelhança as espalhou pelas três partições: 99,6% das variantes '
      'de alface do teste têm outra variante da mesma fotografia no treino, o equivalente a 23% do teste. É o mesmo defeito '
      'corrigido no quarto ciclo para os 49 originais de míldio, aplicado agora à fonte inteira. Sem essas variantes, o teste '
      'dá 97,8%, e a conclusão sobre as arquiteturas se mantém.'),
    P('**Cultura associada ao rótulo e pastas fora do dataset.** A alface responde por 29,9% do treino, mas 86,6% dela é '
      'saudável e ela vem de apenas 1.231 fotografias originais, 108 doentes; várias culturas da coletânea têm um só rótulo. '
      'Uma rede que aprende "alface tende a ser saudável" ou "foto de lavoura tende a ser doente" acerta muito no teste e '
      'erra fora dele — o padrão de *shortcut learning* descrito por Geirhos et al. (2020). A auditoria também encontrou '
      'pastas de dano por praga que a rotulagem automática não reconheceu e deixou fora do dataset.'),
    P('Na calibração, a temperatura ajustada na validação é menor que 1 nos quatro modelos (0,546 a 0,620), isto é, deixa '
      'as probabilidades mais confiantes. Nas 13 fotografias de campo isso piora a calibração: o ECE é de 0,146 com a '
      'temperatura e de 0,094 sem ela. Dos nove erros de campo dos quatro ciclos, quatro vêm de uma mesma fotografia rotulada '
      'como saudável que mostra manchas na folha; o rótulo será reavaliado.'),

    H2('9.3  Quinto ciclo: o que foi corrigido'),
    P('O quinto ciclo foi montado em um particionamento novo, sem alterar nada do quarto ciclo, que permanece como modelo '
      'servido. As correções foram:'),
    BULLET('Alface por fotografia original.',
           'O identificador no nome do arquivo e o hash juntam as variantes de cada fotografia; as 1.231 fotografias formam '
           '1.172 unidades, repartidas inteiras. Validação e teste recebem uma fotografia por original, sem as variantes '
           'filtradas: o teste de alface passa a ter 176 fotografias, 15 doentes.'),
    BULLET('Pastas de praga rotuladas.',
           'Entram como anômalas 8.266 imagens de pastas que a rotulagem deixava de fora, quase todas de dano por praga; as da '
           'coletânea seguem a partição de qualquer quase-duplicata já existente.'),
    BULLET('Amostragem por cultura.',
           'Cada cultura mantém a sua fatia do treino, mas dentro dela saudável e anômala são sorteadas meio a meio; a perda '
           'deixa de ter pesos de classe, para não compensar a classe duas vezes.'),
    BULLET('Lote retido.',
           'Um lote de espinafre-de-malabar (574 imagens) fica fora do treino e da validação. Outro lote da mesma espécie '
           'continua no treino: a medida é de troca de lote, mais branda que a de fonte nunca vista do terceiro ciclo.'),
    BULLET('Mesma partição no resto.',
           'A coletânea e o PlantVillage mantêm a partição do quarto ciclo, o que permite comparar os dois ciclos imagem a '
           'imagem nas mesmas fotografias.'),
    P('O conjunto tem 152.057 imagens: 112.458 de treino, 20.612 de validação, 18.413 de teste e 574 retidas. A auditoria de '
      'quase-duplicatas encontrou 1,7% do teste a distância de órbita ≤ 4 de alguma imagem de treino, contra 23% de variantes '
      'no quarto ciclo; parte desses pares são colisões entre culturas diferentes, de modo que o valor é um teto. As quatro '
      'arquiteturas foram treinadas em GPU L4, no Google Colab, com as mesmas receitas do quarto ciclo, em cerca de 12 horas.'),

    H2('9.4  Quinto ciclo: resultados'),
    TABLE([1950, 850, 1480, 1480, 1480, 880, 950], [
        ['Modelo', 'Teste', 'Lavoura', 'Espec.', 'Sens.', 'Alface', 'Retido'],
        ['EfficientNet-B0', '97,4', '86,3 → 88,8', '76,6 → 83,4', '96,0 → 94,2', '96,7', '87,9'],
        ['MobileNetV2', '96,9', '86,6 → 90,1', '78,3 → 90,2', '94,9 → 90,0', '100', '88,4'],
        ['ResNet-50', '96,9', '87,8 → 89,6', '81,4 → 89,0', '94,2 → 90,2', '100', '90,1'],
        ['ViT-B/16', '96,9', '83,4 → 84,5', '69,8 → 72,1', '97,0 → 96,9', '100', '85,8'],
    ]),
    LEGENDA('Tabela 15 — Quinto ciclo: acurácia balanceada (%) no teste do ciclo (18.413 imagens, sem as variantes de '
            'alface; não comparável à Tabela 13). Lavoura, Espec. e Sens.: acurácia balanceada, especificidade e '
            'sensibilidade nas culturas fotografadas em lavoura (N = 4.530), quarto ciclo → quinto ciclo, mesmas imagens. '
            'Alface: uma fotografia por original (N = 176). Retido: lote de espinafre fora do treino (N = 574).'),
    P('A correção agiu onde estavam os erros. Nas culturas fotografadas em lavoura, as três CNNs ganham de 1,8 a 3,5 pontos '
      'de acurácia balanceada e de 6,8 a 11,9 pontos de especificidade: passam a emitir bem menos falsos alarmes em folhas '
      'sadias. O MobileNetV2, que declarava doente quase tudo o que vinha de lavoura (especificidade de 78,3%, sensibilidade '
      'de 94,9%), fica equilibrado em 90,2% e 90,0%. O ViT-B/16 quase não muda (McNemar, p = 0,66).'),
    P('O ganho tem custo. A sensibilidade nas culturas de lavoura cai de 1,8 a 4,9 pontos nas CNNs e, contado imagem a '
      'imagem, o quarto ciclo acerta mais: o McNemar exato favorece o quarto ciclo no EfficientNet-B0 (126 contra 94 '
      'imagens, p = 0,04) e, com folga, no MobileNetV2 e no ResNet-50 (p < 10⁻¹¹), porque três em cada quatro imagens desse '
      'subconjunto são anômalas e cada acerto conta igual. A acurácia balanceada pesa as duas classes por igual e sobe. As '
      'duas leituras são verdadeiras e precisam ser citadas juntas: o quinto ciclo não é melhor em tudo, é outro ponto de '
      'operação. Qual dos dois serve melhor depende do custo de cada erro na horta — um falso alarme custa uma inspeção; uma '
      'doença perdida pode custar o canteiro.'),
    P('No lote de espinafre retido, os quatro modelos ficam entre 85,8% e 90,1%, cerca de dez pontos abaixo das fontes de '
      'treino, mesmo com outro lote da espécie no treino; é a medida de mudança de fonte mais limpa que o trabalho tem, e a '
      'ordem de grandeza é a mesma da queda por fonte disjunta descrita para amêndoa por Workneh et al. (2026). Nas 176 '
      'fotografias de alface sem variantes, os modelos acertam de 96,7% a 100%, mas há apenas 15 doentes, e o limite inferior '
      'do intervalo chega a 90%: a pergunta sobre a cultura-alvo continua dependendo do campo.'),
    P('Dois resultados merecem registro à parte. Primeiro, a validação não ordena os modelos pelo que importa: ela põe '
      'EfficientNet-B0 e ViT-B/16 à frente (F1 macro de 0,972, contra 0,960 e 0,961), e na lavoura e no lote retido a ordem '
      'se inverte. Segundo, nas 13 fotografias do conjunto histórico, o EfficientNet-B0 do quinto ciclo acerta 11 dos 12 '
      'diagnósticos que emite, com uma abstenção, e empata com o quarto ciclo (cada um acerta uma fotografia que o outro '
      'erra; p = 1): com essa amostra, as versões continuam indistinguíveis.'),

    H2('9.5  O que muda'),
    P('O modelo servido continua o EfficientNet-B0 do quarto ciclo até a avaliação em campo. Os modelos do quarto e do quinto '
      'ciclos foram congelados antes da coleta e serão comparados, às cegas, sobre as mesmas fotografias; se a horta mostrar o '
      'padrão da lavoura — falsos alarmes em folhas sadias —, o EfficientNet-B0 do quinto ciclo é o candidato, com a perda de '
      'sensibilidade dita explicitamente. Um sexto ciclo, com fotografias de horta no treino, depende da separação por horta '
      'em conjuntos de avaliação e de adaptação proposta na seção 8.5.'),
    P('As lições desta etapa somam-se às do capítulo 5 e da seção 8.6. Uma média esconde onde o modelo erra: os 98,48% '
      'juntavam 99,7% em estúdio e 86% em lavoura, e as previsões por imagem já estavam salvas. Uma lição aprendida em um '
      'lote precisa ser aplicada ao dataset inteiro: o particionamento por proveniência feito para o míldio faltava na fonte '
      'de alface. Documentos precisam ser conferidos contra os artefatos, e não uns contra os outros. E corrigir um atalho '
      'muda o ponto de operação, e não só a acurácia: por isso especificidade e sensibilidade são reportadas lado a lado.'),
    QUEBRA,
]

# ------------------------------------------------------------- referência nova
ref_workneh = REF('WORKNEH, A. D. et al. Leakage-aware, calibrated, and explainable deep learning for robust almond disease '
                  'classification. **Computers and Electronics in Agriculture**, v. 256, 112464, 2026. DOI: '
                  '10.1016/j.compag.2026.112464.')

# ------------------------------------------------------------- Apêndice A
ap = ps[494]
trs = re.findall(r'<w:tr[ >].*?</w:tr>', ap, re.S)
modelo = trs[-1]
novas = ''
for ev, art in (('Comparação das quatro arquiteturas (capítulo 9.1)', 'results/arquiteturas/comparacao_arquiteturas.md'),
                ('Auditoria de consistência (capítulo 9.2)', 'results/auditoria/auditoria_13out.{json,md}; docs/FICHA_TECNICA_FITOVISION.md'),
                ('Quinto ciclo: particionamento e auditoria', 'data_c5/split_metadata.json; results/ciclo5/leakage_report.json'),
                ('Quinto ciclo: avaliação por subconjunto e quarto × quinto ciclo', 'results/ciclo5/avaliacao_ciclo5.md; results/ciclo5/ciclo4_x_ciclo5.md'),
                ('Quinto ciclo: pesos e calibração', 'weights/ciclo5/')):
    linha = modelo
    for tc, txt in zip(re.findall(r'<w:tc>.*?</w:tc>', modelo, re.S), (ev, art)):
        ts = re.findall(r'<w:t[^>]*>[^<]*</w:t>', tc)
        novo = tc.replace(ts[0], f'<w:t xml:space="preserve">{escape(txt)}</w:t>', 1)
        for t in ts[1:]:
            novo = novo.replace(t, '<w:t></w:t>', 1)
        linha = linha.replace(tc, novo, 1)
    novas += linha
ap = ap.replace(trs[-1], trs[-1] + novas, 1)

body = ps[0:47] + nota + ps[47:445] + cap9 + ps[445:491] + [ref_workneh] + ps[491:494] + [ap] + ps[495:]
open(path, 'w', encoding='utf8').write(head + ''.join(body) + '</w:body></w:document>')
print('blocos:', len(body))
