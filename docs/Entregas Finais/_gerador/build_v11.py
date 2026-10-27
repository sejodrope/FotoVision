# Monta a v11 do artigo a partir do document.xml da v10: o orientador pediu 10 páginas
# (27/10/2026), e a v10 tinha 11. Sai o Quadro 2 (objetivos x dados x técnicas; a Seção 3.3 já
# descreve as técnicas por pergunta) e o texto é enxugado sem mudar resultados.
# Uso: python build_v11.py <dir v10 descompactado> <dir de saida descompactado>
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


def apaga_quadro(legenda, fonte):
    """Apaga legenda, tabela e fonte de um quadro, conferindo que são vizinhos."""
    global d
    i = d.index(acha(legenda))
    j = d.index(acha(fonte)) + len(acha(fonte))
    trecho = d[i:j]
    assert trecho.count('<w:tbl>') == 1 and len(re.findall(r'<w:p[ >]', trecho.split('</w:tbl>')[1])) == 1
    d = d[:i] + d[j:]


apaga_quadro('Quadro 2. Relação entre objetivos', 'Fonte: elaboração própria (2026).')
troca('A análise responde às perguntas Q1 a Q3', [(
    'A análise responde às perguntas Q1 a Q3, e o Quadro 2 relaciona cada objetivo específico aos '
    'dados e às técnicas. Adota-se α = 0,05, com',
    'A análise responde às perguntas Q1 a Q3, com α = 0,05 e')])

# ---------------------------------------------------------------- resumo e abstract
troca_xml(' baseado em transfer learning, treinado com imagens públicas auditadas contra vazamento de '
          'dados e dotado de calibração de confiança e abstenção,',
          ' treinado com imagens públicas auditadas contra vazamento e dotado de calibração e abstenção,')
troca_xml('Um quinto ciclo, que separa a alface pela fotografia original e equilibra os rótulos dentro de '
          'cada cultura, reduz', 'Um quinto ciclo, que corrige atalhos nos dados, reduz')
troca_xml(' based on transfer learning, trained on public images audited for data leakage and equipped '
          'with confidence calibration and abstention,',
          ' trained on public images audited for leakage and equipped with calibration and abstention,')
troca_xml('A fifth cycle that splits lettuce by original photograph and balances labels within each crop '
          'cuts', 'A fifth cycle that removes data shortcuts cuts')

# ---------------------------------------------------------------- 1 e 2
substitui('A inteligência artificial (IA) consolidou-se', [])
troca('Na agricultura, uma das aplicações mais estudadas', [(
    'Na agricultura, uma das aplicações mais estudadas é',
    'Na agricultura, uma das aplicações mais estudadas do aprendizado profundo é')])
substitui('Esta seção apresenta os modelos, os conceitos', [])
troca('As CNNs extraem hierarquias', [(
    'Na versão original de 1.000 classes, a ResNet-50, que introduziu as conexões residuais, tem 25,6 '
    'milhões de parâmetros (He et al., 2016); a MobileNetV2, com convoluções separáveis e blocos '
    'residuais invertidos, 3,4 milhões (Sandler et al., 2018); e a EfficientNet-B0, que equilibra '
    'profundidade, largura e resolução por escalonamento composto, 5,3 milhões (Tan; Le, 2019).',
    'A ResNet-50 introduziu as conexões residuais (He et al., 2016); a MobileNetV2 usa convoluções '
    'separáveis e blocos residuais invertidos (Sandler et al., 2018); e a EfficientNet-B0 equilibra '
    'profundidade, largura e resolução por escalonamento composto (Tan; Le, 2019).')])

# ---------------------------------------------------------------- 3.3
substitui('Para Q3, cada fotografia de campo tem três desfechos', [])
troca('Como as amostras de campo são pequenas', [(
    'Como as amostras de campo são pequenas,',
    'Em campo (Q3), as abstenções têm taxa própria, as métricas de acerto são calculadas sobre os '
    'diagnósticos emitidos, e as imagens sem folha medem a guarda de vegetação. Como as amostras '
    'são pequenas,')])
troca('Ainda em Q3, a acurácia de campo', [
    ('H1 e H3 são testadas pelo teste exato de Freeman-Halton, H3 também pelo de Cochran-Armitage, e '
     'H2 pelo teste exato de Fisher (Agresti, 2013);',
     'As três são testadas pelos testes exatos de Fisher (H2) e de Freeman-Halton (H1 e H3) e, em '
     'H3, também pelo de Cochran-Armitage (Agresti, 2013);'),
    ('e classificado quanto à causa provável: sintoma sutil, enquadramento amplo, fundo com '
     'vegetação, iluminação ou rótulo duvidoso.', 'e classificado quanto à causa provável.')])

# ---------------------------------------------------------------- 3.1
troca('O dataset binário', [(
    'A rotulagem é automática, por busca em largura na árvore de diretórios, com cada decisão '
    'pasta-rótulo registrada em um mapa de auditoria.',
    'A rotulagem é automática, pela árvore de diretórios, com cada decisão registrada para auditoria.')])
troca('As três CNNs são inicializadas', [(
    'O ViT-B/16 segue o mesmo protocolo de dados, perda, seleção e calibração, com hiperparâmetros '
    'usuais no ajuste fino de *transformers*: taxa de',
    'O ViT-B/16 segue o mesmo protocolo, com taxa de')])

# ---------------------------------------------------------------- 4 e 5
troca('Na Tabela 3, a correção agiu', [(
    'dez pontos abaixo das fontes de treino; na alface sem variantes, entre 96,7% e 100%, com só 15 '
    'doentes.', 'dez pontos abaixo das fontes de treino.')])
troca('A coleta com fundo real responde a Q3', [(
    ' Vê-se que o *domain gap* surge mesmo entre fontes de folha isolada e se concentra na lavoura, e '
    'corrigir atalhos nos dados (quinto ciclo) reduz falsos alarmes ao custo de sensibilidade.', '')])

# ---------------------------------------------------------------- rodada 3: introdução, revisão, 3.2
troca('Os trabalhos relacionados mostram um contraste', [
    ('mas sobre imagens de estúdio — folha isolada, fundo homogêneo, iluminação controlada.',
     'mas em imagens de estúdio, com folha isolada e fundo homogêneo.'),
    ('Estudos recentes passaram a combinar validação sem vazamento de dados, calibração da confiança '
     'e abstenção', 'Estudos recentes combinam validação sem vazamento, calibração e abstenção')])
troca('O problema deste trabalho nasce', [(
    'A questão passa a ser quanto um sistema desse tipo vale em uso real; e, como um sistema que erra '
    'com aparência de certeza transfere confiança indevida ao produtor, a calibração da confiança e a '
    'capacidade de abster-se fazem parte do objeto de estudo.',
    'Como um sistema que erra com aparência de certeza transfere confiança indevida ao produtor, a '
    'calibração e a abstenção fazem parte do objeto de estudo.')])
troca('Vazamento de dados designa', [(
    'Em imagens, a forma mais comum é a duplicação quase exata — a mesma fotografia com recorte, '
    'compressão ou rotação diferentes — distribuída entre treino e teste por um particionamento '
    'aleatório por arquivo.',
    'Em imagens, a forma mais comum é a mesma fotografia, com recorte, compressão ou rotação '
    'diferentes, no treino e no teste.')])
troca('A busca por trabalhos relacionados', [(
    'A busca por trabalhos relacionados foi feita na Web of Science, em outubro de 2026, combinando '
    'doença de plantas e aprendizado profundo com generalização em campo, vazamento de dados, '
    'calibração e hortaliças folhosas;',
    'A busca foi feita na Web of Science, em outubro de 2026 (doença de plantas e aprendizado profundo '
    'com generalização em campo, vazamento, calibração e hortaliças folhosas);')])
troca('Os dados de campo são coletados', [(
    '(manchas, clorose, oídio ou míldio, perfurações, murcha)', '(manchas, clorose, míldio, furos, murcha)')])

# ---------------------------------------------------------------- rodada 4: Seções 4 e 5
substitui('As Tabelas 1 e 2 vão até o quarto ciclo', [])
troca('Quanto a Q2, o temperature scaling', [(
    'reduziu o ECE na validação de 0,0509 para 0,0027 no EfficientNet-B0 e de 0,0472 para 0,0022 no '
    'ViT-B/16.', 'reduziu o ECE de validação a 0,0027 no EfficientNet-B0 e a 0,0022 no ViT-B/16.')])
troca('Quanto ao Quadro 1, a queda', [(
    'O ViT-B/16, empatado no teste e pior na lavoura, confirma a sensibilidade do ViT puro ao ambiente '
    '(Shafay et al., 2025), e o ajuste fino completo não reduz o *domain gap* (Nguyen et al., 2026).',
    'O ViT-B/16, pior na lavoura, confirma a sensibilidade do ViT puro ao ambiente (Shafay et al., '
    '2025).')])
troca('Em Q1, EfficientNet-B0 e ViT-B/16 empatam', [(
    'empatam e superam as outras duas redes com significância estatística; como o *transformer*, com 21 '
    'vezes mais parâmetros, não supera a CNN no teste e é o pior nas culturas de lavoura, o '
    'EfficientNet-B0 é a arquitetura indicada, e o limite está nos dados, não na arquitetura.',
    'empatam e superam as outras duas redes; como o *transformer*, 21 vezes maior, não supera a CNN no '
    'teste e é o pior na lavoura, o EfficientNet-B0 é a arquitetura indicada, e o limite está nos dados.')])

# ---------------------------------------------------------------- rodada 5: linhas curtas
troca('Na agricultura, uma das aplicações mais estudadas', [(
    'segmento no qual o diagnóstico fitossanitário ainda depende predominantemente de inspeção visual '
    'manual.', 'segmento em que o diagnóstico ainda depende de inspeção visual.')])
troca('Os trabalhos relacionados mostram um contraste', [
    ('treinadas no dataset PlantVillage', 'treinadas no PlantVillage'),
    ('fenômeno denominado *domain gap*', 'o chamado *domain gap*'),
    ('uma revisão sistemática recente situa', 'uma revisão recente situa'),
    ('em partições do próprio conjunto de dados', 'em partições do próprio conjunto')])
troca('As probabilidades de redes profundas', [
    ('por um escalar *T* aprendido na validação sem alterar a classe predita',
     'por um escalar *T* aprendido na validação'),
    ('acurácia observada por faixas de confiança', 'acurácia observada por faixas'),
    ('permite rejeitar antes da inferência', 'rejeita antes da inferência')])
troca('A busca foi feita na Web of Science', [(
    'Não foi localizado estudo que reúna, para hortaliças folhosas, auditoria de vazamento, calibração '
    'com abstenção e avaliação fora da fonte de treino.',
    'Não se localizou estudo que reúna, em hortaliças folhosas, auditoria de vazamento, calibração com '
    'abstenção e avaliação fora da fonte.')])
troca('Os dados de campo são coletados', [
    ('com telefone celular', 'com celular'),
    ('proporção próxima de 50/50 entre folhas saudáveis e anômalas', 'cerca de metade de folhas anômalas')])
troca('Em Q1, EfficientNet-B0 e ViT-B/16 empatam', [(
    'a abstenção ajustada na validação não protege', 'a abstenção não protege')])

# ---------------------------------------------------------------- rodada 6: depois da Tabela 1
troca('A Tabela 1 tem uma ressalva', [(
    'subiu para 88,4% (IC 95%: 85,3% a 91,2%; p de Holm = 5,9×10⁻¹²), mas',
    'subiu para 88,4% (p de Holm = 5,9×10⁻¹²), mas')])
troca('Teste do ciclo 5: 18.413 imagens', [
    ('Lavoura: mandioca, arroz, cana e café (N = 4.530), com Espec. e Sens. no mesmo subconjunto; '
     'ciclo 4 → ciclo 5, mesmas imagens.',
     'Lavoura: mandioca, arroz, cana e café (N = 4.530), também em Espec. e Sens.; ciclo 4 → ciclo 5.'),
    ('Retido: lote de espinafre, N = 574.', 'Retido: espinafre, N = 574.')])
troca('Quanto ao Quadro 1, a queda', [(
    'as culturas de lavoura têm 86,3% (IC 95%: 84,5% a 88,0%) no EfficientNet-B0',
    'as culturas de lavoura têm 86,3% no EfficientNet-B0')])
troca('A principal contribuição é um protocolo', [
    ('cada modelo foi treinado uma única vez, sem variar a semente;', 'um treino por modelo, sem variar a semente;'),
    ('recalibrar temperatura e limiar com fotografias de campo separadas por horta,',
     'recalibrar com fotografias de campo,')])

# ---------------------------------------------------------------- rodada 7
troca('Teste do ciclo 5: 18.413 imagens', [(
    'Teste do ciclo 5: 18.413 imagens, sem variantes de alface.', 'Teste do ciclo 5: 18.413 imagens.')])
troca('Quanto a Q2, o temperature scaling', [
    ('(0,990 contra 0,874; Mann-Whitney, p < 0,001)', '(0,990 contra 0,874; p < 0,001)'),
    ('O ViT-B/16 tem o menor ECE da Tabela 1 e a menor fração de erros interceptados: boa calibração '
     'média não garante que os erros fiquem nas confianças baixas.',
     'O ViT-B/16 tem o menor ECE e a menor fração de erros interceptados: boa calibração média não '
     'põe os erros nas confianças baixas.')])
troca('A Tabela 1 tem uma ressalva', [
    ('que o modelo do terceiro ciclo nunca viu, a acurácia balanceada do EfficientNet-B0 foi de 75,4%',
     'que o modelo do terceiro ciclo nunca viu, o EfficientNet-B0 teve 75,4%'),
    ('subiu para 88,4% (p de Holm = 5,9×10⁻¹²), mas', 'subiu para 88,4%, mas')])
troca('Quanto ao Quadro 1, a queda', [(
    ' O ViT-B/16, pior na lavoura, confirma a sensibilidade do ViT puro ao ambiente (Shafay et al., '
    '2025).', '')])
troca('Este trabalho avaliou em que medida', [(
    'Este trabalho avaliou em que medida um sistema de triagem fitossanitária para hortaliças folhosas '
    'mantém desempenho confiável fora das condições de treino. Cumpridos os objetivos (a) a (c), a '
    'resposta é que', 'Cumpridos os objetivos (a) a (c), a resposta à questão de pesquisa é que')])

# ---------------------------------------------------------------- rodada 8: linhas curtas
troca('A busca foi feita na Web of Science', [
    ('Dois estudos adotam protocolo semelhante ao deste:', 'Dois estudos têm protocolo semelhante:'),
    ('calibração com abstenção e avaliação fora da fonte', 'calibração, abstenção e avaliação fora da fonte')])
troca('O dataset binário', [
    ('com vazamento residual de 0,04% no teste por esse critério.', 'com vazamento residual de 0,04% no teste.'),
    ('23% do teste são rotações, recortes e filtros de fotografias de alface',
     '23% do teste são variantes de fotografias de alface')])
troca('Um quinto ciclo corrige três defeitos', [
    ('achados em auditoria posterior:', 'achados na auditoria:'),
    ('fica fora do treino, onde permanece outro lote da espécie.', 'fica fora do treino.')])
troca('Os dados de campo são coletados', [
    ('(aproximada, média e ampla)', '(perto, média e ampla)'),
    ('(solo, mão, parede), para verificar', '(solo, mão), para testar')])
troca('Quanto a Q2, o temperature scaling', [
    ('Quanto a Q2, o', 'Em Q2, o'),
    ('não põe os erros nas confianças baixas.', 'não põe os erros em confiança baixa.')])
troca('A Tabela 1 tem uma ressalva', [(
    'Com o lote no treino, o resultado nele subiu para 88,4%', 'Com o lote no treino, subiu para 88,4%')])
troca('Quanto ao Quadro 1, a queda', [
    ('indica que a diferença está no protocolo, e não na cultura', 'indica que a diferença é de protocolo, não de cultura'),
    ('no EfficientNet-B0 do ciclo 4 e concentram', 'no ciclo 4 e concentram'),
    ('aqui, o de campo é de 0,146.', 'aqui, 0,146 em campo.')])
troca('Cumpridos os objetivos (a) a (c)', [(
    'a resposta à questão de pesquisa é que', 'a resposta até aqui é que')])

open(path, 'w', encoding='utf8').write(d)
print('ok')
