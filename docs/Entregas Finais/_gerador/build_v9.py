# Monta a v9 do artigo a partir do document.xml da v8 (entrega "Postagem IV - parcial II -
# 5. Conclusão", 27/10/2026): reescreve a Seção 5 como conclusão completa (resposta à questão
# de pesquisa, Q1 a Q3, contribuições, limitações e trabalhos futuros) e enxuga outras seções
# para manter as 11 páginas. A coleta de campo ainda não foi feita; a conclusão diz isso.
# Uso: python build_v9.py <dir v8 descompactado> <dir de saida descompactado>
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


# ---------------------------------------------------------------- 5. Conclusão
CONCLUSAO = [
    'Este trabalho avaliou em que medida um sistema de triagem fitossanitária para hortaliças '
    'folhosas mantém desempenho confiável fora das condições de treino. Cumpridos os objetivos '
    '(a) a (c), a resposta até aqui é que o desempenho medido depende mais do protocolo de '
    'avaliação que da arquitetura: 98,48% de acurácia balanceada nas fontes de treino, 75,4% em '
    'uma fonte nunca vista e 84,6% de acertos em 13 fotografias de campo (IC 95%: 57,8% a 95,7%).',

    'Em Q1, EfficientNet-B0 e ViT-B/16 empatam e superam as outras duas redes com significância '
    'estatística; como o *transformer*, com 21 vezes mais parâmetros, não supera a CNN no teste '
    'e é o pior nas culturas de lavoura, o EfficientNet-B0 é a arquitetura indicada, e o limite '
    'está nos dados, não na arquitetura. Em Q2, as probabilidades são bem '
    'calibradas na distribuição de treino, mas o limiar de abstenção deixa passar mais de dois '
    'terços dos erros, e em campo os erros vieram com confiança acima de 0,94: a abstenção '
    'ajustada na validação não protege o produtor onde mais importa.',

    'Q3 e os objetivos (d) a (f) dependem da coleta de campo com fundo real, ainda em andamento, '
    'mas o sentido da resposta já aparece: o *domain gap* surge mesmo entre fontes de folha '
    'isolada e se concentra nas culturas de lavoura, e corrigir atalhos nos dados (quinto '
    'ciclo) reduz os falsos alarmes ao custo de sensibilidade.',

    'A principal contribuição é um protocolo de avaliação reprodutível (teste intra-fonte, '
    'fonte retida e campo com rótulo prévio à inferência, e versões comparadas por testes '
    'pareados), que expôs o vazamento dos 99,01% do TCC-I e os limites da calibração fora do '
    'treino.',

    'As conclusões têm limites: cada modelo foi treinado uma única vez, sem variar a semente; a '
    'tarefa é binária e não identifica a doença; parte das culturas entra apenas como *proxy* '
    'visual; e a evidência de campo se restringe a 13 fotografias de alface. Como trabalhos '
    'futuros, além da coleta, propõem-se recalibrar temperatura e limiar com fotografias de '
    'campo separadas por horta, testar extratores congelados de modelos de fundação (Nguyen et '
    'al., 2026), repetir o treino com várias sementes e definir, com produtores, o custo de cada '
    'tipo de erro, que decide entre os perfis dos dois ciclos.',
]
substitui('No domínio de teste, EfficientNet-B0 e ViT-B/16 empatam', CONCLUSAO)
substitui('A próxima etapa é a coleta de campo', [])

# ---------------------------------------------------------------- cortes para caber em 11 páginas
# Quase todos tiram a última linha curta de um parágrafo sem tirar conteúdo; a frase sobre
# o treino único (Seção 4) e a da diversidade de captura passaram para a conclusão.
troca('A inteligência artificial (IA)', [(
    '(Russell; Norvig, 2022). Redes neurais sustentam hoje aplicações em diagnóstico médico por '
    'imagem, inspeção industrial e agricultura de precisão, e o',
    '(Russell; Norvig, 2022), e o')])
troca('As CNNs extraem hierarquias', [(
    'mas ampliar o volume de dados tem retorno decrescente se a diversidade das condições de '
    'captura não aumenta',
    'mas mais dados rendem pouco se a diversidade das condições de captura não aumenta')])
troca('O domain gap é a discrepância', [
    ('em condições controladas e em condições de uso', 'em condições controladas e de uso'),
    ('treinados em imagens de estúdio', 'treinados em estúdio'),
    ('com solo, sombra, sobreposição de folhas e iluminação variável',
     'com solo, sombra e iluminação variável'),
    ('de espécies-*proxy* para espécies-alvo, pois o PlantVillage não contém as hortaliças-alvo.',
     'de espécies-*proxy* para as hortaliças-alvo, ausentes do PlantVillage.')])
troca('A pesquisa é quantitativa', [(
    'e registro dos parâmetros e métricas de cada execução.', 'e registro de cada execução.')])
troca('O dataset binário', [(
    'O critério não alcança variantes muito alteradas da mesma fotografia:',
    'O critério não alcança variantes muito alteradas:')])
troca('As três CNNs são inicializadas', [(
    'hiperparâmetros de otimização usuais', 'hiperparâmetros usuais')])
troca('Para Q2, a calibração é avaliada', [
    ('faixas de confiança de igual largura', 'faixas de igual largura'),
    ('cuja confiança cai na faixa', 'com confiança na faixa'),
    ('acompanhado do diagrama de fiabilidade', 'com diagrama de fiabilidade')])
troca('O mecanismo de abstenção é avaliado', [
    ('(fração de imagens que recebem diagnóstico)', '(fração de imagens diagnosticadas)'),
    ('(acurácia entre as diagnosticadas)', '(acurácia entre elas)'),
    ('(confiança inferior a τ)', '(confiança < τ)')])
troca('Para Q3, cada fotografia de campo', [
    ('As abstenções são reportadas como taxa própria', 'As abstenções têm taxa própria'),
    ('e as métricas de acerto, calculadas sobre', 'e as métricas de acerto, sobre')])

troca('Como as amostras de campo são pequenas', [
    ('leva intervalo de confiança de Wilson', 'leva intervalo de Wilson'),
    ('mais de 40 pontos de amplitude com 13', 'mais de 40 pontos com 13')])
troca('EfficientNet-B0 e ViT-B/16 empatam:', [
    ('empatam: mesma acurácia balanceada e mesmo IC 95% (98,32% a 98,64%), sem diferença',
     'empatam, com o mesmo IC 95% (98,32% a 98,64%) e sem diferença'),
    ('com 21 vezes mais parâmetros que o EfficientNet-B0 e latência',
     'com 21 vezes mais parâmetros e latência'),
    (' Isso responde a Q1 neste teste; com um único treino por modelo, a conclusão não se '
     'estende às arquiteturas em geral.', '')])
troca('A Tabela 1 deve ser lida com uma ressalva', [
    ('A Tabela 1 deve ser lida com uma ressalva: o teste é extraído das mesmas seis fontes do '
     'treino, de modo que os 98,48%',
     'A Tabela 1 tem uma ressalva: o teste vem das mesmas seis fontes do treino, e os 98,48%'),
    ('Incorporado o lote ao treino,', 'Com o lote no treino,'),
    ('mas o ganho é intra-fonte: o modelo aprende a aparência das fontes de treino, e isso não '
     'informa sobre o campo.',
     'mas o ganho é intra-fonte e não informa sobre o campo.')])
troca('A Tabela 2 mostra por que', [
    ('central. Em pontos estimados, o desempenho de campo teria piorado nos três primeiros '
     'ciclos, mesmo com a correção de defeitos reais, como 3.972 imagens de alface saudável '
     'rotuladas como doentes; mas os intervalos de Wilson de todos os ciclos se sobrepõem, e com '
     'N = 13 versões e arquiteturas são indistinguíveis. A amostra revelou o defeito de '
     'rotulagem, mas',
     'central: os intervalos de Wilson se sobrepõem, e com N = 13 versões e arquiteturas são '
     'indistinguíveis. A amostra revelou um defeito real, 3.972 imagens de alface saudável '
     'rotuladas como doentes, mas'),
    ('não tem poder para decidir entre modelos. Os nove erros emitidos nos quatro primeiros '
     'ciclos, concentrados em quatro fotografias,',
     'não decide entre modelos. Os nove erros dos quatro primeiros ciclos, em quatro '
     'fotografias,'),
    ('que compartilha a distribuição do treino', 'da mesma distribuição do treino')])
troca('Na Tabela 3, a correção agiu', [(
    ' A validação, que põe EfficientNet-B0 e ViT-B/16 à frente (F1 macro de 0,972 contra 0,96), '
    'não ordena os modelos como os dados de fora.',
    '')])
troca('Esses resultados são coerentes com o Quadro 1', [
    ('tem a ordem de grandeza dos', 'é da ordem dos'),
    ('Esses resultados são coerentes', 'Os resultados são coerentes'),
    ('e, como no estudo com amêndoa, só aparece', 'e, como na amêndoa, só aparece'),
    ('O ViT-B/16, que empata no teste e é o pior nas culturas de lavoura, confirma',
     'O ViT-B/16, empatado no teste e pior na lavoura, confirma'),
    ('A literatura atribui a perda à pouca diversidade das condições de captura',
     'A perda vem da pouca diversidade de captura'),
    ('no EfficientNet-B0 do quarto ciclo', 'no EfficientNet-B0 do ciclo 4'),
    ('as culturas fotografadas em lavoura têm', 'as culturas de lavoura têm'),
    ('enquanto aqui o ECE de campo fica em 0,146 após a calibração na validação, o que recomenda '
     'reservar fotografias de campo para recalibração.',
     'enquanto aqui o ECE de campo é de 0,146, o que recomenda recalibrar em campo.')])

open(path, 'w', encoding='utf8').write(d)
print('ok')
