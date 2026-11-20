# Gera a página de estudo (HTML único) a partir de roteiro_data.py e do GUIA_BANCA_TCC2.md.
# Uso: python gera_pagina_estudo.py <GUIA_BANCA_TCC2.md> <saida.html>
import json, re, sys, html
from roteiro_data import SLIDES

guia = open(sys.argv[1], encoding='utf-8').read()


def md(t):
    t = html.escape(t.strip())
    t = re.sub(r'\*\*\[(ATUALIZAR|RESPONDER)[^\]]*\]\*\*', '', t)
    t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'\*(.+?)\*', r'<i>\1</i>', t)
    t = re.sub(r'`(.+?)`', r'<code>\1</code>', t)
    return re.sub(r'\s+', ' ', t).strip()


# cartão de números
a = guia.index('### Para saber de cor')
b = guia.index('### Para consultar')
numeros = []
for ln in guia[a:b].splitlines():
    m = re.match(r'\|\s*\*\*(.+?)\*\*\s*\|\s*(.+?)\s*\|$', ln)
    if m:
        numeros.append({'n': md(m.group(1)), 'o': md(m.group(2))})

# perguntas
a = guia.index('## 11. Perguntas prováveis da banca')
b = guia.index('### Se não souber responder')
perguntas, cat = [], ''
blocos = re.split(r'\n(?=### |\*\*\d+\. )', guia[a:b])
for bl in blocos:
    bl = bl.strip()
    if bl.startswith('### '):
        cat = bl.splitlines()[0][4:].strip()
        resto = '\n'.join(bl.splitlines()[1:]).strip()
        if not resto.startswith('**'):
            continue
        bl = resto
    m = re.match(r'\*\*(\d+)\. ["“](.+?)["”]\*\*(.*)', bl, re.S)
    if not m or '[RESPONDER]' in bl:
        continue
    perguntas.append({'k': int(m.group(1)), 'c': cat.replace('Sobre ', '').capitalize(), 'p': md(m.group(2)),
                      'r': md(m.group(3))})

limit = [
    ('Campo pequeno', '23 fotos, só de alface; o lote com fundo real não tem folhas doentes. Sensibilidade e domain gap em campo sem medida.', 'Os próprios dados'),
    ('Mesmas fontes públicas', 'Treino e teste da mesma origem; os conjuntos públicos estão longe do uso real; parte das culturas é só proxy visual.', 'Xu et al. (2024)'),
    ('Calibração fora do campo', 'Do laboratório para a lavoura, a confiança segue bem acima da acurácia. Aqui: ECE de 0,003 para 0,146.', 'Xiang; Shi; Zhu (2026)'),
    ('Tarefa binária', 'Não nomeia a doença; sintomas parecidos podem ter causas diferentes.', 'Barbedo (2016)'),
    ('Uma semente por modelo', 'O desempenho varia entre sementes; o empate EfficientNet-B0 × ViT pode não se repetir.', 'Coakley; Gundersen (2026)'),
]
dados = {'slides': [{k: s[k] for k in ('titulo', 'secao', 'tempo', 'pagina', 'mensagem', 'fala', 'transicao', 'se_perguntarem')}
                    for s in SLIDES],
         'numeros': numeros, 'perguntas': perguntas,
         'limit': [{'t': a, 'd': b, 'f': c} for a, b, c in limit]}
modelo = open(__file__.replace('gera_pagina_estudo.py', 'pagina_estudo_modelo.html'), encoding='utf-8').read()
open(sys.argv[2], 'w', encoding='utf-8').write(modelo.replace('/*DADOS*/null', json.dumps(dados, ensure_ascii=False)))
print('ok', len(numeros), 'numeros', len(perguntas), 'perguntas')
