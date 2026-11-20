# -*- coding: utf-8 -*-
"""Quadro de atendimento às observações do orientador (artigo v14). Uso: python gera_atendimento.py <saida.docx>"""
import sys
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OBS = [
 # (observação do orientador, como foi atendida, onde no v14)
 ('Formatação no Word: parágrafo com 0 pt antes, 0 pt depois e espaçamento simples.',
  'Todo o texto em espaçamento simples, com 0 pt antes e depois. Títulos de seção mantêm 6 pt antes, para separar as seções.', 'Todo o artigo'),
 ('Resumo deve ser apresentado com espaçamento simples.', 'Resumo e abstract em espaçamento simples, em bloco único.', 'p. 1'),
 ('Incluir palavras-chave.', 'Incluídas três palavras-chave, separadas por ponto (Guia, Apêndice C), e as keywords correspondentes.', 'p. 1'),
 ('Artigo é relato de pesquisa: redigir no tempo passado.',
  'Resumo, abstract, procedimentos, resultados e conclusão no passado. Conceitos da literatura permanecem no presente.', 'Todo o artigo'),
 ('Incluir objetivo.', 'O resumo traz o objetivo de forma explícita.', 'p. 1'),
 ('Objetivo não está claro.',
  'Objetivo reformulado: avaliar em que medida o sistema de triagem mantém desempenho confiável em fotografias de campo, em correspondência direta com a questão de pesquisa.', 'p. 1 e 2'),
 ('Uniformizar o objetivo em todo o trabalho.', 'A mesma formulação do objetivo é usada em todas as ocorrências.', 'p. 1, 2, 4 e 8'),
 ('O objetivo deve estar no resumo, na introdução, nos procedimentos metodológicos e na conclusão.',
  'Resumo; Seção 1, último parágrafo; Seção 3, primeiro parágrafo; Seção 5, primeiro parágrafo.', 'p. 1, 2, 4 e 8'),
 ('Incluir no resumo a metodologia, os resultados e as conclusões.',
  'O resumo apresenta contexto, objetivo, método (quatro arquiteturas, teste de McNemar, três situações de avaliação), resultados e conclusão.', 'p. 1'),
 ('Se há mais três questões, serão quatro TCCs.',
  'Retiradas as perguntas Q1 a Q3. Permanece uma única questão de pesquisa; as três situações de avaliação são descritas como procedimento.', 'p. 2 e 4'),
 ('No TCC II não há objetivo geral nem específicos; são tópicos do TCC I.', 'Retirado o parágrafo de objetivo geral e objetivos específicos.', 'p. 2'),
 ('Remover espaços entre parágrafos.', 'Atendido no texto inteiro.', 'Todo o artigo'),
 ('Incluir recuo nos parágrafos.', 'Recuo de 1,25 cm na primeira linha (Guia, 3.4).', 'Todo o artigo'),
 ('O trabalho deve ser redigido em espaçamento simples.', 'Atendido no texto inteiro.', 'Todo o artigo'),
 ('Incluir texto introdutório na Revisão da literatura.', 'Incluído parágrafo que apresenta as subseções 2.1 a 2.3.', 'p. 2'),
 ('Adequar as citações à ABNT, conforme o Guia.',
  'Sistema autor-data: sobrenomes em maiúsculas dentro dos parênteses e em minúsculas fora deles; "et al." a partir de quatro autores; ponto e vírgula entre autores e entre obras, em ordem alfabética (Guia, 3.11).', 'Todo o artigo'),
 ('Figuras, quadros e tabelas não são autoexplicativos: todos precisam ser comentados.',
  'Cada ilustração é anunciada antes e comentada depois: Quadro 1 e Tabelas 1, 2 e 3.', 'p. 3, 6 e 7'),
 ('Quadros e tabelas identificados acima e com a fonte abaixo.',
  'Título acima no formato "Tabela n – título"; nota e fonte abaixo. Todas as abreviaturas das tabelas são definidas nas notas.', 'p. 3, 6 e 7'),
 ('Figuras, quadros e tabelas centralizados, com descrição e fonte alinhadas à esquerda.',
  'Tabelas centralizadas; título, nota e fonte à esquerda. Tabelas abertas nas laterais e quadro fechado (Guia, 2.4.2).', 'p. 3, 6 e 7'),
 ('Conclusão: resgatar o contexto da introdução.', 'O primeiro parágrafo retoma o contexto e o problema.', 'p. 8'),
 ('Conclusão: resgatar a questão de pesquisa e o objetivo.', 'O primeiro parágrafo retoma o objetivo; o terceiro responde à questão de pesquisa.', 'p. 8'),
 ('Conclusão: apresentar o que se pôde concluir com os resultados.', 'O segundo e o terceiro parágrafos apresentam o que se concluiu e a contribuição.', 'p. 8'),
 ('Conclusão: sugerir trabalhos futuros com base nas questões levantadas.',
  'O quarto parágrafo traz cinco trabalhos futuros, ligados às limitações do estudo.', 'p. 8'),
 ('Cuidar com afirmações categóricas.',
  'Frases reformuladas com o devido alcance ("os resultados indicam", "mostrou-se a opção mais adequada neste estudo", "parece"). Retirada a afirmação "o limite está nos dados".', 'Seções 4 e 5'),
 ('Adequar as referências à ABNT, conforme o Guia.',
  'Ordem alfabética; alinhadas à esquerda; espaçamento simples com uma linha em branco entre elas; título do periódico ou do livro em negrito; eventos com número, ano, local, editora e páginas; volume, fascículo e número do artigo conferidos no Crossref; DOI.', 'p. 8 a 10'),
]

EXTRAS = [
 'Siglas escritas por extenso na primeira ocorrência (TCC, CNN, ViT, ECE, IC, ROC, GPU) e símbolos das equações definidos (Guia, 2.4.3 e 2.4.4).',
 'Termos estrangeiros em itálico, inclusive nos títulos das subseções 2.1 e 2.2 (Guia, 3.8).',
 'Retirado o parágrafo que previa análise com Grad-CAM, que não foi executada; a referência correspondente saiu da lista.',
 'Ajustes de precisão: métrica nomeada em cada célula do Quadro 1; falsos alarmes do lote fotografado em casa explicitados; ECE de campo indicado como valor indicativo (N = 13).',
 'Mantido o limite de 10 páginas do modelo de artigo de 2026.',
]

DECIDIR = [
 'Numeração das seções: o modelo de artigo do curso usa "1. Introdução" (com ponto); o Guia (3.6, d) não usa sinal entre o número e o título. Mantive o modelo do curso.',
 'Ordem e fonte do modelo do curso: Abstract antes do Resumo e legendas em Arial 10, como no modelo de 2026; o Guia sugere Resumo antes e legendas em tamanho 11.',
 'Referências de artigos on-line: usei o DOI. Se preferir "Disponível em: ... Acesso em: ..." em todas, o texto passa de 10 páginas e será preciso cortar conteúdo.',
 'Workneh et al.: publicado on-line em 2026, no volume 256 da revista, datado de janeiro de 2027. Citei como 2026.',
]


def borda(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    b = OxmlElement('w:tcBorders')
    for side in ('top', 'left', 'bottom', 'right'):
        e = OxmlElement(f'w:{side}')
        e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), '4'); e.set(qn('w:color'), '000000')
        b.append(e)
    tcPr.append(b)


def par(doc, texto, bold=False, size=11, after=4, align=None):
    p = doc.add_paragraph()
    r = p.add_run(texto); r.bold = bold; r.font.size = Pt(size)
    p.paragraph_format.space_after = Pt(after); p.paragraph_format.space_before = Pt(0)
    if align: p.alignment = align
    return p


def main(saida):
    doc = Document()
    st = doc.styles['Normal']; st.font.name = 'Times New Roman'; st.font.size = Pt(11)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    for s in doc.sections:
        s.top_margin = Cm(2.5); s.bottom_margin = Cm(2); s.left_margin = Cm(2.5); s.right_margin = Cm(2)
    par(doc, 'Atendimento às observações do orientador', bold=True, size=13, after=2, align=WD_ALIGN_PARAGRAPH.CENTER)
    par(doc, 'TCC II · Detecção de anomalias fitossanitárias em hortaliças utilizando visão computacional e aprendizado de máquina',
        size=11, after=2, align=WD_ALIGN_PARAGRAPH.CENTER)
    par(doc, 'José Pedro Vieira Silva · Orientador: Prof. Leanderson André · Artigo v14, 19/11/2026',
        size=11, after=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    par(doc, 'As observações da entrega 1 foram aplicadas com base no Guia para Apresentação de Trabalhos Acadêmicos da '
             'Univille (2019). O quadro indica como cada observação foi atendida e onde conferir no artigo v14. '
             'As páginas valem para a versão em PDF enviada junto.', after=8)
    t = doc.add_table(rows=1, cols=4); t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
    larg = [Cm(0.9), Cm(5.4), Cm(7.9), Cm(2.3)]
    for col, w in zip(t.columns, larg): col.width = w
    for c, h, w in zip(t.rows[0].cells, ('Nº', 'Observação', 'Como foi atendida', 'Onde'), larg):
        c.width = w; c.paragraphs[0].add_run(h).bold = True; borda(c)
        c.paragraphs[0].paragraph_format.space_after = Pt(1)
    for i, (o, a, onde) in enumerate(OBS, 1):
        row = t.add_row().cells
        for c, txt, w in zip(row, (str(i), o, a, onde), larg):
            c.width = w; c.paragraphs[0].add_run(txt).font.size = Pt(10); borda(c)
            pf = c.paragraphs[0].paragraph_format; pf.space_after = Pt(1); pf.space_before = Pt(1); pf.line_spacing = 1.0
    par(doc, '', after=4)
    par(doc, 'Ajustes complementares feitos na revisão', bold=True, after=4)
    for e in EXTRAS:
        p = doc.add_paragraph(e, style='List Bullet'); p.paragraph_format.space_after = Pt(2)
    par(doc, '', after=4)
    par(doc, 'Pontos para a decisão do orientador', bold=True, after=4)
    for e in DECIDIR:
        p = doc.add_paragraph(e, style='List Bullet'); p.paragraph_format.space_after = Pt(2)
    doc.save(saida)
    print('ok', saida, len(OBS), 'observações')


if __name__ == '__main__':
    main(sys.argv[1])
