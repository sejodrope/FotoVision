# -*- coding: utf-8 -*-
"""Confere a ordem dos elementos de propriedade no document.xml (o Word recusa ou "repara" o
arquivo quando a ordem do schema OOXML não é seguida). Uso: python valida_ooxml.py <arquivo.docx>"""
import sys, zipfile
from lxml import etree

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
ORDEM = {
 'pPr': 'pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd '
        'tabs suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi '
        'adjustRightInd snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc textDirection '
        'textAlignment textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange',
 'rPr': 'rStyle rFonts b bCs i iCs caps smallCaps strike dstrike outline shadow emboss imprint noProof '
        'snapToGrid vanish webHidden color spacing w kern position sz szCs highlight u effect bdr shd fitText '
        'vertAlign rtl cs em lang eastAsianLayout specVanish oMath',
 'tblPr': 'tblStyle tblpPr tblOverlap bidiVisual tblStyleRowBandSize tblStyleColBandSize tblW jc '
          'tblCellSpacing tblInd tblBorders shd tblLayout tblCellMar tblLook',
 'tcPr': 'cnfStyle tcW gridSpan hMerge vMerge tcBorders shd noWrap tcMar textDirection tcFitText vAlign hideMark',
 'tblBorders': 'top left start bottom right end insideH insideV',
}
ORDEM = {k: v.split() for k, v in ORDEM.items()}


def valida(path):
    root = etree.fromstring(zipfile.ZipFile(path).read('word/document.xml'))
    erros = 0
    for nome, seq in ORDEM.items():
        for el in root.iter(W + nome):
            pos = [seq.index(etree.QName(c).localname) for c in el if etree.QName(c).localname in seq]
            desconhecidos = [etree.QName(c).localname for c in el if etree.QName(c).localname not in seq]
            if pos != sorted(pos) or len(pos) != len(set(pos)) or desconhecidos:
                erros += 1
                if erros <= 10:
                    print('fora de ordem:', nome, [etree.QName(c).localname for c in el])
    print('OK: ordem OOXML válida' if not erros else f'{erros} elemento(s) com problema')
    return erros


if __name__ == '__main__':
    sys.exit(1 if valida(sys.argv[1]) else 0)
