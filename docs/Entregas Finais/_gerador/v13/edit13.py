import re,sys
from xml.sax.saxutils import escape
p='v13/word/document.xml'
x=open(p,encoding='utf-8').read()
def R(t,i=False,b=False):
    rp='<w:rPr>'+('<w:b w:val="1"/><w:bCs w:val="1"/>' if b else '')+('<w:i w:val="1"/><w:iCs w:val="1"/>' if i else '')+'<w:rtl w:val="0"/></w:rPr>'
    return '<w:r>'+rp+'<w:t xml:space="preserve">'+escape(t)+'</w:t></w:r>'
def P(runs): return '<w:p><w:pPr><w:rPr/></w:pPr>'+''.join(runs)+'</w:p>'
def sub(a,b):
    global x
    a=escape(a); b=escape(b)
    assert x.count(a)==1,(a,x.count(a)); x=x.replace(a,b)
def para_bounds(key):
    k=x.find(escape(key)); assert k>0,key
    return x.rfind('<w:p ',0,k) if x.rfind('<w:p ',0,k)>x.rfind('<w:p>',0,k) else x.rfind('<w:p>',0,k), x.find('</w:p>',k)+6
def delpara(key):
    global x
    s,e=para_bounds(key); x=x[:s]+x[e:]
def setpara(key,runs):
    global x
    s,e=para_bounds(key); x=x[:s]+P(runs)+x[e:]
# --- citations of classic statistical papers / textbook
sub(' (McNemar, 1947; Dietterich, 1998)',' (Dietterich, 1998)')
sub(' de Holm (1979) nas',' de Holm nas')
sub(' (Wilson, 1927), e a',', e a')
sub(' dominam a classificação de imagens (Russell; Norvig, 2022).',' dominam a classificação de imagens.')
sub(' também pelo de Cochran-Armitage (Agresti, 2013); as diferenças de proporções são reportadas com intervalo de Newcombe (1998), e os resultados, interpretados como exploratórios.',' também pelo de Cochran-Armitage, e os resultados são interpretados como exploratórios.')
for k in ['MCNEMAR, Q.','HOLM, S.','WILSON, E. B.','RUSSELL, S.','AGRESTI, A.','NEWCOMBE, R. G.']: delpara(k)
# --- results: field paragraph (the "no measure" point moves to limitations)
sub(' não provocaram falso alarme, mas, sem folhas doentes, a sensibilidade em campo e o ',' não provocaram falso alarme. No lote de casa')
s,e=para_bounds('No lote de campo, as quatro'); seg=x[s:e]
seg2=re.sub(r'<w:r [^>]*><w:rPr><w:i w:val="1"/><w:iCs w:val="1"/><w:rtl w:val="0"/></w:rPr><w:t xml:space="preserve">domain gap</w:t></w:r>','',seg)
assert seg2!=seg
seg2=seg2.replace(' seguem sem medida. No lote de casa, todas',', todas'); assert seg2.count(', todas as diagnosticadas')==1
x=x[:s]+seg2+x[e:]
# --- limitations
setpara('Limitações do estudo. ',[R('Limitações do estudo. ',b=True),
 R('Os resultados precisam ser lidos com cinco ressalvas. (i) A evidência de campo é pequena: 23 fotografias, só de alface, e o lote com fundo real não tem folhas doentes; por isso, a sensibilidade e o '),R('domain gap',i=True),
 R(' em campo seguem sem medida. (ii) Treino e teste vêm das mesmas fontes públicas, que, como a maioria dos conjuntos da área, estão longe das condições reais de uso (Xu et al., 2024); além disso, parte das culturas entra só como '),R('proxy',i=True),
 R(' visual. (iii) A calibração foi ajustada fora do campo: do laboratório para a lavoura, o '),R('temperature scaling',i=True),
 R(' reduz o ECE, mas a confiança continua bem acima da acurácia (Xiang; Shi; Zhu, 2026), como no ECE de 0,146 obtido aqui. (iv) A tarefa é binária e não nomeia a doença, e sintomas parecidos podem ter causas diferentes (Barbedo, 2016), o que torna o próprio rótulo uma fonte de erro. (v) Cada modelo foi treinado uma única vez, e o desempenho de uma rede é uma distribuição entre sementes, não um valor único (Coakley; Gundersen, 2026); o empate entre EfficientNet-B0 e ViT-B/16 pode não se repetir com outras sementes.')])
# --- conclusion
setpara('A coleta com fundo real responde a Q3',[R('Q3 fica respondida só em parte, pelas limitações já descritas, e os objetivos (d) a (f) seguem abertos.')])
open(p,'w',encoding='utf-8').write(x); print('ok')
