import re,sys
from xml.sax.saxutils import escape
path=sys.argv[1]
x=open(path,encoding='utf-8').read()
def R(t,i=False,b=False):
    rp='<w:rPr>'+('<w:b w:val="1"/><w:bCs w:val="1"/>' if b else '')+('<w:i w:val="1"/><w:iCs w:val="1"/>' if i else '')+'<w:rtl w:val="0"/></w:rPr>'
    return '<w:r>'+rp+'<w:t xml:space="preserve">'+escape(t)+'</w:t></w:r>'
def P(runs): return '<w:p><w:pPr><w:rPr/></w:pPr>'+''.join(runs)+'</w:p>'
# 1. limitations paragraph after last paragraph of section 4
lim=P([R('Limitações do estudo. ',b=True),
 R('Os resultados têm cinco limitações. (i) A evidência de campo é pequena e incompleta: 23 fotografias, só de alface, e o lote com fundo real não tem folhas doentes, de modo que a sensibilidade e o '),R('domain gap',i=True),
 R(' em campo não foram medidos, e os intervalos da Tabela 2 não separam versões. (ii) Treino e teste vêm das mesmas seis fontes públicas, que, como a maioria dos conjuntos da área, refletem pouco as condições reais de uso (Xu et al., 2024); a deduplicação não alcança variantes muito alteradas, parte das culturas é só '),R('proxy',i=True),
 R(' visual e o espinafre é '),R('Basella alba',i=True),R(', não '),R('Spinacia oleracea',i=True),
 R('. (iii) A calibração foi ajustada na validação intra-fonte; na passagem do laboratório para o campo, o '),R('temperature scaling',i=True),
 R(' reduz o ECE, mas a confiança continua bem acima da acurácia (Xiang; Shi; Zhu, 2026), o que é coerente com o ECE de 0,146 observado aqui. (iv) A tarefa é binária e a rotulagem, por diretório: o sistema não nomeia a doença, e sintomas semelhantes podem ter causas distintas ou coexistir na mesma folha (Barbedo, 2016), o que faz do rótulo uma fonte de ruído, como mostrou o defeito corrigido no terceiro ciclo. (v) Cada modelo foi treinado uma vez, com semente fixa, mas o desempenho de redes como ResNet e ViT é uma distribuição entre sementes, não um valor único (Coakley; Gundersen, 2026); o teste de McNemar compara os modelos treinados, não as arquiteturas em geral, e o empate entre EfficientNet-B0 e ViT-B/16 pode mudar com outras sementes.')])
k=x.find('No lote de campo, as quatro'); e=x.find('</w:p>',k)+6
x=x[:e]+lim+x[e:]
# 2. conclusion: drop the old "Há limites" sentence
old_start=' Há limites: um treino por modelo'
s=x.find(old_start); assert s>0
endtxt='todas de plantas sadias.'
t=x.find(endtxt,s)+len(endtxt)
seg=x[s:t]
# keep any run boundaries: remove text but keep xml tags balanced -> rebuild: cut text from start to end across runs
tags=re.findall(r'<[^>]+>',seg)
x=x[:s]+''.join(tags)+x[t:]
# remove now-empty italic 'proxy' run leftovers (empty w:t) – harmless, but clean
x=re.sub(r'<w:r [^>]*><w:rPr>(?:(?!</w:rPr>).)*</w:rPr><w:t xml:space="preserve"></w:t></w:r>','',x)
# 3. references
def ref(before_key, author_title, journal, rest, after=False):
    global x
    k=x.find(before_key); ps=x.rfind('<w:p ',0,k); pe=x.find('</w:p>',k)+6
    tmpl=x[ps:pe]
    runs=re.findall(r'<w:r [^>]*>.*?</w:r>',tmpl,flags=re.S)
    assert len(runs)==3, len(runs)
    new=tmpl
    for r,txt in zip(runs,[author_title,journal,rest]):
        new=new.replace(r, re.sub(r'(<w:t[^>]*>)[^<]*(</w:t>)', lambda m:m.group(1)+escape(txt)+m.group(2), r),1)
    new=new.replace('w14:paraId="','w14:paraId="7')  # avoid duplicate ids
    new=re.sub(r'w14:paraId="7([0-9A-F]{8})"',lambda m:'w14:paraId="7'+m.group(1)[1:]+'"',new)
    x = x[:pe]+new+x[pe:] if after else x[:ps]+new+x[ps:]
ref('BARBEDO, J. G. A. Impact','BARBEDO, J. G. A. A review on the main challenges in automatic plant disease identification based on visible range images. ','Biosystems Engineering',', v. 144, p. 52-60, 2016. DOI: 10.1016/j.biosystemseng.2016.01.017.')
ref('DIETTERICH, T. G.','COAKLEY, K. L.; GUNDERSEN, O. E. Measuring deep learning performance: an empirical study of performance distributions across architectures and tasks. ','Scientific Reports',', v. 16, 20474, 2026. DOI: 10.1038/s41598-026-49656-z.')
ref('WORKNEH, A. D.','XU, M. et al. Plant disease recognition datasets in the age of deep learning: challenges and opportunities. ','Frontiers in Plant Science',', v. 15, 1452551, 2024. DOI: 10.3389/fpls.2024.1452551.',after=True)
ref('WORKNEH, A. D.','XIANG, K.; SHI, D.; ZHU, X. Quantifying the reliability gap in cross-domain plant disease classification: benchmarking the limited efficacy of standard mitigation techniques under controlled-to-field shift. ','Frontiers in Plant Science',', v. 17, 1826962, 2026. DOI: 10.3389/fpls.2026.1826962.',after=True)
open(path,'w',encoding='utf-8').write(x)
print('ok')
