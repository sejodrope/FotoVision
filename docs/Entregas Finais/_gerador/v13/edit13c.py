exec(open('edit13.py',encoding='utf-8').read().split("# --- citations")[0])
sub('Os resultados precisam ser lidos com cinco ressalvas. (i)','Há cinco ressalvas. (i)')
sub(' (Barbedo, 2016), o que torna o próprio rótulo uma fonte de erro. (v)',' (Barbedo, 2016). (v)')
sub(' uma única vez, e o desempenho de uma rede é uma distribuição entre sementes, não um valor único (Coakley; Gundersen, 2026); o empate entre EfficientNet-B0 e ViT-B/16 pode não se repetir com outras sementes.',' uma única vez, e o desempenho de uma rede varia entre sementes (Coakley; Gundersen, 2026): o empate entre EfficientNet-B0 e ViT-B/16 pode não se repetir.')
# merge Q3 sentence into the Q1/Q2 paragraph
delpara('Q3 fica respondida só em parte')
sub(': a abstenção não protege o produtor onde mais importa.',': a abstenção não protege o produtor onde mais importa. Q3 fica respondida só em parte, e os objetivos (d) a (f) seguem abertos.')
open(p,'w',encoding='utf-8').write(x); print('ok')
