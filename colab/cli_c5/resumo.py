# Linhas importantes do log de treino (sem barras de progresso) e os pesos ja prontos.
import os, re
txt = open('/content/out/execucao.log', encoding='utf-8', errors='replace').read().replace('\r', '\n')
chave = re.compile(r'Epoch +\d+ \||Early stopping|Melhor F1|accuracy=|[0-9]/5 |Treino |Calibra|T =|ECE|Traceback|Error|COMPLETO')
linhas = [l.strip() for l in txt.split('\n') if chave.search(l) and '%|' not in l]
for l in linhas[-40:]:
    print(l)
# o fim de cada modelo sempre, mesmo fora da janela acima
print('RESUMO:', ' || '.join(l for l in linhas if l.startswith(('Melhor F1', '3/5', 'ECE depois'))))
w = '/content/fitovision/backend/weights/ciclo5'
print('pesos:', sorted(os.listdir(w)) if os.path.isdir(w) else '-')
