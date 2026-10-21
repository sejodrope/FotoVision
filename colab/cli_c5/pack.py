# Empacota pesos, resultados e logs do ciclo 5 (sem os checkpoints de retomada) e corta
# em pedacos de 32 MB para o download. Pode rodar no meio do treino.
import hashlib, os, shutil, zipfile

WORK, OUT = '/content/fitovision/backend', '/content/out'
DEST, PARTS = '/content/c5_resultados.zip', '/content/rparts'
log = open(f'{OUT}/execucao.log', encoding='utf-8', errors='replace').read() if os.path.exists(f'{OUT}/execucao.log') else ''
print('STATUS', 'COMPLETO' if 'CICLO 5 COMPLETO' in log else 'INCOMPLETO', flush=True)
with zipfile.ZipFile(DEST, 'w', zipfile.ZIP_STORED) as z:
    for sub in ('weights/ciclo5', 'results/ciclo5', 'logs/ciclo5'):
        for raiz, _, nomes in os.walk(f'{WORK}/{sub}'):
            for n in nomes:
                if n.endswith('_last.pt'):
                    continue
                caminho = os.path.join(raiz, n)
                z.write(caminho, os.path.relpath(caminho, WORK))
    if log:
        z.write(f'{OUT}/execucao.log', 'logs/ciclo5/execucao_colab.log')
    print('conteudo:', sorted({os.path.dirname(n) for n in z.namelist()}), flush=True)
    print('pesos:', [n for n in z.namelist() if n.endswith('.pth')], flush=True)
shutil.rmtree(PARTS, ignore_errors=True); os.makedirs(PARTS)
h, i, linhas = hashlib.sha256(), 0, []
with open(DEST, 'rb') as f:
    while chunk := f.read(32 << 20):
        h.update(chunk)
        open(f'{PARTS}/p{i:04d}', 'wb').write(chunk)
        linhas.append(f'p{i:04d} {len(chunk)}'); i += 1
open(f'{PARTS}/manifest.txt', 'w').write(h.hexdigest() + '\n' + '\n'.join(linhas) + '\n')
print(f'PACOTE {os.path.getsize(DEST) / 1e6:.0f} MB em {i} pedacos', flush=True)
