# Empacota pesos, resultados e logs (sem o checkpoint de retomada) no mesmo layout
# da ultima celula do notebook, e corta em pedacos de 32 MB para o download.
import hashlib, os, shutil, zipfile

OUT, DEST, PARTS = '/content/out', '/content/vit_resultados.zip', '/content/rparts'
log = open(f'{OUT}/execucao.log', encoding='utf-8', errors='replace').read()
print('STATUS', 'COMPLETO' if 'ViT COMPLETO' in log else 'INCOMPLETO', flush=True)
with zipfile.ZipFile(DEST, 'w', zipfile.ZIP_STORED) as z:
    for sub in ('weights', 'results', 'logs'):
        for raiz, _, nomes in os.walk(f'{OUT}/{sub}'):
            for n in nomes:
                if n.endswith('_last.pt'):
                    continue
                caminho = os.path.join(raiz, n)
                z.write(caminho, os.path.join(sub, 'vit', os.path.relpath(caminho, f'{OUT}/{sub}')))
    z.write(f'{OUT}/execucao.log', 'logs/vit/execucao.log')
shutil.rmtree(PARTS, ignore_errors=True); os.makedirs(PARTS)
h, i, linhas = hashlib.sha256(), 0, []
with open(DEST, 'rb') as f:
    while chunk := f.read(32 << 20):
        h.update(chunk)
        open(f'{PARTS}/p{i:04d}', 'wb').write(chunk)
        linhas.append(f'p{i:04d} {len(chunk)}'); i += 1
open(f'{PARTS}/manifest.txt', 'w').write(h.hexdigest() + '\n' + '\n'.join(linhas) + '\n')
print(f'PACOTE {os.path.getsize(DEST) / 1e6:.0f} MB em {i} pedacos', flush=True)
