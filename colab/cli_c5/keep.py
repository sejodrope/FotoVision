# Mantem uma celula em execucao (~4 min; curta para nao segurar na fila a consulta do chunkup.sh) enquanto os dados sobem: em 14/10 o runtime
# foi recolhido no meio do envio, com o kernel ocioso.
import glob, os, time
for _ in range(4):
    ps = glob.glob('/content/parts/*')
    print(f'[keep {time.strftime("%H:%M:%S")}] {len(ps)} pedacos, '
          f'{sum(os.path.getsize(p) for p in ps) / 1e9:.2f} GB', flush=True)
    time.sleep(60)
