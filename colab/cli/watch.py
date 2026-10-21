# Fica na celula ate o run_vit.py terminar: mantem o kernel ocupado (a VM nao e
# recolhida por inatividade) e ecoa as linhas novas do log, sem as barras de progresso.
import os, subprocess, time

LOG = '/content/out/execucao.log'
pos, t_last = 0, time.time()
while True:
    alive = subprocess.run('pgrep -f "[r]un_vit.py"', shell=True, capture_output=True).returncode == 0
    if os.path.exists(LOG):
        with open(LOG, 'rb') as f:
            f.seek(pos); chunk = f.read(); pos = f.tell()
        for line in chunk.decode('utf-8', 'replace').replace('\r', '\n').split('\n'):
            s = line.strip()
            if s and 'it/s]' not in s and 's/it]' not in s and '%|' not in s:
                print(s, flush=True); t_last = time.time()
    if not alive:
        print('== run_vit.py terminou ==', flush=True); break
    if time.time() - t_last > 240:
        gpu = subprocess.run('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader',
                             shell=True, capture_output=True, text=True).stdout.strip()
        print(f'[{time.strftime("%H:%M:%S")}] rodando | GPU {gpu}', flush=True); t_last = time.time()
    time.sleep(20)
