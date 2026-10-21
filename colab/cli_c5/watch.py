# Fica na celula ate o run_ciclo5.py terminar (ou ate ~28 min): mantem o kernel ocupado e
# ecoa as linhas novas do log, sem as barras de progresso.
import os, subprocess, time

LOG = '/content/out/execucao.log'
POS = '/content/out/.watch_pos'
pos = int(open(POS).read()) if os.path.exists(POS) else 0
t0 = t_last = time.time()
while True:
    alive = subprocess.run('pgrep -f "[r]un_ciclo5.py"', shell=True, capture_output=True).returncode == 0
    if os.path.exists(LOG):
        with open(LOG, 'rb') as f:
            f.seek(pos); chunk = f.read(); pos = f.tell()
        open(POS, 'w').write(str(pos))
        for line in chunk.decode('utf-8', 'replace').replace('\r', '\n').split('\n'):
            s = line.strip()
            if s and 'it/s]' not in s and 's/it]' not in s and '%|' not in s:
                print(s, flush=True); t_last = time.time()
    if not alive:
        print('== run_ciclo5.py terminou ==', flush=True); break
    if time.time() - t_last > 300:
        gpu = subprocess.run('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader',
                             shell=True, capture_output=True, text=True).stdout.strip()
        print(f'[{time.strftime("%H:%M:%S")}] rodando | GPU {gpu}', flush=True); t_last = time.time()
    if time.time() - t0 > 1680:
        break
    time.sleep(20)
