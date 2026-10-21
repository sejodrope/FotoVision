import os, subprocess, sys

WORK, OUT = '/content/fitovision/backend', '/content/out'
os.makedirs(OUT, exist_ok=True)
alive = subprocess.run('pgrep -f "[r]un_ciclo5.py"', shell=True, capture_output=True, text=True).stdout.strip()
if alive:
    print('run_ciclo5.py ja esta rodando, pid', alive.split())
else:
    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'pydantic-settings'], check=True)
    log = open(f'{OUT}/execucao.log', 'a')
    # Ordem: o modelo servido primeiro, para ficar pronto o quanto antes.
    p = subprocess.Popen(
        [sys.executable, '-u', 'run_ciclo5.py', '--modelos',
         'efficientnet_b0', 'vit_b_16', 'mobilenet_v2', 'resnet50',
         '--sem-auditoria', '--workers', str(min(8, os.cpu_count() or 2))],
        cwd=WORK, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    print('lancado, pid', p.pid)
