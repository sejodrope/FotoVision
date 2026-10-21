import os, subprocess, sys

WORK, OUT = '/content/fitovision/backend', '/content/out'
os.makedirs(OUT, exist_ok=True)
alive = subprocess.run('pgrep -f "[r]un_vit.py"', shell=True, capture_output=True, text=True).stdout.strip()
if alive:
    print('run_vit.py ja esta rodando, pid', alive.split())
else:
    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'pydantic-settings'], check=True)
    log = open(f'{OUT}/execucao.log', 'a')
    p = subprocess.Popen(
        [sys.executable, '-u', 'run_vit.py',
         '--weights-dir', f'{OUT}/weights', '--logs-dir', f'{OUT}/logs',
         '--results-dir', f'{OUT}/results', '--workers', str(os.cpu_count())],
        cwd=WORK, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    print('lancado, pid', p.pid)
