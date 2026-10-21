import glob, os, subprocess
ps = glob.glob('/content/parts/*')
print(len(ps), 'pedacos', round(sum(os.path.getsize(p) for p in ps) / 1e9, 2), 'GB enviados')
print(subprocess.run('pgrep -f "[r]un_ciclo5.py" >/dev/null && echo treino VIVO; tail -c 600 /content/out/execucao.log 2>/dev/null',
                     shell=True, capture_output=True, text=True, errors='replace').stdout)
