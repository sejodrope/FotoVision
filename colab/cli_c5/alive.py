import subprocess
print('VIVO' if subprocess.run('pgrep -f "[r]un_ciclo5.py"', shell=True, capture_output=True).returncode == 0 else 'PARADO')
