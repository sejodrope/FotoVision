import os, subprocess, time, zipfile

WORK = '/content/fitovision/backend'
os.makedirs(WORK, exist_ok=True)
for z in ['fitovision_codigo.zip', 'fitovision_dados_test.zip', 'fitovision_dados_retida.zip',
          'fitovision_dados_val.zip', 'fitovision_dados_train.zip']:
    src, feito = f'/content/{z}', f'{WORK}/.extraido_{z}'
    # O código é sempre reextraído: pode ter mudado entre duas rodadas.
    if os.path.exists(feito) and z != 'fitovision_codigo.zip':
        print(z, '- ja extraido', flush=True); continue
    if not os.path.exists(src):
        print(z, '- AUSENTE', flush=True); continue
    t0 = time.time()
    bad = zipfile.ZipFile(src).testzip() if os.path.getsize(src) < 5e6 else None
    assert bad is None, bad
    subprocess.run(['unzip', '-q', '-o', src, '-d', WORK], check=True)
    open(feito, 'w').close()
    print(f'{z} - {os.path.getsize(src) / 1e9:.2f} GB - {time.time() - t0:.0f} s', flush=True)
print(subprocess.run('df -h /content | tail -1; ls /content/fitovision/backend/data_c5 2>/dev/null',
                     shell=True, capture_output=True, text=True).stdout)
