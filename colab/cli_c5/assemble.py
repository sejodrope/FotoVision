# Junta os pedacos de cada zip, confere tamanho e SHA-256 com os do PC e apaga os pedacos.
import glob, hashlib, json, os

ESPERADO = json.load(open('/content/esperado.json'))
for z, (size, sha) in ESPERADO.items():
    dst = f'/content/{z}'
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        print(z, '- ja montado', flush=True); continue
    parts = sorted(glob.glob(f'/content/parts/{z}.*'))
    total = sum(os.path.getsize(p) for p in parts)
    if total != size:
        print(f'{z} - INCOMPLETO: {len(parts)} pedacos, {total} de {size} bytes', flush=True); continue
    h = hashlib.sha256()
    with open(dst, 'wb') as out:
        for p in parts:
            with open(p, 'rb') as f:
                while chunk := f.read(1 << 24):
                    out.write(chunk); h.update(chunk)
    if h.hexdigest() != sha:
        os.remove(dst)
        print(f'{z} - SHA DIFERENTE ({h.hexdigest()[:16]} != {sha[:16]})', flush=True); continue
    for p in parts:
        os.remove(p)
    print(f'{z} - OK, {size / 1e9:.2f} GB, sha confere', flush=True)
