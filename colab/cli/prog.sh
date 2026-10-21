#!/usr/bin/env bash
export PATH="$HOME/.local/bin:$PATH"
echo "$(date +%H:%M:%S) uploads locais ativos: $(pgrep -fc 'bin/colab upload')"
colab exec -s vit --timeout 60 <<'EOF' 2>&1 | grep -v bogus | tail -6
import os, glob, collections
c = collections.Counter(); b = collections.Counter()
for f in glob.glob('/content/parts/*'):
    k = os.path.basename(f).rsplit('.', 1)[0]; c[k] += 1; b[k] += os.path.getsize(f)
for k in sorted(c): print(f'{k}: {c[k]} pedacos, {b[k] / 1e6:.0f} MB')
print('total MB', round(sum(b.values()) / 1e6))
EOF
