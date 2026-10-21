#!/usr/bin/env bash
# uso: chunkup.sh <zip> [inicio] [passo]   — envia os pedacos inicio, inicio+passo, ...
# Retomavel: pula os pedacos que ja estao no runtime com o tamanho certo.
export PATH="$HOME/.local/bin:$PATH"
E="$(ls -d /mnt/c/Users/Jos*Pedro/Documents/WORK/PROJECTS/FitoVision/colab_export_c5)"
z="$1"; start="${2:-0}"; step="${3:-1}"
MB=${MB:-32}
size=$(stat -c %s "$E/$z")
n=$(( (size + MB*1024*1024 - 1) / (MB*1024*1024) ))
tmp="$HOME/fv5/tmp_${z}_${start}"; mkdir -p "$tmp"
have="$tmp/have.txt"
timeout 400 colab exec -s c5 --timeout 360 2>/dev/null > "$have" <<EOF
import os, glob
os.makedirs('/content/parts', exist_ok=True)
for f in sorted(glob.glob('/content/parts/$z.*')): print(os.path.basename(f), os.path.getsize(f))
EOF
t0=$(date +%s); sent=0
for (( i=start; i<n; i+=step )); do
  part=$(printf '%s.%04d' "$z" "$i")
  want=$(( i == n-1 ? size - i*MB*1024*1024 : MB*1024*1024 ))
  if grep -q "^$part $want\$" "$have"; then continue; fi
  for try in 1 2 3 4; do
    dd if="$E/$z" of="$tmp/$part" bs=1M skip=$(( i*MB )) count=$MB status=none
    out=$(timeout 300 colab upload -s c5 "$tmp/$part" "/content/parts/$part" 2>&1 | tail -1)
    rm -f "$tmp/$part"
    if [[ "$out" == *Uploaded* ]]; then break; fi
    if [[ "$out" == *"Session 'c5' not found"* ]]; then echo "SESSAO c5 not found - abortando $z"; exit 9; fi
    echo "  falha $part (tentativa $try): ${out:0:160}"; sleep 10
  done
  sent=$(( sent + want ))
  el=$(( $(date +%s) - t0 ))
  echo "$part ok | $(( sent/1048576 )) MB em ${el}s | $(( el > 0 ? sent*8/1000000/el : 0 )) Mbps"
done
echo "FIM $z inicio=$start passo=$step"
