#!/usr/bin/env bash
# uso: baixa_modelo.sh <modelo>  — traz os pesos e a calibracao de um modelo ja pronto,
# direto pelos arquivos (sem usar o kernel, que fica ocupado com o watch.py).
export PATH="$HOME/.local/bin:$PATH"
m="$1"; S=c5
B=/content/fitovision/backend
P="$(ls -d /mnt/c/Users/Jos*Pedro/Documents/WORK/PROJECTS/FitoVision)/backend"
mkdir -p "$P/weights/ciclo5" "$P/logs/ciclo5"
for f in "weights/ciclo5/${m}_binary.pth" "weights/ciclo5/${m}_binary_calibration.json"; do
  for try in 1 2 3; do
    timeout 300 colab download -s $S "$B/$f" "$P/$f" 2>&1 | grep -v bogus | tail -1
    [ -s "$P/$f" ] && break; sleep 5
  done
done
ls -la "$P/weights/ciclo5/"
