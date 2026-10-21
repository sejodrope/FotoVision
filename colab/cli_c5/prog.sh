#!/usr/bin/env bash
# Progresso no runtime c5 (envio e treino), sem atrapalhar o orq.sh.
export PATH="$HOME/.local/bin:$PATH"
cp /mnt/c/Users/Jos*Pedro/Documents/WORK/PROJECTS/FitoVision/colab/cli_c5/prog.py ~/fv5/prog.py
colab exec -s c5 -f ~/fv5/prog.py --timeout 60 2>&1 | grep -v bogus
