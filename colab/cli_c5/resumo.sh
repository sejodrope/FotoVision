#!/usr/bin/env bash
export PATH="$HOME/.local/bin:$PATH"
cp /mnt/c/Users/Jos*Pedro/Documents/WORK/PROJECTS/FitoVision/colab/cli_c5/resumo.py ~/fv5/resumo.py
timeout 150 colab exec -s c5 -f ~/fv5/resumo.py --timeout 120 2>&1 | grep -v bogus
