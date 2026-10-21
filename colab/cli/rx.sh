#!/usr/bin/env bash
# uso: rx.sh <script.py> [timeout_s]  — executa um script de ~/fv no runtime
export PATH="$HOME/.local/bin:$PATH"
colab exec -s vit -f "$HOME/fv/$1" --timeout "${2:-120}" 2>&1 | grep --line-buffered -v bogus
