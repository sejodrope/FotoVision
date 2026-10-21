#!/usr/bin/env bash
# Empacota o que existir em /content (pode rodar no meio do treino, para trazer os
# modelos ja prontos) e baixa em pedacos, conferindo o SHA-256.
export PATH="$HOME/.local/bin:$PATH"
E="$(ls -d /mnt/c/Users/Jos*Pedro/Documents/WORK/PROJECTS/FitoVision/colab_export_c5)"
DEST="$E/retorno"; S=c5
say() { echo "[baixa $(date +%H:%M:%S)] $*"; }
colab exec -s $S -f "$HOME/fv5/pack.py" --timeout 900 2>&1 | grep -v bogus | tee ~/fv5/pack.out
mkdir -p "$DEST" ~/fv5/rparts; rm -f ~/fv5/rparts/*
colab download -s $S /content/rparts/manifest.txt ~/fv5/rparts/manifest.txt 2>&1 | tail -1
sha=$(head -1 ~/fv5/rparts/manifest.txt)
tail -n +2 ~/fv5/rparts/manifest.txt | while read -r p size; do
  for try in 1 2 3 4 5; do
    colab download -s $S "/content/rparts/$p" ~/fv5/rparts/$p >/dev/null 2>&1
    [ "$(stat -c %s ~/fv5/rparts/$p 2>/dev/null)" = "$size" ] && break
    sleep 5
  done
done
cat $(tail -n +2 ~/fv5/rparts/manifest.txt | cut -d' ' -f1 | sed "s|^|$HOME/fv5/rparts/|") > ~/fv5/c5_resultados.zip
got=$(sha256sum ~/fv5/c5_resultados.zip | cut -d' ' -f1)
if [ -n "$sha" ] && [ "$got" = "$sha" ]; then
  cp ~/fv5/c5_resultados.zip "$DEST/c5_resultados.zip"
  say "download conferido ($(du -h ~/fv5/c5_resultados.zip | cut -f1)) -> $DEST/c5_resultados.zip"
else
  say "ERRO: sha nao confere"
fi
colab usage 2>&1 | tr '\n' ' '; echo
