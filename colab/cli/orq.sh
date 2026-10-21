#!/usr/bin/env bash
# Orquestrador do ViT no Colab: runtime -> envio -> conferencia -> treino -> download -> desliga.
# Pode ser relancado: cada etapa pula o que ja esta feito.
export PATH="$HOME/.local/bin:$PATH"
cd ~/fv
E="$(ls -d /mnt/c/Users/Jos*Pedro/Documents/WORK/PROJECTS/FitoVision/colab_export)"
DEST="$(dirname "$E")/colab_export/retorno"
say() { echo "[orq $(date +%H:%M:%S)] $*"; }
rx()  { colab exec -s vit -f "$HOME/fv/$1" --timeout "${2:-120}" 2>&1 | grep --line-buffered -v bogus; }

if colab status -s vit 2>&1 | grep -q 'Hardware: L4'; then
  say "runtime 'vit' ja existe"
else
  say "criando runtime L4"; colab new -s vit --gpu L4 2>&1 | tail -2
  colab status -s vit 2>&1 | grep -q 'Hardware: L4' || { say "ERRO: runtime nao criado"; exit 1; }
fi

say "enviando codigo e manifesto de hashes"
colab upload -s vit "$E/fitovision_codigo.zip" /content/fitovision_codigo.zip 2>&1 | tail -1
colab upload -s vit ~/fv/esperado.json /content/esperado.json 2>&1 | tail -1

if rx extract.py 60 | grep -q 'fitovision_dados_train.zip - ja extraido'; then
  say "dados ja extraidos"
else
  say "enviando dados (11,4 GB em pedacos de 32 MB)"
  bash ~/fv/upload_all.sh 2>&1 | grep --line-buffered -E 'concluido|TODOS|falha'
  say "remontando e conferindo SHA-256"
  rx assemble.py 1800
  say "extraindo"
  rx extract.py 1800 | grep -E 'zip|overlay'
fi
rx extract.py 60 | grep -q 'fitovision_dados_train.zip - ja extraido' || { say "ERRO: dados incompletos; runtime mantido"; exit 2; }
colab usage 2>&1 | tr '\n' ' '; echo

say "lancando run_vit.py"
rx launch.py 300
for i in 1 2 3 4 5 6; do
  rx watch.py 1200
  rx alive.py 60 | grep -q VIVO || break
  say "watch caiu com o treino vivo; reconectando ($i)"; sleep 20
done

say "empacotando resultados"
rx pack.py 900 | tee ~/fv/pack.out
mkdir -p "$DEST" ~/fv/rparts; rm -f ~/fv/rparts/*
colab download -s vit /content/rparts/manifest.txt ~/fv/rparts/manifest.txt 2>&1 | tail -1
sha=$(head -1 ~/fv/rparts/manifest.txt)
tail -n +2 ~/fv/rparts/manifest.txt | while read -r p size; do
  for try in 1 2 3 4 5; do
    colab download -s vit "/content/rparts/$p" ~/fv/rparts/$p >/dev/null 2>&1
    [ "$(stat -c %s ~/fv/rparts/$p 2>/dev/null)" = "$size" ] && break
    sleep 5
  done
done
cat $(tail -n +2 ~/fv/rparts/manifest.txt | cut -d' ' -f1 | sed "s|^|$HOME/fv/rparts/|") > ~/fv/vit_resultados.zip
got=$(sha256sum ~/fv/vit_resultados.zip | cut -d' ' -f1)
colab usage 2>&1 | tr '\n' ' '; echo
if [ -n "$sha" ] && [ "$got" = "$sha" ]; then
  cp ~/fv/vit_resultados.zip "$DEST/vit_resultados.zip"
  say "download conferido ($(du -h ~/fv/vit_resultados.zip | cut -f1)) -> $DEST/vit_resultados.zip"
  if grep -q 'STATUS COMPLETO' ~/fv/pack.out; then
    colab stop -s vit 2>&1 | tail -1; say "FIM_OK runtime desligado"
  else
    say "FIM_INCOMPLETO: run_vit.py nao chegou ao fim; runtime MANTIDO para diagnostico"
  fi
else
  say "FIM_ERRO_DOWNLOAD: sha nao confere; runtime MANTIDO (resultados seguem em /content/out)"
fi
