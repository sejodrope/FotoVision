#!/usr/bin/env bash
E="$(ls -d /mnt/c/Users/Jos*Pedro/Documents/WORK/PROJECTS/FitoVision/colab_export)"
cd "$E"
{
  echo "{"
  first=1
  for z in fitovision_dados_test.zip fitovision_dados_val.zip fitovision_dados_train.zip; do
    [ $first = 1 ] || echo ","
    first=0
    printf '"%s": [%s, "%s"]' "$z" "$(stat -c %s "$z")" "$(sha256sum "$z" | cut -d' ' -f1)"
  done
  echo; echo "}"
} > ~/fv/esperado.json
cat ~/fv/esperado.json
