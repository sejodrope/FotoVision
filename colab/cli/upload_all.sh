#!/usr/bin/env bash
# Envia os tres zips de dados em pedacos, P conexoes em paralelo. Cada rodada repete
# o que tiver falhado (o chunkup.sh pula o que ja esta no runtime).
export PATH="$HOME/.local/bin:$PATH"
P=${P:-4}
rm -rf ~/fv/tmp_*
for z in "${@:-fitovision_dados_test.zip fitovision_dados_val.zip fitovision_dados_train.zip}"; do
 for z in $z; do
  for round in 1 2 3; do
    for (( w=0; w<P; w++ )); do
      bash ~/fv/chunkup.sh "$z" "$w" "$P" 2>&1 | grep -v bogus | sed "s/^/[w$w] /" &
      sleep 3
    done
    wait
  done
  echo "=== $z concluido $(date +%H:%M:%S) ==="
 done
done
echo TODOS_ENVIADOS
