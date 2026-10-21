#!/usr/bin/env bash
# Interrompe o upload_all/chunkup (sem tocar no orq.sh, que segue para a remontagem).
for nome in upload_all.sh chunkup.sh; do
  for p in $(pgrep -f "fv5/$nome"); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done
done
sleep 1
for p in $(pgrep -f "colab upload -s c5"); do kill "$p" 2>/dev/null; done
sleep 2
ps -eo pid,etime,args | grep -E "upload_all|chunkup|colab upload" | grep -v grep | cut -c1-120
echo parado
