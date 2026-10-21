#!/usr/bin/env bash
# Orquestrador do ciclo 5 no Colab: runtime -> envio -> conferencia -> treino das quatro
# arquiteturas (run_ciclo5.py) -> download -> desliga. Mesma logica do colab/cli/orq.sh
# do ViT; pode ser relancado: cada etapa pula o que ja esta feito.
export PATH="$HOME/.local/bin:$PATH"
cd ~/fv5
E="$(ls -d /mnt/c/Users/Jos*Pedro/Documents/WORK/PROJECTS/FitoVision/colab_export_c5)"
DEST="$E/retorno"
S=c5
say() { echo "[orq $(date +%H:%M:%S)] $*"; }
rx()  { colab exec -s $S -f "$HOME/fv5/$1" --timeout "${2:-120}" 2>&1 | grep --line-buffered -v bogus; }

if colab status -s $S 2>&1 | grep -q 'Hardware: L4'; then
  say "runtime '$S' ja existe"
else
  say "criando runtime L4"; colab new -s $S --gpu L4 2>&1 | tail -2
  colab status -s $S 2>&1 | grep -q 'Hardware: L4' || { say "ERRO: runtime nao criado"; exit 1; }
fi

say "enviando codigo e manifesto de hashes"
colab upload -s $S "$E/fitovision_codigo.zip" /content/fitovision_codigo.zip 2>&1 | tail -1
colab upload -s $S ~/fv5/esperado.json /content/esperado.json 2>&1 | tail -1
rx extract.py 120 | grep -E 'codigo'

if rx extract.py 60 | grep -q 'fitovision_dados_train.zip - ja extraido'; then
  say "dados ja extraidos"
else
  say "enviando dados em pedacos de 32 MB"
  ( while true; do timeout 330 colab exec -s $S -f "$HOME/fv5/keep.py" --timeout 300 2>&1 \
      | grep --line-buffered -E 'keep' | tail -1; sleep 5; done ) &
  KEEP=$!
  bash ~/fv5/upload_all.sh 2>&1 | grep --line-buffered -E 'concluido|TODOS|not found'
  kill $KEEP 2>/dev/null; pkill -f 'fv5/keep.py' 2>/dev/null
  colab status -s $S 2>&1 | grep -q 'Hardware: L4' || { say "ERRO: runtime recolhido durante o envio"; exit 4; }
  say "remontando e conferindo SHA-256"
  rx assemble.py 1800
  say "extraindo"
  rx extract.py 1800 | grep -E 'zip'
fi
rx extract.py 60 | grep -q 'fitovision_dados_train.zip - ja extraido' || { say "ERRO: dados incompletos; runtime mantido"; exit 2; }
colab usage 2>&1 | tr '\n' ' '; echo

# A auditoria de quase-duplicatas roda no PC (--sem-auditoria no Colab): so treina com
# o veredicto OK no relatorio local.
REP="$(dirname "$E")/backend/results/ciclo5/leakage_report.json"
until [ -f "$REP" ]; do say "aguardando a auditoria local ($REP)"; sleep 120; done
grep -q '"verdict": "OK"' "$REP" || { say "ERRO: auditoria nao aprovou o split; runtime mantido"; exit 3; }
say "auditoria local OK"

say "lancando run_ciclo5.py"
rx launch.py 300
for i in $(seq 1 40); do
  rx watch.py 1800
  rx alive.py 60 | grep -q VIVO || break
  say "watch saiu com o treino vivo; reconectando ($i)"; sleep 20
done

bash ~/fv5/baixa.sh
if grep -q 'STATUS COMPLETO' ~/fv5/pack.out && [ -f "$DEST/c5_resultados.zip" ]; then
  colab stop -s $S 2>&1 | tail -1; say "FIM_OK runtime desligado"
else
  say "FIM_INCOMPLETO ou erro no download: runtime MANTIDO para diagnostico"
fi
