#!/usr/bin/env bash
# Baixa o log do treino (API de arquivos, nao ocupa o kernel) e mostra a barra de progresso atual.
export PATH="$HOME/.local/bin:$PATH"
colab download -s vit /content/out/execucao.log ~/fv/execucao.log >/dev/null 2>&1
date +%H:%M:%S
tr '\r' '\n' < ~/fv/execucao.log | grep -aE '%\||Epoch' | tail -${1:-2} | cut -c1-200
