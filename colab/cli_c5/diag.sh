#!/usr/bin/env bash
export PATH="$HOME/.local/bin:$PATH"
colab ls 2>&1 | grep -v bogus
colab usage 2>&1 | grep -v bogus
colab status -s c5 2>&1 | grep -v bogus
