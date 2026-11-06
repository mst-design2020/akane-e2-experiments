#!/bin/bash
# 使い方: DOID_FILE=... bash run28.sh answers.json tag   （分け方5通りを並べて回し、まとめて判定）
D=$(dirname "$0")
for s in 0 1 2 3 4; do python3 "$D/e28.py" "$1" "$2_s$s" $s > "$D/../results/E28_runs/log_$2_s$s.txt" 2>&1 & done
wait; python3 "$D/merge28.py" "$2"
