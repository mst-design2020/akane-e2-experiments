"""E9-1の材料（大きい）から、確かめの答えだけの小さなファイルを作るColabノート。python3 build_notebook_E9-1b.py out.ipynb"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
CHK = open(f"{HERE}/e91_checker.py").read()
body = CHK.split('R = json.load(open(sys.argv[1]))')[1].split('here = os.path.dirname')[0]
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md("""
# AKANE E9-1（続き）：確かめの答えだけを小さなファイルにする
E9-1_results.json が大きく（15MB）こちらで受け取れないため、辞書の確かめの答えだけを取り出します。GPUは要りません（1〜2分）。
「すべてのセルを実行」→ ドライブの `AKANE_アーキテクチャ/E9-1_results/E91_checker_small.json` ができます。
""")
code("""
import json, os
import numpy as np
from google.colab import drive
drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/AKANE_アーキテクチャ/E9-1_results"
DEPTH = 3
R = json.load(open(f"{OUT}/E9-1_results.json"))
""")
code("# 中身（sims/e9/e91_checker.py と同じ）" + body)
code("""
ENTS = sorted(R['chains']); CATS = R['meta']['cats']
small = {'meta': {k: v for k, v in R['meta'].items()}, 'ents': ENTS, 'cats': CATS, 'checkers': {}, 'scores_e8D': {}, 'scores_e8one': {}}
for name in out:
    small['checkers'][name] = ''.join(''.join(str(out[name][f"{w}|{C}"]) if f"{w}|{C}" in out[name] else '-' for C in CATS) for w in ENTS)
for name in score:
    small['scores_' + name] = [int(round(score[name].get(f"{w}|{C}", -0.001) * 1000)) for w in ENTS for C in CATS]
small['n_nodes'] = len(R['nodes']); small['n_steps'] = len(R['steps'])
small['choice_first_rate'] = float(np.mean([v == 0 for w, k, v in R.get('choice_D', [[0, 0, 0]])]))
p = f"{OUT}/E91_checker_small.json"
json.dump(small, open(p, "w"))
print("保存:", p, round(os.path.getsize(p) / 1e6, 2), "MB")
print({k: (v.count('1'), v.count('0'), v.count('-')) for k, v in small['checkers'].items()})
""")
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells))
