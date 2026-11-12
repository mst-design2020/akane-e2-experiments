"""E11-3 の箱だけを Colab の GPU で学習する（計算機では重すぎたため）。python3 build_notebook_E11-3_box.py out.ipynb
中身は e112_box.py・e113_box.py と同じ（数も同じ）。正解は入れない。"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS = [{k: x[k] for k in ("id", "dom", "ent", "cls")} for x in json.load(open(f"{HERE}/../../data/E10/E10-1/e101_items.json"))["items"]]
BOX = open(f"{HERE}/e112_box.py").read()
B13 = open(f"{HERE}/e113_box.py").read().split('if __name__ == "__main__":')[0]
B13 = B13.replace("import e112_box as BXM", "").replace("sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))", "")
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md("""
# AKANE E11-3：箱の形のベクトルを GPU で学習する（部分と全体・病気・地理）

E11-3 の比べのうち、箱の学習だけが手元の計算機では重すぎたので、ここで流します（中身と数は E11-2 と同じ）。
正解はこのノートに入っていません。手で書いた語の一覧もありません。材料は E10-1_results の鎖と着きです。

## 手順
1. 「ランタイム」→「ランタイムのタイプを変更」→ **L4 GPU**（T4 でも可）。
2. 「すべてのセルを実行」。ドライブの許可を聞かれたら許可。
3. 終わると `AKANE_アーキテクチャ/E11-3_results/box_P.json`・`box_D.json`・`box_G.json` ができます。20〜40分の見込み。
""")
code("""
import os, json, gzip, time, sys, types
import numpy as np
from google.colab import drive
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/AKANE_アーキテクチャ"
R101 = f"{BASE}/E10-1_results"; OUT = f"{BASE}/E11-3_results"; os.makedirs(OUT, exist_ok=True)
for n in ["E101_steps", "E101_res_P", "E101_res_D", "E101_res_G"]:
    assert os.path.exists(f"{R101}/{n}.json.gz"), n
os.environ["BOX_DEVICE"] = "cuda"
""")
code("# 箱の中身（analysis/e11/e112_box.py と同じ）\n" + BOX)
code("BXM = types.SimpleNamespace(train=train, score=score)\n"
     "ITEMS = json.loads(r'''" + json.dumps({"items": ITEMS}, ensure_ascii=False) + "''')\n"
     "json.dump(ITEMS, open('/content/e101_items_nogold.json', 'w'))\nprint(len(ITEMS['items']), '主張')")
code("# 場面ごとの学習（analysis/e11/e113_box.py と同じ）\n" + B13)
code("""
for dom in "GDP":
    path = f"{OUT}/box_{dom}.json"
    if os.path.exists(path):
        print(dom, "済み"); continue
    t = time.time(); main(R101, "/content/e101_items_nogold.json", dom, path)
    print(dom, f"{time.time() - t:.0f}秒", flush=True)
""")
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells), "size", os.path.getsize(sys.argv[1]))
