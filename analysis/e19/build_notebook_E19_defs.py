"""E19 点検：600語の辞書の説明文（最初の3つ）を印字する。GPU なし。
python3 build_notebook_E19_defs.py out.ipynb"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
CORE101 = open(f"{HERE}/../e10/e101/e101_core.py").read()
C = json.load(open(f"{HERE}/results/c19.json"))
WORDS = sorted({it[1] for it in C["items"]})
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md("# AKANE E19 点検：説明文に分類の語がそのまま書いてあるか\n\nE19 の 600 語について、Wiktionary の説明文（E10-1 と同じ読み方、最初の3つ）を印字するだけです。GPU は要りません。20分ほど。")
code("CORE101 = " + repr(CORE101) + "\nWORDS = " + repr(WORDS) + "\nDUMP_URL = \"https://dumps.wikimedia.org/enwiktionary/latest/enwiktionary-latest-pages-articles.xml.bz2\"")
code("""
import subprocess, os, bz2, json, time
E101 = {}; exec(CORE101, E101)
dump = "/content/enwiktionary.xml.bz2"
subprocess.run(["wget", "-c", "-q", "-O", dump, DUMP_URL], check=True)
t = time.time()
with bz2.open(dump, "rb") as f: WKD = E101["parse_dump"](f)[0]
print("読み終わり", len(WKD), "語", f"{time.time() - t:.0f}秒")
D = {w: [d for d, lb in WKD.get(w, [])[:3]] for w in WORDS}
print("E19_DEFS_BEGIN"); print(json.dumps(D, ensure_ascii=True)); print("E19_DEFS_END")
""")
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print(len(WORDS), os.path.getsize(sys.argv[1]))
