"""E7 Colabノートを作る。python3 build_notebook_E7.py out.ipynb"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS = [{k: x[k] for k in ("id", "ent", "catw", "stmt")} for x in json.load(open(f"{HERE}/e6_items.json"))]  # 正解は入れない
CORE = open(f"{HERE}/e7_core.py").read()
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md("""
# AKANE E7：辞書の説明文を段々につなぐ（孫娘→人→哺乳類）

このノートは **材料を取るだけ** です（比べは結果のファイルを読んで別に行います）。計画：sims/e6/PLAN_E7.md。

- E6と同じ700個。物の語の説明文（E6で取ったもの）から頭の言葉をたどり、3段まで辞書をつなぎます。
- 途中の語の説明文は、Wiktionaryの全部入りのファイル（ダンプ、約1.5GB）を一度だけ取って読みます（サイトに何度も頼まないので断られません）。
- 各段と「分類に着いたか」を NLI（E6と同じ）で判定します。

## 手順
1. 「ランタイム」→「ランタイムのタイプを変更」→ **L4 GPU**。
2. 「ランタイム」→「すべてのセルを実行」。ドライブの許可を聞かれたら許可。
3. 終わると、ドライブの `AKANE_アーキテクチャ/E7_results/E7_results.json` ができます。
4. 時間の見込み：ダンプを取る 5〜15分、読む 20〜30分（この間GPUは使わないので「GPUを使っていません」の警告が出ますが、×で閉じてください）、NLI 5〜10分。途中で切れても、もう一度「すべてのセルを実行」すれば、済んだ所は飛ばします。
""")
code("""
import os, json, time, bz2, subprocess, sys
from google.colab import drive
drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/AKANE_アーキテクチャ/E7_results"
os.makedirs(OUT, exist_ok=True)
E6_WK = "/content/drive/MyDrive/AKANE_アーキテクチャ/E6_results/E6_wiktionary.json"
NLI_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"   # E6と同じ
DUMP_URL = "https://dumps.wikimedia.org/enwiktionary/latest/enwiktionary-latest-pages-articles.xml.bz2"
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"], check=False)
""")
code("# 中身（sims/e6/e7_core.py と同じ）\n" + CORE)
code("ITEMS = json.loads(r'''" + json.dumps(ITEMS, ensure_ascii=False) + "''')\n"
     "WK6 = json.load(open(E6_WK))\nfor it in ITEMS: it['wk_defs'] = WK6.get(it['ent'], [])\n"
     "print(len(ITEMS), '個', ' 物の語の説明文あり:', sum(1 for it in ITEMS if it['wk_defs']))")
md("## 1. Wiktionaryのダンプを取って読む（25〜45分、GPUは使わない）")
code("""
wkd_path = f"{OUT}/E7_wiktionary_dump_nouns.json"
if os.path.exists(wkd_path):
    WKD = json.load(open(wkd_path)); print("ドライブから読み込み:", len(WKD))
else:
    dump = "/content/enwiktionary.xml.bz2"
    subprocess.run(["wget", "-c", "-q", "--show-progress", "-O", dump, DUMP_URL], check=True)
    print("取れた大きさ:", round(os.path.getsize(dump) / 1e9, 2), "GB")
    t = time.time()
    with bz2.open(dump, "rb") as f:
        WKD = parse_dump(f)
    print("読み終わり:", len(WKD), "語", f"{time.time() - t:.0f}秒")
    json.dump(WKD, open(wkd_path, "w"), ensure_ascii=False)
print("例 cat:", WKD.get("cat", [])[:2])
""")
md("## 2. 辞書をたどって網の材料を作る")
code("""
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])
G = build_graph(ITEMS, lambda w, lim: WKD.get(w, [])[:lim], nlp)
steps, arrive = nli_jobs(ITEMS, G)
print("語の数:", len(G), " 段の組:", len(steps), " 着きの組:", len(arrive))
for w in ["kitten", "granddaughter", "barbell"]:
    if w in G: print(w, [x["genus"] for x in G[w]])
""")
md("## 3. NLIで判定する（5〜10分、GPU）")
code("""
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
ntok = AutoTokenizer.from_pretrained(NLI_ID)
nm = AutoModelForSequenceClassification.from_pretrained(NLI_ID).cuda().eval()
lab = {v.lower(): int(k) for k, v in nm.config.id2label.items()}
ORDER = [lab["entailment"], lab["neutral"], lab["contradiction"]]
def nli_probs(P, H, B=64):
    out = []
    for i in range(0, len(P), B):
        enc = ntok(P[i:i + B], H[i:i + B], return_tensors="pt", padding=True, truncation=True, max_length=192).to("cuda")
        with torch.no_grad():
            pr = torch.softmax(nm(**enc).logits.float(), -1)[:, ORDER]
        out += pr.cpu().numpy().round(4).tolist()
        if (i // B) % 200 == 0: print(f"{i}/{len(P)}")
    return out
t = time.time()
S = nli_probs([premise(w, G[w][k]["def"]) for w, k in steps], [hyp(w, G[w][k]["genus"]) for w, k in steps])
A = nli_probs([premise(w, G[w][k]["def"]) for w, k, C in arrive], [hyp(w, C) for w, k, C in arrive])
res = {"meta": {"nli": NLI_ID, "dump": DUMP_URL, "date": time.strftime("%Y-%m-%d"), "n_items": len(ITEMS)},
       "G": G, "steps": [[w, k, p[0], p[2]] for (w, k), p in zip(steps, S)],
       "arrive": [[w, k, C, p[0], p[2]] for (w, k, C), p in zip(arrive, A)]}
path = f"{OUT}/E7_results.json"
json.dump(res, open(path, "w"), ensure_ascii=False)
print("保存:", path, round(os.path.getsize(path) / 1e6, 1), "MB", f"{time.time() - t:.0f}秒")
""")
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells))
