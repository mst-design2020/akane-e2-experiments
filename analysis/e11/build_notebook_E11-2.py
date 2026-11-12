"""E11-2 Colabノート（E8n の方法D の鎖を6段まで伸ばす）。python3 build_notebook_E11-2.py out.ipynb
計画：plans/E11-2_deep_chain_and_box.md。材料を取るだけ（深い鎖・箱・いいとこ取りの比べは e112_analyze.py で別に）。"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..")
OLD = [{k: x[k] for k in ("id", "ent", "catw")} for x in json.load(open(f"{R}/e6/e6_items.json"))]  # 正解は入れない
NEW = [{k: x[k] for k in ("id", "ent", "catw")} for x in json.load(open(f"{R}/e8/e8_new_items.json"))]
CORE8 = open(f"{R}/nolist/e8n_core.py").read()
CORE = open(f"{HERE}/e112_core.py").read()
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md("""
# AKANE E11-2：辞書の鎖を3段から6段まで伸ばす

このノートは **材料を取るだけ** です（比べは結果のファイルを読んで別に行います）。計画：plans/E11-2_deep_chain_and_box.md。

- E8n（一覧なし版）の結果をそのまま読み、その先（4〜6段目）だけを足します。3段までは E8n と同じです。
- 途中の語の意味は、E8n と同じく Gemma が選びます。頭の名詞は文法で候補を出し、NLI で選びます。
- 正解はこのノートに入っていません。手で書いた語の一覧もありません。

## 手順
1. 「ランタイム」→「ランタイムのタイプを変更」→ **L4 GPU**。
2. 鍵マークの `HF_TOKEN` にアクセスを許可（E6〜E11-1と同じ）。
3. 「ランタイム」→「すべてのセルを実行」。ドライブの許可を聞かれたら許可。
4. 終わると、ドライブの `AKANE_アーキテクチャ/E11-2_results/E11-2_results.json`（と .gz）ができます。
5. 時間の見込み：10〜20分（モデルの読み込みが大半。足す組は数千ほどの見込み）。
""")
code("""
import os, json, time, subprocess, sys, gzip, shutil
import numpy as np
from google.colab import drive, userdata
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/AKANE_アーキテクチャ"
OUT = f"{BASE}/E11-2_results"
os.makedirs(OUT, exist_ok=True)
E8N = f"{BASE}/E8n_results/E8n_results.json"
WKD_PATH = f"{BASE}/E8_results/E8_wiktionary_dump_senses.json"
for p in (E8N, WKD_PATH):
    assert os.path.exists(p), "見つかりません: " + p
NLI_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"   # E6〜E8nと同じ
GEMMA_ID = "google/gemma-2-2b-it"
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"], check=False)
""")
code("# 中身1（analysis/nolist/e8n_core.py と同じ。手書きの語の一覧はありません）\n" + CORE8)
code("# 中身2（analysis/e11/e112_core.py と同じ。手書きの語の一覧はありません）\n" + CORE)
code("OLD = json.loads(r'''" + json.dumps(OLD, ensure_ascii=False) + "''')\n"
     "NEW = json.loads(r'''" + json.dumps(NEW, ensure_ascii=False) + "''')\n"
     "ITEMS = OLD + NEW\n"
     "r8 = json.load(open(E8N))\n"
     "L = r8['L']\n"
     "CHOICE = {(w, k): v for w, k, v in r8['choice']['D']}\n"
     "ST = {(w, k): (e, c) for w, k, e, c in r8['steps']}\n"
     "AR = {(w, k, C): (e, c) for w, k, C, e, c in r8['arrive']}\n"
     "WKD = json.load(open(WKD_PATH))\n"
     "# 確かめ：3段で切ると E8n の方法D の鎖と同じ\n"
     "CH8 = {i: nd for i, m, nd, lk in r8['chains'] if m == 'D'}\n"
     "bad = sum(set(follow(L, CHOICE, it['ent'], 3)[0]) != {tuple(n) for n in CH8[it['id']]} for it in ITEMS)\n"
     "assert bad == 0, f'3段の鎖が E8n と {bad} 問で違う'\n"
     "print(len(ITEMS), '問  語', len(L), '  3段の鎖は E8n と一致')")
md("## 1. NLI と Gemma を読み込む（GPU、2〜4分）")
code("""
import torch, spacy, hashlib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
nlp = spacy.load("en_core_web_sm", disable=["ner"])
ntok = AutoTokenizer.from_pretrained(NLI_ID)
nm = AutoModelForSequenceClassification.from_pretrained(NLI_ID).cuda().eval()
lab = {v.lower(): int(k) for k, v in nm.config.id2label.items()}
ORDER = [lab["entailment"], lab["neutral"], lab["contradiction"]]
tok = AutoTokenizer.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), padding_side="left")
gm = AutoModelForCausalLM.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), torch_dtype=torch.float16, device_map="cuda").eval()
NUM = [tok.encode(str(i), add_special_tokens=False)[0] for i in range(1, MAX_DEFS_MID + 1)]

def nli(trip, B=64):
    # (w, k, c) → (含意, 矛盾)。前提「"w" means: 説明文」、仮説「W is a type of c.」（E8n と同じ）
    P = [premise(w, L[w][k]["def"]) for w, k, c in trip]; H = [hyp(w, c) for w, k, c in trip]
    out = []
    for i in range(0, len(P), B):
        enc = ntok(P[i:i + B], H[i:i + B], return_tensors="pt", padding=True, truncation=True, max_length=192).to("cuda")
        with torch.no_grad():
            pr = torch.softmax(nm(**enc).logits.float(), -1)[:, ORDER].cpu().numpy()
        out += [(round(float(p[0]), 4), round(float(p[2]), 4)) for p in pr]
    return out

DPROB = {}
def gemma(jobs, B=16):
    # (w, k) → 頭の名詞 g の意味の番号（0から）。聞き方は E8n の方法D と同じ
    out = []
    for i in range(0, len(jobs), B):
        bt = jobs[i:i + B]
        qs = [choice_prompt(w, L[w][k]["def"], L[w][k]["genus"], [x["def"] for x in L[L[w][k]["genus"]]]) for w, k in bt]
        ps = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            last = gm(**enc, logits_to_keep=1).logits[:, -1, :].float()
        for (w, k), row in zip(bt, last):
            n = len(L[L[w][k]["genus"]]); pr = torch.softmax(row[NUM[:n]], -1).cpu().numpy()
            DPROB[(w, k)] = pr.round(4).tolist(); out.append(int(pr.argmax()))
    return out
print("読み込み済み")
""")
md("## 2. 4〜6段目を足す（GPU、10〜25分）")
code("""
t = time.time()
added = extend(ITEMS, L, CHOICE, ST, AR, WKD, nlp, nli, gemma, maxd=MAX_DEPTH)
print("足した物:", added, f"{time.time() - t:.0f}秒")
CHAINS = chains(ITEMS, L, CHOICE, MAX_DEPTH)
import collections
print("段の数ごとの点:", dict(sorted(collections.Counter(d for nd, lk in CHAINS.values() for w, k, d in nd).items())))
""")
md("## 3. 保存")
code("""
res = {"meta": {"nli": NLI_ID, "gemma": GEMMA_ID, "base": "E8n_results/E8n_results.json", "max_depth": MAX_DEPTH,
                "date": time.strftime("%Y-%m-%d"), "added": added},
       "L": L,
       "choice": [[w, k, v] for (w, k), v in CHOICE.items()],
       "dprob_new": [[w, k, v] for (w, k), v in DPROB.items()],
       "steps": [[w, k, e, c] for (w, k), (e, c) in ST.items()],
       "arrive": [[w, k, C, e, c] for (w, k, C), (e, c) in AR.items()],
       "chains": [[i, nd, lk] for i, (nd, lk) in CHAINS.items()]}
path = f"{OUT}/E11-2_results.json"
json.dump(res, open(path, "w"), ensure_ascii=False)
with open(path, "rb") as a, gzip.open(path + ".gz", "wb") as b:
    shutil.copyfileobj(a, b)
print("保存:", path, round(os.path.getsize(path) / 1e6, 1), "MB（圧縮", round(os.path.getsize(path + ".gz") / 1e6, 1), "MB）")
""")
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells), "size", os.path.getsize(sys.argv[1]))
