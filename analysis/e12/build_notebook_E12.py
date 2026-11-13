"""E12 Colabノート。python3 build_notebook_E12.py items_all.json out.ipynb
正解（人が付けた意味）はノートに入れない。材料を取るだけ（比べは e12_analyze.py で）。"""
import json, os, sys, gzip, base64
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
src, dst = sys.argv[1], sys.argv[2]
it = json.load(open(src))["items"]
heads = {}
for x in it:
    heads.setdefault(x["head"], x["cand_defs"])
    assert heads[x["head"]] == x["cand_defs"]
ITEMS = [[x["id"], x["kind"], x["word"], x["defn"], x["head"]] for x in it]
rng = np.random.default_rng(0)
hidx = [i for i, x in enumerate(it) if x["kind"] == "head"]
G0 = sorted(rng.choice(hidx, size=5000, replace=False).tolist())
blob = base64.b64encode(gzip.compress(json.dumps({"items": ITEMS, "heads": heads, "g0": G0}).encode())).decode()
CORE = open(f"{HERE}/e12_core.py").read()
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md(f"""
# AKANE E12：辞書の説明文の頭の語の意味を、Gemmaにたくさん選ばせる

このノートは **材料を取るだけ** です（比べは結果のファイルを読んで別に行います）。計画：plans/E12_gloss_head_sense.md。

- 問題は WordNet 3.0 の名詞の説明文 {len(ITEMS):,} 問（頭の語 {len(hidx):,}、「〜の」の後ろの語 {len(ITEMS) - len(hidx):,}）。
- 聞き方は E8 の方法D と同じ文です。候補はくじで並べ替えた順とその逆順の2通りで聞きます。
- 正解（人が付けた意味）はこのノートに入っていません。手で書いた語の一覧もありません。

## 手順
1. 「ランタイム」→「ランタイムのタイプを変更」→ **L4 GPU**。
2. 鍵マークの `HF_TOKEN` にアクセスを許可（いつもと同じ）。
3. 「ランタイム」→「すべてのセルを実行」。ドライブの許可を聞かれたら許可。
4. 終わると、ドライブの `AKANE_アーキテクチャ/E12_results/E12_results.json.gz` ができます。
5. 時間の見込み：40〜70分。途中で切れても、もう一度「すべてのセルを実行」すれば済んだ所（1万問ごとに保存）は飛ばします。
""")
code("""
import os, json, time, gzip, base64
import numpy as np
from google.colab import drive, userdata
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/AKANE_アーキテクチャ"
OUT = f"{BASE}/E12_results"
os.makedirs(OUT, exist_ok=True)
GEMMA_ID = "google/gemma-2-2b-it"   # E6〜E11と同じ
EMB_ID = "sentence-transformers/all-mpnet-base-v2"   # E8の方法Cと同じ
""")
code("# 中身（analysis/e12/e12_core.py と同じ。手書きの語の一覧はありません）\n" + CORE)
code("DATA = json.loads(gzip.decompress(base64.b64decode('" + blob + "')))\n"
     "ITEMS, HEADS, G0 = DATA['items'], DATA['heads'], DATA['g0']\n"
     "print(len(ITEMS), '問  頭の語の種類', len(HEADS), '  WordNetの順で聞く分', len(G0))")
md("## 1. 説明文どうしの近さ（方法C、GPU、2〜4分）")
code("""
import torch
from sentence_transformers import SentenceTransformer
ck = f"{OUT}/ck_sim.json"
if os.path.exists(ck):
    SIM = json.load(open(ck))
else:
    emb = SentenceTransformer(EMB_ID, device="cuda")
    texts = sorted({d for x in ITEMS for d in [x[3]]} | {d for v in HEADS.values() for d in v})
    V = emb.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
    pos = {t: i for i, t in enumerate(texts)}
    SIM = [[round(float(V[pos[x[3]]] @ V[pos[d]]), 4) for d in HEADS[x[4]]] for x in ITEMS]
    json.dump(SIM, open(ck, "w"))
    del emb; torch.cuda.empty_cache()
print("近さ 済み", len(SIM))
""")
md("## 2. Gemma に意味を選ばせる（方法D、GPU、40〜60分。1万問ごとにドライブへ保存）")
code("""
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), padding_side="left")
gm = AutoModelForCausalLM.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), torch_dtype=torch.float16, device_map="cuda").eval()
NUM = [tok.encode(str(i), add_special_tokens=False)[0] for i in range(1, MAX_SENSES + 1)]
assert len(set(NUM)) == MAX_SENSES

def ask(jb, ck, B=32):
    if os.path.exists(ck):
        c = json.load(open(ck))
        if c.get("n") == len(jb) and c.get("head") == jb[0][0][:200]:
            print("ドライブから読み込み", ck); return c["out"]
    out = []; t = time.time()
    order = sorted(range(len(jb)), key=lambda i: len(jb[i][0]))   # 長さをそろえて速く
    res = [None] * len(jb)
    for s in range(0, len(order), B):
        idx = order[s:s + B]
        ps = [tok.apply_chat_template([{"role": "user", "content": jb[i][0]}], tokenize=False, add_generation_prompt=True) for i in idx]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            last = gm(**enc, logits_to_keep=1).logits[:, -1, :].float()
        for row, i in zip(last, idx):
            res[i] = torch.softmax(row[NUM[:len(jb[i][1])]], -1).cpu().numpy().round(4).tolist()
        if (s // B) % 200 == 0: print(f"  {s}/{len(jb)}  {time.time() - t:.0f}秒")
    json.dump({"n": len(jb), "head": jb[0][0][:200], "out": res}, open(ck, "w"))
    return res

STEP = 10000
GD = []
for c0 in range(0, len(ITEMS), STEP):
    part = ITEMS[c0:c0 + STEP]
    jb = jobs(part, HEADS, seed=c0, shuffle=True)
    GD += merge(jb, ask(jb, f"{OUT}/ck_gemma_{c0}.json"), 2)
    print("済み", len(GD), "/", len(ITEMS))
# 参考：WordNet の順のまま聞く（最初の意味が先頭に来る。E8 は辞書の順のまま聞いていた）
jb0 = jobs([ITEMS[i] for i in G0], HEADS, shuffle=False)
G0P = merge(jb0, ask(jb0, f"{OUT}/ck_gemma_wnorder.json"), 1)
del gm; torch.cuda.empty_cache()
print("Gemma 済み", len(GD), len(G0P))
""")
md("## 3. 保存")
code("""
res = {"meta": {"gemma": GEMMA_ID, "emb": EMB_ID, "date": time.strftime("%Y-%m-%d"), "n": len(ITEMS),
                "data": "Princeton WordNet 3.0 gloss corpus (noun), tag=man, head by spaCy"},
       "ids": [x[0] for x in ITEMS], "sim": SIM, "gemma": GD, "g0_idx": G0, "gemma_wnorder": G0P}
path = f"{OUT}/E12_results.json.gz"
with gzip.open(path, "wt") as f:
    json.dump(res, f)
print("保存:", path, round(os.path.getsize(path) / 1e6, 2), "MB")
""")
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(dst, "w"), ensure_ascii=False, indent=1)
print("cells", len(cells), "size", os.path.getsize(dst))
