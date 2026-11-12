"""E11-1 Colabノート（物の語の「普通の意味」をGemmaに選ばせる）。python3 build_notebook_E11-1.py out.ipynb
計画：plans/E11-1_usual_sense.md。材料を取るだけ（比べは e111_analyze.py で別に）。"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..")
OLD = [{k: x[k] for k in ("id", "ent", "catw")} for x in json.load(open(f"{R}/e6/e6_items.json"))]  # 正解は入れない
NEW = [{k: x[k] for k in ("id", "ent", "catw")} for x in json.load(open(f"{R}/e8/e8_new_items.json"))]
CORE = open(f"{HERE}/e111_core.py").read()
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md("""
# AKANE E11-1：物の語そのものの「普通の意味」を、Gemmaに選ばせる

このノートは **材料を取るだけ** です（比べは結果のファイルを読んで別に行います）。計画：plans/E11-1_usual_sense.md。

- ① WordNet の名詞 2,000語で、文なしで「一番よく使う意味」を選ばせ、SemCor の回数（人が意味を付けた文で数えた回数）と照らします。
- ② E8n の物の語（約900語）で、同じ聞き方で意味を選ばせます。NLI と鎖は E8n の結果をそのまま使うので、流し直しません。
- ②の正解（常識の判定）はこのノートに入っていません。①の正解（SemCor の回数）はノートの中で数えますが、Gemma には見せません（候補はくじで並べ替えて見せます）。
- 手で書いた語の一覧はありません。

## 手順
1. 「ランタイム」→「ランタイムのタイプを変更」→ **L4 GPU**。
2. 鍵マークの `HF_TOKEN` にアクセスを許可（E6〜E10と同じ）。
3. 「ランタイム」→「すべてのセルを実行」。ドライブの許可を聞かれたら許可。
4. 終わると、ドライブの `AKANE_アーキテクチャ/E11-1_results/E11-1_results.json` ができます。
5. 時間の見込み：20〜35分。途中で切れても、もう一度「すべてのセルを実行」すれば済んだ所は飛ばします。
""")
code("""
import os, json, time, subprocess, sys
import numpy as np
from google.colab import drive, userdata
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/AKANE_アーキテクチャ"
OUT = f"{BASE}/E11-1_results"
os.makedirs(OUT, exist_ok=True)
E8N = f"{BASE}/E8n_results/E8n_results.json"
assert os.path.exists(E8N), "見つかりません: " + E8N
GEMMA_ID = "google/gemma-2-2b-it"   # E6〜E10と同じ
EMB_ID = "sentence-transformers/all-mpnet-base-v2"   # E8と同じ
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers", "nltk"], check=False)
""")
code("# 中身（analysis/e11/e111_core.py と同じ。手書きの語の一覧はありません）\n" + CORE)
code("OLD = json.loads(r'''" + json.dumps(OLD, ensure_ascii=False) + "''')\n"
     "NEW = json.loads(r'''" + json.dumps(NEW, ensure_ascii=False) + "''')\n"
     "ITEMS = OLD + NEW\n"
     "L = json.load(open(E8N))['L']\n"
     "print(len(OLD), '+', len(NEW), '問  語の意味の一覧', len(L))")
md("## 1. WordNet の名詞と SemCor の回数（GPUは使わない、1〜3分）")
code("""
import nltk
nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
from nltk.corpus import wordnet as wn
WN = semcor_items(wn, n=2000, seed=0)
print("①の語:", len(WN), " 例:", WN[0]["w"], [d[:40] for d in WN[0]["defs"]][:3])
""")
md("## 2. 語と説明文の近さ（S1、GPU、1〜2分）")
code("""
import torch
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer(EMB_ID, device="cuda")
S1 = []
for it in WN:
    v = emb.encode([it["w"]] + it["defs"], normalize_embeddings=True, show_progress_bar=False)
    S1.append((v[1:] @ v[0]).round(4).tolist())
del emb; torch.cuda.empty_cache()
print("S1 済み", len(S1))
""")
md("## 3. Gemma に意味を選ばせる（①・②・参考R4、GPU、15〜30分）")
code("""
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), padding_side="left")
gm = AutoModelForCausalLM.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), torch_dtype=torch.float16, device_map="cuda").eval()
NUM = [tok.encode(str(i), add_special_tokens=False)[0] for i in range(1, MAX_SENSES + 1)]
assert len(set(NUM)) == MAX_SENSES

def ask(prompts, sizes, ck, B=16):
    if os.path.exists(ck):
        c = json.load(open(ck))
        if c.get("n") == len(prompts) and c.get("head") == prompts[0][:200]:
            print("ドライブから読み込み", ck); return c["out"]
    out = []; t = time.time()
    for i in range(0, len(prompts), B):
        ps = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in prompts[i:i + B]]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            last = gm(**enc, logits_to_keep=1).logits[:, -1, :].float()
        for row, n in zip(last, sizes[i:i + B]):
            out.append(torch.softmax(row[NUM[:n]], -1).cpu().numpy().round(4).tolist())
        if (i // B) % 100 == 0: print(f"{i}/{len(prompts)}  {time.time() - t:.0f}秒")
    json.dump({"n": len(prompts), "head": prompts[0][:200], "out": out}, open(ck, "w"))
    return out

# ① WordNet：くじの順とその逆順の2通り
J1 = jobs_usual([(it["w"], it["defs"]) for it in WN], seed=0)
P1 = ask([j[0] for j in J1], [len(j[1]) for j in J1], f"{OUT}/ck_wn.json")
S2 = merge(J1, P1, len(WN))
# ② E8n の物の語（意味が2つ以上の物だけ）
ENTS = sorted({it["ent"] for it in ITEMS if len(L.get(it["ent"], [])) >= 2})
J2 = jobs_usual([(w, [x["def"] for x in L[w]]) for w in ENTS], seed=0)
P2 = ask([j[0] for j in J2], [len(j[1]) for j in J2], f"{OUT}/ck_ent.json")
R2 = dict(zip(ENTS, merge(J2, P2, len(ENTS))))
# 参考 R4：分類を見せて選ばせる（主張には使わない）
IT4 = [it for it in ITEMS if len(L.get(it["ent"], [])) >= 2]
J4 = jobs_leak([(it["ent"], [x["def"] for x in L[it["ent"]]], it["catw"]) for it in IT4], seed=0)
P4 = ask([j[0] for j in J4], [len(j[1]) for j in J4], f"{OUT}/ck_leak.json")
R4 = {it["id"]: p for it, p in zip(IT4, merge(J4, P4, len(IT4)))}
del gm; torch.cuda.empty_cache()
print("S2 一番上に並べた候補を選んだ割合（くじの順の1通り目、並べ順の癖の目安）:", np.mean([int(np.argmax(p)) == 0 for p in P1[0::2]]).round(3))
print("R2 Wiktionaryの最初の意味を選んだ割合:", np.mean([int(np.argmax(v)) == 0 for v in R2.values()]).round(3))
""")
md("## 4. 保存")
code("""
res = {"meta": {"gemma": GEMMA_ID, "emb": EMB_ID, "e8n": "E8n_results/E8n_results.json", "date": time.strftime("%Y-%m-%d"),
                "wordnet": "nltk wordnet 3.0, lemma.count()", "n_wn": len(WN), "n_ent": len(ENTS)},
       "wn": [dict(it, s1=s1, s2=s2) for it, s1, s2 in zip(WN, S1, S2)],
       "ent_p": R2, "leak_p": {str(k): v for k, v in R4.items()}}
path = f"{OUT}/E11-1_results.json"
json.dump(res, open(path, "w"), ensure_ascii=False)
print("保存:", path, round(os.path.getsize(path) / 1e6, 2), "MB")
""")
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells), "size", os.path.getsize(sys.argv[1]))
