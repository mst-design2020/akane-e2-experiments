"""E8n Colabノート（一覧なしでE7・E8を確かめ直す）。python3 build_notebook_E8n.py REPO_analysis_e8_dir out.ipynb
計画：nolist/PLAN_E8n.md。E8のノート（公開 analysis/e8/build_notebook_E8.py）から変えたのは、頭の名詞の取り方・方法B・打ち消しだけ。"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
E8 = sys.argv[1]
OLD = [{k: x[k] for k in ("id", "ent", "catw")} for x in json.load(open(f"{E8}/../e6/e6_items.json"))]  # 正解は入れない
NEW = [{k: x[k] for k in ("id", "ent", "catw")} for x in json.load(open(f"{E8}/e8_new_items.json"))]
CORE = open(f"{HERE}/e8n_core.py").read()
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md("""
# AKANE E8n：公開済みのE7・E8を、手書きの一覧なしで確かめ直す

このノートは **材料を取るだけ** です（比べは結果のファイルを読んで別に行います）。計画：sims/e10/nolist/PLAN_E8n.md。

- E8と同じ700個＋300個（正解はこのノートに入っていません）。
- 説明文の頭の名詞を、**手で書いた語の一覧を使わずに**取ります（文法で候補を出し、NLIで選ぶ）。
- E8が保存したWiktionaryの読み取り（ドライブの E8_results）をそのまま使うので、ダンプは取り直しません。

## 手順
1. 「ランタイム」→「ランタイムのタイプを変更」→ **L4 GPU**。
2. 鍵マークの `HF_TOKEN` にアクセスを許可（E6・E8と同じ）。
3. 「ランタイム」→「すべてのセルを実行」。ドライブの許可を聞かれたら許可。
4. 終わると、ドライブの `AKANE_アーキテクチャ/E8n_results/E8n_results.json` ができます。
5. 時間の見込み：40〜70分（NLI 20〜40分、Gemma 10〜15分）。途中で切れても、もう一度「すべてのセルを実行」すれば済んだ所は飛ばします。
""")
code("""
import os, json, time, subprocess, sys
import numpy as np
from google.colab import drive, userdata
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/AKANE_アーキテクチャ"
OUT = f"{BASE}/E8n_results"
os.makedirs(OUT, exist_ok=True)
E6_WK = f"{BASE}/E6_results/E6_wiktionary.json"
WKD_PATH = f"{BASE}/E8_results/E8_wiktionary_dump_senses.json"
for p in (E6_WK, WKD_PATH):
    assert os.path.exists(p), "見つかりません: " + p
NLI_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"   # E6〜E8と同じ
GEMMA_ID = "google/gemma-2-2b-it"
EMB_ID = "sentence-transformers/all-mpnet-base-v2"
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], check=False)
""")
code("# 中身（sims/e10/nolist/e8n_core.py と同じ。手書きの語の一覧はありません）\n" + CORE)
code("OLD = json.loads(r'''" + json.dumps(OLD, ensure_ascii=False) + "''')\n"
     "NEW = json.loads(r'''" + json.dumps(NEW, ensure_ascii=False) + "''')\n"
     "WK6 = json.load(open(E6_WK))\n"
     "for it in OLD: it['wk_defs'] = WK6.get(it['ent'], [])\n"
     "for it in NEW: it['wk_defs'] = None   # 新しい問題の物の語はダンプから（E8と同じ）\n"
     "ITEMS = OLD + NEW\n"
     "WKD = json.load(open(WKD_PATH))\n"
     "print(len(OLD), '+', len(NEW), '個  ダンプの語', len(WKD))")
md("## 1. 語の意味の一覧と、頭の名詞の候補（GPUは使わない、5〜15分）")
code("""
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])
lex_path = f"{OUT}/E8n_lexicon.json"
if os.path.exists(lex_path):
    L = json.load(open(lex_path)); print("ドライブから読み込み")
else:
    L = build_lexicon(ITEMS, WKD, nlp)
    json.dump(L, open(lex_path, "w"), ensure_ascii=False)
PAIRS = sorted({(w, k, g) for w in L for k, x in enumerate(L[w]) for g in x["cands"] if g != w and L.get(g)})
print("語:", len(L), " 候補の組:", len(PAIRS))
print("例 person:", [(x["def"][:40], x["cands"]) for x in L.get("person", [])[:4]])
""")
md("## 2. NLIで頭の名詞を選ぶ（GPU、15〜30分）")
code("""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
ntok = AutoTokenizer.from_pretrained(NLI_ID)
nm = AutoModelForSequenceClassification.from_pretrained(NLI_ID).cuda().eval()
lab = {v.lower(): int(k) for k, v in nm.config.id2label.items()}
ORDER = [lab["entailment"], lab["neutral"], lab["contradiction"]]
import hashlib
def nli_probs(P, H, B=64, ck=None):
    key = hashlib.md5("\\n".join(p + "\\t" + h for p, h in zip(P, H)).encode()).hexdigest()
    out = []; t = time.time(); start = 0
    if ck and os.path.exists(ck):
        c = json.load(open(ck))
        if c.get("key") == key:
            out = c["out"]; start = len(out); print("続きから", start)
        else:
            print("前の途中経過は別の組なので使わない")
    for i in range(start, len(P), B):
        enc = ntok(P[i:i + B], H[i:i + B], return_tensors="pt", padding=True, truncation=True, max_length=192).to("cuda")
        with torch.no_grad():
            pr = torch.softmax(nm(**enc).logits.float(), -1)[:, ORDER]
        out += pr.cpu().numpy().round(4).tolist()
        if (i // B) % 300 == 0:
            print(f"{i}/{len(P)}  {time.time() - t:.0f}秒")
            if ck: json.dump({"key": key, "out": out}, open(ck, "w"))
    if ck: json.dump({"key": key, "out": out}, open(ck, "w"))
    return out
SP = nli_probs([premise(w, L[w][k]["def"]) for w, k, g in PAIRS], [hyp(w, g) for w, k, g in PAIRS], ck=f"{OUT}/ck_pairs.json")
SC = {p: s[0] for p, s in zip(PAIRS, SP)}; SCC = {p: s[2] for p, s in zip(PAIRS, SP)}
set_genus(L, SC)
EDGES = edges(L, sorted({it["ent"] for it in ITEMS}))
multi = [(w, k) for w, k in EDGES if len(L[L[w][k]["genus"]]) > 1]
print("段:", len(EDGES), " 意味を選ぶ必要のある段:", len(multi))
print("例 person:", [(x["def"][:40], x["cands"], x["genus"]) for x in L.get("person", [])[:4]])
""")
md("## 3. 意味の選び方 C（前の説明文に近い意味）と D（Gemmaに選ばせる）（GPU、10〜15分）")
code("""
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer(EMB_ID, device="cuda")
texts = sorted({x["def"] for w in L for x in L[w]})
V = emb.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
vec = {t: v for t, v in zip(texts, V)}
CHOICE_C = {}
for w, k in multi:
    g = L[w][k]["genus"]; p = vec[L[w][k]["def"]]
    CHOICE_C[(w, k)] = int(np.argmax([float(p @ vec[x["def"]]) for x in L[g]]))
del emb; torch.cuda.empty_cache()

from transformers import AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), padding_side="left")
gm = AutoModelForCausalLM.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), torch_dtype=torch.float16, device_map="cuda").eval()
NUM = [tok.encode(str(i), add_special_tokens=False)[0] for i in range(1, MAX_DEFS_MID + 1)]
CHOICE_D, DPROB = {}, {}
t = time.time(); B = 16
for i in range(0, len(multi), B):
    batch = multi[i:i + B]
    qs = [choice_prompt(w, L[w][k]["def"], L[w][k]["genus"], [x["def"] for x in L[L[w][k]["genus"]]]) for w, k in batch]
    ps = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs]
    enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad():
        last = gm(**enc, logits_to_keep=1).logits[:, -1, :].float()
    for (w, k), row in zip(batch, last):
        n = len(L[L[w][k]["genus"]]); lg = row[NUM[:n]]
        pr = torch.softmax(lg, -1).cpu().numpy()
        CHOICE_D[(w, k)] = int(pr.argmax()); DPROB[(w, k)] = pr.round(4).tolist()
    if (i // B) % 100 == 0: print(f"{i}/{len(multi)}  {time.time() - t:.0f}秒")
del gm; torch.cuda.empty_cache()
CH = {"C": CHOICE_C, "D": CHOICE_D}
for m in "CD":
    print(m, "最初の意味以外を選んだ割合:", np.mean([v != 0 for v in CH[m].values()]).round(3))
""")
md("## 4. 鎖を作り、NLIで着きを判定して保存（GPU、5〜15分）")
code("""
CHAINS, need_step, need_arr = {}, set(), set()
for it in ITEMS:
    for m in "ACD":
        nodes, links = chain_nodes(L, it["ent"], m, CH.get(m))
        CHAINS[(it["id"], m)] = (sorted(nodes), sorted(links))
        for (w, k) in nodes:
            need_arr.add((w, k, it["catw"]))
            if L[w][k]["genus"]: need_step.add((w, k))
need_step, need_arr = sorted(need_step), sorted(need_arr)
print("段の組:", len(need_step), " 着きの組:", len(need_arr))
A = nli_probs([premise(w, L[w][k]["def"]) for w, k, C in need_arr], [hyp(w, C) for w, k, C in need_arr], ck=f"{OUT}/ck_arrive.json")
res = {"meta": {"nli": NLI_ID, "gemma": GEMMA_ID, "emb": EMB_ID, "dump": "E8_results/E8_wiktionary_dump_senses.json（E8と同じ）",
                "date": time.strftime("%Y-%m-%d"), "n_old": len(OLD), "n_new": len(NEW), "head_noun": "一覧なし（文法の候補をNLIで選ぶ）"},
       "L": L,
       "choice": {m: [[w, k, v] for (w, k), v in CH[m].items()] for m in "CD"},
       "dprob": [[w, k, v] for (w, k), v in DPROB.items()],
       "chains": [[i, m, [list(x) for x in nd], [[list(a), list(b)] for a, b in lk]] for (i, m), (nd, lk) in CHAINS.items()],
       "steps": [[w, k, SC[(w, k, L[w][k]["genus"])], SCC[(w, k, L[w][k]["genus"])]] for w, k in need_step],
       "arrive": [[w, k, C, p[0], p[2]] for (w, k, C), p in zip(need_arr, A)]}
path = f"{OUT}/E8n_results.json"
json.dump(res, open(path, "w"), ensure_ascii=False)
import gzip, shutil
with open(path, "rb") as a, gzip.open(path + ".gz", "wb") as b:
    shutil.copyfileobj(a, b)
print("保存:", path, round(os.path.getsize(path) / 1e6, 1), "MB（圧縮", round(os.path.getsize(path + ".gz") / 1e6, 1), "MB）")
""")
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells), "size", os.path.getsize(sys.argv[2]))
