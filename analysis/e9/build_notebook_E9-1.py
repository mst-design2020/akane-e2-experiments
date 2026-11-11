"""E9-1 Colabノートを作る。python3 build_notebook_E9-1.py out.ipynb
E8の辞書の鎖（意味はGemmaが選ぶ＝D）を、E4-2の物1,681語すべてに作り、鎖の点から46分類すべてへの着きをNLIで測る。正解は入れない。"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(f"{HERE}/../e4/e42_data.json"))
ENTS = sorted(D["ents"])
CATS = list(D["cats"].values())
CORE = open(f"{HERE}/../e8/e8_core.py").read()
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md("""
# AKANE E9-1：辞書の確かめを、物1,681語ぜんぶの分類46個に広げる

このノートは **材料を取るだけ** です（比べは結果のファイルを読んで別に行います）。計画：sims/e9/PLAN_E9-1.md。

- E8と同じ辞書の鎖（Wiktionaryの説明文をたどる。途中の語の意味はGemmaが選ぶ）を、E4-2の物1,681語すべてに作ります。
- 鎖の点から46分類すべてへ「〜は〜の一種か」をNLIで判定します。正解はこのノートに入っていません。

## 手順
1. 「ランタイム」→「ランタイムのタイプを変更」→ **L4 GPU**。
2. 鍵マークの `HF_TOKEN` にアクセスを許可（E8と同じ）。
3. 「ランタイム」→「すべてのセルを実行」。ドライブの許可を聞かれたら許可。
4. 終わると、ドライブの `AKANE_アーキテクチャ/E9-1_results/E9-1_results.json` ができます。
5. 時間の見込み：Gemmaの意味選び 15〜25分、NLI 40〜70分。E8のダンプの控えがドライブにあれば、ダンプは取り直しません。
6. 途中で切れても、もう一度「すべてのセルを実行」すれば、済んだ所は飛ばします（途中の控えをドライブに保存しています）。
""")
code("""
import os, json, time, bz2, subprocess, sys
import numpy as np
from google.colab import drive, userdata
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/AKANE_アーキテクチャ"
OUT = f"{BASE}/E9-1_results"
os.makedirs(OUT, exist_ok=True)
NLI_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"   # E6〜E8と同じ
GEMMA_ID = "google/gemma-2-2b-it"
DUMP_URL = "https://dumps.wikimedia.org/enwiktionary/latest/enwiktionary-latest-pages-articles.xml.bz2"
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"], check=False)
""")
code("# 中身（sims/e8/e8_core.py と同じ）\n" + CORE)
code("ENTS = json.loads(r'''" + json.dumps(ENTS, ensure_ascii=False) + "''')\n"
     "CATS = json.loads(r'''" + json.dumps(CATS, ensure_ascii=False) + "''')\n"
     "ITEMS = [{'ent': w, 'wk_defs': None} for w in ENTS]   # 物の語の説明文もダンプから\n"
     "print(len(ENTS), '語', len(CATS), '分類')")
md("## 1. Wiktionaryの意味（E8の控えを使う。なければダンプを取る：25〜45分）")
code("""
cands = [f"{OUT}/E9-1_wiktionary_dump_senses.json", f"{BASE}/E8_results/E8_wiktionary_dump_senses.json"]
got = [p for p in cands if os.path.exists(p)]
if got:
    WKD = json.load(open(got[0])); print("ドライブから読み込み:", got[0], len(WKD))
else:
    dump = "/content/enwiktionary.xml.bz2"
    subprocess.run(["wget", "-c", "-q", "--show-progress", "-O", dump, DUMP_URL], check=True)
    t = time.time()
    with bz2.open(dump, "rb") as f:
        WKD = parse_dump(f)
    print("読み終わり:", len(WKD), "語", f"{time.time() - t:.0f}秒")
    json.dump(WKD, open(cands[0], "w"), ensure_ascii=False)
print("説明文のある物の語の割合:", np.mean([w in WKD for w in ENTS]).round(3))
""")
md("## 2. 語の意味の一覧と、意味を選ぶ必要のある段")
code("""
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])
L = build_lexicon(ITEMS, WKD, nlp)
EDGES = edges(L, ENTS)
multi = [(w, k) for w, k in EDGES if len(L[L[w][k]["genus"]]) > 1]
print("語:", len(L), " 段:", len(EDGES), " 意味を選ぶ必要のある段:", len(multi))
""")
md("## 3. 途中の語の意味をGemmaに選ばせる（GPU、15〜25分、途中の控えあり）")
code("""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
cpath = f"{OUT}/E9-1_choice_D.json"
CHOICE_D = {tuple(k[:2]): k[2] for k in json.load(open(cpath))} if os.path.exists(cpath) else {}
todo = [x for x in multi if x not in CHOICE_D]
print("済み:", len(CHOICE_D), " 残り:", len(todo))
if todo:
    tok = AutoTokenizer.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), padding_side="left")
    gm = AutoModelForCausalLM.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), torch_dtype=torch.float16, device_map="cuda").eval()
    NUM = [tok.encode(str(i), add_special_tokens=False)[0] for i in range(1, MAX_DEFS_MID + 1)]
    t = time.time(); B = 16
    for i in range(0, len(todo), B):
        batch = todo[i:i + B]
        qs = [choice_prompt(w, L[w][k]["def"], L[w][k]["genus"], [x["def"] for x in L[L[w][k]["genus"]]]) for w, k in batch]
        ps = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            last = gm(**enc, logits_to_keep=1).logits[:, -1, :].float()
        for (w, k), row in zip(batch, last):
            n = len(L[L[w][k]["genus"]])
            CHOICE_D[(w, k)] = int(row[NUM[:n]].argmax())
        if (i // B) % 100 == 0:
            print(f"{i}/{len(todo)}  {time.time() - t:.0f}秒")
            json.dump([[w, k, v] for (w, k), v in CHOICE_D.items()], open(cpath, "w"))
    json.dump([[w, k, v] for (w, k), v in CHOICE_D.items()], open(cpath, "w"))
    del gm; torch.cuda.empty_cache()
print("最初の意味以外を選んだ割合:", np.mean([v != 0 for v in CHOICE_D.values()]).round(3))
""")
md("## 4. 鎖を作り、NLIで判定する（GPU、40〜70分、途中の控えあり）")
code("""
CHAINS, nodes_all = {}, set()
for w in ENTS:
    nodes, links = chain_nodes(L, w, "D", CHOICE_D)
    CHAINS[w] = (sorted(nodes), sorted(links)); nodes_all |= set(nodes)
NODES = sorted(nodes_all)
STEPS = sorted({(w, k) for w, k in NODES if L[w][k]["genus"]})
print("点:", len(NODES), " 段:", len(STEPS), " 着きの組:", len(NODES) * len(CATS))
from transformers import AutoModelForSequenceClassification
ntok = AutoTokenizer.from_pretrained(NLI_ID)
nm = AutoModelForSequenceClassification.from_pretrained(NLI_ID, torch_dtype=torch.float16).cuda().eval()
lab = {v.lower(): int(k) for k, v in nm.config.id2label.items()}
ORDER = [lab["entailment"], lab["contradiction"]]
def nli_probs(P, H, B=128):
    out = np.zeros((len(P), 2), np.float32)
    for i in range(0, len(P), B):
        enc = ntok(P[i:i + B], H[i:i + B], return_tensors="pt", padding=True, truncation=True, max_length=192).to("cuda")
        with torch.no_grad():
            pr = torch.softmax(nm(**enc).logits.float(), -1)[:, ORDER]
        out[i:i + len(pr)] = pr.cpu().numpy()
    return out
S = nli_probs([premise(w, L[w][k]["def"]) for w, k in STEPS], [hyp(w, L[w][k]["genus"]) for w, k in STEPS])
print("段 済み")
# 着き：点ごとに46分類。2,000点ずつ控えを保存
CH_N = 2000
A = np.zeros((len(NODES), len(CATS), 2), np.float32)
t = time.time()
for s in range(0, len(NODES), CH_N):
    p = f"{OUT}/E9-1_arrive_{s}.npy"
    part = NODES[s:s + CH_N]
    if os.path.exists(p) and np.load(p).shape[0] == len(part):
        A[s:s + len(part)] = np.load(p); continue
    P = [premise(w, L[w][k]["def"]) for w, k in part for C in CATS]
    H = [hyp(w, C) for w, k in part for C in CATS]
    r = nli_probs(P, H).reshape(len(part), len(CATS), 2)
    np.save(p, r); A[s:s + len(part)] = r
    print(f"着き {s + len(part)}/{len(NODES)}  {time.time() - t:.0f}秒")
res = {"meta": {"nli": NLI_ID, "gemma": GEMMA_ID, "dump": DUMP_URL, "date": time.strftime("%Y-%m-%d"),
                "n_ents": len(ENTS), "cats": CATS},
       "L": L,
       "choice_D": [[w, k, v] for (w, k), v in CHOICE_D.items()],
       "chains": {w: [[list(x) for x in nd], [[list(a), list(b)] for a, b in lk]] for w, (nd, lk) in CHAINS.items()},
       "steps": [[w, k, round(float(p[0]), 4), round(float(p[1]), 4)] for (w, k), p in zip(STEPS, S)],
       "nodes": [list(x) for x in NODES],
       "arrive_e": np.round(A[:, :, 0], 4).tolist(),
       "arrive_c": np.round(A[:, :, 1], 4).tolist()}
path = f"{OUT}/E9-1_results.json"
json.dump(res, open(path, "w"), ensure_ascii=False)
print("保存:", path, round(os.path.getsize(path) / 1e6, 1), "MB")
""")
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells))
