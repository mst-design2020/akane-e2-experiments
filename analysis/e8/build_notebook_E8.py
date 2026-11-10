"""E8 Colabノートを作る。python3 build_notebook_E8.py out.ipynb"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
OLD = [{k: x[k] for k in ("id", "ent", "catw")} for x in json.load(open(f"{HERE}/../e6/e6_items.json"))]  # 正解は入れない
NEW = [{k: x[k] for k in ("id", "ent", "catw")} for x in json.load(open(f"{HERE}/e8_new_items.json"))]
CORE = open(f"{HERE}/e8_core.py").read()
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md("""
# AKANE E8：鎖の途中の語の意味を選ぶ（取り違えを防ぐ）＋新しい問題

このノートは **材料を取るだけ** です（比べは結果のファイルを読んで別に行います）。計画：sims/e6/PLAN_E8.md。

- いままでの700個＋新しい300個（正解はこのノートに入っていません）。
- Wiktionaryのダンプをもう一度取り、今度は「俗語・古い」などの印を残して読みます。
- 途中の語の意味の選び方を4通り（選ばない／印のない最初の意味／前の説明文に近い意味／Gemmaに選ばせる）で鎖を作り、NLIで判定します。

## 手順
1. 「ランタイム」→「ランタイムのタイプを変更」→ **L4 GPU**。
2. 鍵マークの `HF_TOKEN` にアクセスを許可（E6と同じ）。
3. 「ランタイム」→「すべてのセルを実行」。ドライブの許可を聞かれたら許可。
4. 終わると、ドライブの `AKANE_アーキテクチャ/E8_results/E8_results.json` ができます。
5. 時間の見込み：ダンプを取って読む 25〜45分（この間「GPUを使っていません」の警告が出ますが×で閉じてください）、Gemma・NLI 25〜40分。途中で切れても、もう一度「すべてのセルを実行」すれば済んだ所は飛ばします。
""")
code("""
import os, json, time, bz2, subprocess, sys
import numpy as np
from google.colab import drive, userdata
drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/AKANE_アーキテクチャ/E8_results"
os.makedirs(OUT, exist_ok=True)
E6_WK = "/content/drive/MyDrive/AKANE_アーキテクチャ/E6_results/E6_wiktionary.json"
NLI_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"   # E6・E7と同じ
GEMMA_ID = "google/gemma-2-2b-it"
EMB_ID = "sentence-transformers/all-mpnet-base-v2"
DUMP_URL = "https://dumps.wikimedia.org/enwiktionary/latest/enwiktionary-latest-pages-articles.xml.bz2"
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], check=False)
""")
code("# 中身（sims/e8/e8_core.py と同じ）\n" + CORE)
code("OLD = json.loads(r'''" + json.dumps(OLD, ensure_ascii=False) + "''')\n"
     "NEW = json.loads(r'''" + json.dumps(NEW, ensure_ascii=False) + "''')\n"
     "WK6 = json.load(open(E6_WK))\n"
     "for it in OLD: it['wk_defs'] = WK6.get(it['ent'], [])\n"
     "for it in NEW: it['wk_defs'] = None   # 新しい問題の物の語はダンプから\n"
     "ITEMS = OLD + NEW\n"
     "print(len(OLD), '+', len(NEW), '個')")
md("## 1. Wiktionaryのダンプを取って、印を残して読む（25〜45分、GPUは使わない）")
code("""
wkd_path = f"{OUT}/E8_wiktionary_dump_senses.json"
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
for w in ["person", "cat", "device"]:
    print(w, [(d[:50], lb) for d, lb in WKD.get(w, [])[:6]])
print("新しい問題の物の語で説明文のある割合:", np.mean([it["ent"] in WKD for it in NEW]).round(3))
""")
md("## 2. 語の意味の一覧を作る")
code("""
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])
L = build_lexicon(ITEMS, WKD, nlp)
EDGES = edges(L, sorted({it["ent"] for it in ITEMS}))
multi = [(w, k) for w, k in EDGES if len(L[L[w][k]["genus"]]) > 1]
print("語:", len(L), " 段:", len(EDGES), " 意味を選ぶ必要のある段:", len(multi))
print("person の意味と印:", [(x["def"][:40], x["labels"], x["genus"]) for x in L.get("person", [])])
""")
md("## 3. 意味の選び方 C（前の説明文に近い意味）と D（Gemmaに選ばせる）（GPU、10〜15分）")
code("""
import torch
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

from transformers import AutoTokenizer, AutoModelForCausalLM
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
md("## 4. 鎖を作り、NLIで判定する（GPU、10〜15分）")
code("""
CHAINS, need_step, need_arr = {}, set(), set()
for it in ITEMS:
    for m in "ABCD":
        nodes, links = chain_nodes(L, it["ent"], m, CH.get(m))
        CHAINS[(it["id"], m)] = (sorted(nodes), sorted(links))
        for (w, k) in nodes:
            need_arr.add((w, k, it["catw"]))
            for gw in GROUP_WORDS: need_arr.add((w, k, gw))
            if L[w][k]["genus"]: need_step.add((w, k))
need_step, need_arr = sorted(need_step), sorted(need_arr)
print("段の組:", len(need_step), " 着きの組:", len(need_arr))
from transformers import AutoModelForSequenceClassification
ntok = AutoTokenizer.from_pretrained(NLI_ID)
nm = AutoModelForSequenceClassification.from_pretrained(NLI_ID).cuda().eval()
lab = {v.lower(): int(k) for k, v in nm.config.id2label.items()}
ORDER = [lab["entailment"], lab["neutral"], lab["contradiction"]]
def nli_probs(P, H, B=64):
    out = []; t = time.time()
    for i in range(0, len(P), B):
        enc = ntok(P[i:i + B], H[i:i + B], return_tensors="pt", padding=True, truncation=True, max_length=192).to("cuda")
        with torch.no_grad():
            pr = torch.softmax(nm(**enc).logits.float(), -1)[:, ORDER]
        out += pr.cpu().numpy().round(4).tolist()
        if (i // B) % 300 == 0: print(f"{i}/{len(P)}  {time.time() - t:.0f}秒")
    return out
S = nli_probs([premise(w, L[w][k]["def"]) for w, k in need_step], [hyp(w, L[w][k]["genus"]) for w, k in need_step])
A = nli_probs([premise(w, L[w][k]["def"]) for w, k, C in need_arr], [hyp(w, C) for w, k, C in need_arr])
res = {"meta": {"nli": NLI_ID, "gemma": GEMMA_ID, "emb": EMB_ID, "dump": DUMP_URL, "date": time.strftime("%Y-%m-%d"),
                "n_old": len(OLD), "n_new": len(NEW)},
       "L": L,
       "choice": {m: [[w, k, v] for (w, k), v in CH[m].items()] for m in "CD"},
       "dprob": [[w, k, v] for (w, k), v in DPROB.items()],
       "chains": [[i, m, [list(x) for x in nd], [[list(a), list(b)] for a, b in lk]] for (i, m), (nd, lk) in CHAINS.items()],
       "steps": [[w, k, p[0], p[2]] for (w, k), p in zip(need_step, S)],
       "arrive": [[w, k, C, p[0], p[2]] for (w, k, C), p in zip(need_arr, A)]}
path = f"{OUT}/E8_results.json"
json.dump(res, open(path, "w"), ensure_ascii=False)
print("保存:", path, round(os.path.getsize(path) / 1e6, 1), "MB")
""")
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells))
