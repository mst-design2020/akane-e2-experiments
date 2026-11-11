"""E9n Colabノート（E9-1〜E9-3の辞書の確かめを一覧なしで作り直す）。python3 build_notebook_E9n.py REPO_analysis_dir out.ipynb
計画：nolist/PLAN_E9n.md。公開の E9-1（build_notebook_E9-1.py）と E9-1b（小さくする）を1本にし、頭の名詞の取り方だけ E8n と同じ一覧なしに変えた。"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
AN = sys.argv[1]
D = json.load(open(f"{AN}/e4/e42_data.json"))
ENTS = sorted(D["ents"])
CATS = list(D["cats"].values())
CORE = open(f"{HERE}/e8n_core.py").read()
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md("""
# AKANE E9n：E9-1〜E9-3の辞書の確かめを、手書きの一覧なしで作り直す

このノートは **材料を取るだけ** です（比べは結果のファイルを読んで別に行います）。計画：sims/e10/nolist/PLAN_E9n.md。

- 公開のE9-1と同じ物1,681語×46分類に、辞書の鎖（意味はGemmaが選ぶ）を作ります。
- 違いは1つだけ：説明文の頭の名詞を、**手で書いた語の一覧を使わずに**取ります（文法で候補を出し、NLIで選ぶ。E8nと同じ）。
- E8が保存したWiktionaryの読み取り（ドライブの E8_results）をそのまま使います。正解はこのノートに入っていません。

## 手順
1. 「ランタイム」→「ランタイムのタイプを変更」→ **L4 GPU**。
2. 鍵マークの `HF_TOKEN` にアクセスを許可（E8と同じ）。
3. 「ランタイム」→「すべてのセルを実行」。ドライブの許可を聞かれたら許可。
4. 終わると、ドライブの `AKANE_アーキテクチャ/E9n_results/E9n_checker_scores.json` ができます（1〜3MBの小さなファイル）。
5. 時間の見込み：1時間半〜2時間（頭の名詞のNLI 10〜20分、Gemma 15〜25分、着きのNLI 40〜70分）。途中で切れても、もう一度「すべてのセルを実行」すれば済んだ所は飛ばします。
""")
code("""
import os, json, time, subprocess, sys, hashlib, gzip, shutil
import numpy as np
from google.colab import drive, userdata
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/AKANE_アーキテクチャ"
OUT = f"{BASE}/E9n_results"
os.makedirs(OUT, exist_ok=True)
WKD_PATH = f"{BASE}/E8_results/E8_wiktionary_dump_senses.json"
assert os.path.exists(WKD_PATH), "見つかりません: " + WKD_PATH
NLI_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"   # E6〜E9と同じ
GEMMA_ID = "google/gemma-2-2b-it"
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"], check=False)
""")
code("# 中身（sims/e10/nolist/e8n_core.py と同じ。手書きの語の一覧はありません）\n" + CORE)
code("ENTS = json.loads(r'''" + json.dumps(ENTS, ensure_ascii=False) + "''')\n"
     "CATS = json.loads(r'''" + json.dumps(CATS, ensure_ascii=False) + "''')\n"
     "ITEMS = [{'ent': w, 'wk_defs': None} for w in ENTS]   # 物の語の説明文もダンプから（公開のE9-1と同じ）\n"
     "WKD = json.load(open(WKD_PATH))\n"
     "print(len(ENTS), '語', len(CATS), '分類  ダンプの語', len(WKD), ' 説明文のある物の語', round(float(np.mean([w in WKD for w in ENTS])), 3))")
md("## 1. 語の意味の一覧と、頭の名詞の候補（GPUは使わない、5〜15分）")
code("""
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])
lex_path = f"{OUT}/E9n_lexicon.json"
if os.path.exists(lex_path):
    L = json.load(open(lex_path)); print("ドライブから読み込み")
else:
    L = build_lexicon(ITEMS, WKD, nlp)
    json.dump(L, open(lex_path, "w"), ensure_ascii=False)
PAIRS = sorted({(w, k, g) for w in L for k, x in enumerate(L[w]) for g in x["cands"] if g != w and L.get(g)})
print("語:", len(L), " 候補の組:", len(PAIRS))
""")
md("## 2. NLIで頭の名詞を選ぶ（GPU、10〜20分）")
code("""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
ntok = AutoTokenizer.from_pretrained(NLI_ID)
nm = AutoModelForSequenceClassification.from_pretrained(NLI_ID, torch_dtype=torch.float16).cuda().eval()
lab = {v.lower(): int(k) for k, v in nm.config.id2label.items()}
ORDER = [lab["entailment"], lab["contradiction"]]
def nli_raw(P, H, B=128):
    out = np.zeros((len(P), 2), np.float32)
    for i in range(0, len(P), B):
        enc = ntok(P[i:i + B], H[i:i + B], return_tensors="pt", padding=True, truncation=True, max_length=192).to("cuda")
        with torch.no_grad():
            pr = torch.softmax(nm(**enc).logits.float(), -1)[:, ORDER]
        out[i:i + len(pr)] = pr.cpu().numpy()
    return out
def nli_probs(P, H, ck, CH_N=50000):
    \"\"\"控え（ck_*.npy）は入力の組の md5 が合う時だけ使う\"\"\"
    out = np.zeros((len(P), 2), np.float32); t = time.time()
    for s in range(0, len(P), CH_N):
        key = hashlib.md5("\\n".join(p + "\\t" + h for p, h in zip(P[s:s + CH_N], H[s:s + CH_N])).encode()).hexdigest()[:12]
        p = f"{OUT}/{ck}_{s}_{key}.npy"
        if os.path.exists(p):
            out[s:s + CH_N] = np.load(p); continue
        r = nli_raw(P[s:s + CH_N], H[s:s + CH_N]); np.save(p, r); out[s:s + len(r)] = r
        print(f"{ck} {s + len(r)}/{len(P)}  {time.time() - t:.0f}秒")
    return out
SP = nli_probs([premise(w, L[w][k]["def"]) for w, k, g in PAIRS], [hyp(w, g) for w, k, g in PAIRS], "ck_pairs")
SC = {p: float(s[0]) for p, s in zip(PAIRS, SP)}
set_genus(L, SC)
EDGES = edges(L, ENTS)
multi = [(w, k) for w, k in EDGES if len(L[L[w][k]["genus"]]) > 1]
print("段:", len(EDGES), " 意味を選ぶ必要のある段:", len(multi))
for w in ENTS[:3]:   # 例を3つ表示するだけ
    print(w, [(x["def"][:40], x["cands"], x["genus"]) for x in L.get(w, [])[:3]])
""")
md("## 3. 途中の語の意味をGemmaに選ばせる（GPU、15〜25分、途中の控えあり）")
code("""
from transformers import AutoModelForCausalLM
cpath = f"{OUT}/E9n_choice_D.json"
# 控えは (語, 意味の番号, 頭の名詞) で持つ（頭の名詞が変わった時に古い選びを使わないため）
old = {(a, b, g): v for a, b, g, v in json.load(open(cpath))} if os.path.exists(cpath) else {}
CHOICE_D = {(w, k): old[(w, k, L[w][k]["genus"])] for w, k in multi if (w, k, L[w][k]["genus"]) in old}
todo = [x for x in multi if x not in CHOICE_D]
save = lambda: json.dump([[w, k, L[w][k]["genus"], v] for (w, k), v in CHOICE_D.items()], open(cpath, "w"))
print("済み:", len(CHOICE_D), " 残り:", len(todo))
if todo:
    del nm; torch.cuda.empty_cache()
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
            print(f"{i}/{len(todo)}  {time.time() - t:.0f}秒"); save()
    save()
    del gm; torch.cuda.empty_cache()
    nm = AutoModelForSequenceClassification.from_pretrained(NLI_ID, torch_dtype=torch.float16).cuda().eval()
print("最初の意味以外を選んだ割合:", np.mean([v != 0 for v in CHOICE_D.values()]).round(3))
""")
md("## 4. 鎖を作り、NLIで着きを判定する（GPU、40〜70分、途中の控えあり）")
code("""
CHAINS, nodes_all = {}, set()
for w in ENTS:
    nodes, links = chain_nodes(L, w, "D", CHOICE_D)
    CHAINS[w] = (sorted(nodes), sorted(links)); nodes_all |= set(nodes)
NODES = sorted(nodes_all)
nid = {x: i for i, x in enumerate(NODES)}
STEPS = {(w, k): SC[(w, k, L[w][k]["genus"])] for w, k in NODES if L[w][k]["genus"]}
print("点:", len(NODES), " 段:", len(STEPS), " 着きの組:", len(NODES) * len(CATS))
P = [premise(w, L[w][k]["def"]) for w, k in NODES for C in CATS]
H = [hyp(w, C) for w, k in NODES for C in CATS]
AE = nli_probs(P, H, "ck_arrive")[:, 0].reshape(len(NODES), len(CATS))
del P, H
""")
md("## 5. 物の主張ごとの辞書の答えを作って保存（公開の e91_checker.py・E9-1b と同じ計算）")
code("""
cidx = {c: j for j, c in enumerate(CATS)}
out = {'e8D': {}, 'e8one': {}}; score = {'e8D': {}, 'e8one': {}}
for w, (nodes, lk) in CHAINS.items():
    if not nodes:
        continue
    ch = {x: [] for x in nodes}
    for a, b in lk:
        ch[a].append(b)
    roots = [x for x in nodes if x[0] == w]
    for C in CATS:
        j = cidx[C]
        arr = {x: (1.0 if x[0] == C else float(AE[nid[x], j])) for x in nodes}
        memo = {}
        def f(x, d):
            if (x, d) in memo:
                return memo[(x, d)]
            best = arr[x]
            if best < 1.0 and d > 0:
                e = STEPS.get(x, 0.0)
                for c in ch[x]:
                    best = max(best, e * f(c, d - 1))
            memo[(x, d)] = best
            return best
        s6 = max(f(r, DEPTH) for r in roots) if roots else 0.0
        s1 = max(arr[r] for r in roots) if roots else 0.0
        key = f"{w}|{C}"
        score['e8D'][key] = round(s6, 4); score['e8one'][key] = round(s1, 4)
        out['e8D'][key] = int(s6 > 0.5); out['e8one'][key] = int(s1 > 0.5)
ENTS_S = sorted(CHAINS)
small = {'meta': {"nli": NLI_ID, "gemma": GEMMA_ID, "dump": "E8_results/E8_wiktionary_dump_senses.json（E8と同じ）",
                  "date": time.strftime("%Y-%m-%d"), "n_ents": len(ENTS), "cats": CATS,
                  "head_noun": "一覧なし（文法の候補をNLIで選ぶ、e8n_core.py）"},
         'ents': ENTS_S, 'cats': CATS, 'checkers': {}}
for name in out:
    small['checkers'][name] = ''.join(''.join(str(out[name][f"{w}|{C}"]) if f"{w}|{C}" in out[name] else '-' for C in CATS) for w in ENTS_S)
for name in score:
    small['scores_' + name] = [int(round(score[name].get(f"{w}|{C}", -0.001) * 1000)) for w in ENTS_S for C in CATS]
small['n_nodes'] = len(NODES); small['n_steps'] = len(STEPS)
small['choice_first_rate'] = float(np.mean([v == 0 for v in CHOICE_D.values()]))
small['genus'] = {w: [x["genus"] for x in L[w]] for w in ENTS}   # 物の語の頭の名詞（点検用）
p = f"{OUT}/E9n_checker_scores.json"
json.dump(small, open(p, "w"), ensure_ascii=False)
with open(p, "rb") as a, gzip.open(p + ".gz", "wb") as b:
    shutil.copyfileobj(a, b)
print("保存:", p, round(os.path.getsize(p) / 1e6, 2), "MB（圧縮", round(os.path.getsize(p + ".gz") / 1e6, 2), "MB）")
print({k: (v.count('1'), v.count('0'), v.count('-')) for k, v in small['checkers'].items()})
""")
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells), "size", os.path.getsize(sys.argv[2]))
