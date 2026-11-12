"""E10 Colabノートを作る。python3 build_notebook_E10.py e10_items_nogold.json out.ipynb
材料を取るだけ（正解は入れない）。結果はドライブ AKANE_アーキテクチャ/E10_results/ に、10MBより小さい .json.gz で分けて保存。"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(sys.argv[1]))
META = D["meta"]
ITEMS = [[it["dom"], it["ent"], it["cls"], it["stmt"], it.get("rev")] for it in D["items"]]
import hashlib
MD5 = hashlib.md5(json.dumps(ITEMS).encode()).hexdigest()[:10]
ORDER = {d: list(dict.fromkeys(it[1] for it in ITEMS if it[0] == d)) for d in "PDG"}
CORDER = {d: list(dict.fromkeys(it[2] for it in ITEMS if it[0] == d)) for d in "PDG"}
def remake(order, corder, meta):
    """ノートの中で主張の文を作り直す（e10_data.py と同じ書き方）。"""
    cap = lambda s: s[:1].upper() + s[1:]
    art = lambda w: ("an " if w[:1].lower() in "aeiou" else "a ") + w
    out = []
    for x in order["P"]:
        for W in corder["P"]:
            ph = meta["wholes"][W]
            out.append(["P", x, W, f"{cap(x)} is a part of {ph}.", f"{cap(ph.split(' ', 1)[1])} is a part of {art(x)}."])
    for x in order["D"]:
        for l in corder["D"]:
            lb = meta["labels"][l]
            out.append(["D", x, l, f"{cap(x)} is {lb['art']} {lb['name']}.", None])
    for x in order["G"]:
        for g in corder["G"]:
            out.append(["G", x, g, f"{x} is located in {meta['regions'][g]}.", None])
    return out
assert remake(ORDER, CORDER, META) == ITEMS, "作り直しが一致しない"
CORE = open(f"{HERE}/e10_core.py").read()
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md("""
# AKANE E10：辞書の確かめは、別の問い（部分と全体）・別の分野（病気・地理）でも効くか

このノートは **材料を取るだけ** です（比べは結果のファイルを読んで別に行います）。計画：sims/e10/PLAN_E10.md。

- 3つの場面の主張 約4.7万（部分と全体・病気・地理、物×分類の全部の組）。正解はこのノートに入っていません。
- Gemmaに全部の主張を「はい／いいえ」で聞きます（部分と全体は逆向きの文も）。
- Wiktionaryのダンプを取り、名詞と固有名詞（地名）の説明文を読みます。
- E8と同じ鎖（説明文をたどる、途中の語の意味はGemmaが選ぶ）に、「〜の一部」「〜にある」の段を足し、NLIで判定します。

## 手順
1. 「ランタイム」→「ランタイムのタイプを変更」→ **L4 GPU**。
2. 鍵マークの `HF_TOKEN` にアクセスを許可（E8と同じ）。
3. 「ランタイム」→「すべてのセルを実行」。ドライブの許可を聞かれたら許可。
4. 終わると、ドライブの `AKANE_アーキテクチャ/E10_results/` に `E10_*.json.gz`（数個）ができます。最後のセルが一覧と大きさを出します。
5. 時間の見込み：Gemmaの「はい／いいえ」10〜15分、ダンプを取って読む 30〜50分（この間「GPUを使っていません」の警告が出ますが×で閉じてください）、語の一覧 5〜10分、Gemmaの意味選び 15〜25分、NLI 90〜150分。合わせて3〜4時間。
6. 途中で切れても、もう一度「すべてのセルを実行」すれば、済んだ所は飛ばします（途中の控えをドライブに保存しています）。
""")
code("""
import os, json, time, bz2, gzip, subprocess, sys, hashlib
import numpy as np
from google.colab import drive, userdata
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/AKANE_アーキテクチャ"
OUT = f"{BASE}/E10_results"
os.makedirs(OUT, exist_ok=True)
NLI_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"   # E6〜E9と同じ
GEMMA_ID = "google/gemma-2-2b-it"
DUMP_URL = "https://dumps.wikimedia.org/enwiktionary/latest/enwiktionary-latest-pages-articles.xml.bz2"
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"], check=False)
def save_gz(obj, name):
    p = f"{OUT}/{name}.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False)
    print("保存:", name, round(os.path.getsize(p) / 1e6, 2), "MB"); return p
def load_gz(name):
    p = f"{OUT}/{name}.json.gz"
    if not os.path.exists(p): return None
    with gzip.open(p, "rt", encoding="utf-8") as f: return json.load(f)
""")
code("# 中身（sims/e10/e10_core.py と同じ）\n" + CORE)
import inspect as _i
code("META = json.loads(r'''" + json.dumps(META, ensure_ascii=False) + "''')\n"
     "ORDER = json.loads(r'''" + json.dumps(ORDER, ensure_ascii=False) + "''')\n"
     "CORDER = json.loads(r'''" + json.dumps(CORDER, ensure_ascii=False) + "''')\n"
     + _i.getsource(remake) +
     "ITEMS = remake(ORDER, CORDER, META)\n"
     f"assert hashlib.md5(json.dumps(ITEMS).encode()).hexdigest()[:10] == '{MD5}', '主張の作り直しが合わない'\n"
     "ENTS = {d: sorted({it[1] for it in ITEMS if it[0] == d}) for d in 'PDG'}\n"
     "CLASSES = {d: sorted({it[2] for it in ITEMS if it[0] == d}) for d in 'PDG'}\n"
     "print(len(ITEMS), '主張', {d: (len(ENTS[d]), len(CLASSES[d])) for d in 'PDG'})\n"
     "print('確認の印:', hashlib.md5(json.dumps(ITEMS).encode()).hexdigest()[:10])")
md("## 1. Gemmaに全部の主張を「はい／いいえ」で聞く（GPU、10〜15分）")
code("""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
G = load_gz("E10_gemma")
if G is None:
    tok = AutoTokenizer.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), padding_side="left")
    gm = AutoModelForCausalLM.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), torch_dtype=torch.float16, device_map="cuda").eval()
    YES = tok.encode("Yes", add_special_tokens=False)[0]; NO = tok.encode("No", add_special_tokens=False)[0]
    Q = [it[3] for it in ITEMS] + [it[4] for it in ITEMS if it[4]]
    def ask(stmts, B=64):
        out = []; t = time.time()
        for i in range(0, len(stmts), B):
            ps = [tok.apply_chat_template([{"role": "user", "content": gemma_q(s)}], tokenize=False, add_generation_prompt=True) for s in stmts[i:i + B]]
            enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                last = gm(**enc, logits_to_keep=1).logits[:, -1, :].float()
            out += (last[:, YES] - last[:, NO]).cpu().numpy().round(3).tolist()
            if (i // B) % 100 == 0: print(f"{i}/{len(stmts)}  {time.time() - t:.0f}秒")
        return out
    lg = ask(Q)
    n = len(ITEMS)
    rev = iter(lg[n:])
    G = {"logit": lg[:n], "rev": [next(rev) if it[4] else None for it in ITEMS], "model": GEMMA_ID}
    save_gz(G, "E10_gemma")
    del gm; torch.cuda.empty_cache()
lg = np.array(G["logit"])
for d in "PDG":
    m = np.array([it[0] == d for it in ITEMS]); print(d, "はいの割合", round(float((lg[m] > 0).mean()), 3))
""")
md("## 2. Wiktionaryのダンプを取って、名詞と固有名詞を読む（30〜50分、GPUは使わない）")
code("""
wk_path = f"{OUT}/E10_wiktionary_senses.json.gz"
if os.path.exists(wk_path):
    with gzip.open(wk_path, "rt", encoding="utf-8") as f: WKD = json.load(f)
    print("ドライブから読み込み:", len(WKD))
else:
    dump = "/content/enwiktionary.xml.bz2"
    subprocess.run(["wget", "-c", "-q", "--show-progress", "-O", dump, DUMP_URL], check=True)
    print("取れた大きさ:", round(os.path.getsize(dump) / 1e9, 2), "GB")
    t = time.time()
    with bz2.open(dump, "rb") as f:
        WKD = parse_dump(f)
    print("読み終わり:", len(WKD), "語", f"{time.time() - t:.0f}秒")
    with gzip.open(wk_path, "wt", encoding="utf-8") as f: json.dump(WKD, f, ensure_ascii=False)
for w in ["pedal", "pneumonia", "Osaka", "Japan", "Maputo"]:
    k = lookup(WKD, w); print(w, "→", [d for d, lb in WKD.get(k, [])[:3]] if k else "なし")
for d in "PDG":
    print(d, "説明文のある物の割合:", round(float(np.mean([lookup(WKD, w) is not None for w in ENTS[d]])), 3))
""")
md("## 3. 語の意味の一覧（鎖に出てくる語を集める、5〜10分）")
code("""
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])
L = load_gz("E10_lexicon")
if L is None:
    t = time.time()
    L = build_lexicon(ENTS, WKD, nlp)
    print("語:", len(L), f"{time.time() - t:.0f}秒")
    save_gz(L, "E10_lexicon")
del WKD
NEED = set()   # 意味を選ぶ必要のある段 (w,k,g)
def walk(dom, wide):
    for x in ENTS[dom]:
        front, seen = [(x, k) for k in range(len(L.get(x, [])))], set()
        for _ in range(DEPTH):
            nxt = []
            for (w, k) in front:
                for g, typ in next_of(L, w, k, dom, wide):
                    if len(L[g]) > 1: NEED.add((w, k, g))
                    for j in range(len(L[g])):
                        if (g, j) not in seen: seen.add((g, j)); nxt.append((g, j))
            front = nxt
for d in "PDG": walk(d, False)
walk("P", True)
NEED = sorted(NEED)
print("意味を選ぶ必要のある段:", len(NEED))
""")
md("## 4. 途中の語の意味をGemmaに選ばせる（GPU、15〜25分、途中の控えあり）")
code("""
CH = load_gz("E10_choice_D") or []
CHOICE_D = {(w, k, g): v for w, k, g, v in CH}
todo = [x for x in NEED if x not in CHOICE_D]
print("済み:", len(CHOICE_D), " 残り:", len(todo))
if todo:
    tok = AutoTokenizer.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), padding_side="left")
    gm = AutoModelForCausalLM.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), torch_dtype=torch.float16, device_map="cuda").eval()
    NUM = [tok.encode(str(i), add_special_tokens=False)[0] for i in range(1, MAX_DEFS_MID + 1)]
    t = time.time(); B = 16
    for i in range(0, len(todo), B):
        batch = todo[i:i + B]
        qs = [choice_prompt(w, L[w][k]["def"], g, [x["def"] for x in L[g]]) for w, k, g in batch]
        ps = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            last = gm(**enc, logits_to_keep=1).logits[:, -1, :].float()
        for (w, k, g), row in zip(batch, last):
            CHOICE_D[(w, k, g)] = int(row[NUM[:len(L[g])]].argmax())
        if (i // B) % 200 == 0:
            print(f"{i}/{len(todo)}  {time.time() - t:.0f}秒")
            save_gz([[w, k, g, v] for (w, k, g), v in CHOICE_D.items()], "E10_choice_D")
    save_gz([[w, k, g, v] for (w, k, g), v in CHOICE_D.items()], "E10_choice_D")
    del gm; torch.cuda.empty_cache()
print("最初の意味以外を選んだ割合:", round(float(np.mean([v != 0 for v in CHOICE_D.values()])), 3))
""")
md("## 5. 鎖を作り、NLIで判定する（GPU、60〜100分、途中の控えあり）")
code("""
VARIANTS = [("D", False), ("B", False)]
CHAINS = {}
for d in "PDG":
    for m, wide in VARIANTS + ([("D", True)] if d == "P" else []):
        tag = m + ("w" if wide else "")
        for x in ENTS[d]:
            nodes, links = chain_nodes(L, x, d, m, CHOICE_D, wide)
            CHAINS[(d, tag, x)] = (sorted(nodes), sorted(links))
STEPS = sorted({(d, a[0], a[1], b[0], typ) for (d, tag, x), (nd, lk) in CHAINS.items() for a, b, typ in lk})
NODES = {d: sorted({n for (dd, tag, x), (nd, lk) in CHAINS.items() if dd == d for n in nd}) for d in "PDG"}
print("段:", len(STEPS), " 点:", {d: len(v) for d, v in NODES.items()},
      " 着きの組:", sum(len(NODES[d]) * len(CLASSES[d]) * (2 if d == "D" else 1) for d in "PDG"))
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
S = load_gz("E10_steps")
if S is None:
    t = time.time()
    r = nli_probs([premise(w, L[w][k]["def"]) for d, w, k, g, typ in STEPS], [step_hyp(w, g, typ, d) for d, w, k, g, typ in STEPS])
    S = [[d, w, k, g, typ, round(float(p[0]), 4), round(float(p[1]), 4)] for (d, w, k, g, typ), p in zip(STEPS, r)]
    save_gz(S, "E10_steps"); print(f"段 済み {time.time() - t:.0f}秒")
ARR = {}
for d in "PDG":
    nh = 2 if d == "D" else 1
    A = np.zeros((len(NODES[d]), len(CLASSES[d]), nh), np.float32)
    CH_N = 1000; t = time.time()
    for s in range(0, len(NODES[d]), CH_N):
        p = f"{OUT}/E10_arrive_{d}_{s}.npy"
        part = NODES[d][s:s + CH_N]
        if os.path.exists(p) and np.load(p).shape[0] == len(part):
            A[s:s + len(part)] = np.load(p); continue
        P, H = [], []
        for w, k in part:
            for c in CLASSES[d]:
                hs = arrive_hyps(w, c, d, META)
                hs = hs + [hs[0]] * (nh - len(hs))
                for h in hs: P.append(premise(w, L[w][k]["def"])); H.append(h)
        r = nli_probs(P, H)[:, 0].reshape(len(part), len(CLASSES[d]), nh)
        np.save(p, r); A[s:s + len(part)] = r
        print(f"{d} 着き {s + len(part)}/{len(NODES[d])}  {time.time() - t:.0f}秒")
    ARR[d] = A
""")
md("## 6. 結果を小さく分けて保存（どれも10MBより小さく）")
code("""
used = sorted({n[0] for d in "PDG" for n in NODES[d]})
LX = {w: [{"def": x["def"], "labels": x["labels"], "genus": x["genus"], "loc": x["loc"], "locw": x.get("locw", {})} for x in L[w]] for w in used}
save_gz({"meta": {"nli": NLI_ID, "gemma": GEMMA_ID, "dump": DUMP_URL, "date": time.strftime("%Y-%m-%d"),
                  "items_md5": hashlib.md5(json.dumps(ITEMS).encode()).hexdigest()[:10]}, "L": LX}, "E10_L_used")
save_gz([[w, k, g, v] for (w, k, g), v in CHOICE_D.items()], "E10_choice_D")
for d in "PDG":
    ch = {f"{tag}|{x}": [[list(n) for n in nd], [[list(a), list(b), t] for a, b, t in lk]]
          for (dd, tag, x), (nd, lk) in CHAINS.items() if dd == d}
    save_gz({"chains": ch, "nodes": [list(n) for n in NODES[d]], "classes": CLASSES[d],
             "arrive": np.round(ARR[d], 3).tolist()}, f"E10_res_{d}")
print("\\nドライブの E10_results の中身:")
for f in sorted(os.listdir(OUT)):
    if f.endswith(".gz"): print(f, round(os.path.getsize(f"{OUT}/{f}") / 1e6, 2), "MB")
""")
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells), "size", round(os.path.getsize(sys.argv[2]) / 1e6, 2), "MB")
