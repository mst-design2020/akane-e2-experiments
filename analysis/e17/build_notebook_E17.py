"""E17 Colabノート（TADORI を人が点を付けた HyperLex で測る：Gemma 2B・9B と TADORI の材料を取る）。
python3 build_notebook_E17.py out.ipynb
計画：sims/e17/PLAN_E17.md。材料を取るだけ（比べは e17_analyze.py で別に）。"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..")
CORE8 = open(f"{R}/e10/nolist/e8n_core.py").read()
CORE14 = open(f"{R}/e14/e14_core.py").read()
CORE15 = open(f"{R}/e15/e15_core.py").read()
CORE101 = open(f"{R}/e10/e101/e101_core.py").read()
import base64
B112 = base64.b64encode(open(f"{R}/e11/results/E11-2_results.json.gz", "rb").read()).decode()
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md("""
# AKANE E17：TADORI を、人が点を付けたデータ（HyperLex）で測る

今までの「常識の正解」は AI（Claude・Gemini）が付けていました。ここでは、研究者が人に点を付けてもらった公開データ **HyperLex**（「X はどれだけ Y の一種か」を約10人が0〜6点で付けた、名詞2,163組）で測ります。

このノートは **材料を取るだけ** です（比べは結果のファイルを読んで別に行います）。計画：sims/e17/PLAN_E17.md。

- Gemma-2 **2B** に「X は Y の一種ですか」と聞く（今までと同じ聞き方）。
- **TADORI**（辞書の鎖、E8n と同じ一覧なしの作り方、3段）で確かめる。
- 大きな AI の参考として、Gemma-2 **9B** にも同じ問いを聞く。
- 参考として、WordNet で Y が X の上位語かも調べる。
- HyperLex は GitHub の原本からこのノートの中で取ります（データはリポジトリに入れません）。人の点はこのノートの計算には使いません。手で書いた語の一覧もありません。

## 手順
1. 「ランタイム」→「ランタイムのタイプを変更」→ **L4 GPU**。
2. 鍵マークの `HF_TOKEN` にアクセスを許可（今までと同じ）。Gemma-2 9B は Hugging Face で利用の同意が要るかもしれません。
3. 「ランタイム」→「すべてのセルを実行」。**ドライブはつなぎません**（v2）。E11-2 の結果はノートに埋め込み、辞書は Wiktionary のダンプから E10-1 と同じ読み方で作り直します（30〜50分）。
4. 最後のセルに、比べに使う値（2,163組ぶん）が印字されます。ノートを保存すると、その値がドライブのノートに残ります。
5. 時間の見込み：1時間半〜2時間。
""")
code("""
import os, json, time, subprocess, sys, gzip, collections, re, random
import numpy as np
from google.colab import userdata
# v2（11-14）：ドライブをつながない版（ドライブの許可の窓を押せないため）。材料はノートの中と公開の所から取り、結果は最後のセルに印字する。
OUT = "/content/E17_results"
os.makedirs(OUT, exist_ok=True)
NLI_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"   # E6〜E14 と同じ
GEMMA_ID = "google/gemma-2-2b-it"
GEMMA9_ID = "google/gemma-2-9b-it"
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
code("# 中身1（sims/e10/nolist/e8n_core.py と同じ。手書きの語の一覧はありません）\n" + CORE8)
code("# 中身2（sims/e14/e14_core.py と同じ。new_word だけ使います）\n" + CORE14)
code("# 中身3（sims/e15/e15_core.py と同じ）\n" + CORE15)
code("# 材料：E11-2 の結果（E11-2_results.json.gz をそのまま埋め込み）と、E10-1 の辞書の読み方（e101_core.py）\nimport base64\nB112 = " + repr(B112) + "\nCORE101 = " + repr(CORE101) + "\nDUMP_URL = \"https://dumps.wikimedia.org/enwiktionary/latest/enwiktionary-latest-pages-articles.xml.bz2\"")
code("""
# HyperLex（Vulić ほか 2017）を原本から取る。名詞だけ。人の点は保存するが、計算には使わない。
import urllib.request, zipfile, io
HL_URL = "https://raw.githubusercontent.com/ivulic/hyperlex/master/hyperlex-data.zip"
zf = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(HL_URL).read()))
name = [n for n in zf.namelist() if n.lower().endswith("hyperlex-nouns.txt")][0]
rows = [l.split() for l in zf.read(name).decode("utf-8").splitlines()[1:] if l.strip()]
ITEMS = [{"id": i, "ent": r[0], "catw": r[1]} for i, r in enumerate(rows)]
HUMAN = [float(r[4]) for r in rows]
SETS = {"hl": ITEMS}
r12 = json.loads(gzip.decompress(base64.b64decode(B112)).decode('utf-8'))
L = r12['L']
CH = {'n': {(w, k): v for w, k, v in r12['choice']}, 'g': {}}
ST = {'n': {(w, k): tuple(v) for w, k, *v in r12['steps']}, 'g': {}}
AR = {(w, k, C): tuple(v) for w, k, C, *v in r12['arrive']}
SC = {}
WKD = load_gz("E17_wkd")
if WKD is None:
    # E10-1 と同じ読み方（sims/e10/e101/e101_core.py の parse_dump、ほかの中身と名前がぶつからないよう別の場所で動かす）
    E101 = {}; exec(CORE101, E101)
    dump = "/content/enwiktionary.xml.bz2"
    subprocess.run(["wget", "-c", "-q", "-O", dump, DUMP_URL], check=True)
    print("ダンプ", round(os.path.getsize(dump) / 1e9, 2), "GB"); t = time.time()
    import bz2
    with bz2.open(dump, "rb") as f: WKD = E101["parse_dump"](f)[0]
    print("読み終わり", len(WKD), "語", f"{time.time() - t:.0f}秒"); save_gz(WKD, "E17_wkd")
print(len(ITEMS), '組（HyperLex 名詞）  語', len(L), '  辞書の見出し', len(WKD))
print(ITEMS[:3])
""")
md("## 1. NLI と Gemma を読み込む（GPU、2〜4分）")
code("""
import torch, spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
nlp = spacy.load("en_core_web_sm", disable=["ner"])
ntok = AutoTokenizer.from_pretrained(NLI_ID)
nm = AutoModelForSequenceClassification.from_pretrained(NLI_ID).cuda().eval()
lab = {v.lower(): int(k) for k, v in nm.config.id2label.items()}
ORDER = [lab["entailment"], lab["neutral"], lab["contradiction"]]
tok = AutoTokenizer.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), padding_side="left")
gm = AutoModelForCausalLM.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), torch_dtype=torch.float16, device_map="cuda").eval()
NUM = [tok.encode(str(i), add_special_tokens=False)[0] for i in range(1, MAX_DEFS_MID + 1)]
YES = tok.encode("Yes", add_special_tokens=False)[0]; NO = tok.encode("No", add_special_tokens=False)[0]

def _nli(P, H, B=64):
    out = []
    for i in range(0, len(P), B):
        enc = ntok(P[i:i + B], H[i:i + B], return_tensors="pt", padding=True, truncation=True, max_length=192).to("cuda")
        with torch.no_grad():
            pr = torch.softmax(nm(**enc).logits.float(), -1)[:, ORDER].cpu().numpy()
        out += [(round(float(p[0]), 4), round(float(p[2]), 4)) for p in pr]
    return out
def nli(trip):
    # (w, k, c)：前提「"w" means: 説明文」、仮説「W is a type of c.」（E8n と同じ）
    return _nli([premise(w, L[w][k]["def"]) for w, k, c in trip], [hyp(w, c) for w, k, c in trip])
def _pick(qs, ns, B=16):
    out = []
    for i in range(0, len(qs), B):
        ps = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs[i:i + B]]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            last = gm(**enc, logits_to_keep=1).logits[:, -1, :].float()
        for row, n in zip(last, ns[i:i + B]):
            out.append(int(torch.softmax(row[NUM[:n]], -1).argmax()))
    return out
def yes_no(qs, B=16):
    out = []
    for i in range(0, len(qs), B):
        ps = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs[i:i + B]]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            r = gm(**enc, logits_to_keep=1).logits[:, -1, :].float()
        out.extend((r[:, YES] - r[:, NO]).cpu().numpy().round(3).tolist())
    return out
print(genus_prompt("kitten", "A young cat.", ["cat"]))
print(arrive_prompt("kitten", "mammal", ['- "kitten": A young cat.', '- "cat": A small domesticated carnivorous mammal.']))
print(yes_no([arrive_prompt("kitten", "mammal", ['- "kitten": A young cat.', '- "cat": A small domesticated carnivorous mammal.']),
              arrive_prompt("kitten", "mammal", [])]))
""")
md("## 2. 鎖を作る（E8n と同じ作り方、3段まで。GPU、10〜30分）")
code("""
DONE = {"n": set(), "g": set()}
def grow(ents, mode, maxd=MAXD):
    # 物の語から maxd 段まで、まだ親の語を決めていない意味だけ決める。
    # n：候補のうち NLI（説明文 ⇒ W is a type of 候補）が一番の物（E8n と同じ）。E11-2 で決めた物はそのまま使う。
    # g：候補が2つ以上なら Gemma に選ばせる（genus_prompt）。段の強さは同じ NLI の値。
    key = gkey(mode)
    front = sorted(set(ents))
    for w in front: new_word(L, w, WKD, nlp)
    for d in range(maxd):
        todo = []
        for w in front:
            for k, x in enumerate(L.get(w, [])):
                if (w, k) in DONE[mode]: continue
                DONE[mode].add((w, k))
                if mode == "n" and x.get("genus") is not None and (w, k) in ST["n"]: continue   # E11-2 で決めた物
                for c in x["cands"]:
                    if c != w: new_word(L, c, WKD, nlp)
                cs = [c for c in x["cands"] if c != w and L.get(c)]
                if cs: todo.append((w, k, cs))
                else: x[key] = None; CH[mode].pop((w, k), None)
        trip = sorted({(w, k, c) for w, k, cs in todo for c in cs if (w, k, c) not in SC})
        if trip: SC.update(zip(trip, nli(trip)))
        if mode == "n":
            for w, k, cs in todo:
                g = max(cs, key=lambda c: SC[(w, k, c)][0])
                if L[w][k].get("genus") != g: CH["n"].pop((w, k), None)   # 親が替わったら前の意味選びは使わない
                L[w][k]["genus"] = g; ST["n"][(w, k)] = SC[(w, k, g)]
        else:
            many = [(w, k, cs) for w, k, cs in todo if len(cs) > 1]
            pk = _pick([genus_prompt(w, L[w][k]["def"], cs) for w, k, cs in many], [len(cs) for w, k, cs in many]) if many else []
            for (w, k, cs), i in zip(many, pk): L[w][k]["ggenus"] = cs[i]
            for w, k, cs in todo:
                if len(cs) == 1: L[w][k]["ggenus"] = cs[0]
                ST["g"][(w, k)] = SC[(w, k, L[w][k]["ggenus"])]
        # 親の語の意味（2つ以上ある時だけ Gemma、E8n の方法D と同じ聞き方）
        jobs = [(w, k) for w, k, cs in todo if len(L[L[w][k][key]]) > 1 and (w, k) not in CH[mode]]
        if jobs:
            qs = [choice_prompt(w, L[w][k]["def"], L[w][k][key], [x["def"] for x in L[L[w][k][key]]]) for w, k in jobs]
            for (w, k), v in zip(jobs, _pick(qs, [len(L[L[w][k][key]]) for w, k in jobs])): CH[mode][(w, k)] = int(v)
        nxt = set()
        for w in front:
            for k in range(len(L.get(w, []))):
                s = step_from(L, CH, ST, mode, w, k)
                if s: nxt.add(s[0][0])
        print(f"  {mode} 段{d + 1}: 決めた意味 {len(todo)}  NLI {len(trip)}  語 {len(L)}")
        front = sorted(nxt)

saved = load_gz("E17_chains")
if saved is not None:
    L = saved["L"]
    CH = {m: {(w, k): v for w, k, v in saved["CH"][m]} for m in "ng"}
    ST = {m: {(w, k): tuple(v) for w, k, *v in saved["ST"][m]} for m in "ng"}
    print("ドライブから読み込み")
else:
    t = time.time()
    grow([it["ent"] for it in ITEMS], "n")
    save_gz({"L": L, "CH": {m: [[w, k, v] for (w, k), v in CH[m].items()] for m in "ng"},
             "ST": {m: [[w, k, *v] for (w, k), v in ST[m].items()] for m in "ng"}}, "E17_chains")
    print(f"鎖 {time.time() - t:.0f}秒")
print("辞書に見出しがある X:", sum(1 for it in ITEMS if L.get(it["ent"])), "/", len(ITEMS))
""")
md("## 3. 最後の段（NLI）と Gemma 2B の答え（GPU、10〜20分）")
code("""
saved = load_gz("E17_arrive")
if saved is not None:
    AR = {(w, k, C): tuple(v) for w, k, C, *v in saved["AR"]}; G2 = saved["G2"]
    print("ドライブから読み込み")
else:
    t = time.time()
    REACH = {it["id"]: reach(L, CH, ST, "n", it["ent"]) for it in ITEMS}
    trip = sorted({(w, k, it["catw"]) for it in ITEMS for (w, k) in REACH[it["id"]] if w != it["catw"] and (w, k, it["catw"]) not in AR})
    if trip: AR.update(zip(trip, nli(trip)))
    print("NLI の着き 新しく", len(trip))
    G2 = yes_no([arrive_prompt(it["ent"], it["catw"], []) for it in ITEMS])
    save_gz({"AR": [[w, k, C, *v] for (w, k, C), v in AR.items()], "G2": G2}, "E17_arrive")
    print(f"着きと 2B {time.time() - t:.0f}秒")
print("2B の Yes−No（最初の5組）:", G2[:5])
""")
md("## 4. Gemma-2 9B に同じ問いを聞く（参考：大きな AI。GPU、5〜15分）")
code("""
saved = load_gz("E17_g9")
if saved is not None:
    G9 = saved["G9"]; print("ドライブから読み込み")
else:
    import gc
    del gm; gc.collect(); torch.cuda.empty_cache()
    tok = AutoTokenizer.from_pretrained(GEMMA9_ID, token=userdata.get("HF_TOKEN"), padding_side="left")
    gm = AutoModelForCausalLM.from_pretrained(GEMMA9_ID, token=userdata.get("HF_TOKEN"), torch_dtype=torch.bfloat16, device_map="cuda").eval()
    YES = tok.encode("Yes", add_special_tokens=False)[0]; NO = tok.encode("No", add_special_tokens=False)[0]
    t = time.time()
    G9 = yes_no([arrive_prompt(it["ent"], it["catw"], []) for it in ITEMS], B=8)
    save_gz({"G9": G9, "model": GEMMA9_ID}, "E17_g9")
    print(f"9B {time.time() - t:.0f}秒")
print("9B の Yes−No（最初の5組）:", G9[:5])
""")
md("## 5. 参考：WordNet で Y が X の上位語か（CPU、1分）")
code("""
import nltk
nltk.download("wordnet", quiet=True)
from nltk.corpus import wordnet as wn
def wn_ref(x, y):
    xs, ys = wn.synsets(x, pos="n"), set(wn.synsets(y, pos="n"))
    if not xs or not ys: return [None, None]
    up = set()
    for s in xs: up |= set(s.closure(lambda z: z.hypernyms() + z.instance_hypernyms()))
    sim = max((a.path_similarity(b) or 0.0) for a in xs for b in ys)
    return [int(bool(up & ys)), round(float(sim), 4)]
WN = [wn_ref(it["ent"], it["catw"]) for it in ITEMS]
print("WordNet で上位語:", sum(1 for v in WN if v[0] == 1), "/", len(WN))
save_gz({"items": ITEMS, "human": HUMAN, "L": L, "CH": {m: [[w, k, v] for (w, k), v in CH[m].items()] for m in "ng"},
         "ST": {m: [[w, k, *v] for (w, k), v in ST[m].items()] for m in "ng"},
         "AR": [[w, k, C, *v] for (w, k, C), v in AR.items()], "G2": G2, "G9": G9, "WN": WN,
         "meta": {"nli": NLI_ID, "gemma2b": GEMMA_ID, "gemma9b": GEMMA9_ID, "hyperlex": HL_URL, "date": time.strftime("%Y-%m-%d")}}, "E17_results")
print("おわり")
""")
md("## 6. 比べに使う値を印字する（ドライブをつながない版なので、ここに出た値を読みます）")
code("""
T, HAS, PLEN = [], [], []
for it in ITEMS:
    x, C = it["ent"], it["catw"]
    best = reach(L, CH, ST, "n", x)
    v = {n: best[n][0] * (1.0 if n[0] == C else AR.get((n[0], n[1], C), (0.0,))[0]) for n in best}
    T.append(round(max(v.values()), 5) if v else 0.0); HAS.append(int(bool(L.get(x))))
    PLEN.append(len(best[max(v, key=v.get)][1]) - 1 if v else -1)
print("E17_COMPACT_BEGIN")
print(json.dumps({"G2": G2, "G9": G9, "T": T, "has": HAS, "plen": PLEN, "WN": WN,
                  "meta": {"nli": NLI_ID, "gemma2b": GEMMA_ID, "gemma9b": GEMMA9_ID, "hyperlex": HL_URL, "dump": DUMP_URL,
                           "date": time.strftime("%Y-%m-%d"), "n_words": len(L), "n_wkd": len(WKD)}}, separators=(",", ":")))
print("E17_COMPACT_END")
""")
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells), "size", os.path.getsize(sys.argv[1]))
