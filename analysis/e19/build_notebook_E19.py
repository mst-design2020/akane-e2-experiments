"""E19 Colabノート（珍しい語で TADORI が効くか。WordNet×wordfreq×Wiktionary）。
python3 build_notebook_E19.py out.ipynb
計画：sims/e19/PLAN_E19.md。比べは e19_analyze.py をノートに埋め込む。"""
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
CATS = json.load(open(f"{R}/e10/e103/k_items.json"))["cats"]   # 46分類（E4-2・E10-3 K と同じ問いの定義）
AN19 = open(f"{HERE}/e19_analyze.py").read().split("\nif __name__ ==")[0]
md("""
# AKANE E19：AI が知らない珍しい言葉なら、辞書をたどると効くか

E17・E18 で、よく出る言葉では TADORI を足しても Gemma 2B はほとんど良くなりませんでした。AI が知らない **珍しい言葉** でなら効くかを、先に決まりを書いて確かめます。計画：sims/e19/PLAN_E19.md。

- 言葉：WordNet の名詞を、wordfreq の出やすさで「よく出る・中くらい・珍しい」に分け、各 200 語（Wiktionary に見出しがある語）。
- 問い：46 分類それぞれに「X は Y の一種か」。
- 比べる方法：2B だけ／2B に説明文を見せるだけ／TADORI／2B または TADORI／9B（大きな AI）。
- しきい値は「よく出る」段で決めます。手で書いた語の一覧はありません。**ドライブはつなぎません。**

## 手順
1. L4 GPU。2. `HF_TOKEN` の許可。3. 1つ目のセルを流してから、2つ目で Ctrl+F10。時間の見込み：1時間半ほど。
""")
code("""
import os, json, time, subprocess, sys, gzip, collections, re, random
import numpy as np
from google.colab import userdata
# v2（11-14）：ドライブをつながない版（ドライブの許可の窓を押せないため）。材料はノートの中と公開の所から取り、結果は最後のセルに印字する。
OUT = "/content/E19_results"
os.makedirs(OUT, exist_ok=True)
NLI_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"   # E6〜E14 と同じ
GEMMA_ID = "google/gemma-2-2b-it"
GEMMA9_ID = "google/gemma-2-9b-it"
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wordfreq"], check=False)
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
code("CATS = " + repr(CATS) + "   # 46分類：WordNet の番号 → 問いに使う名前（E4-2・E10-3 K と同じ）")
code("""
# 材料を読む：E11-2 の鎖、Wiktionary（E10-1 と同じ読み方）、WordNet、wordfreq
r12 = json.loads(gzip.decompress(base64.b64decode(B112)).decode('utf-8'))
L = r12['L']
CH = {'n': {(w, k): v for w, k, v in r12['choice']}, 'g': {}}
ST = {'n': {(w, k): tuple(v) for w, k, *v in r12['steps']}, 'g': {}}
AR = {(w, k, C): tuple(v) for w, k, C, *v in r12['arrive']}
SC = {}
WKD = load_gz("E19_wkd")
if WKD is None:
    E101 = {}; exec(CORE101, E101)
    dump = "/content/enwiktionary.xml.bz2"
    subprocess.run(["wget", "-c", "-q", "-O", dump, DUMP_URL], check=True)
    print("ダンプ", round(os.path.getsize(dump) / 1e9, 2), "GB"); t = time.time()
    import bz2
    with bz2.open(dump, "rb") as f: WKD = E101["parse_dump"](f)[0]
    print("読み終わり", len(WKD), "語", f"{time.time() - t:.0f}秒"); save_gz(WKD, "E19_wkd")
import nltk
nltk.download("wordnet", quiet=True)
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency
CSYN = {wn.synset(s): name for s, name in CATS.items()}
CNAMES = set(CATS.values())
def gold_of(x):
    # X のどれかの名詞の意味から上位語（instance も）をたどって、その分類に着けば 1
    up = set()
    for s in wn.synsets(x, pos="n"):
        up |= {s} | set(s.closure(lambda z: z.hypernyms() + z.instance_hypernyms()))
    return {name: int(s in up) for s, name in CSYN.items()}
pool = {}
for x in sorted(set(wn.all_lemma_names(pos="n"))):
    if not re.fullmatch(r"[a-z]{3,}", x) or x in CNAMES: continue
    g = gold_of(x)
    if sum(g.values()) == 0: continue
    z = zipf_frequency(x, "en")
    pool[x] = ("common" if z >= 3.0 else "mid" if z >= 1.5 else "rare", round(z, 2), g)
rng = random.Random(0)
ITEMS, PICK = [], {}
for bk in ("common", "mid", "rare"):
    allw = [x for x in pool if pool[x][0] == bk]
    have = [x for x in allw if WKD.get(x)]
    PICK[bk] = sorted(rng.sample(have, min(200, len(have))))
    print(bk, "WordNet の語", len(allw), " Wiktionary に見出し", len(have), f"({len(have) / max(1, len(allw)):.0%})")
    for x in PICK[bk]:
        for s, name in CATS.items():
            ITEMS.append({"id": len(ITEMS), "ent": x, "catw": name, "bucket": bk, "zipf": pool[x][1], "gold": pool[x][2][name]})
print(len(ITEMS), "組  一種の組", sum(it["gold"] for it in ITEMS), "  例", {bk: PICK[bk][:8] for bk in PICK})
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

saved = load_gz("E19_chains")
if saved is not None:
    L = saved["L"]
    CH = {m: {(w, k): v for w, k, v in saved["CH"][m]} for m in "ng"}
    ST = {m: {(w, k): tuple(v) for w, k, *v in saved["ST"][m]} for m in "ng"}
    print("ドライブから読み込み")
else:
    t = time.time()
    grow([it["ent"] for it in ITEMS], "n")
    save_gz({"L": L, "CH": {m: [[w, k, v] for (w, k), v in CH[m].items()] for m in "ng"},
             "ST": {m: [[w, k, *v] for (w, k), v in ST[m].items()] for m in "ng"}}, "E19_chains")
    print(f"鎖 {time.time() - t:.0f}秒")
print("辞書に見出しがある X:", sum(1 for it in ITEMS if L.get(it["ent"])), "/", len(ITEMS))
""")
md("## 3. 最後の段（NLI）と Gemma 2B の答え（GPU、10〜20分）")
code("""
saved = load_gz("E19_arrive")
if saved is not None:
    AR = {(w, k, C): tuple(v) for w, k, C, *v in saved["AR"]}; G2 = saved["G2"]; S = saved["S"]
    print("ドライブから読み込み")
else:
    t = time.time()
    REACH = {it["id"]: reach(L, CH, ST, "n", it["ent"]) for it in ITEMS}
    trip = sorted({(w, k, it["catw"]) for it in ITEMS for (w, k) in REACH[it["id"]] if w != it["catw"] and (w, k, it["catw"]) not in AR})
    if trip: AR.update(zip(trip, nli(trip)))
    print("NLI の着き 新しく", len(trip))
    G2 = yes_no([arrive_prompt(it["ent"], it["catw"], []) for it in ITEMS])
    # S：X の辞書の説明文（最初の3つまで）を見せて聞く（E15 と同じ聞き方）
    S = yes_no([arrive_prompt(it["ent"], it["catw"], [f'- "{it["ent"]}": {d["def"]}' for d in L[it["ent"]][:3]]) for it in ITEMS])
    save_gz({"AR": [[w, k, C, *v] for (w, k, C), v in AR.items()], "G2": G2, "S": S}, "E19_arrive")
    print(f"着きと 2B {time.time() - t:.0f}秒")
print("2B の Yes−No（最初の5組）:", G2[:5], " 説明文あり:", S[:5])
""")
md("## 4. Gemma-2 9B に同じ問いを聞く（参考：大きな AI。GPU、5〜15分）")
code("""
saved = load_gz("E19_g9")
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
    save_gz({"G9": G9, "model": GEMMA9_ID}, "E19_g9")
    print(f"9B {time.time() - t:.0f}秒")
print("9B の Yes−No（最初の5組）:", G9[:5])
""")
md("## 5. 比べ（PLAN_E19.md の決まり。e19_analyze.py をそのまま埋め込み）")
code("# sims/e19/e19_analyze.py と同じ\n" + AN19)
code("""
T = []
for it in ITEMS:
    x, C = it["ent"], it["catw"]
    best = reach(L, CH, ST, "n", x)
    v = {n: best[n][0] * (1.0 if n[0] == C else AR.get((n[0], n[1], C), (0.0,))[0]) for n in best}
    T.append(round(max(v.values()), 5) if v else 0.0)
C19 = {"items": [[it["bucket"], it["ent"], it["catw"], it["zipf"]] for it in ITEMS], "gold": [it["gold"] for it in ITEMS],
       "G2": G2, "G9": G9, "S": S, "T": T,
       "meta": {"nli": NLI_ID, "gemma2b": GEMMA_ID, "gemma9b": GEMMA9_ID, "dump": DUMP_URL, "date": time.strftime("%Y-%m-%d"), "n_wkd": len(WKD)}}
json.dump(C19, open("/content/c19.json", "w"))
res = main("/content/c19.json", "/content/out19.json")
""")
md("## 6. 値を印字する（ドライブをつながない版なので、ここに出た値を読みます）")
code("""
import base64
blob = base64.b64encode(gzip.compress(json.dumps(C19, separators=(",", ":")).encode())).decode()
print("E19_C_BEGIN"); print(blob); print("E19_C_END")
print("E19_OUT_BEGIN"); print(json.dumps({k: v for k, v in res.items() if k != "cs_todo"}, ensure_ascii=False, separators=(",", ":"))); print("E19_OUT_END")
""")
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells), "size", os.path.getsize(sys.argv[1]))
