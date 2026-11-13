"""E15 Colabノート（鎖の最後の段の判定を、道を見せた Gemma に／親の語を Gemma に選ばせる）。
python3 build_notebook_E15.py out.ipynb
計画：plans/E15_last_step.md。材料を取るだけ（比べは e15_analyze.py で別に）。"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..")
OLD = [{k: x[k] for k in ("id", "ent", "catw")} for x in json.load(open(f"{R}/e6/e6_items.json"))]   # 正解は入れない
NEW = [{k: x[k] for k in ("id", "ent", "catw")} for x in json.load(open(f"{R}/e8/e8_new_items.json"))]
KALL = json.load(open(f"{R}/e10/e103/k_items.json"))["items"]
KCS = json.load(open(f"{R}/e10/e103/cs_K_claude.json"))
KIT = [{"id": int(i), "ent": KALL[int(i)]["ent"], "catw": KALL[int(i)]["cname"]} for i in KCS]   # 正解は入れない
CORE8 = open(f"{R}/e10/nolist/e8n_core.py").read()
CORE14 = open(f"{R}/e14/e14_core.py").read()
CORE15 = open(f"{HERE}/e15_core.py").read()
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md("""
# AKANE E15：鎖の最後の段の判定と、親の語の選び方

E14 で、正しいのに「いいえ」と言っていた問題の多くは、**正しい途中の語（例 子猫→猫、妻→人間）まで着いているのに、最後の「分類に着く」判定で落ちていた**ことが分かりました。今の判定は、その語の辞書の説明文1つで NLI に聞いています（人間の説明文に「哺乳類」と書いてないと落ちる）。

このノートは **材料を取るだけ** です（比べは結果のファイルを読んで別に行います）。計画：plans/E15_last_step.md。

- **最後の段**：たどった道の説明文を全部見せて、Gemma に「この説明文から、x は C の一種ですか」と聞きます。対照として、道の最後の説明文だけ・関係のない別の問題の道・説明文なし、でも聞きます。
- **親の語**：説明文の中のどの語が「〇〇の一種」の〇〇に当たるかを、Gemma に選ばせます（今までは NLI の値で選んでいた）。
- 問題は2組：E8 の 1,000問（直し方を見る）と、E10-3 の「一種か」600問（確かめ。501語×46分類から常識で採点した600問）。
- 正解はこのノートに入っていません。手で書いた語の一覧もありません。

## 手順
1. 「ランタイム」→「ランタイムのタイプを変更」→ **L4 GPU**。
2. 鍵マークの `HF_TOKEN` にアクセスを許可（今までと同じ）。
3. 「ランタイム」→「すべてのセルを実行」。ドライブの許可を聞かれたら許可。
4. 終わると、ドライブの `AKANE_アーキテクチャ/E15_results/E15_results.json.gz` ができます。
5. 時間の見込み：40〜70分。途中で切れても、もう一度「すべてのセルを実行」すれば、済んだ所は飛ばします。
""")
code("""
import os, json, time, subprocess, sys, gzip, collections, re, random
import numpy as np
from google.colab import drive, userdata
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/AKANE_アーキテクチャ"
OUT = f"{BASE}/E15_results"
os.makedirs(OUT, exist_ok=True)
E112 = f"{BASE}/E11-2_results/E11-2_results.json"
WIKT = f"{BASE}/E10-1_results/E101_wiktionary.json.gz"   # E10-1 で読んだ名詞の辞書
for p in (E112, WIKT):
    assert os.path.exists(p), "見つかりません: " + p
NLI_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"   # E6〜E14 と同じ
GEMMA_ID = "google/gemma-2-2b-it"
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
code("# 中身1（analysis/nolist/e8n_core.py と同じ。手書きの語の一覧はありません）\n" + CORE8)
code("# 中身2（analysis/e14/e14_core.py と同じ。new_word だけ使います）\n" + CORE14)
code("# 中身3（analysis/e15/e15_core.py と同じ）\n" + CORE15)
code("OLD = json.loads(r'''" + json.dumps(OLD, ensure_ascii=False) + "''')\n"
     "NEW = json.loads(r'''" + json.dumps(NEW, ensure_ascii=False) + "''')\n"
     "KIT = json.loads(r'''" + json.dumps(KIT, ensure_ascii=False) + "''')\n"
     "SETS = {'e8': OLD + NEW, 'k': KIT}\n"
     "r12 = json.load(open(E112))\n"
     "L = r12['L']\n"
     "CH = {'n': {(w, k): v for w, k, v in r12['choice']}, 'g': {}}\n"
     "ST = {'n': {(w, k): tuple(v) for w, k, *v in r12['steps']}, 'g': {}}\n"
     "AR = {(w, k, C): tuple(v) for w, k, C, *v in r12['arrive']}\n"
     "with gzip.open(WIKT, 'rt', encoding='utf-8') as f: WKD = json.load(f)['n']\n"
     "print(len(SETS['e8']), '問（E8）', len(SETS['k']), '問（E10-3 の一種か）  語', len(L), '  辞書の見出し', len(WKD))")
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
md("## 2. 鎖を作る（親の語：NLI の作り方 n と Gemma の作り方 g、3段まで。GPU、10〜20分）")
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
                else: x[key] = None
        trip = sorted({(w, k, c) for w, k, cs in todo for c in cs if (w, k, c) not in SC})
        if trip: SC.update(zip(trip, nli(trip)))
        if mode == "n":
            for w, k, cs in todo:
                g = max(cs, key=lambda c: SC[(w, k, c)][0]); L[w][k]["genus"] = g; ST["n"][(w, k)] = SC[(w, k, g)]
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

saved = load_gz("E15_chains")
if saved is not None:
    L = saved["L"]
    CH = {m: {(w, k): v for w, k, v in saved["CH"][m]} for m in "ng"}
    ST = {m: {(w, k): tuple(v) for w, k, *v in saved["ST"][m]} for m in "ng"}
    print("ドライブから読み込み")
else:
    SC = {}
    t = time.time()
    grow([it["ent"] for it in SETS["k"]], "n")
    grow([it["ent"] for s in SETS.values() for it in s], "g")
    save_gz({"L": L, "CH": {m: [[w, k, v] for (w, k), v in CH[m].items()] for m in "ng"},
             "ST": {m: [[w, k, *v] for (w, k), v in ST[m].items()] for m in "ng"}}, "E15_chains")
    print(f"鎖 {time.time() - t:.0f}秒")
same = sum(1 for w in L for x in L[w] if x.get("ggenus") and x.get("ggenus") == x.get("genus"))
both = sum(1 for w in L for x in L[w] if x.get("ggenus") and x.get("genus"))
print("親の語が NLI と Gemma で同じ:", same, "/", both)
""")
md("## 3. 最後の段を聞く（NLI と Gemma、GPU、20〜40分）")
code("""
saved = load_gz("E15_arrive")
if saved is not None:
    AR = {(w, k, C): tuple(v) for w, k, C, *v in saved["AR"]}; GA = saved["GA"]
    print("ドライブから読み込み")
else:
    t = time.time()
    REACH = {}
    for sn, its in SETS.items():
        for it in its:
            for mode in "ng":
                REACH[(sn, it["id"], mode)] = reach(L, CH, ST, mode, it["ent"])
    # NLI の着き（今までのやり方）：まだ無い (w,k,C) だけ
    trip = sorted({(w, k, it["catw"]) for sn, its in SETS.items() for it in its for mode in "ng"
                   for (w, k) in REACH[(sn, it["id"], mode)] if w != it["catw"] and (w, k, it["catw"]) not in AR})
    if trip: AR.update(zip(trip, nli(trip)))
    print("NLI の着き 新しく", len(trip))
    # Gemma の着き：path（道を全部）・last（道の最後の説明文だけ）・ctrl（関係のない別の問題の、同じ長さの道）・none（説明文なし、1問1回）
    rng = random.Random(15)
    Q, KEYS = [], []
    for sn, its in SETS.items():
        pool = collections.defaultdict(list)
        for it in its:
            for mode in "ng":
                for n, (p, path) in REACH[(sn, it["id"], mode)].items():
                    pool[len(path)].append((it["ent"], path))
        for it in its:
            x, C = it["ent"], it["catw"]
            Q.append(arrive_prompt(x, C, [])); KEYS.append([sn, it["id"], "-", "none", "", -1])
            for mode in "ng":
                for n, (p, path) in REACH[(sn, it["id"], mode)].items():
                    Q.append(arrive_prompt(x, C, path_lines(L, path))); KEYS.append([sn, it["id"], mode, "path", n[0], n[1]])
                    Q.append(arrive_prompt(x, C, path_lines(L, path[-1:]))); KEYS.append([sn, it["id"], mode, "last", n[0], n[1]])
                    other = [q for e, q in pool[len(path)] if e != x] or [q for ln in pool for e, q in pool[ln] if e != x]
                    Q.append(arrive_prompt(x, C, path_lines(L, rng.choice(other)))); KEYS.append([sn, it["id"], mode, "ctrl", n[0], n[1]])
    # 同じ聞き方は1回だけ聞く
    uq = sorted(set(Q)); print("Gemma に聞く", len(Q), "（重なりを除いて", len(uq), "）")
    V = dict(zip(uq, yes_no(uq)))
    GA = [k + [V[q]] for k, q in zip(KEYS, Q)]
    save_gz({"AR": [[w, k, C, *v] for (w, k, C), v in AR.items()], "GA": GA}, "E15_arrive")
    print(f"着き {time.time() - t:.0f}秒")
save_gz({"items": SETS, "L": L, "CH": {m: [[w, k, v] for (w, k), v in CH[m].items()] for m in "ng"},
         "ST": {m: [[w, k, *v] for (w, k), v in ST[m].items()] for m in "ng"},
         "AR": [[w, k, C, *v] for (w, k, C), v in AR.items()], "GA": GA}, "E15_results")
ex = [g for g in GA if g[0] == "e8" and g[3] == "path"][:5]
for g in ex: print(g)
""")
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells), "size", os.path.getsize(sys.argv[1]))
