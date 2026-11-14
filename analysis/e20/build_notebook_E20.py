"""E20 Colabノート（説明文に答えが書いていない所で鎖が効くか。小さいAI 3つ、WordNet と SemEval-2018）。
python3 build_notebook_E20.py out.ipynb
計画：sims/e20/PLAN_E20.md。比べは e20_analyze.py をノートに埋め込む。"""
import json, os, sys, base64
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..")
CORE8 = open(f"{R}/e10/nolist/e8n_core.py").read()
CORE14 = open(f"{R}/e14/e14_core.py").read()
CORE15 = open(f"{R}/e15/e15_core.py").read()
CORE101 = open(f"{R}/e10/e101/e101_core.py").read()
B112 = base64.b64encode(open(f"{R}/e11/results/E11-2_results.json.gz", "rb").read()).decode()
CATS = json.load(open(f"{R}/e10/e103/k_items.json"))["cats"]
AN20 = open(f"{HERE}/e20_analyze.py").read().split("\nif __name__ ==")[0]
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md("""
# AKANE E20：説明文に答えが書いていない所で、辞書の鎖は効くか

E19 で、AI が知らない言葉には辞書の説明文を見せるのが大きく効きました。ただ、鎖（TADORI）の上乗せは小さく、同着でした。
E20 では、**説明文に分類の語が書いていない組**（1段では届かない所）で、鎖を足すと良くなるかを確かめます。
小さいAI 3つ（Gemma 2B・Qwen 3B・Llama 3B）で測り、人が作った公開の正解（SemEval-2018）でも測ります。計画：sims/e20/PLAN_E20.md。

## 手順
1. L4 GPU。2. `HF_TOKEN` の許可。3. 1つ目のセルを流してから、2つ目で Ctrl+F10。時間の見込み：2〜3時間。ドライブはつなぎません。
""")
code("""
import os, json, time, subprocess, sys, gzip, collections, re, random, glob, zipfile
import numpy as np
from google.colab import userdata
OUT = "/content/E20_results"
os.makedirs(OUT, exist_ok=True)
NLI_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"   # E6〜E19 と同じ
MODELS = {"gemma2b": "google/gemma-2-2b-it", "qwen3b": "Qwen/Qwen2.5-3B-Instruct", "llama3b": "meta-llama/Llama-3.2-3B-Instruct"}
FALLBACK = {"llama3b": "microsoft/Phi-3.5-mini-instruct"}   # Llama の許可が無い時
GEMMA9_ID = "google/gemma-2-9b-it"
SEMEVAL_GDRIVE = "1qc812LUPMfcGTsH324O1gZzDuMLzFT_F"   # SemEval-2018 Task 9 の配布 zip（CRIM の get_data.sh と同じ所）
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wordfreq", "gdown"], check=False)
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
code("# 材料：E11-2 の結果と、E10-1 の辞書の読み方\nimport base64\nB112 = " + repr(B112) + "\nCORE101 = " + repr(CORE101) + "\nDUMP_URL = \"https://dumps.wikimedia.org/enwiktionary/latest/enwiktionary-latest-pages-articles.xml.bz2\"")
code("CATS = " + repr(CATS) + "   # 46分類（E4-2・E10-3 K・E19 と同じ）")
code("""
# 材料を読む：E11-2 の鎖、Wiktionary、WordNet、wordfreq、SemEval-2018
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])
r12 = json.loads(gzip.decompress(base64.b64decode(B112)).decode('utf-8'))
L = r12['L']
CH = {'n': {(w, k): v for w, k, v in r12['choice']}, 'g': {}}
ST = {'n': {(w, k): tuple(v) for w, k, *v in r12['steps']}, 'g': {}}
AR = {(w, k, C): tuple(v) for w, k, C, *v in r12['arrive']}
SC = {}
WKD = load_gz("E20_wkd")
if WKD is None:
    E101 = {}; exec(CORE101, E101)
    dump = "/content/enwiktionary.xml.bz2"
    subprocess.run(["wget", "-c", "-q", "-O", dump, DUMP_URL], check=True)
    print("ダンプ", round(os.path.getsize(dump) / 1e9, 2), "GB"); t = time.time()
    import bz2
    with bz2.open(dump, "rb") as f: WKD = E101["parse_dump"](f)[0]
    print("読み終わり", len(WKD), "語", f"{time.time() - t:.0f}秒"); save_gz(WKD, "E20_wkd")
import nltk
nltk.download("wordnet", quiet=True)
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency
def bucket(z): return "common" if z >= 3.0 else "mid" if z >= 1.5 else "rare"
CSYN = {wn.synset(s): name for s, name in CATS.items()}
CNAMES = set(CATS.values())
def gold_of(x):
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
    pool[x] = (bucket(z), round(z, 2), g)
# E19 の 600 語（種0、E19 と同じ選び方）を作り直して除く
rng0 = random.Random(0); E19W = set(); HAVE = {}
for bk in ("common", "mid", "rare"):
    HAVE[bk] = [x for x in pool if pool[x][0] == bk and WKD.get(x)]
    E19W |= set(rng0.sample(HAVE[bk], min(200, len(HAVE[bk]))))
rng1 = random.Random(1); ITEMS = []; PICK = {}
for bk in ("common", "mid", "rare"):
    rest = [x for x in HAVE[bk] if x not in E19W]; rng1.shuffle(rest); PICK[bk] = []
    for x in rest:
        if len(PICK[bk]) == 200: break
        if new_word(L, x, WKD, nlp): PICK[bk].append(x)
    PICK[bk].sort()
    for x in PICK[bk]:
        for s, name in CATS.items():
            ITEMS.append({"src": "W", "ent": x, "catw": name, "bucket": bk, "zipf": pool[x][1], "gold": pool[x][2][name], "split": "-"})
print("W:", {bk: len(PICK[bk]) for bk in PICK}, "E19 と重なり", len(E19W & set(sum(PICK.values(), []))), " 組", len(ITEMS), " 一種", sum(it["gold"] for it in ITEMS))
# SemEval-2018 Task 9 英語（1A）テスト
# 配布 zip（Google ドライブ）は 11-14 に取れなくなっていた → 同じ 1A 英語テストの語と正解を Hugging Face の写しから読み、同じ形の zip を作る
import pandas as pd
if not os.path.exists("/content/semeval.zip"):
    df = pd.read_parquet("https://huggingface.co/datasets/feliphe-galiza/SemEval2018-Task9-preds-1A.english/resolve/main/data/train-00000-of-00001.parquet")
    with zipfile.ZipFile("/content/semeval.zip", "w") as zf:
        zf.writestr("SemEval2018-Task9/test/data/1A.english.test.data.txt", "".join(t + "\\tConcept\\n" for t in df["input"]))
        zf.writestr("SemEval2018-Task9/test/gold/1A.english.test.gold.txt", "".join("\\t".join(h.strip() for h in g.split(",")) + "\\n" for g in df["grund_truth"]))
zipfile.ZipFile("/content/semeval.zip").extractall("/content/semeval")
fd = glob.glob("/content/semeval/**/1A.english.test.data.txt", recursive=True)[0]
fg = glob.glob("/content/semeval/**/1A.english.test.gold.txt", recursive=True)[0]
terms = [l.split("\\t")[0].strip().lower() for l in open(fd, encoding="utf-8")]
golds = [[h.strip().lower() for h in l.rstrip("\\n").split("\\t") if h.strip()] for l in open(fg, encoding="utf-8")]
print("SemEval 1A テスト", len(terms), "語  例", list(zip(terms[:3], golds[:3])))
allh = sorted({h for g in golds for h in g})
rngE = random.Random(0); ok = []
for x, g in zip(terms, golds):
    if re.fullmatch(r"[a-z]{3,}", x) and g and WKD.get(x) and new_word(L, x, WKD, nlp): ok.append((x, g))
half = set(rngE.sample(sorted({x for x, g in ok}), len({x for x, g in ok}) // 2))
nE = 0
for x, g in ok:
    gs = set(g); pos = [h for h in g if h != x]
    pos = rngE.sample(pos, min(3, len(pos)))
    neg = rngE.sample([h for h in allh if h not in gs and h != x], len(pos))
    z = round(zipf_frequency(x, "en"), 2)
    for h, lab in [(h, 1) for h in pos] + [(h, 0) for h in neg]:
        ITEMS.append({"src": "E", "ent": x, "catw": h, "bucket": bucket(z), "zipf": z, "gold": lab, "split": "tune" if x in half else "test"}); nE += 1
for i, it in enumerate(ITEMS): it["id"] = i
print("E: 語", len(ok), " 組", nE, " 段", collections.Counter(it["bucket"] for it in ITEMS if it["src"] == "E"))
# 説明文（最初の3つ）に Y が語として（s・es も）書いてあるか
def literal(x, C):
    t = " ".join(d["def"] for d in L[x][:3]).lower()
    return int(bool(re.search(r"\\b" + re.escape(C.lower()) + r"(s|es)?\\b", t)))
for it in ITEMS: it["lit"] = literal(it["ent"], it["catw"])
for s in "WE":
    pos = [it for it in ITEMS if it["src"] == s and it["gold"] == 1]
    print(s, "一種の組", len(pos), " うち説明文に書いてある", sum(it["lit"] for it in pos))
""")
md("## 1. NLI を読み込む・AI を読み込む仕組み（GPU）")
code("""
import torch, gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
ntok = AutoTokenizer.from_pretrained(NLI_ID)
nm = AutoModelForSequenceClassification.from_pretrained(NLI_ID).cuda().eval()
lab = {v.lower(): int(k) for k, v in nm.config.id2label.items()}
ORDER = [lab["entailment"], lab["neutral"], lab["contradiction"]]
tok = gm = None
def load_lm(mid):
    global tok, gm, NUM, YES, NO
    if gm is not None:
        gm = None; gc.collect(); torch.cuda.empty_cache()
    tok = AutoTokenizer.from_pretrained(mid, token=userdata.get("HF_TOKEN"), padding_side="left")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dt = torch.float16 if mid.endswith("gemma-2-2b-it") else torch.bfloat16   # 2B は E19 と同じ
    gm = AutoModelForCausalLM.from_pretrained(mid, token=userdata.get("HF_TOKEN"), torch_dtype=dt, device_map="cuda").eval()
    NUM = [tok.encode(str(i), add_special_tokens=False)[-1] for i in range(1, MAX_DEFS_MID + 1)]
    YES = tok.encode("Yes", add_special_tokens=False)[-1]; NO = tok.encode("No", add_special_tokens=False)[-1]
    print("読み込み", mid, " Yes/No", tok.convert_ids_to_tokens([YES, NO]), " 数", tok.convert_ids_to_tokens(NUM[:3]))
def _nli(P, H, B=64):
    out = []
    for i in range(0, len(P), B):
        enc = ntok(P[i:i + B], H[i:i + B], return_tensors="pt", padding=True, truncation=True, max_length=192).to("cuda")
        with torch.no_grad():
            pr = torch.softmax(nm(**enc).logits.float(), -1)[:, ORDER].cpu().numpy()
        out += [(round(float(p[0]), 4), round(float(p[2]), 4)) for p in pr]
    return out
def nli(trip):
    return _nli([premise(w, L[w][k]["def"]) for w, k, c in trip], [hyp(w, c) for w, k, c in trip])
def _last_logits(qs, B, f):
    # 長さの順に並べて束ねる（速さのため）。各行はすぐ f で小さな値にする（全語の値を GPU に溜めない：11-14 v1 でメモリ不足）。返す順は元の順。
    ps = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs]
    order = sorted(range(len(ps)), key=lambda i: len(ps[i])); res = [None] * len(ps)
    for s in range(0, len(order), B):
        idx = order[s:s + B]
        enc = tok([ps[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            last = gm(**enc, logits_to_keep=1).logits[:, -1, :].float()
        for i, row in zip(idx, last): res[i] = f(row, i)
        if s // B % 200 == 0: print(f"    {s}/{len(order)}", flush=True)
    return res
def _pick(qs, ns, B=16):
    return _last_logits(qs, B, lambda r, i: int(torch.softmax(r[NUM[:ns[i]]], -1).argmax()))
def yes_no(qs, B=16):
    return _last_logits(qs, B, lambda r, i: round(float(r[YES] - r[NO]), 3))
""")
md("## 2. 鎖を作る（E19 と同じ作り。親の語は NLI、意味選びは Gemma 2B。GPU、20〜40分）")
code("""
load_lm(MODELS["gemma2b"])
DONE = {"n": set(), "g": set()}
def grow(ents, mode, maxd=MAXD):
    # E19 と同じ（sims/e19/build_notebook_E19.py の grow）
    key = gkey(mode)
    front = sorted(set(ents))
    for w in front: new_word(L, w, WKD, nlp)
    for d in range(maxd):
        todo = []
        for w in front:
            for k, x in enumerate(L.get(w, [])):
                if (w, k) in DONE[mode]: continue
                DONE[mode].add((w, k))
                if mode == "n" and x.get("genus") is not None and (w, k) in ST["n"]: continue
                for c in x["cands"]:
                    if c != w: new_word(L, c, WKD, nlp)
                cs = [c for c in x["cands"] if c != w and L.get(c)]
                if cs: todo.append((w, k, cs))
                else: x[key] = None; CH[mode].pop((w, k), None)
        trip = sorted({(w, k, c) for w, k, cs in todo for c in cs if (w, k, c) not in SC})
        if trip: SC.update(zip(trip, nli(trip)))
        for w, k, cs in todo:
            g = max(cs, key=lambda c: SC[(w, k, c)][0])
            if L[w][k].get("genus") != g: CH["n"].pop((w, k), None)
            L[w][k]["genus"] = g; ST["n"][(w, k)] = SC[(w, k, g)]
        jobs = [(w, k) for w, k, cs in todo if len(L[L[w][k][key]]) > 1 and (w, k) not in CH[mode]]
        if jobs:
            qs = [choice_prompt(w, L[w][k]["def"], L[w][k][key], [x["def"] for x in L[L[w][k][key]]]) for w, k in jobs]
            for (w, k), v in zip(jobs, _pick(qs, [len(L[L[w][k][key]]) for w, k in jobs])): CH[mode][(w, k)] = int(v)
        nxt = set()
        for w in front:
            for k in range(len(L.get(w, []))):
                s = step_from(L, CH, ST, mode, w, k)
                if s: nxt.add(s[0][0])
        print(f"  {mode} 段{d + 1}: 決めた意味 {len(todo)}  NLI {len(trip)}  語 {len(L)}", flush=True)
        front = sorted(nxt)
t = time.time()
grow([it["ent"] for it in ITEMS], "n")
print(f"鎖 {time.time() - t:.0f}秒  辞書に見出しがある X:", sum(1 for it in ITEMS if L.get(it["ent"])), "/", len(ITEMS))
""")
md("## 3. AI ごとに：途中の語の意味選び・鎖の点 T・説明文なし G・説明文あり S（GPU、AI 1つにつき 20〜40分）")
code("""
RES = {}
def choose_senses(CHm):
    # 物の語から3段まで、親の語に意味が2つ以上ある所だけ、このAIに選ばせる（E8n の方法D と同じ聞き方）
    front = {(x, k) for x in {it["ent"] for it in ITEMS} for k in range(len(L.get(x, [])))}
    for d in range(MAXD):
        jobs = sorted({(w, k) for w, k in front if L[w][k].get("genus") and L[w][k]["genus"] in L and L[w][k]["genus"] != w
                       and len(L[L[w][k]["genus"]]) > 1 and (w, k) not in CHm})
        if jobs:
            qs = [choice_prompt(w, L[w][k]["def"], L[w][k]["genus"], [x["def"] for x in L[L[w][k]["genus"]]]) for w, k in jobs]
            for (w, k), v in zip(jobs, _pick(qs, [len(L[L[w][k]["genus"]]) for w, k in jobs])): CHm[(w, k)] = int(v)
        nxt = set()
        for n in front:
            s = step_from(L, {"n": CHm}, ST, "n", *n)
            if s: nxt.add(s[0])
        print(f"  意味選び 段{d + 1}: 聞いた {len(jobs)}", flush=True)
        front = nxt
    return CHm
def t_scores(CHm):
    CHx = {"n": CHm}
    REACH = {it["id"]: reach(L, CHx, ST, "n", it["ent"]) for it in ITEMS}
    trip = sorted({(w, k, it["catw"]) for it in ITEMS for (w, k) in REACH[it["id"]] if w != it["catw"] and (w, k, it["catw"]) not in AR})
    if trip: AR.update(zip(trip, nli(trip)))
    T = []
    for it in ITEMS:
        best = REACH[it["id"]]; C = it["catw"]
        v = [best[n][0] * (1.0 if n[0] == C else AR.get((n[0], n[1], C), (0.0,))[0]) for n in best]
        T.append(round(max(v), 5) if v else 0.0)
    print("  NLI の着き 新しく", len(trip))
    return T
def run_model(key, mid):
    t = time.time()
    if key != "gemma2b":
        try: load_lm(mid)
        except Exception as e:
            print("読み込めない", mid, repr(e)[:200]); mid = FALLBACK[key]; load_lm(mid)
    CHm = dict(CH["n"]) if key == "gemma2b" else {}
    CHm = choose_senses(CHm)
    T = t_scores(CHm)
    G = yes_no([arrive_prompt(it["ent"], it["catw"], []) for it in ITEMS])
    S = yes_no([arrive_prompt(it["ent"], it["catw"], [f'- "{it["ent"]}": {d["def"]}' for d in L[it["ent"]][:3]]) for it in ITEMS])
    RES[key] = {"model": mid, "G": G, "S": S, "T": T, "n_choice": len(CHm)}
    save_gz(RES[key], f"E20_{key}")
    print(f"{key} {mid} {time.time() - t:.0f}秒  G[:3] {G[:3]} S[:3] {S[:3]} T[:3] {T[:3]}", flush=True)
run_model("gemma2b", MODELS["gemma2b"])
""")
code('run_model("qwen3b", MODELS["qwen3b"])')
code('run_model("llama3b", MODELS["llama3b"])')
md("## 4. Gemma-2 9B に説明文なしで聞く（参考：大きな AI。GPU、20〜40分）")
code("""
load_lm(GEMMA9_ID)
t = time.time()
G9 = yes_no([arrive_prompt(it["ent"], it["catw"], []) for it in ITEMS], B=8)
save_gz({"G9": G9}, "E20_g9")
print(f"9B {time.time() - t:.0f}秒", G9[:5])
""")
md("## 5. 比べ（PLAN_E20.md の決まり。e20_analyze.py をそのまま埋め込み）")
code("# sims/e20/e20_analyze.py と同じ\n" + AN20)
code("""
C20 = {"items": [[it["src"], it["bucket"], it["ent"], it["catw"], it["zipf"], it["lit"], it["split"]] for it in ITEMS],
       "gold": [it["gold"] for it in ITEMS], "M": {k: {"G": v["G"], "S": v["S"], "T": v["T"]} for k, v in RES.items()}, "G9": G9,
       "meta": {"nli": NLI_ID, "models": {k: v["model"] for k, v in RES.items()}, "gemma9b": GEMMA9_ID, "dump": DUMP_URL,
                "date": time.strftime("%Y-%m-%d"), "n_wkd": len(WKD)}}
json.dump(C20, open("/content/c20.json", "w"))
res = main("/content/c20.json", "/content/out20.json")
""")
md("## 6. 値を印字する（ここに出た値を読みます）")
code("""
blob = base64.b64encode(gzip.compress(json.dumps(C20, separators=(",", ":")).encode())).decode()
print("E20_C_BEGIN"); print(blob); print("E20_C_END")
print("E20_OUT_BEGIN"); print(json.dumps({k: v for k, v in res.items() if k != "cs_todo"}, ensure_ascii=False, separators=(",", ":"))); print("E20_OUT_END")
""")
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells), "size", os.path.getsize(sys.argv[1]))
