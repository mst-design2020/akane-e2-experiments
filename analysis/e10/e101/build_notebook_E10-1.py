"""E10-1 Colabノートを作る。python3 build_notebook_E10-1.py e101_items_nogold.json out.ipynb
材料を取るだけ（正解は入れない）。計画：sims/e10/PLAN_E10-1.md。結果はドライブ AKANE_アーキテクチャ/E10-1_results/ に .json.gz（どれも10MB未満）。
方法の中身は e101_core.py（手書きの語の一覧なし）。比べ0のためだけに、E10の手書きつきの鎖（e10_core.py）を別の名前の場所（OLD）で動かす。"""
import json, os, sys, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(sys.argv[1]))
META = D["meta"]
ITEMS = [[it["dom"], it["ent"], it["cls"], it["stmt"], it.get("rev")] for it in D["items"]]
MD5 = hashlib.md5(json.dumps(ITEMS).encode()).hexdigest()[:10]
ORDER = {d: list(dict.fromkeys(it[1] for it in ITEMS if it[0] == d)) for d in "PDG"}
CORDER = {d: list(dict.fromkeys(it[2] for it in ITEMS if it[0] == d)) for d in "PDG"}


def remake(order, corder, meta):
    """ノートの中で主張の文を作り直す（e101_data.py と同じ書き方）。"""
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
CORE = open(f"{HERE}/e101_core.py").read()
OLD_SRC = open(f"{HERE}/../e10_core.py").read()
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md("""
# AKANE E10-1：辞書の確かめの弱点を、手書きの一覧なしで直し、新しい問題で確かめる

このノートは **材料を取るだけ** です（比べは結果のファイルを読んで別に行います）。計画：sims/e10/PLAN_E10-1.md。

- E10で使っていない新しい問題（部分と全体・病気・地理）。正解はこのノートに入っていません。
- 辞書の鎖は **手で書いた語の一覧を使いません**（頭の名詞は文法で候補を出してNLIで選ぶ、「〜of／in〜」は全部たどってNLIで決める）。
- 弱点の直し：直し1・2（部分と全体の受け継ぎ・全体の側から見る）は材料をここで取り、計算は後で。直し3（向き）はGemmaに2択で聞きます。直し4（病名を分ける）は分けた語を辞書で引きます。
- 比べのためだけに、E10の手書きの一覧つきの鎖も同じ材料で作ります（OLD）。

## 手順
1. 「ランタイム」→「ランタイムのタイプを変更」→ **L4 GPU**。
2. 鍵マークの `HF_TOKEN` にアクセスを許可（E8〜E10と同じ）。
3. 「ランタイム」→「すべてのセルを実行」。ドライブの許可を聞かれたら許可。
4. 終わると、ドライブの `AKANE_アーキテクチャ/E10-1_results/` に `E101_*.json.gz`（数個）ができます。最後のセルが一覧と大きさを出します。
5. 時間の見込み：Gemma 10〜15分、ダンプを取って読む 30〜50分（この間「GPUを使っていません」の警告が出ますが×で閉じてください）、語の一覧 10〜20分、NLIとGemmaの意味選び 90〜150分。合わせて3〜4時間。
6. 途中で切れても、もう一度「すべてのセルを実行」すれば、済んだ所は飛ばします（途中の控えをドライブに保存しています）。
""")
code("""
import os, json, time, bz2, gzip, subprocess, sys, hashlib
import numpy as np
from google.colab import drive, userdata
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/AKANE_アーキテクチャ"
OUT = f"{BASE}/E10-1_results"
os.makedirs(OUT, exist_ok=True)
NLI_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"   # E6〜E10と同じ
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
code("# 中身（sims/e10/e101/e101_core.py と同じ。手書きの語の一覧はありません）\n" + CORE)
code("# 比べ0のためだけ：E10の鎖（sims/e10/e10_core.py、手書きの一覧つき）を OLD という別の場所で使う\n"
     "OLD_SRC = " + repr(OLD_SRC) + "\nOLD = {}\nexec(OLD_SRC, OLD)\nprint('OLD:', sorted(k for k in OLD if k.isupper())[:8])")
code("META = json.loads(r'''" + json.dumps(META, ensure_ascii=False) + "''')\n"
     "ORDER = json.loads(r'''" + json.dumps(ORDER, ensure_ascii=False) + "''')\n"
     "CORDER = json.loads(r'''" + json.dumps(CORDER, ensure_ascii=False) + "''')\n"
     + __import__("inspect").getsource(remake) +
     "ITEMS = remake(ORDER, CORDER, META)\n"
     f"assert hashlib.md5(json.dumps(ITEMS).encode()).hexdigest()[:10] == '{MD5}', '主張の作り直しが合わない'\n"
     "ENTS = {d: sorted({it[1] for it in ITEMS if it[0] == d}) for d in 'PDG'}\n"
     "CLASSES = {d: sorted({it[2] for it in ITEMS if it[0] == d}) for d in 'PDG'}\n"
     "WHOLES = sorted(META['wholes'])\n"
     "print(len(ITEMS), '主張', {d: (len(ENTS[d]), len(CLASSES[d])) for d in 'PDG'}, '確認の印:', hashlib.md5(json.dumps(ITEMS).encode()).hexdigest()[:10])")
md("## 1. Gemmaに全部の主張を「はい／いいえ」で聞き、部分と全体は向きを2択でも聞く（GPU、10〜15分）")
code("""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), padding_side="left")
gm = AutoModelForCausalLM.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), torch_dtype=torch.float16, device_map="cuda").eval()
def last_logits(qs, a, b, B=64, log_every=100):
    # 各問いの最後の位置で、語a と語b の値の差だけを残す（全部の語の値をGPUに溜めるとメモリが足りない）
    out = []; t = time.time()
    for i in range(0, len(qs), B):
        ps = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs[i:i + B]]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            r = gm(**enc, logits_to_keep=1).logits[:, -1, :].float()
            out.extend((r[:, a] - r[:, b]).cpu().numpy().round(3).tolist()); del r
        if (i // B) % log_every == 0: print(f"{i}/{len(qs)}  {time.time() - t:.0f}秒")
    return out
G = load_gz("E101_gemma")
if G is None:
    YES = tok.encode("Yes", add_special_tokens=False)[0]; NO = tok.encode("No", add_special_tokens=False)[0]
    Q = [gemma_q(it[3]) for it in ITEMS] + [gemma_q(it[4]) for it in ITEMS if it[4]]
    lg = [float(v) for v in last_logits(Q, YES, NO)]
    n = len(ITEMS); rev = iter(lg[n:])
    G = {"logit": lg[:n], "rev": [next(rev) if it[4] else None for it in ITEMS], "model": GEMMA_ID}
    save_gz(G, "E101_gemma")
DIR = load_gz("E101_direction")
if DIR is None:
    A_ = tok.encode("A", add_special_tokens=False)[0]; B_ = tok.encode("B", add_special_tokens=False)[0]
    PI = [i for i, it in enumerate(ITEMS) if it[0] == "P"]
    Q = [dir_prompt(ITEMS[i][3], ITEMS[i][4]) for i in PI] + [dir_prompt(ITEMS[i][4], ITEMS[i][3]) for i in PI]
    d = [float(v) for v in last_logits(Q, A_, B_)]
    n = len(PI)
    DIR = {"idx": PI, "ab": d[:n], "ba": d[n:]}   # ab：A＝「x は W の一部」、ba：B＝「x は W の一部」
    save_gz(DIR, "E101_direction")
lg = np.array(G["logit"])
for d in "PDG":
    m = np.array([it[0] == d for it in ITEMS]); print(d, "はいの割合", round(float((lg[m] > 0).mean()), 3))
""")
md("## 2. Wiktionaryのダンプを取って読む（名詞・固有名詞・形容詞・部分の一覧・全体の一覧、30〜50分、GPUは使わない）")
code("""
wk = load_gz("E101_wiktionary")
if wk is not None:
    WKD, WKA, MERO, HOLO = wk["n"], wk["a"], wk["mero"], wk["holo"]; print("ドライブから読み込み:", len(WKD))
else:
    dump = "/content/enwiktionary.xml.bz2"
    subprocess.run(["wget", "-c", "-q", "--show-progress", "-O", dump, DUMP_URL], check=True)
    print("取れた大きさ:", round(os.path.getsize(dump) / 1e9, 2), "GB")
    t = time.time()
    with bz2.open(dump, "rb") as f:
        WKD, WKA, MERO, HOLO = parse_dump(f)
    print("読み終わり:", len(WKD), "語", f"{time.time() - t:.0f}秒")
    save_gz({"n": WKD, "a": WKA, "mero": MERO, "holo": HOLO, "date": time.strftime("%Y-%m-%d")}, "E101_wiktionary")
print("名詞", len(WKD), "形容詞", len(WKA), "部分の一覧", len(MERO), "全体の一覧", len(HOLO))
for d in "PDG":
    print(d, "説明文のある物の割合:", round(float(np.mean([lookup(WKD, w) is not None for w in ENTS[d]])), 3))
# 病気の「〜の一部か」の器官名を分類名から作る（手書きの部位名なし）
for l, lb in META["labels"].items():
    lb["system"] = system_of(lb["name"], WKD, WKA)
print({l: lb["system"] for l, lb in META["labels"].items()})
# 直し4：辞書にない病名を、最後の語（種類の候補）と残り（部位・形容詞）に分ける
SPLIT = {}
for x in ENTS["D"]:
    if lookup(WKD, x) is None and split_name(x):
        kind, rest = split_name(x)
        SPLIT[x] = {"kind": kind, "rest": rest, "rest_last": rest.split()[-1]}
print("辞書にない病名:", sum(lookup(WKD, x) is None for x in ENTS["D"]), " 分けられた:", len(SPLIT))
""")
md("## 3. 語の意味の一覧（一覧なし・E10の手書きつきの両方、10〜20分）")
code("""
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])
D_EXTRA = sorted({v[k] for v in SPLIT.values() for k in ("kind", "rest", "rest_last")})
ADJ_OK = {v[k] for v in SPLIT.values() for k in ("rest", "rest_last")}
LX = load_gz("E101_lexicon")
if LX is None:
    t = time.time()
    L = build_lexicon({"P": ENTS["P"] + WHOLES, "D": ENTS["D"] + D_EXTRA, "G": ENTS["G"]}, WKD, nlp, WKA, ADJ_OK)
    LO = OLD["build_lexicon"](ENTS, WKD, nlp, wide_doms=())
    print("語:", len(L), "（E10の手書きつき:", len(LO), "）", f"{time.time() - t:.0f}秒")
    save_gz({"L": L, "LO": LO}, "E101_lexicon")
else:
    L, LO = LX["L"], LX["LO"]
PAIRS = sorted({(w, k, g) for w in L for k, x in enumerate(L[w]) for g in x["cands"] if g != w and L.get(g)})
print("頭の名詞の候補の組:", len(PAIRS))
""")
md("## 4. NLIで頭の名詞を選び、Gemmaに途中の語の意味を選ばせる（GPU、30〜50分、途中の控えあり）")
code("""
ntok = AutoTokenizer.from_pretrained(NLI_ID)
nm = AutoModelForSequenceClassification.from_pretrained(NLI_ID, torch_dtype=torch.float16).cuda().eval()
lab = {v.lower(): int(k) for k, v in nm.config.id2label.items()}
ORD = [lab["entailment"], lab["contradiction"]]
def nli_raw(P, H, B=128):
    out = np.zeros((len(P), 2), np.float32)
    for i in range(0, len(P), B):
        enc = ntok(P[i:i + B], H[i:i + B], return_tensors="pt", padding=True, truncation=True, max_length=192).to("cuda")
        with torch.no_grad():
            pr = torch.softmax(nm(**enc).logits.float(), -1)[:, ORD]
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
SP = nli_probs([premise(w, L[w][k]["def"]) for w, k, g in PAIRS], [isa_hyp(w, g) for w, k, g in PAIRS], "ck_pairs")
SC = {p: float(s[0]) for p, s in zip(PAIRS, SP)}
set_genus(L, SC)
# 直し4：種類の候補の語が「病気」と言えるか（その語の説明文 ⇒ "{語} is a disease."）
KIND = sorted({v["kind"] for v in SPLIT.values()})
KP = [(w, k) for w in KIND for k in range(len(L.get(w, [])))]
KS = nli_probs([premise(w, L[w][k]["def"]) for w, k in KP], [disease_hyp(w) for w, k in KP], "ck_kind")
KIND_SC = {w: 0.0 for w in KIND}
for (w, k), s in zip(KP, KS): KIND_SC[w] = max(KIND_SC[w], float(s[0]))
print("種類の語と言えた:", sum(v > 0.5 for v in KIND_SC.values()), "/", len(KIND))
# 意味を選ぶ必要のある段（一覧なしの鎖と、E10の手書きつきの鎖の両方）
NEED = set()
def walk(starts, nxt_fn, LL, depth):
    for x in starts:
        front, seen = [(x, k) for k in range(len(LL.get(x, [])))], set()
        for _ in range(depth):
            nxt = []
            for (w, k) in front:
                for g, typ in nxt_fn(w, k):
                    if len(LL[g]) > 1: NEED.add((w, k, g))
                    for j in range(len(LL[g])):
                        if (g, j) not in seen: seen.add((g, j)); nxt.append((g, j))
            front = nxt
walk(ENTS["P"], lambda w, k: next_of(L, w, k), L, DEPTH)
walk(WHOLES, lambda w, k: next_of(L, w, k, True), L, 2)
walk(ENTS["D"] + D_EXTRA, lambda w, k: next_of(L, w, k), L, DEPTH)
walk(ENTS["G"], lambda w, k: next_of(L, w, k), L, DEPTH)
for d in "PDG":
    walk(ENTS[d], lambda w, k, d=d: OLD["next_of"](LO, w, k, d, False), LO, DEPTH)
NEED = sorted(NEED)
CH = load_gz("E101_choice") or []
CHOICE = {(w, k, g): v for w, k, g, v in CH}
todo = [x for x in NEED if x not in CHOICE]
print("意味を選ぶ必要のある段:", len(NEED), " 済み:", len(CHOICE), " 残り:", len(todo))
if todo:
    NUM = [tok.encode(str(i), add_special_tokens=False)[0] for i in range(1, MAX_DEFS_MID + 1)]
    t = time.time(); B = 16
    for i in range(0, len(todo), B):
        batch = todo[i:i + B]
        LL = [L if (w in L and k < len(L[w]) and g in L) else LO for w, k, g in batch]
        qs = [choice_prompt(w, ll[w][k]["def"], g, [x["def"] for x in ll[g]]) for (w, k, g), ll in zip(batch, LL)]
        ps = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            last = gm(**enc, logits_to_keep=1).logits[:, -1, :].float()
        for (w, k, g), row, ll in zip(batch, last, LL):
            CHOICE[(w, k, g)] = int(row[NUM[:len(ll[g])]].argmax())
        if (i // B) % 200 == 0:
            print(f"{i}/{len(todo)}  {time.time() - t:.0f}秒")
            save_gz([[w, k, g, v] for (w, k, g), v in CHOICE.items()], "E101_choice")
    save_gz([[w, k, g, v] for (w, k, g), v in CHOICE.items()], "E101_choice")
print("最初の意味以外を選んだ割合:", round(float(np.mean([v != 0 for v in CHOICE.values()])), 3))
""")
md("## 5. 鎖を作り、NLIで段と着きを判定する（GPU、60〜100分、途中の控えあり）")
code("""
CHAINS = {}
for d in "PDG":
    starts = ENTS[d] + (D_EXTRA if d == "D" else [])
    for x in starts:
        CHAINS[(d, "N", x)] = chain_nodes(L, x, CHOICE)
    for x in ENTS[d]:
        CHAINS[(d, "D", x)] = OLD["chain_nodes"](LO, x, d, "D", CHOICE, False)
WCH = {W: chain_nodes(L, W, CHOICE, depth=2, isa_only=True) for W in WHOLES}   # 直し1：全体の上の種類
STEPS = sorted({(d, tag, a[0], a[1], b[0], typ) for (d, tag, x), (nd, lk) in CHAINS.items() for a, b, typ in lk if not (tag == "N" and typ == "isa")})
LEX = lambda tag: L if tag == "N" else LO
r = nli_probs([premise(w, LEX(tag)[w][k]["def"]) for d, tag, w, k, g, typ in STEPS],
              [step_hyp(w, g, typ, d) if tag == "N" else OLD["step_hyp"](w, g, typ, d) for d, tag, w, k, g, typ in STEPS], "ck_steps")
ST = [[d, tag, w, k, g, typ, round(float(p[0]), 4)] for (d, tag, w, k, g, typ), p in zip(STEPS, r)]
# 一覧なしの一種の段は、頭の名詞を選んだ時のNLIの値をそのまま使う
for (d, tag, x), (nd, lk) in CHAINS.items():
    if tag == "N":
        for a, b, typ in lk:
            if typ == "isa": ST.append([d, "N", a[0], a[1], b[0], "isa", round(SC.get((a[0], a[1], b[0]), 0.0), 4)])
for W, (nd, lk) in WCH.items():
    for a, b, typ in lk: ST.append(["W", "N", a[0], a[1], b[0], "isa", round(SC.get((a[0], a[1], b[0]), 0.0), 4)])
ST = [list(t) for t in sorted({tuple(s) for s in ST})]
save_gz(ST, "E101_steps")
# 着き：場面ごとに、一覧なし・手書きつきの両方の点。どちらも同じダンプの同じ説明文なので、(語, 意味の番号) が同じなら同じ点
def def_of(w, k):
    return L[w][k]["def"] if w in L and k < len(L[w]) else LO[w][k]["def"]
NODES = {d: sorted({(n[0], n[1]) for (dd, tag, x), (nd, lk) in CHAINS.items() if dd == d for n in nd}) for d in "PDG"}
ARR = {}
for d in "PDG":
    nh = 2 if d == "D" else 1
    nodes = NODES[d]
    P_, H_ = [], []
    for w, k in nodes:
        for c in CLASSES[d]:
            hs = arrive_hyps(w, c, d, META); hs = hs + [hs[0]] * (nh - len(hs))
            for h in hs: P_.append(premise(w, def_of(w, k))); H_.append(h)
    ARR[d] = nli_probs(P_, H_, f"ck_arrive_{d}")[:, 0].reshape(len(nodes), len(CLASSES[d]), nh)
    print(d, "着き 済み", len(nodes), "点")
""")
md("## 6. 結果を小さく分けて保存（どれも10MBより小さく）")
code("""
used = sorted({n[0] for d in "PDG" for n in NODES[d]} | set(WHOLES) | {n[0] for W, (nd, lk) in WCH.items() for n in nd})
LU = {w: [{"def": x["def"], "genus": x.get("genus"), "loc": x["loc"], "cands": x["cands"]} for x in L[w]] for w in used if w in L}
LOU = {w: [{"def": x["def"], "genus": x["genus"], "loc": x["loc"]} for x in LO[w]] for w in used if w in LO}
WLEM = {W: [x["lem"] for x in L.get(W, [])] for W in WHOLES}
rel = set(ENTS["P"]) | set(WHOLES)
save_gz({"meta": {"nli": NLI_ID, "gemma": GEMMA_ID, "dump": DUMP_URL, "date": time.strftime("%Y-%m-%d"),
                  "items_md5": hashlib.md5(json.dumps(ITEMS).encode()).hexdigest()[:10], "labels": META["labels"]},
         "L": LU, "LO": LOU, "whole_lemmas": WLEM,
         "mero": {w: MERO[k] for w in rel for k in [lookup(MERO, w)] if k},
         "holo": {w: HOLO[k] for w in rel for k in [lookup(HOLO, w)] if k},
         "split": SPLIT, "kind_disease": KIND_SC,
         "whole_chains": {W: [[list(n) for n in nd], [[list(a), list(b), t] for a, b, t in lk]] for W, (nd, lk) in WCH.items()}}, "E101_L_used")
save_gz([[w, k, g, v] for (w, k, g), v in CHOICE.items()], "E101_choice")
for d in "PDG":
    ch = {f"{tag}|{x}": [[list(n) for n in nd], [[list(a), list(b), t] for a, b, t in lk]]
          for (dd, tag, x), (nd, lk) in CHAINS.items() if dd == d}
    save_gz({"chains": ch, "nodes": [list(n) for n in NODES[d]], "classes": CLASSES[d],
             "arrive": np.round(ARR[d], 3).tolist()}, f"E101_res_{d}")
print("\\nドライブの E10-1_results の中身:")
for f in sorted(os.listdir(OUT)):
    if f.endswith(".gz"): print(f, round(os.path.getsize(f"{OUT}/{f}") / 1e6, 2), "MB")
""")
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells), "size", round(os.path.getsize(sys.argv[2]) / 1e6, 2), "MB")
