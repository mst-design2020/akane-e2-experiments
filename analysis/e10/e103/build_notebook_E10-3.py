"""E10-3 Colabノートを作る。python3 build_notebook_E10-3.py e101_items_nogold.json k_items_nogold.json out.ipynb
材料を取るだけ（正解は入れない）。計画：sims/e10/PLAN_E10-3.md。結果はドライブ AKANE_アーキテクチャ/E10-3_results/ に .json.gz。
N の P の一般的な文は E10-2 の結果（E10-2_results/E102_facts_N.json.gz）を読む。辞書は E10-1 の物。"""
import json, os, sys, hashlib, inspect
HERE = os.path.dirname(os.path.abspath(__file__))


def remake(order, corder, meta):
    """ノートの中で主張の文を作り直す（E10・E10-1・E10-2 と同じ書き方）。"""
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


D = json.load(open(sys.argv[1]))
meta = {k: D["meta"][k] for k in ("wholes", "labels", "regions")}
items = [[it["dom"], it["ent"], it["cls"], it["stmt"], it.get("rev")] for it in D["items"]]
order = {d: list(dict.fromkeys(it[1] for it in items if it[0] == d)) for d in "PDG"}
corder = {d: list(dict.fromkeys(it[2] for it in items if it[0] == d)) for d in "PDG"}
assert remake(order, corder, meta) == items
SET_N = dict(meta=meta, order=order, corder=corder, md5=hashlib.md5(json.dumps(items).encode()).hexdigest()[:10], n=len(items))
K = json.load(open(sys.argv[2]))
SET_K = dict(ents=K["ents"], cats=K["cats"], n=len(K["ents"]) * len(K["cats"]))
CORE = open(f"{HERE}/../e102/e102_core.py").read() + "\n\n" + open(f"{HERE}/e103_core.py").read().replace("import re\n", "")
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md(f"""
# AKANE E10-3：自分で書かせた文との比べ・「一種か」にも資料を見せる・意味のずれた文を除く

このノートは **材料を取るだけ** です（比べは結果のファイルを読んで別に行います）。計画：sims/e10/PLAN_E10-3.md。

- N の P：E10-1 の部分と全体（{sum(1 for it in items if it[0] == 'P'):,} 主張）。⑫ Gemma に自分で文を書かせて見せる／⑪f 意味の合う一般的な文だけ見せる。
- K：「一種か」（{SET_K['n']:,} 主張）。⑩ 辞書の説明文／⑪g 一般的な文／⑫ 自分で書いた文、対照 ⑩s・⑪s。
- 正解はこのノートに入っていません。手で書いた語の一覧は使いません。

## 手順
1. 「ランタイム」→「ランタイムのタイプを変更」→ **L4 GPU**（**ハイRAM** があれば選ぶ）。
2. 鍵マークの `HF_TOKEN` にアクセスを許可（これまでと同じ）。
3. 「ランタイム」→「すべてのセルを実行」。ドライブの許可を聞かれたら許可。
4. 終わると、ドライブの `AKANE_アーキテクチャ/E10-3_results/` に `E103_*.json.gz` ができます。最後のセルが一覧を出します。
5. 時間の見込み：一般的な文の読み込みと文探し 5分、Gemma に書かせる 20〜30分、Gemma に聞く 40〜50分、NLI 10分。合わせて1.5時間ほど。
6. 途中で切れても、もう一度「すべてのセルを実行」すれば、済んだ所（書かせる所は2,000件ごと）は飛ばします。
""")
code("""
import os, json, time, gzip, subprocess, sys, hashlib, random
import numpy as np
from google.colab import drive, userdata
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/AKANE_アーキテクチャ"
OUT = f"{BASE}/E10-3_results"
R102 = f"{BASE}/E10-2_results"                          # E10-2 で見つけた一般的な文
WIKT = f"{BASE}/E10-1_results/E101_wiktionary.json.gz"   # E10-1 で読んだ辞書
os.makedirs(OUT, exist_ok=True)
NLI_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
GEMMA_ID = "google/gemma-2-2b-it"
GKB_ID = "community-datasets/generics_kb"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lemminflect", "datasets"], check=False)
def save_gz(obj, name, d=None):
    p = f"{d or OUT}/{name}.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False)
    print("保存:", name, round(os.path.getsize(p) / 1e6, 2), "MB"); return p
def load_gz(name, d=None):
    p = f"{d or OUT}/{name}.json.gz"
    if not os.path.exists(p): return None
    with gzip.open(p, "rt", encoding="utf-8") as f: return json.load(f)
""")
code("# 中身（sims/e10/e102/e102_core.py と e103/e103_core.py と同じ。手書きの語の一覧はありません）\nimport re\n" + CORE)
code("SET_N = json.loads(r'''" + json.dumps(SET_N, ensure_ascii=False) + "''')\n"
     "SET_K = json.loads(r'''" + json.dumps(SET_K, ensure_ascii=False) + "''')\n"
     + inspect.getsource(remake) +
     "ITEMS_N = remake(SET_N['order'], SET_N['corder'], SET_N['meta'])\n"
     "assert hashlib.md5(json.dumps(ITEMS_N).encode()).hexdigest()[:10] == SET_N['md5'], '主張の作り直しが合わない'\n"
     "PIDX = [i for i, it in enumerate(ITEMS_N) if it[0] == 'P']\n"
     "NP = [(ITEMS_N[i][1], ITEMS_N[i][2], ITEMS_N[i][3]) for i in PIDX]   # (x, W, 文)\n"
     "KI = [(x, c, k_stmt(x, c)) for x in SET_K['ents'] for c in SET_K['cats']]   # (x, 分類名, 文)\n"
     "print('N の P', len(NP), ' K', len(KI)); print(NP[0]); print(KI[0])")
md("## 1. 辞書と一般的な文を読む（N の P は E10-2 の物、K は新しく探す。GPUなし、5分）")
code("""
with gzip.open(WIKT, "rt", encoding="utf-8") as f: WKD = json.load(f)["n"]
print("辞書の見出し:", len(WKD))
F102 = load_gz("E102_facts_N", R102); assert F102 is not None, "E10-2_results/E102_facts_N.json.gz が見つかりません"
FN = [[F102["sents"][str(j)] for j in F102["items"][i]] for i in PIDX]   # N の P の各主張の文
FK = load_gz("E103_facts_K")
if FK is None:
    from lemminflect import getAllInflections
    from datasets import load_dataset
    try:
        ds = load_dataset(GKB_ID, "generics_kb", split="train")
    except Exception as e:
        print("全部の版が読めないので best の版にします:", e); ds = load_dataset(GKB_ID, "generics_kb_best", split="train")
    cols = ds.column_names; scol = [c for c in cols if "sentence" in c][0]
    df = ds.to_pandas(); sents = df[scol].astype(str).tolist()   # 列はリストにしてから使う（E10-2 の直し）
    scores = df["score"].astype(float).tolist() if "score" in cols else [0.0] * len(sents); del df
    TERMS = {}
    for x, c, _ in KI:
        for term in (x, c):
            if term not in TERMS: TERMS[term] = forms(term, getAllInflections)
    print("文の数:", len(sents), "探す語:", len(TERMS))
    t = time.time(); HITS, KEPT = term_hits(sents, scores, TERMS); print(f"文を読んだ {time.time() - t:.0f}秒")
    FK = {"items": [[KEPT[j] for j in sorted(HITS[x] & HITS[c])[:K_FACT]] for x, c, _ in KI], "source": GKB_ID}
    save_gz(FK, "E103_facts_K"); del sents, scores, HITS, KEPT
FK = FK["items"]
print("文が見つかった割合  N の P:", round(float(np.mean([len(f) > 0 for f in FN])), 3), " K:", round(float(np.mean([len(f) > 0 for f in FK])), 3))
""")
md("## 2. Gemma を読み込み、自分で文を書かせる（⑫、GPU、20〜30分。2,000件ごとに保存）")
code("""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), padding_side="left")
gm = AutoModelForCausalLM.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), torch_dtype=torch.float16, device_map="cuda").eval()
YES = tok.encode("Yes", add_special_tokens=False)[0]; NO = tok.encode("No", add_special_tokens=False)[0]
chat = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
def yes_no(qs, B=32, log_every=200):
    out = []; t = time.time()
    for i in range(0, len(qs), B):
        enc = tok([chat(q) for q in qs[i:i + B]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            r = gm(**enc, logits_to_keep=1).logits[:, -1, :].float()
            out.extend((r[:, YES] - r[:, NO]).cpu().numpy().round(3).tolist()); del r
        if (i // B) % log_every == 0: print(f"  {i}/{len(qs)}  {time.time() - t:.0f}秒")
    return out
def write(qs, B=64):
    enc = tok([chat(q) for q in qs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad():
        g = gm.generate(**enc, max_new_tokens=SELF_NEW_TOKENS, do_sample=False)
    return tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
def write_all(name, qs, B=64, every=2000):
    done = load_gz(name) or {"text": []}; T = done["text"]; t = time.time()
    while len(T) < len(qs):
        j = len(T); chunk = qs[j:j + every]
        for k in range(0, len(chunk), B): T.extend(write(chunk[k:k + B]))
        save_gz({"text": T, "model": GEMMA_ID}, name); print(f"  {len(T)}/{len(qs)}  {time.time() - t:.0f}秒")
    return T
SELF_N = [parse_lines(s) for s in write_all("E103_self_N", [self_prompt(x, W) for x, W, _ in NP])]
SELF_K = [parse_lines(s) for s in write_all("E103_self_K", [self_prompt(x, c) for x, c, _ in KI])]
print("書いた行の数の平均  N の P:", round(float(np.mean([len(s) for s in SELF_N])), 2), " K:", round(float(np.mean([len(s) for s in SELF_K])), 2))
print(NP[0][2], SELF_N[0])
""")
md("## 3. 意味の確かめ（⑪f）と、Gemma への問い（GPU、40〜50分。方法ごとに保存）")
code("""
# 3-1 N の P の一般的な文ごとに「問いと同じ意味で使われているか」
name = "E103_sense_N"
if load_gz(name) is None:
    qs, owner = [], []
    for i, ((x, W, stmt), fs) in enumerate(zip(NP, FN)):
        for f_ in fs: qs.append(sense_prompt(stmt, f_, x, W)); owner.append(i)
    v = yes_no(qs); per = [[] for _ in NP]
    for i, s in zip(owner, v): per[i].append(s)
    save_gz({"logit": per}, name)
SENSE = load_gz(name)["logit"]
FNF = [[f_ for f_, s in zip(fs, ss) if s > 0] for fs, ss in zip(FN, SENSE)]
print("意味が合うと残った文の割合:", round(sum(map(len, FNF)) / max(1, sum(map(len, FN))), 3))
# 3-2 問いを作る（対照の入れ替えは種1、E10-2 と同じ考え）
rng = random.Random(1)
def derange(L):
    sh = L[:]; rng.shuffle(sh); return dict(zip(sh, sh[1:] + sh[:1]))
sx, sc = derange(sorted({x for x, _, _ in KI})), derange(sorted({c for _, c, _ in KI}))
withf = [i for i, f_ in enumerate(FK) if f_]; sf = derange(withf)
Q = {
    "N_self": [fact_prompt(s, stmt) for (_, _, stmt), s in zip(NP, SELF_N)],
    "N_genF": [fact_prompt(f_, stmt) for (_, _, stmt), f_ in zip(NP, FNF)],
    "K_def": [def_prompt(x, defs_text(WKD, x), c, defs_text(WKD, c), stmt) for x, c, stmt in KI],
    "K_defS": [def_prompt(sx[x], defs_text(WKD, sx[x]), sc[c], defs_text(WKD, sc[c]), stmt) for x, c, stmt in KI],
    "K_gen": [fact_prompt(f_, stmt) for (_, _, stmt), f_ in zip(KI, FK)],
    "K_genS": [fact_prompt(FK[sf[i]] if i in sf else [], stmt) for i, (_, _, stmt) in enumerate(KI)],
    "K_self": [fact_prompt(s, stmt) for (_, _, stmt), s in zip(KI, SELF_K)],
}
print({k: len(v) for k, v in Q.items()}); print(Q["K_def"][0]); print(); print(Q["N_self"][0])
for m, qs in Q.items():
    name = f"E103_gemma_{m}"
    if load_gz(name) is not None: print("済み:", name); continue
    print(name, len(qs)); save_gz({"logit": yes_no(qs), "model": GEMMA_ID}, name)
# 確かめ：K の聞き方なしの値を最初の200個だけもう一度（E4-2 の値と合うか）
if load_gz("E103_check_plain_K") is None:
    save_gz({"logit": yes_no([gemma_q(stmt) for _, _, stmt in KI[:200]])}, "E103_check_plain_K")
del gm; torch.cuda.empty_cache()
""")
md("## 4. NLI（文ごと。N の P は全部の文と意味の合う文、K は全部の文。GPU、10分）")
code("""
ntok = AutoTokenizer.from_pretrained(NLI_ID)
nm = AutoModelForSequenceClassification.from_pretrained(NLI_ID).cuda().eval()
lab = {v.lower(): int(k) for k, v in nm.config.id2label.items()}
ORD = [lab["entailment"], lab["contradiction"]]
def nli(P, H, B=128):
    out = np.zeros((len(P), 2), np.float32)
    for i in range(0, len(P), B):
        enc = ntok(P[i:i + B], H[i:i + B], return_tensors="pt", padding=True, truncation=True, max_length=192).to("cuda")
        with torch.no_grad():
            pr = torch.softmax(nm(**enc).logits.float(), -1)[:, ORD]
        out[i:i + len(pr)] = pr.cpu().numpy()
    return out
for name, its, facts in (("E103_nli_N", NP, FN), ("E103_nli_K", KI, FK)):
    if load_gz(name) is not None: print("済み:", name); continue
    P, H, owner = [], [], []
    for i, ((_, _, stmt), fs) in enumerate(zip(its, facts)):
        for f_ in fs: P.append(f_); H.append(stmt); owner.append(i)
    pr = nli(P, H) if P else np.zeros((0, 2)); per = [[] for _ in its]
    for e, i in zip(pr[:, 0].round(4).tolist(), owner): per[i].append(e)
    save_gz({"ent": per, "model": NLI_ID}, name)
""")
md("## 5. できたファイル")
code("""
for f in sorted(os.listdir(OUT)):
    print(f, round(os.path.getsize(f"{OUT}/{f}") / 1e6, 2), "MB")
""")
nb = dict(cells=cells, metadata={"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}, nbformat=4, nbformat_minor=5)
json.dump(nb, open(sys.argv[3], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells), "size", round(os.path.getsize(sys.argv[3]) / 1e6, 2), "MB")
