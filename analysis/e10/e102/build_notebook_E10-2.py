"""E10-2 Colabノートを作る。python3 build_notebook_E10-2.py e101_items_nogold.json e10_items_nogold.json out.ipynb
材料を取るだけ（正解は入れない）。計画：sims/e10/PLAN_E10-2.md。結果はドライブ AKANE_アーキテクチャ/E10-2_results/ に .json.gz。
辞書は E10-1 で読んだもの（ドライブの E10-1_results/E101_wiktionary.json.gz）をそのまま使う。"""
import json, os, sys, hashlib, inspect
HERE = os.path.dirname(os.path.abspath(__file__))


def remake(order, corder, meta):
    """ノートの中で主張の文を作り直す（E10・E10-1 と同じ書き方）。"""
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


SETS = {}
for tag, path in (("N", sys.argv[1]), ("O", sys.argv[2])):
    D = json.load(open(path))
    meta = {k: D["meta"][k] for k in ("wholes", "labels", "regions")}
    items = [[it["dom"], it["ent"], it["cls"], it["stmt"], it.get("rev")] for it in D["items"]]
    order = {d: list(dict.fromkeys(it[1] for it in items if it[0] == d)) for d in "PDG"}
    corder = {d: list(dict.fromkeys(it[2] for it in items if it[0] == d)) for d in "PDG"}
    assert remake(order, corder, meta) == items, tag
    SETS[tag] = dict(meta=meta, order=order, corder=corder, md5=hashlib.md5(json.dumps(items).encode()).hexdigest()[:10], n=len(items))
CORE = open(f"{HERE}/e102_core.py").read()
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md(f"""
# AKANE E10-2：辞書の説明文・一般的な文を見せてからGemmaに聞く

このノートは **材料を取るだけ** です（比べは結果のファイルを読んで別に行います）。計画：sims/e10/PLAN_E10-2.md。

- 問題：E10-1 の問題（{SETS['N']['n']:,} 主張、主）と E10 の問題（{SETS['O']['n']:,} 主張、副）。正解はこのノートに入っていません。
- ⑩ 辞書（Wiktionary）の説明文を見せて聞く／⑩s 関係のない語の説明文を見せる（対照、E10-1 の問題だけ）
- ⑪g 一般的な文の資料（GenericsKB）から両方の語を含む文を最大5つ見せて聞く／⑪s 別の主張の文を見せる（対照、E10-1 の問題だけ）
- ⑪n 同じ文をNLIにかけて、主張が成り立つか
- 手で書いた語の一覧は使いません（語の形は lemminflect のデータで作る）。

## 手順
1. 「ランタイム」→「ランタイムのタイプを変更」→ **L4 GPU**。
2. 鍵マークの `HF_TOKEN` にアクセスを許可（E8〜E10-1と同じ）。
3. 「ランタイム」→「すべてのセルを実行」。ドライブの許可を聞かれたら許可。
4. 終わると、ドライブの `AKANE_アーキテクチャ/E10-2_results/` に `E102_*.json.gz` ができます。最後のセルが一覧と大きさを出します。
5. 時間の見込み：一般的な文の資料を取って索引を作る 10〜20分、Gemma 約20万回 60〜90分、NLI 10〜20分。合わせて1.5〜2時間。
6. 途中で切れても、もう一度「すべてのセルを実行」すれば、済んだ所は飛ばします。
""")
code("""
import os, json, time, gzip, subprocess, sys, hashlib, random
import numpy as np
from google.colab import drive, userdata
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/AKANE_アーキテクチャ"
OUT = f"{BASE}/E10-2_results"
WIKT = f"{BASE}/E10-1_results/E101_wiktionary.json.gz"   # E10-1 で読んだ辞書
os.makedirs(OUT, exist_ok=True)
NLI_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"   # E6〜E10-1と同じ
GEMMA_ID = "google/gemma-2-2b-it"
GKB_ID = "community-datasets/generics_kb"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lemminflect", "datasets"], check=False)
def save_gz(obj, name):
    p = f"{OUT}/{name}.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False)
    print("保存:", name, round(os.path.getsize(p) / 1e6, 2), "MB"); return p
def load_gz(name):
    p = f"{OUT}/{name}.json.gz"
    if not os.path.exists(p): return None
    with gzip.open(p, "rt", encoding="utf-8") as f: return json.load(f)
""")
code("# 中身（sims/e10/e102/e102_core.py と同じ。手書きの語の一覧はありません）\n" + CORE)
code("SETS = json.loads(r'''" + json.dumps(SETS, ensure_ascii=False) + "''')\n"
     + inspect.getsource(remake) +
     "ITEMS = {}\n"
     "for tag, S in SETS.items():\n"
     "    ITEMS[tag] = remake(S['order'], S['corder'], S['meta'])\n"
     "    assert hashlib.md5(json.dumps(ITEMS[tag]).encode()).hexdigest()[:10] == S['md5'], ('主張の作り直しが合わない', tag)\n"
     "    print(tag, len(ITEMS[tag]), '主張', {d: (len(S['order'][d]), len(S['corder'][d])) for d in 'PDG'})")
md("## 1. 辞書を読む（E10-1 で保存したもの）と、一般的な文の資料（GenericsKB）を取って、各主張の文を探す（GPUなし、10〜20分）")
code("""
with gzip.open(WIKT, "rt", encoding="utf-8") as f: WKD = json.load(f)["n"]
print("辞書の見出し:", len(WKD))
from lemminflect import getAllInflections
FACTS = {tag: load_gz(f"E102_facts_{tag}") for tag in ITEMS}
if any(v is None for v in FACTS.values()):
    from datasets import load_dataset
    try:
        ds = load_dataset(GKB_ID, "generics_kb", split="train")
    except Exception as e:
        print("全部の版が読めないので best の版にします:", e); ds = load_dataset(GKB_ID, "generics_kb_best", split="train")
    cols = ds.column_names; print("列:", cols, "文の数:", len(ds))
    scol = [c for c in cols if "sentence" in c][0]
    # 列は Python のリストにしてから使う（datasets の列を1つずつ引くと1回数ミリ秒かかり、340万文で何時間にもなる。11-12 直し）
    df = ds.to_pandas(); sents = df[scol].astype(str).tolist()
    scores = df["score"].astype(float).tolist() if "score" in cols else [0.0] * len(sents); del df
    from collections import defaultdict
    TERMS = {}
    for tag, its in ITEMS.items():
        meta = SETS[tag]["meta"]
        for d, x, c, stmt, _ in its:
            for term in (x, class_term(d, c, meta)):
                if term not in TERMS: TERMS[term] = forms(term, getAllInflections)
    print("探す語:", len(TERMS))
    t = time.time(); HITS, KEPT = term_hits(sents, scores, TERMS); print(f"文を読んだ {time.time() - t:.0f}秒")
    for tag, its in ITEMS.items():
        if FACTS[tag] is not None: continue
        meta = SETS[tag]["meta"]; per, used = [], {}
        for d, x, c, stmt, _ in its:
            ids = sorted(HITS[x] & HITS[class_term(d, c, meta)])[:K_FACT]
            per.append(ids)
            for i in ids: used[i] = KEPT[i]
        FACTS[tag] = {"items": per, "sents": {str(i): s for i, s in used.items()}, "n_corpus": len(sents), "source": GKB_ID}
        save_gz(FACTS[tag], f"E102_facts_{tag}")
for tag, F_ in FACTS.items():
    per = F_["items"]
    for d in "PDG":
        m = [len(p) > 0 for p, it in zip(per, ITEMS[tag]) if it[0] == d]
        print(tag, d, "文が見つかった主張の割合", round(float(np.mean(m)), 3))
""")
md("## 2. Gemmaへの問いを作る（対照の入れ替えは決まった種で）")
code("""
def build_prompts(tag, with_controls):
    its, meta = ITEMS[tag], SETS[tag]["meta"]; per = FACTS[tag]["items"]; S_ = FACTS[tag]["sents"]
    rng = random.Random(1)
    Q = {"def": [], "gen": []}
    if with_controls: Q["defS"] = []; Q["genS"] = []
    swap_x, swap_c, swap_f = {}, {}, {}
    if with_controls:
        for d in "PDG":
            xs = sorted({it[1] for it in its if it[0] == d}); cs = sorted({it[2] for it in its if it[0] == d})
            for L, M in ((xs, swap_x), (cs, swap_c)):
                sh = L[:]; rng.shuffle(sh)
                for a, b in zip(sh, sh[1:] + sh[:1]): M[(d, a)] = b
            withf = [i for i, it in enumerate(its) if it[0] == d and per[i]]
            sh = withf[:]; rng.shuffle(sh)
            for a, b in zip(sh, sh[1:] + sh[:1]): swap_f[a] = b
    for i, (d, x, c, stmt, _) in enumerate(its):
        w = class_term(d, c, meta)
        Q["def"].append(def_prompt(x, defs_text(WKD, x), w, defs_text(WKD, w), stmt))
        Q["gen"].append(fact_prompt([S_[str(j)] for j in per[i]], stmt))
        if with_controls:
            x2, c2 = swap_x[(d, x)], swap_c[(d, c)]; w2 = class_term(d, c2, meta)
            Q["defS"].append(def_prompt(x2, defs_text(WKD, x2), w2, defs_text(WKD, w2), stmt))
            j = swap_f.get(i)
            Q["genS"].append(fact_prompt([S_[str(k)] for k in per[j]] if j is not None else [], stmt))
    return Q
PROMPTS = {"N": build_prompts("N", True), "O": build_prompts("O", False)}
print({t: {k: len(v) for k, v in q.items()} for t, q in PROMPTS.items()})
print(PROMPTS["N"]["def"][0]); print(); print(PROMPTS["N"]["gen"][0])
""")
md("## 3. Gemmaに聞く（GPU、60〜90分。方法ごとにドライブに保存）")
code("""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), padding_side="left")
gm = AutoModelForCausalLM.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), torch_dtype=torch.float16, device_map="cuda").eval()
YES = tok.encode("Yes", add_special_tokens=False)[0]; NO = tok.encode("No", add_special_tokens=False)[0]
def yes_no(qs, B=32, log_every=200):
    # 各問いの最後の位置で、Yes と No の値の差だけを残す（全部の語の値をGPUに溜めない）
    out = []; t = time.time()
    for i in range(0, len(qs), B):
        ps = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs[i:i + B]]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            r = gm(**enc, logits_to_keep=1).logits[:, -1, :].float()
            out.extend((r[:, YES] - r[:, NO]).cpu().numpy().round(3).tolist()); del r
        if (i // B) % log_every == 0: print(f"  {i}/{len(qs)}  {time.time() - t:.0f}秒")
    return out
for tag, Q in PROMPTS.items():
    for m, qs in Q.items():
        name = f"E102_gemma_{tag}_{m}"
        if load_gz(name) is not None: print("済み:", name); continue
        print(name, len(qs)); save_gz({"logit": yes_no(qs), "model": GEMMA_ID}, name)
# 確かめ：説明文なしの聞き方を、E10-1 の問題の最初の200個だけもう一度（E10-1 の値と合うか）
if load_gz("E102_check_plain") is None:
    chk = yes_no([gemma_q(it[3]) for it in ITEMS["N"][:200]])
    save_gz({"logit": chk}, "E102_check_plain")
del gm; torch.cuda.empty_cache()
""")
md("## 4. 一般的な文をNLIにかける（⑪n、GPU、10〜20分）")
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
for tag in ITEMS:
    name = f"E102_nli_{tag}"
    if load_gz(name) is not None: print("済み:", name); continue
    per = FACTS[tag]["items"]; S_ = FACTS[tag]["sents"]
    P, H, owner = [], [], []
    for i, ids in enumerate(per):
        for j in ids: P.append(S_[str(j)]); H.append(ITEMS[tag][i][3]); owner.append(i)
    pr = nli(P, H) if P else np.zeros((0, 2))
    ent = [None] * len(per); con = [None] * len(per)
    for (e, c), i in zip(pr.tolist(), owner):
        if ent[i] is None or e > ent[i]: ent[i], con[i] = round(e, 4), round(c, 4)
    save_gz({"ent": ent, "con": con, "n_pairs": len(P), "model": NLI_ID}, name)
""")
md("## 5. できたファイル")
code("""
for f in sorted(os.listdir(OUT)):
    print(f, round(os.path.getsize(f"{OUT}/{f}") / 1e6, 2), "MB")
""")
nb = dict(cells=cells, metadata={"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}, nbformat=4, nbformat_minor=5)
json.dump(nb, open(sys.argv[3], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells), "size", round(os.path.getsize(sys.argv[3]) / 1e6, 2), "MB")
