"""E14 Colabノート（辞書の鎖の各点に、一般的な文・Gemmaが書いた文の「AはBの一種」を継ぎ足す）。
python3 build_notebook_E14.py out.ipynb
計画：plans/E14_fill_gaps.md。材料を取るだけ（比べは e14_analyze.py で別に）。"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..")
OLD = [{k: x[k] for k in ("id", "ent", "catw", "stmt")} for x in json.load(open(f"{R}/e6/e6_items.json"))]  # 正解は入れない
NEW = [{k: x[k] for k in ("id", "ent", "catw", "stmt")} for x in json.load(open(f"{R}/e8/e8_new_items.json"))]
CORE8 = open(f"{R}/nolist/e8n_core.py").read()
CORE = open(f"{HERE}/e14_core.py").read()
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md("""
# AKANE E14：辞書の鎖の行き止まりに、別の資料の「AはBの一種」を継ぎ足す

このノートは **材料を取るだけ** です（比べは結果のファイルを読んで別に行います）。計画：plans/E14_fill_gaps.md。

- E11-2 の6段の鎖（E8n の方法D を伸ばした物）をそのまま読み、鎖の全部の語に、2つの資料から文を足します。
  - 一般的な文（GenericsKB の全部の版。WordNet・ConceptNet から作った文は除く）
  - Gemma に自分で書かせた1文
- 文の主語がその語の時だけ使い、文の中のほかの名詞から親を NLI で選びます。親の語の意味は Gemma が選びます。
- 正解はこのノートに入っていません。手で書いた語の一覧もありません。

## 手順
1. 「ランタイム」→「ランタイムのタイプを変更」→ **L4 GPU**。
2. 鍵マークの `HF_TOKEN` にアクセスを許可（今までと同じ）。
3. 「ランタイム」→「すべてのセルを実行」。ドライブの許可を聞かれたら許可。
4. 終わると、ドライブの `AKANE_アーキテクチャ/E14_results/E14_results.json.gz` ができます。
5. 時間の見込み：1時間ほど（一般的な文の読み込み5分、継ぎ足し40分、直接見せる聞き方5分）。
   途中で切れても、もう一度「すべてのセルを実行」すれば、済んだ所（継ぎ足し）は飛ばします。
""")
code("""
import os, json, time, subprocess, sys, gzip, shutil, collections, re
import numpy as np
from google.colab import drive, userdata
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/AKANE_アーキテクチャ"
OUT = f"{BASE}/E14_results"
os.makedirs(OUT, exist_ok=True)
E112 = f"{BASE}/E11-2_results/E11-2_results.json"
WIKT = f"{BASE}/E10-1_results/E101_wiktionary.json.gz"   # E10-1 で読んだ名詞の辞書（新しく着く語に使う）
for p in (E112, WIKT):
    assert os.path.exists(p), "見つかりません: " + p
NLI_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"   # E6〜E11と同じ
GEMMA_ID = "google/gemma-2-2b-it"
GKB_ID = "community-datasets/generics_kb"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=False)
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
code("# 中身2（analysis/e14/e14_core.py と同じ。手書きの語の一覧はありません）\n" + CORE)
code("OLD = json.loads(r'''" + json.dumps(OLD, ensure_ascii=False) + "''')\n"
     "NEW = json.loads(r'''" + json.dumps(NEW, ensure_ascii=False) + "''')\n"
     "ITEMS = OLD + NEW\n"
     "r12 = json.load(open(E112))\n"
     "L = r12['L']\n"
     "CHOICE = {(w, k): v for w, k, v in r12['choice']}\n"
     "ST = {(w, k): tuple(v) for w, k, *v in r12['steps']}\n"
     "AR = {(w, k, C): tuple(v) for w, k, C, *v in r12['arrive']}\n"
     "# 確かめ：資料を足さない（srcs=()）と、E11-2 の6段の鎖と同じ\n"
     "CH12 = {i: nd for i, nd, lk in r12['chains']}\n"
     "bad = sum({(w, k) for (t, w, k) in follow(L, CHOICE, {}, it['ent'], 6, ())[0]} != {(w, k) for w, k, d in CH12[it['id']]} for it in ITEMS)\n"
     "assert bad == 0, f'資料なしの鎖が E11-2 と {bad} 問で違う'\n"
     "with gzip.open(WIKT, 'rt', encoding='utf-8') as f: WKD = json.load(f)['n']\n"
     "print(len(ITEMS), '問  語', len(L), '  資料なしの鎖は E11-2 と一致  辞書の見出し', len(WKD))")
md("## 1. 一般的な文（GenericsKB の全部の版）を読む（GPUなし、5分）")
code("""
from datasets import load_dataset
ds = load_dataset(GKB_ID, "generics_kb", split="train")   # 全部の版だけ（best は WordNet・ConceptNet から合成した文を含む）
cols = ds.column_names; print("列:", cols, "文の数:", len(ds))
df = ds.to_pandas(); del ds
scol = [c for c in cols if "sentence" in c][0]
if "source" in cols:
    print("出どころ:", df["source"].astype(str).value_counts().to_dict())
    bad = df["source"].astype(str).str.lower().str.contains("wordnet|conceptnet")   # 答え漏れ防止：出どころの欄の値で除く
    print("WordNet・ConceptNet と書かれた出どころの文（除く）:", int(bad.sum())); df = df[~bad]
df = df.sort_values("score", ascending=False) if "score" in cols else df
SENTS = df[scol].astype(str).tolist()
if "term" in cols:
    TERM = df["term"].astype(str).str.lower().tolist()
    BY_TERM = collections.defaultdict(list)
    for i, t in enumerate(TERM):
        if len(BY_TERM[t]) < 3 * N_GKB_READ: BY_TERM[t].append(i)
    print("見出しの語:", len(BY_TERM))
else:
    BY_TERM = None; print("term の列がないので、文を読んで探します（遅い）")
del df
TOK = re.compile(r"[a-z][a-z\\-]*")
def gkb_sents(words):
    # score の高い順に N_GKB_READ 文まで（主語かどうかは後で確かめる）
    if BY_TERM is not None:
        return {w: [SENTS[i] for i in BY_TERM.get(w, [])][:N_GKB_READ] for w in words}
    want = set(words); out = {w: [] for w in words}
    for s in SENTS:
        for w in set(TOK.findall(s.lower())) & want:
            if len(out[w]) < N_GKB_READ: out[w].append(s)
    return out
print(gkb_sents(["zoologist", "human"]))
""")
md("## 2. NLI と Gemma を読み込む（GPU、2〜4分）")
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
def nli_sent(trip):
    # (文, w, c)：前提＝文そのもの、仮説「W is a type of c.」
    return _nli([s for s, w, c in trip], [hyp(w, c) for s, w, c in trip])

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
def gemma_choice(jobs):
    # (w, k) → 頭の名詞 g の意味の番号。聞き方は E8n の方法D と同じ
    qs = [choice_prompt(w, L[w][k]["def"], L[w][k]["genus"], [x["def"] for x in L[L[w][k]["genus"]]]) for w, k in jobs]
    return _pick(qs, [len(L[L[w][k]["genus"]]) for w, k in jobs])
def gemma_sense(jobs):
    # (文, 語, [説明文]) → 文の中のその語の意味の番号
    return _pick([sense_in_sentence_prompt(s, w, ds) for s, w, ds in jobs], [len(ds) for s, w, ds in jobs])
SELF = {}
def self_write(words, B=32):
    todo = [w for w in words if w not in SELF]
    for i in range(0, len(todo), B):
        bt = todo[i:i + B]
        ps = [tok.apply_chat_template([{"role": "user", "content": self_prompt(w)}], tokenize=False, add_generation_prompt=True) for w in bt]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = gm.generate(**enc, max_new_tokens=SELF_TOKENS, do_sample=False)
        for w, row in zip(bt, g[:, enc["input_ids"].shape[1]:]):
            SELF[w] = {"raw": tok.decode(row, skip_special_tokens=True), "sent": None}
            SELF[w]["sent"] = clean_self(SELF[w]["raw"])
    return {w: SELF[w]["sent"] for w in words}
def yes_no(qs, B=32):
    out = []
    for i in range(0, len(qs), B):
        ps = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs[i:i + B]]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            r = gm(**enc, logits_to_keep=1).logits[:, -1, :].float()
        out.extend((r[:, YES] - r[:, NO]).cpu().numpy().round(3).tolist())
    return out
print("読み込み済み")
""")
md("## 3. 継ぎ足す（GPU、30〜50分）")
code("""
saved = load_gz("E14_grow")
if saved is not None:
    L, GR, SELF, STAT = saved["L"], saved["GR"], saved["self"], saved["stat"]
    CHOICE = {(w, k): v for w, k, v in saved["choice"]}
    ST = {(w, k): tuple(v) for w, k, *v in saved["steps"]}
    AR = {(w, k, C): tuple(v) for w, k, C, *v in saved["arrive"]}
    ARG = {(w, i, C): tuple(v) for w, i, C, *v in saved["arrive_g"]}
    print("ドライブから読み込み:", STAT)
else:
    GR, ARG = {}, {}
    t = time.time()
    STAT = grow(ITEMS, L, CHOICE, ST, AR, GR, ARG, WKD, nlp, nli, nli_sent, gemma_choice, gemma_sense, gkb_sents, self_write)
    print("足した物:", STAT, f"{time.time() - t:.0f}秒")
    save_gz({"L": L, "GR": GR, "self": SELF, "stat": STAT,
             "choice": [[w, k, v] for (w, k), v in CHOICE.items()],
             "steps": [[w, k, *v] for (w, k), v in ST.items()],
             "arrive": [[w, k, C, *v] for (w, k, C), v in AR.items()],
             "arrive_g": [[w, i, C, *v] for (w, i, C), v in ARG.items()]}, "E14_grow")
CHAINS = chains(ITEMS, L, CHOICE, GR)
print("点の種類と段:", dict(sorted(collections.Counter((t, d) for nd, lk in CHAINS.values() for t, w, k, d in nd).items())))
ex = [w for w in ("zoologist", "human", "person", "granddaughter") if GR.get(w)]
for w in ex: print(w, [(g["src"], g["sent"], g["parent"], g["step"][0]) for g in GR[w]])
""")
md("## 4. 対照：同じ文を Gemma に直接見せて聞く（⑪x、GPU、5分）")
code("""
def fact_prompt(facts, stmt):   # E10-2 と同じ
    body = "\\n".join(f"- {f}" for f in facts) if facts else "(no facts found)"
    return (f"Facts:\\n{body}\\n\\n"
            f"Using these facts and your own knowledge, is the following statement true? Answer Yes or No only.\\n"
            f"Statement: {stmt}")
DIRECT = load_gz("E14_direct")
if DIRECT is None:
    fg = [[g["sent"] for g in GR.get(it["ent"], []) if g["src"] == "g"] for it in ITEMS]
    fgs = [[g["sent"] for g in GR.get(it["ent"], [])] for it in ITEMS]
    DIRECT = {"ids": [it["id"] for it in ITEMS],
              "g": yes_no([fact_prompt(f, it["stmt"]) for f, it in zip(fg, ITEMS)]),
              "gs": yes_no([fact_prompt(f, it["stmt"]) for f, it in zip(fgs, ITEMS)]),
              "n_g": [len(f) for f in fg], "n_gs": [len(f) for f in fgs]}
    save_gz(DIRECT, "E14_direct")
print("物の語に一般的な文があった問題:", sum(1 for n in DIRECT["n_g"] if n), "/", len(ITEMS))
""")
md("## 5. 保存")
code("""
res = {"meta": {"nli": NLI_ID, "gemma": GEMMA_ID, "gkb": GKB_ID, "base": "E11-2_results/E11-2_results.json",
                "dict_new_words": "E10-1_results/E101_wiktionary.json.gz", "max_depth": MAX_DEPTH, "k_gkb": K_GKB,
                "n_gkb_read": N_GKB_READ, "date": time.strftime("%Y-%m-%d"), "stat": STAT},
       "L": L, "GR": GR, "self": SELF,
       "choice": [[w, k, v] for (w, k), v in CHOICE.items()],
       "steps": [[w, k, *v] for (w, k), v in ST.items()],
       "arrive": [[w, k, C, *v] for (w, k, C), v in AR.items()],
       "arrive_g": [[w, i, C, *v] for (w, i, C), v in ARG.items()],
       "chains": [[i, nd, lk] for i, (nd, lk) in CHAINS.items()],
       "direct": DIRECT}
save_gz(res, "E14_results")
""")
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells), "size", os.path.getsize(sys.argv[1]))
