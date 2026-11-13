"""E14-2 Colabノート（継ぎ足しの資料を Wikipedia の記事の最初の1文に替える。判定は Gemma、継ぎ足しは分析で1段まで）。
python3 build_notebook_E14-2.py out.ipynb
計画：plans/E14-2_wikipedia_first_sentence.md。材料を取るだけ（比べは e14_analyze.py で別に）。"""
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
# AKANE E14-2 v2：継ぎ足しの資料を Wikipedia の最初の1文に替える（取り方を直した版）

v1 では Wikipedia に「アクセスが多すぎ」と断られ、半分以上の語が取れませんでした。v2 は 1回で20語まとめて聞き、1本ずつ1秒あけ、断られたら待ちます。取れない語が5%を超えたら止まります。

このノートは **材料を取るだけ** です（比べは結果のファイルを読んで別に行います）。計画：plans/E14-2_wikipedia_first_sentence.md。

- E11-2 の6段の鎖をそのまま読み、鎖の全部の語に、**Wikipedia のその語の記事の最初の1文**を足します（Wikipedia の公開 API で記事の冒頭を取ります。曖昧さ回避のページは使いません）。
- 文の主語がその語の時だけ使い、文の中のほかの名詞から親を選びます。判定（「この文は X が Y の一種だと言っていますか？」）と親の語の意味選びは Gemma です（E14-1 と同じ）。
- 一般的な文（GenericsKB）・自作文は使いません。
- 正解はこのノートに入っていません。手で書いた語の一覧もありません。

## 手順
1. 「ランタイム」→「ランタイムのタイプを変更」→ **L4 GPU**。
2. 鍵マークの `HF_TOKEN` にアクセスを許可（今までと同じ）。
3. 「ランタイム」→「すべてのセルを実行」。ドライブの許可を聞かれたら許可。
4. 終わると、ドライブの `AKANE_アーキテクチャ/E14-2_results/E142v2_grow.json.gz` ができます。
5. 時間の見込み：40〜60分（Wikipedia は1段に数分、継ぎ足し30〜40分）。
   途中で切れても、もう一度「すべてのセルを実行」すれば、済んだ所（継ぎ足し）は飛ばします。
""")
code("""
import os, json, time, subprocess, sys, gzip, shutil, collections, re
import numpy as np
from google.colab import drive, userdata
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/AKANE_アーキテクチャ"
OUT = f"{BASE}/E14-2_results"
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
# E14-1：継ぎ足しの文の判定を Gemma に（聞き方は1つだけ、PLAN_E14-1.md）
def says_prompt(s, w, c):
    return f"Sentence: {s}\\n\\nDoes this sentence say that a \\"{w}\\" is a kind of \\"{c}\\"? Answer Yes or No only."
def gemma_says(trip, B=32):
    d = yes_no([says_prompt(s, w, c) for s, w, c in trip], B)
    return [(round(float(1 / (1 + np.exp(-x))), 4), 0.0) for x in d]
nli_sent = gemma_says
gkb_sents = lambda words: {}      # E14-2 では使わない
self_write = lambda words: {}     # E14-2 では使わない
print(says_prompt("Granddaughters trump even a four-homer win any time.", "granddaughter", "win"))
print(gemma_says([("Granddaughters trump even a four-homer win any time.", "granddaughter", "win"),
                  ("Hominids are primates with no tails.", "hominid", "primate")]))
# E14-2：Wikipedia の記事の最初の1文（公開の要約 API。曖昧さ回避など standard でないページは使わない）
import requests
UA = {"User-Agent": "AKANE-research-colab/1.1 (non-commercial research; low request rate)"}
API = "https://en.wikipedia.org/w/api.php"
WIKI = load_gz("E142v2_wiki") or {}
N_REQ, N_429 = [0], [0]
def _wiki_batch(words):
    # v2：1回の問い合わせで20語まで（記事の冒頭の説明だけ、平文）。1本ずつ・1秒あける。断られたら言われた時間だけ待つ。
    # 曖昧さ回避のページ（pageprops に disambiguation）と記事のない語は None。転送（hominid→Hominidae）はたどる。
    titles = {w: w[:1].upper() + w[1:] for w in words}
    params = {"action": "query", "format": "json", "formatversion": "2", "prop": "extracts|pageprops",
              "exintro": "1", "explaintext": "1", "exlimit": "20", "ppprop": "disambiguation", "redirects": "1",
              "maxlag": "5", "titles": "|".join(titles.values())}
    for i in range(8):
        time.sleep(1.0)
        try:
            N_REQ[0] += 1
            r = requests.get(API, params=params, headers=UA, timeout=30)
            if r.status_code == 429 or r.status_code >= 500 or "maxlag" in r.text[:200]:
                N_429[0] += 1
                time.sleep(float(r.headers.get("Retry-After", 0) or 0) or min(60, 5 * 2 ** i))
                continue
            if r.status_code != 200:
                time.sleep(min(60, 5 * 2 ** i)); continue
            q = r.json().get("query", {})
            go = {}
            for x in q.get("normalized", []): go[x["from"]] = x["to"]
            red = {x["from"]: x["to"] for x in q.get("redirects", [])}
            pages = {p["title"]: p for p in q.get("pages", [])}
            out = {}
            for w, t in titles.items():
                t = go.get(t, t); t = red.get(t, t)
                p = pages.get(t)
                if not p or p.get("missing") or p.get("invalid") or "disambiguation" in (p.get("pageprops") or {}):
                    out[w] = None
                else:
                    out[w] = {"title": p["title"], "extract": p.get("extract") or ""}
            return out
        except Exception:
            time.sleep(min(60, 5 * 2 ** i))
    return {w: {"error": True} for w in words}
def wiki_sents(words):
    todo = [w for w in words if w not in WIKI or (WIKI[w] or {}).get("error")]
    if todo:
        t = time.time()
        for s in range(0, len(todo), 20):
            WIKI.update(_wiki_batch(todo[s:s + 20]))
        save_gz(WIKI, "E142v2_wiki")
        print(f"  Wikipedia {len(todo)}語 {time.time() - t:.0f}秒  取れた {sum(1 for w in todo if WIKI[w] and WIKI[w].get('extract'))}  "
              f"記事なし・曖昧さ回避 {sum(1 for w in todo if WIKI[w] is None)}  失敗 {sum(1 for w in todo if WIKI[w] and WIKI[w].get('error'))}  "
              f"問い合わせ {N_REQ[0]}  待たされた {N_429[0]}")
        bad = sum(1 for w in todo if WIKI[w] and WIKI[w].get('error'))
        assert bad <= 0.05 * len(todo), "Wikipedia に断られ続けています。少し時間をおいて、もう一度「すべてのセルを実行」してください（取れた所は残ります）"
    out = {}
    for w in words:
        r = WIKI.get(w)
        if r and r.get("extract"):
            sents = list(nlp(r["extract"][:600]).sents)
            if sents:
                out[w] = sents[0].text.strip()
    return out
print(wiki_sents(["zoologist", "hominid", "kitten", "ant", "artist", "bathtub", "mercury"]))
print("読み込み済み")
""")
md("## 3. 継ぎ足す（GPU、40〜60分）。終わったら E142v2_grow.json.gz がドライブにできます")
code("""
saved = load_gz("E142v2_grow")
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
    STAT = grow(ITEMS, L, CHOICE, ST, AR, GR, ARG, WKD, nlp, nli, nli_sent, gemma_choice, gemma_sense, gkb_sents, self_write, wiki_sents=wiki_sents)
    print("足した物:", STAT, f"{time.time() - t:.0f}秒")
    save_gz({"L": L, "GR": GR, "self": SELF, "stat": STAT,
             "choice": [[w, k, v] for (w, k), v in CHOICE.items()],
             "steps": [[w, k, *v] for (w, k), v in ST.items()],
             "arrive": [[w, k, C, *v] for (w, k, C), v in AR.items()],
             "arrive_g": [[w, i, C, *v] for (w, i, C), v in ARG.items()]}, "E142v2_grow")
CHAINS = chains(ITEMS, L, CHOICE, GR)
print("点の種類と段:", dict(sorted(collections.Counter((t, d) for nd, lk in CHAINS.values() for t, w, k, d in nd).items())))
ex = [w for w in ("zoologist", "human", "person", "granddaughter") if GR.get(w)]
for w in ex: print(w, [(g["src"], g["sent"], g["parent"], g["step"][0]) for g in GR[w]])
""")
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells), "size", os.path.getsize(sys.argv[1]))
