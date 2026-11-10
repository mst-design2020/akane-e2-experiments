"""E6 Colabノートを作る。python3 build_notebook_E6.py out.ipynb"""
import json, sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS = [{k: x[k] for k in ("id", "ent", "stmt", "wn_defs")} for x in json.load(open(f"{HERE}/e6_items.json"))]  # 正解は入れない
CORE = open(f"{HERE}/e6_colab.py").read().split('if __name__ == "__main__":')[0]
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md("""
# AKANE E6：Gemmaの自信ある間違いを、辞書の説明文で直せるか

このノートは **答えを取るだけ** です（比べは結果のファイルを読んで別に行います）。計画：sims/e6/PLAN_E6.md。

- 700個の「〜は〜の一種か」に、辞書の説明文（WordNet／Wiktionary）を足して、Gemma-2 2Bにもう一度聞き、NLI（DeBERTa大）でも判定します。
- 正解はこのノートに入っていません。

## 手順
1. 「ランタイム」→「ランタイムのタイプを変更」→ **L4 GPU**（なければ T4）。
2. 鍵マークの `HF_TOKEN` にアクセスを許可。
3. 「ランタイム」→「すべてのセルを実行」。ドライブの許可を聞かれたら許可。
4. 終わると、ドライブの `AKANE_アーキテクチャ/E6_results/E6_results.json` ができます。10〜20分の見込み。
""")
code("""
import os, json, time, urllib.parse, requests
import numpy as np
from google.colab import drive, userdata
drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/AKANE_アーキテクチャ/E6_results"
os.makedirs(OUT, exist_ok=True)
GEMMA_ID = "google/gemma-2-2b-it"
NLI_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"   # E2-12・E3と同じ
""")
code("# 中身（sims/e6/e6_colab.py と同じ）\n" + CORE)
code("ITEMS = json.loads(r'''" + json.dumps(ITEMS, ensure_ascii=False) + "''')\nprint(len(ITEMS), '個')")
md("## 1. Wiktionaryの説明文を取る（5〜10分。取ったものはドライブに残し、もう一度実行すると取れていない語だけ取り直す）")
code("""
wk_path = f"{OUT}/E6_wiktionary.json"
words = sorted(set(it["ent"] for it in ITEMS))
import collections
WK = json.load(open(wk_path)) if os.path.exists(wk_path) else {}
todo = [w for w in words if not WK.get(w)]   # 取れていない語だけ取り直す（もう一度実行すると続きから）
print("取れている語:", len(words) - len(todo), " これから取る語:", len(todo))
S = requests.Session(); S.headers["User-Agent"] = "research-notebook/0.2 (non-commercial; python-requests)"
STATUS = collections.Counter(); FIRST_ERR = {}
def req(url, **kw):
    for k in range(6):                      # 断られたら待って取り直す
        try:
            r = S.get(url, timeout=30, **kw)
        except Exception as e:
            STATUS["通信エラー"] += 1; time.sleep(2 ** k); continue
        STATUS[r.status_code] += 1
        if r.status_code == 200:
            return r
        FIRST_ERR.setdefault(r.status_code, r.text[:300])
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(float(r.headers.get("Retry-After") or 0) or 2 ** (k + 1)); continue
        return None
    return None
def get_json(w):
    r = req("https://en.wiktionary.org/api/rest_v1/page/definition/" + urllib.parse.quote(w), params={"redirect": "true"})
    return r.json() if r is not None else None
def get_wikitext(w):  # 予備（上で取れなかった語だけ）
    r = req("https://en.wiktionary.org/w/api.php", params={"action": "parse", "page": w, "prop": "wikitext",
            "format": "json", "formatversion": 2, "redirects": 1})
    return r.json().get("parse", {}).get("wikitext", "") if r is not None else ""
for i in range(0, len(todo), 50):          # 50語ごとにドライブへ保存
    WK.update(fetch_all(todo[i:i + 50], get_json, get_wikitext, log=lambda s: None, sleep=0.5))
    json.dump(WK, open(wk_path, "w"), ensure_ascii=False)
    print(f"{min(i + 50, len(todo))}/{len(todo)}  取れた語の合計 {sum(1 for w in words if WK.get(w))}  返事の種類 {dict(STATUS)}")
for it in ITEMS: it["wk_defs"] = WK.get(it["ent"], [])
got = sum(1 for w in words if WK.get(w))
print("説明文が取れた語:", got, "/", len(words), " 返事の種類:", dict(STATUS))
print("例:", ITEMS[0]["ent"], WK.get(ITEMS[0]["ent"]))
for code_, txt in FIRST_ERR.items(): print("断られた例", code_, txt)
if got < len(words) * 0.7:
    raise RuntimeError("説明文の取れた語が7割未満です。この出力をClaudeに見せてください（もう一度このセルを実行すると続きから取ります）。")
""")
md("## 2. Gemmaにもう一度聞く・NLIで判定する")
code("""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), padding_side="left")
gm = AutoModelForCausalLM.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), torch_dtype=torch.float16, device_map="cuda").eval()
YES = tok.encode("Yes", add_special_tokens=False)[0]; NO = tok.encode("No", add_special_tokens=False)[0]
def gemma_logits(qs, B=16):
    out = []
    for i in range(0, len(qs), B):
        ps = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs[i:i + B]]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            last = gm(**enc, logits_to_keep=1).logits[:, -1, :].float()
        out += (last[:, YES] - last[:, NO]).cpu().numpy().tolist()
    return out
ntok = AutoTokenizer.from_pretrained(NLI_ID)
nm = AutoModelForSequenceClassification.from_pretrained(NLI_ID).cuda().eval()
lab = {v.lower(): int(k) for k, v in nm.config.id2label.items()}
ORDER = [lab["entailment"], lab["neutral"], lab["contradiction"]]
def nli_probs(P, H, B=32):
    out = []
    for i in range(0, len(P), B):
        enc = ntok(P[i:i + B], H[i:i + B], return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        with torch.no_grad():
            pr = torch.softmax(nm(**enc).logits.float(), -1)[:, ORDER]
        out += pr.cpu().numpy().tolist()
    return out
t = time.time()
res = run(ITEMS, gemma_logits, nli_probs)
path = f"{OUT}/E6_results.json"
json.dump({"meta": {"mock": False, "gemma": GEMMA_ID, "nli": NLI_ID, "n": len(res)}, "items": res}, open(path, "w"), ensure_ascii=False)
print("保存:", path, f"{time.time() - t:.0f}秒")
g = np.array([[it["gemma"][k] for k in ["none", "S1", "S2", "S3"]] for it in res])
print("「はい」の割合 説明文なし/S1/S2/S3:", np.round((g > 0).mean(0), 3))
""")
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells))
