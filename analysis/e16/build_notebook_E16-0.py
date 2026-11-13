"""E16-0 Colabノートを作る。python3 build_notebook_E16-0.py e160_items.json out.ipynb
結果はドライブ AKANE_アーキテクチャ/E16-0_results/E160_results.json.gz。計画：sims/pbit/e160/PLAN_E16-0.md"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(sys.argv[1]))
CORE = open(f"{HERE}/e160_core.py").read()
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md(f"""
# AKANE E16-0：Gemma は「逆向きの問い」に弱いか

同じ「AはBの一種」を、2つの向きの4択で聞きます。
- 前向き：「Abolitionist is a type of which? A. person B. animal …」（物 → 分類）
- 逆向き：「Which is a type of person? A. abolitionist B. llama …」（分類 → 物）
- 組は {len(D['items'])}（物 {len(D['ents'])}、分類 46）。外れの選択肢はやさしい（でたらめ）と難しい（兄弟の分類）の2通り。正解の位置は A〜D に1回ずつ。
- おまけ：答えを自由に書かせる版（前向き「〜は何の一種？」、逆向き「〜の種類を20挙げて」）。
- 手で書いた語の一覧は使っていません。

## 手順
1. 「ランタイム」→「ランタイムのタイプを変更」→ **L4 GPU**（T4でも動きます）。
2. 鍵マークの `HF_TOKEN` にアクセスを許可（これまでと同じ）。
3. 「ランタイム」→「すべてのセルを実行」。ドライブの許可を聞かれたら許可。
4. 15分ほどで、ドライブの `AKANE_アーキテクチャ/E16-0_results/E160_results.json.gz` ができます。最後のセルが速報を出します。
""")
code("""
import os, json, time, gzip, subprocess, sys
import numpy as np
from google.colab import drive, userdata
drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/AKANE_アーキテクチャ/E16-0_results"
os.makedirs(OUT, exist_ok=True)
GEMMA_ID = "google/gemma-2-2b-it"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nltk"], check=False)
import nltk; nltk.download("wordnet", quiet=True)
from nltk.corpus import wordnet as wn
""")
code("# 中身（sims/pbit/e160/e160_core.py と同じ）\n" + CORE)
code("DATA = json.loads(" + repr(json.dumps(D, ensure_ascii=False)) + ")\nITEMS, CATS, ENTS = DATA['items'], DATA['cats'], DATA['ents']\nMC = mc_set(ITEMS, CATS)\nprint('4択の問い', len(MC)); print(MC[0]['q']); print(); print(MC[-1]['q'])")
md("## 1. Gemma に4択で聞く（5〜10分）")
code("""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), padding_side="left")
gm = AutoModelForCausalLM.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), torch_dtype=torch.float16, device_map="cuda").eval()
chat = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
LT = [tok.encode(c, add_special_tokens=False)[0] for c in LET]
assert len(set(LT)) == 4
def letters(qs, B=32):
    out = []; t = time.time()
    for i in range(0, len(qs), B):
        enc = tok([chat(q) for q in qs[i:i + B]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            r = gm(**enc, logits_to_keep=1).logits[:, -1, :].float()
        out.extend(torch.log_softmax(r[:, LT], -1).cpu().numpy().round(4).tolist()); del r
        if (i // B) % 50 == 0: print(f"  {i}/{len(qs)}  {time.time() - t:.0f}秒")
    return out
LP = letters([m["q"] for m in MC])
""")
md("## 2. 自由に書かせる（5分）")
code("""
def write(qs, n_new, B=32):
    out = []
    for i in range(0, len(qs), B):
        enc = tok([chat(q) for q in qs[i:i + B]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = gm.generate(**enc, max_new_tokens=n_new, do_sample=False)
        out.extend(tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return out
OF = write([open_f_prompt(x) for x in ENTS], 10)
OR = write([open_r_prompt(CATS[c]) for c in CATS], 240, B=16)
print(ENTS[0], "→", OF[0]); print(list(CATS.values())[0], "→", parse_list(OR[0])[:8])
""")
md("## 3. 自由に書いた答えを WordNet で確かめる（参考。WordNet の採点は狭いことがある）")
code("""
def syns(w):
    return wn.synsets(w.strip().replace(" ", "_"), pos=wn.NOUN)
def ups(s):
    return set(s.closure(lambda z: z.hypernyms() + z.instance_hypernyms())) | {s}
def f_ok(ent, ans):
    a = ans.strip().strip(".").lower()
    cand = syns(a) or syns(a.split()[-1]) if a else []
    U = set().union(*[ups(s) - {s} for s in syns(ent)]) if syns(ent) else set()
    return bool(set(cand) & U)
def r_ok(cat, word):
    C = wn.synset(cat)
    return any(C in ups(s) and s != C for s in (syns(word) or (syns(word.split()[-1]) if word.split() else [])))
F_OK = [f_ok(x, a) for x, a in zip(ENTS, OF)]
R_LIST = [parse_list(t) for t in OR]
R_OK = [[r_ok(c, w) for w in L] for c, L in zip(CATS, R_LIST)]
print("前向き 自由回答の正しさ", round(float(np.mean(F_OK)), 3))
print("逆向き 自由回答の正しさ（挙げた物のうち）", round(float(np.mean([v for L in R_OK for v in L])), 3), " 1分類あたり挙げた数", round(float(np.mean([len(L) for L in R_LIST])), 1))
""")
md("## 4. 保存と速報")
code("""
res = dict(model=GEMMA_ID, mc=[dict(m, lp=l) for m, l in zip(MC, LP)], open_f=OF, open_r=OR, f_ok=F_OK, r_list=R_LIST, r_ok=R_OK)
for m in res["mc"]: m.pop("q")
with gzip.open(f"{OUT}/E160_results.json.gz", "wt", encoding="utf-8") as f: json.dump(res, f, ensure_ascii=False)
print("保存しました", OUT)
acc = {}
for m in res["mc"]:
    acc.setdefault((m["d"], m["lv"]), []).append(int(np.argmax(m["lp"]) == m["pos"]))
for k in sorted(acc): print("前向き" if k[0] == "f" else "逆向き", "やさしい" if k[1] == "easy" else "難しい", round(float(np.mean(acc[k])), 3))
""")
nb = dict(nbformat=4, nbformat_minor=0, metadata=dict(colab=dict(provenance=[]), kernelspec=dict(name="python3", display_name="Python 3"), accelerator="GPU"), cells=cells)
json.dump(nb, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
print("書きました", sys.argv[2], len(cells), "セル")
