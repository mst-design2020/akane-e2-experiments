"""E12-1 Colabノート。python3 build_notebook_E12-1.py items_all.json out.ipynb
E12 で WordNet の順でも聞いた 5,000問（くじ seed 0）を使う。正解はノートに入れない。"""
import json, os, sys, gzip, base64
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
src, dst = sys.argv[1], sys.argv[2]
it = json.load(open(src))["items"]
rng = np.random.default_rng(0)
hidx = [i for i, x in enumerate(it) if x["kind"] == "head"]
G0 = sorted(rng.choice(hidx, size=5000, replace=False).tolist())   # E12 の参考と同じ5,000問
ROWS = [[i, it[i]["word"], it[i]["defn"], it[i]["head"]] for i in G0]
HEADS = {}
for i in G0:
    HEADS.setdefault(it[i]["head"], it[i]["cand_defs"])
blob = base64.b64encode(gzip.compress(json.dumps({"rows": ROWS, "heads": HEADS}).encode())).decode()
CORE = open(f"{HERE}/e121_core.py").read()
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md(f"""
# AKANE E12-1：Gemma は WordNet の説明文を覚えていて当てただけか

このノートは **材料を取るだけ** です。計画：plans/E12-1_memorized_or_not.md。E12 と同じ5,000問（頭の語の種類 {len(HEADS):,}）。

- B：定義される語の名前を隠して聞く。
- C：名前を隠し、説明文と候補の説明文を別のAI（Qwen 2.5 1.5B）で言い換えてから聞く。
- D：説明文の前半を見せて後半を Gemma に言わせ、覚えている度合いを測る。
- 正解はこのノートに入っていません。手で書いた語の一覧もありません。

## 手順
1. **L4 GPU**、鍵マークの `HF_TOKEN` を許可、「すべてのセルを実行」。
2. 終わると `AKANE_アーキテクチャ/E12-1_results/E12-1_results.json.gz` ができます。
3. 時間の見込み：25〜45分（言い換えが一番長い）。途中で切れても、済んだ所は飛ばします。
""")
code("""
import os, json, time, gzip, base64
import numpy as np, torch
from google.colab import drive, userdata
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/AKANE_アーキテクチャ"
OUT = f"{BASE}/E12-1_results"
os.makedirs(OUT, exist_ok=True)
GEMMA_ID = "google/gemma-2-2b-it"
PARA_ID = "Qwen/Qwen2.5-1.5B-Instruct"   # 言い換え役（Gemma と別の会社のAI）
NLI_ID = "cross-encoder/nli-deberta-v3-base"   # 言い換えが同じ意味かの目安
""")
code("# 中身（analysis/e12/e121_core.py と同じ）\n" + CORE)
code("DATA = json.loads(gzip.decompress(base64.b64decode('" + blob + "')))\n"
     "ROWS, HEADS = DATA['rows'], DATA['heads']\n"
     "print(len(ROWS), '問  頭の語', len(HEADS))")
md("## 1. 言い換え（Qwen、GPU、15〜25分）")
code("""
from transformers import AutoTokenizer, AutoModelForCausalLM
def gen(model_id, prompts, ck, max_new=60, B=48, token=None):
    if os.path.exists(ck):
        c = json.load(open(ck))
        if c.get("n") == len(prompts) and c.get("head") == prompts[0][:200]:
            print("ドライブから読み込み", ck); return c["out"]
    tk = AutoTokenizer.from_pretrained(model_id, token=token, padding_side="left")
    m = AutoModelForCausalLM.from_pretrained(model_id, token=token, torch_dtype=torch.float16, device_map="cuda").eval()
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i])); res = [None] * len(prompts); t = time.time()
    for s in range(0, len(order), B):
        idx = order[s:s + B]
        ps = [tk.apply_chat_template([{"role": "user", "content": prompts[i]}], tokenize=False, add_generation_prompt=True) for i in idx]
        enc = tk(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            o = m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tk.pad_token_id or tk.eos_token_id)
        for r, i in zip(o[:, enc["input_ids"].shape[1]:], idx):
            res[i] = tk.decode(r, skip_special_tokens=True).strip().split("\\n")[0].strip()
        if (s // B) % 50 == 0: print(f"  {s}/{len(prompts)}  {time.time() - t:.0f}秒")
    del m; torch.cuda.empty_cache()
    json.dump({"n": len(prompts), "head": prompts[0][:200], "out": res}, open(ck, "w"))
    return res

CDEFS = sorted({d for v in HEADS.values() for d in v[:MAX_SENSES]})
P_T = gen(PARA_ID, [para_prompt(r[2], keep=r[3]) for r in ROWS], f"{OUT}/ck_para_target.json")
P_C = dict(zip(CDEFS, gen(PARA_ID, [para_prompt(d) for d in CDEFS], f"{OUT}/ck_para_cands.json")))
keep = [has_word(p, r[3]) for p, r in zip(P_T, ROWS)]
print("説明文の言い換えに頭の語が残った割合:", np.mean(keep).round(3))
print("例:", ROWS[0][2], " → ", P_T[0])
""")
md("## 2. 言い換えが同じ意味かの目安（NLI、両向き、GPU、2〜4分）")
code("""
from transformers import AutoModelForSequenceClassification
ntk = AutoTokenizer.from_pretrained(NLI_ID)
nm = AutoModelForSequenceClassification.from_pretrained(NLI_ID).to("cuda").eval()
ENT = [i for i, l in nm.config.id2label.items() if l.lower().startswith("entail")][0]
def ent(pairs, B=128):
    out = []
    for s in range(0, len(pairs), B):
        a, b = zip(*pairs[s:s + B])
        enc = ntk(list(a), list(b), return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        with torch.no_grad():
            out += torch.softmax(nm(**enc).logits.float(), -1)[:, ENT].cpu().numpy().round(4).tolist()
    return out
def both(xs, ys):
    f, b = ent(list(zip(xs, ys))), ent(list(zip(ys, xs)))
    return [round(min(u, v), 4) for u, v in zip(f, b)]
NLI_T = both([r[2] for r in ROWS], P_T)
NLI_C = dict(zip(CDEFS, both(CDEFS, [P_C[d] for d in CDEFS])))
del nm; torch.cuda.empty_cache()
print("両向きで同じ意味（0.5以上）の割合  説明文:", np.mean(np.array(NLI_T) >= .5).round(3), " 候補:", np.mean(np.array(list(NLI_C.values())) >= .5).round(3))
""")
md("## 3. Gemma に聞く（B・C）と、後半を言わせる（D）（GPU、8〜15分）")
code("""
tok = AutoTokenizer.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), padding_side="left")
gm = AutoModelForCausalLM.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), torch_dtype=torch.float16, device_map="cuda").eval()
NUM = [tok.encode(str(i), add_special_tokens=False)[0] for i in range(1, MAX_SENSES + 1)]
assert len(set(NUM)) == MAX_SENSES
def ask(jb, ck, B=32):
    if os.path.exists(ck):
        c = json.load(open(ck))
        if c.get("n") == len(jb) and c.get("head") == jb[0][0][:200]:
            print("ドライブから読み込み", ck); return c["out"]
    order = sorted(range(len(jb)), key=lambda i: len(jb[i][0])); res = [None] * len(jb)
    for s in range(0, len(order), B):
        idx = order[s:s + B]
        ps = [tok.apply_chat_template([{"role": "user", "content": jb[i][0]}], tokenize=False, add_generation_prompt=True) for i in idx]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            last = gm(**enc, logits_to_keep=1).logits[:, -1, :].float()
        for row, i in zip(last, idx):
            res[i] = torch.softmax(row[NUM[:len(jb[i][1])]], -1).cpu().numpy().round(4).tolist()
    json.dump({"n": len(jb), "head": jb[0][0][:200], "out": res}, open(ck, "w"))
    return res
def merge2(jb, probs):
    out = []
    for i in range(0, len(jb), 2):
        acc = np.zeros(len(jb[i][1]))
        for j in (i, i + 1):
            for pos, orig in enumerate(jb[j][1]): acc[orig] += probs[j][pos] / 2
        out.append(acc.round(4).tolist())
    return out
jbB = jobs_hidden([(r[2], r[3], HEADS[r[3]]) for r in ROWS], seed=1)
GB = merge2(jbB, ask(jbB, f"{OUT}/ck_B.json"))
jbC = jobs_hidden([(p, r[3], [P_C[d] for d in HEADS[r[3]][:MAX_SENSES]]) for p, r in zip(P_T, ROWS)], seed=1)
GC = merge2(jbC, ask(jbC, f"{OUT}/ck_C.json"))
del gm; torch.cuda.empty_cache()
# D：前半を見せて後半を言わせる（ここだけ文を作らせる）
HALF = [split_half(r[2]) for r in ROWS]
DI = [i for i, h in enumerate(HALF) if h]
DOUT = gen(GEMMA_ID, [recall_prompt(ROWS[i][1], HALF[i][0]) for i in DI], f"{OUT}/ck_D.json", max_new=40, token=userdata.get("HF_TOKEN"))
MEM = [None] * len(ROWS)
for i, o in zip(DI, DOUT): MEM[i] = round(lcs_recall(HALF[i][1], o), 4)
print("後半を言わせた数", len(DI), " 後半の語を同じ順で出せた割合の平均", np.mean([m for m in MEM if m is not None]).round(3))
""")
md("## 4. 保存")
code("""
res = {"meta": {"gemma": GEMMA_ID, "para": PARA_ID, "nli": NLI_ID, "date": time.strftime("%Y-%m-%d"), "n": len(ROWS)},
       "idx": [r[0] for r in ROWS], "gemma_B": GB, "gemma_C": GC, "para_target": P_T, "para_target_keep": keep,
       "nli_target": NLI_T, "para_cands": P_C, "nli_cands": NLI_C, "mem": MEM, "recall_out": dict(zip([ROWS[i][0] for i in DI], DOUT))}
path = f"{OUT}/E12-1_results.json.gz"
with gzip.open(path, "wt") as f: json.dump(res, f)
print("保存:", path, round(os.path.getsize(path) / 1e6, 2), "MB")
""")
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(dst, "w"), ensure_ascii=False, indent=1)
print("cells", len(cells), "size", os.path.getsize(dst))
