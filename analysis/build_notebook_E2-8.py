# E2-8 の Colab ノートを作る。使い方: python3 build_notebook.py 出力.ipynb 出力.py
import json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))
md("""
# AKANE E2-8：3つ目の分野（病気の分類）で「疑う網」を確かめる

問い：Gemmaの作るルールに外れが多い分野でも、ルールを疑う網は「矛盾を数えるだけ」より間違いを拾えるか。
あわせて、**どの組を確かめるかもGemmaに選ばせる**。

Gemmaへの質問の例：
- 口の答え：「Measles is a viral infection.（はしかはウイルス感染症）」は本当？
- ルール："Is it true that no cancer or other tumor is a nervous system disease?"（がんで、しかも神経の病気、というものはない？）
- 組選び：「がん」と見せて、他の21の分類から「同時にはあり得ないもの」の番号を挙げさせる

正解は研究者が作っている病気の分類表（Human Disease Ontology、2026-10-29版）。

**このノートでやること**：Gemmaに約3,800問を聞いて、答えをドライブに保存するだけ。計算はこちらで行う。
**時間の目安**（T4 GPU）：10〜20分。計画は plans/E2-8_disease.md（結果を見る前に固定）。
""")
md("""
## 0. 準備
1. 「ランタイム」→「ランタイムのタイプを変更」→ **T4 GPU**。
2. 鍵マークの `HF_TOKEN`（前と同じもの）にアクセスを許可。
3. 「ランタイム」→「すべてのセルを実行」。
""")
code("""
# ===== 設定 =====
USE_MOCK_MOUTH = False      # True: 作り物の口（動作確認用。結果は研究に使わない）
SAVE_TO_DRIVE  = True
MODEL_ID = "google/gemma-2-2b-it"
DATA_SEED = 0
""")
code("""
import os, json, time, re, itertools, collections
import numpy as np
OUT = "/content/E2_results" if os.path.exists("/content") else "E2_results"
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = "/content/drive/MyDrive/AKANE_アーキテクチャ/E2_results"
os.makedirs(OUT, exist_ok=True)
print("保存先:", OUT)
""")
md("## 1. 病気のテストを作る（分類表をダウンロードして作る）")
code(open(os.path.join(HERE, "data_dis.py")).read())
md("""
## 2. 質問を作る
- 口の答え：全部の問題（"Is the following statement true?"、前と同じ聞き方）
- ルール：ラベルの組231組 × 順番2通り（"Is it true that no X is a Y?"、はい＝両立しない）
- 組選び：ラベル22個それぞれ（Gemmaに番号を書かせる）
""")
code("""
MOUTH_Q = [f"Is the following statement true? Answer Yes or No only.\\nStatement: {s}" for s in STMT]
RULE_QS, RULE_Q = [], []
for j, (p, q) in enumerate(PAIRS):
    for o, (x, y) in enumerate([(p, q), (q, p)]):
        RULE_QS.append((j, o)); RULE_Q.append(f"Is it true that no {nm(x)} is {an(y)}? Answer Yes or No only.")
CHOOSE = []
for x in sorted(LABELS):
    opts = [y for y in sorted(LABELS) if y != x]
    lst = "\\n".join(f"{n + 1}. {nm(y)}" for n, y in enumerate(opts))
    CHOOSE.append((x, opts, f"Here is a list of disease categories:\\n{lst}\\n\\n"
                   f"Which of these categories can never apply to the same disease as \\"{nm(x)}\\"? "
                   f"That is, no disease can be both {an(x)} and that category. Answer with the numbers only, separated by commas."))
print(f"口 {len(MOUTH_Q)}問、ルール {len(RULE_Q)}問、組選び {len(CHOOSE)}問")
print(MOUTH_Q[22]); print(RULE_Q[0]); print(CHOOSE[0][2][:200], "…")
""")
md("## 3. Gemmaに聞く（保存するので2回目からは飛ばせる）")
code("""
cache = f"{OUT}/E2-8_answers_{'MOCK' if USE_MOCK_MOUTH else MODEL_ID.split('/')[-1]}.json"
def parse_pick(text, opts):
    nums = [int(n) for n in re.findall(r"\\d+", text)]
    return sorted({opts[n - 1] for n in nums if 1 <= n <= len(opts)})
if os.path.exists(cache):
    d = json.load(open(cache))
    if d["stmt"] != STMT or d["rule_q"] != RULE_Q: raise SystemExit("保存済みの答えと問題が合わない。ファイルを消してやり直す")
    print("保存済みの答えを読み込んだ:", cache)
else:
    if USE_MOCK_MOUTH:   # 作り物の口：間違い12%、ルールは「両立しない」寄り
        r = np.random.default_rng(123)
        ML = 2.0 * (2 * GOLD - 1) + r.normal(0, 1.5, len(GOLD))
        fl = r.random(len(GOLD)) < 0.12; ML[fl] = -np.sign(ML[fl]) * np.abs(r.normal(2, 0.7, fl.sum()))
        tr = np.array([TRUTH[PAIRS[j]] for j, o in RULE_QS])
        RL = 1.5 * (2 * tr - 1) + 1.0 + r.normal(0, 1.5, len(tr))
        texts = [", ".join(str(n + 1) for n, y in enumerate(o) if (truth_excl(x, y) and r.random() < .6) or r.random() < .1) for x, o, q in CHOOSE]
    else:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from google.colab import userdata
        tok = AutoTokenizer.from_pretrained(MODEL_ID, token=userdata.get("HF_TOKEN"), padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=userdata.get("HF_TOKEN"),
                                                     torch_dtype=torch.float16, device_map="cuda").eval()
        YES = tok.encode("Yes", add_special_tokens=False)[0]; NO = tok.encode("No", add_special_tokens=False)[0]
        pr = lambda s: tok.apply_chat_template([{"role": "user", "content": s}], tokenize=False, add_generation_prompt=True)
        def yesno(Q, B=64):
            L = np.zeros(len(Q)); t = time.time()
            for i in range(0, len(Q), B):
                enc = tok([pr(q) for q in Q[i:i + B]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.no_grad(): last = model(**enc).logits[:, -1, :].float()
                L[i:i + B] = (last[:, YES] - last[:, NO]).cpu().numpy()
                if (i // B) % 20 == 0: print(f"  {i + B}/{len(Q)}  {time.time() - t:.0f}秒", flush=True)
            return L
        print("口の答え"); ML = yesno(MOUTH_Q)
        print("ルール"); RL = yesno(RULE_Q)
        print("組選び"); texts = []
        for x, o, q in CHOOSE:
            enc = tok([pr(q)], return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.no_grad(): g = model.generate(**enc, max_new_tokens=80, do_sample=False)
            texts.append(tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    d = {"model": "MOCK" if USE_MOCK_MOUTH else MODEL_ID, "stmt": STMT, "mouth": ML.tolist(),
         "rule_q": RULE_Q, "rule_qs": RULE_QS, "rule_logit": RL.tolist(),
         "pairs": PAIRS, "truth": [TRUTH[p] for p in PAIRS],
         "choose": [{"label": x, "options": o, "text": t, "picked": parse_pick(t, o)} for (x, o, q), t in zip(CHOOSE, texts)]}
    json.dump(d, open(cache, "w"), ensure_ascii=False); print("保存:", cache)
""")
md("## 4. ざっと確かめる（ここまでで終わり。結果のファイルをこちらで読みます）")
code("""
ML = np.array(d["mouth"]); print(f"口の正解率 {((ML > 0) == GOLD).mean():.3f}（問題 {len(GOLD)}）")
v = np.zeros((len(PAIRS), 2))
for (j, o), l in zip(d["rule_qs"], d["rule_logit"]): v[j, o] = l
g = v.mean(1) > 0; tt = np.array(d["truth"])
print(f"ルール（全部の組）：正しい判断 {(g == tt).mean():.1%}、間違ったルール {(~tt & g).sum()}/{(~tt).sum()}、見逃し {(tt & ~g).sum()}/{tt.sum()}")
ch = {tuple(sorted((c["label"], y))) for c in d["choose"] for y in c["picked"]}
cw = sum(not truth_excl(*p) for p in ch)
print(f"Gemmaが選んだ組 {len(ch)}、そのうち実は両立する組 {cw}")
for c in d["choose"][:3]: print(" ", nm(c["label"]), "→", c["text"][:80].replace(chr(10), " "))
""")
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [{"cell_type": t, "metadata": {}, "source": s, **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells]}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
open(sys.argv[2], "w").write("\n\n".join(s for t, s in cells if t == "code"))
