# E3 50件の試しの Colab ノートを作る。使い方: python3 build_notebook_E3_pilot.py 出力.ipynb
import json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))
src = lambda f: open(os.path.join(HERE, f)).read()

md("""
# AKANE E3：資料を根拠に「分からない」と言えるか（48件の試し）

本番（4,000件）の前に、48件（またぐ記事2・3・4本 × 支える・反する、各8件）で次を確かめる。
- Gemma-2 2B が主張を部分に分けられるか（2つの分け方：そのまま／「隠れた人や物」を使う）
- 隠れた人や物（例「映画Xの監督」）の候補を、資料から出せるか
- Gemmaの答えと自信、NLI（DeBERTa大）で資料の文ごとの判定

結果はこちらで目で確かめる（分け方・資料の付け方の出来）。計画は plans/E3_evidence_abstain.md（固定済み）。
**時間の目安**（T4 GPU）：10〜20分。
""")
md("""
## 0. 準備
1. 「ランタイム」→「ランタイムのタイプを変更」→ **T4 GPU**。
2. 鍵マークの `HF_TOKEN`（前と同じもの）にアクセスを許可。
3. 「ランタイム」→「すべてのセルを実行」。
""")
code("""
# ===== 設定 =====
USE_MOCK = False        # True: 作り物のAIで流れだけ確かめる（結果は研究に使わない）
SAVE_TO_DRIVE = True
GEMMA_ID = "google/gemma-2-2b-it"
NLI_ID   = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"   # E2-12の「NLI大」と同じ
""")
code("""
import os, json
OUT = "/content/E3_results"
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = "/content/drive/MyDrive/AKANE_アーキテクチャ/E3_results"
os.makedirs(OUT, exist_ok=True)
print("保存先:", OUT)
""")
md("## 1. 道具（データの読み込み・聞き方・本体）")
code(src("e3_pilot.py"))
md("## 2. AIを読み込む（Gemma-2 2B と NLI）")
code("""
if USE_MOCK:
    gen, p_true, nli = mock_models()
else:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
    from google.colab import userdata
    HF = userdata.get("HF_TOKEN")
    gtok = AutoTokenizer.from_pretrained(GEMMA_ID, token=HF, padding_side="left")
    gm = AutoModelForCausalLM.from_pretrained(GEMMA_ID, token=HF, torch_dtype=torch.float16,
                                              device_map="cuda", attn_implementation="eager").eval()
    def chat(p): return gtok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    def gen(prompts, sample=False, bs=8, max_new=160):
        out = []
        for i in range(0, len(prompts), bs):
            enc = gtok([chat(p) for p in prompts[i:i+bs]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                g = gm.generate(**enc, max_new_tokens=max_new, do_sample=sample,
                                temperature=0.7 if sample else None, top_p=0.95 if sample else None)
            out += gtok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return out
    T_ID = gtok.encode("True", add_special_tokens=False)[0]; F_ID = gtok.encode("False", add_special_tokens=False)[0]
    def p_true(prompts, bs=8):
        out = []
        for i in range(0, len(prompts), bs):
            enc = gtok([chat(p) for p in prompts[i:i+bs]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad(): lg = gm(**enc).logits[:, -1, :].float()
            out += torch.softmax(lg[:, [T_ID, F_ID]], -1)[:, 0].cpu().tolist()
        return out
    ntok = AutoTokenizer.from_pretrained(NLI_ID)
    nm = AutoModelForSequenceClassification.from_pretrained(NLI_ID).cuda().eval()
    IDX = []
    for name in NAMES:
        hit = [i for i, l in nm.config.id2label.items() if l.lower().startswith(name[:6])]
        assert len(hit) == 1, nm.config.id2label; IDX.append(int(hit[0]))
    def nli(P, H, bs=64):
        out = []
        for i in range(0, len(P), bs):
            enc = ntok(P[i:i+bs], H[i:i+bs], return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
            with torch.no_grad(): out += nm(**enc).logits.float().softmax(-1)[:, IDX].cpu().tolist()
        return out
print("準備できた")
""")
md("## 3. 48件を選んで実行")
code("""
data = load_hover("/content/hover_dev.json")
items = pick_pilot(data, per_group=8, seed=0)
print(len(items), "件")
items = run_pilot(items, gen, p_true, nli)
res = {"mock": USE_MOCK, "gemma": GEMMA_ID, "nli": NLI_ID, "items": items}
json.dump(res, open(f"{OUT}/E3_pilot.json", "w"), ensure_ascii=False)
print("保存:", f"{OUT}/E3_pilot.json")
for it in items[:3]:
    print("\\n主張:", it["claim"], "| 正解:", it["label"], "| Gemmaの自信(正しい):", round(it["verdict"]["p_true"], 2))
    print(" 分け方A:", it["decomp"]["A"])
    print(" 分け方B:", it["decomp"]["B"]["hidden"], it["decomp"]["B"]["facts"])
    print(" 候補:", it["decomp"]["B"]["assigns"])
""")
md("## 4. ここまで。E3_results の E3_pilot.json をこちらで読みます。")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"colab": {"name": "E3_資料で分からないと言えるか_試し48件.ipynb"}, "accelerator": "GPU",
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(True),
                 **({"execution_count": None, "outputs": []} if t == "code" else {})} for t, s in cells]}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells))
