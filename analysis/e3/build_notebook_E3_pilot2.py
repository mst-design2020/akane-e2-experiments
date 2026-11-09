# E3 試し2回目の Colab ノートを作る。使い方: python3 build_notebook_E3_pilot2.py 出力.ipynb
import json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))
src = lambda f: open(os.path.join(HERE, f)).read()

md("""
# AKANE E3：資料を根拠に「分からない」と言えるか（試し2回目、同じ48件）

1回目は Gemma-2 2B の分け方が下手だった（取りこぼし・付け足しが多い）。今回は同じ48件で：
- 分ける・隠れた人や物を埋める役を、**2B と 9B（4ビットに縮めたもの）で並べる**
- 聞き方の細かい直し（見本の文が混ざらない、名前で答える、分からなければ unknown）
- 資料の判定（NLI）を、**文1つ** と **記事の段落1つ** の両方で取る
- 答えと自信は1回目と同じ 2B

計画は plans/E3_evidence_abstain.md（固定済み、末尾に変更を記録）。
**時間の目安**（T4 GPU）：30〜50分（9Bの読み込みに5〜10分）。
""")
md("""
## 0. 準備
1. 「ランタイム」→「ランタイムのタイプを変更」→ **T4 GPU**。
2. 鍵マークの `HF_TOKEN`（前と同じもの）にアクセスを許可。
3. **9Bを初めて使う時だけ**：https://huggingface.co/google/gemma-2-9b-it を開いて、利用条件に同意（2Bと同じ画面）。
4. 「ランタイム」→「すべてのセルを実行」。
""")
code("""
# ===== 設定 =====
USE_MOCK = False        # True: 作り物のAIで流れだけ確かめる（結果は研究に使わない）
SAVE_TO_DRIVE = True
GEMMA_ID = "google/gemma-2-2b-it"
GEMMA9_ID = "google/gemma-2-9b-it"
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
if not USE_MOCK:
    import importlib.util, subprocess
    if importlib.util.find_spec("bitsandbytes") is None:   # 入っていない時だけ入れる（他の部品は上げない）
        subprocess.run('pip -q install bitsandbytes --no-deps', shell=True, check=True)
print("保存先:", OUT)
""")
md("## 1. 道具（データの読み込み・聞き方・本体）")
code(src("e3_pilot2.py"))
md("## 2. AIの道具（読み込みと聞き方）")
code("""
import gc
if USE_MOCK:
    gen, p_true, nli = mock_models()
    def load_gemma(mid, four_bit=False): return gen, p_true
    def free(): pass
else:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
    from google.colab import userdata
    HF = userdata.get("HF_TOKEN")
    G = {}
    def free():
        G.clear(); gc.collect(); torch.cuda.empty_cache()
    def load_gemma(mid, four_bit=False):
        free()
        tok = AutoTokenizer.from_pretrained(mid, token=HF, padding_side="left")
        kw = dict(token=HF, device_map="cuda", attn_implementation="eager")
        if four_bit:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        else:
            kw["torch_dtype"] = torch.float16
        m = AutoModelForCausalLM.from_pretrained(mid, **kw).eval()
        G["tok"], G["m"] = tok, m
        chat = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        def gen(prompts, sample=False, bs=8, max_new=160):
            out = []
            for i in range(0, len(prompts), bs):
                enc = tok([chat(p) for p in prompts[i:i+bs]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    g = m.generate(**enc, max_new_tokens=max_new, do_sample=sample,
                                   temperature=0.7 if sample else None, top_p=0.95 if sample else None)
                out += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            return out
        T_ID = tok.encode("True", add_special_tokens=False)[0]; F_ID = tok.encode("False", add_special_tokens=False)[0]
        def p_true(prompts, bs=8):
            out = []
            for i in range(0, len(prompts), bs):
                enc = tok([chat(p) for p in prompts[i:i+bs]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.no_grad(): lg = m(**enc).logits[:, -1, :].float()
                out += torch.softmax(lg[:, [T_ID, F_ID]], -1)[:, 0].cpu().tolist()
            return out
        return gen, p_true
    def load_nli():
        free()
        ntok = AutoTokenizer.from_pretrained(NLI_ID)
        nm = AutoModelForSequenceClassification.from_pretrained(NLI_ID).cuda().eval()
        G["m"] = nm
        IDX = []
        for name in NAMES:
            hit = [i for i, l in nm.config.id2label.items() if l.lower().startswith(name[:6])]
            assert len(hit) == 1, nm.config.id2label; IDX.append(int(hit[0]))
        def nli(P, H, bs=32):
            out = []
            for i in range(0, len(P), bs):
                enc = ntok(P[i:i+bs], H[i:i+bs], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
                with torch.no_grad(): out += nm(**enc).logits.float().softmax(-1)[:, IDX].cpu().tolist()
            return out
        return nli
print("道具の準備できた")
""")
md("## 3. 48件を選んで実行")
code("""
data = load_hover("/content/hover_dev.json")
items = pick_pilot(data, per_group=8, seed=0)
print(len(items), "件")
prep(items)
OUTF = f"{OUT}/E3_pilot2.json"
def save(): json.dump({"mock": USE_MOCK, "gemma": GEMMA_ID, "gemma9": GEMMA9_ID, "nli": NLI_ID, "items": items}, open(OUTF, "w"), ensure_ascii=False); print("保存:", OUTF)
if os.path.exists(OUTF):   # 途中で止まった時は続きから
    old = json.load(open(OUTF))["items"]
    if [o["id"] for o in old] == [it["id"] for it in items]: items = old; print("続きから")
if "decomp" not in items[0] or "2b" not in items[0]["decomp"]:
    gen, p_true = load_gemma(GEMMA_ID)                 # 2B：答えと自信・分ける
    do_verdict(items, p_true); save()
    do_decomp(items, gen, "2b"); save()
if "9b" not in items[0]["decomp"]:
    gen, p_true = load_gemma(GEMMA9_ID, four_bit=True)  # 9B：分ける
    do_decomp(items, gen, "9b"); save()
if USE_MOCK: nli_f = nli
else: nli_f = load_nli()
do_nli(items, nli_f); save()
for it in items[:3]:
    print("\\n主張:", it["claim"], "| 正解:", it["label"], "| Gemmaの自信(正しい):", round(it["verdict"]["p_true"], 2))
    for tag in ("2b", "9b"):
        B = it["decomp"][tag]["B"]
        print(f" [{tag}] 分け方B:", B["hidden"], B["facts"], "| 候補:", B["assigns"])
""")
md("## 4. ここまで。E3_results の E3_pilot2.json をこちらで読みます。")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"colab": {"name": "E3_資料で分からないと言えるか_試し2回目.ipynb"}, "accelerator": "GPU",
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(True),
                 **({"execution_count": None, "outputs": []} if t == "code" else {})} for t, s in cells]}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells))
