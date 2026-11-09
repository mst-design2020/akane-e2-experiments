# E3 本番の Colab ノートを作る。使い方: python3 build_notebook_E3_main.py 出力.ipynb
import json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))
src = lambda f: open(os.path.join(HERE, f)).read()

md("""
# AKANE E3 本番：資料を根拠に「分からない」と言えるか

HoVer の主張（試しの48件を除く）について、次を集める。網の計算と比べはこちら（Claude）でやる。
1. Gemma-2 2B：資料を見て答え（正しい／間違い）と自信
2. Gemma-2 9B（4ビット）：主張を「隠れた人や物」つきの部分に分ける → 隠れた人や物の候補を資料から出す
3. NLI（DeBERTa大）：段落ごと・文ごとに、元の主張と部分の主張を判定

**途中で切れても大丈夫**：もう一度「すべてのセルを実行」で続きから動く。
計画は plans/E3_evidence_abstain.md（固定、変更は末尾に記録）。
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
PER_GROUP = None        # None: 全部（3,952件）。数字にすると「記事数×正解」ごとにその件数だけ選ぶ
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
code(src("e3_main.py"))
md("## 2. AIの道具（読み込みと聞き方）")
code("""
import gc
if USE_MOCK:
    gen, p_true, nli = mock_models()
    def load_gemma(mid, four_bit=False): return gen, p_true
    def free(): pass
else:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
    def batched(fn, xs, bs):
        # bs件ずつ fn に渡す。メモリが足りなければ、その回だけ半分ずつにして続ける
        out, i = [], 0
        while i < len(xs):
            n = bs
            while True:
                try:
                    out += fn(xs[i:i+n]); break
                except torch.cuda.OutOfMemoryError:
                    gc.collect(); torch.cuda.empty_cache()
                    if n == 1: raise
                    n = max(1, n // 2); print("メモリ不足→まとめて聞く数を", n, "に")
            i += n
        return out
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
            def one(ps):
                enc = tok([chat(p) for p in ps], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    g = m.generate(**enc, max_new_tokens=max_new, do_sample=sample,
                                   temperature=0.7 if sample else None, top_p=0.95 if sample else None)
                return tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            return batched(one, prompts, bs)
        T_ID = tok.encode("True", add_special_tokens=False)[0]; F_ID = tok.encode("False", add_special_tokens=False)[0]
        def p_true(prompts, bs=8):
            def one(ps):
                enc = tok([chat(p) for p in ps], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                # 最後の1語の分だけ計算する（全部の語の分を作るとメモリが足りなくなる。答えは同じ）
                with torch.no_grad(): lg = m(**enc, logits_to_keep=1).logits[:, -1, :].float()
                return torch.softmax(lg[:, [T_ID, F_ID]], -1)[:, 0].cpu().tolist()
            return batched(one, prompts, bs)
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
            def one(ph):
                enc = ntok([a for a, _ in ph], [b for _, b in ph], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
                with torch.no_grad(): return nm(**enc).logits.float().softmax(-1)[:, IDX].cpu().tolist()
            return batched(one, list(zip(P, H)), bs)
        return nli
print("道具の準備できた")
""")
md("## 3. 本番を実行（段階ごとに保存。切れたら、もう一度すべてのセルを実行）")
code("""
data = load_hover("/content/hover_dev.json")
pilot_ids = {e["id"] for e in pick_pilot(data, per_group=8, seed=0)}
OUTF = f"{OUT}/E3_main.json"
items = pick_main(data, pilot_ids, per_group=PER_GROUP)
if os.path.exists(OUTF):
    old = json.load(open(OUTF))["items"]
    if [o["id"] for o in old] == [it["id"] for it in items]: items = old; print("続きから")
prep(items)
print(len(items), "件")
def save():
    tmp = OUTF + ".tmp"
    json.dump({"meta": {"mock": USE_MOCK, "gemma": GEMMA_ID, "gemma9": GEMMA9_ID, "nli": NLI_ID, "per_group": PER_GROUP},
               "items": items}, open(tmp, "w"), ensure_ascii=False)
    os.replace(tmp, OUTF)
big = (not USE_MOCK) and torch.cuda.get_device_properties(0).total_memory > 20e9
BS = 16 if big else 8
print("GPUが大きい:", big, "まとめて聞く数:", BS)
if any("verdict" not in it for it in items):
    gen, p_true = load_gemma(GEMMA_ID)
    stage_verdict(items, lambda ps: p_true(ps, bs=BS), save)
if any(it["label"] == "supports" and len(it["paras"]) >= 2 and "verdict_drop" not in it for it in items):
    gen, p_true = load_gemma(GEMMA_ID)
    stage_verdict_drop(items, lambda ps: p_true(ps, bs=BS), save)
if any("assigns" not in it.get("decomp9b", {}) for it in items):
    gen, p_true = load_gemma(GEMMA9_ID, four_bit=True)
    g = lambda ps, **kw: gen(ps, bs=BS, **kw)
    stage_decomp(items, g, save)
    stage_fill(items, g, save)
if any("nli" not in it for it in items):
    nli_f = nli if USE_MOCK else load_nli()
    stage_nli(items, lambda P, H: nli_f(P, H, bs=64 if big else 32), save)
print("全部済み:", OUTF)
it = items[0]; print(it["claim"]); print(it["decomp9b"]["hidden"], it["decomp9b"]["facts"], it["decomp9b"]["assigns"])
""")
md("## 4. ここまで。E3_results の E3_main.json をこちらで読みます。")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"colab": {"name": "E3_資料で分からないと言えるか_本番_直し.ipynb"}, "accelerator": "GPU",
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(True),
                 **({"execution_count": None, "outputs": []} if t == "code" else {})} for t, s in cells]}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells))
