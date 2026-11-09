"""E4-1 Colab ノート（Gemmaに はい/いいえ を答えさせるだけ）を作る。python3 build_notebook_E4-1.py out.ipynb"""
import json, sys
D = json.load(open("e41_data.json"))
STMT = [c["stmt"] for c in D["claims"]]
cells = []
md = lambda s: cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))
code = lambda s: cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n")))
md("""
# AKANE E4-1：Gemmaに「〜は〜の一種か」を約1万問答えさせる

このノートは **Gemma-2 2B の答え（はい/いいえ の確からしさ）を取るだけ** です。網の計算と比べは、結果のファイルを読んで別に行います。

- 問い：10,864問（物の主張 7,906・分類どうしの一種 2,008・両立しない 950）。正解はWordNetから作ってあり、このノートには入っていません。
- 計画：sims/e4/PLAN_E4-1.md（固定済み）

## 手順
1. 「ランタイム」→「ランタイムのタイプを変更」→ **L4 GPU**（なければ T4）。
2. 鍵マークの `HF_TOKEN` にアクセスを許可。
3. 「ランタイム」→「すべてのセルを実行」。ドライブの許可を聞かれたら許可。
4. 終わると、ドライブの `AKANE_アーキテクチャ/E4_results/E41_gemma-2-2b-it.json` ができます（約1MB）。L4で5〜10分の見込み。
""")
code("""
import os, json, time
import numpy as np
from google.colab import drive, userdata
drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/AKANE_アーキテクチャ/E4_results"
os.makedirs(OUT, exist_ok=True)
MODEL_ID = "google/gemma-2-2b-it"
""")
code("STMT = json.loads(r'''" + json.dumps(STMT, ensure_ascii=False) + "''')\nprint(len(STMT), '問')")
code("""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL_ID, token=userdata.get("HF_TOKEN"), padding_side="left")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=userdata.get("HF_TOKEN"),
                                             torch_dtype=torch.float16, device_map="cuda").eval()
YES = tok.encode("Yes", add_special_tokens=False)[0]; NO = tok.encode("No", add_special_tokens=False)[0]
def prompt(s):
    msg = [{"role": "user", "content": f"Is the following statement true? Answer Yes or No only.\\nStatement: {s}"}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
LOGIT = np.zeros(len(STMT)); B = 64; t = time.time()
for i in range(0, len(STMT), B):
    enc = tok([prompt(s) for s in STMT[i:i + B]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad():
        last = model(**enc).logits[:, -1, :].float()
    LOGIT[i:i + B] = (last[:, YES] - last[:, NO]).cpu().numpy()
    if (i // B) % 20 == 0: print(f"{min(i + B, len(STMT))}/{len(STMT)}  {time.time() - t:.0f}秒")
path = f"{OUT}/E41_{MODEL_ID.split('/')[-1]}.json"
json.dump({"model": MODEL_ID, "stmt": STMT, "logit": LOGIT.tolist()}, open(path, "w"))
print("保存:", path, " はいと答えた割合:", round(float((LOGIT > 0).mean()), 3))
""")
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
