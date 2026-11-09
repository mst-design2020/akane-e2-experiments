# E2-12 の Colab ノートを作る。使い方: python3 build_notebook_E2-12.py 出力.ipynb（リポジトリ版。プロジェクトで実行したノートは colab/E2-12_links_gemma.ipynb）
import json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))
src = lambda f: open(os.path.join(HERE, f)).read()

md("""
# AKANE E2-12：つながりを、いろいろなAIに付けさせて比べる

E2-11では、主張どうしのつながりを**くじでわざと**反対にした。今回は、つながりを**実際のAI 4つ**に付けさせる。
外れは作り物ではなく、そのAIが本当にやる外れになる。

- つながりを付けるAI：NLI専用の小・中・大（DeBERTa系）と、Gemma-2 2Bに聞く形
- 主張・木・自信・正解はE2-10と同じ（BTPropの公開データ、FactCheckGPT と WikiBio）
- 比べる：AIの自信／BTProp（つながりを全部信じる）／ルールを全部信じるBP／ルールも疑うBP

計画は sims/e2_btprop/PLAN_E2-12.md（結果を見る前に固定）。
**時間の目安**（T4 GPU）：つながりの判定 15〜25分、計算 5〜10分。
""")
md("""
## 0. 準備
1. 「ランタイム」→「ランタイムのタイプを変更」→ **T4 GPU**。
2. 鍵マークの `HF_TOKEN`（前と同じもの）にアクセスを許可。
3. 「ランタイム」→「すべてのセルを実行」。途中で止まったら、もう一度「すべてのセルを実行」で続きから動く（判定は保存される）。
""")
code("""
# ===== 設定 =====
USE_MOCK_NLI  = False   # True: 作り物の判定（動作確認用。結果は研究に使わない）
SAVE_TO_DRIVE = True
NLI_MODELS = {   # 名前: Hugging Faceの名前
    "NLI小(DeBERTa-xsmall)": "cross-encoder/nli-deberta-v3-xsmall",
    "NLI中(DeBERTa-base)":   "cross-encoder/nli-deberta-v3-base",
    "NLI大(DeBERTa-large)":  "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
}
GEMMA_ID = "google/gemma-2-2b-it"
""")
code("""
import os, json, time, subprocess, collections
import numpy as np
OUT = "/content/E2_results"
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = "/content/drive/MyDrive/AKANE_アーキテクチャ/E2_results"
os.makedirs(OUT, exist_ok=True)
LABELS_JSON = f"{OUT}/E2-10_wikibio_labels.json"     # E2-10で保存したWikiBioの正解
assert os.path.exists(LABELS_JSON), "E2_results に E2-10_wikibio_labels.json が見つからない"
# BTProp（公開の木・自信・コード）と FactCheckGPT の正解を取ってくる
if not os.path.exists("/content/BTProp"):
    subprocess.run("git clone -q -b master https://github.com/UCSB-NLP-Chang/BTProp.git /content/BTProp", shell=True, check=True)
os.chdir("/content/BTProp"); os.makedirs("eval_dataset", exist_ok=True)
if not os.path.exists("eval_dataset/Factbench.jsonl"):
    subprocess.run("wget -q -O eval_dataset/Factbench.jsonl https://raw.githubusercontent.com/yuxiaw/Factcheck-GPT/main/Factbench.jsonl", shell=True, check=True)
!pip -q install pyyaml scikit-learn
print("保存先:", OUT)
""")
md("## 1. WikiBioの正解を、BTPropの文の順に合わせる（E2-10と同じ手順、1分弱）")
code(src("align_wikibio.py").replace("sys.argv[1]", "LABELS_JSON"))
md("## 2. つながりを判定する組を集める（評価に届く組＋聞き方選び用の200組）")
code(src("e212_pairs.py").split("if __name__")[0] + """
print('実際に聞く組', len(RUN), '／聞き方選び', len(HELD))
""")
md("""
## 3. つながりを判定する
各組を両方向（親→子、子→親）で「含まれる／どちらとも言えない／矛盾する」の3択に。
Gemmaは聞き方を3つ用意し、**評価に使わない200組**で元の判定と一番合うものを1回だけ選んでから、全部に使う。
""")
code(src("e212_nli.py"))
md("## 4. 計算（BTProp・BPを、つながりを替えて比べる）")
code("""
nli_json = f"{OUT}/E2-12_nli.json"
for ds in ["factcheckgpt", "wikibio"]:
    subprocess.run(f"python /content/e212.py {ds} '{nli_json}' '{OUT}/E212_{ds}.json' | tail -8", shell=True, check=True)
""")
md("## 5. ここまでで終わり。E2_results の E2-12_nli.json・E212_factcheckgpt.json・E212_wikibio.json をこちらで読みます。")

# e212.py はファイルとして置く（セル4で使う）
cells.insert(5, ("code", "%%writefile /content/e212.py\n" + src("e212.py")))

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"colab": {"name": "E2-12_つながりをAIに付けさせる.ipynb"}, "accelerator": "GPU",
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(True),
                 **({"execution_count": None, "outputs": []} if t == "code" else {})} for t, s in cells]}
json.dump(nb, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("cells", len(cells))
