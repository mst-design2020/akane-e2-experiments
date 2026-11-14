"""E20 figure: gain from adding the TADORI chain to "show the definition" (SO - S),
balanced accuracy, majority vote of three AI judges, 95% intervals (bootstrap by word).
Run from analysis/e20 after judge_combine.py:  python fig_e20.py  -> figures/E20/e20_gain_en.png, e20_gain_ja.png"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..", "results", "E20", "judge", "summary_judges.json")
F = os.path.join(HERE, "..", "..", "figures", "E20")
os.makedirs(F, exist_ok=True)
s = json.load(open(R))["vote"]
MODELS = [("gemma2b", "Gemma-2 2B", "#2a78d6"), ("qwen3b", "Qwen2.5 3B", "#d9822b"), ("llama3b", "Phi-3.5 mini", "#3c9a5b")]
KEYS = ["W|main|common", "W|main|mid", "W|main|rare", "E|main|all"]
T = {"en": dict(x=["WordNet nouns\ncommon", "WordNet nouns\nmid", "WordNet nouns\nrare", "SemEval-2018\nTask 9 (test)"],
                y="Gain in balanced accuracy\n(definition + chain) − (definition only)",
                title="Where the TADORI chain adds to showing the definition (E20)",
                sub="Pairs whose first 3 definitions do not contain the category word. Labels: majority of 3 AI judges. Bars: 95% intervals.",
                bar="decision margin 0.03"),
     "ja": dict(x=["WordNet の名詞\nよく出る", "WordNet の名詞\nふつう", "WordNet の名詞\n珍しい", "SemEval-2018\nTask 9（試験）"],
                y="釣り合った正しさの上がり幅\n（説明文＋鎖）−（説明文だけ）",
                title="TADORI の鎖が「説明文を見せる」に上乗せになる所（E20）",
                sub="説明文の頭3つに分類の語が出ない組。正解は 3つのAIの多数決。縦線は 95% の幅。",
                bar="「効く」の基準 0.03")}
for lang, t in T.items():
    if lang == "ja":
        f = "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf"
        if os.path.exists(f):
            font_manager.fontManager.addfont(f); plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    w = 0.26
    for i, (m, name, col) in enumerate(MODELS):
        for j, k in enumerate(KEYS):
            d = s[k][f"{m}|SO-S"]
            x = j + (i - 1) * w
            ax.bar(x, d["diff"], w * 0.9, color=col, label=name if j == 0 else None)
            ax.plot([x, x], [d["lo"], d["hi"]], color="#0b0b0b", lw=1.2)
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.axhline(0.03, color="#52514e", lw=0.8, ls="--")
    ax.text(len(KEYS) - 0.55, 0.032, t["bar"], fontsize=8, color="#52514e", ha="right", va="bottom")
    ax.set_xticks(range(len(KEYS))); ax.set_xticklabels(t["x"], fontsize=9)
    ax.set_ylabel(t["y"], fontsize=9)
    ax.set_title(t["title"] + "\n", fontsize=11, loc="left")
    ax.text(0, 1.02, t["sub"], transform=ax.transAxes, fontsize=7.5, color="#52514e")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{F}/e20_gain_{lang}.png"); plt.close(fig)
print("ok")
