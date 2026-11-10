"""先頭の図：「〜は〜の一種」の700個で、Gemmaの答えを外から確かめる方法ごとのAURC（E6〜E8、小さいほど良い）。
python3 fig_overview.py → figures/E8/overview_en.png, overview_ja.png（数字は results/E6〜E8 の analysis.json から）"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "..", "results")
e8 = json.load(open(f"{R}/E8/E8_analysis.json"))["methods"]
e7 = json.load(open(f"{R}/E7/E7_analysis.json"))["methods"]
rows = [  # (英, 日, 値, 強調)
    ("Gemma's own confidence", "Gemmaの自信だけ", e8["1_conf"]["mean"]["aurc_claude"], False),
    ("E6: one dictionary definition + NLI", "E6：辞書の説明文1段＋NLI", e8["3_one_step"]["mean"]["aurc_claude"], False),
    ("E7: chain of definitions (all senses)", "E7：説明文を3段つなぐ（意味を選ばない）", e7["6_maxprod"]["mean"]["aurc_claude"], False),
    ("E8: chain, Gemma picks each word's sense", "E8：つなぐ＋Gemmaが語の意味を選ぶ", e8["6_maxprod_D"]["mean"]["aurc_claude"], True),
    ("E8: same + learned combination", "E8：同じ＋手がかりを学習で合わせる", e8["8_learn_D"]["mean"]["aurc_claude"], True),
]
INK, INK2, GRID, BG, ACC, BASE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6", "#b9b7b0"
def draw(lang, path):
    if lang == "ja":
        plt.rcParams["font.family"] = font_manager.FontProperties(fname="/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf").get_name()
        font_manager.fontManager.addfont("/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf")
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=200); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    labels = [r[0] if lang == "en" else r[1] for r in rows][::-1]; vals = [r[2] for r in rows][::-1]; acc = [r[3] for r in rows][::-1]
    ax.barh(range(len(vals)), vals, height=0.56, color=[ACC if a else BASE for a in acc])
    for i, v in enumerate(vals):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlim(0, 0.18); ax.tick_params(axis="x", colors=INK2, labelsize=8); ax.tick_params(axis="y", length=0)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    if lang == "en":
        t = "Checking a small model's \"X is a type of Y\" answers from outside"
        s = "AURC, lower is better: average error rate when the most confident answers are given first. 700 items, common-sense labels."
    else:
        t = "小さいAIの「XはYの一種」という答えを外から確かめる"
        s = "AURC（小さいほど良い）：確かな順に答えた時の誤りの割合の平均。700個、常識の判定。"
    fig.text(0.01, 0.97, t, fontsize=11, color=INK, weight="bold", va="top")
    fig.text(0.01, 0.89, s, fontsize=7.5, color=INK2, va="top")
    fig.subplots_adjust(left=0.42 if lang == "en" else 0.40, right=0.97, top=0.80, bottom=0.10)
    fig.savefig(path, facecolor=BG); plt.close(fig)
os.makedirs(f"{H}/../../figures/E8", exist_ok=True)
draw("en", f"{H}/../../figures/E8/overview_en.png"); draw("ja", f"{H}/../../figures/E8/overview_ja.png")
print([round(r[2], 3) for r in rows])
