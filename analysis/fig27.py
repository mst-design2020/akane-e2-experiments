import os, json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont(os.environ.get("JP_FONT", "NotoSansJP.ttf")); plt.rcParams["font.family"] = fm.FontProperties(fname=os.environ.get("JP_FONT", "NotoSansJP.ttf")).get_name()
K = [("①口の自信", "①口の自信"), ("②言い換えのぶれ", "②言い換え\nのぶれ"), ("③近所の一貫性", "③近所の\n一貫性"), ("④組み合わせ", "④組み合わせ"), ("数える", "数える"), ("BP", "BP\n（ルールを疑う網）")]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, (t, title) in zip(axs, [("wn", "ものの分類（2,074問）"), ("geo", "地理（3,586問）")]):
    R = json.load(open(f"../results/E27_{t}.json"))
    v = [np.mean([r["auc"]["全体"][k] for r in R]) for k, _ in K]
    c = ["#9aa5b1"]*4 + ["#6b8fb5", "#c8553d"]
    ax.bar(range(len(K)), v, color=c)
    for i, x in enumerate(v): ax.text(i, x+0.005, f"{x:.3f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(K))); ax.set_xticklabels([l for _, l in K], fontsize=9)
    ax.set_ylim(0.5, 0.9); ax.set_title(title, pad=10); ax.set_ylabel("Gemmaの間違いを拾う力（AUROC、10回平均）")
fig.suptitle("E2-7 世界で使われている間違い探しとの比べ（Gemma-2 2B、ルールもGemmaが作った）")
fig.tight_layout(); fig.savefig("../results/E2-7_図.png", dpi=150)
