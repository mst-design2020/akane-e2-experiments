import os, json, numpy as np, subprocess, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont(os.environ.get("JP_FONT", "NotoSansJP.ttf"))
JP=[f.name for f in font_manager.fontManager.ttflist if "CJK" in f.name or "Noto Sans JP" == f.name or "Noto Sans JP" in f.name or "IPA" in f.name]
plt.rcParams["font.family"]=JP[0] if JP else "DejaVu Sans"
R={}
for f in ["C_A","C_F0.1","C_F0.2","C_F0.3","gemma_B","E25_P2"]:
    for r in json.load(open(f"../results/{f}.json")): R.setdefault(r["cond"],[]).append(r)
CONDS=[("A","人が書いた\nルール\n（外れ0割）"),("F0.1","わざと\n1割外す"),("F0.2","わざと\n2割外す"),("F0.3","わざと\n3割外す"),("B","Gemmaのルール\n前回の聞き方\n（外れ約7割）"),("P2","Gemmaのルール\n聞き方を変えた\n（外れ約1割強）")]
SIGS=[("口の自信","Gemmaの自信だけ","#b5b3ad"),("数える","矛盾を数えるだけ","#2a78d6"),("BP","つながりで考える（BP）","#eb6834"),("揺れ","つながりで考える（p-bitの揺れ）","#1baf7a")]
INK,MUTED,GRID="#0b0b0b","#52514e","#e4e3df"
fig,ax=plt.subplots(figsize=(15,5.6)); bw=0.19
for gi,(c,lab) in enumerate(CONDS):
    for si,(k,l,col) in enumerate(SIGS):
        v=np.array([r["auc"]["全体"][k] for r in R[c]]); x=gi+(si-1.5)*(bw+0.015)
        ax.bar(x,v.mean()-0.5,bw,bottom=0.5,color=col,edgecolor="white",linewidth=1.5,label=l if gi==0 else None)
        ax.plot([x,x],[v.min(),v.max()],color=INK,lw=1)
        ax.text(x,v.max()+0.005,f"{v.mean():.2f}",ha="center",va="bottom",fontsize=8.5,color=INK)
ax.set_xticks(range(len(CONDS))); ax.set_xticklabels([c[1] for c in CONDS],fontsize=10,color=INK)
ax.set_ylim(0.5,1.0); ax.set_ylabel("Gemmaの間違いを見つける力（AUROC）",color=INK)
ax.set_title("ルールが外れるほど、つながりで考える良さは小さくなる",fontsize=13,loc="left",color=INK,pad=14)
ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.2),ncol=4,frameon=False,fontsize=9.5)
ax.grid(axis="y",color=GRID,lw=0.8); ax.set_axisbelow(True)
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED)
fig.text(0.01,0.005,"ものの分類の○×問題（正解はWordNet）、口：Gemma-2 2B。棒は10回の平均、細い線は最小〜最大。0.5は当てずっぽう、1.0は完全。",fontsize=8.5,color=MUTED)
fig.tight_layout(); fig.savefig("../results/E2-5_図.png",dpi=150,bbox_inches="tight",facecolor="white")
