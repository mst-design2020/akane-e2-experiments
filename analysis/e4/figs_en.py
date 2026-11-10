import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf')
plt.rcParams.update({'font.family':'IPAPGothic','font.size':11,'axes.edgecolor':'#b8b6b0','axes.labelcolor':'#52514e',
 'xtick.color':'#52514e','ytick.color':'#52514e','figure.facecolor':'#fcfcfb','axes.facecolor':'#fcfcfb'})
BLUE,ORANGE,GRAY,TXT='#2a78d6','#eb6834','#a9a8a2','#0b0b0b'
def clean(ax):
    for s in ['top','right','left']: ax.spines[s].set_visible(False)
    ax.tick_params(axis='y',length=0); ax.grid(axis='x',color='#e6e4df',lw=0.8); ax.set_axisbelow(True)

# 図1：3つの実験のAURC（網＝青、網なし＝灰）
panels=[('E4-1 Real network (20% labels used)',[('Gemma confidence',.142,0),('Count violated rules',.053,0),('Features + LR',.009,0),('Small networks',.059,1),('Large network, BP',.062,1),('Large network, p-bit',.080,1)]),
 ('E4-2 Learned rule weights (20% labels used)',[('Gemma confidence',.248,0),('Features + LR',.086,0),('Learned small networks',.107,1),('Learned large network, BP',.382,1),('Large network, p-bit',.382,1),('Cautious network',.079,1)]),
 ('E4-3 No labels at all',[('Gemma confidence',.248,0),('Batch Calibration',.243,0),('Count violated rules',.151,0),('Large network',.157,1),('Cautious network',.168,1),('Small networks',.280,1)])]
fig,axs=plt.subplots(3,1,figsize=(8,9.2),sharex=True)
for ax,(t,rows) in zip(axs,panels):
    rows=rows[::-1]
    y=range(len(rows))
    ax.barh(y,[r[1] for r in rows],color=[BLUE if r[2] else GRAY for r in rows],height=0.6)
    ax.set_yticks(list(y)); ax.set_yticklabels([r[0] for r in rows])
    for i,r in enumerate(rows): ax.text(r[1]+0.005,i,f'{r[1]:.3f}',va='center',fontsize=10,color=TXT)
    ax.set_title(t,loc='left',fontsize=12,color=TXT); clean(ax)
axs[-1].set_xlabel('AURC (risk-coverage area, lower is better)')
axs[-1].set_xlim(0,0.43)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=BLUE,label='Network methods'),Patch(color=GRAY,label='Non-network methods')],loc='upper right',frameon=False,ncol=2)
fig.tight_layout(rect=(0,0,1,0.97)); fig.savefig('fig1_aurc_en.png',dpi=200)

# 図2：正解表と常識の食い違い（種類ごと）
types=[('True (T)',178,1,21),('Gemma\'s most-believed false (G)',132,24,44),('Sibling/uncle false (H)',195,5,0),('Easy false (E)',100,0,0)]
fig,ax=plt.subplots(figsize=(8,3.4))
types=types[::-1]
for i,(n,a,m,d) in enumerate(types):
    tot=a+m+d; l=0
    for v,c in [(a,BLUE),(m,GRAY),(d,ORANGE)]:
        if v: ax.barh(i,v/tot*100,left=l,color=c,height=0.6,edgecolor='#fcfcfb',linewidth=2)
        l+=v/tot*100
    if d: ax.text(100.5,i,f'disagree {d/tot*100:.0f}%',va='center',fontsize=10,color=TXT)
ax.set_yticks(range(len(types))); ax.set_yticklabels([t[0] for t in types]); ax.set_xlim(0,122); ax.set_xticks([0,25,50,75,100])
ax.set_xlabel('Share (%; 200 items each, E 100)'); clean(ax)
ax.legend(handles=[Patch(color=BLUE,label='WordNet agrees'),Patch(color=GRAY,label='Debatable'),Patch(color=ORANGE,label='Common sense disagrees')],loc='lower center',bbox_to_anchor=(0.45,1.0),ncol=3,frameon=False,fontsize=10)
fig.tight_layout(); fig.savefig('fig2_labels_en.png',dpi=200)

# 図3：正解表を替えると順位が変わる
rows=[('Gemma confidence',.263,.151),('Features + LR',.087,.056),('Learned small networks',.165,.054),('Cautious network',.099,.039)][::-1]
fig,ax=plt.subplots(figsize=(8,3.6)); h=0.36
for i,(n,w,c) in enumerate(rows):
    ax.barh(i+h/2,w,height=h,color=GRAY); ax.barh(i-h/2,c,height=h,color=BLUE)
    ax.text(w+0.004,i+h/2,f'{w:.3f}',va='center',fontsize=10,color=TXT); ax.text(c+0.004,i-h/2,f'{c:.3f}',va='center',fontsize=10,color=TXT)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(0,0.3)
ax.set_xlabel('AURC (700 sampled items, mean of 5 splits, lower is better)'); clean(ax)
ax.legend(handles=[Patch(color=GRAY,label='Scored with WordNet labels'),Patch(color=BLUE,label='Scored with common-sense labels')],loc='lower right',frameon=False)
fig.tight_layout(); fig.savefig('fig3_rescore_en.png',dpi=200)
