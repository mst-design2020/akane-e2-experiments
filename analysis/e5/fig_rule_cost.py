# Figure for E5: loss of Gemma-made rules (R2 - R1 AURC), scored by answer table vs common sense.
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf')
plt.rcParams.update({'font.family':'IPAPGothic','font.size':11,'axes.edgecolor':'#b8b6b0','axes.labelcolor':'#52514e',
 'xtick.color':'#52514e','ytick.color':'#52514e','figure.facecolor':'#fcfcfb','axes.facecolor':'#fcfcfb'})
BLUE,ORANGE,TXT='#2a78d6','#eb6834','#0b0b0b'
import os
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../results/E5/')
M=[('2_count','矛盾を数える'),('3_trust_net','ルールを全部信じる網'),('4_doubt_net','ルールも疑う網'),('5_learn','特徴を学習で組み合わせる')]
def loss(f,a,b,key):
    S=json.load(open(D+f))['splits']; return [np.mean([s[a][m][key]-s[b][m][key] for s in S]) for m,_ in M]
panels=[('分類（WordNet）',loss('E5_results_claude_labels.json','R2_gemma','R1_wordnet','aurc'),loss('E5_results_claude_labels.json','R2_gemma','R1_wordnet','aurc_common')),
        ('病気',loss('E5-2_results_claude_labels.json','R2_gemma','R1_doid','aurc'),loss('E5-2_results_claude_labels.json','R2_gemma','R1_doid','aurc_common'))]
fig,axs=plt.subplots(1,2,figsize=(10,4.2),sharey=True)
y=np.arange(len(M))[::-1]
for ax,(t,a,c) in zip(axs,panels):
    ax.barh(y+0.18,a,height=0.34,color=BLUE,label='正解表で採点')
    ax.barh(y-0.18,c,height=0.34,color=ORANGE,label='常識で採点')
    for yy,v in zip(y+0.18,a): ax.text(max(v,0)+0.001,yy,f'{(0.0 if abs(v)<5e-4 else v):.3f}',va='center',fontsize=9,color=TXT)
    for yy,v in zip(y-0.18,c): ax.text(max(v,0)+0.001,yy,f'{(0.0 if abs(v)<5e-4 else v):.3f}',va='center',fontsize=9,color=TXT)
    ax.axvline(0.02,color='#a9a8a2',ls='--',lw=0.8); ax.text(0.0205,y[0]+0.55,'差ありの目安 0.02',fontsize=8,color='#52514e')
    ax.set_title(t,loc='left',fontsize=12,color=TXT); ax.set_xlim(-0.005,0.055)
    for s in ['top','right','left']: ax.spines[s].set_visible(False)
    ax.tick_params(axis='y',length=0); ax.grid(axis='x',color='#e6e4df',lw=0.8); ax.set_axisbelow(True)
axs[0].set_yticks(y); axs[0].set_yticklabels([n for _,n in M])
fig.supxlabel('Gemmaのルールの損（AURCの増え方、大きいほど損）',fontsize=10,color='#52514e')
axs[1].legend(loc='lower right',frameon=False)
fig.suptitle('AIが作ったルールの損：正解表で採点すると見え、常識で採点するとほぼ消える',x=0.01,ha='left',fontsize=12,color=TXT)
fig.tight_layout(); fig.savefig(os.path.join(D,'../../figures/E5/fig4_rule_cost.png'),dpi=200)
