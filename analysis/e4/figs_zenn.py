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
panels=[('E4-1 本物の網（正解を2割使う）',[('Gemmaの自信',.142,0),('矛盾を数える',.053,0),('特徴＋学習',.009,0),('小さな網',.059,1),('大きな網 BP',.062,1),('大きな網 p-bit方式',.080,1)]),
 ('E4-2 重みを学習した網（正解を2割使う）',[('Gemmaの自信',.248,0),('特徴＋学習',.086,0),('学習した小さな網',.107,1),('学習した大きな網 BP',.382,1),('大きな網 p-bit方式',.382,1),('慎重な網',.079,1)]),
 ('E4-3 正解を使わない',[('Gemmaの自信',.248,0),('癖消し（Batch Cal.）',.243,0),('矛盾を数える',.151,0),('大きな網',.157,1),('慎重な網',.168,1),('小さな網',.280,1)])]
fig,axs=plt.subplots(3,1,figsize=(8,9.2),sharex=True)
for ax,(t,rows) in zip(axs,panels):
    rows=rows[::-1]
    y=range(len(rows))
    ax.barh(y,[r[1] for r in rows],color=[BLUE if r[2] else GRAY for r in rows],height=0.6)
    ax.set_yticks(list(y)); ax.set_yticklabels([r[0] for r in rows])
    for i,r in enumerate(rows): ax.text(r[1]+0.005,i,f'{r[1]:.3f}',va='center',fontsize=10,color=TXT)
    ax.set_title(t,loc='left',fontsize=12,color=TXT); clean(ax)
axs[-1].set_xlabel('AURC（「分からない」と控えたときの誤りの面積、小さいほど良い）')
axs[-1].set_xlim(0,0.43)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=BLUE,label='網を使う方法'),Patch(color=GRAY,label='網を使わない方法')],loc='upper right',frameon=False,ncol=2)
fig.tight_layout(rect=(0,0,1,0.97)); fig.savefig('fig1_aurc.png',dpi=200)

# 図2：正解表と常識の食い違い（種類ごと）
types=[('正しい（T）',178,1,21),('Gemmaが一番信じたうそ（G）',132,24,44),('兄弟・おじのうそ（H）',195,5,0),('易しいうそ（E）',100,0,0)]
fig,ax=plt.subplots(figsize=(8,3.4))
types=types[::-1]
for i,(n,a,m,d) in enumerate(types):
    tot=a+m+d; l=0
    for v,c in [(a,BLUE),(m,GRAY),(d,ORANGE)]:
        if v: ax.barh(i,v/tot*100,left=l,color=c,height=0.6,edgecolor='#fcfcfb',linewidth=2)
        l+=v/tot*100
    if d: ax.text(100.5,i,f'食い違い {d/tot*100:.0f}%',va='center',fontsize=10,color=TXT)
ax.set_yticks(range(len(types))); ax.set_yticklabels([t[0] for t in types]); ax.set_xlim(0,122); ax.set_xticks([0,25,50,75,100])
ax.set_xlabel('割合（%、各200個・Eは100個）'); clean(ax)
ax.legend(handles=[Patch(color=BLUE,label='正解表（WordNet）と常識が一致'),Patch(color=GRAY,label='どちらとも言える'),Patch(color=ORANGE,label='常識では逆')],loc='lower center',bbox_to_anchor=(0.45,1.0),ncol=3,frameon=False,fontsize=10)
fig.tight_layout(); fig.savefig('fig2_labels.png',dpi=200)

# 図3：正解表を替えると順位が変わる
rows=[('Gemmaの自信',.263,.151),('特徴＋学習',.087,.056),('学習した小さな網',.165,.054),('慎重な網',.099,.039)][::-1]
fig,ax=plt.subplots(figsize=(8,3.6)); h=0.36
for i,(n,w,c) in enumerate(rows):
    ax.barh(i+h/2,w,height=h,color=GRAY); ax.barh(i-h/2,c,height=h,color=BLUE)
    ax.text(w+0.004,i+h/2,f'{w:.3f}',va='center',fontsize=10,color=TXT); ax.text(c+0.004,i-h/2,f'{c:.3f}',va='center',fontsize=10,color=TXT)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(0,0.3)
ax.set_xlabel('AURC（抜き出した700個、5回の平均、小さいほど良い）'); clean(ax)
ax.legend(handles=[Patch(color=GRAY,label='正解表（WordNet）で採点'),Patch(color=BLUE,label='常識で採点')],loc='lower right',frameon=False)
fig.tight_layout(); fig.savefig('fig3_rescore.png',dpi=200)
