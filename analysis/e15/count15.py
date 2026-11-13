import sys,json,numpy as np,collections
sys.path.insert(0,'/mnt/project-files/akane-research/analysis/e14')
import e142_compare as C
from nltk.corpus import wordnet as wn
M=C.M; A=M.A
r14=M.jload(sys.argv[1])
old,new=A.labels(); allit=old+new; no=len(old)
y=np.array([int(x["mine"]>=1) for x in allit])
r={"L":r14["L"],"GR":r14["GR"],"CH":{(w,k):v for w,k,v in r14["choice"]},"ST":{(w,k):tuple(v) for w,k,*v in r14["steps"]},
   "AR":{(w,k,Cc):tuple(v) for w,k,Cc,*v in r14["arrive"]},"ARG":{(w,i,Cc):tuple(v) for w,i,Cc,*v in r14["arrive_g"]}}
def under(word,cat):
    cs=wn.synsets(cat.replace(' ','_'),'n'); ws=wn.synsets(word.replace(' ','_'),'n')
    for s in ws:
        H=set(s.closure(lambda x:x.hypernyms()+x.instance_hypernyms()))|{s}
        if any(c in H for c in cs): return True
    return False
for name,srcs,d,g in (("N3",(),3,0),("W1",("w",),6,1)):
    cnt=collections.Counter(); ex=collections.defaultdict(list)
    for i,it in enumerate(allit):
        G=C.GraphB(it,r,srcs,d,g); s,_=G.maxprod()
        if not(y[i]==1 and s<=0.5): continue
        best={}
        def walk(n,dd,p,gl):
            if p<=best.get(n,(-1,))[0] and n in best: return
            best[n]=(p,)
            if dd<=0: return
            for c in G.ch[n]:
                e,cost=G.edge(n,c); g2=gl-(c[0]=="g")
                if cost>dd or g2<0: continue
                walk(c,dd-cost,p*e,g2)
        for rt in G.roots:
            gl=g-(rt[0]=="g")
            if gl>=0: walk(rt,d,1.0,gl)
        good=[(n,p[0]) for n,p in best.items() if n[1]!=it["ent"] and under(n[1],it["catw"])]
        goodhi=[(n,p) for n,p in good if p>=0.5]
        if goodhi: k="A 正しい途中の語まで着いた→最後の段で落ちた"
        elif good: k="B 正しい途中の語はあるが途中の段が弱い"
        elif under(it["ent"],it["catw"]): k="C 正しい途中の語に着かない（親の選び違い・行き止まり）"
        else: k="D WordNetでも一種でない（正解が微妙／WordNetの抜け）"
        cnt[k]+=1; ex[k].append(f'{it["ent"]}->{it["catw"]}'+(f' via {goodhi[0][0][1]}({goodhi[0][1]:.2f}) arr {G.arr(goodhi[0][0]):.3f}' if goodhi else ''))
    print(name, sum(cnt.values()), dict(sorted(cnt.items())))
    for k in sorted(ex): print(' ',k[:1], ex[k][:14])
