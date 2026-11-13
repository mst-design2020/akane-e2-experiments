"""E14-3 の後の見立て（事後、主張には使わない）：N3 でも W1 でも「いいえ」のままの正しい答えを、分類と、たどれた語で並べる。
E8_DIR=../e8 python3 e143_diag.py ../../data/E8n_chains_nli.json ../../data/E14/E14-3_results.json.gz"""
import sys,os,json,numpy as np,collections
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import e142_compare as C
M=C.M; A=M.A
r8,r14=M.jload(sys.argv[1]),M.jload(sys.argv[2])
old,new=A.labels(); allit=old+new; no=len(old)
y=np.array([int(x["mine"]>=1) for x in allit])
r={"L":r14["L"],"GR":r14["GR"],"CH":{(w,k):v for w,k,v in r14["choice"]},"ST":{(w,k):tuple(v) for w,k,*v in r14["steps"]},
   "AR":{(w,k,Cc):tuple(v) for w,k,Cc,*v in r14["arrive"]},"ARG":{(w,i,Cc):tuple(v) for w,i,Cc,*v in r14["arrive_g"]}}
n3=[C.GraphB(it,r,(),3,0).maxprod() for it in allit]; w1=[C.GraphB(it,r,("w",),6,1).maxprod() for it in allit]
cat=collections.Counter(); ends=collections.Counter(); out=[]
for i,it in enumerate(allit):
    if y[i]==1 and n3[i][0]<=0.5 and w1[i][0]<=0.5:
        cat[it["catw"]]+=1
        G=C.GraphB(it,r,("w",),6,1)
        # all reachable nodes: deepest words
        reach=set()
        def walk(n,d):
            if (n,d) in reach or d<0: return
            reach.add((n,d))
            for c in G.ch[n]:
                e,cost=G.edge(n,c); walk(c,d-cost)
        for rt in G.roots: walk(rt,6)
        words=sorted({n[1] for n,d in reach})
        gr=r["GR"].get(it["ent"],[])
        out.append((i<no, it["ent"],it["catw"],round(w1[i][0],3), [g["parent"]+' | '+g.get("sent","")[:90] for g in gr][:1], words[:25]))
print(cat.most_common())
for o in out[:45]: print(o)
