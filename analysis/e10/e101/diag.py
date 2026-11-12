import sys, json, numpy as np, collections
sys.path.insert(0,"/mnt/project-files/akane-research/sims/e10/e101")
import e101_analyze as A
items,data,info=A.build("e101_items.json","res")
cs={int(k):v for k,v in json.load(open("cs_labels_claude.json")).items()}
for dom in "PDG":
    D=data[dom]; F=D["F"]; y=D["y"]; lg=D["lg"]
    print("==",dom,"n",len(y),"true",y.sum(),"Gemma yes",(lg>0).mean().round(3),"acc",((lg>0)==y).mean().round(3))
    for k in ("oneN","sixN","oneD","sixD"):
        v=F[k]; print(f" {k}: >0.5 rate {np.mean(v>0.5):.3f}; among true {np.mean(v[y==1]>0.5):.3f}; precision {y[v>0.5].mean() if (v>0.5).any() else 0:.3f}")
    if dom=="P":
        for k in ("fix1","fix2"):
            v=F[k]; print(f" {k}: >0.5 {np.mean(v>0.5):.3f} among true {np.mean(v[y==1]>0.5):.3f} prec {y[v>0.5].mean() if (v>0.5).any() else 0:.3f}")
        dr=F["dir"]; print(" dir mean true",dr[y==1].mean().round(3),"false",dr[y==0].mean().round(3))
    ids=[items[i]["id"] for i in D["idx"]]; c=np.array([cs.get(i,-1) for i in ids]); m=(c>=0)&(c!=2)
    print(" cs n",m.sum(),"cs-true",(c[m]==1).mean().round(3),"agree gold",(c[m]==y[m]).mean().round(3))
D=data["D"]; F=D["F"]; y=D["y"]
m=(F["has"]==0)&(F["fix4_ok"]>0)
print("D fix4 used items",m.sum(),"ents", len({D['its'][i]['ent'] for i in np.where(m)[0]}),"true among",y[m].sum())
v=F["fix4"][m]; yy=y[m]
print(" fix4>0.5 rate",(v>0.5).mean().round(3),"among true",(v[yy==1]>0.5).mean().round(3),"prec",yy[v>0.5].mean().round(3) if (v>0.5).any() else 0)
LU=json.load(open("/dev/null")) if 0 else None
