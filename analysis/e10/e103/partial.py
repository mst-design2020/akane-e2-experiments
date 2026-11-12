import sys,json,numpy as np
sys.path.insert(0,'/mnt/project-files/akane-research/sims/e10/e103')
import e103_analyze as A
E2=A.E2; lgz=A.lgz
it_n,r101,r102,r103,cs_n=sys.argv[1:6]
items=json.load(open(it_n))["items"]
_,d101,_=E2.build101(it_n,r101)
gN=lgz(f"{r101}/E101_gemma.json.gz")["logit"]
D=E2.load_set("N",items,gN,r102,{d:d101[d]["F"]["oneN"] for d in "PDG"})["P"]
selfv=np.array(lgz(f"{r103}/E103_gemma_N_self.json.gz")["logit"],float)
genF=np.array(lgz(f"{r103}/E103_gemma_N_genF.json.gz")["logit"],float)
sense=lgz(f"{r103}/E103_sense_N.json.gz")["logit"]
print(len(selfv),len(D["y"]),len(sense))
kept=np.array([sum(s>0 for s in ss) for ss in sense]); nf=D["nf"]
print("facts",(nf>0).mean(),"kept/facts",kept.sum()/nf.sum(),"kept_any",(kept>0).mean())
cs={int(k):v for k,v in json.load(open(cs_n)).items()}
ycs=np.array([cs.get(items[i]["id"],-1) for i in D["idx"]]); y=D["y"]
fold=A.folds([it["ent"] for it in D["its"]]); runs=[]
for sp in range(5):
    tune=fold==sp; ev=~tune
    P,_,freq=E2.methods(D,tune)
    P={k:P[k] for k in ("A","gen","def","Xpp") if k in P}
    P["self"]=A.cal(selfv,y,tune); P["genF"]=A.cal(genF,y,tune)
    runs.append(A.score(P,y,ev,ycs,freq))
comp={"1 gen v self":("gen","self"),"1 self v A":("self","A"),"1 gen v A":("gen","A"),"3 genF v gen":("genF","gen")}
R=A.summarize(runs,comp)
for key in ("cs","cs_rare","cs_common","table"):
    if "mean_"+key in R:
        print(key,{k:round(v,3) for k,v in R["mean_"+key].items()})
        for c,v in R["main_"+key].items(): print("   ",c,v and (v["判定"],v["回"]))
