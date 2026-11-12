import json,gzip,numpy as np,sys
S='/tmp/claude-0/-home-user/4394446b-3cce-5fb8-b7fa-8c62205a2ad7/scratchpad'
L=lambda p: json.load(gzip.open(p,'rt'))
it=json.load(open(f'{S}/e101/e101_items.json'))['items']
A=np.array(L(f'{S}/e101/res/E101_gemma.json.gz')['logit'],float)
G={m:np.array(L(f'{S}/e102/res/E102_gemma_N_{m}.json.gz')['logit'],float) for m in ['def','gen','defS','genS']}
F=L(f'{S}/e102/res/E102_facts_N.json.gz'); NL=L(f'{S}/e102/res/E102_nli_N.json.gz')
has=np.array([len(x)>0 for x in F['items']]); ent=np.array([e if e is not None else np.nan for e in NL['ent']],float)
cc={int(k):v for k,v in json.load(open(f'{S}/e101/cs_labels_claude.json')).items()}
cg={int(k):v for k,v in json.load(open(f'{S}/e101/cs_labels_gemini.json')).items()}
P=np.array([x['dom']=='P' for x in it]); gold=np.array([x['gold'] for x in it])
ids=[i for i in cc if P[i] and cc[i]!=2]
def acc(v,idx,lab): 
    idx=[i for i in idx if lab.get(i,2)!=2] if isinstance(lab,dict) else idx
    y=np.array([lab[i] if isinstance(lab,dict) else lab[i] for i in idx]); return round(float(np.mean((v[idx]>0)==y)),3), len(idx)
print("P cs items",len(ids),"has facts",int(has[ids].sum()))
for nm,sub in (("facts",[i for i in ids if has[i]]),("nofacts",[i for i in ids if not has[i]])):
    print(nm,"A claude",acc(A,sub,cc),"gen",acc(G['gen'],sub,cc),"genS",acc(G['genS'],sub,cc),"| gemini A",acc(A,sub,cg),"gen",acc(G['gen'],sub,cg),"| wordnet A",acc(A,sub,gold),"gen",acc(G['gen'],sub,gold))
# label disagreement
d=[i for i in ids if cg.get(i,2)!=2 and cg[i]!=cc[i]]; print("claude!=gemini",len(d),"of",len([i for i in ids if cg.get(i,2)!=2]),"claude yes",sum(cc[i] for i in d))
dw=[i for i in ids if gold[i]!=cc[i]]; print("claude!=wordnet",len(dw),"claude yes among",sum(cc[i] for i in dw))
print(" rates yes: claude",np.mean([cc[i] for i in ids]),"gemini",np.mean([cg[i] for i in ids if cg.get(i,2)!=2]),"wordnet",gold[ids].mean())
# facts directly supporting: NLI ent>0.5 among true(claude) with facts
t=[i for i in ids if has[i] and cc[i]==1]; f=[i for i in ids if has[i] and cc[i]==0]
print("NLI ent>0.5: true-with-facts",np.mean(ent[t]>0.5).round(3),len(t)," false-with-facts",np.mean(ent[f]>0.5).round(3),len(f))
print("gen yes rate true-with-facts",np.mean(G['gen'][t]>0).round(3),"A",np.mean(A[t]>0).round(3)," false-with-facts gen",np.mean(G['gen'][f]>0).round(3),"A",np.mean(A[f]>0).round(3))
# Mean logit shift overall P
Pi=np.where(P)[0]; print("mean logit P: A",A[Pi].mean().round(2),"gen",G['gen'][Pi].mean().round(2),"def",G['def'][Pi].mean().round(2))
print("nofacts P: gen-A mean", (G['gen'][Pi][~has[Pi]]-A[Pi][~has[Pi]]).mean().round(2))
