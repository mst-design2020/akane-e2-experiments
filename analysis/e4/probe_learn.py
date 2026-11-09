import json, numpy as np
from sklearn.linear_model import LogisticRegression
import e41
D=json.load(open('e41_data.json')); A=json.load(open('E41_gemma-2-2b-it.json'))
L=np.array(A['logit']); y=np.array([c['gold'] for c in D['claims']]); n=len(L)
big=e41.Net(n,D['factors']); a=(L>0).astype(np.int8); v1,v0=big.counts(a)
F=np.zeros((n,6)); F[:,0]=L; F[:,1]=v1; F[:,2]=v0
for w in D['ents']:
    ids=np.array(D['ent_claims'][w])
    for i in ids: o=L[ids[ids!=i]]; F[i,3]=o.max(); F[i,4]=o.mean()
cs=np.zeros(n); cc=np.zeros(n)
for t,v in D['factors']:
    if t=='trans': cs[v[0]]+=L[v[1]]; cc[v[0]]+=1
F[:,5]=cs/np.maximum(cc,1)
ents=D['ents']; r=np.random.default_rng(1000); perm=r.permutation(len(ents)); te=set(ents[i] for i in perm[:len(ents)//5])
xt=np.array([i for w in ents if w in te for i in D['ent_claims'][w]]); xe=np.array([i for w in ents if w not in te for i in D['ent_claims'][w]])
names=['logit','v1','v0','max_other','mean_other','mean_c']
def run(cols):
    c=LogisticRegression(max_iter=2000).fit(F[xt][:,cols],y[xt]); return e41.selective(c.predict_proba(F[xe][:,cols])[:,1],y[xe])['aurc']
print('all',round(run(list(range(6))),4))
for k in range(6): print('only',names[k],round(run([k]),4),' drop',names[k],round(run([j for j in range(6) if j!=k]),4))
print('logit+mean_c',round(run([0,5]),4),'logit+v1v0',round(run([0,1,2]),4))
