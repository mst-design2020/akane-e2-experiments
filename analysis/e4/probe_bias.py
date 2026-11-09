import json, numpy as np, e41
D=json.load(open('e41_data.json')); A=json.load(open('E41_gemma-2-2b-it.json'))
L=np.array(A['logit']); y=np.array([c['gold'] for c in D['claims']]); n=len(L)
big=e41.Net(n,D['factors']); sm,src,xmap=e41.small_nets(D)
ents=D['ents']
for sp in range(5):
    r=np.random.default_rng(1000+sp); perm=r.permutation(len(ents)); te=set(ents[i] for i in perm[:len(ents)//5])
    xt=np.array([i for w in ents if w in te for i in D['ent_claims'][w]]); xe=np.array([i for w in ents if w not in te for i in D['ent_claims'][w]])
    best=None
    for b in [0,1,2,3,4]:
        for lam in e41.LAMS:
            for W in e41.WS:
                p=big.bp(lam*(L+b),W,maxit=200)[0]; s=e41.selective(p[xt],y[xt])['aurc']
                if best is None or s<best[0]: best=(s,b,lam,W)
    _,b,lam,W=best
    pb=big.bp(lam*(L+b),W)[0]; ps=sm.bp(lam*(L[src]+b),W)[0]
    # conf with offset tuned
    bc=min([0,1,2,3,4],key=lambda bb:e41.selective(e41.sig(L[xt]+bb),y[xt])['aurc'])
    print(sp,'b,lam,W',b,lam,W,'conf+b',round(e41.selective(e41.sig(L[xe]+bc),y[xe])['aurc'],4),
          'big',round(e41.selective(pb[xe],y[xe])['aurc'],4),'small',round(e41.selective(np.array([ps[xmap[i]] for i in xe]),y[xe])['aurc'],4),flush=True)
