import sys,json,numpy as np
sys.argv=['x']; sys.path.insert(0,'.')
import e102_analyze as E
from sklearn.linear_model import LogisticRegression
S='/tmp/claude-0/-home-user/4394446b-3cce-5fb8-b7fa-8c62205a2ad7/scratchpad'; R='/mnt/project-files/akane-research/sims/e10'
orig=E.methods
def methods(D,tune):
    P,ent,freq=orig(D,tune); y=D['y']; has=(D['nf']>0).astype(float)
    def cal(X): return LogisticRegression(max_iter=1000).fit(X[tune],y[tune]).predict_proba(X)[:,1]
    P['A+has']=cal(np.column_stack([D['lg'],has])); P['A+has+nf']=cal(np.column_stack([D['lg'],has,D['nf']]))
    P['gen_hasonly']=np.where(has>0,P['gen'],P['A'])   # 文がある所だけ⑪g、ない所は①
    P['A_hasonly_genS']=np.where(has>0,P['genS'],P['A']) if 'genS' in P else P['A']
    return P,ent,freq
E.methods=methods
for lab in ['claude','gemini','agree']:
    items_n=json.load(open(f'{S}/e101/e101_items.json'))['items']
    _,d101,_=E.build101(f'{S}/e101/e101_items.json',f'{S}/e101/res'); one={d:d101[d]['F']['oneN'] for d in 'PDG'}
    gN=E.lgz(f'{S}/e101/res/E101_gemma.json.gz')['logit']
    SN=E.load_set('N',items_n,gN,f'{S}/e102/res',one)
    cs={int(k):v for k,v in json.load(open(f'{S}/e101/cs_labels_{lab}.json')).items()}
    SN={'P':SN['P']}
    import types
    rep={}
    D=SN['P']; y=D['y']
    ents=sorted({it['ent'] for it in D['its']}); eidx={e:j for j,e in enumerate(ents)}; ent=np.array([eidx[it['ent']] for it in D['its']])
    rng=np.random.default_rng(0); perm=rng.permutation(len(ents)); fold=np.zeros(len(ents),int)
    for f in range(5): fold[perm[f::5]]=f
    ycs=np.array([cs.get(items_n[i]['id'],-1) for i in D['idx']])
    out={}
    for sp in range(5):
        tune=fold[ent]==sp; ev=~tune; P,_,_=methods(D,tune); m=ev&(ycs>=0)&(ycs!=2); yc=(ycs[m]==1).astype(int)
        for k,v in P.items(): out.setdefault(k,[]).append(E.aurc(v[m],yc))
        for k,v in P.items(): out.setdefault('WN_'+k,[]).append(E.aurc(v[ev],y[ev]))
    print(lab,{k:round(float(np.mean(v)),3) for k,v in out.items() if not k.startswith('WN_') and k in ('A','gen','genS','A+has','A+has+nf','gen_hasonly','Cpp','Xpp')})
    print('  A+has vs gen per run',[ (round(a,3),round(b,3)) for a,b in zip(out['A+has'],out['gen'])])
    if lab=='claude': print('  WordNet',{k[3:]:round(float(np.mean(v)),3) for k,v in out.items() if k.startswith('WN_') and k[3:] in ('A','gen','A+has','gen_hasonly')})
