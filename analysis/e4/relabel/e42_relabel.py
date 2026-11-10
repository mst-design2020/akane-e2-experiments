"""Run from the repo root after placing e42_data.json and the answers file next to analysis/e4 (see analysis/e4/e42_data.py).
E4-2の主な方法の予測を出し直し、正解表（WordNet）と Claudeの常識判定（relabel/sample）で比べる。p-bit・格子は省く。"""
import json, sys, numpy as np
sys.path.insert(0, 'analysis/e4'); sys.path.insert(0, 'analysis/e4/relabel')
from sklearn.linear_model import LogisticRegression
import e42
from e41 import sig, selective
from my_labels import LAB
D = json.load(open('e42_data.json')); allc = D['claims']; Y = np.array([c['gold'] for c in allc])
L = np.array(json.load(open('data/E4-2_answers_gemma-2-2b-it.json'))['logit'])
var, factors, ent_x, xtype = e42.build(D, L)
n = len(var); logit = L[var]; y = Y[var]; kind = np.array([allc[i]['kind'] for i in var])
T = ["trans", "disj", "anti", "cd"]; big = e42.WNet(n, factors); ftypes = np.array([t for t, _ in factors])
sm_facs, src, xmap = e42.small_copy(factors, ent_x); smnet = e42.WNet(len(src), sm_facs); smt = np.array([t for t, _ in sm_facs])
ents = list(ent_x); a = (logit > 0).astype(np.int8); SC = big.signed_counts(a, ftypes, T)
feat = np.zeros((n, 6)); feat[:, 0] = logit; v1, v0 = big.counts(a); feat[:, 1] = v1; feat[:, 2] = v0
for w in ents:
    ids = np.array(ent_x[w])
    for i in ids:
        o = logit[ids[ids != i]]; feat[i, 3] = o.max(); feat[i, 4] = o.mean()
cs = np.zeros(n); cc = np.zeros(n)
for t, v in factors:
    if t == "trans": cs[v[0]] += logit[v[1]]; cc[v[0]] += 1
feat[:, 5] = cs / np.maximum(cc, 1); feat3 = np.hstack([feat, SC[:, :2]])
S = json.load(open('analysis/e4/relabel/sample.json'))
node = np.array([s['node'] for s in S]); mine = np.array([LAB.get(s['id'], 0) for s in S])
out = {}
for sp in [0, 1, 2, 3, 4]:
    r = np.random.default_rng(1000 + sp); perm = r.permutation(len(ents)); tune = set(ents[i] for i in perm[: len(ents) // 5])
    xt = np.array([i for w in ents if w in tune for i in ent_x[w]])
    P = {"1_conf": sig(logit)}
    P["2_conf_cal"] = LogisticRegression(max_iter=2000).fit(logit[xt, None], y[xt]).predict_proba(logit[:, None])[:, 1]
    P["3_learn"] = LogisticRegression(max_iter=2000).fit(feat3[xt], y[xt]).predict_proba(feat3)[:, 1]
    lrw = LogisticRegression(max_iter=2000, C=1e4).fit(np.hstack([logit[:, None], SC[:, :2]])[xt], y[xt])
    lam, Wt, Wd = lrw.coef_[0]; b = lrw.intercept_[0]; Wt, Wd = max(Wt, 0), max(Wd, 0); Wo = (Wt + Wd) / 2
    Wm = {"trans": Wt, "disj": Wd, "anti": Wo, "cd": Wo}; Wf = np.array([Wm[t] for t in ftypes]); u = lam * logit + b
    P["5_big"] = big.bp(u, Wf)[0]
    ps = smnet.bp(u[src], np.array([Wm[t] for t in smt]))[0]; p4 = np.zeros(n)
    for i, j in xmap.items(): p4[i] = ps[j]
    P["4_small"] = p4
    P["5c_cautious"] = big.bp(u, Wf, frozen=kind != "x")[0]
    ev = np.array([s['ent'] not in tune for s in S])   # 評価用の名詞だけ
    res = {}
    for k, p in P.items():
        pp = p[node[ev]]
        res[k] = dict(wordnet=selective(pp, y[node[ev]])['aurc'],
                      mine_2true=selective(pp, mine[ev] >= 1)['aurc'],
                      mine_2false=selective(pp, mine[ev] == 1)['aurc'])
        for tt in 'GT':
            m = np.array([s['type'] == tt for s in S])[ev]
            res[k][f'acc_{tt}_wordnet'] = float(np.mean((pp[m] >= .5) == y[node[ev]][m]))
            res[k][f'acc_{tt}_mine'] = float(np.mean((pp[m] >= .5) == (mine[ev][m] >= 1)))
    out[sp] = res
    print(sp, {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in res.items()}, flush=True)
json.dump(out, open('results/E4/E42_relabel.json', 'w'), indent=1)
