"""E16-1 第3段：(i)だけ・(iii)だけが替えた主張を常識で採点（plans/E16-1_wobble_routing.md 末尾）。cs/ の判定を使う。
python3 e161d.py ../e4/e42_data.json ../../data/E9n_checker_scores.json （E161_arrays_s0.npz と cs/ を同じ場所に置く）"""
import json, sys, numpy as np
D = json.load(open(sys.argv[1])); CK = json.load(open(sys.argv[2]))
CATS = list(D['cats']); K = len(CATS); ents = sorted(D['ents']); E = len(ents); ei = {w: i for i, w in enumerate(ents)}; ci = {c: i for i, c in enumerate(CATS)}
Y = np.zeros((E, K))
for c in D['claims']:
    if c['kind'] == 'x': Y[ei[c['ent']], ci[c['cat']]] = c['gold']
sc = np.array([-1 if x is None else x for x in CK['scores_e8D']], float).reshape(E, K); tad = (sc > 500).ravel()
ans = (np.load('E161_arrays_s0.npz')['p'] >= 0.5).ravel(); y = Y.ravel()
S = json.load(open('cs/sets.json')); todo = json.load(open('cs/todo_keys.json'))
lab = {int(k): v for k, v in S['pre'].items()}
for f in ['a', 'b', 'c', 'd', 'e']:
    for t in open(f'cs/lab_{f}.txt').read().split():
        j, v = t.split(':'); lab[todo[int(j)]] = int(v)
cs = {k: (1.0 if v >= 1 else 0.0) for k, v in lab.items()}
rng = np.random.default_rng(0); out = {}
for b, st in S['sets'].items():
    r = {}
    for nm in ['only_i', 'only_iii']:
        ks = np.array(st[nm]); c = np.array([cs[k] for k in ks])
        r[nm] = dict(n=len(ks), cs_gain=(tad[ks] == c).astype(int) - (ans[ks] == c).astype(int), wn_gain=(tad[ks] == y[ks]).astype(int) - (ans[ks] == y[ks]).astype(int))
    gi, gt = r['only_i']['cs_gain'], r['only_iii']['cs_gain']
    diff = gi.sum() - gt.sum()
    boot = [rng.choice(gi, len(gi)).sum() - rng.choice(gt, len(gt)).sum() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    wn = int(r['only_i']['wn_gain'].sum() - r['only_iii']['wn_gain'].sum())
    out[b] = dict(n_only_i=len(gi), n_only_iii=len(gt), cs_net_only_i=int(gi.sum()), cs_net_only_iii=int(gt.sum()), cs_diff=int(diff), ci95=[float(lo), float(hi)],
                  wn_net_only_i=int(r['only_i']['wn_gain'].sum()), wn_net_only_iii=int(r['only_iii']['wn_gain'].sum()), wn_diff=wn,
                  agree_wn_cs=float(np.mean([cs[k] == y[k] for k in list(st['only_i']) + list(st['only_iii'])])))
pos = sum(v['ci95'][0] > 0 for v in out.values()); neg = sum(v['ci95'][1] < 0 for v in out.values())
out['verdict'] = '常識でも矛盾の順の勝ちは残る' if pos >= 2 else ('WordNetの癖による勝ちだった' if neg >= 2 else '常識では同着')
print(json.dumps(out, ensure_ascii=False, indent=1)); json.dump(out, open('E161d_out.json', 'w'), ensure_ascii=False, indent=1)
