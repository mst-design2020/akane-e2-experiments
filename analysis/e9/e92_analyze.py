"""E9-2 のまとめ（PLAN_E9-2.md）。主な採点：確かめが入った物の、まだ確かめていない物の主張。
D 対 学習の一番よいもの（C・C+・C++・それぞれの _sub）。差0.02以上が5回中4回以上で差あり。"""
import json, collections, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(H, '..', '..', 'results', 'E9', 'E92_results.json')))['results']
g = collections.defaultdict(list)
for r in R:
    g[(r['budget'], r['select'])].append(r)
out = {}
for key, rs in sorted(g.items()):
    if key[0] >= 1.0:
        continue
    L = [m for m in ['C', 'Csub', 'Cp', 'Cp_sub', 'Cpp', 'Cpp_sub'] if m in rs[0]]
    f = lambda m: np.array([r[m]['aurc_same_ent_checked'] for r in rs])
    row = {m: round(float(f(m).mean()), 4) for m in L + ['D', 'Fpp'] if m in rs[0]}
    best = min(L, key=lambda m: f(m).mean()); d = f(best) - f('D')
    row.update(best=best, D_better_002=int((d >= 0.02).sum()), D_worse_002=int((d <= -0.02).sum()),
               Fpp_minus_Cpp=round(float((f('Cpp') - f('Fpp')).mean()), 4),
               common={m: round(float(np.mean([r[m]['aurc_common'] for r in rs])), 4) for m in ['C', 'Cpp', 'D', 'Fpp']})
    out[f'{key[0]}|{key[1]}'] = row
    print(key, row)
json.dump(out, open(os.path.join(H, '..', '..', 'results', 'E9', 'E92_analysis.json'), 'w'), ensure_ascii=False, indent=1)
