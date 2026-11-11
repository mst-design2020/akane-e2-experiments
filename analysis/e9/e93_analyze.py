"""E9-3 のまとめ（PLAN_E9-3.md の決まり：差が相手の値の20%以上で、5回中4回以上同じ向きなら差あり）"""
import json, glob, os, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = []
for f in sorted(glob.glob(os.path.join(HERE, '..', '..', 'results', 'E9', 'E93_results_s*.json'))):
    R += json.load(open(f))['results']
g = collections.defaultdict(list)
for r in R:
    g[(r['budget'], r['select'])].append(r)
M = ['A', 'K', 'C', 'Cpp', 'D', 'Fpp']


def verdict(mine, opp):
    rel = (opp - mine) / opp
    if (rel >= 0.2).sum() >= 4:
        return '網が上'
    if (rel <= -0.2).sum() >= 4:
        return '網が下'
    return '同着'


out = {}
for key in sorted(g):
    rs = sorted(g[key], key=lambda r: r['split'])
    row = dict(n=len(rs))
    for score in ['aurc_main', 'aurc_all', 'aurc_common_all', 'aurc_common_main']:
        if any(r['A'].get(score) is None for r in rs):
            continue
        a = {m: np.array([r[m][score] for r in rs]) for m in M}
        best = min(['C', 'Cpp'], key=lambda m: a[m].mean())
        row[score] = {m: round(float(a[m].mean()), 4) for m in M}
        row[score + '_judge'] = dict(best_learner=best, D_vs_best=verdict(a['D'], a[best]),
                                     D_rel=[round(float(x), 3) for x in (a[best] - a['D']) / a[best]],
                                     Fpp_vs_Cpp=verdict(a['Fpp'], a['Cpp']),
                                     Fpp_rel=[round(float(x), 3) for x in (a['Cpp'] - a['Fpp']) / a['Cpp']])
    row['check_wrong_rate'] = round(float(np.mean([r['check_wrong'] / max(r['n_checked'], 1) for r in rs])), 3)
    out[f'{key[0]}|{key[1]}'] = row
    print(key, row.get('aurc_main'), row.get('aurc_main_judge'), '| all', row.get('aurc_all'), row.get('aurc_all_judge', {}).get('D_vs_best'), row.get('aurc_all_judge', {}).get('Fpp_vs_Cpp'))
main = [out.get(f'{b}|net', {}).get('aurc_main_judge', {}) for b in (0.02, 0.05, 0.1)]
wins = sum(1 for j in main if j.get('D_vs_best') == '網が上' or j.get('Fpp_vs_Cpp') == '網が上')
out['_verdict'] = dict(net_wins_of_3=wins, result='網の勝ちが残った' if wins >= 2 else '網の勝ちは残らず')
print(out['_verdict'])
json.dump(out, open(os.path.join(HERE, '..', '..', 'results', 'E9', 'E93_analysis.json'), 'w'), ensure_ascii=False, indent=1)
