"""E9 の結果のまとめ（PLAN_E9.md の決まり：差0.02以上が5回中4回以上で「差あり」）"""
import json, collections, numpy as np, os
R = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'E9', 'E9_results.json')))['results']
M = ['A', 'B', 'B2', 'C', 'D', 'F']
g = collections.defaultdict(list)
for r in R:
    g[(r['noise'], r['budget'], r['select'])].append(r)
out = {}
for key, rs in sorted(g.items()):
    a = {m: np.array([r[m]['aurc'] for r in rs]) for m in M}
    bestsimple = min(['B', 'B2', 'C'], key=lambda m: a[m].mean())
    dD = a[bestsimple] - a['D']; dF = a['C'] - a['F']
    verdict = lambda d: ('網が上' if (d >= 0.02).sum() >= 4 else '網が下' if (d <= -0.02).sum() >= 4 else '同着')
    row = {m: round(float(a[m].mean()), 4) for m in M}
    row.update(best_simple=bestsimple, D_vs_best=round(float(dD.mean()), 4), D_verdict=verdict(dD),
               F_vs_C=round(float(dF.mean()), 4), F_wins=int((dF > 0).sum()), F_verdict=verdict(dF),
               common={m: round(float(np.mean([r[m]['aurc_common'] for r in rs])), 4) for m in ['C', 'D', 'F']},
               same_ent={m: round(float(np.mean([r[m].get('aurc_same_ent_checked', np.nan) for r in rs])), 4) for m in ['C', 'D', 'F']},
               accG={m: round(float(np.mean([r[m]['acc_G'] for r in rs])), 3) for m in ['A', 'C', 'D', 'F']},
               n=len(rs))
    out['|'.join(map(str, key))] = row
    print(key, row)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'E9', 'E9_analysis.json'), 'w'), ensure_ascii=False, indent=1)
