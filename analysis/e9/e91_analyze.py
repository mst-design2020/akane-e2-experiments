"""E9-1 のまとめ（PLAN_E9-1.md の決まり）。主な採点：確かめが入った物の、まだ確かめていない物の主張。
主な比べ：D 対 C・Csubのよい方、net 2/5/10% で差0.02以上が5回中4回以上 → その条件で網が上。3条件中2つ以上で「網の勝ちが残った」。"""
import json, collections, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(H, '..', '..', 'results', 'E9', 'E91_results.json')))['results']
g = collections.defaultdict(list)
for r in R:
    g[(r['noise'], r['budget'], r['select'])].append(r)
out = {}
for key, rs in sorted(g.items()):
    noise, budget, sel = key
    row = {'n': len(rs), 'n_checked': rs[0]['n_checked'], 'n_wrong_checks': float(np.mean([r['n_flip'] for r in rs]))}
    M = [m for m in ['A', 'B', 'B2', 'C', 'Csub', 'D', 'F'] if m in rs[0]]
    for fld in ['aurc_same_ent_checked', 'aurc', 'aurc_all_incl_checked', 'aurc_common']:
        row[fld] = {m: round(float(np.mean([r[m].get(fld, np.nan) for r in rs])), 4) for m in M}
    if 'Csub' in rs[0] and budget < 1.0:
        f = lambda m: np.array([r[m]['aurc_same_ent_checked'] for r in rs])
        best = np.minimum(f('C'), f('Csub')); d = best - f('D'); fF = f('C') - f('F')
        row['D_minus_best_same_ent'] = round(float(d.mean()), 4)
        row['D_wins_002'] = int((d >= 0.02).sum()); row['D_loses_002'] = int((d <= -0.02).sum())
        row['F_vs_C_same_ent'] = round(float(fF.mean()), 4)
    out['|'.join(map(str, key))] = row
    print(key, {k: v for k, v in row.items() if k not in ('aurc_common',)})
main = [out.get(f'e8D|{b}|net', {}).get('D_wins_002', 0) >= 4 for b in (0.02, 0.05, 0.1)]
out['verdict'] = {'net_2_5_10_D_wins': main, 'survived': sum(main) >= 2}
print('判定', out['verdict'])
json.dump(out, open(os.path.join(H, '..', '..', 'results', 'E9', 'E91_analysis.json'), 'w'), ensure_ascii=False, indent=1)
