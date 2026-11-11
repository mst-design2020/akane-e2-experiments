"""E9-1：Colabの材料（E9-1_results.json）から、物の主張ごとの辞書の確かめの答えを作る → E91_checker.json
e8D  ＝ E8の⑥D（鎖を3段まで、途中の意味はGemmaが選ぶ、一番強い道の掛け算）。0.5より大きければ「はい」。
e8one＝ 1段だけ（物の語の説明文から、分類への含むの一番大きい値）。0.5より大きければ「はい」。
説明文のない語は確かめられない（答えなし）。
使い方: python3 e91_checker.py E9-1_results.json
"""
import json, sys, os
import numpy as np
DEPTH = 3
R = json.load(open(sys.argv[1]))
L = R['L']; CATS = R['meta']['cats']; cidx = {c: j for j, c in enumerate(CATS)}
nid = {tuple(x): i for i, x in enumerate(R['nodes'])}
AE = np.array(R['arrive_e'], dtype=np.float32)
ST = {(w, k): e for w, k, e, c in R['steps']}
out = {'e8D': {}, 'e8one': {}}
score = {'e8D': {}, 'e8one': {}}
for w, (nd, lk) in R['chains'].items():
    nodes = [tuple(x) for x in nd]
    if not nodes:
        continue
    ch = {x: [] for x in nodes}
    for a, b in lk:
        ch[tuple(a)].append(tuple(b))
    roots = [x for x in nodes if x[0] == w]
    for C in CATS:
        j = cidx[C]
        arr = {x: (1.0 if x[0] == C else float(AE[nid[x], j])) for x in nodes}
        memo = {}
        def f(x, d):
            if (x, d) in memo:
                return memo[(x, d)]
            best = arr[x]
            if best < 1.0 and d > 0:
                e = ST.get(x, 0.0)
                for c in ch[x]:
                    best = max(best, e * f(c, d - 1))
            memo[(x, d)] = best
            return best
        s6 = max(f(r, DEPTH) for r in roots) if roots else 0.0
        s1 = max(arr[r] for r in roots) if roots else 0.0
        key = f"{w}|{C}"
        score['e8D'][key] = round(s6, 4); score['e8one'][key] = round(s1, 4)
        out['e8D'][key] = int(s6 > 0.5); out['e8one'][key] = int(s1 > 0.5)
here = os.path.dirname(os.path.abspath(__file__))
json.dump({'checkers': out, 'scores': score, 'meta': R['meta']}, open(os.path.join(here, 'E91_checker.json'), 'w'))
print({k: (len(v), float(np.mean(list(v.values())))) for k, v in out.items()})
