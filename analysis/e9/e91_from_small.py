"""data/E9-1_checker_scores.json（Colab E9-1b の出力 E91_checker_small.json と同じ）→ E91_checker.json（e9.py real が読む形）
使い方: python3 e91_from_small.py ../../data/E9-1_checker_scores.json"""
import json, sys, os
S = json.load(open(sys.argv[1]))
ents, cats = S['ents'], S['cats']
out = {'checkers': {}, 'scores': {}, 'meta': S['meta']}
for name, s in S['checkers'].items():
    d = {}
    for i, w in enumerate(ents):
        for j, C in enumerate(cats):
            ch = s[i * len(cats) + j]
            if ch != '-':
                d[f"{w}|{C}"] = int(ch)
    out['checkers'][name] = d
for name in ('e8D', 'e8one'):
    v = S.get('scores_' + name, [])
    out['scores'][name] = {f"{w}|{C}": v[i * len(cats) + j] / 1000 for i, w in enumerate(ents) for j, C in enumerate(cats) if v and v[i * len(cats) + j] >= 0}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'E91_checker.json'), 'w'))
print({k: len(v) for k, v in out['checkers'].items()}, S.get('n_nodes'), S.get('choice_first_rate'))
