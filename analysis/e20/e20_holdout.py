"""E20 check added after review (2026-11-14): the common bucket was used both to tune the thresholds and to report SO - S.
Here the common-bucket words are split in half (seeds 0-4): thresholds are tuned on one half (all pairs) and
SO - S is measured on the other half (main subset), with the same bootstrap and rule as e20_analyze.py.
Run from analysis/e20 after judge_combine.py:  python e20_holdout.py  -> results/E20/E20_holdout.json"""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e20_analyze as A
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "..", "data", "E20")
R = os.path.join(HERE, "..", "..", "results", "E20")
c = json.load(open(f"{D}/c20.json")); it = c["items"]
gold = np.array(c["gold"], int); keys = [f"{r[0]}|{r[2]}|{r[3]}" for r in it]
src = np.array([r[0] for r in it]); b = np.array([r[1] for r in it])
lit = np.array([int(r[5]) for r in it]); X = np.array([r[2] for r in it])


def labels(name):
    if name == "gold": return gold
    cs = json.load(open(f"{R}/judge/cs_{name}.json")); y = gold.copy()
    for j, k in enumerate(keys):
        if k in cs: y[j] = cs[k]
    return y


out = {}
for name in ("vote", "gold", "claude", "chatgpt", "gemini"):
    y = labels(name); runs = []
    for seed in range(5):
        words = sorted(set(X[(src == "W") & (b == "common")]))
        np.random.default_rng(seed).shuffle(words)
        half = set(words[:len(words) // 2]); inA = np.array([x in half for x in X])
        tm = (src == "W") & (b == "common") & inA
        ev = (src == "W") & (b == "common") & ~inA & ((y == 0) | (lit == 0))
        row = {}
        for i, m in enumerate(c["M"]):
            S = np.array(c["M"][m]["S"], float); T = np.array(c["M"][m]["T"], float)
            ps = A.fit1(S[tm], y[tm]); pso = A.fit_or(S[tm], T[tm], y[tm])
            d = A.boot(((S > pso[0]) | (T >= pso[1]))[ev], (S > ps)[ev], y[ev], X[ev], 10 + i)
            row[m] = {**d, "flag": A.flag(d)}
        runs.append(row)
    out[name] = runs
    print(name, [" ".join(f"{m}:{r[m]['diff']:+.3f}{r[m]['flag']}" for m in r) for r in runs])
json.dump(out, open(f"{R}/E20_holdout.json", "w"), ensure_ascii=False, indent=1)
