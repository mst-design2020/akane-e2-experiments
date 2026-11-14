"""E19 check added after review (2026-11-14): thresholds are tuned on the common bucket, so the common-bucket
results were measured on the tuning words. Here the common words are split in half (seeds 0-4): thresholds are
tuned on one half and the flags are measured on the other half, with the same rules as e19_analyze.py.
Run from analysis/e19 after judge_combine.py:  python e19_holdout.py  -> results/E19/E19_holdout.json"""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e19_analyze as A
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "..", "data", "E19")
R = os.path.join(HERE, "..", "..", "results", "E19")
c = json.load(open(f"{D}/c19.json"))
b = np.array([it[0] for it in c["items"]]); X = np.array([it[1] for it in c["items"]])
keys = [f"{it[1]}|{it[2]}" for it in c["items"]]
G2, G9, S, T = (np.array(c[k], float) for k in ("G2", "G9", "S", "T"))
gold = np.array(c["gold"], int)


def labels(name):
    if name == "wordnet": return gold
    cs = json.load(open(f"{R}/judge/cs_{name}.json")); y = gold.copy()
    for j, k in enumerate(keys):
        if k in cs: y[j] = cs[k]
    return y


def th(v, y, ge=False):
    return A.fit(lambda t: (v >= t) if ge else (v > t), [A.cands(v)], y)[0][0]


out = {}
for name in ("wordnet", "claude", "chatgpt", "gemini", "vote"):
    y = labels(name); runs = []
    words = sorted(set(X[b == "common"]))
    for seed in range(5):
        w = list(words); np.random.default_rng(seed).shuffle(w); half = set(w[:len(w) // 2])
        inA = np.array([x in half for x in X]); tm = (b == "common") & inA; ev = (b == "common") & ~inA
        YES = {"G2": G2 > th(G2[tm], y[tm]), "G9": G9 > th(G9[tm], y[tm]), "S": S > th(S[tm], y[tm])}
        p = A.fit(lambda t, u: (G2[tm] > t) | (T[tm] >= u), [A.cands(G2[tm]), A.cands(T[tm])], y[tm])[0]
        YES["TO"] = (G2 > p[0]) | (T >= p[1])
        row = {}
        for key, a_, b_, f, sd in (("A_TO_vs_G2", "TO", "G2", A.flag_ab, 1), ("B_TO_vs_S", "TO", "S", A.flag_ab, 2),
                                   ("C_TO_vs_G9", "TO", "G9", A.flag_c, 3), ("ref_S_vs_G2", "S", "G2", A.flag_ab, 4)):
            d = A.boot(YES[a_][ev], YES[b_][ev], y[ev], X[ev], sd); row[key] = {**d, "flag": f(d)}
        runs.append(row)
    out[name] = runs
    print(name, [" ".join(f"{k[:1]}:{r[k]['diff']:+.3f}{r[k]['flag']}" for k in r) for r in runs])
json.dump(out, open(f"{R}/E19_holdout.json", "w"), ensure_ascii=False, indent=1)
