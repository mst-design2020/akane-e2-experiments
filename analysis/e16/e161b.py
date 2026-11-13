"""E16-1 第2段：どこをTADORIで確かめるか。計画 PLAN_E16-1.md「第2段の計画」。
python3 e161b.py ../../e4/e42_data.json ../../e10/nolist/results/E9n/E9n_checker_scores.json ../../e4/relabel/sample_labeled.json out.json（E161_arrays_s*.npz を読む）"""
import json, sys
import numpy as np
D = json.load(open(sys.argv[1])); CK = json.load(open(sys.argv[2])); SAMPLE = json.load(open(sys.argv[3]))
CATS = list(D["cats"]); K = len(CATS); ci = {c: i for i, c in enumerate(CATS)}
ents = sorted(D["ents"]); E = len(ents); ei = {w: i for i, w in enumerate(ents)}
assert CK["ents"] == ents
Y = np.zeros((E, K))
for c in D["claims"]:
    if c["kind"] == "x": Y[ei[c["ent"]], ci[c["cat"]]] = c["gold"]
sc = np.array([-1 if x is None else x for x in CK["scores_e8D"]], float).reshape(E, K)
can = sc >= 0; tadori = sc > 500
samp = [(ei[s["ent"]], ci[[c for c in CATS if D["cats"][c] == s["cat"]][0]], 1.0 if s["mine"] >= 1 else 0.0) for s in SAMPLE if s["ent"] in ei]
BUD = [0.01, 0.025, 0.05]; n_all = E * K
out = {}
for seed in range(5):
    A = np.load(f"E161_arrays_s{seed}.npz"); p = A["p"]; ans = p >= 0.5
    shaky = (p >= 0.1) & (p <= 0.9); rng = np.random.default_rng(100 + seed)
    pri = {"i_conflict": A["C"], "ii_uncertain": -np.abs(p - 0.5), "iii_tree": -A["pred3"],
           "iv_ignorance": np.where(shaky, -A["M"], -np.inf), "v_random": rng.random((E, K)), "vi_tree_MC": -A["pred4"]}
    base_wrong = int((ans != Y).sum()); base_cs = sum(int(ans[e, c] != y) for e, c, y in samp)
    res = {}
    for b in BUD:
        n = int(round(b * n_all)); r = {}
        for name, s in pri.items():
            s = np.where(can, s, -np.inf).ravel(); pick = np.argsort(-s, kind="stable")[:n]
            pick = pick[np.isfinite(s[pick])]
            new = ans.copy().ravel(); new[pick] = tadori.ravel()[pick]; new = new.reshape(E, K)
            r[name] = dict(fixed=base_wrong - int((new != Y).sum()), picked=len(pick), wrong_in_pick=int((ans.ravel()[pick] != Y.ravel()[pick]).sum()),
                           fixed_cs=base_cs - sum(int(new[e, c] != y) for e, c, y in samp))
        res[str(b)] = r
    out[seed] = dict(base_wrong=base_wrong, res=res)
    print(seed, base_wrong, {b: {k: v["fixed"] for k, v in r.items()} for b, r in res.items()})
win = {}
for b in BUD:
    w = 0
    for s in out:
        r = out[s]["res"][str(b)]; best = max(r["ii_uncertain"]["fixed"], r["iii_tree"]["fixed"])
        w += r["i_conflict"]["fixed"] >= 1.1 * best
    win[str(b)] = w
out["rule"] = dict(wins_per_budget=win, works=sum(v >= 4 for v in win.values()) >= 2)
print(json.dumps(out["rule"]))
json.dump(out, open(sys.argv[4], "w"), ensure_ascii=False, indent=1)
