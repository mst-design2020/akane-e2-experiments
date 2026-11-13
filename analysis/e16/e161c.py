"""E16-1 第2段の追加：決まりを破る所を数えるだけの比べ (vii)。計画 PLAN_E16-1.md 末尾。
python3 e161c.py ../../e4/e42_data.json ../../e4/E42_gemma-2-2b-it.json ../../e10/nolist/results/E9n/E9n_checker_scores.json ../../e4/relabel/sample_labeled.json out.json"""
import json, sys
import numpy as np
D = json.load(open(sys.argv[1])); G = json.load(open(sys.argv[2])); CK = json.load(open(sys.argv[3])); SAMPLE = json.load(open(sys.argv[4]))
CATS = list(D["cats"]); K = len(CATS); ci = {c: i for i, c in enumerate(CATS)}
ents = sorted(D["ents"]); E = len(ents); ei = {w: i for i, w in enumerate(ents)}
assert CK["ents"] == ents
A = np.zeros((K, K))
for c, ps in D["parent"].items():
    for p in ps: A[ci[c], ci[p]] = 1
Dm = np.zeros((K, K))
for k, v in D["disjoint"].items():
    if v:
        a, b = k.split("|"); Dm[ci[a], ci[b]] = Dm[ci[b], ci[a]] = 1
L = G["logit"]; LG = np.zeros((E, K)); Y = np.zeros((E, K))
for j, c in enumerate(D["claims"]):
    if c["kind"] == "x":
        LG[ei[c["ent"]], ci[c["cat"]]] = L[j]; Y[ei[c["ent"]], ci[c["cat"]]] = c["gold"]
X = (LG > 0).astype(float)
v = X * ((1 - X) @ A.T) + (1 - X) * (X @ A) + X * (X @ Dm.T)   # 主張ごとの破り数
sc = np.array([-1 if x is None else x for x in CK["scores_e8D"]], float).reshape(E, K)
can = sc >= 0; tadori = sc > 500
samp = [(ei[s["ent"]], ci[[c for c in CATS if D["cats"][c] == s["cat"]][0]], 1.0 if s["mine"] >= 1 else 0.0) for s in SAMPLE if s["ent"] in ei]
BUD = [0.01, 0.025, 0.05]; n_all = E * K
print("破りのある主張", int((v > 0).sum()), "そのうちGemmaの間違い", int(((v > 0) & (X != Y)).sum()))
out = {"n_viol_claims": int((v > 0).sum()), "n_viol_wrong_gemma": int(((v > 0) & (X != Y)).sum())}
for seed in range(5):
    Ar = np.load(f"E161_arrays_s{seed}.npz"); p = Ar["p"]; ans = p >= 0.5; U = Ar["U"]
    au = np.abs(U); pri = {"i_conflict": Ar["C"], "iii_tree": -Ar["pred3"], "vii_viol": v - au / (au.max() + 1)}
    base_wrong = int((ans != Y).sum()); base_cs = sum(int(ans[e, c] != y) for e, c, y in samp)
    res = {}
    for b in BUD:
        n = int(round(b * n_all)); r = {}
        for name, s in pri.items():
            s = np.where(can, s, -np.inf).ravel(); pick = np.argsort(-s, kind="stable")[:n]
            new = ans.copy().ravel(); new[pick] = tadori.ravel()[pick]; new = new.reshape(E, K)
            r[name] = dict(fixed=base_wrong - int((new != Y).sum()), wrong_in_pick=int((ans.ravel()[pick] != Y.ravel()[pick]).sum()),
                           viol_in_pick=int((v.ravel()[pick] > 0).sum()), fixed_cs=base_cs - sum(int(new[e, c] != y) for e, c, y in samp))
        res[str(b)] = r
    out[seed] = dict(base_wrong=base_wrong, res=res)
    print(seed, {b: {k: x["fixed"] for k, x in r.items()} for b, r in res.items()})
win = {str(b): sum(out[s]["res"][str(b)]["i_conflict"]["fixed"] >= 1.1 * out[s]["res"][str(b)]["vii_viol"]["fixed"] for s in range(5)) for b in BUD}
out["rule"] = dict(wins_per_budget=win, works=sum(x >= 4 for x in win.values()) >= 2)
out["mean"] = {str(b): {k: float(np.mean([out[s]["res"][str(b)][k]["fixed"] for s in range(5)])) for k in pri} for b in BUD}
out["mean_cs"] = {str(b): {k: float(np.mean([out[s]["res"][str(b)][k]["fixed_cs"] for s in range(5)])) for k in pri} for b in BUD}
print(json.dumps({k: out[k] for k in ["rule", "mean", "mean_cs"]}, ensure_ascii=False))
json.dump(out, open(sys.argv[5], "w"), ensure_ascii=False, indent=1)
