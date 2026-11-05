import json,glob,sys,numpy as np
def judge(RES, a, b, sub="全体"):
    A = np.array([r["auc"][sub][a] for r in RES]); B = np.array([r["auc"][sub][b] for r in RES])
    ok = ~(np.isnan(A)|np.isnan(B)); A, B = A[ok], B[ok]
    d = (A-B).mean(); w = int((A > B).sum()); n = len(A)
    if d >= 0.02 and w >= 0.8*n: v = f"{a}勝ち"
    elif d <= -0.02 and (B > A).sum() >= 0.8*n: v = f"{b}勝ち"
    elif abs(d) < 0.02: v = "同着"
    else: v = "?"
    return f"{a}-{b} {d:+.3f}({w}/{n}) {v}"
for f in sys.argv[1:]:
    R=json.load(open(f))
    for c in sorted(set(r["cond"] for r in R)):
        RES=[r for r in R if r["cond"]==c]
        for sub in ["全体","どのルールも破っていない主張"]:
            m={k:np.nanmean([r["auc"][sub][k] for r in RES]) for k in ["口の自信","数える","MaxSAT","BP","揺れ"]}
            print(c, sub[:4], len(RES), " ".join(f"{k}{v:.3f}" for k,v in m.items()), "|", judge(RES,"BP","数える",sub), "|", judge(RES,"揺れ","数える",sub), "|", judge(RES,"BP","口の自信",sub))
        print("  acc", {k: round(np.mean([r["acc"][k] for r in RES]),3) for k in ["口","MaxSAT","BP","揺れ"]}, "rules", RES[0].get("n_rules"), "conv", sum(r["bp_converged"] for r in RES))
