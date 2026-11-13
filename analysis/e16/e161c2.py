"""参考：網なしの流れ（Gemmaの答え→破る所をTADORIで確かめる）の最後の間違い数と、網＋矛盾の順の最後の間違い数を比べる。e161c.py と同じ引数。"""
import json, sys, runpy
import numpy as np
sys.argv = sys.argv[:5] + ["/dev/null"]
g = runpy.run_path("e161c.py")
X, v, Y, can, tadori, BUD, n_all = g["X"], g["v"], g["Y"], g["can"], g["tadori"], g["BUD"], g["n_all"]
res = {}
for seed in range(5):
    Ar = np.load(f"E161_arrays_s{seed}.npz"); p = Ar["p"]; U = Ar["U"]; au = np.abs(U)
    gem = U > 0; net = p >= 0.5
    r = {"gemma_alone": int((gem != Y).sum()), "net_alone": int((net != Y).sum())}
    for b in BUD:
        n = int(round(b * n_all))
        for nm, base, s in [("gemma+viol", gem, v - au / (au.max() + 1)), ("gemma+unsure", gem, -au), ("net+conflict", net, Ar["C"])]:
            s = np.where(can, s, -np.inf).ravel(); pick = np.argsort(-s, kind="stable")[:n]
            new = base.copy().ravel(); new[pick] = tadori.ravel()[pick]
            r[f"{nm}@{b}"] = int((new.reshape(Y.shape) != Y).sum())
    res[seed] = r
m = {k: float(np.mean([res[s][k] for s in res])) for k in res[0]}
print(json.dumps(m, ensure_ascii=False, indent=1)); json.dump(dict(seeds=res, mean=m), open("E161c2_out.json", "w"), ensure_ascii=False, indent=1)
