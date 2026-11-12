"""事後の見方（計画には入れていない）：いいとこ取りの何が効いたか。1つずつ足す・1つずつ抜く。
python3 e112_ablate.py E8n_results.json E11-1_results.json E11-2_results.json out.json"""
import json, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e112_analyze as M
A, e6, GB, sig, judge, Z = M.A, M.e6, M.GB, M.sig, M.judge, M.Z
r8, r11, r12 = (json.load(open(p)) for p in sys.argv[1:4])
res, allit, S, DEP, PATH, L, Dk = M.run(r8, r11, r12, log=lambda *a: None)
L8, ST8, AR8, CH8 = A.load(r8); old, new = A.labels(); no = len(old)
io = np.arange(no); inw = np.arange(no, len(allit))
y = np.array([int(x["mine"] >= 1) for x in allit]); yo, yn = y[io], y[inw]
eo = np.array([x["ent"] for x in old]); en = np.array([x["ent"] for x in new])
LG = np.array([it["logit"] for it in allit])
its8 = [A.Item(it, "D", *CH8[(it["id"], "D")], L8, ST8, AR8) for it in allit]
S11, _ = Z.scores(allit, its8, r11["ent_p"], r11.get("leak_p", {}))
F = [A.x_feats(it, L8, AR8) for it in allit]
X5 = np.column_stack([LG, np.array([f[0] for f in F]), np.array([f[1] for f in F])])
parts = {"deep": [S[Dk], DEP[Dk]], "box": [S["BX"]], "R3": [S11["R3_weight"]], "R1": [S11["R1_first"]], "R2": [S11["R2_gemma"]]}
base = [X5, S["R0_all"], DEP["R0_all"]]
def ev(cols):
    X = np.column_stack(cols); per = M.gb_cv(X[io], yo, eo)
    q = GB().fit(X[io], yo).predict_proba(X[inw])[:, 1]
    return {"old_mean": float(np.mean(per)), "old_per": per, "new": e6.aurc(q, yn)}
out = {"base": ev(base)}
for k in ("deep", "box", "R3"):
    out["add_" + k] = ev(base + parts[k])
allc = base + parts["deep"] + parts["box"] + parts["R3"]
out["all"] = ev(allc)
for k in ("deep", "box", "R3"):
    out["drop_" + k] = ev(base + sum([parts[j] for j in ("deep", "box", "R3") if j != k], []))
out["add_R1"] = ev(base + parts["R1"]); out["add_R2"] = ev(base + parts["R2"])
json.dump(out, open(sys.argv[4], "w"), indent=1)
for k, v in out.items(): print(f"{k:12s} 700 {v['old_mean']:.4f}  300 {v['new']:.4f}")
