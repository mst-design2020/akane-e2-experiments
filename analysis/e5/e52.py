"""E5-2：病気（E2-8のデータ）で、AIが作ったルールの外れの損。計画: PLAN_E5-2.md（固定）。
使い方（analysis/e5 で）: python3 e52.py ../../data/E2-8_answers_gemma-2-2b-it.json labels_E5-2/E5-2_labels_claude.json out.json
（Geminiの判定は labels_E5-2/E5-2_labels_gemini.json、常識で採点しないなら NONE）
data_dis.py（analysis/）でE2-8の問いを作り直す（DOIDを取ってくる）。
"""
import json, sys, os, time
import numpy as np
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "e4"))
import e42
from e41 import sig, selective
DATA_SEED = 0
_cwd = os.getcwd(); os.chdir(os.path.join(HERE, ".."))
exec(open(os.path.join(HERE, "..", "data_dis.py")).read())
os.chdir(_cwd)
BIG = 30.0
SOURCES = ["R0_none", "R1_doid", "R2_gemma", "R3_fix20", "R3_fix50", "R3_fix80", "R3a_fix_missed", "R3b_fix_overclaimed", "R4_strong"]
T = ["trans", "disj"]


def main():
    d = json.load(open(sys.argv[1])); assert d["stmt"] == STMT
    L = np.array(d["mouth"]); N = len(STMT)
    v = np.zeros((len(PAIRS), 2))
    for (j, o), l in zip(d["rule_qs"], d["rule_logit"]): v[j, o] = l
    Z = v.mean(1); TR = np.array([TRUTH[p] for p in PAIRS])
    n = N + len(PAIRS)
    logit = np.concatenate([L, Z]); y = np.concatenate([GOLD, TR]).astype(bool)
    isrule = np.arange(n) >= N
    factors = [("trans", list(r)) for r in TRANS]
    for j, p in enumerate(PAIRS):
        for i, k in INST[p]:
            factors.append(("disj", [i, k, N + j]))
    net = e42.WNet(n, factors); ftypes = np.array([t for t, _ in factors])
    grp = {i: (CITY[i] if CITY[i] else ("k", STMT[i].split(" is a kind")[0])) for i in range(N)}
    mem = {}
    for i, g in grp.items(): mem.setdefault(g, []).append(i)
    f34 = np.zeros((n, 2))
    for g, ids in mem.items():
        ids = np.array(ids)
        for i in ids:
            o = L[ids[ids != i]]
            if len(o): f34[i] = o.max(), o.mean()
    lab = None
    if sys.argv[2] != "NONE":
        S = json.load(open(sys.argv[2]))
        lab = {s["idx"]: s["label"] >= 1 for s in S if s["part"] == "item"}
        snode = np.array(sorted(lab)); smine = np.array([lab[i] for i in snode])
    g = Z > 0
    err = (g != TR)
    ucities = sorted(set(c for c in CITY if c))
    out = dict(rules=dict(gemma_yes=int(g.sum()), truth=int(TR.sum()), overclaimed=int((g & ~TR).sum()), missed=int((~g & TR).sum())), splits=[])
    t0 = time.time()
    for sp in range(5):
        rng = np.random.default_rng(sp)
        calc = set(rng.choice(ucities, int(round(0.2 * len(ucities))), replace=False))
        xt = np.array([i for i in range(N) if CITY[i] in calc]); xe = np.array([i for i in range(N) if CITY[i] not in calc])
        def score(p):
            dd = selective(p[xe], y[xe])
            if lab is not None:
                m = np.array([CITY[i] not in calc for i in snode])
                dd["aurc_common"] = selective(p[snode[m]], smine[m])["aurc"]
                dd["aurc_doid_sample"] = selective(p[snode[m]], y[snode[m]])["aurc"]
            return dd
        res = {"1_conf": score(sig(logit))}
        rfix = np.random.default_rng(5000 + sp)
        for src in SOURCES:
            s = g.copy(); soft = np.ones(len(PAIRS), bool)
            if src == "R0_none": s[:] = False; soft[:] = False
            elif src == "R1_doid": s = TR.copy(); soft[:] = False
            elif src.startswith("R3_fix"):
                ids = np.where(err)[0]; fx = rfix.choice(ids, int(round(int(src[-2:]) / 100 * len(ids))), replace=False)
                s[fx] = TR[fx]; soft[fx] = False
            elif src == "R3a_fix_missed": fx = err & TR; s[fx] = True; soft[fx] = False
            elif src == "R3b_fix_overclaimed": fx = err & ~TR; s[fx] = False; soft[fx] = False
            elif src == "R4_strong":
                thr = np.median(Z[g]); drop = g & (Z < thr); s[drop] = False; soft[drop] = False
            st = np.concatenate([(L > 0), s]).astype(np.int8)
            R = {}
            R["2_count"] = score(sig(logit + net.local(st[None, :], 1.0)[0]))
            SC = net.signed_counts(st, ftypes, T); v1, v0 = net.counts(st)
            feat = np.column_stack([logit, v1, v0, f34, SC])
            R["5_learn"] = score(LogisticRegression(max_iter=3000).fit(feat[xt], y[xt]).predict_proba(feat)[:, 1])
            lrw = LogisticRegression(max_iter=3000, C=1e4).fit(np.column_stack([logit, SC])[xt], y[xt])
            lam, Wt, Wd = lrw.coef_[0]; b = lrw.intercept_[0]; Wt, Wd = max(Wt, 0.0), max(Wd, 0.0)
            Wf = np.where(ftypes == "trans", Wt, Wd)
            u = lam * logit + b
            u3 = np.where(isrule, np.where(st == 1, BIG, -BIG), u)
            p3, c3, _ = net.bp(u3, Wf, frozen=isrule); R["3_trust_net"] = score(p3); R["3_trust_net"]["converged"] = c3
            soft_all = np.concatenate([np.zeros(N, bool), soft])
            u4 = np.where(soft_all, logit, u3)
            p4, c4, _ = net.bp(u4, Wf, frozen=isrule); R["4_doubt_net"] = score(p4); R["4_doubt_net"]["converged"] = c4
            R["weights"] = dict(lam=float(lam), b=float(b), trans=float(Wt), disj=float(Wd))
            R["rule_acc_doid"] = float(np.mean(s == TR))
            res[src] = R
            print(sp, src, {k: round(v["aurc"], 4) for k, v in R.items() if isinstance(v, dict) and "aurc" in v},
                  ({k: round(v["aurc_common"], 4) for k, v in R.items() if isinstance(v, dict) and "aurc_common" in v} if lab else ""),
                  round(time.time() - t0), flush=True)
        out["splits"].append(res)
    json.dump(out, open(sys.argv[3], "w"), indent=1)


if __name__ == "__main__":
    main()
