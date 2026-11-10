"""E5：AIが作ったルールの外れは、答え直しでどれだけ損か。計画: PLAN_E5.md（固定）。
使い方（analysis/e5 で）: python3 e5.py ../e4/e42_data.json ../../data/E4-2_answers_gemma-2-2b-it.json labels/sample_labeled.json out.json
常識の判定json の "mine"（1=正しい, 2=どちらとも, 0=違う。1以上を正しいとして採点）を差し替えると、Geminiの判定でも採点できる（labels/sample_gemini*.json）。
"""
import json, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e4"))
import e42
from e41 import sig, selective

BIG = 30.0
SPLITS = [0, 1, 2, 3, 4]
SOURCES = ["R0_none", "R1_wordnet", "R2_gemma", "R3_fix20", "R3_fix50", "R3_fix80",
           "R3a_fix_isa", "R3b_fix_disj", "R4_strong"]
T = ["trans", "disj", "anti", "cd"]


def rule_state(src, kind, logit, y, rng):
    """ルール（分類どうしの主張）の はい/いいえ と、Gemmaの確率で置く(疑う)ものの印 を返す。物の主張は触らない。"""
    r = (kind != "x"); g = (logit > 0)
    s = g.copy(); soft = r.copy()
    err = r & (g != y)
    if src == "R0_none":
        s = np.zeros_like(g); soft = np.zeros_like(r)
    elif src == "R1_wordnet":
        s = y.astype(bool).copy(); soft = np.zeros_like(r)
    elif src.startswith("R3_fix"):
        f = int(src[-2:]) / 100
        ids = np.where(err)[0]
        fix = rng.choice(ids, size=int(round(f * len(ids))), replace=False)
        s[fix] = y[fix]; soft[fix] = False
    elif src == "R3a_fix_isa":
        fix = err & (kind == "c"); s[fix] = y[fix]; soft[fix] = False
    elif src == "R3b_fix_disj":
        fix = err & (kind == "d"); s[fix] = y[fix]; soft[fix] = False
    elif src == "R4_strong":
        for k in ["c", "d"]:
            m = (kind == k) & g
            thr = np.median(logit[m])
            drop = m & (logit < thr)
            s[drop] = False; soft[drop] = False
    s = np.where(r, s, g)  # 物の主張はGemmaの答え（矛盾を数える時の周り）
    return s.astype(np.int8), soft & r


def main():
    D = json.load(open(sys.argv[1]))
    allc = D["claims"]; Y = np.array([c["gold"] for c in allc])
    if sys.argv[2] == "MOCK":
        r0 = np.random.default_rng(123)
        L = 2.0 * (2 * Y - 1) - 1.0 + r0.normal(0, 2.0, len(Y))
    else:
        A = json.load(open(sys.argv[2]))
        assert A["stmt"] == [c["stmt"] for c in allc], "答えと問いが合わない"
        L = np.array(A["logit"])
    var, factors, ent_x, xtype = e42.build(D, L)
    n = len(var); logit = L[var]; y = Y[var].astype(bool)
    kind = np.array([allc[i]["kind"] for i in var])
    big = e42.WNet(n, factors); ftypes = np.array([t for t, _ in factors])
    ents = list(ent_x)
    S = json.load(open(sys.argv[3]))
    snode = np.array([s["node"] for s in S]); smine = np.array([s["mine"] >= 1 for s in S])
    stype = np.array([s["type"] for s in S]); sent = [s["ent"] for s in S]
    # 物ごとの他の候補の自信（特徴、ルールと関係ない）
    f34 = np.zeros((n, 2))
    for w in ents:
        ids = np.array(ent_x[w])
        for i in ids:
            o = logit[ids[ids != i]]; f34[i] = o.max(), o.mean()
    trans_first = [(v[0], v[1]) for t, v in factors if t == "trans"]
    tf_x = np.array([a for a, _ in trans_first]); tf_c = np.array([c for _, c in trans_first])
    # ルールの外れの数（網の中の分類どうしの主張）
    rs = {}
    for k in ["c", "d"]:
        m = kind == k; g = logit[m] > 0
        rs[k] = dict(n=int(m.sum()), gemma_yes=int(g.sum()), yes_correct=int((g & y[m]).sum()),
                     no_wrong=int((~g & y[m]).sum()), errors=int((g != y[m]).sum()))
    print("網", n, "主張", len(factors), "ルール", rs, flush=True)
    t0 = time.time(); out = dict(rules=rs, splits=[])
    for sp in SPLITS:
        r = np.random.default_rng(1000 + sp)
        perm = r.permutation(len(ents)); tune = set(ents[i] for i in perm[: len(ents) // 5])
        xt = np.array([i for w in ents if w in tune for i in ent_x[w]])
        xe = np.array([i for w in ents if w not in tune for i in ent_x[w]])
        sev = np.array([e not in tune for e in sent])
        def score(p):
            d = selective(p[xe], y[xe])
            pe = p[snode[sev]]
            d["aurc_common"] = selective(pe, smine[sev])["aurc"]
            d["aurc_wn_sample"] = selective(pe, y[snode[sev]])["aurc"]
            for tt in "TGHE":
                m = np.array([xtype[i] == tt for i in xe])
                d[f"acc_{tt}"] = float(np.mean((p[xe][m] >= .5) == y[xe][m]))
            for tt in "TG":
                m = stype[sev] == tt
                d[f"acc_{tt}_common"] = float(np.mean((pe[m] >= .5) == smine[sev][m]))
            return d
        res = {"1_conf": score(sig(logit))}
        rng_fix = np.random.default_rng(5000 + sp)
        for src in SOURCES:
            st, soft = rule_state(src, kind, logit, y, rng_fix)
            R = {}
            R["2_count"] = score(sig(logit + big.local(st[None, :], 1.0)[0]))
            SC = big.signed_counts(st, ftypes, T)
            v1, v0 = big.counts(st)
            cs = np.bincount(tf_x, st[tf_c], n) / np.maximum(np.bincount(tf_x, minlength=n), 1)
            feat = np.column_stack([logit, v1, v0, f34, cs, SC[:, :2]])
            lr = LogisticRegression(max_iter=3000).fit(feat[xt], y[xt])
            R["5_learn"] = score(lr.predict_proba(feat)[:, 1])
            Xw = np.column_stack([logit, SC[:, :2]])
            lrw = LogisticRegression(max_iter=3000, C=1e4).fit(Xw[xt], y[xt])
            lam, Wt, Wd = lrw.coef_[0]; b = lrw.intercept_[0]
            Wt, Wd = max(Wt, 0.0), max(Wd, 0.0); Wo = (Wt + Wd) / 2
            Wm = {"trans": Wt, "disj": Wd, "anti": Wo, "cd": Wo}
            Wf = np.array([Wm[t] for t in ftypes])
            frozen = kind != "x"
            u = lam * logit + b
            u3 = np.where(frozen, np.where(st == 1, BIG, -BIG), u)
            p3, c3, _ = big.bp(u3, Wf, frozen=frozen)
            R["3_trust_net"] = score(p3); R["3_trust_net"]["converged"] = c3
            u4 = np.where(soft, logit, u3)
            p4, c4, _ = big.bp(u4, Wf, frozen=frozen)
            R["4_doubt_net"] = score(p4); R["4_doubt_net"]["converged"] = c4
            R["weights"] = dict(lam=float(lam), b=float(b), trans=float(Wt), disj=float(Wd))
            m = kind != "x"
            R["rule_acc"] = float(np.mean(st[m] == y[m]))
            res[src] = R
            print(sp, src, {k: round(v["aurc"], 4) for k, v in R.items() if isinstance(v, dict) and "aurc" in v},
                  "常識", {k: round(v["aurc_common"], 4) for k, v in R.items() if isinstance(v, dict) and "aurc" in v},
                  round(time.time() - t0), flush=True)
        out["splits"].append(res)
    out["meta"] = dict(n=n, factors=len(factors), seconds=time.time() - t0)
    json.dump(out, open(sys.argv[4], "w"), indent=1)


if __name__ == "__main__":
    main()
