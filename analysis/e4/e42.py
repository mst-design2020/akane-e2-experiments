"""E4-2：候補選び → 学習した網 と比べ。計画: PLAN_E4-2.md（固定）。
使い方: python3 e42.py e42_data.json <答えのjson|MOCK> <出力json>
"""
import json, sys, time, itertools, os
import numpy as np
from sklearn.linear_model import LogisticRegression
import e41
from e41 import sig, selective, BAD

SWEEPS = int(os.environ.get("E42_SWEEPS", 10000))
BETAS = np.array([1.0, 0.8, 0.6, 0.45])
SPLITS = [0, 1, 2, 3, 4]
LAMS, OFFS, WS = [0.25, 0.5, 1.0, 2.0], [0, 1, 2, 3, 4, 5], [0.5, 1.0, 2.0, 4.0, 8.0]


# ---------- 候補選び（PLAN の「候補の選び方」） ----------
def build(D, logit):
    CATS = list(D["cats"]); dis = D["disjoint"]; parent = D["parent"]
    idx = {}
    for i, c in enumerate(D["claims"]):
        key = ("x", c["ent"], c["cat"]) if c["kind"] == "x" else (c["kind"], c["cat"], c["cat2"])
        idx[key] = i
    isa = lambda a, b: D["claims"][idx[("c", a, b)]]["gold"]
    rng = np.random.default_rng(42)
    sel, ftype = {}, {}
    for w, tc in D["ents"].items():
        true = list(rng.choice(tc, size=min(4, len(tc)), replace=False))
        notc = [c for c in CATS if c not in tc]
        G = max(notc, key=lambda c: logit[idx[("x", w, c)]])
        anc_t = set(itertools.chain.from_iterable(parent[t] for t in true))
        hard = [c for c in notc if c != G and (set(parent[c]) & anc_t or (not parent[c] and any(not parent[t] for t in true)))]
        if hard:
            H = rng.choice(hard)
        else:
            H = sorted([c for c in notc if c != G], key=lambda c: -logit[idx[("x", w, c)]])[0]
        easy = [c for c in notc if c not in (G, H) and all(dis[f"{c}|{t}"] for t in true)]
        cand = true + [G, H] + ([rng.choice(easy)] if easy else [])
        sel[w] = cand
        ftype[w] = {G: "G", H: "H", **({cand[-1]: "E"} if easy else {})}
    # 網
    var, vmap = [], {}
    def V(key):
        if key not in vmap:
            vmap[key] = len(var); var.append(idx[key])
        return vmap[key]
    factors, ent_x, xtype = [], {}, {}
    for w, cand in sel.items():
        ent_x[w] = [V(("x", w, c)) for c in cand]
        for c in cand:
            xtype[vmap[("x", w, c)]] = ftype[w].get(c, "T")
        for a, b in itertools.permutations(cand, 2):
            factors.append(("trans", [vmap[("x", w, a)], V(("c", a, b)), vmap[("x", w, b)]]))
        for a, b in itertools.combinations(sorted(cand), 2):
            if isa(a, b) or isa(b, a):
                continue
            factors.append(("disj", [vmap[("x", w, a)], vmap[("x", w, b)], V(("d", a, b))]))
    for (k, a, b), j in list(vmap.items()):
        if k == "c" and ("c", b, a) in vmap and a < b:
            factors.append(("anti", [j, vmap[("c", b, a)]]))
        if k == "d":
            for kk in [("c", a, b), ("c", b, a)]:
                if kk in vmap:
                    factors.append(("cd", [vmap[kk], j]))
    return np.array(var), factors, ent_x, xtype


# ---------- 重みがルールごとに違う網 ----------
class WNet(e41.Net):
    def bp(self, u, Wf, damp=0.5, maxit=500, tol=1e-6, frozen=None):
        fi, si, vi, bi = self.fi, self.si, self.vi, self.bi
        r = np.zeros(len(vi)); c = 1.0 - np.exp(-Wf[fi]); conv = False
        fz = np.zeros(len(vi), bool) if frozen is None else frozen[vi]
        for it in range(maxit):
            tot = u.copy(); np.add.at(tot, vi, r)
            q = sig(tot[vi] - r)
            qb = np.where(bi == 1, q, 1 - q)
            lq = np.log(np.clip(qb, 1e-300, 1))
            S = np.zeros((self.F, 3)); S[fi, si] = lq
            P = np.exp(S.sum(1)[fi] - lq)
            m = np.log(np.clip(1 - c * P, 1e-300, 1))
            new = np.where(bi == 1, m, -m); new[fz] = 0.0
            rn = damp * r + (1 - damp) * new
            d = np.max(np.abs(rn - r)); r = rn
            if d < tol:
                conv = True; break
        tot = u.copy(); np.add.at(tot, vi, r)
        return sig(tot), conv, it + 1

    def signed_counts(self, a, types, T):
        """状態aで、主張iを本当にすると破れる(−)・うそにすると破れる(+)の数を、ルールの種類ごとに"""
        match = (a[self.vi] == self.bi)
        cnt = np.zeros(self.F); np.add.at(cnt, self.fi, match)
        others = (cnt[self.fi] - match) == (self.ar[self.fi] - 1)
        sgn = np.where(self.bi == 1, -1.0, 1.0) * others
        out = np.zeros((self.n, len(T)))
        for k, t in enumerate(T):
            m = types[self.fi] == t
            np.add.at(out[:, k], self.vi[m], sgn[m])
        return out

    def energy(self, X, u, Wf):
        m = (X[:, self.var] == self.bad) | ~self.mask[None]
        viol = m.all(2)
        return -(X @ u) + viol.astype(float) @ Wf

    def pt(self, u, Wf, rng, sweeps=SWEEPS, betas=BETAS):
        groups = self.coloring()
        prep = []
        for g in groups:
            fs = np.unique(self.fi[np.isin(self.vi, g)])
            gi = np.isin(self.var[fs], g) & self.mask[fs]
            ff, ss = np.nonzero(gi)
            prep.append((g, fs, ff, ss))
        K = len(betas); B = betas.copy()
        X = rng.integers(0, 2, size=(K, self.n)).astype(np.int8)
        burn = sweeps // 10; acc = np.zeros(self.n); swaps = 0
        for t in range(sweeps):
            for g, fs, ff, ss in prep:
                m = (X[:, self.var[fs]] == self.bad[fs]) & self.mask[fs]
                cnt = m.sum(2)
                others = (cnt[:, ff] - m[:, ff, ss]) == (self.ar[fs][ff] - 1)
                b = self.bad[fs][ff, ss]
                contrib = np.where(b == 1, -1.0, 1.0) * Wf[fs][ff] * others
                field = np.zeros((K, self.n)); np.add.at(field.T, self.var[fs][ff, ss], contrib.T)
                loc = u[g] + field[:, g]
                X[:, g] = (rng.random((K, len(g))) < sig(B[:, None] * loc)).astype(np.int8)
            E = self.energy(X, u, Wf)
            for k in range(K - 1):                       # 隣の温度と入れ替え
                if rng.random() < np.exp(min(0.0, (B[k] - B[k + 1]) * (E[k] - E[k + 1]))):
                    X[[k, k + 1]] = X[[k + 1, k]]; E[[k, k + 1]] = E[[k + 1, k]]; swaps += 1
            if t >= burn:
                acc += X[0]
        return acc / (sweeps - burn), swaps / (sweeps * (K - 1))


def small_copy(factors, ent_x):
    """物ごとの小さな網（分類どうしの主張を物ごとに写す）。戻り値: ルール一覧, 写し→元, 物の主張→写し"""
    ent_of = {i: w for w, ids in ent_x.items() for i in ids}
    new_id, src, facs = {}, [], []
    def N(w, i):
        if (w, i) not in new_id:
            new_id[(w, i)] = len(src); src.append(i)
        return new_id[(w, i)]
    by_w = {}
    for t, v in factors:
        if t in ("trans", "disj"):
            w = ent_of[v[0]]
            facs.append((t, [N(w, i) for i in v]))
            for i in v:
                by_w.setdefault(w, set()).add(i)
    for t, v in factors:
        if t in ("anti", "cd"):
            for w, s_ in by_w.items():
                if all(i in s_ for i in v):
                    facs.append((t, [N(w, i) for i in v]))
    for i, w in ent_of.items():
        N(w, i)
    return facs, np.array(src), {i: new_id[(ent_of[i], i)] for i in ent_of}


def main():
    D = json.load(open(sys.argv[1]))
    allc = D["claims"]; Y = np.array([c["gold"] for c in allc])
    if sys.argv[2] == "MOCK":
        r0 = np.random.default_rng(123)
        L = 2.0 * (2 * Y - 1) - 1.0 + r0.normal(0, 2.0, len(Y))
    else:
        A = json.load(open(sys.argv[2]))
        if A["stmt"] != [c["stmt"] for c in allc]:
            raise SystemExit("答えと問いが合わない")
        L = np.array(A["logit"])
    var, factors, ent_x, xtype = build(D, L)
    n = len(var); logit = L[var]; y = Y[var]
    kind = np.array([allc[i]["kind"] for i in var])
    T = ["trans", "disj", "anti", "cd"]
    big = WNet(n, factors)
    ftypes = np.array([t for t, _ in factors])
    sm_facs, src, xmap = small_copy(factors, ent_x)
    smnet = WNet(len(src), sm_facs)
    sm_ftypes = np.array([t for t, _ in sm_facs])
    ents = list(ent_x)
    a = (logit > 0).astype(np.int8)
    SC = big.signed_counts(a, ftypes, T)             # 主張×ルール種類
    # E4-1と同じ特徴
    feat = np.zeros((n, 6)); feat[:, 0] = logit
    v1, v0 = big.counts(a); feat[:, 1] = v1; feat[:, 2] = v0
    for w in ents:
        ids = np.array(ent_x[w])
        for i in ids:
            o = logit[ids[ids != i]]; feat[i, 3] = o.max(); feat[i, 4] = o.mean()
    cs = np.zeros(n); cc = np.zeros(n)
    for t, v in factors:
        if t == "trans":
            cs[v[0]] += logit[v[1]]; cc[v[0]] += 1
    feat[:, 5] = cs / np.maximum(cc, 1)
    feat3 = np.hstack([feat, SC[:, :2]])
    t0 = time.time(); out = dict(splits=[])
    print("網", n, "主張", len(factors), "ルール", time.time() - t0, flush=True)
    grid = {}
    for sp in SPLITS:
        r = np.random.default_rng(1000 + sp)
        perm = r.permutation(len(ents)); tune = set(ents[i] for i in perm[: len(ents) // 5])
        xt = np.array([i for w in ents if w in tune for i in ent_x[w]])
        xe = np.array([i for w in ents if w not in tune for i in ent_x[w]])
        res = {}
        res["1_conf"] = selective(sig(logit[xe]), y[xe])
        lr2 = LogisticRegression(max_iter=2000).fit(logit[xt, None], y[xt])
        p2 = lr2.predict_proba(logit[:, None])[:, 1]
        res["2_conf_cal"] = selective(p2[xe], y[xe])
        lr3 = LogisticRegression(max_iter=2000).fit(feat3[xt], y[xt])
        res["3_learn"] = selective(lr3.predict_proba(feat3[xe])[:, 1], y[xe])
        # 重みの学習
        Xw = np.hstack([logit[:, None], SC[:, :2]])
        lrw = LogisticRegression(max_iter=2000, C=1e4).fit(Xw[xt], y[xt])
        lam, Wt, Wd = lrw.coef_[0]; b = lrw.intercept_[0]
        Wt, Wd = max(Wt, 0.0), max(Wd, 0.0); Wo = (Wt + Wd) / 2
        Wmap = {"trans": Wt, "disj": Wd, "anti": Wo, "cd": Wo}
        Wf = np.array([Wmap[t] for t in ftypes])
        u = lam * logit + b
        res["weights"] = dict(lam=float(lam), b=float(b), trans=float(Wt), disj=float(Wd))
        p5, conv, its = big.bp(u, Wf)
        res["5_big"] = selective(p5[xe], y[xe]); res["5_big"]["converged"] = conv
        sm_W = np.array([Wmap[t] for t in sm_ftypes])
        ps, cs_, _ = smnet.bp(u[src], sm_W)
        res["4_small"] = selective(np.array([ps[xmap[i]] for i in xe]), y[xe]); res["4_small"]["converged"] = cs_
        frozen = kind != "x"
        p5c, convc, _ = big.bp(u, Wf, frozen=frozen)
        res["5c_cautious"] = selective(p5c[xe], y[xe]); res["5c_cautious"]["converged"] = convc
        for k in ["c", "d"]:
            m = kind == k
            res[f"cat_{k}_conf_cal"] = selective(p2[m], y[m])
            res[f"cat_{k}_5_big"] = selective(p5[m], y[m])
            res[f"cat_{k}_5c"] = selective(sig(u)[m], y[m])
        p6, swap = big.pt(u, Wf, np.random.default_rng(7 + sp))
        res["6_pbit_pt"] = selective(p6[xe], y[xe]); res["6_pbit_pt"]["swap_rate"] = swap
        res["6_minus_5_mean_abs"] = float(np.mean(np.abs(p6 - p5)))
        # ⑦ 参考：1つの強さを格子で（ずらし付き）
        best = None
        for lm in LAMS:
            for of in OFFS:
                for W in WS:
                    key = (lm, of, W)
                    if key not in grid:
                        grid[key] = big.bp(lm * (logit + of), np.full(len(factors), W), maxit=300)[0]
                    s = selective(grid[key][xt], y[xt])["aurc"]
                    if best is None or s < best[0]:
                        best = (s, key)
        res["7_grid"] = selective(grid[best[1]][xe], y[xe]); res["7_grid"]["key"] = best[1]
        # うその種類ごとの正しさ（全部答えた時）
        for name, p in [("1_conf", sig(logit)), ("2_conf_cal", p2), ("5_big", p5), ("5c_cautious", p5c), ("6_pbit_pt", p6)]:
            res[f"bytype_{name}"] = {tt: float(np.mean((p[xe][[xtype[i] == tt for i in xe]] >= 0.5) == y[xe][[xtype[i] == tt for i in xe]]))
                                     for tt in ["T", "G", "H", "E"]}
        out["splits"].append(res)
        print("split", sp, {k: round(v["aurc"], 4) for k, v in res.items() if isinstance(v, dict) and "aurc" in v},
              res["weights"], round(time.time() - t0), flush=True)
    out["meta"] = dict(n=n, factors=len(factors), seconds=time.time() - t0,
                       xtype_counts={tt: int(sum(1 for v in xtype.values() if v == tt)) for tt in "TGHE"})
    json.dump(out, open(sys.argv[3], "w"), indent=1)


if __name__ == "__main__":
    main()
