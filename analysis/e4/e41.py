"""E4-1 網の計算と比べ。計画: PLAN_E4-1.md（固定の節）。
使い方: python3 e41.py e41_data.json <答えのjson|MOCK> <出力json>
答えのjson: {"stmt": [...], "logit": [...]}（Colab E4-1 ノートの出力）
"""
import json, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression

BAD = {"trans": (1, 1, 0), "disj": (1, 1, 1), "anti": (1, 1), "cd": (1, 1)}
LAMS = [0.25, 0.5, 1.0, 2.0]
WS = [0.5, 1.0, 2.0, 4.0, 8.0]
SPLITS = [0, 1, 2, 3, 4]
COVER = [1.0, 0.9, 0.7, 0.5]
import os
CHAINS, SWEEPS = 4, int(os.environ.get('E41_SWEEPS', 10000))


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class Net:
    """2値の主張と「1つの悪い組み合わせだけ罰する」ルールの網。"""
    def __init__(self, n, factors):
        self.n = n
        F = len(factors)
        self.F = F
        self.var = -np.ones((F, 3), dtype=np.int64)
        self.bad = np.zeros((F, 3), dtype=np.int8)
        self.ar = np.zeros(F, dtype=np.int64)
        for k, (t, v) in enumerate(factors):
            b = BAD[t]
            self.var[k, :len(v)] = v
            self.bad[k, :len(v)] = b
            self.ar[k] = len(v)
        self.mask = self.var >= 0
        self.fi, self.si = np.nonzero(self.mask)            # 差し込み口（ルール, 場所）
        self.vi = self.var[self.fi, self.si]
        self.bi = self.bad[self.fi, self.si].astype(float)
        self._col = None

    # ---------- BP ----------
    def bp(self, u, W, damp=0.5, maxit=500, tol=1e-6):
        fi, si, vi, bi = self.fi, self.si, self.vi, self.bi
        r = np.zeros(len(vi))                               # ルール→主張 の対数オッズ
        c = 1.0 - np.exp(-W)
        conv = False
        for it in range(maxit):
            tot = u.copy(); np.add.at(tot, vi, r)
            L = tot[vi] - r                                 # 主張→ルール（そのルール抜き）
            q = sig(L)
            qb = np.where(bi == 1, q, 1 - q)                # 悪い側にいる確率
            logq = np.log(np.clip(qb, 1e-300, 1))
            S = np.zeros((self.F, 3)); S[fi, si] = logq
            P = np.exp(S.sum(1)[fi] - logq)                 # 他の口が全部悪い側にいる確率
            m = np.log(np.clip(1 - c * P, 1e-300, 1))
            new = np.where(bi == 1, m, -m)
            rn = damp * r + (1 - damp) * new
            d = np.max(np.abs(rn - r)) if len(r) else 0
            r = rn
            if d < tol:
                conv = True
                break
        tot = u.copy(); np.add.at(tot, vi, r)
        return sig(tot), conv, it + 1

    # ---------- その主張を 1/0 にした時の場の差（状態 X: 鎖×主張） ----------
    def local(self, X, W):
        fi, si, vi, bi = self.fi, self.si, self.vi, self.bi
        match = (X[:, vi] == bi)                            # 鎖×差し込み口
        cnt = np.zeros((X.shape[0], self.F)); np.add.at(cnt.T, fi, match.T)
        others = (cnt[:, fi] - match) == (self.ar[fi] - 1)  # 他の口が全部悪い側
        contrib = np.where(bi == 1, -W, W) * others
        out = np.zeros((X.shape[0], self.n)); np.add.at(out.T, vi, contrib.T)
        return out

    def counts(self, a):
        """状態aで、主張を本当にすると破れる数・うそにすると破れる数"""
        match = (a[self.vi] == self.bi)
        cnt = np.zeros(self.F); np.add.at(cnt, self.fi, match)
        others = (cnt[self.fi] - match) == (self.ar[self.fi] - 1)
        v1 = np.zeros(self.n); v0 = np.zeros(self.n)
        np.add.at(v1, self.vi, others * (self.bi == 1))
        np.add.at(v0, self.vi, others * (self.bi == 0))
        return v1, v0

    def coloring(self):
        if self._col is None:
            nb = [set() for _ in range(self.n)]
            for row, a in zip(self.var, self.ar):
                vs = row[:a]
                for x in vs:
                    nb[x].update(vs)
            col = -np.ones(self.n, dtype=int)
            for i in np.argsort([-len(s) for s in nb], kind="stable"):
                used = {col[j] for j in nb[i] if j != i}
                k = 0
                while k in used:
                    k += 1
                col[i] = k
            self._col = [np.where(col == k)[0] for k in range(col.max() + 1)]
        return self._col

    # ---------- p-bit方式（鎖を並べて揺らす） ----------
    def pbit(self, u, W, rng, chains=CHAINS, sweeps=SWEEPS, beta=1.0):
        groups = self.coloring()
        # 色ごとに、その色の主張に触れる差し込み口だけで場を計算する
        prep = []
        for g in groups:
            ing = np.isin(self.vi, g)
            fs = np.unique(self.fi[ing])
            prep.append((g, fs))
        X = rng.integers(0, 2, size=(chains, self.n)).astype(np.int8)
        burn = sweeps // 10
        acc = np.zeros((chains, self.n))
        fi, si, vi, bi = self.fi, self.si, self.vi, self.bi
        for t in range(sweeps):
            for g, fs in prep:
                Xf = X[:, self.var[fs]]                         # 鎖×ルール×3
                m = (Xf == self.bad[fs]) & self.mask[fs]
                cnt = m.sum(2)
                gi = np.isin(self.var[fs], g) & self.mask[fs]   # この色の主張の口
                ff, ss = np.nonzero(gi)
                others = (cnt[:, ff] - m[:, ff, ss]) == (self.ar[fs][ff] - 1)
                b = self.bad[fs][ff, ss]
                contrib = np.where(b == 1, -W, W) * others
                field = np.zeros((X.shape[0], self.n)); np.add.at(field.T, self.var[fs][ff, ss], contrib.T)
                loc = u[g] + field[:, g]
                X[:, g] = (rng.random((X.shape[0], len(g))) < sig(beta * loc)).astype(np.int8)
            if t >= burn:
                acc += X
        return (acc / (sweeps - burn)).mean(0)


def selective(p, y):
    ans = p >= 0.5
    ok = (ans == y).astype(float)
    order = np.argsort(-np.abs(p - 0.5), kind="stable")
    cum = np.cumsum(ok[order]) / np.arange(1, len(ok) + 1)
    return dict(aurc=float(np.mean(1 - cum)),
                acc={str(c): float(cum[max(1, int(round(c * len(ok)))) - 1]) for c in COVER})


def small_nets(D):
    """物ごとの小さな網：分類どうしの主張を物ごとに写す。戻り値: Net, 写し→元の主張"""
    claims = D["claims"]
    ent_of_x = {}
    for w, ids in D["ent_claims"].items():
        for i in ids:
            ent_of_x[i] = w
    new_id, src, facs = {}, [], []
    for t, v in D["factors"]:
        if t not in ("trans", "disj"):
            continue
        w = ent_of_x[v[0]]
        nv = []
        for i in v:
            key = (w, i)
            if key not in new_id:
                new_id[key] = len(src); src.append(i)
            nv.append(new_id[key])
        facs.append((t, nv))
    # 分類どうしのルール（anti, cd）も物ごとに写す
    by_w = {}
    for (w, i), j in new_id.items():
        by_w.setdefault(w, {})[i] = j
    for t, v in D["factors"]:
        if t in ("anti", "cd"):
            for w, mp in by_w.items():
                if all(i in mp for i in v):
                    facs.append((t, [mp[i] for i in v]))
    # 物の主張でルールに出てこないもの
    for i, w in ent_of_x.items():
        if (w, i) not in new_id:
            new_id[(w, i)] = len(src); src.append(i)
    xmap = {i: new_id[(ent_of_x[i], i)] for i in ent_of_x}
    return Net(len(src), facs), np.array(src), xmap


def main():
    D = json.load(open(sys.argv[1]))
    claims = D["claims"]; n = len(claims)
    stmt = [c["stmt"] for c in claims]
    y = np.array([c["gold"] for c in claims])
    kind = np.array([c["kind"] for c in claims])
    rng0 = np.random.default_rng(123)
    if sys.argv[2] == "MOCK":
        logit = 2.0 * (2 * y - 1) + rng0.normal(0, 2.0, n)
        flip = rng0.random(n) < 0.15
        logit[flip] = -logit[flip]
    else:
        A = json.load(open(sys.argv[2]))
        if A["stmt"] != stmt:
            raise SystemExit("答えと主張が合わない")
        logit = np.array(A["logit"])
    big = Net(n, D["factors"])
    sm, src, xmap = small_nets(D)
    ents = D["ents"]
    t0 = time.time()

    def u_of(lam):
        return lam * logit

    # BP の結果を (λ,W) ごとにためておく
    bp_big, bp_small = {}, {}
    for lam in LAMS:
        for W in WS:
            bp_big[(lam, W)] = big.bp(u_of(lam), W)
    print("BP grid", time.time() - t0, flush=True)

    xs_all = np.array(sorted(xmap))
    out = dict(splits=[], meta={})
    pbit_cache = {}
    for sp in SPLITS:
        r = np.random.default_rng(1000 + sp)
        perm = r.permutation(len(ents))
        tune_e = set(ents[i] for i in perm[: len(ents) // 5])
        xt = np.array([i for w in ents if w in tune_e for i in D["ent_claims"][w]])
        xe = np.array([i for w in ents if w not in tune_e for i in D["ent_claims"][w]])
        best = min(bp_big, key=lambda k: selective(bp_big[k][0][xt], y[xt])["aurc"])
        lam, W = best
        res = {"lam": lam, "W": W}
        u = u_of(lam)
        p1 = sig(u)
        res["1_conf"] = selective(p1[xe], y[xe])
        a = (logit > 0).astype(np.int8)
        loc = big.local(a[None, :], W)[0]
        res["2_count"] = selective(sig(u + loc)[xe], y[xe])
        # ③ 特徴＋学習
        v1, v0 = big.counts(a)
        feat = np.zeros((n, 6))
        feat[:, 0] = logit; feat[:, 1] = v1; feat[:, 2] = v0
        for w in ents:
            ids = np.array(D["ent_claims"][w])
            for i in ids:
                o = logit[ids[ids != i]]
                feat[i, 3] = o.max(); feat[i, 4] = o.mean()
        csum = np.zeros(n); ccnt = np.zeros(n)
        for t, v in D["factors"]:
            if t == "trans":
                csum[v[0]] += logit[v[1]]; ccnt[v[0]] += 1
        feat[:, 5] = csum / np.maximum(ccnt, 1)
        clf = LogisticRegression(max_iter=2000).fit(feat[xt], y[xt])
        res["3_learn"] = selective(clf.predict_proba(feat[xe])[:, 1], y[xe])
        # ④ 小さな網
        if best not in bp_small:
            bp_small[best] = sm.bp(lam * logit[src], W)
        ps = bp_small[best][0]
        p4 = np.array([ps[xmap[i]] for i in xe])
        res["4_small_bp"] = selective(p4, y[xe]); res["4_small_bp"]["converged"] = bp_small[best][1]
        # ⑤ 大きな網 BP
        p5, conv, its = bp_big[best]
        res["5_big_bp"] = selective(p5[xe], y[xe]); res["5_big_bp"]["converged"] = conv; res["5_big_bp"]["iters"] = its
        res["5_big_bp_c"] = selective(p5[kind == "c"], y[kind == "c"])
        res["5_big_bp_d"] = selective(p5[kind == "d"], y[kind == "d"])
        res["1_conf_c"] = selective(p1[kind == "c"], y[kind == "c"])
        res["1_conf_d"] = selective(p1[kind == "d"], y[kind == "d"])
        # ⑥ p-bit方式
        if best not in pbit_cache:
            pbit_cache[best] = big.pbit(u, W, np.random.default_rng(7))
        p6 = pbit_cache[best]
        res["6_big_pbit"] = selective(p6[xe], y[xe])
        res["6_minus_5_mean_abs"] = float(np.mean(np.abs(p6 - p5)))
        # 信じすぎ（2W）
        p5o, convo, _ = big.bp(u, 2 * W)
        res["5_big_bp_2W"] = selective(p5o[xe], y[xe]); res["5_big_bp_2W"]["converged"] = convo
        if (lam, 2 * W, "o") not in pbit_cache:
            pbit_cache[(lam, 2 * W, "o")] = big.pbit(u, 2 * W, np.random.default_rng(8))
        res["6_big_pbit_2W"] = selective(pbit_cache[(lam, 2 * W, "o")][xe], y[xe])
        out["splits"].append(res)
        print("split", sp, {k: round(v["aurc"], 4) for k, v in res.items() if isinstance(v, dict)}, lam, W, time.time() - t0, flush=True)
    out["meta"] = dict(n=n, factors=len(D["factors"]), colors=len(big.coloring()), seconds=time.time() - t0,
                       conf_acc_x=float(((logit > 0) == y)[kind == "x"].mean()))
    json.dump(out, open(sys.argv[3], "w"), indent=1)


if __name__ == "__main__":
    main()
