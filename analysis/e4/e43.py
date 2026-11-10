"""E4-3：正解なしで網は効くか。計画: PLAN_E4-3.md（固定）。
使い方: python3 e43.py e42_data.json <答えのjson|MOCK> <出力json>
正解(gold)は評価(selective)と、推し量った「本当の割合」との比べだけに使う。
"""
import json, sys, time
import numpy as np
import e42
from e41 import sig, selective

NB, MAXIT, TOL = 10, 50, 1e-4
T0 = time.time()
SPLITS = [0, 1, 2, 3, 4]


def bins_of(logit, kind):
    """主張の種類ごとに、logitを10段（同じ数ずつ）に分ける"""
    b = np.zeros(len(logit), int)
    for k in np.unique(kind):
        m = kind == k
        edges = np.quantile(logit[m], np.linspace(0, 1, NB + 1)[1:-1])
        b[m] = np.searchsorted(edges, logit[m], side="right")
    return b


def mstep(q, b, kind, prior=None):
    """周辺確率 q から、種類ごとの 本当の割合π と 各段の出る割合 e1,e0 を推し量り、主張ごとの一つ目の値 u を返す。
    prior(h)：Gemmaの答えを見ない網だけの周辺確率。網があると、π＝qの平均 をそのまま足すと網の押しを二重に数えて
    暴走する（作り物で確認）ので、網ありの時は「網だけの模型の種類ごとの平均＝qの平均」になる偏り h を合わせる。"""
    u = np.zeros(len(q)); est = {}
    ks = list(np.unique(kind)); h = np.zeros(len(q))
    for k in ks:
        m = kind == k
        qk, bk = q[m], b[m]
        e1 = np.bincount(bk, qk, NB) + 1e-3; e0 = np.bincount(bk, 1 - qk, NB) + 1e-3
        e1 /= e1.sum(); e0 /= e0.sum(); pi = float(np.clip(qk.mean(), 1e-4, 1 - 1e-4))
        u[m] = np.log(e1[bk] / e0[bk]); h[m] = np.log(pi / (1 - pi))
        est[k] = pi
    if prior is not None:
        if getattr(prior, "h", None) is not None:          # 前回の偏りから始める（速くするだけ）
            for k in ks:
                m = kind == k; h[m] = prior.h[m]
        lo = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
        for _ in range(8):                               # 前回から続けるので少しずつで足りる
            pm = prior(h)
            step = 0.0
            for k in ks:
                m = kind == k
                d = lo(est[k]) - lo(pm[m].mean()); h[m] += d; step = max(step, abs(d))
            if step < 1e-2:
                break
        prior.h = h.copy()
    return u + h, est


class Prior:
    def __init__(self, f):
        self.f, self.h = f, None
    def __call__(self, h):
        return self.f(h)


def em(b, kind, q0, infer, prior=None):
    q = q0.copy(); hist = []
    for it in range(MAXIT):
        u, est = mstep(q, b, kind, prior)
        qn = infer(u)
        d = float(np.max(np.abs(qn - q))); q = qn; hist.append(d)
        print("  em", it, round(time.time() - T0), round(d, 5), {k: round(v, 3) for k, v in est.items()}, flush=True)
        if d < TOL:
            break
    u, est = mstep(q, b, kind, prior)
    return q, u, est, dict(iters=it + 1, last_change=hist[-1])


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
    var, factors, ent_x, xtype = e42.build(D, L)
    n = len(var); logit = L[var]; y = Y[var]
    kind = np.array([allc[i]["kind"] for i in var])
    big = e42.WNet(n, factors)
    sm_facs, src, xmap = e42.small_copy(factors, ent_x)
    smnet = e42.WNet(len(src), sm_facs)
    ents = list(ent_x)
    t0 = time.time(); out = dict(splits=[], meta={})
    b = bins_of(logit, kind)
    q0 = sig(logit)                                   # Gemmaの答えをそのまま信じる所から
    truth = {k: float(y[kind == k].mean()) for k in np.unique(kind)}
    P, info = {}, {}
    # ② 矛盾を数える（W=1, λ=1）
    a = (logit > 0).astype(np.int8)
    P["2_count"] = sig(logit + big.local(a[None, :], 1.0)[0])
    # ③ 網なしの癖の推し量り
    P["3_em_nonet"], u3, est3, it3 = em(b, kind, q0, sig)
    info["3_em_nonet"] = dict(pi=est3, **it3)
    # ④ 大きな網（W=2が主、1・4は参考）
    for W in [2.0, 1.0, 4.0]:
        Wf = np.full(len(factors), W)
        conv = {}
        def inf(u):
            p, c, _ = big.bp(u, Wf); conv["c"] = c; return p
        name = f"4_em_big_W{W:g}"
        P[name], _, est, it = em(b, kind, q0, inf, prior=Prior(lambda h, Wf=Wf: big.bp(h, Wf, maxit=200)[0]))
        info[name] = dict(pi=est, converged=conv["c"], **it)
        print(name, info[name], round(time.time() - t0), flush=True)
    # ⑤ 慎重な網：分類どうしの主張は③の値に固定、物の主張だけEM
    Wf = np.full(len(factors), 2.0); frozen = kind != "x"; conv = {}
    def inf5(u):
        uu = np.where(frozen, u3, u)
        p, c, _ = big.bp(uu, Wf, frozen=frozen); conv["c"] = c; return p
    def prior5(h):
        return big.bp(np.where(frozen, u3, h), Wf, frozen=frozen, maxit=200)[0]
    P["5_em_cautious"], _, est5, it5 = em(b, kind, q0, inf5, prior=Prior(prior5))
    info["5_em_cautious"] = dict(pi=est5, converged=conv["c"], **it5)
    print("5", info["5_em_cautious"], round(time.time() - t0), flush=True)
    # ⑥ 小さな網：写しの周辺確率を元の主張ごとに平均して M-step
    smW = np.full(len(sm_facs), 2.0); conv = {}
    cnt = np.bincount(src, minlength=n)
    def inf6(u):
        ps, c, _ = smnet.bp(u[src], smW); conv["c"] = c
        return np.bincount(src, ps, n) / np.maximum(cnt, 1)
    def prior6(h):
        ps_, _, _ = smnet.bp(h[src], smW, maxit=200); return np.bincount(src, ps_, n) / np.maximum(cnt, 1)
    _, u6, est6, it6 = em(b, kind, q0, inf6, prior=Prior(prior6))
    ps, _, _ = smnet.bp(u6[src], smW)
    p6 = np.bincount(src, ps, n) / np.maximum(cnt, 1)
    for i in xmap:
        p6[i] = ps[xmap[i]]
    P["6_em_small"] = p6
    info["6_em_small"] = dict(pi=est6, converged=conv["c"], **it6)
    print("6", info["6_em_small"], round(time.time() - t0), flush=True)
    P["1_conf"] = sig(logit)
    for sp in SPLITS:
        r = np.random.default_rng(1000 + sp)
        perm = r.permutation(len(ents)); tune = set(ents[i] for i in perm[: len(ents) // 5])
        xe = np.array([i for w in ents if w not in tune for i in ent_x[w]])
        res = {k: selective(p[xe], y[xe]) for k, p in P.items()}
        # ⑦ Batch Calibration：評価する物の主張の束で「はい／いいえ」の確率の平均をとり、その比の対数を引く
        pe = sig(logit[xe]); shift = np.log(pe.mean() / (1 - pe).mean())
        res["7_batchcal"] = selective(sig(logit[xe] - shift), y[xe]); res["7_batchcal"]["shift"] = float(shift)
        for name in ["1_conf", "3_em_nonet", "4_em_big_W2", "5_em_cautious"]:
            p = P[name]
            res[f"bytype_{name}"] = {tt: float(np.mean((p[xe][[xtype[i] == tt for i in xe]] >= 0.5) == y[xe][[xtype[i] == tt for i in xe]]))
                                     for tt in ["T", "G", "H", "E"]}
        p7 = sig(logit - shift)
        res["bytype_7_batchcal"] = {tt: float(np.mean((p7[xe][[xtype[i] == tt for i in xe]] >= 0.5) == y[xe][[xtype[i] == tt for i in xe]]))
                                    for tt in ["T", "G", "H", "E"]}
        out["splits"].append(res)
        print("split", sp, {k: round(v["aurc"], 4) for k, v in res.items() if "aurc" in v}, flush=True)
    out["meta"] = dict(n=n, factors=len(factors), seconds=time.time() - t0, true_rate=truth, em=info)
    json.dump(out, open(sys.argv[3], "w"), indent=1)


if __name__ == "__main__":
    main()
