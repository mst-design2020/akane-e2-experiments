"""E4-0: 作り物の網で、BP と p-bit方式（揺らして数える）が分かれる所を探す。
計画: PLAN_E4-0.md（固定の節）。
使い方: python3 e40.py [--quick]
"""
import json, sys, time, itertools
import numpy as np
from multiprocessing import Pool
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

M = 0.6                 # AIの自信の当たり方（全部答えると約73%）
DEGREES = [2, 3, 4, 6]
EPS = [0.01, 0.03, 0.10, 0.30]
TRUST = [1.0, 2.0]      # 正しい強さ / 信じすぎ
SEEDS = [0, 1, 2, 3, 4]
N_BIG = 2000
N_SMALL = 20
N_SMALL_GRAPHS = 20
SWEEPS = [100, 1000, 10000]
COVER = [1.0, 0.9, 0.7, 0.5]


# ---------- 網を作る ----------
def regular_graph(n, d, rng):
    """どの主張も線がd本の、くじで作る網（自己ループ・二重線なし）。"""
    for _ in range(200):
        stubs = np.repeat(np.arange(n), d)
        rng.shuffle(stubs)
        e = stubs.reshape(-1, 2)
        e = np.sort(e, axis=1)
        # 悪い線を入れ替えで直す
        for _ in range(100000):
            key = e[:, 0] * n + e[:, 1]
            _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
            bad = np.where((e[:, 0] == e[:, 1]) | (cnt[inv] > 1))[0]
            if len(bad) == 0:
                return e
            a = bad[0]
            b = rng.integers(len(e))
            if b == a:
                continue
            (u, v), (x, y) = e[a], e[b]
            if rng.random() < 0.5:
                na, nb = sorted((u, x)), sorted((v, y))
            else:
                na, nb = sorted((u, y)), sorted((v, x))
            e[a], e[b] = na, nb
    raise RuntimeError("graph")


def make_instance(n, d, eps, trust, rng):
    y = rng.choice([-1, 1], size=n)
    h = rng.normal(M * y, 1.0)
    f = M * h                                   # 主張の場（正しく割り引いた強さ）
    e = regular_graph(n, d, rng)
    r = y[e[:, 0]] * y[e[:, 1]]                 # 正しいルールの向き
    flip = rng.random(len(e)) < eps
    r = np.where(flip, -r, r)
    jstar = 0.5 * np.log((1 - eps) / eps)
    J = trust * jstar * r
    return dict(y=y, f=f, e=e, r=r, J=J, jstar=jstar, trust=trust, n=n)


def coloring(n, e):
    nbr = [[] for _ in range(n)]
    for a, b in e:
        nbr[a].append(b); nbr[b].append(a)
    col = -np.ones(n, dtype=int)
    for i in np.argsort([-len(x) for x in nbr], kind="stable"):
        used = {col[j] for j in nbr[i]}
        c = 0
        while c in used:
            c += 1
        col[i] = c
    return col


# ---------- 方法 ----------
def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def m1_conf(I):
    return sig(2 * I["f"])


def m2_count(I):
    n, e, J, f = I["n"], I["e"], I["J"], I["f"]
    a = np.sign(f)
    v = np.zeros(n)
    np.add.at(v, e[:, 0], J * a[e[:, 1]])
    np.add.at(v, e[:, 1], J * a[e[:, 0]])
    return sig(2 * (f + v))


def features(I):
    n, e, r, f = I["n"], I["e"], I["r"], I["f"]
    a = np.sign(f)
    vs = np.zeros(n); vt = np.zeros(n); br = np.zeros(n); deg = np.zeros(n)
    for (p, q) in [(0, 1), (1, 0)]:
        np.add.at(vs, e[:, p], r * a[e[:, q]])
        np.add.at(vt, e[:, p], r * np.tanh(f[e[:, q]]))
        np.add.at(br, e[:, p], (r * a[e[:, p]] * a[e[:, q]] < 0).astype(float))
        np.add.at(deg, e[:, p], 1.0)
    return np.stack([f, vs, vt, br, deg], 1)


def m3_learn(I, rng):
    X = features(I); t = (I["y"] > 0).astype(int)
    idx = rng.permutation(I["n"]); half = I["n"] // 2
    p = np.zeros(I["n"])
    for tr, te in [(idx[:half], idx[half:]), (idx[half:], idx[:half])]:
        clf = LogisticRegression(max_iter=1000).fit(X[tr], t[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p


def m4_bp(I, damp=0.5, maxit=500, tol=1e-6):
    n, e, J, f = I["n"], I["e"], I["J"], I["f"]
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    Jd = np.concatenate([J, J])
    m = len(e)
    rev = np.concatenate([np.arange(m, 2 * m), np.arange(m)])
    u = np.zeros(2 * m)
    tJ = np.tanh(Jd)
    conv = False
    for it in range(maxit):
        H = f.copy(); np.add.at(H, dst, u)
        cav = H[src] - u[rev]
        new = np.arctanh(np.clip(tJ * np.tanh(cav), -1 + 1e-15, 1 - 1e-15))
        un = damp * u + (1 - damp) * new
        diff = np.max(np.abs(un - u))
        u = un
        if diff < tol:
            conv = True
            break
    H = f.copy(); np.add.at(H, dst, u)
    return sig(2 * H), conv, it + 1


def _gibbs_setup(I):
    n, e, J = I["n"], I["e"], I["J"]
    W = sp.coo_matrix((np.concatenate([J, J]), (np.concatenate([e[:, 0], e[:, 1]]),
                       np.concatenate([e[:, 1], e[:, 0]]))), shape=(n, n)).tocsr()
    col = coloring(n, e)
    groups = [np.where(col == c)[0] for c in range(col.max() + 1)]
    rows = [W[g] for g in groups]
    return groups, rows


def m5_pbit(I, sweeps, rng, setup=None, beta=1.0):
    groups, rows = setup or _gibbs_setup(I)
    f = I["f"]
    s = rng.choice([-1.0, 1.0], size=I["n"])
    burn = sweeps // 10
    acc = np.zeros(I["n"])
    for t in range(sweeps):
        for g, Wg in zip(groups, rows):
            loc = f[g] + Wg @ s
            s[g] = np.where(rng.random(len(g)) < sig(2 * beta * loc), 1.0, -1.0)
        if t >= burn:
            acc += s
    return 0.5 * (1 + acc / (sweeps - burn))


def m6_anneal(I, rng, setup=None, sweeps=1000):
    groups, rows = setup or _gibbs_setup(I)
    f = I["f"]
    s = rng.choice([-1.0, 1.0], size=I["n"])
    for beta in np.geomspace(0.1, 3.0, sweeps):
        for g, Wg in zip(groups, rows):
            loc = f[g] + Wg @ s
            s[g] = np.where(rng.random(len(g)) < sig(2 * beta * loc), 1.0, -1.0)
    return s


def exact(I):
    n = I["n"]
    S = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.int8)
    E = S @ I["f"]
    for (a, b), j in zip(I["e"], I["J"]):
        E = E + j * (S[:, a] * S[:, b])
    w = np.exp(E - E.max())
    w /= w.sum()
    return 0.5 * (1 + w @ S)


# ---------- 物差し ----------
def selective(p, y):
    ans = np.where(p >= 0.5, 1, -1)
    ok = (ans == y).astype(float)
    conf = np.abs(p - 0.5)
    order = np.argsort(-conf, kind="stable")
    okc = ok[order]
    cum = np.cumsum(okc) / np.arange(1, len(okc) + 1)
    aurc = float(np.mean(1 - cum))
    accs = {str(c): float(cum[max(1, int(round(c * len(okc)))) - 1]) for c in COVER}
    return dict(aurc=aurc, acc=accs)


# ---------- 1つの条件・くじ ----------
def run_big(args):
    d, eps, trust, seed = args
    rng = np.random.default_rng([seed, d, int(eps * 1000), int(trust * 10), 1])
    I = make_instance(N_BIG, d, eps, trust, rng)
    y = I["y"]
    out = {}
    out["1_conf"] = selective(m1_conf(I), y)
    out["2_count"] = selective(m2_count(I), y)
    out["3_learn"] = selective(m3_learn(I, rng), y)
    pbp, conv, its = m4_bp(I)
    out["4_bp"] = selective(pbp, y); out["4_bp"]["converged"] = conv; out["4_bp"]["iters"] = its
    setup = _gibbs_setup(I)
    pg = {}
    for sw in SWEEPS:
        pg[sw] = m5_pbit(I, sw, rng, setup)
        out[f"5_pbit_{sw}"] = selective(pg[sw], y)
        out[f"5_pbit_{sw}"]["mean_abs_diff_vs_bp"] = float(np.mean(np.abs(pg[sw] - pbp)))
    s6 = m6_anneal(I, rng, setup)
    out["6_anneal_acc100"] = float(np.mean(s6 == y))
    out["n_colors"] = len(setup[0])
    return dict(d=d, eps=eps, trust=trust, seed=seed, res=out)


def run_small(args):
    d, eps, trust, seed = args
    rng = np.random.default_rng([seed, d, int(eps * 1000), int(trust * 10), 2])
    errs = {"4_bp": [], **{f"5_pbit_{sw}": [] for sw in SWEEPS}}
    conv = []
    for g in range(N_SMALL_GRAPHS):
        I = make_instance(N_SMALL, d, eps, trust, rng)
        pe = exact(I)
        pbp, c, _ = m4_bp(I); conv.append(c)
        errs["4_bp"].append(float(np.mean(np.abs(pbp - pe))))
        setup = _gibbs_setup(I)
        for sw in SWEEPS:
            errs[f"5_pbit_{sw}"].append(float(np.mean(np.abs(m5_pbit(I, sw, rng, setup) - pe))))
    return dict(d=d, eps=eps, trust=trust, seed=seed,
                mae_vs_exact={k: float(np.mean(v)) for k, v in errs.items()},
                bp_conv_rate=float(np.mean(conv)))


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    global_conds = list(itertools.product(DEGREES, EPS, TRUST, SEEDS))
    if quick:
        N_BIG = 400; SWEEPS[:] = [100, 1000]; N_SMALL_GRAPHS = 3
        global_conds = [(d, e, t, 0) for d in [2, 6] for e in [0.01, 0.3] for t in TRUST]
    t0 = time.time()
    with Pool(4) as pool:
        small = pool.map(run_small, global_conds)
        print("small done", time.time() - t0, flush=True)
        big = pool.map(run_big, global_conds)
    print("big done", time.time() - t0, flush=True)
    name = "E40_quick.json" if quick else "E40_results.json"
    json.dump(dict(big=big, small=small, seconds=time.time() - t0), open(name, "w"), indent=1)
    print("saved", name)
