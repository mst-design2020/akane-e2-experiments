"""E8 のコードが README「E6〜E8 の計算のしかた」の式どおりかを確かめる。
python3 check_formulas_e8.py ../../data/E8_chains_nli.json   （CPU で1〜2分）
1) 一番強い道 s(X)：式どおりに道を全部数え上げて最大を取り、e8_analyze.Item.maxprod と比べる（全部の問い×A〜D）。
2) 網：点が少ない問いで、式の確率分布を全部の組み合わせで数え上げ、Item.bp（輪のあるBP）と比べる。"""
import json, sys, os, itertools, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8_analyze as e8

r = json.load(open(sys.argv[1]))
L, ST, AR, CH = e8.load(r)
old, new = e8.labels()
items = old + new


def paths_score(x):
    """式：s(X) = max over 物の語の意味 k, 道 (X,k)=n0→n1→…→nt (t≤3, 途中の点は C そのものでない) of
    (Π_{i<t} e(n_i)) · a(n_t)。C そのものの点は a=1 で、その先へは進まない。"""
    best = 0.0
    def walk(n, prod, t):
        nonlocal best
        a = x.arr[x.idx[n]]
        best = max(best, prod * a)
        if a >= 1.0 or t == e8.DEPTH:
            return
        e = ST.get(n, (0.0, 0.0))[0]
        for c in x.ch[n]:
            walk(c, prod * e, t + 1)
    for r0 in x.roots:
        walk(r0, 1.0, 0)
    return best


md = 0.0; n = 0
for it in items:
    for m in e8.METHODS:
        x = e8.Item(it, m, *CH[(it["id"], m)], L, ST, AR)
        md = max(md, abs(paths_score(x) - x.maxprod()[0])); n += 1
print(f"1) 一番強い道：{n} 組で、式と maxprod の最大のずれ {md:.2e}")


def exact(x, beta, b, eps, gamma):
    """式：P(z) ∝ exp(z_R·b + Σ_n z_n·u_n) · Π_{(p→c)} (1 − ε·w_pc)^{[z_p=0, z_c=1]}。P(z_R = 1) を数え上げで。"""
    nn = len(x.nodes)
    u = beta * e8.lgt(np.clip(x.arr, 1e-4, 1 - 1e-4)) + b - gamma * x.neg
    u = np.where(x.arr >= 1.0, 30.0, u)
    E = [(nn, x.idx[r0], 1.0) for r0 in x.roots]
    for p, cs in x.ch.items():
        e = ST.get(p, (0.0, 0.0))[0]
        E += [(x.idx[p], x.idx[c], e) for c in cs if c != p]
    Z = np.full(2, -np.inf)
    for z in itertools.product([0, 1], repeat=nn + 1):
        z = np.array(z)
        lw = z[nn] * b + float(z[:nn] @ u) + sum(np.log(1 - eps * w + 1e-12) for a, c, w in E if z[a] == 0 and z[c] == 1)
        Z[z[nn]] = np.logaddexp(Z[z[nn]], lw)
    return float(e8.sig(Z[1] - Z[0]))


rng = np.random.default_rng(0)
tree, loopy = [], []
for it in items:
    for m in "BCD":
        x = e8.Item(it, m, *CH[(it["id"], m)], L, ST, AR)
        if not (2 <= len(x.nodes) <= 13):
            continue
        parents = {r0: 1 for r0 in x.roots}  # 根（物の語）からの段も数える
        for p, cs in x.ch.items():
            for c in cs: parents[c] = parents.get(c, 0) + 1
        is_tree = all(v <= 1 for v in parents.values())
        pr = (float(rng.choice([0.25, 0.5, 1.0])), float(rng.choice([-1.0, 0.0, 1.0])), float(rng.choice([0.8, 0.95])), float(rng.choice([0.0, 2.0])))
        d = abs(exact(x, *pr) - x.bp(*pr, iters=200))
        (tree if is_tree else loopy).append(d)
    if len(tree) >= 60 and len(loopy) >= 20:
        break
print(f"2) 網：木の形 {len(tree)} 問でBPと数え上げの最大のずれ {max(tree):.2e}"
      + (f"、輪のある形 {len(loopy)} 問で {max(loopy):.3f}" if loopy else "（輪のある小さな問いはなし）"))
