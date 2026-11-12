"""E9 の網が README の式どおりかを確かめる。CPU で数秒。
(1) E9-3 の網（e93.py の bp）、(2) E9〜E9-2 の網（analysis/e4/e42.py の WNet.bp、確かめた主張を固定して使う形）。
使い方: python check_formulas_e9.py

小さな網（分類 3〜7、ルール 1〜7）を 200 個くじで作り、e93.py の bp の答えを、
全部の組み合わせを数え上げた正確な確率と比べる。
確かめた主張（固定）は、まわりからの知らせを受け取らず、自分の場だけをルールの相手に送る。
これは「固定した主張を、ルールごとに場だけを持つ写しに分けた網」を数え上げたものと同じになるので、そう数える。
e93.py はデータを読み込むので、bp の関数だけを取り出して使う。
"""
import ast, itertools, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'e93.py'), encoding='utf-8').read()
fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == 'bp'][0]
BP_SRC = ast.get_source_segment(src, fn)
sig = lambda x: 1 / (1 + np.exp(-x))


def exact(U, Wt, Wd, frozen, FA, FB, FT, NC):
    """式：P(z) ∝ exp(Σ u_j z_j) Π_I (1-c_t)^[z_a=1,z_b=0] Π_D (1-c_d)^[z_a=1,z_b=1]、c = 1-exp(-W)"""
    c = 1 - np.exp(-np.where(FT == 0, Wt, Wd))
    free = [j for j in range(NC) if not frozen[j]]
    copies = [(f, s) for f in range(len(FA)) for s, v in (('a', FA[f]), ('b', FB[f])) if frozen[v]]
    num = np.zeros(NC); Z = 0.0
    for zz in itertools.product([0, 1], repeat=len(free) + len(copies)):
        z = dict(zip(free, zz[:len(free)])); cz = dict(zip(copies, zz[len(free):]))
        w = np.exp(sum(U[j] * z[j] for j in free) + sum(U[FA[f] if s == 'a' else FB[f]] * v for (f, s), v in cz.items()))
        for f in range(len(FA)):
            za = cz[(f, 'a')] if frozen[FA[f]] else z[FA[f]]
            zb = cz[(f, 'b')] if frozen[FB[f]] else z[FB[f]]
            if (za == 1 and zb == 0) if FT[f] == 0 else (za == 1 and zb == 1):
                w *= 1 - c[f]
        Z += w
        for j in free:
            num[j] += w * z[j]
    return num / Z, free


def has_loop(FA, FB, frozen, NC):
    """固定した主張を写しに分けたあとの網に、輪があるか"""
    edges = [(a, b) for a, b in zip(FA, FB) if not frozen[a] and not frozen[b]]
    if len({tuple(sorted(e)) for e in edges}) < len(edges):
        return True
    parent = list(range(NC))
    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = root(a), root(b)
        if ra == rb:
            return True
        parent[ra] = rb
    return False


rng = np.random.default_rng(0)
worst = {False: 0.0, True: 0.0}; cnt = {False: 0, True: 0}
for t in range(200):
    NC = int(rng.integers(3, 8))
    pairs = [(a, b) for a in range(NC) for b in range(NC) if a != b]
    nf = int(rng.integers(1, min(8, len(pairs))))
    sel = rng.choice(len(pairs), nf, replace=False)
    FA = np.array([pairs[i][0] for i in sel]); FB = np.array([pairs[i][1] for i in sel]); FT = rng.integers(0, 2, nf)
    IA = np.zeros((nf, NC)); IA[np.arange(nf), FA] = 1
    IB = np.zeros((nf, NC)); IB[np.arange(nf), FB] = 1
    g = dict(np=np, sig=sig, FA=FA, FB=FB, FT=FT, IA=IA, IB=IB, NE=1, NF=nf)
    exec(BP_SRC, g)
    U = rng.normal(0, 1.5, (1, NC)); Wt, Wd = rng.uniform(0.2, 3, 2)
    frozen = np.zeros((1, NC), bool)
    if rng.random() < .7:
        frozen[0, rng.integers(NC)] = True
    p = g['bp'](U, Wt, Wd, frozen, maxit=2000, tol=1e-12)[0]
    ex, free = exact(U[0], Wt, Wd, frozen[0], FA, FB, FT, NC)
    loop = has_loop(FA, FB, frozen[0], NC)
    d = float(np.abs(p[free] - ex[free]).max()) if free else 0.0
    worst[loop] = max(worst[loop], d); cnt[loop] += 1
print("(1) E9-3 の網（e93.py の bp）")
print(f"  木の形 {cnt[False]} 個：最大のずれ {worst[False]:.1e}")
print(f"  輪のある形 {cnt[True]} 個：最大のずれ {worst[True]:.3f}（輪のある確率伝搬は近似）")

# ---- (2) E9〜E9-2：E4-2 の網（WNet）。ルールは「破れる組み合わせ」のとき exp(-W) を掛ける ----
import sys
sys.path.insert(0, os.path.join(HERE, '..', 'e4'))
import e42
from e41 import BAD
ARITY = {"trans": 3, "disj": 3, "anti": 2, "cd": 2}
rng = np.random.default_rng(1)
worst2 = {False: 0.0, True: 0.0}; cnt2 = {False: 0, True: 0}
for t in range(200):
    n = int(rng.integers(3, 8))
    factors = []
    for _ in range(int(rng.integers(1, 5))):
        ty = str(rng.choice(list(ARITY)))
        factors.append((ty, [int(v) for v in rng.choice(n, ARITY[ty], replace=False)]))
    net = e42.WNet(n, factors)
    u = rng.normal(0, 1.5, n); Wf = rng.uniform(0.2, 3, len(factors))
    frozen = rng.random(n) < 0.3
    p = net.bp(u, Wf, maxit=5000, tol=1e-13, frozen=frozen)[0]
    # 数え上げ：固定した主張は、ルールごとに場だけを持つ写しに分ける
    free = [j for j in range(n) if not frozen[j]]
    slots = [(f, k) for f, (ty, v) in enumerate(factors) for k, j in enumerate(v) if frozen[j]]
    num = np.zeros(n); Z = 0.0
    for zz in itertools.product([0, 1], repeat=len(free) + len(slots)):
        z = dict(zip(free, zz[:len(free)])); cz = dict(zip(slots, zz[len(free):]))
        w = np.exp(sum(u[j] * z[j] for j in free) + sum(u[factors[f][1][k]] * v for (f, k), v in cz.items()))
        for f, (ty, v) in enumerate(factors):
            vals = [cz[(f, k)] if frozen[j] else z[j] for k, j in enumerate(v)]
            if tuple(vals) == BAD[ty]:
                w *= np.exp(-Wf[f])
        Z += w
        for j in free:
            num[j] += w * z[j]
    ex = num / Z
    # 輪：写しに分けたあとの、自由な主張とルールの二部グラフに輪があるか
    nodes = {('v', j) for j in free} | {('f', f) for f in range(len(factors))}
    edges = [(('f', f), ('v', j)) for f, (ty, v) in enumerate(factors) for j in v if not frozen[j]]
    parent = {x: x for x in nodes}
    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x
    loop = False
    for a, b in edges:
        ra, rb = root(a), root(b)
        if ra == rb:
            loop = True
        else:
            parent[ra] = rb
    d = float(np.abs(p[free] - ex[free]).max()) if free else 0.0
    worst2[loop] = max(worst2[loop], d); cnt2[loop] += 1
print("(2) E9〜E9-2 の網（e42.WNet.bp、確かめを固定）")
print(f"  木の形 {cnt2[False]} 個：最大のずれ {worst2[False]:.1e}")
print(f"  輪のある形 {cnt2[True]} 個：最大のずれ {worst2[True]:.3f}（輪のある確率伝搬は近似）")
