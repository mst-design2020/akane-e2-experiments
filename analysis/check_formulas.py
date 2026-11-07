# Checks that the code computes the model written in README ("How the computation works").
# On small networks it enumerates every assignment of the README formula and compares the exact
# answer with the repository's BP (bpw), p-bit simulation (pbitw) and MaxSAT (maxsat26).
# Usage: cd analysis; python check_formulas.py      (about 1 minute, CPU only)
import ast, itertools, os
import numpy as np
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
HERE = os.path.dirname(os.path.abspath(__file__)); g = globals()
def grab(path, names):
    for node in ast.parse(open(path).read()).body:
        if (isinstance(node, ast.FunctionDef) and node.name in names) or \
           (isinstance(node, ast.Assign) and any(getattr(t, "id", None) in names for t in node.targets)):
            exec(compile(ast.Module([node], []), path, "exec"), g)
grab(os.path.join(HERE, "base.py"), {"A8", "VIOL", "coloring"})
grab(os.path.join(HERE, "e24.py"), {"bpw", "pbitw"})
grab(os.path.join(HERE, "e26.py"), {"maxsat26"})

def log_weight(x, u, ell, zeta, lam, W, r, C, I, E):
    """log of the unnormalized probability, written exactly as in README."""
    s = lam * (ell @ x) - lam * r * (zeta @ u)
    s -= W * sum(x[a] * x[b] * (1 - x[c]) for a, b, c in C)                         # chaining rules
    s -= W * sum(x[a] * x[b] * (1 - u[k]) for k, pr in enumerate(I) for a, b in pr)  # doubted exclusions
    s -= W * sum(x[a] * x[b] for a, b in E)                                          # trusted exclusions
    return s

def exact(ell, zeta, lam, W, r, C, I, E):
    n, K = len(ell), len(zeta); Z = 0.0; m = np.zeros(n + K); best = (-np.inf, None)
    for bits in itertools.product([0, 1], repeat=n + K):
        v = np.array(bits); s = log_weight(v[:n], v[n:], ell, zeta, lam, W, r, C, I, E)
        Z += np.exp(s); m += np.exp(s) * v
        if s > best[0]: best = (s, v)
    return m / Z, best[0]

def to_repo(ell, zeta, r, C, I, E):
    """Repository layout: n statements, then DUMMY (always 0), then one node per label pair."""
    global DUMMY
    n = len(ell); DUMMY = n; u0 = n + 1
    lg = np.r_[ell, -60.0, r * (-zeta)]
    rules = [tuple(c) for c in C] + [(min(a, b), max(a, b), DUMMY) for a, b in E] \
          + [(min(a, b), max(a, b), u0 + k) for k, pr in enumerate(I) for a, b in pr]
    return lg, rules

NETS = {  # (C, I, E) on 7 statements and 2 label pairs
    "tree":  ([(0, 1, 2)], [[(2, 3)], [(4, 5)]], [(5, 6)]),
    "loopy": ([(0, 1, 2), (2, 3, 4)], [[(0, 5), (1, 5), (4, 6)], [(3, 6), (0, 6)]], [(1, 3)]),
}
rng = np.random.default_rng(0); worst = {}
for trial in range(40):
    name = "tree" if trial % 2 == 0 else "loopy"; C, I, E = NETS[name]
    n, K = 7, 2
    ell, zeta = rng.normal(0, 1.5, n), rng.normal(0, 1.5, K)
    lam, W, r = rng.choice([0.5, 1, 2]), rng.choice([1, 2, 4]), rng.choice([0.5, 1, 2])
    ex, best = exact(ell, zeta, lam, W, r, C, I, E)
    lg, rules = to_repo(ell, zeta, r, C, I, E)
    drop = lambda p: np.r_[p[:n], p[n + 1:]]
    pb, _ = bpw(lg, rules, lam, W, iters=2000, tol=1e-10)
    pp = pbitw(lg, rules, lam, W, np.random.default_rng(trial), coloring(len(lg), rules), chains=200, burn=200, sweeps=2000)
    ms = drop(maxsat26(lg, rules, lam, W))
    for k, v in {f"BP ({name})": np.abs(drop(pb) - ex).max(), "p-bit simulation": np.abs(drop(pp) - ex).max(),
                 "MaxSAT (log-weight gap to the best assignment)": best - log_weight(ms[:n], ms[n:], ell, zeta, lam, W, r, C, I, E)}.items():
        worst[k] = max(worst.get(k, 0.0), float(v))
print("Largest difference from exact enumeration over 40 random small networks:")
for k, v in worst.items(): print(f"  {k}: {v:.4f}")
print("Expected: BP (tree) 0, p-bit simulation within sampling noise (< 0.01), MaxSAT 0. "
      "BP on networks with loops is an approximation, so a small difference is normal.")
