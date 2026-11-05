import os, sys; exec(open(os.environ.get("BASE","base.py")).read())
# ---- label pairs ----
PAIRS = collections.Counter()
for name in set(CITIES):
    L = [("K",k) for k in COUNTRIES if ("CC",name,k) in idx] + [("Z",z) for z in CONT if ("CL",name,z) in idx]
    for p,q in itertools.combinations(L,2): PAIRS[tuple(sorted((p,q)))] += 1
for k in COUNTRIES:
    for p,q in itertools.combinations([("Z",z) for z in CONT if ("CK",k,z) in idx],2): PAIRS[tuple(sorted((p,q)))] += 1
PAIRS = sorted(PAIRS)
def compat(p,q):
    if p[0]==q[0]: return False
    k,z = (p,q) if p[0]=="K" else (q,p)
    return cont_of[k[1]]==z[1]
TRUTH = {pq: (not compat(*pq)) for pq in PAIRS}    # True = 両立しない
TRANS = [r for r in RULES if r[2] != DUMMY]
def key_of(lab, name):
    return ("CC",name,lab[1]) if lab[0]=="K" else ("CL",name,lab[1])
def build(excl, wmul=None):
    """excl: dict pair->bool(両立しない). wmul: dict pair->weight multiplier"""
    rules, mult = [tuple(r) for r in TRANS], [1.0]*len(TRANS)
    seen = set(rules)
    for pq, ex in excl.items():
        if not ex: continue
        p,q = pq; w = 1.0 if wmul is None else wmul[pq]
        inst = []
        for name in set(CITIES):
            a, b = key_of(p,name), key_of(q,name)
            if a in idx and b in idx: inst.append((idx[a], idx[b]))
        if p[0]=="Z" and q[0]=="Z":
            for k in COUNTRIES:
                if ("CK",k,p[1]) in idx and ("CK",k,q[1]) in idx: inst.append((idx[("CK",k,p[1])], idx[("CK",k,q[1])]))
        for i,j in inst:
            r = (min(i,j), max(i,j), DUMMY)
            if r not in seen: seen.add(r); rules.append(r); mult.append(w)
    return rules, np.array(mult)

# ---- generalized solvers (W can be per-rule) ----
def pbitw(lg, rules, lam, W, rng, col, chains=10, burn=100, sweeps=400):
    R = np.array(rules); n = len(lg); a, b, h = R[:,0], R[:,1], R[:,2]
    groups = [np.nonzero(col == c)[0] for c in range(col.max()+1)]
    x = (rng.random((chains,n)) < 0.5).astype(float); x[:,DUMMY] = 0; acc = np.zeros(n)
    for t in range(burn+sweeps):
        for g in groups:
            dE = np.tile(-lam*lg, (chains,1))
            xa, xb, xh = x[:,a], x[:,b], x[:,h]
            np.add.at(dE.T, a, (W*xb*(1-xh)).T); np.add.at(dE.T, b, (W*xa*(1-xh)).T); np.add.at(dE.T, h, (-W*xa*xb).T)
            p1 = 1/(1+np.exp(np.clip(dE[:,g],-50,50)))
            x[:,g] = (rng.random((chains,len(g))) < p1).astype(float); x[:,DUMMY] = 0
        if t >= burn: acc += x.mean(0)
    return acc/sweeps
def bpw(lg, rules, lam, W, damp=0.5, iters=200, tol=1e-6):
    R = np.array(rules); n = len(lg); F = len(R)
    un = np.stack([np.zeros(n), lam*lg], 1)
    lt = -np.broadcast_to(np.asarray(W,float),(F,))[:,None]*VIOL[None,:]
    msg = np.zeros((F,3,2)); conv = False
    for it in range(iters):
        tot = un.copy()
        for j in range(3): np.add.at(tot, R[:,j], msg[:,j])
        vin = tot[R] - msg; vin[R == DUMMY] = (0.0, -1e9)
        s = lt + sum(vin[:,j,A8[:,j]] for j in range(3))
        new = np.zeros_like(msg)
        for j in range(3):
            for v in (0,1):
                sel = A8[:,j] == v
                new[:,j,v] = np.logaddexp.reduce(s[:,sel] - vin[:,j,v][:,None], axis=1)
        new -= new.max(2, keepdims=True); new = damp*msg + (1-damp)*new
        d = np.abs(new-msg).max(); msg = new
        if d < tol: conv = True; break
    tot = un.copy()
    for j in range(3): np.add.at(tot, R[:,j], msg[:,j])
    return 1/(1+np.exp(np.clip(tot[:,0]-tot[:,1],-50,50))), conv
def maxsatw(lg, rules, lam, W):
    n = DUMMY; wc = WCNF(); Wv = np.broadcast_to(np.asarray(W,float),(len(rules),))
    for v in range(n):
        wc.append([(v+1) if lg[v] > 0 else -(v+1)], weight=max(1,int(round(100*lam*abs(lg[v])))))
    for (p,q,h),w in zip(rules,Wv):
        wc.append([-(p+1),-(q+1)] + ([h+1] if h != DUMMY else []), weight=max(1,int(round(100*w))))
    with RC2(wc) as r: m = r.compute()
    return np.append((np.array(m[:n]) > 0).astype(int), 0)

ucities = sorted(set(CITIES))
def run_split(split, rules, mult, seeds=(1,2), tag=""):
    col = coloring(DUMMY+1, rules)
    rng = np.random.default_rng(split)
    calc = set(rng.choice(ucities, int(round(0.2*len(ucities))), replace=False))
    cal = np.array([i for i in range(DUMMY) if CITY[i] in calc]); test = np.array([i for i in range(DUMMY) if CITY[i] not in calc])
    cl = LOGIT[cal]; scale = 2.0/cl.std()
    shift = max(np.quantile(cl, np.linspace(0.02,0.98,97)), key=lambda b: ((cl > b) == GOLD[cal]).mean())
    lg = np.append((LOGIT-shift)*scale, -60.0); m = (lg > 0).astype(int); gold = np.append(GOLD, 0)
    best = {}
    for lam, W in itertools.product([0.5,1,2],[1,2,4]):
        acc = {"maxsat": (maxsatw(lg,rules,lam,W*mult)[cal] == gold[cal]).mean(),
               "bp": ((bpw(lg,rules,lam,W*mult)[0][cal] > .5) == gold[cal]).mean(),
               "pbit": ((pbitw(lg,rules,lam,W*mult,np.random.default_rng(0),col)[cal] > .5) == gold[cal]).mean()}
        for k,a in acc.items():
            if k not in best or a > best[k][0]: best[k] = (a,lam,W)
    ms = maxsatw(lg, rules, best["maxsat"][1], best["maxsat"][2]*mult)
    pb, conv = bpw(lg, rules, best["bp"][1], best["bp"][2]*mult)
    vio = violations(m, rules); conf = np.abs(1/(1+np.exp(-lg))-.5); err = (m != gold).astype(int)
    out = []
    for ps in seeds:
        pp = pbitw(lg, rules, best["pbit"][1], best["pbit"][2]*mult, np.random.default_rng(1000*split+ps), col)
        sig = {"口の自信": -conf, "数える": vio-conf/10, "MaxSAT": (ms != m)-conf/10,
               "BP": np.where(m == 1, 1-pb, pb), "揺れ": np.where(m == 1, 1-pp, pp)}
        subsets = {"全体": test, "ことば→種類": np.array([i for i in test if KIND[i] == "ことば→種類"]),
                   "どのルールも破っていない主張": test[vio[test] == 0]}
        auc = {}
        for sn, Sx in subsets.items():
            y = err[Sx]
            auc[sn] = {k: (float(roc_auc_score(y, v[Sx])) if 0 < y.sum() < len(y) else float("nan")) for k,v in sig.items()}
            auc[sn]["主張"] = int(len(Sx)); auc[sn]["口の間違い"] = int(y.sum())
        accd = {"口": float((m[test] == gold[test]).mean()), "MaxSAT": float((ms[test] == gold[test]).mean()),
                "BP": float(((pb[test] > .5) == gold[test]).mean()), "揺れ": float(((pp[test] > .5) == gold[test]).mean())}
        out.append({"cond": tag, "split": split, "pbit_seed": ps, "auc": auc, "acc": accd,
                    "best": {k: v[1:] for k,v in best.items()}, "bp_converged": bool(conv), "n_rules": len(rules)})
        print(f"[{tag}] 分け方{split} 乱数{ps}: " + "  ".join(f"{k} {auc['全体'][k]:.3f}" for k in ["口の自信","数える","MaxSAT","BP","揺れ"]), flush=True)
    return out

def flip(excl, frac, seed):
    r = np.random.default_rng(10_000 + seed); keys = list(excl)
    sel = set(r.choice(len(keys), int(round(frac*len(keys))), replace=False))
    return {k: (not v) if i in sel else v for i,(k,v) in enumerate(keys and excl.items())}

def judge(RES, a, b, sub="全体"):
    A = np.array([r["auc"][sub][a] for r in RES]); B = np.array([r["auc"][sub][b] for r in RES])
    ok = ~(np.isnan(A)|np.isnan(B)); A, B = A[ok], B[ok]
    d = (A-B).mean(); w = int((A > B).sum()); n = len(A)
    if d >= 0.02 and w >= 0.8*n: v = f"{a}の勝ち"
    elif d <= -0.02 and (B > A).sum() >= 0.8*n: v = f"{b}の勝ち"
    elif abs(d) < 0.02: v = "同着"
    else: v = "はっきりしない"
    return f"{a} 対 {b}：差 {d:+.3f}（{w}/{n}回で{a}が上）→ {v}"
