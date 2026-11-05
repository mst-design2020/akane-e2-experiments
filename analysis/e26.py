exec(open("e24.py").read())
def maxsat26(lg, rules, lam, W):
    n = len(lg); wc = WCNF()
    for v in range(n):
        if v == DUMMY: continue
        wc.append([(v+1) if lg[v] > 0 else -(v+1)], weight=max(1, int(round(100*lam*abs(lg[v])))))
    wc.append([-(DUMMY+1)], weight=10**7)
    for p, q, h in rules:
        wc.append([-(p+1), -(q+1)] + ([h+1] if h != DUMMY else []), weight=max(1, int(round(100*W))))
    with RC2(wc) as r: m = r.compute()
    return (np.array(m[:n]) > 0).astype(int)
def load_rule_belief(path, form):
    d = json.load(open(path)); RL = np.array(d["logit"])
    if "qs" in d:
        v = np.zeros((len(PAIRS), 2))
        for (s, i, f, o, q), l in zip(d["qs"], RL):
            if s == "test" and f == form: v[i, o] = l
        a = v.mean(1)
    else:
        n = len(PAIRS); a = (RL[:n] + RL[n:]) / 2
    yme = {"P0": False, "P1": False, "P2": True}[form]
    return a if yme else -a          # >0 = Gemma thinks 両立しない
def build26(z):
    rules = [tuple(r) for r in TRANS]; seen = set(rules); u0 = DUMMY + 1
    for j, (p, q) in enumerate(PAIRS):
        u = u0 + j; inst = []
        for name in set(CITIES):
            a, b = key_of(p, name), key_of(q, name)
            if a in idx and b in idx: inst.append((idx[a], idx[b]))
        if p[0] == "Z" and q[0] == "Z":
            for k in COUNTRIES:
                if ("CK", k, p[1]) in idx and ("CK", k, q[1]) in idx: inst.append((idx[("CK", k, p[1])], idx[("CK", k, q[1])]))
        for i, k in inst: rules.append((min(i, k), max(i, k), u))
    zf = 2 * z / z.std()
    return rules, -zf                       # u=1 means rule does NOT hold → field −z
def run26(split, z, hard_rules, seeds=(1, 2), tag=""):
    rules, ufield = build26(z)
    nall = DUMMY + 1 + len(PAIRS)
    col = coloring(nall, rules)
    rng = np.random.default_rng(split)
    calc = set(rng.choice(ucities, int(round(0.2*len(ucities))), replace=False))
    cal = np.array([i for i in range(DUMMY) if CITY[i] in calc]); test = np.array([i for i in range(DUMMY) if CITY[i] not in calc])
    cl = LOGIT[cal]; scale = 2.0/cl.std()
    shift = max(np.quantile(cl, np.linspace(0.02, 0.98, 97)), key=lambda b: ((cl > b) == GOLD[cal]).mean())
    base = np.append((LOGIT-shift)*scale, -60.0); gold = np.append(GOLD, 0); m = (base > 0).astype(int)
    mk = lambda lr: np.concatenate([base, lr*ufield])
    best = {}
    for lam, W, lr in itertools.product([0.5, 1, 2], [1, 2, 4], [0.5, 1, 2]):
        lg = mk(lr)
        acc = {"maxsat": (maxsat26(lg, rules, lam, W)[cal] == gold[cal]).mean(),
               "bp": ((bpw(lg, rules, lam, W)[0][cal] > .5) == gold[cal]).mean(),
               "pbit": ((pbitw(lg, rules, lam, W, np.random.default_rng(0), col)[cal] > .5) == gold[cal]).mean()}
        for k, a in acc.items():
            if k not in best or a > best[k][0]: best[k] = (a, lam, W, lr)
    b = best["maxsat"]; ms = maxsat26(mk(b[3]), rules, b[1], b[2])
    b = best["bp"]; pb, conv = bpw(mk(b[3]), rules, b[1], b[2])
    vio = violations(m, hard_rules); conf = np.abs(1/(1+np.exp(-base))-.5); err = (m != gold).astype(int)
    T = np.array([TRUTH[p] for p in PAIRS]); u0 = DUMMY + 1
    rule_true_bp = 1 - pb[u0:]
    out = []
    for ps in seeds:
        b = best["pbit"]; pp = pbitw(mk(b[3]), rules, b[1], b[2], np.random.default_rng(1000*split+ps), col)
        mm = m[:DUMMY+1]
        sig = {"口の自信": -conf, "数える": vio-conf/10, "MaxSAT": (ms[:DUMMY+1] != mm)-conf/10,
               "BP": np.where(mm == 1, 1-pb[:DUMMY+1], pb[:DUMMY+1]), "揺れ": np.where(mm == 1, 1-pp[:DUMMY+1], pp[:DUMMY+1])}
        subsets = {"全体": test, "ことば→種類": np.array([i for i in test if KIND[i] in ("ことば→種類", "市→国")], dtype=int),
                   "どのルールも破っていない主張": test[vio[test] == 0]}
        auc = {}
        for sn, Sx in subsets.items():
            y = err[Sx]
            auc[sn] = {k: (float(roc_auc_score(y, v[Sx])) if 0 < y.sum() < len(y) else float("nan")) for k, v in sig.items()}
            auc[sn]["主張"] = int(len(Sx)); auc[sn]["口の間違い"] = int(y.sum())
        accd = {"口": float((m[test] == gold[test]).mean()), "MaxSAT": float((ms[test] == gold[test]).mean()),
                "BP": float(((pb[test] > .5) == gold[test]).mean()), "揺れ": float(((pp[test] > .5) == gold[test]).mean())}
        rt_pb = 1 - pp[u0:]
        rule = {"AUROC_最初の自信": float(roc_auc_score(T, z)), "AUROC_BP": float(roc_auc_score(T, rule_true_bp)),
                "AUROC_揺れ": float(roc_auc_score(T, rt_pb)),
                "正しさ_最初": float(((z > 0) == T).mean()), "正しさ_BP": float(((rule_true_bp > .5) == T).mean()),
                "害の3本": {f"{PAIRS[j][0][1]}|{PAIRS[j][1][1]}": [float(1/(1+np.exp(-2*z[j]/z.std()*b[3]))), float(rule_true_bp[j]), float(rt_pb[j])]
                           for j in np.nonzero(~T & (z > 0))[0]}}
        out.append({"cond": tag, "split": split, "pbit_seed": ps, "auc": auc, "acc": accd, "rule": rule,
                    "best": {k: v[1:] for k, v in best.items()}, "bp_converged": bool(conv)})
        print(f"[{tag}] 分け方{split} 乱数{ps}: " + "  ".join(f"{k} {auc['全体'][k]:.3f}" for k in ["口の自信", "数える", "MaxSAT", "BP", "揺れ"])
              + f" | rule AUROC z {rule['AUROC_最初の自信']:.3f} BP {rule['AUROC_BP']:.3f}", flush=True)
    return out
if __name__ == "__main__" and len(sys.argv) > 2:
    path, form, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    z = load_rule_belief(path, form)
    hard, _ = build({p: bool(x) for p, x in zip(PAIRS, z > 0)})
    ALL = []
    for s in SPLITS:
        ALL += run26(s, z, hard, tag=tag)
        json.dump(ALL, open(f"../results/E26_{tag}.json", "w"), ensure_ascii=False, indent=1)
