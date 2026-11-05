exec(open("e24.py").read())
d = json.load(open(sys.argv[1])); forms = sys.argv[2].split(",")
QS = d["qs"]; RL = np.array(d["logit"]); YME = {"P0": False, "P1": False, "P2": True}
assert [tuple(tuple(x) for x in p) for p in d["test_pairs"]] == PAIRS
def excl_of(setname, form, n):
    v = np.zeros((n, 2))
    for (s, i, f, o, q), l in zip(QS, RL):
        if s == setname and f == form: v[i, o] = l
    a = v.mean(1); return (a > 0) if YME[form] else (a < 0)
dt = np.array([t for *_, t in d["dev"]]); nd = len(dt)
bal = lambda p, t: 0.5 * (p[t].mean() + (~p[~t]).mean())
dp = {f: excl_of("dev", f, nd) for f in ["P0", "P1", "P2"]}; dp["P12"] = dp["P1"] & dp["P2"]
sc = {f: bal(p, dt) for f, p in dp.items()}; print("dev", {k: round(v, 3) for k, v in sc.items()})
best = max(["P1", "P0", "P2", "P12"], key=lambda f: sc[f]); print("best", best)
T = np.array([TRUTH[p] for p in PAIRS])
tp = {f: excl_of("test", f, len(PAIRS)) for f in ["P1", "P2"]}; tp["P12"] = tp["P1"] & tp["P2"]
AN = {"mammal.n.01", "bird.n.01", "fish.n.01", "insect.n.01", "reptile.n.01"}
clear = np.array([(p[0][0] == "K" and q[0] == "K" and p[1] in AN and q[1] in AN) or (p[0] == "Z" and q[0] == "Z") or not TRUTH[(p, q)] for p, q in PAIRS])
for f, g in tp.items():
    print(f, f"acc {(g == T).mean():.3f} miss {(T & ~g).sum()}/{T.sum()} wrong {(~T & g).sum()}/{(~T).sum()} | clear({clear.sum()}) acc {(g[clear] == T[clear]).mean():.3f}")
ALL = []
for f in forms:
    rules, mult = build({p: bool(x) for p, x in zip(PAIRS, tp[f])})
    for s in SPLITS: ALL += run_split(s, rules, mult, tag=f)
    json.dump(ALL, open(f"../results/E25_{'_'.join(forms)}.json", "w"), ensure_ascii=False, indent=1)
