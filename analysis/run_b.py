exec(open("e24.py").read())
d = json.load(open(sys.argv[1])); tag = sys.argv[2]
P = [tuple(tuple(x) for x in p) for p in d["pairs"]]; assert P == PAIRS, "pairs mismatch"
n = len(P); L = np.array(d["logit"]); avg = (L[:n] + L[n:]) / 2
T = np.array([TRUTH[p] for p in PAIRS]); G = avg < 0
print(f"rule acc {(G==T).mean():.3f} miss {(T&~G).sum()}/{T.sum()} wrong {(~T&G).sum()}/{(~T).sum()} order-flip {((L[:n]<0)!=(L[n:]<0)).sum()}")
ALL = []
for c in sys.argv[3].split(","):
    if c == "B": rules, mult = build({p: bool(g) for p, g in zip(PAIRS, G)})
    elif c == "Bw":
        pno = 1/(1+np.exp(avg)); rules, mult = build({p: True for p in PAIRS}, {p: float(w) for p, w in zip(PAIRS, pno)})
    for s in SPLITS: ALL += run_split(s, rules, mult, tag=c)
    json.dump(ALL, open(f"../results/{tag}_{c}.json","w"), ensure_ascii=False, indent=1)
