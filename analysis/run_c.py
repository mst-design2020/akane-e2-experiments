exec(open("e24.py").read())
import time
print("pairs", len(PAIRS), "両立しない", sum(TRUTH.values()))
rA, mA = build(TRUTH); print("A rules", len(rA), "E2-3 rules", len(RULES))
conds = sys.argv[1].split(",")
ALL = []
t = time.time()
for c in conds:
    for s in SPLITS:
        if c == "E23": rules, mult = [tuple(r) for r in RULES], np.ones(len(RULES))
        elif c == "A": rules, mult = rA, mA
        else:
            f = float(c[1:]); rules, mult = build(flip(TRUTH, f, s))
        ALL += run_split(s, rules, mult, tag=c)
    print(c, f"{time.time()-t:.0f}s", flush=True)
    json.dump(ALL, open(f"../results/C_{sys.argv[1].replace(',','_')}.json","w"), ensure_ascii=False, indent=1)
