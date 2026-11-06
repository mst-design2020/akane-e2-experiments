# E2-9: 外れの量だけ変える。使い方: python3 e29.py dis|wn split
import os, sys, json, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SR = HERE
DOM, SPLIT = sys.argv[1], int(sys.argv[2]); LEVELS = [float(x) for x in os.environ.get("LEVELS", "0,0.1,0.2,0.3,0.5").split(",")]
NOTRANS = os.environ.get("NOTRANS") == "1"; SUF = "_notrans" if NOTRANS else ""
if DOM == "dis":
    g = {"__name__": "e28", "__file__": os.path.join(HERE, "e28.py")}
    sys.argv = ["e28", os.path.join(HERE, "..", "data", "E2-8_answers_gemma-2-2b-it.json"), "x"]
    exec(open(os.path.join(HERE, "e28.py")).read(), g)
    Zg, T = g["Z"].copy(), g["T"]
    def run(zf, hard, tag):
        g["Z"] = zf
        return g["run28"](SPLIT, np.arange(len(g["PAIRS"])), hard, tag=tag)
    hard_of = lambda zf: g["hard_of"](zf > 0)
else:
    cwd = os.getcwd(); os.chdir(SR)
    g = {"__name__": "e26"}; sys.argv = ["e26"]
    exec(open("e26.py").read(), g)
    Zg = g["load_rule_belief"](os.path.join(HERE, "..", "data", "E2-5_rules_gemma-2-2b-it.json"), "P2")
    T = np.array([g["TRUTH"][p] for p in g["PAIRS"]])
    def run(zf, hard, tag): return g["run26"](SPLIT, zf, hard, tag=tag)
    hard_of = lambda zf: g["build"]({p: bool(x) for p, x in zip(g["PAIRS"], zf > 0)})[0]
if NOTRANS: g["TRANS"] = []          # 人が書いたつながりのルールを抜く
bpw, LOGIT, GOLD, DUMMY = g["bpw"], g["LOGIT"], g["GOLD"], g["DUMMY"]
CITY = g["CITY"]; ucities = sorted(set(c for c in CITY if c))
from sklearn.metrics import roc_auc_score
def fixed_bp(hard):          # 疑わない網（ルールを固定で信じる）
    rng = np.random.default_rng(SPLIT)
    calc = set(rng.choice(ucities, int(round(0.2*len(ucities))), replace=False))
    cal = np.array([i for i in range(DUMMY) if CITY[i] in calc]); test = np.array([i for i in range(DUMMY) if CITY[i] not in calc])
    cl = LOGIT[cal]; scale = 2.0/cl.std()
    shift = max(np.quantile(cl, np.linspace(0.02, 0.98, 97)), key=lambda b: ((cl > b) == GOLD[cal]).mean())
    lg = np.append((LOGIT-shift)*scale, -60.0); m = (lg > 0).astype(int); gold = np.append(GOLD, 0)
    best = max(itertools.product([0.5, 1, 2], [1, 2, 4]), key=lambda p: ((bpw(lg, hard, *p)[0][cal] > .5) == gold[cal]).mean())
    pb = bpw(lg, hard, *best)[0]; err = (m != gold).astype(int)
    return float(roc_auc_score(err[test], np.where(m == 1, 1-pb, pb)[test]))
base = np.where(T, 1.0, -1.0) * np.abs(Zg)
r = np.random.default_rng(10000 + SPLIT); order = r.permutation(len(T))
OP = os.path.join(HERE, "..", "results", "E29_runs", f"E29{SUF}_{DOM}_s{SPLIT}.json")
OUT = json.load(open(OP)) if os.path.exists(OP) else []      # 途中から続ける（くじは同じなので結果は変わらない）
done = {x["f"] for x in OUT}
for f in LEVELS:
    if f in done: continue
    zf = base.copy(); fl = order[:int(round(f*len(T)))]; zf[fl] = -zf[fl]
    hard = hard_of(zf)
    res = run(zf, hard, f"{DOM}_f{f}")
    fb = fixed_bp(hard)
    for x in res: x["f"] = f; x["wrong_rate"] = float(((zf > 0) != T).mean()); x["auc"]["全体"]["疑わないBP"] = fb
    OUT += res
    json.dump(OUT, open(OP, "w"), ensure_ascii=False, indent=1)
    print(f"[{DOM} s{SPLIT} f{f}] 外れ率 {OUT[-1]['wrong_rate']:.2f} 疑わないBP {fb:.3f}", flush=True)
