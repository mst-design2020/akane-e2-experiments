# E2-7: usage  BASE=base.py|base_geo.py python3 e27.py answers.json DOM(wn|geo) mode(run|reuse:<E26 json>) tag
import os, sys
_argv = sys.argv; sys.argv = sys.argv[:1]
exec(open("e26.py").read())
sys.argv = _argv
from sklearn.ensemble import HistGradientBoostingClassifier as GBC
ANS, DOM, MODE, TAG = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
d = json.load(open(ANS)); RL = np.array(d["logit"]); QS = d["qs"]
sig_ = lambda x: 1/(1+np.exp(-x))
# ---- paraphrase answers, checked against statement text ----
pA = np.full(DUMMY, np.nan); pB = np.full(DUMMY, np.nan)
for (k, i, o, a, q), l in zip(QS, RL):
    if k == DOM + "_paraA":
        assert q.endswith("Statement: " + STMT[i]), (i, q); pA[i] = l
    elif k == DOM + "_paraB":
        assert q.startswith(STMT[i] + " Is this"), (i, q); pB[i] = l
assert not np.isnan(pA).any() and not np.isnan(pB).any()
PT_A, PT_B = sig_(-pA), sig_(pB)          # P(true) under each paraphrase
# ---- rule belief ----
if DOM == "geo":
    gp = [tuple(tuple(x) for x in pq) for pq in d["geo_pairs"]]
    assert gp == list(PAIRS), "地理の組が合わない"
    assert d["geo_truth"] == [TRUTH[p] for p in PAIRS]
    v = np.zeros((len(PAIRS), 2))
    for (k, j, o, a, q), l in zip(QS, RL):
        if k == "georule": v[j, o] = l
    Z = v.mean(1)                          # yes = 両立しない
else:
    Z = load_rule_belief("../data/E2-5_rules_gemma-2-2b-it.json", "P2")
T = np.array([TRUTH[p] for p in PAIRS])
print(f"[{DOM}] ルールの正しさ {((Z > 0) == T).mean():.3f}  見逃し {(T & (Z <= 0)).sum()}/{T.sum()}  間違ったルール {(~T & (Z > 0)).sum()}/{(~T).sum()}")
print(f"[{DOM}] 言い換えA正解率 {((PT_A > .5) == GOLD).mean():.3f}  言い換えB {((PT_B > .5) == GOLD).mean():.3f}  元 {((LOGIT > 0) == GOLD).mean():.3f}")
HARD, _ = build({p: bool(x) for p, x in zip(PAIRS, Z > 0)})
# neighbours: same CITY; CK statements (CITY None) grouped by kind
grp = {}
for (t, a, b), i in idx.items(): grp[i] = ("c", a) if t != "CK" else ("k", a)
members = collections.defaultdict(list)
for i, g in grp.items(): members[g].append(i)
def baselines(split):
    rng = np.random.default_rng(split)
    calc = set(rng.choice(ucities, int(round(0.2*len(ucities))), replace=False))
    cal = np.array([i for i in range(DUMMY) if CITY[i] in calc]); test = np.array([i for i in range(DUMMY) if CITY[i] not in calc])
    cl = LOGIT[cal]; scale = 2.0/cl.std()
    shift = max(np.quantile(cl, np.linspace(0.02, 0.98, 97)), key=lambda b: ((cl > b) == GOLD[cal]).mean())
    base = (LOGIT-shift)*scale; m = (base > 0).astype(int); err = (m != GOLD).astype(int)
    pm = sig_(base); own = np.where(m == 1, pm, 1-pm)         # confidence in own answer
    conf = np.abs(pm-.5)
    para = np.where(m == 1, (1-PT_A + 1-PT_B)/2, (PT_A + PT_B)/2)   # prob. of opposite answer
    ncb = np.zeros(DUMMY)
    for i in range(DUMMY):
        nb = [j for j in members[grp[i]] if j != i]
        ncb[i] = own[i] * (np.exp(np.mean(np.log(own[nb] + 1e-12))) if nb else 1.0)
    vio = violations(np.append(m, 0), HARD)[:DUMMY]
    X = np.c_[conf, para, ncb, vio]; mu, sd = X[cal].mean(0), X[cal].std(0) + 1e-9
    lr = GBC(max_iter=300, learning_rate=0.05, random_state=0).fit((X[cal]-mu)/sd, err[cal])
    ens = lr.predict_proba((X-mu)/sd)[:, 1]
    S = {"①口の自信": -conf, "②言い換えのぶれ": para, "③近所の一貫性": -ncb, "④組み合わせ": ens, "数える(確認)": vio-conf/10}
    y = err[test]
    return {k: float(roc_auc_score(y, s[test])) for k, s in S.items()}, {"lr_coef": [0]}
if MODE == "run":
    RES = []
    for s in SPLITS:
        RES += run26(s, Z, HARD, tag=TAG)
        json.dump(RES, open(f"../results/E26_{TAG}.json", "w"), ensure_ascii=False, indent=1)
else:
    RES = json.load(open(MODE.split(":", 1)[1]))
OUT = []
for s in SPLITS:
    b, info = baselines(s)
    print(f"[{TAG}] 分け方{s}: " + "  ".join(f"{k} {v:.3f}" for k, v in b.items()) + f"  coef {np.round(info['lr_coef'], 2)}", flush=True)
    for r in RES:
        if r["split"] == s:
            r = dict(r); r["auc"] = dict(r["auc"]); r["auc"]["全体"] = {**r["auc"]["全体"], **b}; r["lr"] = info; OUT.append(r)
json.dump(OUT, open(f"/tmp/claude-0/-home-user/b16b4c71-981a-5e41-88b0-1a772d256980/scratchpad/E27GB_{TAG}.json", "w"), ensure_ascii=False, indent=1)
for a, bb in [("BP", "数える"), ("揺れ", "数える"), ("BP", "①口の自信"), ("BP", "②言い換えのぶれ"), ("BP", "③近所の一貫性"), ("BP", "④組み合わせ")]:
    print(judge(OUT, a, bb))
print("平均:", {k: round(float(np.mean([r["auc"]["全体"][k] for r in OUT])), 3) for k in OUT[0]["auc"]["全体"] if isinstance(OUT[0]["auc"]["全体"][k], float)})
