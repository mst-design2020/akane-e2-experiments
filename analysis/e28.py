# E2-8 計算（Claude側）。使い方: python3 e28.py answers.json tag [splits]
import os, sys, ast, json, itertools, collections
import numpy as np
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__)); SR = HERE
DATA_SEED = 0
exec(open(os.path.join(HERE, "data_dis.py")).read())
def grab(path, names):      # 前の実験の計算部分（関数）をそのまま使う
    src = open(path).read(); tree = ast.parse(src); g = globals()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.Assign)) and (getattr(node, "name", None) in names or
                (isinstance(node, ast.Assign) and any(getattr(t, "id", None) in names for t in node.targets))):
            exec(compile(ast.Module([node], []), path, "exec"), g)
grab(os.path.join(SR, "base.py"), {"coloring", "A8", "VIOL", "violations"})
grab(os.path.join(SR, "e24.py"), {"pbitw", "bpw", "judge"})
grab(os.path.join(SR, "e26.py"), {"maxsat26"})
ANS = sys.argv[1]; TAG = sys.argv[2]
SPLITS = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1, 2, 3, 4]
d = json.load(open(ANS))
assert d["stmt"] == STMT, "問題が合わない"
assert [tuple(p) for p in d["pairs"]] == PAIRS and d["truth"] == [TRUTH[p] for p in PAIRS]
LOGIT = np.array(d["mouth"])
v = np.zeros((len(PAIRS), 2))
for (j, o), l in zip(d["rule_qs"], d["rule_logit"]): v[j, o] = l
Z = v.mean(1)                               # >0 = Gemmaは「両立しない」と思う
T = np.array([TRUTH[p] for p in PAIRS])
CH = set()                                  # Gemmaが選んだ組
for c in d["choose"]:
    for n in c["picked"]: CH.add(tuple(sorted((c["label"], n))))
CHI = np.array([p in CH for p in PAIRS])
def rstat(mask, pred):
    t = T[mask]; p = pred[mask]
    return {"組": int(mask.sum()), "正しい判断": float((p == t).mean()) if mask.any() else float("nan"),
            "間違ったルール(両立するのに両立しないと言った)": int((~t & p).sum()), "見逃し": int((t & ~p).sum()),
            "両立しない(正解)": int(t.sum())}
RS = {"全部の組（P2の聞き方）": rstat(np.ones(len(PAIRS), bool), Z > 0),
      "Gemmaが選んだ組": rstat(CHI, np.ones(len(PAIRS), bool)),
      "Gemmaが選んだ組（P2で両立しない）": rstat(CHI, Z > 0)}
for k, s in RS.items(): print(f"[{TAG}] ルール {k}: {s}")
print(f"[{TAG}] 口の正解率 {((LOGIT > 0) == GOLD).mean():.3f}")
def hard_of(sel):
    rules = list(TRANS); seen = set(rules)
    for j in np.nonzero(sel)[0]:
        for i, k in INST[PAIRS[j]]:
            r = (min(i, k), max(i, k), DUMMY)
            if r not in seen: seen.add(r); rules.append(r)
    return rules
def build28(use):            # use: 使う組の番号。組ごとに「本当か」の札（u=1でルールが成り立たない）
    rules = list(TRANS); u0 = DUMMY + 1
    for n, j in enumerate(use):
        for i, k in INST[PAIRS[j]]: rules.append((min(i, k), max(i, k), u0 + n))
    return rules
ucities = sorted(set(c for c in CITY if c))
grp = {i: (CITY[i] if CITY[i] else ("k", STMT[i].split(" is a kind")[0])) for i in range(DUMMY)}
members = collections.defaultdict(list)
for i, g in grp.items(): members[g].append(i)
sig_ = lambda x: 1/(1+np.exp(-x))
def run28(split, use, hard, seeds=(1, 2), tag=""):
    use = np.asarray(use, int); z = Z[use]
    rules = build28(use); ufield = -(2*z/Z.std())
    nall = DUMMY + 1 + len(use); col = coloring(nall, rules)
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
    vio = violations(m, hard); conf = np.abs(sig_(base)-.5); err = (m != gold).astype(int)
    # 他の方法：①口の自信 ③近所の一貫性（同じ病気の他の問題） ④組み合わせ（①③数える）
    bm = base[:DUMMY]; pm = sig_(bm); own = np.where(m[:DUMMY] == 1, pm, 1-pm)
    ncb = np.array([own[i]*(np.exp(np.mean(np.log(own[[j for j in members[grp[i]] if j != i]] + 1e-12)))
                            if len(members[grp[i]]) > 1 else 1.0) for i in range(DUMMY)])
    X = np.c_[conf[:DUMMY], ncb, vio[:DUMMY]]; mu, sd = X[cal].mean(0), X[cal].std(0)+1e-9
    lrm = LogisticRegression(C=1.0, max_iter=1000).fit((X[cal]-mu)/sd, err[cal]); ens = np.append(lrm.decision_function((X-mu)/sd), 0)
    ncb = np.append(ncb, 1)
    rule_true_bp = 1 - pb[DUMMY+1:]; Tu = T[use]
    out = []
    for ps in seeds:
        b = best["pbit"]; pp = pbitw(mk(b[3]), rules, b[1], b[2], np.random.default_rng(1000*split+ps), col)
        mm = m[:DUMMY+1]
        sg = {"口の自信": -conf, "③近所の一貫性": -ncb, "④組み合わせ": ens, "数える": vio-conf/10,
              "MaxSAT": (ms[:DUMMY+1] != mm)-conf/10,
              "BP": np.where(mm == 1, 1-pb[:DUMMY+1], pb[:DUMMY+1]), "揺れ": np.where(mm == 1, 1-pp[:DUMMY+1], pp[:DUMMY+1])}
        subsets = {"全体": test, "どのルールも破っていない主張": test[vio[test] == 0]}
        auc = {}
        for sn, Sx in subsets.items():
            y = err[Sx]
            auc[sn] = {k: (float(roc_auc_score(y, s[Sx])) if 0 < y.sum() < len(y) else float("nan")) for k, s in sg.items()}
            auc[sn]["主張"] = int(len(Sx)); auc[sn]["口の間違い"] = int(y.sum())
        accd = {"口": float((m[test] == gold[test]).mean()), "MaxSAT": float((ms[test] == gold[test]).mean()),
                "BP": float(((pb[test] > .5) == gold[test]).mean()), "揺れ": float(((pp[test] > .5) == gold[test]).mean())}
        rt_pp = 1 - pp[DUMMY+1:]
        ok = 0 < Tu.sum() < len(Tu)
        rule = {"組": int(len(use)), "AUROC_最初の自信": float(roc_auc_score(Tu, z)) if ok else float("nan"),
                "AUROC_BP": float(roc_auc_score(Tu, rule_true_bp)) if ok else float("nan"),
                "正しさ_最初": float(((z > 0) == Tu).mean()), "正しさ_BP": float(((rule_true_bp > .5) == Tu).mean())}
        out.append({"cond": tag, "split": split, "pbit_seed": ps, "auc": auc, "acc": accd, "rule": rule,
                    "best": {k: v[1:] for k, v in best.items()}, "bp_converged": bool(conv), "n_rules": len(rules)})
        print(f"[{tag}] 分け方{split} 乱数{ps}: " + "  ".join(f"{k} {auc['全体'][k]:.3f}" for k in sg)
              + f" | ルール見抜き 最初{rule['AUROC_最初の自信']:.3f} BP{rule['AUROC_BP']:.3f}", flush=True)
    return out
if __name__ == "__main__":
    allp = np.arange(len(PAIRS)); chp = np.nonzero(CHI)[0]
    RES = {"B": [], "A": []}
    for s in SPLITS:
        RES["B"] += run28(s, allp, hard_of(Z > 0), tag=TAG + "_全部の組")
        RES["A"] += run28(s, chp, hard_of(CHI), tag=TAG + "_Gemmaが選んだ組")
        json.dump({"rules": RS, "res": RES}, open(os.path.join(HERE, "..", "results", "E28_runs", f"E28_{TAG}.json"), "w"), ensure_ascii=False, indent=1)
    B, A = RES["B"], RES["A"]
    print("問い1:", judge(B, "BP", "数える")); print("記録:", judge(B, "揺れ", "数える"))
    print("記録:", judge(B, "BP", "④組み合わせ")); print("記録:", judge(B, "BP", "口の自信"))
    AB = [{"auc": {"全体": {"BP(選んだ組)": a["auc"]["全体"]["BP"], "BP(全部の組)": b["auc"]["全体"]["BP"]}}} for a, b in zip(A, B)]
    print("問い3:", judge(AB, "BP(選んだ組)", "BP(全部の組)"))
    print("記録（選んだ組の中で）:", judge(A, "BP", "数える"))
    for k, R in RES.items():
        print(k, "平均:", {kk: round(float(np.nanmean([r["auc"]["全体"][kk] for r in R])), 3) for kk in R[0]["auc"]["全体"] if isinstance(R[0]["auc"]["全体"][kk], float)})
