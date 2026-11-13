"""E16-2：E16-1（網を揺らして矛盾の揺れをTADORIに回す）を病気・地理で繰り返す。計画 PLAN_E16-2.md。
python3 e162.py <E10-1のコードと e101_items.json・results/ がある場所> doid_2026-10-29.obo out.json
（公開リポジトリでは analysis/e10/e101 に data/E10/E10-1 の中身を results/ として、e101_items.json を並べて置く）
手書きの語の一覧はない（病気の決まりは分類表から、地理は「地域は重ならない」だけ）。"""
import json, sys, time, os, itertools, collections, ast
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
E101 = sys.argv[1]; sys.path.insert(0, E101)
import e101_analyze as ea
items, data, _ = ea.build(os.path.join(E101, "e101_items.json"), os.path.join(E101, "results"))
CS = {int(k): v for k, v in json.load(open(os.path.join(E101, "results", "cs_labels_claude.json"))).items()}
sig = lambda x: 1 / (1 + np.exp(-np.clip(x, -50, 50)))
SWEEPS = int(os.environ.get("SWEEPS", 1000)); CHAINS = 4; GRID = [0.5, 1, 2, 4, 8]; BUD = [0.01, 0.025, 0.05]

# ---- 病気の分類表（E2-8と同じ読み方）
TERMS = {}; cur = None
for line in open(sys.argv[2], encoding="utf-8"):
    line = line.rstrip("\n")
    if line == "[Term]": cur = {"isa": [], "obs": False}; continue
    if line.startswith("["): cur = None; continue
    if cur is None: continue
    if line.startswith("id: "): cur["id"] = line[4:]; TERMS[cur["id"]] = cur
    elif line.startswith("is_a: "): cur["isa"].append(line[6:].split(" ")[0])
    elif line.startswith("is_obsolete: true"): cur["obs"] = True
TERMS = {k: v for k, v in TERMS.items() if not v["obs"]}
sys.setrecursionlimit(10000); _anc = {}
def anc(k):
    if k not in _anc:
        s = set()
        for p in TERMS[k]["isa"]:
            if p in TERMS: s |= {p} | anc(p)
        _anc[k] = s
    return _anc[k]


def gibbs(U, A, Dm, W1, W2, sweeps, chains, rng, read=True):
    n, K = U.shape; X = (rng.random((chains, n, K)) < 0.5).astype(float)
    burn = sweeps // 5; P = np.zeros((n, K)); M = np.zeros((n, K)); N = np.zeros((n, K)); cnt = 0
    for t in range(sweeps):
        rec = read and t >= burn
        for c in range(K):
            t1 = -W1 * ((1 - X) @ A[c]); t2 = W1 * (X @ A[:, c]); t3 = -W2 * (X @ Dm[c])
            f = U[None, :, c] + t1 + t2 + t3
            X[:, :, c] = rng.random((chains, n)) < sig(f)
            if rec:
                M[:, c] += (np.abs(U[None, :, c]) + np.abs(t1) + np.abs(t2) + np.abs(t3)).sum(0); N[:, c] += np.abs(f).sum(0)
        if t >= burn: P += X.sum(0); cnt += chains
    return P / cnt, M / max(cnt, 1), N / max(cnt, 1)


def aurc(score, ok):
    o = np.argsort(-score, kind="stable"); cum = np.cumsum(ok[o]) / np.arange(1, len(ok) + 1)
    return float(np.mean(1 - cum))


def viol(X, A, Dm):  # 主張ごとの破り数
    return X * ((1 - X) @ A.T) + (1 - X) * (X @ A) + X * (X @ Dm.T)


out = {}; t0 = time.time()
for dom in "DG":
    Dd = data[dom]; its = Dd["its"]
    ents = sorted({it["ent"] for it in its}); cls = sorted({it["cls"] for it in its})
    ei = {e: i for i, e in enumerate(ents)}; ci = {c: i for i, c in enumerate(cls)}; E, K = len(ents), len(cls)
    LG = np.full((E, K), np.nan); Y = np.zeros((E, K)); TS = np.full((E, K), -1.0); IDS = np.zeros((E, K), int)
    sixN = Dd["F"]["sixN"]; has = Dd["F"]["has"] > 0
    for j, it in enumerate(its):
        e, c = ei[it["ent"]], ci[it["cls"]]
        LG[e, c] = Dd["lg"][j]; Y[e, c] = it["gold"]; IDS[e, c] = it["id"]
        if has[j]: TS[e, c] = sixN[j]
    assert not np.isnan(LG).any()
    A = np.zeros((K, K)); Dm = np.zeros((K, K))
    if dom == "D":
        src = open(os.path.join(E101, "e101_data.py"), encoding="utf-8").read()   # 分類→DOIDの表（問いの定義、E10-1・E2-8と同じ）を読む
        tree = ast.parse(src); LAB = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "LABELS": LAB = ast.literal_eval(node.value)
        LID = {l: LAB[l][0] for l in cls}
        for p in cls:
            for q in cls:
                if p != q and LID[q] in anc(LID[p]): A[ci[p], ci[q]] = 1
        both = collections.Counter()
        for k in TERMS:
            L = {l for l in cls if LID[l] in anc(k) or LID[l] == k}
            for p, q in itertools.combinations(sorted(L), 2): both[(p, q)] += 1
        for p, q in itertools.combinations(cls, 2):
            if both[(p, q)] == 0: Dm[ci[p], ci[q]] = Dm[ci[q], ci[p]] = 1
    else:
        Dm = 1 - np.eye(K)
    rules = dict(n_imp=int(A.sum()), n_disj=int(Dm.sum() / 2), imp=[[cls[i], cls[j]] for i, j in zip(*np.nonzero(A))])
    print(dom, E, K, rules["n_imp"], rules["n_disj"], rules["imp"], flush=True)
    G0 = Y; bad = int((viol(G0, A, Dm)).sum()); print(" 正解が破る決まり", bad, flush=True)
    can = TS >= 0; tad = TS > 0.5
    csl = [(e, c, 1.0 if CS[IDS[e, c]] >= 1 else 0.0) for e in range(E) for c in range(K) if IDS[e, c] in CS]
    Xraw = (LG > 0).astype(float); vraw = viol(Xraw, A, Dm)
    n_all = E * K; R_dom = {"rules": rules, "gold_viol": bad, "E": E, "K": K, "n_can": int(can.sum()), "seeds": {}}
    for seed in range(5):
        rng = np.random.default_rng(seed); perm = rng.permutation(E); folds = np.array_split(perm, 5)
        R = {k: np.zeros((E, K)) for k in ("p", "M", "N", "U")}; meta = []
        for f in range(5):
            te = folds[f]; tr = np.concatenate([folds[g] for g in range(5) if g != f])
            lr = LogisticRegression().fit(LG[tr].reshape(-1, 1), Y[tr].ravel()); a, b = lr.coef_[0, 0], lr.intercept_[0]; U = a * LG + b
            sub = tr[rng.permutation(len(tr))[:400]]; best = None
            for W1 in GRID:
                for W2 in GRID:
                    p, _, _ = gibbs(U[sub], A, Dm, W1, W2, 200, 2, rng, read=False)
                    s = aurc(np.abs(p - 0.5).ravel(), ((p >= 0.5) == Y[sub]).ravel().astype(float))
                    if best is None or s < best[0]: best = (s, W1, W2)
            _, W1, W2 = best; meta.append(dict(fold=f, W1=W1, W2=W2))
            p, M, N = gibbs(U[te], A, Dm, W1, W2, SWEEPS, CHAINS, rng)
            R["p"][te], R["M"][te], R["N"][te], R["U"][te] = p, M, N, U[te]
        p = R["p"]; U = R["U"]; C = R["M"] - R["N"]; ans = p >= 0.5; ok = (ans == Y).astype(float); au = np.abs(U)
        pred3 = np.zeros((E, K)); Xf = np.stack([p, U], -1)
        for f in range(5):
            te = folds[f]; tr = np.concatenate([folds[g] for g in range(5) if g != f])
            m = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(Xf[tr].reshape(-1, 2), ok[tr].ravel())
            pred3[te] = m.predict_proba(Xf[te].reshape(-1, 2))[:, 1].reshape(len(te), K)
        gem = U > 0; vcal = viol(gem.astype(float), A, Dm); tb = au / (au.max() + 1)
        pri = {"i_conflict": C, "ii_uncertain": -np.abs(p - 0.5), "iii_tree": -pred3, "viia_viol_raw": vraw - tb, "viib_viol_cal": vcal - tb}
        base_wrong = int((ans != Y).sum()); base_cs = sum(int(ans[e, c] != y) for e, c, y in csl)
        res = {}; ref = {"gemma_alone": int((gem != Y).sum()), "net_alone": base_wrong}
        for bgt in BUD:
            n = int(round(bgt * n_all)); r = {}
            for name, s in pri.items():
                s = np.where(can, s, -np.inf).ravel(); pick = np.argsort(-s, kind="stable")[:n]
                new = ans.copy().ravel(); new[pick] = tad.ravel()[pick]; new = new.reshape(E, K)
                r[name] = dict(fixed=base_wrong - int((new != Y).sum()), wrong_in_pick=int((ans.ravel()[pick] != Y.ravel()[pick]).sum()),
                               fixed_cs=base_cs - sum(int(new[e, c] != y) for e, c, y in csl))
            res[str(bgt)] = r
            for nm, s in [("gemma+viol", vcal - tb), ("gemma+unsure", -au)]:
                s = np.where(can, s, -np.inf).ravel(); pick = np.argsort(-s, kind="stable")[:n]
                new = gem.copy().ravel(); new[pick] = tad.ravel()[pick]; ref[f"{nm}@{bgt}"] = int((new.reshape(E, K) != Y).sum())
            ref[f"net+conflict@{bgt}"] = base_wrong - r["i_conflict"]["fixed"]
        shaky = (p >= 0.1) & (p <= 0.9); wrong = ok == 0
        medC = np.median(C[shaky]) if shaky.any() else 0; medM = np.median(R["M"][shaky]) if shaky.any() else 0
        cf = shaky & (C >= medC); ig = shaky & (R["M"] < medM) & (C < medC)
        typ = dict(n_wrong=int(wrong.sum()), firm_wrong=int((wrong & ~shaky).sum()), conflict=[int(cf.sum()), int((cf & wrong).sum())], ignorance=[int(ig.sum()), int((ig & wrong).sum())])
        R_dom["seeds"][seed] = dict(meta=meta, base_wrong=base_wrong, base_cs=base_cs, res=res, ref=ref, types=typ)
        print(dom, seed, base_wrong, {b: {k: v["fixed"] for k, v in r.items()} for b, r in res.items()}, f"{time.time() - t0:.0f}秒", flush=True)
    S = R_dom["seeds"]; win = {}
    for bgt in BUD:
        w = 0
        for s in S:
            r = S[s]["res"][str(bgt)]; best = max(r[k]["fixed"] for k in r if k != "i_conflict")
            w += r["i_conflict"]["fixed"] >= 1.1 * best and r["i_conflict"]["fixed"] > 0
        win[str(bgt)] = int(w)
    R_dom["rule"] = dict(wins_per_budget=win, works=sum(v >= 4 for v in win.values()) >= 2)
    R_dom["mean"] = {str(b): {k: float(np.mean([S[s]["res"][str(b)][k]["fixed"] for s in S])) for k in S[0]["res"][str(b)]} for b in BUD}
    R_dom["mean_cs"] = {str(b): {k: float(np.mean([S[s]["res"][str(b)][k]["fixed_cs"] for s in S])) for k in S[0]["res"][str(b)]} for b in BUD}
    R_dom["mean_ref"] = {k: float(np.mean([S[s]["ref"][k] for s in S])) for k in S[0]["ref"]}
    print(dom, json.dumps({k: R_dom[k] for k in ["rule", "mean", "mean_cs", "mean_ref"]}, ensure_ascii=False), flush=True)
    out[dom] = R_dom
json.dump(out, open(sys.argv[3], "w"), ensure_ascii=False, indent=1)
