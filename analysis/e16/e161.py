"""E16-1：網を揺らして「知らない」「矛盾」を読む。計画 PLAN_E16-1.md。
python3 e161.py ../../e4/e42_data.json ../../e4/E42_gemma-2-2b-it.json ../../e4/relabel/sample_labeled.json out.json
手書きの語の一覧はない（分類の木と両立しない組は e42_data から）。"""
import json, sys, time, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
D = json.load(open(sys.argv[1])); L = np.array(json.load(open(sys.argv[2]))["logit"])
SAMPLE = json.load(open(sys.argv[3]))
CATS = list(D["cats"]); K = len(CATS); ci = {c: i for i, c in enumerate(CATS)}
A = np.zeros((K, K))                                   # A[c,p]=1：p は c の先祖
for c, ps in D["parent"].items():
    for p in ps: A[ci[c], ci[p]] = 1
Dm = np.zeros((K, K))                                  # 両立しない組
for k, v in D["disjoint"].items():
    if v:
        a, b = k.split("|"); Dm[ci[a], ci[b]] = Dm[ci[b], ci[a]] = 1
ents = sorted(D["ents"]); E = len(ents); ei = {w: i for i, w in enumerate(ents)}
LG = np.zeros((E, K)); Y = np.zeros((E, K))
for j, c in enumerate(D["claims"]):
    if c["kind"] == "x":
        LG[ei[c["ent"]], ci[c["cat"]]] = L[j]; Y[ei[c["ent"]], ci[c["cat"]]] = c["gold"]
sig = lambda x: 1 / (1 + np.exp(-np.clip(x, -50, 50)))
SWEEPS = int(os.environ.get("SWEEPS", 1000)); CHAINS = 4


def gibbs(U, W1, W2, sweeps, chains, rng, read=True):
    n = U.shape[0]; X = (rng.random((chains, n, K)) < 0.5).astype(float)
    burn = sweeps // 5; P = np.zeros((n, K)); M = np.zeros((n, K)); N = np.zeros((n, K)); cnt = 0
    for t in range(sweeps):
        rec = read and t >= burn
        for c in range(K):
            t1 = -W1 * ((1 - X) @ A[c])                 # 先祖が0なら下へ
            t2 = W1 * (X @ A[:, c])                      # 子孫が1なら上へ
            t3 = -W2 * (X @ Dm[c])                       # 両立しない物が1なら下へ
            f = U[None, :, c] + t1 + t2 + t3
            X[:, :, c] = rng.random((chains, n)) < sig(f)
            if rec:
                M[:, c] += (np.abs(U[None, :, c]) + np.abs(t1) + np.abs(t2) + np.abs(t3)).sum(0)
                N[:, c] += np.abs(f).sum(0)
        if t >= burn:
            P += X.sum(0); cnt += chains
    return P / cnt, M / max(cnt, 1), N / max(cnt, 1)


def meanfield_read(U, P, W1, W2):
    t1 = -W1 * ((1 - P) @ A.T); t2 = W1 * (P @ A); t3 = -W2 * (P @ Dm.T)
    M = np.abs(U) + np.abs(t1) + np.abs(t2) + np.abs(t3); N = np.abs(U + t1 + t2 + t3)
    return M, N


def aurc(score, ok):                                   # score が大きいほど先に答える
    o = np.argsort(-score, kind="stable"); cum = np.cumsum(ok[o]) / np.arange(1, len(ok) + 1)
    return float(np.mean(1 - cum))


def viol_count(Xb):                                    # Gemmaの答えが破る決まりの数（物ごと）
    v1 = (Xb[:, :, None] * (1 - Xb[:, None, :]) * A[None]).sum((1, 2))
    v2 = (Xb[:, :, None] * Xb[:, None, :] * Dm[None]).sum((1, 2)) / 2
    return v1 + v2


samp = [(ei[s["ent"]], [c for c in CATS if D["cats"][c] == s["cat"]][0], s["mine"]) for s in SAMPLE if s["ent"] in ei]
GRID = [0.5, 1, 2, 4, 8]
out = {"seeds": {}}; t0 = time.time()
for seed in range(5):
    rng = np.random.default_rng(seed); perm = rng.permutation(E); folds = np.array_split(perm, 5)
    R = {k: np.zeros((E, K)) for k in ("p", "M", "N", "Mmf", "Nmf", "U")}; meta = []
    for f in range(5):
        te = folds[f]; tr = np.concatenate([folds[g] for g in range(5) if g != f])
        lr = LogisticRegression().fit(LG[tr].reshape(-1, 1), Y[tr].ravel())
        a, b = lr.coef_[0, 0], lr.intercept_[0]; U = a * LG + b
        best = None
        sub = tr[rng.permutation(len(tr))[:400]]
        for W1 in GRID:
            for W2 in GRID:
                p, _, _ = gibbs(U[sub], W1, W2, 200, 2, rng, read=False)
                s = aurc(np.abs(p - 0.5).ravel(), ((p >= 0.5) == Y[sub]).ravel().astype(float))
                if best is None or s < best[0]: best = (s, W1, W2)
        _, W1, W2 = best; meta.append(dict(fold=f, a=a, b=b, W1=W1, W2=W2))
        p, M, N = gibbs(U[te], W1, W2, SWEEPS, CHAINS, rng)
        Mm, Nm = meanfield_read(U[te], p, W1, W2)
        R["p"][te], R["M"][te], R["N"][te], R["Mmf"][te], R["Nmf"][te], R["U"][te] = p, M, N, Mm, Nm, U[te]
        print(f"seed{seed} fold{f} W1={W1} W2={W2}  {time.time() - t0:.0f}秒", flush=True)
    p = R["p"]; ok = ((p >= 0.5) == Y).astype(float); C = R["M"] - R["N"]; Cmf = R["Mmf"] - R["Nmf"]
    vc = viol_count((LG > 0).astype(float))[:, None].repeat(K, 1)
    # (a) 間違いの分かれ方
    wrong = ok == 0; shaky = (p >= 0.1) & (p <= 0.9)
    medC = np.median(C[shaky]); medM = np.median(R["M"][shaky])
    conf_ = shaky & (C >= medC); ign = shaky & (R["M"] < medM) & (C < medC); oth = shaky & ~conf_ & ~ign
    cnt = lambda m: int((m & wrong).sum())
    typ = dict(n_wrong=int(wrong.sum()), firm_wrong=cnt(~shaky), conflict=cnt(conf_), ignorance=cnt(ign), other=cnt(oth),
               err_rate=dict(firm=float(wrong[~shaky].mean()), conflict=float(wrong[conf_].mean()), ignorance=float(wrong[ign].mean()), other=float(wrong[oth].mean())),
               n=dict(firm=int((~shaky).sum()), conflict=int(conf_.sum()), ignorance=int(ign.sum()), other=int(oth.sum())))
    # (b) 控え：物の分け方は同じ5つ、木は4つで学習・1つで予想
    feats = {"3_tree": [p, R["U"]], "4_tree+MC": [p, R["U"], R["M"], C], "5_tree+MCmf": [p, R["U"], R["Mmf"], Cmf], "6_tree+viol": [p, R["U"], vc]}
    sc = {"1_p": np.abs(p - 0.5), "2_gemma": np.abs(R["U"])}
    for name, fs in feats.items():
        Xf = np.stack([x for x in fs], -1); pred = np.zeros((E, K))
        for f in range(5):
            te = folds[f]; tr = np.concatenate([folds[g] for g in range(5) if g != f])
            m = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(Xf[tr].reshape(-1, Xf.shape[-1]), ok[tr].ravel())
            pred[te] = m.predict_proba(Xf[te].reshape(-1, Xf.shape[-1]))[:, 1].reshape(len(te), K)
        sc[name] = pred
    res = {k: aurc(v.ravel(), ok.ravel()) for k, v in sc.items()}
    np.savez_compressed(f"E161_arrays_s{seed}.npz", p=p, M=R["M"], C=C, U=R["U"], pred3=sc["3_tree"], pred4=sc["4_tree+MC"])
    # 常識の700個（2＝どちらとも→正しい扱い）
    sy = np.array([1.0 if m >= 1 else 0.0 for _, _, m in samp]); idx = [(e, ci[c]) for e, c, _ in samp]
    okc = np.array([float((p[e, c] >= 0.5) == y) for (e, c), y in zip(idx, sy)])
    res_cs = {k: aurc(np.array([v[e, c] for e, c in idx]), okc) for k, v in sc.items()}
    out["seeds"][seed] = dict(meta=meta, types=typ, aurc=res, aurc_common=res_cs, acc_net=float(ok.mean()), acc_gemma=float(((LG > 0) == Y).mean()))
    print(seed, json.dumps(out["seeds"][seed]["aurc"]), json.dumps(typ), flush=True)
S = out["seeds"]
d43 = [S[s]["aurc"]["3_tree"] - S[s]["aurc"]["4_tree+MC"] for s in S]
d45 = [S[s]["aurc"]["5_tree+MCmf"] - S[s]["aurc"]["4_tree+MC"] for s in S]
out["rule"] = dict(gain_MC=d43, works=sum(d >= 0.02 for d in d43) >= 4, gain_vs_meanfield=d45, pbit_specific=sum(d >= 0.02 for d in d45) >= 4)
print(json.dumps(out["rule"]))
json.dump(out, open(sys.argv[4], "w"), ensure_ascii=False, indent=1)
