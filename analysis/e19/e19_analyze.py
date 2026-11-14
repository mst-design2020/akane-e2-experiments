"""E19 の比べ（PLAN_E19.md、決まりは流す前に固定）。
python3 e19_analyze.py c19.json out.json [cs.json]
c19.json：items=[[段, X, Y, zipf]], gold（WordNet 0/1）, G2, G9, S, T。
cs.json（任意）：{"X|Y": 0/1} 常識の判定。あれば正解をそれで置き換えてもう一度比べる。
しきい値は「よく出る」段で、釣り合った正しさが一番の所。"""
import sys, json, numpy as np

NB, MARGIN, Q = 2000, 0.03, np.arange(1, 100)
BUCKETS = ("common", "mid", "rare")


def ba(yes, y):
    return 0.5 * (yes[y == 1].mean() + (1 - yes[y == 0]).mean())


def cands(v):
    return np.unique(np.percentile(v, Q))


def fit(rule, cols, y):
    best, bp = -1, None
    for p in np.array(np.meshgrid(*cols, indexing="ij")).reshape(len(cols), -1).T:
        v = ba(rule(*p), y)
        if v > best: best, bp = v, tuple(float(a) for a in p)
    return bp, round(float(best), 4)


def boot(ya, yb, y, cid, seed):
    rng = np.random.default_rng(seed); u = np.unique(cid); idx = {c: np.where(cid == c)[0] for c in u}; d = []
    for _ in range(NB):
        i = np.concatenate([idx[c] for c in rng.choice(u, len(u))])
        if y[i].min() == y[i].max(): continue
        d.append(ba(ya[i], y[i]) - ba(yb[i], y[i]))
    d = np.array(d)
    return {"diff": round(float(ba(ya, y) - ba(yb, y)), 4),
            "lo": round(float(np.percentile(d, 2.5)), 4), "hi": round(float(np.percentile(d, 97.5)), 4)}


def flag_ab(d):
    if d["diff"] >= MARGIN and d["lo"] > 0: return "効く"
    if d["diff"] < 0 and d["hi"] < 0: return "悪くなる"
    return "同着"


def flag_c(d):
    if d["diff"] >= MARGIN and d["lo"] > 0: return "9Bより上"
    if d["diff"] >= -MARGIN and d["hi"] > 0: return "9Bに届く"
    return "届かない"


def run(c, y):
    b = np.array([it[0] for it in c["items"]]); X = np.array([it[1] for it in c["items"]])
    G2, G9, S, T = (np.array(c[k], float) for k in ("G2", "G9", "S", "T"))
    tm = b == "common"
    P = {}
    P["G2"] = fit(lambda t: G2[tm] > t, [cands(G2[tm])], y[tm])
    P["G9"] = fit(lambda t: G9[tm] > t, [cands(G9[tm])], y[tm])
    P["S"] = fit(lambda t: S[tm] > t, [cands(S[tm])], y[tm])
    P["T"] = fit(lambda t: T[tm] >= t, [cands(T[tm])], y[tm])
    P["TO"] = fit(lambda t, u: (G2[tm] > t) | (T[tm] >= u), [cands(G2[tm]), cands(T[tm])], y[tm])
    YES = {"G2": G2 > P["G2"][0][0], "G9": G9 > P["G9"][0][0], "S": S > P["S"][0][0], "T": T >= P["T"][0][0],
           "TO": (G2 > P["TO"][0][0]) | (T >= P["TO"][0][1])}
    out = {"params": {k: {"p": v[0], "ba_tune": v[1]} for k, v in P.items()}, "by_bucket": {}, "flags": {}}
    for bk in BUCKETS:
        m = b == bk
        out["by_bucket"][bk] = {"n_pairs": int(m.sum()), "n_words": int(len(set(X[m]))), "n_yes_gold": int(y[m].sum()),
                                **{k: {"ba": round(float(ba(v[m], y[m])), 4), "yes_on_true": round(float(v[m][y[m] == 1].mean()), 4),
                                       "no_on_false": round(float(1 - v[m][y[m] == 0].mean()), 4)} for k, v in YES.items()}}
        dA = boot(YES["TO"][m], YES["G2"][m], y[m], X[m], 1)
        dB = boot(YES["TO"][m], YES["S"][m], y[m], X[m], 2)
        dC = boot(YES["TO"][m], YES["G9"][m], y[m], X[m], 3)
        dS = boot(YES["S"][m], YES["G2"][m], y[m], X[m], 4)
        out["flags"][bk] = {"A_TO_vs_G2": {**dA, "flag": flag_ab(dA)}, "B_TO_vs_S": {**dB, "flag": flag_ab(dB)},
                            "C_TO_vs_G9": {**dC, "flag": flag_c(dC)}, "ref_S_vs_G2": {**dS, "flag": flag_ab(dS)}}
    return out, YES


def main(cp, op, csp=None):
    c = json.load(open(cp))
    y = np.array(c["gold"], int)
    res, YES = run(c, y)
    out = {"wordnet": res}
    # 常識で判定する組（珍しい段で、比べる2つの答えが違う組。各比べ最大1,000組、種0）
    b = np.array([it[0] for it in c["items"]]); rng = np.random.default_rng(0); todo = {}
    for other in ("G2", "S", "G9"):
        i = np.where((b == "rare") & (YES["TO"] != YES[other]))[0]
        if len(i) > 1000: i = np.sort(rng.choice(i, 1000, replace=False))
        out.setdefault("cs_counts", {})[other] = int(len(i))
        for j in i: todo[f'{c["items"][j][1]}|{c["items"][j][2]}'] = None
    out["cs_todo"] = sorted(todo)
    if csp:
        cs = json.load(open(csp))
        y2 = y.copy()
        for j, it in enumerate(c["items"]):
            k = f"{it[1]}|{it[2]}"
            if k in cs: y2[j] = int(cs[k])
        out["common_sense"] = run(c, y2)[0]
        out["cs_n"] = len(cs)
    json.dump(out, open(op, "w"), ensure_ascii=False, indent=1)
    for name in ("wordnet", "common_sense"):
        if name not in out: continue
        r = out[name]; print("==", name)
        for bk in BUCKETS:
            print(bk, {k: v["ba"] for k, v in r["by_bucket"][bk].items() if isinstance(v, dict)})
            print("  ", {k: (v["diff"], v["lo"], v["hi"], v["flag"]) for k, v in r["flags"][bk].items()})
    print("常識で判定する組:", len(out["cs_todo"]), out["cs_counts"])
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
