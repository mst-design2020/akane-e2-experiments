"""E20 の比べ（PLAN_E20.md、決まりは流す前に固定）。
python3 e20_analyze.py c20.json out.json [cs.json]
c20.json：items=[[src, 段, X, Y, zipf, lit, split]]（src は W か E、lit は説明文に Y が書いてあれば 1、split は E の tune/test）、
gold（0/1）、M={AI名: {"G":[], "S":[], "T":[]}}、G9=[]。
cs.json（任意）：{"src|X|Y": 0/1} 常識の判定。あれば正解をそれで置き換え、しきい値も決め直してもう一度比べる。"""
import sys, json, numpy as np

NB, MARGIN, M3, Q = 2000, 0.03, 0.01, np.arange(1, 100)
Q19 = np.arange(5, 100, 5)
BUCKETS = ("common", "mid", "rare")


def ba(yes, y):
    return 0.5 * (yes[y == 1].mean() + (1 - yes[y == 0]).mean())


def cands(v, q=Q):
    return np.unique(np.percentile(v, q))


def fit1(v, y, ge=False):
    best, bp = -1, None
    for t in cands(v):
        b = ba(v >= t if ge else v > t, y)
        if b > best: best, bp = b, float(t)
    return bp


def fit_or(a, t, y):
    best, bp = -1, None
    for th in cands(a):
        ya = a > th
        for ta in cands(t):
            b = ba(ya | (t >= ta), y)
            if b > best: best, bp = b, (float(th), float(ta))
    return bp


def fit_sa(a, t, y):
    best, bp = -1, None
    for th in cands(a, Q19):
        ya = a > th; d = np.abs(a - th)
        for ta in cands(t, Q19):
            yt = t >= ta
            for m in cands(d, Q19):
                b = ba(ya | ((d <= m) & yt), y)
                if b > best: best, bp = b, (float(th), float(ta), float(m))
    return bp


def boot(ya, yb, y, cid, seed):
    rng = np.random.default_rng(seed); u = np.unique(cid); idx = {c: np.where(cid == c)[0] for c in u}; d = []
    for _ in range(NB):
        i = np.concatenate([idx[c] for c in rng.choice(u, len(u))])
        if y[i].min() == y[i].max(): continue
        d.append(ba(ya[i], y[i]) - ba(yb[i], y[i]))
    d = np.array(d)
    return {"diff": round(float(ba(ya, y) - ba(yb, y)), 4),
            "lo": round(float(np.percentile(d, 2.5)), 4), "hi": round(float(np.percentile(d, 97.5)), 4)}


def flag(d):
    if d["diff"] >= MARGIN and d["lo"] > 0: return "効く"
    if d["diff"] < 0 and d["hi"] < 0: return "悪くなる"
    return "同着"


def masks(c, y):
    it = c["items"]
    src = np.array([r[0] for r in it]); b = np.array([r[1] for r in it]); lit = np.array([int(r[5]) for r in it])
    sp = np.array([r[6] for r in it])
    tune = {"W": (src == "W") & (b == "common"), "E": (src == "E") & (sp == "tune")}
    ev = {"W": src == "W", "E": (src == "E") & (sp == "test")}
    main = {s: ev[s] & ((y == 0) | (lit == 0)) for s in ev}
    return src, b, lit, tune, ev, main


def decide(c, y):
    """AIごと・src ごとにしきい値を決め、全部の組の「はい」を返す。"""
    src, b, lit, tune, ev, main = masks(c, y)
    YES, P = {}, {}
    G9 = np.array(c["G9"], float)
    for s in ("W", "E"):
        tm = tune[s]
        th = fit1(G9[tm], y[tm]); YES[(s, "9B", "G")] = G9 > th; P[f"{s}|9B|G"] = th
        for m, v in c["M"].items():
            G, S, T = (np.array(v[k], float) for k in "GST")
            pg = fit1(G[tm], y[tm]); ps = fit1(S[tm], y[tm]); pt = fit1(T[tm], y[tm], ge=True)
            pso = fit_or(S[tm], T[tm], y[tm]); pto = fit_or(G[tm], T[tm], y[tm]); psa = fit_sa(S[tm], T[tm], y[tm])
            YES[(s, m, "G")] = G > pg; YES[(s, m, "S")] = S > ps; YES[(s, m, "T")] = T >= pt
            YES[(s, m, "SO")] = (S > pso[0]) | (T >= pso[1]); YES[(s, m, "TO")] = (G > pto[0]) | (T >= pto[1])
            use = np.abs(S - psa[0]) <= psa[2]
            YES[(s, m, "SA")] = (S > psa[0]) | (use & (T >= psa[1])); YES[(s, m, "SA_use")] = use
            P[f"{s}|{m}"] = {"G": pg, "S": ps, "T": pt, "SO": pso, "TO": pto, "SA": psa}
    return YES, P


def run(c, y):
    src, b, lit, tune, ev, main = masks(c, y)
    X = np.array([r[2] for r in c["items"]])
    YES, P = decide(c, y)
    out = {"params": P, "flags": {}, "ba": {}, "count": {}}
    models = list(c["M"])
    for s in ("W", "E"):
        mm = main[s]
        out["count"][s] = {"main_pairs": int(mm.sum()), "main_yes": int(y[mm].sum()), "all_pairs": int(ev[s].sum()),
                           "all_yes": int(y[ev[s]].sum()), "lit_yes": int((y[ev[s]] * lit[ev[s]]).sum()), "words": int(len(set(X[ev[s]])))}
        for scope, base in (("main", mm), ("all", ev[s])):
            for bk in ("all",) + BUCKETS:
                m_ = base if bk == "all" else base & (b == bk)
                if m_.sum() == 0 or y[m_].min() == y[m_].max(): continue
                key = f"{s}|{scope}|{bk}"
                out["ba"][key] = {f"{m}|{k}": round(float(ba(YES[(s, m, k)][m_], y[m_])), 4)
                                  for m in models for k in ("G", "S", "T", "SO", "TO", "SA")}
                out["ba"][key]["9B|G"] = round(float(ba(YES[(s, "9B", "G")][m_], y[m_])), 4)
                out["ba"][key]["yes_on_true"] = {f"{m}|{k}": round(float(YES[(s, m, k)][m_][y[m_] == 1].mean()), 4)
                                                 for m in models for k in ("S", "SO")}
                if scope == "main" and (bk == "all" or s == "W"):
                    fl = {}
                    for i, m in enumerate(models):
                        d1 = boot(YES[(s, m, "SO")][m_], YES[(s, m, "S")][m_], y[m_], X[m_], 10 + i)
                        fl[f"{m}|SO-S"] = {**d1, "flag": flag(d1)}
                        if bk == "all":
                            d9 = boot(YES[(s, m, "SO")][m_], YES[(s, "9B", "G")][m_], y[m_], X[m_], 20 + i)
                            fl[f"{m}|SO-9B"] = {**d9, "flag": flag(d9)}
                            if s == "W":
                                d3 = boot(YES[(s, m, "SA")][m_], YES[(s, m, "SO")][m_], y[m_], X[m_], 30 + i)
                                share = float(YES[(s, m, "SA_use")][m_].mean())
                                ok = d3["diff"] >= -M3 and d3["lo"] > -MARGIN and share <= 0.5
                                fl[f"{m}|SA-SO"] = {**d3, "chain_share": round(share, 3), "flag": "迷った時だけで足りる" if ok else "足りない"}
                    n = sum(fl[f"{m}|SO-S"]["flag"] == "効く" for m in models)
                    fl["verdict"] = f"{n}/{len(models)} で効く → " + ("効く" if n >= 2 else "言えない")
                    out["flags"][key] = fl
    return out, YES


def main(cp, op, csp=None):
    c = json.load(open(cp))
    y = np.array(c["gold"], int)
    res, YES = run(c, y)
    out = {"gold": res}
    src, b, lit, tune, ev, main_ = masks(c, y)
    rng = np.random.default_rng(0); todo = []
    for s in ("W", "E"):
        diff = np.zeros(len(y), bool)
        for m in c["M"]: diff |= YES[(s, m, "SO")] != YES[(s, m, "S")]
        i = np.where(main_[s] & diff)[0]
        out.setdefault("cs_counts", {})[s] = int(len(i))
        if len(i) > 800: i = np.sort(rng.choice(i, 800, replace=False))
        todo += [f'{s}|{c["items"][j][2]}|{c["items"][j][3]}' for j in i]
    out["cs_todo"] = sorted(set(todo))
    if csp:
        cs = json.load(open(csp)); y2 = y.copy()
        for j, it in enumerate(c["items"]):
            k = f"{it[0]}|{it[2]}|{it[3]}"
            if k in cs: y2[j] = int(cs[k])
        out["common_sense"] = run(c, y2)[0]; out["cs_n"] = len(cs)
    json.dump(out, open(op, "w"), ensure_ascii=False, indent=1)
    for name in ("gold", "common_sense"):
        if name not in out: continue
        print("==", name)
        for k, v in out[name]["flags"].items():
            print(k, {kk: (vv["diff"], vv["lo"], vv["hi"], vv["flag"]) if isinstance(vv, dict) else vv for kk, vv in v.items()})
    print("常識で判定する組:", len(out["cs_todo"]), out["cs_counts"])
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
