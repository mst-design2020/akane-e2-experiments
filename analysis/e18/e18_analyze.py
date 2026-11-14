"""E18 の比べ（PLAN_E18.md、決まりは流す前に固定）。
python3 e18_analyze.py c18.json hyperlex-data.zip out.json
c18.json：ノートの最後の値（items=[set, type, cid, X, Y]、G2, G2R, G9, T, has, plen）。
しきい値は HyperLex の学習＋確かめ（人の点 3 以上＝一種）で、釣り合った正しさが一番の所。試すのは BLESS だけ。"""
import sys, json, zipfile, re, collections, numpy as np

NB, MID, MARGIN, Q = 2000, 3.0, 0.03, np.arange(1, 100)
NEG = ("mero", "rev", "coord", "random-n")


def hl_tune(zp):
    z = zipfile.ZipFile(zp)
    def rows(pat):
        n = [n for n in z.namelist() if re.search(pat, n)][0]
        return [l.split() for l in z.read(n).decode("utf-8").splitlines()[1:] if l.strip()]
    hum = {(r[0], r[1]): float(r[4]) for r in rows(r"hyperlex-nouns\.txt$")}
    tune = set()
    for pat in (r"random/hyperlex_training_all_random\.txt$", r"random/hyperlex_dev_all_random\.txt$"):
        tune |= {(r[0], r[1]) for r in rows(pat) if r[2] == "N"}
    return hum, tune


def ba(yes, y):
    return 0.5 * (yes[y == 1].mean() + (1 - yes[y == 0]).mean())


def cands(v):
    return np.unique(np.percentile(v, Q))


def fit(rule, cols, y):
    """rule(params) -> yes（bool の並び）。cols は各パラメータの候補。HyperLex で一番の物（同点は最初）。"""
    best, bp = -1, None
    for p in np.array(np.meshgrid(*cols, indexing="ij")).reshape(len(cols), -1).T:
        v = ba(rule(*p), y)
        if v > best: best, bp = v, tuple(float(a) for a in p)
    return bp, round(float(best), 4)


def bl_ba(yes, typ):
    return 0.5 * yes[typ == "hyper"].mean() + 0.5 * np.mean([1 - yes[typ == t].mean() for t in NEG])


def boot(yes_a, yes_b, typ, cid, seed):
    rng = np.random.default_rng(seed); u = np.unique(cid); idx = {c: np.where(cid == c)[0] for c in u}; d = []
    for _ in range(NB):
        i = np.concatenate([idx[c] for c in rng.choice(u, len(u))])
        try: d.append(bl_ba(yes_a[i], typ[i]) - bl_ba(yes_b[i], typ[i]))
        except Exception: pass
    d = np.array(d); d = d[~np.isnan(d)]
    return {"diff": round(float(bl_ba(yes_a, typ) - bl_ba(yes_b, typ)), 4),
            "lo": round(float(np.percentile(d, 2.5)), 4), "hi": round(float(np.percentile(d, 97.5)), 4)}


def flag_ab(d):
    if d["diff"] >= MARGIN and d["lo"] > 0: return "効く"
    if d["diff"] < 0 and d["hi"] < 0: return "悪くなる"
    return "同着"


def main(cp, zp, op):
    c = json.load(open(cp, encoding="utf-8"))
    it = c["items"]; S = np.array([r[0] for r in it]); typ = np.array([r[1] for r in it]); cid = np.array([r[2] for r in it])
    G2, G2R, G9, T, has = (np.array(c[k], float) for k in ("G2", "G2R", "G9", "T", "has"))
    hum, tune = hl_tune(zp)
    m = np.array([s == "hl" and (r[3], r[4]) in tune for s, r in zip(S, it)])
    y = np.array([int(hum[(r[3], r[4])] >= MID) for r, k in zip(it, m) if k])
    D = G2 - G2R
    R = {"G2": (lambda a: G2 > a, [G2]), "G9": (lambda a: G9 > a, [G9]),
         "D": (lambda a, b: (G2 > a) & (D > b), [G2, D]),
         "V": (lambda a, b: (G2 > a) & ((has == 0) | (T >= b)), [G2, T])}
    out = {"n_tune": int(m.sum()), "tune_pos": int(y.sum()), "bless": dict(collections.Counter(typ[S == "bl"].tolist())), "params": {}, "tune_ba": {}}
    YES = {}
    for k, (rule, srcs) in R.items():
        cols = [cands(v[m]) for v in srcs]
        sub = lambda *p, rule=rule: rule(*p)[m]
        p, b = fit(sub, cols, y)
        out["params"][k], out["tune_ba"][k] = p, b
        YES[k] = rule(*p).astype(float)
    b = S == "bl"; tb, cb = typ[b], cid[b]
    out["bless_ba"] = {k: round(float(bl_ba(v[b], tb)), 4) for k, v in YES.items()}
    out["bless_yes_rate"] = {k: {t: round(float(v[b][tb == t].mean()), 3) for t in ("hyper",) + NEG} for k, v in YES.items()}
    out["bless_x_in_dict"] = round(float(has[b].mean()), 3)
    dA = boot(YES["V"][b], YES["G2"][b], tb, cb, 1); out["flag_A_V_vs_G2"] = {**dA, "flag": flag_ab(dA)}
    dB = boot(YES["V"][b], YES["D"][b], tb, cb, 2); out["flag_B_V_vs_D"] = {**dB, "flag": flag_ab(dB)}
    dC = boot(YES["V"][b], YES["G9"][b], tb, cb, 3)
    out["flag_C_V_vs_G9"] = {**dC, "flag": "9Bより上" if dC["lo"] > 0 else ("9Bに届かない" if dC["hi"] < 0 else "9Bと同着")}
    out["ref_D_vs_G2"] = boot(YES["D"][b], YES["G2"][b], tb, cb, 4)
    json.dump(out, open(op, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
