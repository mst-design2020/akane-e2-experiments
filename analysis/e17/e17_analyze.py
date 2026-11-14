"""E17 の比べ（PLAN_E17.md、決まりは流す前に固定）。
python3 e17_analyze.py E17_results.json.gz hyperlex-data.zip out.json
- T（TADORI）＝道の強さの掛け算 × 着き（NLI）の一番大きい物（E15 の n_nli と同じ）。点の語が Y そのものなら着き 1。
- S＝X の辞書の見出しがあれば T、無ければ σ(G2)。
- 人の点（0〜6）は HyperLex の原本の zip から読む（結果のファイルにも入っているが、分け方と種類は zip から）。"""
import sys, os, json, gzip, zipfile, collections, re, numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingRegressor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "e15"))
import e15_core as F

NB, SEEDS, MID, MARGIN = 2000, range(5), 3.0, 0.05
sig = lambda v: 1 / (1 + np.exp(-np.asarray(v, float)))


def read_zip(zp):
    z = zipfile.ZipFile(zp)
    def rows(pat):
        n = [n for n in z.namelist() if re.search(pat, n)][0]
        return [l.split() for l in z.read(n).decode("utf-8").splitlines()[1:] if l.strip()]
    nouns = rows(r"hyperlex-nouns\.txt$")
    split = {}
    for part, pat in (("train", r"random/hyperlex_training_all_random\.txt$"), ("dev", r"random/hyperlex_dev_all_random\.txt$"),
                      ("test", r"random/hyperlex_test_all_random\.txt$")):
        for r in rows(pat):
            if r[2] == "N": split[(r[0], r[1])] = part
    return nouns, split


def tscore(res):
    L = res["L"]
    CH = {(w, k): v for w, k, v in res["CH"]["n"]}
    ST = {(w, k): tuple(v) for w, k, *v in res["ST"]["n"]}
    AR = {(w, k, C): v[0] for w, k, C, *v in res["AR"]}
    T, has, plen = [], [], []
    for it in res["items"]:
        x, C = it["ent"], it["catw"]
        best = F.reach(L, {"n": CH}, {"n": ST}, "n", x)
        arr = {n: (1.0 if n[0] == C else AR.get((n[0], n[1], C), 0.0)) for n in best}
        v = {n: best[n][0] * arr[n] for n in best}
        T.append(max(v.values()) if v else 0.0)
        has.append(int(bool(L.get(x))))
        plen.append(len(best[max(v, key=v.get)][1]) - 1 if v else -1)
    return np.array(T), np.array(has), np.array(plen)


def rho(a, b):
    return float(spearmanr(a, b).correlation)


def boot_diff(h, a, b, seed=0):
    """ρ(a) − ρ(b) と、組を引き直した 95% の幅。"""
    rng = np.random.default_rng(seed); n = len(h); d = []
    for _ in range(NB):
        i = rng.integers(0, n, n); d.append(rho(h[i], a[i]) - rho(h[i], b[i]))
    return {"diff": round(rho(h, a) - rho(h, b), 4), "lo": round(float(np.percentile(d, 2.5)), 4), "hi": round(float(np.percentile(d, 97.5)), 4)}


def flag_ab(d):
    if d["diff"] >= MARGIN and d["lo"] > 0: return "効く"
    if d["diff"] < 0 and d["hi"] < 0: return "悪くなる"
    return "同着"


def flag_c(d):
    return "9Bより上" if d["lo"] > 0 else ("9Bに届かない" if d["hi"] < 0 else "9Bと同着")


def learn(X, y, tr, te):
    p = np.zeros(te.sum())
    for s in SEEDS:
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=s).fit(X[tr], y[tr])
        p += m.predict(X[te]) / len(SEEDS)
    return p


def main(rp, zp, op):
    res = json.load(gzip.open(rp, "rt", encoding="utf-8") if rp.endswith(".gz") else open(rp, encoding="utf-8"))
    nouns, split = read_zip(zp)
    if "T" in res:   # v2：ノートの最後のセルに印字した値（T などはノートの中で同じ式で計算済み）
        n = len(res["T"])
    else:
        its = res["items"]; n = len(its)
        assert all(its[i]["ent"] == nouns[i][0] and its[i]["catw"] == nouns[i][1] for i in range(n))
    h = np.array([float(r[4]) for r in nouns[:n]])
    typ = [re.sub(r"-\d+$", "", r[3]) if r[3].startswith("r-hyp") else r[3] for r in nouns[:n]]
    part = np.array([split.get((r[0], r[1]), "none") for r in nouns[:n]])
    G2, G9 = np.array(res["G2"], float), np.array(res["G9"], float)
    if "T" in res:
        T, has, plen = np.array(res["T"], float), np.array(res["has"]), np.array(res["plen"])
    else:
        T, has, plen = tscore(res)
    S = np.where(has == 1, T, sig(G2))
    out = {"n": n, "coverage": {"x_in_dict": int(has.sum()), "T_pos": int((T > 0).sum())},
           "split": dict(collections.Counter(part.tolist()))}
    M = {"G2": G2, "G9": G9, "T": T, "S": S}
    WN = res.get("WN")
    if WN and any(v[1] is not None for v in WN):
        M["WN_ref"] = np.array([v[1] if v[1] is not None else 0.0 for v in WN])
    yb = (h >= MID).astype(int)
    out["rho_all"] = {k: round(rho(h, v), 4) for k, v in M.items()}
    out["auroc_all"] = {k: round(float(roc_auc_score(yb, v)), 4) for k, v in M.items()} if 0 < yb.sum() < n else None
    out["rho_x_in_dict"] = {k: round(rho(h[has == 1], v[has == 1]), 4) for k, v in M.items()}
    # 旗A・旗C（学習なし、全部の組）
    dA = boot_diff(h, S, G2, 1); out["flag_A"] = {**dA, "flag": flag_ab(dA)}
    dC1 = boot_diff(h, S, G9, 2); out["flag_C_S_vs_G9"] = {**dC1, "flag": flag_c(dC1)}
    out["ref_T_vs_G2"] = boot_diff(h, T, G2, 3)
    # 学習あり（学習＋確かめ → 試験）
    tr, te = np.isin(part, ["train", "dev"]), part == "test"
    if tr.sum() > 20 and te.sum() > 20:
        P = {"LG2": learn(G2[:, None], h, tr, te),
             "LG2T": learn(np.c_[G2, T, has, plen], h, tr, te),
             "LG9": learn(G9[:, None], h, tr, te)}
        ht = h[te]
        out["rho_test"] = {k: round(rho(ht, v), 4) for k, v in P.items()}
        dB = boot_diff(ht, P["LG2T"], P["LG2"], 4); out["flag_B"] = {**dB, "flag": flag_ab(dB)}
        dC2 = boot_diff(ht, P["LG2T"], P["LG9"], 5); out["flag_C_LG2T_vs_LG9"] = {**dC2, "flag": flag_c(dC2)}
    # 種類ごとの平均（見るだけ）
    by = collections.defaultdict(list)
    for i, t in enumerate(typ): by[t].append(i)
    out["by_type"] = {t: {"n": len(ix), "human": round(float(h[ix].mean()), 2), "S": round(float(S[ix].mean()), 3),
                          "T": round(float(T[ix].mean()), 3), "sigG2": round(float(sig(G2[ix]).mean()), 3),
                          "sigG9": round(float(sig(G9[ix]).mean()), 3)} for t, ix in sorted(by.items())}
    json.dump(out, open(op, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in out if k != "by_type"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
