"""E12 分析。python3 e12_analyze.py items_all.json local.json E12_results.json.gz out.json
同点はでたらめに割る扱い（同点の中に正解があれば 1/同点数 を当たりとする）。
差の幅は「頭の語」ごとにまとめて引き直す（同じ頭の語が何度も出るため）、1000回。"""
import sys, json, gzip, collections
import numpy as np
import gzip as _gz
jl = lambda p: __import__("json").load(_gz.open(p, "rt") if str(p).endswith(".gz") else open(p))   # .json と .json.gz の両方


def hit(scores, gold):
    s = np.asarray(scores, float)
    top = np.flatnonzero(s >= s.max() - 1e-12)
    return (gold in top) / len(top)


def boot_diff(a, b, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    gi = collections.defaultdict(list)
    for i, g in enumerate(groups):
        gi[g].append(i)
    keys = list(gi)
    sa = np.array([np.sum(a[gi[k]]) for k in keys]); sb = np.array([np.sum(b[gi[k]]) for k in keys])
    cnt = np.array([len(gi[k]) for k in keys])
    ds = []
    for _ in range(n):
        p = rng.integers(0, len(keys), len(keys))
        ds.append((sa[p].sum() - sb[p].sum()) / cnt[p].sum())
    return float(np.mean(a) - np.mean(b)), [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]


def main(items_p, local_p, res_p, out_p):
    it = jl(items_p)["items"]
    L = jl(local_p)
    R = json.load(gzip.open(res_p, "rt"))
    assert R["ids"] == [x["id"] for x in it] and [x["id"] for x in L] == R["ids"]
    rng = np.random.default_rng(0)
    M = {}
    M["S0 でたらめ"] = np.array([1 / x["n"] for x in it])
    M["S1 最初の意味"] = np.array([float(x["gold"] == 0) for x in it])
    M["S2 Lesk"] = np.array([hit(l["lesk"], x["gold"]) for x, l in zip(it, L)])
    M["S3 近さ"] = np.array([hit(s, x["gold"]) for x, s in zip(it, R["sim"])])
    M["G Gemma"] = np.array([hit(p, x["gold"]) for x, p in zip(it, R["gemma"])])
    M["参考T WordNetの木"] = np.array([hit(l["tree"], x["gold"]) for x, l in zip(it, L)])
    conf = np.array([max(p) for p in R["gemma"]])
    kind = np.array([x["kind"] for x in it]); head = [x["head"] for x in it]
    gold0 = np.array([x["gold"] == 0 for x in it]); nS = np.array([x["n"] for x in it])
    out = {"n": {}, "acc": {}, "diff": {}, "sub": {}}

    def block(mask, name):
        idx = np.flatnonzero(mask)
        out["n"][name] = int(len(idx))
        out["acc"][name] = {k: round(float(v[idx].mean()), 4) for k, v in M.items()}
        # 頭の語ごとに平均してから平均（よく出る頭の語に引っぱられない見方）
        byh = collections.defaultdict(list)
        for i in idx:
            byh[head[i]].append(i)
        out["acc"][name + "（頭の語ごと平均）"] = {k: round(float(np.mean([v[ii].mean() for ii in byh.values()])), 4) for k, v in M.items()}
        g = [head[i] for i in idx]
        out["diff"][name] = {}
        for other in ("S1 最初の意味", "S2 Lesk", "S3 近さ"):
            d, ci = boot_diff(M["G Gemma"][idx], M[other][idx], g)
            out["diff"][name]["G − " + other] = {"diff": round(d, 4), "ci95": [round(c, 4) for c in ci],
                                                 "差あり": bool(abs(d) >= 0.05 and (ci[0] > 0 or ci[1] < 0))}

    block(kind == "head", "主：頭の語")
    block(kind == "of", "副：「〜の」の後ろの語")
    block((kind == "head") & ~gold0, "副：頭の語・正解が最初の意味でない")
    block((kind == "head") & gold0, "副：頭の語・正解が最初の意味")
    for k in sorted(set(nS)):
        idx = np.flatnonzero((kind == "head") & (nS == k))
        out["sub"][f"意味の数{k}"] = {"n": int(len(idx)), **{m: round(float(M[m][idx].mean()), 4) for m in ("S1 最初の意味", "G Gemma")}}
    h = kind == "head"
    for lo, hi in ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
        idx = np.flatnonzero(h & (conf >= lo) & (conf < hi))
        out["sub"][f"Gemmaの自信{lo}〜{min(hi,1.0)}"] = {"n": int(len(idx)), "割合": round(len(idx) / h.sum(), 4),
                                                   "Gemma正しさ": round(float(M["G Gemma"][idx].mean()), 4) if len(idx) else None}
    g0 = R["g0_idx"]
    out["sub"]["参考 WordNetの順のまま聞く（5000問）"] = {
        "Gemma（順のまま）": round(float(np.mean([hit(p, it[i]["gold"]) for i, p in zip(g0, R["gemma_wnorder"])])), 4),
        "Gemma（並べ替え）": round(float(M["G Gemma"][g0].mean()), 4),
        "最初の意味": round(float(M["S1 最初の意味"][g0].mean()), 4),
        "Gemmaが1番目を選んだ割合（順のまま）": round(float(np.mean([int(np.argmax(p)) == 0 for p in R["gemma_wnorder"]])), 4)}
    json.dump(out, open(out_p, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
