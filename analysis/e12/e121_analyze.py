"""E12-1 分析。python3 e121_analyze.py items_all.json E12_results.json.gz E12-1_results.json.gz out.json"""
import sys, json, gzip
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from e12_analyze import hit, boot_diff
from e121_core import overlap
import gzip as _gz
jl = lambda p: __import__("json").load(_gz.open(p, "rt") if str(p).endswith(".gz") else open(p))   # .json と .json.gz の両方


def main(items_p, r12_p, r121_p, out_p):
    it = jl(items_p)["items"]
    R = json.load(gzip.open(r12_p, "rt")); Q = json.load(gzip.open(r121_p, "rt"))
    idx = Q["idx"]; assert idx == R["g0_idx"]
    pos = {i: k for k, i in enumerate(range(len(it)))}
    X = [it[i] for i in idx]; gold = [x["gold"] for x in X]; head = [x["head"] for x in X]
    A = np.array([hit(R["gemma"][i], g) for i, g in zip(idx, gold)])
    B = np.array([hit(p, g) for p, g in zip(Q["gemma_B"], gold)])
    C = np.array([hit(p, g) for p, g in zip(Q["gemma_C"], gold)])
    F = np.array([float(g == 0) for g in gold])
    S = np.array([hit(R["sim"][i], g) for i, g in zip(idx, gold)])
    mem = np.array([np.nan if m is None else m for m in Q["mem"]])
    out = {"n": len(idx), "acc": {"A 元のまま": A.mean(), "B 名前を隠す": B.mean(), "C 名前を隠し言い換え": C.mean(),
                                  "最初の意味": F.mean(), "近さ": S.mean()}}
    out["acc"] = {k: round(float(v), 4) for k, v in out["acc"].items()}

    def d(a, b, m=None):
        m = np.ones(len(a), bool) if m is None else m
        x, ci = boot_diff(a[m], b[m], [h for h, k in zip(head, m) if k])
        return {"diff": round(x, 4), "ci95": [round(c, 4) for c in ci], "n": int(m.sum()),
                "差あり": bool(abs(x) >= 0.05 and (ci[0] > 0 or ci[1] < 0))}
    out["diff"] = {"A − B": d(A, B), "A − C": d(A, C), "C − 最初の意味": d(C, F), "B − 最初の意味": d(B, F), "C − 近さ": d(C, S)}
    # 言い換えがちゃんと同じ意味で、頭の語も残った物だけ
    nc = Q["nli_cands"]; pc = Q["para_cands"]
    good = np.array([k and t >= .5 and all(nc[c] >= .5 for c in x["cand_defs"][:8]) for k, t, x in
                     zip(Q["para_target_keep"], Q["nli_target"], X)])
    out["言い換えが同じ意味の物だけ"] = {"割合": round(float(good.mean()), 4), "C": round(float(C[good].mean()), 4),
                               "最初の意味": round(float(F[good].mean()), 4), "C − 最初の意味": d(C, F, good), "A − C": d(A, C, good)}
    out["言い換えの変わり方"] = {"説明文の語の重なり(平均)": round(float(np.mean([overlap(x["defn"], p) for x, p in zip(X, Q["para_target"])])), 4),
                          "候補の語の重なり(平均)": round(float(np.mean([overlap(c, pc[c]) for c in pc])), 4),
                          "頭の語が残った割合": round(float(np.mean(Q["para_target_keep"])), 4)}
    ok = ~np.isnan(mem)
    out["D 覚えている度合い"] = {"測れた数": int(ok.sum()), "平均": round(float(np.nanmean(mem)), 4)}
    for name, m in (("ほぼ覚えていない(0.2未満)", ok & (mem < 0.2)), ("中(0.2〜0.5)", ok & (mem >= .2) & (mem < .5)),
                    ("よく覚えている(0.5以上)", ok & (mem >= .5))):
        out["D 覚えている度合い"][name] = {"n": int(m.sum()), "A": round(float(A[m].mean()), 4), "C": round(float(C[m].mean()), 4),
                                    "最初の意味": round(float(F[m].mean()), 4), "A − 最初の意味": d(A, F, m)}
    json.dump(out, open(out_p, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
