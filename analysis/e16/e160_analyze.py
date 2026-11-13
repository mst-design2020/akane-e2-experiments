"""E16-0 の分析。python3 e160_analyze.py e160_items.json E160_results.json.gz out.json
決まり（PLAN_E16-0.md で先に固定）：差＝前向きの正しさ−逆向きの正しさ（4択、4つの位置の平均）。
物ごとに引き直す2,000回の幅（95%）。難しい版・やさしい版のどちらかで 差≥10ポイント かつ 幅の下が0より上 → 続ける（E16-1）。
両方で 差<5ポイント → やめる。それ以外 → 相談。"""
import json, sys, gzip
import numpy as np
items = json.load(open(sys.argv[1]))["items"]
R = json.load(gzip.open(sys.argv[2], "rt"))
acc = {}; pc = {}
for m in R["mc"]:
    k = (m["d"], m["lv"]); lp = np.array(m["lp"])
    acc.setdefault(k, {}).setdefault(m["i"], []).append(int(lp.argmax() == m["pos"]))
    pc.setdefault(k, {}).setdefault(m["i"], []).append(float(np.exp(lp[m["pos"]])))
ent_of = [it["ent"] for it in items]; ents = sorted(set(ent_of))
idx = {e: [i for i, x in enumerate(ent_of) if x == e] for e in ents}
rng = np.random.default_rng(0)
out = {}
for lv in ("easy", "hard"):
    f = np.array([np.mean(acc[("f", lv)][i]) for i in range(len(items))])
    r = np.array([np.mean(acc[("r", lv)][i]) for i in range(len(items))])
    d = f - r; B = []
    for _ in range(2000):
        s = rng.choice(len(ents), len(ents)); ii = [j for e in s for j in idx[ents[e]]]
        B.append(d[ii].mean())
    lo, hi = np.percentile(B, [2.5, 97.5])
    cats = sorted({it["cat"] for it in items})
    macro = np.mean([d[[i for i, it in enumerate(items) if it["cat"] == c]].mean() for c in cats])
    pf = np.mean([np.mean(pc[("f", lv)][i]) for i in range(len(items))]); pr = np.mean([np.mean(pc[("r", lv)][i]) for i in range(len(items))])
    out[lv] = dict(forward=round(f.mean(), 3), reverse=round(r.mean(), 3), gap=round(d.mean(), 3), ci=[round(lo, 3), round(hi, 3)],
                   gap_by_category_mean=round(macro, 3), prob_forward=round(pf, 3), prob_reverse=round(pr, 3),
                   both_right=round(np.mean((f == 1) & (r == 1)), 3), only_forward=round(np.mean((f == 1) & (r < 1)), 3), only_reverse=round(np.mean((f < 1) & (r == 1)), 3))
    print(lv, out[lv])
G = [out[l]["gap"] for l in out]
go = any(out[l]["gap"] >= 0.10 and out[l]["ci"][0] > 0 for l in out)
stop = all(g < 0.05 for g in G)
out["decision"] = "続ける（E16-1）" if go else ("やめる" if stop else "相談")
out["open"] = dict(forward_ok=round(float(np.mean(R["f_ok"])), 3), reverse_ok=round(float(np.mean([v for L in R["r_ok"] for v in L])), 3),
                   reverse_n=round(float(np.mean([len(L) for L in R["r_list"]])), 1))
print("決まりによる判断:", out["decision"]); print("自由回答（参考）:", out["open"])
json.dump(out, open(sys.argv[3], "w"), ensure_ascii=False, indent=1)
