"""E4-0 の結果を条件ごとにまとめる。python3 e40_analyze.py E40_results.json"""
import json, sys
from collections import defaultdict
import numpy as np

D = json.load(open(sys.argv[1]))
M = ["1_conf", "2_count", "3_learn", "4_bp", "5_pbit_100", "5_pbit_1000", "5_pbit_10000"]
by = defaultdict(list)
for r in D["big"]:
    by[(r["d"], r["eps"], r["trust"])].append(r["res"])
sm = defaultdict(list)
for r in D["small"]:
    sm[(r["d"], r["eps"], r["trust"])].append(r)


def judge(a, b):
    """a が b より良い(AURC小)か。差0.02以上かつ5回中4回以上同じ向き。"""
    a, b = np.array(a), np.array(b)
    diff = b - a
    if abs(diff.mean()) >= 0.02 and max((diff > 0).sum(), (diff < 0).sum()) >= 4:
        return "上" if diff.mean() > 0 else "下"
    return "同着"


rows = []
print("d eps trust | " + " ".join(m.replace("5_pbit_", "pb") for m in M) + " | BP収束 pb詰まり | 焼き | BP対pb1万 BP対学習 BP対自信 | 小網MAE bp pb1万")
for k in sorted(by):
    R = by[k]
    au = {m: [x[m]["aurc"] for x in R] for m in M}
    mean = {m: float(np.mean(v)) for m, v in au.items()}
    conv = float(np.mean([x["4_bp"]["converged"] for x in R]))
    stuck = int(sum(x["5_pbit_10000"]["aurc"] > 0.5 for x in R))
    ann = float(np.mean([x["6_anneal_acc100"] for x in R]))
    S = sm[k]
    mae = {m: float(np.mean([s["mae_vs_exact"][m] for s in S])) for m in S[0]["mae_vs_exact"]}
    sconv = float(np.mean([s["bp_conv_rate"] for s in S]))
    j = dict(bp_vs_pbit=judge(au["4_bp"], au["5_pbit_10000"]),
             bp_vs_learn=judge(au["4_bp"], au["3_learn"]),
             bp_vs_conf=judge(au["4_bp"], au["1_conf"]),
             pbit_vs_learn=judge(au["5_pbit_10000"], au["3_learn"]))
    rows.append(dict(d=k[0], eps=k[1], trust=k[2], aurc=mean, bp_conv=conv, pbit_stuck_runs=stuck,
                     anneal_acc100=ann, judge=j, small_mae=mae, small_bp_conv=sconv,
                     acc100={m: float(np.mean([x[m]["acc"]["1.0"] for x in R])) for m in M}))
    print(k, " ".join(f"{mean[m]:.3f}" for m in M), f"| {conv:.1f} {stuck} | {ann:.3f} |",
          j["bp_vs_pbit"], j["bp_vs_learn"], j["bp_vs_conf"], f"| {mae['4_bp']:.3f} {mae['5_pbit_10000']:.3f} (小網BP収束{sconv:.2f})")
json.dump(rows, open(sys.argv[1].replace(".json", "_analysis.json"), "w"), indent=1, ensure_ascii=False)
