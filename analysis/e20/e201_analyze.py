"""E20-1 (plans/E20-1_combine_and_count.md). Run from analysis/e20 after judge_combine.py:
python e201_analyze.py  -> results/E20/E20-1_out.json"""
import json, os, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e20_analyze as A
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "..", "data", "E20")
R = os.path.join(HERE, "..", "..", "results", "E20")
c = json.load(open(f"{D}/c20.json"))
it = c["items"]
keys = np.array([f"{r[0]}|{r[2]}|{r[3]}" for r in it])
gold = np.array(c["gold"])
X = np.array([r[2] for r in it])
models = list(c["M"])


def labels(name):
    if name == "gold": return gold
    cs = json.load(open(f"{R}/judge/cs_{name}.json")); y = gold.copy()
    for j, k in enumerate(keys):
        if k in cs: y[j] = cs[k]
    return y


def lr_yes(F, tm, y):
    sc = StandardScaler().fit(F[tm]); clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    clf.fit(sc.transform(F[tm]), y[tm]); p = clf.decision_function(sc.transform(F))
    th = A.fit1(p[tm], y[tm]); return p > th


out = {}
for name in ("vote", "gold", "claude", "chatgpt", "gemini"):
    y = labels(name)
    src, b, lit, tune, ev, main = A.masks(c, y)
    YES, P = A.decide(c, y)
    res = {}
    for s in ("W", "E"):
        tm = tune[s]
        for m in models:
            G, S, T = (np.array(c["M"][m][k], float) for k in "GST")
            lt = np.log(T + 1e-4)
            YES[(s, m, "LR")] = lr_yes(np.c_[S, lt], tm, y)
            YES[(s, m, "LRg")] = lr_yes(np.c_[G, S, lt], tm, y)
        for bk in ("all",) + (A.BUCKETS if s == "W" else ()):
            m_ = main[s] if bk == "all" else main[s] & (b == bk)
            key = f"{s}|{bk}"; fl = {}
            for i, m in enumerate(models):
                for a, bb, sd in (("LR", "SO", 40), ("LR", "S", 50), ("LRg", "SO", 60)):
                    d = A.boot(YES[(s, m, a)][m_], YES[(s, m, bb)][m_], y[m_], X[m_], sd + i)
                    fl[f"{m}|{a}-{bb}"] = (d["diff"], d["lo"], d["hi"], A.flag(d))
                fl[f"{m}|BA"] = {k: round(float(A.ba(YES[(s, m, k)][m_], y[m_])), 4) for k in ("S", "SO", "LR", "LRg")}
            n = sum(fl[f"{m}|LR-SO"][3] == "効く" for m in models)
            fl["verdict LR-SO"] = f"{n}/3 → " + ("効く" if n >= 2 else "言えない")
            res[key] = fl
    # 2. 数える（主な組・正解「はい」）
    if name == "vote":
        cnt = {}
        for bk in A.BUCKETS:
            m_ = main["W"] & (b == bk) & (y == 1)
            cnt[bk] = {"n": int(m_.sum())}
            for m in models:
                T = np.array(c["M"][m]["T"], float)[m_]; S = np.array(c["M"][m]["S"], float)[m_]
                ta = P[f"W|{m}"]["SO"][1]; sa = P[f"W|{m}"]["SO"][0]
                cnt[bk][m] = {"鎖が届いた": round(float((T >= ta).mean()), 3), "道が着かない(T<0.01)": round(float((T < 0.01).mean()), 3),
                              "着いたが弱い": round(float(((T >= 0.01) & (T < ta)).mean()), 3), "説明文だけで当たる": round(float((S > sa).mean()), 3),
                              "両方とも外す": round(float(((S <= sa) & (T < ta)).mean()), 3)}
        res["count"] = cnt
    out[name] = res
json.dump(out, open(f"{R}/E20-1_out.json", "w"), ensure_ascii=False, indent=1)
for name in out:
    print("==", name)
    for k, v in out[name].items():
        if k == "count": print(json.dumps(v, ensure_ascii=False)); continue
        print(k, v["verdict LR-SO"], {kk: vv for kk, vv in v.items() if "LR-SO" in kk and "|" in kk})
        print("   BA", {m: v[f"{m}|BA"] for m in models})
