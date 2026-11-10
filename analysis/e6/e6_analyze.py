"""E6 の比べ。python3 e6_analyze.py E6_results.json out.json
正解：Claudeの常識判定（mine、2＝正しい）が主。Geminiの判定（2＝正しい）、mine の 2＝違う でも出す。
"""
import json, csv, sys, numpy as np
from sklearn.linear_model import LogisticRegression

import os
D = os.path.dirname(os.path.abspath(__file__))
def find(*cands):
    for c in cands:
        if os.path.exists(os.path.join(D, c)): return os.path.join(D, c)
    raise FileNotFoundError(cands)
sig = lambda x: 1 / (1 + np.exp(-x))
lgt = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6)) - np.log(np.clip(1 - p, 1e-6, 1 - 1e-6))
SRC = ["S1", "S2", "S3"]
W_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]


def aurc(p, y):
    """p＝正しい確率。答え＝p>0.5、確かさ＝max(p,1-p)。確かな順に答えた時の誤り率の平均。"""
    pred = p > 0.5; conf = np.maximum(p, 1 - p); o = np.argsort(-conf, kind="stable")
    err = (pred[o] != y[o]).astype(float)
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


def nli_feats(R, key):
    n = len(R); F = np.zeros((n, 6)); has = np.zeros(n, bool)
    for i, it in enumerate(R):
        k = "S2" if key == "S3" else key
        ps = np.array(it["nli"][k], float).reshape(-1, 3)
        if key == "S3": ps = ps[:1]
        if len(ps) == 0: continue
        has[i] = True; e, c = ps[:, 0], ps[:, 2]
        F[i] = [e.max(), c.max(), e.mean(), c.mean(), 1 - np.prod(1 - e), len(ps)]
    return F, has


def run(res_path):
    R = sorted(json.load(open(res_path))["items"], key=lambda x: x["id"])
    I = {x["id"]: x for x in json.load(open(find("e6_items.json")))}
    g = json.load(open(find("../e5/labels/gemini_items.json", "../e5/gemini_items.json")))
    gl = {int(r["no"]): int(r["label"]) for r in csv.DictReader(open(find("../e5/labels/gemini_labels.csv", "../e5/gemini_labels.csv")))}
    gem = {x["sid"]: gl[x["no"]] for x in g if x.get("sid") is not None}
    ids = [it["id"] for it in R]; n = len(ids)
    typ = np.array([I[i]["type"] for i in ids]); mine = np.array([I[i]["mine"] for i in ids]); L = np.array([I[i]["logit"] for i in ids])
    gm = np.array([gem[i] for i in ids]); ents = np.array([I[i]["ent"] for i in ids])
    Y = {"claude": (mine >= 1).astype(int), "claude_2wrong": (mine == 1).astype(int), "gemini": (gm >= 1).astype(int)}
    y_train = Y["claude"]
    groups = {"G_wrong": (typ == "G") & (mine == 0), "T_right": (typ == "T") & (mine == 1), "G_right": (typ == "G") & (mine >= 1),
              "H_E_wrong": np.isin(typ, ["H", "E"]) & (mine == 0), "G_wrong_yes": (typ == "G") & (mine == 0) & (L > 0)}
    gy = groups["G_wrong_yes"]; qs = np.quantile(L[gy], [1 / 3, 2 / 3])  # Gemmaが「はい」と言った常識でも違うG＝自信ある間違い
    conf3 = {"Gy_low": gy & (L <= qs[0]), "Gy_mid": gy & (L > qs[0]) & (L <= qs[1]), "Gy_high": gy & (L > qs[1])}

    fixed = {"1_conf": sig(L), "1b_conf_rerun": sig(np.array([it["gemma"]["none"] for it in R]))}
    for s in SRC:
        fixed[f"2_in_prompt_{s}"] = sig(np.array([it["gemma"][s] for it in R]))
        F, has = nli_feats(R, s)
        fixed[f"3_nli_{s}"] = np.where(has, F[:, 0], sig(L))  # 一番含む意味の確率。説明文なし→自信
    learned = {}
    for seed in range(5):
        rng = np.random.default_rng(600 + seed); ue = np.unique(ents); fold_of = {e: k % 5 for k, e in enumerate(rng.permutation(ue))}
        fold = np.array([fold_of[e] for e in ents])
        for s in SRC:
            F, has = nli_feats(R, s); ev = np.where(has, lgt(F[:, 4]), 0.0); g2 = np.array([it["gemma"][s] for it in R])
            X5 = np.column_stack([L, F, has]); X5b = np.column_stack([X5, g2])
            p4 = np.zeros(n); p5 = np.zeros(n); p5b = np.zeros(n); ws = []
            for k in range(5):
                tr, te = fold != k, fold == k
                best = min(W_GRID, key=lambda w: -np.mean(y_train[tr] * np.log(sig(L[tr] + w * ev[tr]) + 1e-9) + (1 - y_train[tr]) * np.log(1 - sig(L[tr] + w * ev[tr]) + 1e-9)))
                ws.append(best); p4[te] = sig(L[te] + best * ev[te])
                for X, P in [(X5, p5), (X5b, p5b)]:
                    m = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y_train[tr]); P[te] = m.predict_proba(X[te])[:, 1]
            learned.setdefault(f"4_net_{s}", []).append((p4, ws)); learned.setdefault(f"5_learn_{s}", []).append((p5, None))
            learned.setdefault(f"5b_learn_plus_prompt_{s}", []).append((p5b, None))
        # 説明文なしの学習（比べの土台）
        p0 = np.zeros(n)
        for k in range(5):
            tr, te = fold != k, fold == k
            m = LogisticRegression(max_iter=2000).fit(L[tr, None], y_train[tr]); p0[te] = m.predict_proba(L[te, None])[:, 1]
        learned.setdefault("0_learn_conf_only", []).append((p0, None))

    def describe(p):
        d = {gname: float(np.mean(p[m] <= 0.5)) for gname, m in {**groups, **conf3}.items()}  # 「いいえ」の割合
        d["fix_G_wrong"] = d.pop("G_wrong"); d["break_T_right"] = d.pop("T_right"); d["break_G_right"] = d.pop("G_right")
        d.update({f"aurc_{k}": aurc(p, y) for k, y in Y.items()}); d["acc_claude"] = float(np.mean((p > 0.5) == Y["claude"]))
        return d

    out = {"n": n, "counts": {k: int(m.sum()) for k, m in {**groups, **conf3}.items()}, "G_wrong_yes_logit_cuts": qs.tolist(),
           "has_defs": {s: float(nli_feats(R, s)[1].mean()) for s in SRC}, "methods": {}}
    for k, p in fixed.items():
        out["methods"][k] = {"mean": describe(p)}
    for k, runs in learned.items():
        ds = [describe(p) for p, _ in runs]
        out["methods"][k] = {"mean": {m: float(np.mean([d[m] for d in ds])) for m in ds[0]}, "per_seed": ds}
        if runs[0][1] is not None: out["methods"][k]["w"] = [w for _, w in runs]
    return out


if __name__ == "__main__":
    o = run(sys.argv[1]); json.dump(o, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
    print("数:", o["counts"], "説明文あり:", o["has_defs"])
    print(f"{'方法':32s} 直るG 直るGはい 低 中 高  壊すT 壊すG正 AURC常識 AURC Gemini")
    for k, v in o["methods"].items():
        m = v["mean"]; print(f"{k:32s} {m['fix_G_wrong']:.2f} {m['G_wrong_yes']:.2f} {m['Gy_low']:.2f} {m['Gy_mid']:.2f} {m['Gy_high']:.2f}  {m['break_T_right']:.2f} {m['break_G_right']:.2f}  {m['aurc_claude']:.3f}  {m['aurc_gemini']:.3f}")
