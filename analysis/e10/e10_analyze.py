"""E10 の比べ。python3 e10_analyze.py e10_items.json RESULTS_DIR out.json [common_sense.json]
RESULTS_DIR：Colabの E10_*.json.gz を置いた所。common_sense.json：{"id": 0/1/2,...}（1＝正しい、0＝違う、2＝どちらとも言えない→除く）。
決まり（PLAN_E10.md、結果の前に固定）：物を5つに分け、1つ（2割）で調整・学習、残り8割で評価。5回。
差が相手の値の20%以上、かつ5回中4回以上同じ向き → 差あり。
"""
import json, sys, os, gzip, collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from wordfreq import zipf_frequency as zf
DEPTH = 3
sig = lambda x: 1 / (1 + np.exp(-x))


def aurc(p, y):
    """p＝「はい」の確率。答え＝p>0.5、確かさ＝max(p,1-p)。確かな順に答えた時の誤り率の平均（E6と同じ）。"""
    pred = p > 0.5; conf = np.maximum(p, 1 - p); o = np.argsort(-conf, kind="stable")
    err = (pred[o] != y[o]).astype(float)
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


def lgz(d, name):
    with gzip.open(os.path.join(d, name + ".json.gz"), "rt", encoding="utf-8") as f:
        return json.load(f)


def chain_scores(res, ST, dom, classes):
    """物ごと・分類ごとの ③（1段）・⑥（一番強い道の掛け算）・道の段の数。res＝E10_res_dom。"""
    nid = {tuple(n): i for i, n in enumerate(res["nodes"])}
    A = np.array(res["arrive"], dtype=np.float32).max(axis=2)   # 場面Dは2つの言い方の大きい方
    cidx = {c: j for j, c in enumerate(classes)}
    out = {}
    for key, (nd, lk) in res["chains"].items():
        tag, x = key.split("|", 1)
        nodes = [tuple(n) for n in nd]
        if not nodes:
            continue
        ch = collections.defaultdict(list)
        for a, b, t in lk:
            ch[tuple(a)].append((tuple(b), ST.get((dom, a[0], a[1], b[0], t), 0.0)))
        roots = [n for n in nodes if n[0] == x]
        for C in classes:
            j = cidx[C]
            def arr(n):
                if dom == "P" and n[0] == C:   # 全体の語そのものに着いた
                    return 1.0
                return float(A[nid[n], j])
            memo = {}
            def f(n, d):
                if (n, d) in memo: return memo[(n, d)]
                best, ln = arr(n), 0
                if best < 1.0 and d > 0:
                    for c, e in ch.get(n, []):
                        v, l2 = f(c, d - 1)
                        if e * v > best: best, ln = e * v, l2 + 1
                memo[(n, d)] = (best, ln)
                return memo[(n, d)]
            r6 = [f(r, DEPTH) for r in roots]
            b6 = max(r6, key=lambda r: r[0]) if r6 else (0.0, 0)
            out[(tag, x, C)] = {"one": max(arr(r) for r in roots) if roots else 0.0, "six": b6[0], "len": b6[1],
                                "anyarr": float(max(arr(n) for n in nodes))}
    return out


def run(items_path, rdir, out_path, cs_path=None):
    IT = json.load(open(items_path)); META = IT["meta"]; items = IT["items"]
    G = lgz(rdir, "E10_gemma"); LX = lgz(rdir, "E10_L_used")["L"]
    ST = {(d, w, k, g, t): e for d, w, k, g, t, e, c in lgz(rdir, "E10_steps")}
    CS = {int(k): v for k, v in json.load(open(cs_path)).items()} if cs_path else {}
    report = {}
    for dom in "PDG":
        idx = [i for i, it in enumerate(items) if it["dom"] == dom]
        its = [items[i] for i in idx]
        res = lgz(rdir, f"E10_res_{dom}")
        classes = res["classes"]
        sc = chain_scores(res, ST, dom, classes)
        y = np.array([it["gold"] for it in its]); lg = np.array([G["logit"][i] for i in idx])
        rv = np.array([G["rev"][i] if G["rev"][i] is not None else 0.0 for i in idx])
        ents = sorted({it["ent"] for it in its}); eidx = {e: j for j, e in enumerate(ents)}
        ent = np.array([eidx[it["ent"]] for it in its])
        has = np.array([len(LX.get(it["ent"], [])) > 0 for it in its])
        def get(tag, key):
            return np.array([sc.get((tag, it["ent"], it["cls"]), {}).get(key, 0.0) for it in its])
        one, sixD, sixB, ln = get("D", "one"), get("D", "six"), get("B", "six"), get("D", "len")
        anyarr = get("D", "anyarr")
        sixDw = get("Dw", "six") if dom == "P" else None
        # Gemmaだけの特徴（同じ物の他の分類への答え）
        cidx = {c: j for j, c in enumerate(classes)}
        M = np.full((len(ents), len(classes)), np.nan)
        for i, it in enumerate(its): M[ent[i], cidx[it["cls"]]] = lg[i]
        emax, emean, estd = np.nanmax(M, 1)[ent], np.nanmean(M, 1)[ent], np.nanstd(M, 1)[ent]
        rank = np.array([(M[ent[i]] > lg[i]).sum() for i in range(len(its))])
        nyes = (M > 0).sum(1)[ent]
        freq = np.array([zf(it["ent"], "en") for it in its])
        cls1h = np.eye(len(classes))[[cidx[it["cls"]] for it in its]]
        FG = np.column_stack([lg, emax, emean, estd, rank, nyes, freq, rv, cls1h])
        FD = np.column_stack([FG, one, sixD, sixB, ln, anyarr, has])
        ycs = np.array([CS.get(items[i]["id"], -1) for i in idx])
        runs = []
        rng = np.random.default_rng(0); perm = rng.permutation(len(ents)); fold = np.zeros(len(ents), int)
        for f in range(5): fold[perm[f::5]] = f
        for sp in range(5):
            tune = fold[ent] == sp; ev = ~tune
            pl = LogisticRegression(max_iter=1000).fit(lg[tune, None], y[tune])
            pA = pl.predict_proba(lg[:, None])[:, 1]
            P = {"A": pA}
            for nm_, s in [("one", one), ("sixD", sixD), ("sixB", sixB)] + ([("sixDw", sixDw)] if dom == "P" else []):
                P[nm_] = np.where(has, s, pA)   # 辞書に載っていない物はGemmaの自信のまま
            P["C"] = LogisticRegression(max_iter=5000).fit(FG[tune], y[tune]).predict_proba(FG)[:, 1]
            P["Cpp"] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=sp).fit(FG[tune], y[tune]).predict_proba(FG)[:, 1]
            P["D8"] = LogisticRegression(max_iter=5000).fit(FD[tune], y[tune]).predict_proba(FD)[:, 1]
            P["D8pp"] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=sp).fit(FD[tune], y[tune]).predict_proba(FD)[:, 1]
            r = {"table": {k: aurc(v[ev], y[ev]) for k, v in P.items()}}
            m = ev & (ycs >= 0) & (ycs != 2)
            if m.sum() > 20:
                r["cs"] = {k: aurc(v[m], (ycs[m] == 1).astype(int)) for k, v in P.items()}; r["n_cs"] = int(m.sum())
            # 自信ある間違いが直る・正しい「はい」を壊す（表の正解、評価用）
            conf_wrong = ev & ((lg > 2) & (y == 0) | (lg < -2) & (y == 1))
            r["fix"] = {k: float(np.mean((v[conf_wrong] > 0.5) == y[conf_wrong])) for k, v in P.items()} if conf_wrong.sum() else {}
            ok_yes = ev & (lg > 0) & (y == 1)
            r["break"] = {k: float(np.mean(v[ok_yes] <= 0.5)) for k, v in P.items()} if ok_yes.sum() else {}
            # よく使う語／まれな語（3つ、物ごとの頻度で）
            q = np.quantile(freq[ev], [1 / 3, 2 / 3]); band = np.digitize(freq, q)
            r["freq"] = {str(b): {k: aurc(P[k][ev & (band == b)], y[ev & (band == b)]) for k in ("A", "one", "sixD", "Cpp", "D8pp")}
                         for b in range(3) if (ev & (band == b)).sum() > 50}
            if dom == "G":
                reg = np.array([items[i]["cls"] for i in idx]); home = {}
                for i, it in zip(idx, its):
                    if it["gold"]: home[it["ent"]] = it["cls"]
                hr = np.array([home.get(it["ent"], "") for it in its])
                r["region"] = {g: {"n": int((ev & (hr == g)).sum()), **{k: aurc(P[k][ev & (hr == g)], y[ev & (hr == g)]) for k in ("A", "sixD", "D8pp")}}
                               for g in sorted(set(hr)) if (ev & (hr == g)).sum() > 100}
            runs.append(r)
        def judge(a, b, key="table"):
            va = np.array([r[key][a] for r in runs if key in r]); vb = np.array([r[key][b] for r in runs if key in r])
            if len(va) < 5: return None
            better = np.sum(vb - va >= 0.2 * vb); worse = np.sum(va - vb >= 0.2 * va)
            verdict = "差あり（%sが良い）" % a if better >= 4 else ("差あり（%sが良い）" % b if worse >= 4 else "同着")
            return {a: float(va.mean()), b: float(vb.mean()), "回": f"{int(better)}/5 vs {int(worse)}/5", "判定": verdict}
        rep = {"n_items": len(its), "n_true": int(y.sum()), "coverage_ents": float(np.mean([len(LX.get(e, [])) > 0 for e in ents])),
               "gemma_yes": float((lg > 0).mean()), "gemma_acc": float(((lg > 0) == y).mean()),
               "mean_table": {k: float(np.mean([r["table"][k] for r in runs])) for k in runs[0]["table"]},
               "main_table": {"1 ⑥D対①": judge("sixD", "A"), "2 ⑥D対③": judge("sixD", "one"), "3 ⑧D++対C++": judge("D8pp", "Cpp")},
               "fix": {k: float(np.mean([r["fix"][k] for r in runs if r["fix"]])) for k in runs[0]["fix"]},
               "break": {k: float(np.mean([r["break"][k] for r in runs if r["break"]])) for k in runs[0]["break"]},
               "freq": runs[0]["freq"], "runs": runs}
        if all("cs" in r for r in runs):
            rep["mean_cs"] = {k: float(np.mean([r["cs"][k] for r in runs])) for k in runs[0]["cs"]}
            rep["main_cs"] = {"1 ⑥D対①": judge("sixD", "A", "cs"), "2 ⑥D対③": judge("sixD", "one", "cs"), "3 ⑧D++対C++": judge("D8pp", "Cpp", "cs")}
        if dom == "P":
            both = (lg > 0) & (rv > 0)
            rep["both_directions_yes"] = int(both.sum())
            rep["both_dir_fixed_by_sixD"] = float(np.mean(sixD[both & has] > 0.5)) if (both & has).sum() else None
        report[dom] = rep
        print(dom, json.dumps({k: rep[k] for k in ("n_items", "n_true", "coverage_ents", "gemma_acc", "mean_table", "main_table")}, ensure_ascii=False, indent=0))
    # 全体の判定：2場面以上で（常識の採点で）比べ1と比べ3がともに差あり（辞書が良い）
    key = "main_cs" if all("main_cs" in report[d] for d in "PDG") else "main_table"
    ok = [d for d in "PDG" if report[d][key]["1 ⑥D対①"] and
          report[d][key]["1 ⑥D対①"]["判定"].startswith("差あり（sixD") and report[d][key]["3 ⑧D++対C++"]["判定"].startswith("差あり（D8pp")]
    report["overall"] = {"採点": key, "効いた場面": ok, "判定": "別の問いでも効く" if len(ok) >= 2 else "そうとは言えない"}
    print(report["overall"])
    json.dump(report, open(out_path, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    run(*sys.argv[1:])
