"""E10-1 の比べ（PLAN_E10-1.md、結果の前に固定）。
使い方：
  python3 e101_analyze.py sample e101_items.json RESULTS_DIR cs_sample.json   … 常識で判定する600×3を選ぶ（文だけ、方法の値は出さない）
  python3 e101_analyze.py run e101_items.json RESULTS_DIR out.json [cs.json]   … 比べ（cs.json＝{"id": 0/1/2}、2は除く）
決まり（E10と同じ）：物を5つに分け、1つ（2割）で調整・学習、残り8割で評価。5回。差が相手の値の20%以上、かつ5回中4回以上同じ向き → 差あり。
方法：① A 自信／③ one 辞書1段／⑥N 一覧なしの鎖／⑥D E10の手書きつきの鎖（比べ0だけ）／⑨ 直した版／C++ 木の学習（Gemmaだけ）／
      ⑧N++（Gemma＋一覧なしの鎖）／⑧'++（⑧N++＋直し1〜4と向き）／⑧D++（Gemma＋手書きつきの鎖、参考）。
"""
import json, sys, os, gzip, collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from wordfreq import zipf_frequency as zf
DEPTH = 3
sig = lambda x: 1 / (1 + np.exp(-np.asarray(x, float)))


def aurc(p, y):
    pred = p > 0.5; conf = np.maximum(p, 1 - p); o = np.argsort(-conf, kind="stable")
    err = (pred[o] != y[o]).astype(float)
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


def lgz(d, name):
    with gzip.open(os.path.join(d, name + ".json.gz"), "rt", encoding="utf-8") as f:
        return json.load(f)


class Chains:
    """場面ごとの鎖と着き。tag＝"N"（一覧なし）か "D"（E10の手書きつき）。"""
    def __init__(self, res, ST, dom):
        self.dom = dom
        self.nid = {tuple(n): i for i, n in enumerate(res["nodes"])}
        self.A = np.array(res["arrive"], dtype=np.float32).max(axis=2)
        self.cidx = {c: j for j, c in enumerate(res["classes"])}
        self.ch = {}
        for key, (nd, lk) in res["chains"].items():
            tag, x = key.split("|", 1)
            kids = collections.defaultdict(list)
            for a, b, t in lk:
                kids[tuple(a)].append((tuple(b), t, ST.get((dom, tag, a[0], a[1], b[0], t), 0.0)))
            self.ch[(tag, x)] = ([tuple(n) for n in nd], kids)

    def has(self, tag, x):
        return (tag, x) in self.ch and len(self.ch[(tag, x)][0]) > 0

    def arr(self, n, C):
        if self.dom == "P" and n[0] == C:
            return 1.0
        return float(self.A[self.nid[n], self.cidx[C]])

    def scores(self, tag, x, C, H=None):
        """one（1段）・six（一番強い道）・len・anyarr、H があれば直し1（loc の段で H の語に着いた道の強さ）"""
        if not self.has(tag, x):
            return None
        nodes, kids = self.ch[(tag, x)]
        roots = [n for n in nodes if n[0] == x]
        memo = {}
        def f(n, d):
            if (n, d) in memo: return memo[(n, d)]
            best, ln = self.arr(n, C), 0
            if best < 1.0 and d > 0:
                for c, t, e in kids.get(n, []):
                    v, l2 = f(c, d - 1)
                    if e * v > best: best, ln = e * v, l2 + 1
            memo[(n, d)] = (best, ln); return memo[(n, d)]
        r6 = [f(r, DEPTH) for r in roots]
        b6 = max(r6, key=lambda r: r[0]) if r6 else (0.0, 0)
        out = {"one": max(self.arr(r, C) for r in roots) if roots else 0.0, "six": b6[0], "len": b6[1],
               "anyarr": max(self.arr(n, C) for n in nodes)}
        if H is not None:
            m2 = {}
            def g(n, d):
                if (n, d) in m2: return m2[(n, d)]
                best = 0.0
                if d > 0:
                    for c, t, e in kids.get(n, []):
                        v = e * (1.0 if (t == "loc" and c[0] in H) else g(c, d - 1))
                        best = max(best, v)
                m2[(n, d)] = best; return best
            out["fix1"] = max([g(r, DEPTH) for r in roots] + [0.0])
        return out


def whole_hypernyms(LU, ST):
    """直し1：全体Wの説明文を一種の段だけ2段たどった語（W自身を除く）。全体の3分の1以上に出る語は除く（先に固定した割合）。"""
    H = {}
    for W, (nd, lk) in LU["whole_chains"].items():
        H[W] = {n[0] for n in nd if n[0] != W}
    cnt = collections.Counter(w for s in H.values() for w in s)
    common = {w for w, c in cnt.items() if c >= len(H) / 3}
    return {W: s - common for W, s in H.items()}, sorted(common)


def build(items_path, rdir):
    IT = json.load(open(items_path)); items = IT["items"]
    G = lgz(rdir, "E101_gemma"); DIR = lgz(rdir, "E101_direction"); LU = lgz(rdir, "E101_L_used")
    ST = {}
    for d, tag, w, k, g, t, e in lgz(rdir, "E101_steps"):
        ST[(d, tag, w, k, g, t)] = e
    dirp = {}
    for i, ab, ba in zip(DIR["idx"], DIR["ab"], DIR["ba"]):
        dirp[i] = float((sig(ab) + sig(-ba)) / 2)   # 「x は W の一部」の向きが正しい確からしさ（順を入れ替えた2回の平均）
    H, common = whole_hypernyms(LU, ST)
    lem = {W: set().union(*[set(x) for x in v]) if v else set() for W, v in LU["whole_lemmas"].items()}
    mero = {w: {m.lower() for m in v} for w, v in LU["mero"].items()}
    holo = {w: {m.lower() for m in v} for w, v in LU["holo"].items()}
    data = {}
    for dom in "PDG":
        idx = [i for i, it in enumerate(items) if it["dom"] == dom]
        its = [items[i] for i in idx]
        CH = Chains(lgz(rdir, f"E101_res_{dom}"), ST, dom)
        lg = np.array([G["logit"][i] for i in idx]); y = np.array([it["gold"] for it in its])
        rv = np.array([G["rev"][i] if G["rev"][i] is not None else 0.0 for i in idx])
        F = collections.defaultdict(list)
        for it in its:
            x, C = it["ent"], it["cls"]
            sN = CH.scores("N", x, C, H.get(C) if dom == "P" else None)
            sD = CH.scores("D", x, C)
            F["has"].append(sN is not None)
            for k in ("one", "six", "len", "anyarr"):
                F[k + "N"].append(sN[k] if sN else 0.0); F[k + "D"].append(sD[k] if sD else 0.0)
            F["hasD"].append(sD is not None)
            if dom == "P":
                F["fix1"].append(sN.get("fix1", 0.0) if sN else 0.0)
                f2 = int(x in lem.get(C, set()) or x in mero.get(C, set()) or C in holo.get(x, set()))
                F["fix2"].append(f2)
            if dom == "D":
                sp = LU["split"].get(x)
                ok = bool(sp) and LU["kind_disease"].get(sp["kind"], 0.0) > 0.5
                v = 0.0
                if ok:
                    for part in (sp["kind"], sp["rest"], sp["rest_last"]):
                        s = CH.scores("N", part, C)
                        if s: v = max(v, s["six"])
                F["fix4"].append(v); F["fix4_ok"].append(int(ok))
        F = {k: np.array(v, float) for k, v in F.items()}
        F["dir"] = np.array([dirp.get(i, 1.0) for i in idx]) if dom == "P" else np.ones(len(idx))
        data[dom] = dict(idx=idx, its=its, lg=lg, y=y, rv=rv, F=F)
    return items, data, {"common_hypernyms": common, "n_hyp": {W: len(s) for W, s in H.items()}}


def methods(dom, D, tune):
    lg, y, F = D["lg"], D["y"], D["F"]
    has = F["has"] > 0
    pl = LogisticRegression(max_iter=1000).fit(lg[tune, None], y[tune]); pA = pl.predict_proba(lg[:, None])[:, 1]
    P = {"A": pA, "one": np.where(has, F["oneN"], pA), "sixN": np.where(has, F["sixN"], pA),
         "sixD": np.where(F["hasD"] > 0, F["sixD"], pA)}
    if dom == "P":
        base = np.maximum.reduce([F["oneN"], F["fix1"], F["fix2"]]) * F["dir"]
        P["nine"] = np.where(has | (F["fix2"] > 0), base, pA)
    elif dom == "D":
        P["nine"] = np.where(has, F["sixN"], np.where(F["fix4_ok"] > 0, F["fix4"], pA))
    else:
        P["nine"] = P["sixN"]
    its = D["its"]; ents = sorted({it["ent"] for it in its}); eidx = {e: j for j, e in enumerate(ents)}
    ent = np.array([eidx[it["ent"]] for it in its]); classes = sorted({it["cls"] for it in its}); cidx = {c: j for j, c in enumerate(classes)}
    M = np.full((len(ents), len(classes)), np.nan)
    for i, it in enumerate(its): M[ent[i], cidx[it["cls"]]] = lg[i]
    emax, emean, estd = np.nanmax(M, 1)[ent], np.nanmean(M, 1)[ent], np.nanstd(M, 1)[ent]
    rank = np.array([(M[ent[i]] > lg[i]).sum() for i in range(len(its))]); nyes = (M > 0).sum(1)[ent]
    freq = np.array([zf(it["ent"], "en") for it in its]); cls1h = np.eye(len(classes))[[cidx[it["cls"]] for it in its]]
    FG = np.column_stack([lg, emax, emean, estd, rank, nyes, freq, D["rv"], cls1h])
    FN = np.column_stack([FG, F["oneN"], F["sixN"], F["lenN"], F["anyarrN"], F["has"]])
    FDo = np.column_stack([FG, F["oneD"], F["sixD"], F["lenD"], F["anyarrD"], F["hasD"]])
    extra = {"P": [F.get("fix1"), F.get("fix2"), F["dir"]], "D": [F.get("fix4"), F.get("fix4_ok")], "G": []}[dom]
    FX = np.column_stack([FN] + extra) if extra else FN
    gb = lambda X, sp: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=sp).fit(X[tune], y[tune]).predict_proba(X)[:, 1]
    sp = int(tune.sum()) % 97
    P["C"] = LogisticRegression(max_iter=5000).fit(FG[tune], y[tune]).predict_proba(FG)[:, 1]
    P["Cpp"] = gb(FG, sp); P["N8pp"] = gb(FN, sp); P["F8pp"] = gb(FX, sp); P["D8pp"] = gb(FDo, sp)
    return P, ent, freq


def judge(runs, a, b, key):
    va = np.array([r[key][a] for r in runs if key in r]); vb = np.array([r[key][b] for r in runs if key in r])
    if len(va) < 5: return None
    better = np.sum(vb - va >= 0.2 * vb); worse = np.sum(va - vb >= 0.2 * va)
    verdict = "差あり（%sが良い）" % a if better >= 4 else ("差あり（%sが良い）" % b if worse >= 4 else "同着")
    return {a: float(va.mean()), b: float(vb.mean()), "回": f"{int(better)}/5 vs {int(worse)}/5", "判定": verdict}


def run(items_path, rdir, out_path, cs_path=None):
    items, data, info = build(items_path, rdir)
    CS = {int(k): v for k, v in json.load(open(cs_path)).items()} if cs_path else {}
    report = {"info": info}
    for dom in "PDG":
        D = data[dom]; y, lg = D["y"], D["lg"]
        ents = sorted({it["ent"] for it in D["its"]}); eidx = {e: j for j, e in enumerate(ents)}
        ent = np.array([eidx[it["ent"]] for it in D["its"]])
        rng = np.random.default_rng(0); perm = rng.permutation(len(ents)); fold = np.zeros(len(ents), int)
        for f in range(5): fold[perm[f::5]] = f
        ycs = np.array([CS.get(items[i]["id"], -1) for i in D["idx"]])
        runs = []
        for sp in range(5):
            tune = fold[ent] == sp; ev = ~tune
            P, _, _ = methods(dom, D, tune)
            r = {"table": {k: aurc(v[ev], y[ev]) for k, v in P.items()}}
            m = ev & (ycs >= 0) & (ycs != 2)
            if m.sum() > 20:
                r["cs"] = {k: aurc(v[m], (ycs[m] == 1).astype(int)) for k, v in P.items()}
            cw = ev & (((lg > 2) & (y == 0)) | ((lg < -2) & (y == 1)))
            r["fix"] = {k: float(np.mean((v[cw] > 0.5) == y[cw])) for k, v in P.items()} if cw.sum() else {}
            ok = ev & (lg > 0) & (y == 1)
            r["break"] = {k: float(np.mean(v[ok] <= 0.5)) for k, v in P.items()} if ok.sum() else {}
            if dom == "P":
                both = ev & (lg > 0) & (D["rv"] > 0)
                r["both_dir_acc"] = {k: float(np.mean((v[both] > 0.5) == y[both])) for k, v in P.items()} if both.sum() else {}
            runs.append(r)
        F = D["F"]
        rep = {"n_items": len(y), "n_true": int(y.sum()), "coverage": float(np.mean([F["has"][ent == j].max() for j in range(len(ents))])),
               "mean_table": {k: float(np.mean([r["table"][k] for r in runs])) for k in runs[0]["table"]},
               "fix": {k: float(np.mean([r["fix"][k] for r in runs if r["fix"]])) for k in runs[0]["fix"]},
               "break": {k: float(np.mean([r["break"][k] for r in runs if r["break"]])) for k in runs[0]["break"]}}
        if dom == "D":
            rep["coverage_with_fix4"] = float(np.mean([(F["has"][ent == j].max() > 0) or (F["fix4_ok"][ent == j].max() > 0) for j in range(len(ents))]))
        if dom == "P":
            rep["both_directions_yes"] = int(((lg > 0) & (D["rv"] > 0)).sum())
            rep["both_dir_acc"] = {k: float(np.mean([r["both_dir_acc"][k] for r in runs if r["both_dir_acc"]])) for k in runs[0]["both_dir_acc"]}
            rep["fix2_rate"] = float(F["fix2"].mean())
        for key in ("table", "cs"):
            if all(key in r for r in runs):
                if key == "cs": rep["mean_cs"] = {k: float(np.mean([r["cs"][k] for r in runs])) for k in runs[0]["cs"]}
                c1 = ("nine", "one") if dom == "P" else ("nine", "sixN")
                rep["main_" + key] = {"0 ⑥N対⑥D": judge(runs, "sixN", "sixD", key),
                                      "1 ⑨対" + ("③" if dom == "P" else "⑥N"): judge(runs, *c1, key) if dom != "G" else None,
                                      "2 ⑨対①": judge(runs, "nine", "A", key),
                                      "3 ⑧'++対⑧N++": judge(runs, "F8pp", "N8pp", key) if dom != "G" else None,
                                      "おまけ ③対①": judge(runs, "one", "A", key), "おまけ ⑧N++対C++": judge(runs, "N8pp", "Cpp", key),
                                      "参考 ⑧N++対⑧D++": judge(runs, "N8pp", "D8pp", key)}
        rep["runs"] = runs
        report[dom] = rep
        print(dom, json.dumps({k: rep.get(k) for k in ("n_items", "n_true", "coverage", "mean_table", "main_table", "mean_cs", "main_cs")}, ensure_ascii=False, indent=0))
    key = "main_cs" if all("main_cs" in report[d] for d in "PDG") else "main_table"
    eff = []
    for d in "PD":
        m = report[d][key]; k1 = [k for k in m if k.startswith("1 ")][0]
        if m[k1] and m[k1]["判定"].startswith("差あり（nine") and m["3 ⑧'++対⑧N++"] and m["3 ⑧'++対⑧N++"]["判定"].startswith("差あり（F8pp"):
            eff.append(d)
    ok0 = [d for d in "PDG" if not (report[d][key]["0 ⑥N対⑥D"] and report[d][key]["0 ⑥N対⑥D"]["判定"].startswith("差あり（sixD"))]
    report["overall"] = {"採点": key, "直し方が効いた場面": eff, "一覧をやめても崩れない場面": ok0}
    print(report["overall"])
    json.dump(report, open(out_path, "w"), ensure_ascii=False, indent=1)


def sample(items_path, rdir, out_path, n_rand=300, n_yes=300, seed=0):
    """常識で判定する物：場面ごとに でたらめ300＋「Gemmaか辞書（⑥N・⑨）が はい」から300。文だけを書き出す。"""
    items, data, _ = build(items_path, rdir)
    rng = np.random.default_rng(seed); out = {}
    for dom in "PDG":
        D = data[dom]; F = D["F"]; n = len(D["idx"])
        one_fold = np.zeros(n, bool)
        P, _, _ = methods(dom, D, np.ones(n, bool) if n < 50 else (np.arange(n) % 5 == 0))
        yes = (D["lg"] > 0) | (P["sixN"] > 0.5) | (P["nine"] > 0.5)
        r = rng.choice(n, min(n_rand, n), replace=False).tolist()
        pool = [i for i in np.where(yes)[0].tolist() if i not in set(r)]
        s = r + (rng.choice(pool, min(n_yes, len(pool)), replace=False).tolist() if pool else [])
        rng.shuffle(s)
        out[dom] = [{"id": items[D["idx"][i]]["id"], "stmt": items[D["idx"][i]]["stmt"]} for i in s]
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=0)
    print({d: len(v) for d, v in out.items()})


if __name__ == "__main__":
    {"run": run, "sample": sample}[sys.argv[1]](*sys.argv[2:])
