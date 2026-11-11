"""E8n（一覧なし）の比べ。e8_analyze.py の写しで、方法B・打ち消しの網だけ外した。E8_DIR=公開リポジトリの analysis/e8。
E8 の比べ。python3 e8_analyze.py E8_results.json out.json [drift_sample.json]
正解：700個はE6と同じ（Claudeの常識判定が主、Geminiでも）。新しい300個は e8_new_items_labeled.json（Claudeの常識判定）。"""
import json, sys, os, itertools, csv, numpy as np
from sklearn.linear_model import LogisticRegression
HERE = os.environ.get("E8_DIR") or os.path.dirname(os.path.abspath(__file__))   # 公開リポジトリの analysis/e8
sys.path.insert(0, os.path.join(HERE, "..", "e6"))
import e6_analyze as e6

DEPTH = 3
sig = lambda x: 1 / (1 + np.exp(-x))
lgt = lambda p: np.log(p / (1 - p))
METHODS = "ACD"   # 一覧なし版：Bは札の一覧を使うのでやめた
GRID = list(itertools.product([0.25, 0.5, 1.0], [-1.0, 0.0, 1.0, 2.0], [0.8, 0.95, 0.99]))  # (β, b, ε) E7と同じ
GAMMAS = [0.0]  # 一覧なし版：打ち消し（手書きのまとまり）はやめた

def disjoint_groups(C):
    return []   # 一覧なし版：手書きのまとまりを使わない


GW = {}


class Item:
    def __init__(self, it, m, nodes, links, L, ST, AR):
        self.x, self.C = it["ent"], it["catw"]
        self.nodes = [tuple(n) for n in nodes]
        self.idx = {n: i for i, n in enumerate(self.nodes)}
        self.ch = {n: [] for n in self.nodes}
        for a, b in links:
            self.ch[tuple(a)].append(tuple(b))
        self.L, self.ST, self.AR = L, ST, AR
        dg = [GW[h] for h in disjoint_groups(self.C)]
        self.arr = np.array([1.0 if w == self.C else AR.get((w, k, self.C), (0.0, 0.0))[0] for w, k in self.nodes])
        self.neg = np.array([max([AR.get((w, k, g), (0.0, 0.0))[0] for g in dg] or [0.0]) for w, k in self.nodes])
        self.roots = [n for n in self.nodes if n[0] == self.x and n[1] < len(L.get(self.x, []))]

    def maxprod(self):
        """⑥ 一番強い道の掛け算、⑥b 道の「または」、一番強い道の段の数と道。"""
        memo = {}
        def f(n, d):
            if (n, d) in memo: return memo[(n, d)]
            i = self.idx[n]; a = self.arr[i]
            best, ok, path = a, a > 0.5, [n]
            if a < 1.0 and d > 0:
                e = self.ST.get(n, (0.0, 0.0))[0]
                for c in self.ch[n]:
                    v, ok2, p2 = f(c, d - 1)
                    if e * v > best: best, path = e * v, [n] + p2
                    ok = ok or (e > 0.5 and ok2)
            memo[(n, d)] = (best, ok, path)
            return memo[(n, d)]
        rs = [f(r, DEPTH) for r in self.roots] or [(0.0, False, [])]
        b = max(rs, key=lambda r: r[0])
        return b[0], any(r[1] for r in rs), max(len(b[2]) - 1, 0), b[2]

    def bp(self, beta, b, eps, gamma, iters=40, damp=0.5):
        """⑦/⑦' 網：点＝「この意味はCの一種」＋根（物の語そのもの）。つながり＝「子がCなら親もC」（強さ ε×段の確かさ、根→意味は ε）。
        点の手がかり＝β×logit(着き)＋b −γ×（両立しない分類への着き）。輪のあるBP（全部同時に更新、半分ずつ）。"""
        n = len(self.nodes); R = n
        if not hasattr(self, "_E"):
            a, c, w = [], [], []
            for r in self.roots:
                a.append(R); c.append(self.idx[r]); w.append(1.0)
            for p, cs in self.ch.items():
                e = self.ST.get(p, (0.0, 0.0))[0]
                for ch in cs:
                    if ch != p: a.append(self.idx[p]); c.append(self.idx[ch]); w.append(e)
            self._E = (np.array(a, int), np.array(c, int), np.array(w, float))
        a, c, w = self._E
        u = beta * lgt(np.clip(self.arr, 1e-4, 1 - 1e-4)) + b - gamma * self.neg
        u = np.where(self.arr >= 1.0, 30.0, u)
        un = np.vstack([np.stack([np.zeros(n), u], 1), [[0.0, b]]])
        if len(a) == 0:
            return float(sig(b))
        q = np.log(1.0 - eps * w + 1e-12)
        lp = np.zeros((len(a), 2, 2)); lp[:, 0, 1] = q  # [x_a, x_c]：親a=0 かつ 子c=1 のときだけ弱める
        m_ij = np.zeros((len(a), 2)); m_ji = np.zeros((len(a), 2))
        for _ in range(iters):
            bel = un.copy(); np.add.at(bel, c, m_ij); np.add.at(bel, a, m_ji)
            ha = bel[a] - m_ji
            nij = np.logaddexp(ha[:, 0:1] + lp[:, 0, :], ha[:, 1:2] + lp[:, 1, :]); nij -= nij.max(1, keepdims=True)
            hc = bel[c] - m_ij
            nji = np.logaddexp(hc[:, 0:1] + lp[:, :, 0], hc[:, 1:2] + lp[:, :, 1]); nji -= nji.max(1, keepdims=True)
            m_ij = damp * m_ij + (1 - damp) * nij; m_ji = damp * m_ji + (1 - damp) * nji
        bel = un.copy(); np.add.at(bel, c, m_ij); np.add.at(bel, a, m_ji)
        s = bel[R]
        return float(sig(s[1] - s[0]))


def load(r):
    L = r["L"]
    ST = {(w, k): (e, c) for w, k, e, c in r["steps"]}
    AR = {(w, k, C): (e, c) for w, k, C, e, c in r["arrive"]}
    CH = {(i, m): (nd, lk) for i, m, nd, lk in r["chains"]}
    return L, ST, AR, CH


def x_feats(it, L, AR):
    ps = [AR[(it["ent"], k, it["catw"])] for k in range(len(L.get(it["ent"], []))) if (it["ent"], k, it["catw"]) in AR]
    if not ps:
        return np.zeros(6), False
    e = np.array([p[0] for p in ps]); c = np.array([p[1] for p in ps])
    return np.array([e.max(), c.max(), e.mean(), c.mean(), 1 - np.prod(1 - e), len(ps)]), True


def labels():
    old = sorted(json.load(open(os.path.join(HERE, "..", "e6", "e6_items.json"))), key=lambda x: x["id"])
    new = sorted(json.load(open(os.path.join(HERE, "e8_new_items_labeled.json"))), key=lambda x: x["id"])
    g = json.load(open(e6.find("../e5/labels/gemini_items.json", "../e5/gemini_items.json")))
    gl = {int(r_["no"]): int(r_["label"]) for r_ in csv.DictReader(open(e6.find("../e5/labels/gemini_labels.csv", "../e5/gemini_labels.csv")))}
    gem = {x["sid"]: gl[x["no"]] for x in g if x.get("sid") is not None}
    for x in old: x["gem"] = gem[x["id"]]
    return old, new


def groups_of(items):
    typ = np.array([x["type"] for x in items]); mine = np.array([x["mine"] for x in items]); L = np.array([x["logit"] for x in items])
    return {"fix_conf_err": (typ == "G") & (mine == 0) & (L > 0), "fix_G_wrong": (typ == "G") & (mine == 0),
            "break_T_right": (typ == "T") & (mine == 1), "break_G_right": (typ == "G") & (mine >= 1)}


def describe(p, items, Y):
    d = {k: (float(np.mean(p[m] <= 0.5)) if m.sum() else None) for k, m in groups_of(items).items()}
    d.update({f"aurc_{k}": e6.aurc(p, yy) for k, yy in Y.items()})
    return d


def compute(items, L, ST, AR, CH, log=print):
    """方法ごとの値（⑥・⑥b・段・道）と、⑦'の格子ぜんぶ（β,b,εは36通り、γはその上で5通り）を前もって計算。"""
    out = {}
    for m in METHODS:
        its = [Item(it, m, *CH[(it["id"], m)], L, ST, AR) for it in items]
        mp = [x.maxprod() for x in its]
        s6 = np.array([v[0] for v in mp]); s6b = np.array([1.0 if v[1] else 0.0 for v in mp]); dep = np.array([v[2] for v in mp])
        poly = np.array([len(L.get(v[3][1][0], [])) if len(v[3]) > 1 else 0 for v in mp])
        negx = np.array([max([x.neg[x.idx[r]] for r in x.roots] or [0.0]) for x in its])
        found = np.array([1.0 if any(x.ch[r] for r in x.roots) else 0.0 for x in its])
        bp = {}
        for pr in GRID:
            bp[pr + (0.0,)] = np.array([x.bp(*pr, 0.0) for x in its])
        out[m] = dict(its=its, s6=s6, s6b=s6b, dep=dep, poly=poly, negx=negx, found=found, bp=bp,
                      paths=[v[3] for v in mp], nodes=np.array([len(x.nodes) for x in its]))
        log(f"{m}: 点の平均 {out[m]['nodes'].mean():.1f}  段の数 {dict(zip(*np.unique(dep, return_counts=True)))}")
    return out


def add_gamma(o, pr):
    """(β,b,ε) を決めたあと、γ を足した網の値を計算（まだなら）。"""
    for g in GAMMAS[1:]:
        if pr + (g,) not in o["bp"]:
            o["bp"][pr + (g,)] = np.array([x.bp(*pr, g) for x in o["its"]])


def ll(p, y):
    return -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))


def feats(X5, o, p7):
    return np.column_stack([X5, o["s6"], o["s6b"], o["dep"], p7, o["negx"], o["poly"], o["found"]])


def run(r):
    L, ST, AR, CH = load(r)
    old, new = labels()
    allit = old + new
    O = compute(allit, L, ST, AR, CH)
    no = len(old)
    F = [x_feats(it, L, AR) for it in allit]
    F6 = np.array([f[0] for f in F]); has6 = np.array([f[1] for f in F])
    LG = np.array([it["logit"] for it in allit])
    s3 = np.where(has6, F6[:, 0], sig(LG))
    X5 = np.column_stack([LG, F6, has6])
    io = np.arange(no); inw = np.arange(no, len(allit))
    y_old = np.array([1 if x["mine"] >= 1 else 0 for x in old])
    Y = {"claude": y_old, "claude_2wrong": np.array([1 if x["mine"] == 1 else 0 for x in old]),
         "gemini": np.array([1 if x["gem"] >= 1 else 0 for x in old])}
    ents = np.array([x["ent"] for x in old])
    res = {"n_old": no, "n_new": len(new), "methods": {}}
    res["check_s3_old"] = describe(s3[io], old, Y)
    res["methods"]["1_conf"] = {"mean": describe(sig(LG[io]), old, Y)}
    res["methods"]["3_one_step"] = {"mean": describe(s3[io], old, Y)}
    for m in METHODS:
        res["methods"][f"6_maxprod_{m}"] = {"mean": describe(O[m]["s6"][io], old, Y)}
        res["methods"][f"6b_or_{m}"] = {"mean": describe(np.where(O[m]["s6b"][io] > 0, 0.99, 0.01), old, Y)}
    # 5分割×5回（物で分ける、E6/E7と同じ種）
    runs = {}
    for seed in range(5):
        rng = np.random.default_rng(600 + seed); ue = np.unique(ents); fold_of = {e: k % 5 for k, e in enumerate(rng.permutation(ue))}
        fold = np.array([fold_of[e] for e in ents])
        P = {}; chosen = []
        for k in range(5):
            tr, te = io[fold != k], io[fold == k]; ytr = y_old[fold != k]
            best8 = None
            for m in METHODS:
                o = O[m]
                pr = min(GRID, key=lambda q: ll(o["bp"][q + (0.0,)][tr], ytr))
                add_gamma(o, pr)
                g = min(GAMMAS, key=lambda gg: ll(o["bp"][pr + (gg,)][tr], ytr))
                P.setdefault(f"7_net_{m}", np.zeros(no))[te] = o["bp"][pr + (0.0,)][te]
                P.setdefault(f"7n_net_neg_{m}", np.zeros(no))[te] = o["bp"][pr + (g,)][te]
                X8 = feats(X5, o, o["bp"][pr + (g,)])
                lr = LogisticRegression(max_iter=3000).fit(X8[tr], ytr)
                P.setdefault(f"8_learn_{m}", np.zeros(no))[te] = lr.predict_proba(X8[te])[:, 1]
                trl = ll(lr.predict_proba(X8[tr])[:, 1], ytr)
                if best8 is None or trl < best8[0]: best8 = (trl, m, lr.predict_proba(X8[te])[:, 1])
                chosen.append((k, m, list(pr), g))
            P.setdefault("8_learn_best", np.zeros(no))[te] = best8[2]
            chosen.append((k, "best8", best8[1]))
            P.setdefault("5_learn_E6", np.zeros(no))[te] = LogisticRegression(max_iter=3000).fit(X5[tr], ytr).predict_proba(X5[te])[:, 1]
        for kk, p in P.items():
            runs.setdefault(kk, []).append(describe(p, old, Y))
        runs.setdefault("_chosen", []).append(chosen)
    for kk, ds in runs.items():
        if kk == "_chosen": res["chosen"] = ds; continue
        res["methods"][kk] = {"mean": {m: (float(np.mean([d[m] for d in ds])) if ds[0][m] is not None else None) for m in ds[0]},
                              "per_seed": ds}
    # 新しい300個：700個で全部決めて、1回だけ測る
    ynew = np.array([1 if x["mine"] >= 1 else 0 for x in new])
    best = None
    for m in METHODS:
        a = np.mean([d["aurc_claude"] for d in runs[f"8_learn_{m}"]])
        if best is None or a < best[0]: best = (a, m)
    m = best[1]; o = O[m]
    pr = min(GRID, key=lambda q: ll(o["bp"][q + (0.0,)][io], y_old)); add_gamma(o, pr)
    g = min(GAMMAS, key=lambda gg: ll(o["bp"][pr + (gg,)][io], y_old))
    X8 = feats(X5, o, o["bp"][pr + (g,)])
    p8 = LogisticRegression(max_iter=3000).fit(X8[io], y_old).predict_proba(X8[inw])[:, 1]
    p5 = LogisticRegression(max_iter=3000).fit(X5[io], y_old).predict_proba(X5[inw])[:, 1]
    oA = O["A"]; prA = min(GRID, key=lambda q: ll(oA["bp"][q + (0.0,)][io], y_old)); add_gamma(oA, prA)
    gA = min(GAMMAS, key=lambda gg: ll(oA["bp"][prA + (gg,)][io], y_old))
    p8A = LogisticRegression(max_iter=3000).fit(feats(X5, oA, oA["bp"][prA + (gA,)])[io], y_old).predict_proba(feats(X5, oA, oA["bp"][prA + (gA,)])[inw])[:, 1]
    NP = {"1_conf": sig(LG[inw]), "3_one_step": s3[inw], "5_learn_E6": p5, f"6_maxprod_{m}": o["s6"][inw],
          f"7n_net_neg_{m}": o["bp"][pr + (g,)][inw], f"8_learn_{m}": p8, "8_learn_A": p8A}
    Yn = {"claude": ynew, "claude_2wrong": np.array([1 if x["mine"] == 1 else 0 for x in new])}
    res["new"] = {"chosen_method": m, "params": list(pr) + [g], "methods": {k: describe(v, new, Yn) for k, v in NP.items()}}
    # 物で取り直して差の幅（1000回）
    en = np.array([x["ent"] for x in new]); ue = np.unique(en); rng = np.random.default_rng(8)
    idx_of = {e: np.where(en == e)[0] for e in ue}; diffs = {"8_vs_3": [], "8_vs_1": [], "8_vs_8A": [], "8_vs_5": []}
    for _ in range(1000):
        ii = np.concatenate([idx_of[e] for e in rng.choice(ue, len(ue))])
        a8 = e6.aurc(p8[ii], ynew[ii])
        diffs["8_vs_3"].append(e6.aurc(s3[inw][ii], ynew[ii]) - a8); diffs["8_vs_1"].append(e6.aurc(sig(LG[inw])[ii], ynew[ii]) - a8)
        diffs["8_vs_8A"].append(e6.aurc(p8A[ii], ynew[ii]) - a8); diffs["8_vs_5"].append(e6.aurc(p5[ii], ynew[ii]) - a8)
    res["new"]["boot_diff"] = {k: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in diffs.items()}
    return res, O, allit


def drift_sample(O, allit, L, k=50, seed=88):
    """選び方ごとに、2段以上で着いた道を50本ずつ。選び方を伏せて混ぜる（鍵は別に）。"""
    rng = np.random.default_rng(seed); rows, key = [], []
    for m in METHODS:
        o = O[m]; cand = [i for i in range(len(allit)) if o["dep"][i] >= 2 and o["s6"][i] > 0.5]
        for i in rng.choice(cand, min(k, len(cand)), replace=False):
            p = o["paths"][i]
            rows.append({"stmt": f'{allit[i]["ent"]} -> {allit[i]["catw"]}',
                         "path": [[w, L[w][s]["def"]] for w, s in p]}); key.append([m, int(allit[i]["id"])])
    o = rng.permutation(len(rows))
    return [dict(no=j, **rows[t]) for j, t in enumerate(o)], [[j] + key[t] for j, t in enumerate(o)]


if __name__ == "__main__":
    import time; t = time.time()
    r = json.load(open(sys.argv[1]))
    res, O, allit = run(r)
    json.dump(res, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
    if len(sys.argv) > 3:
        rows, key = drift_sample(O, allit, r["L"])
        json.dump(rows, open(sys.argv[3], "w"), ensure_ascii=False, indent=1)
        json.dump(key, open(sys.argv[3].replace(".json", "_key.json"), "w"))
    print(f"{time.time() - t:.0f}秒")
    print(f"{'方法':22s} 直る  壊すG正  Tにいいえ  AURC常識  AURC Gemini")
    for k, v in res["methods"].items():
        mm = v["mean"]; print(f"{k:22s} {mm['fix_conf_err']:.2f}  {mm['break_G_right']:.2f}  {mm['break_T_right']:.2f}  {mm['aurc_claude']:.3f}  {mm['aurc_gemini']:.3f}")
    print("新しい300個（選んだ方法", res["new"]["chosen_method"], res["new"]["params"], "）")
    for k, mm in res["new"]["methods"].items():
        print(f"{k:22s} {mm['fix_conf_err']}  {mm['break_G_right']}  {mm['break_T_right']}  {mm['aurc_claude']:.3f}")
    print("差（相手−⑧、平均・95%幅）:", res["new"]["boot_diff"])
