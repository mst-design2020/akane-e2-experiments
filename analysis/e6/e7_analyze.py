"""E7 の比べ。python3 e7_analyze.py E7_results.json E6_results.json out.json
正解はE6と同じ（Claudeの常識判定が主、Geminiでも）。"""
import json, sys, os, itertools, numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e6_analyze as e6

DEPTH = 3
sig, lgt = e6.sig, e6.lgt
GRID = list(itertools.product([0.25, 0.5, 1.0], [-1.0, 0.0, 1.0, 2.0], [0.8, 0.95, 0.99]))  # (β, b, ε)


def norm(w):
    return " ".join(str(w).lower().split())


class Item:
    def __init__(self, X, C, G, ST, AR):
        self.X, self.C = X, norm(C)
        self.nodes = sorted(self._reach(G))
        self.idx = {w: i for i, w in enumerate(self.nodes)}
        n = len(self.nodes)
        self.direct = np.zeros(n); self.exact = np.zeros(n, bool)
        self.edges = []  # (i, j, 段の確かさ)
        for w in self.nodes:
            i = self.idx[w]
            self.exact[i] = norm(w) == self.C
            arr = [AR.get((w, k, C), (0.0, 0.0))[0] for k in range(len(G.get(w, [])))]
            self.direct[i] = max(arr) if arr else 0.0
            for k, x in enumerate(G.get(w, [])):
                g = x["genus"]
                if g and g in self.idx and g != w:
                    self.edges.append((i, self.idx[g], ST.get((w, k), (0.0, 0.0))[0]))
        self.G, self.ST, self.AR = G, ST, AR

    def _reach(self, G):
        seen, front = {self.X}, [self.X]
        for _ in range(DEPTH):
            nf = []
            for w in front:
                for x in G.get(w, []):
                    g = x["genus"]
                    if g and g not in seen:
                        seen.add(g); nf.append(g)
            front = nf
        return seen

    def maxprod(self):
        """⑥ 一番強い道の掛け算（Entailer型）と、⑥b 道の「または」（全部の段が0.5超）、一番強い道の段の数。"""
        memo = {}
        def f(w, d):
            key = (w, d)
            if key in memo: return memo[key]
            i = self.idx[w]
            if self.exact[i]: memo[key] = (1.0, True, 0); return memo[key]
            best, ok, dep = 0.0, False, 0
            for k, x in enumerate(self.G.get(w, [])):
                a = self.AR.get((w, k, self.C), self.AR.get((w, k, self.Craw), (0.0, 0.0)))[0]
                if a > best: best, dep = a, 0
                ok = ok or a > 0.5
                g = x["genus"]
                if d > 0 and g and g in self.idx and g != w:
                    e = self.ST.get((w, k), (0.0, 0.0))[0]
                    v, ok2, dp = f(g, d - 1)
                    if e * v > best: best, dep = e * v, dp + 1
                    ok = ok or (e > 0.5 and ok2)
            memo[key] = (best, ok, dep)
            return memo[key]
        return f(self.X, DEPTH)

    def bp(self, beta, b, eps, iters=40, damp=0.5):
        """⑦ 網：点＝「この語はCの一種」、点の手がかり＝直接のNLI、つながり＝段（gがCならwもC、強さ ε×段の確かさ）。輪があれば輪のあるBP。"""
        n = len(self.nodes)
        u = beta * lgt(np.clip(self.direct, 1e-4, 1 - 1e-4)) + b
        u = np.where(self.exact, 30.0, u)
        un = np.stack([np.zeros(n), u], 1)  # log φ(x)
        keep = {}
        for i, j, e in self.edges:  # 同じ2語の間の段（別の意味から）は1つにまとめる：どれかの段が通れば（noisy-OR）
            keep[(i, j)] = keep.get((i, j), 1.0) * (1.0 - eps * e)
        E = [(i, j, np.log(np.array([[1.0, q], [1.0, 1.0]]) + 1e-12)) for (i, j), q in keep.items() if q < 1.0]
        # ψ[x_i, x_j]：x_i=0 かつ x_j=1 のときだけ弱める
        m_ij = {(a, c): np.zeros(2) for a, c, _ in E}; m_ji = {(a, c): np.zeros(2) for a, c, _ in E}
        nb = [[] for _ in range(n)]
        for a, c, _ in E:
            nb[a].append((a, c, "ji")); nb[c].append((a, c, "ij"))  # 点aには c→a の便り、点cには a→c の便り
        def belief(v):
            s = un[v].copy()
            for a, c, dr in nb[v]:
                s += m_ij[(a, c)] if dr == "ij" else m_ji[(a, c)]
            return s
        for _ in range(iters):
            for a, c, lp in E:
                # i→j の便り（i=a から j=c へ）
                ha = belief(a) - m_ji[(a, c)]
                new = np.logaddexp(ha[0] + lp[0], ha[1] + lp[1])  # x_j ごと
                new -= new.max(); m_ij[(a, c)] = damp * m_ij[(a, c)] + (1 - damp) * new
                hc = belief(c) - m_ij[(a, c)]
                new = np.logaddexp(hc[0] + lp[:, 0], hc[1] + lp[:, 1])  # x_i ごと
                new -= new.max(); m_ji[(a, c)] = damp * m_ji[(a, c)] + (1 - damp) * new
        s = belief(self.idx[self.X])
        return float(sig(s[1] - s[0])), len(E), len(self.nodes)


def run(r7, r6):
    G = r7["G"]; ST = {(w, k): (e, c) for w, k, e, c in r7["steps"]}; AR = {(w, k, C): (e, c) for w, k, C, e, c in r7["arrive"]}
    I = {x["id"]: x for x in json.load(open(e6.find("e6_items.json")))}
    R6 = {x["id"]: x for x in r6["items"]}
    ids = sorted(I); n = len(ids)
    items = []
    for i in ids:
        it = Item(I[i]["ent"], I[i]["catw"], G, ST, AR); it.Craw = I[i]["catw"]; items.append(it)
    # E6と同じ数え方の材料
    typ = np.array([I[i]["type"] for i in ids]); mine = np.array([I[i]["mine"] for i in ids]); L = np.array([I[i]["logit"] for i in ids])
    import csv
    g = json.load(open(e6.find("../e5/labels/gemini_items.json", "../e5/gemini_items.json")))
    gl = {int(r_["no"]): int(r_["label"]) for r_ in csv.DictReader(open(e6.find("../e5/labels/gemini_labels.csv", "../e5/gemini_labels.csv")))}
    gem = {x["sid"]: gl[x["no"]] for x in g if x.get("sid") is not None}
    gm = np.array([gem[i] for i in ids]); ents = np.array([I[i]["ent"] for i in ids])
    Y = {"claude": (mine >= 1).astype(int), "claude_2wrong": (mine == 1).astype(int), "gemini": (gm >= 1).astype(int)}
    y = Y["claude"]
    gy = (typ == "G") & (mine == 0) & (L > 0)
    groups = {"fix_conf_err": gy, "fix_G_wrong": (typ == "G") & (mine == 0), "break_T_right": (typ == "T") & (mine == 1),
              "break_G_right": (typ == "G") & (mine >= 1)}

    # E6 ③（同じNLIで1段だけ）を、E6の結果から
    F6, has6 = e6.nli_feats([R6[i] for i in ids], "S2")
    s3_e6 = np.where(has6, F6[:, 0], sig(L))
    s3 = np.array([max([it.AR.get((it.X, k, it.Craw), (0, 0))[0] for k in range(len(G.get(it.X, [])))] or [0]) for it in items])
    mp = [it.maxprod() for it in items]
    s6 = np.array([m[0] for m in mp]); s6b = np.array([1.0 if m[1] else 0.0 for m in mp]); dep = np.array([m[2] for m in mp])
    BP = {}
    for pr in GRID:
        BP[pr] = np.array([it.bp(*pr)[0] for it in items])
    sizes = np.array([[len(it.nodes), len(it.edges)] for it in items])
    loops = np.array([len({(a, c) for a, c, _ in it.edges}) > len(it.nodes) - 1 for it in items])  # 合流（輪）がある

    def describe(p):
        d = {k: float(np.mean(p[m] <= 0.5)) for k, m in groups.items()}
        d["fix_conf_err_n"] = int(groups["fix_conf_err"].sum())
        d.update({f"aurc_{k}": e6.aurc(p, yy) for k, yy in Y.items()})
        d["aurc_claude_loops_only"] = e6.aurc(p[loops], y[loops]) if loops.sum() > 10 else None
        return d

    out = {"n": n, "loops": int(loops.sum()), "mean_nodes": float(sizes[:, 0].mean()), "mean_edges": float(sizes[:, 1].mean()),
           "best_path_depth": {int(k): int(v) for k, v in zip(*np.unique(dep, return_counts=True))},
           "check_s3_vs_e6_maxdiff": float(np.max(np.abs(s3 - F6[:, 0]) * has6)), "methods": {}}
    out["methods"]["3_one_step_E6"] = {"mean": describe(s3_e6)}
    out["methods"]["6_maxprod"] = {"mean": describe(s6)}
    out["methods"]["6b_or_paths"] = {"mean": describe(np.where(s6b > 0, 0.99, 0.01))}
    learned = {}
    for seed in range(5):
        rng = np.random.default_rng(600 + seed); ue = np.unique(ents); fold_of = {e: k % 5 for k, e in enumerate(rng.permutation(ue))}
        fold = np.array([fold_of[e] for e in ents])
        p7 = np.zeros(n); p8 = np.zeros(n); p5 = np.zeros(n); pr_used = []
        X5 = np.column_stack([L, F6, has6])
        for k in range(5):
            tr, te = fold != k, fold == k
            ll = lambda p: -np.mean(y[tr] * np.log(p[tr] + 1e-9) + (1 - y[tr]) * np.log(1 - p[tr] + 1e-9))
            best = min(GRID, key=lambda pr: ll(BP[pr])); pr_used.append(best); p7[te] = BP[best][te]
            X8 = np.column_stack([X5, s6, s6b, dep, BP[best]])
            p8[te] = LogisticRegression(max_iter=3000).fit(X8[tr], y[tr]).predict_proba(X8[te])[:, 1]
            p5[te] = LogisticRegression(max_iter=3000).fit(X5[tr], y[tr]).predict_proba(X5[te])[:, 1]
        learned.setdefault("7_net_bp", []).append((p7, pr_used)); learned.setdefault("8_learn_with_chains", []).append((p8, None))
        learned.setdefault("5_learn_E6", []).append((p5, None))
    for k, runs in learned.items():
        ds = [describe(p) for p, _ in runs]
        out["methods"][k] = {"mean": {m: (float(np.mean([d[m] for d in ds])) if ds[0][m] is not None else None) for m in ds[0]}, "per_seed": ds}
        if runs[0][1] is not None: out["methods"][k]["params"] = [[list(x) for x in pu] for _, pu in runs]
    # ⑦をGemmaなし・格子の真ん中（β=0.5,b=0,ε=0.95）で固定した参考
    out["methods"]["7ref_net_fixed"] = {"mean": describe(BP[(0.5, 0.0, 0.95)])}
    return out


if __name__ == "__main__":
    r7 = json.load(open(sys.argv[1])); r6 = json.load(open(sys.argv[2]))
    import time; t = time.time()
    o = run(r7, r6); json.dump(o, open(sys.argv[3], "w"), ensure_ascii=False, indent=1)
    print(f"{time.time() - t:.0f}秒  輪のある主張 {o['loops']}  点の平均 {o['mean_nodes']:.1f}  一番強い道の段 {o['best_path_depth']}  ③のずれ {o['check_s3_vs_e6_maxdiff']:.3f}")
    print(f"{'方法':24s} 自信ある間違い直る  壊すG正  正しいTにいいえ  AURC常識  AURC Gemini  輪だけAURC")
    for k, v in o["methods"].items():
        m = v["mean"]; lo = m["aurc_claude_loops_only"]
        print(f"{k:24s} {m['fix_conf_err']:.2f}  {m['break_G_right']:.2f}  {m['break_T_right']:.2f}  {m['aurc_claude']:.3f}  {m['aurc_gemini']:.3f}  {lo if lo is None else round(lo, 3)}")
