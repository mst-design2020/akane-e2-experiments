"""E14-2 の比べ（PLAN_E14-2.md、決まりは流す前に固定）。継ぎ足しは1つの道で1回まで（W1）。
E8_DIR=../e8 python3 e142_compare.py E8n_results.json E14-2_for_analysis.json.gz out.json [drift.json]"""
import sys, os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import e14_analyze as M
A, e6, judge, sig = M.A, M.e6, M.judge, M.sig


class GraphB(M.Graph):
    """継ぎ足しの文の点（"g" の点）を通れる回数に上限を付けた一番強い道の掛け算。"""
    def __init__(self, it, r, srcs, maxd, max_g):
        super().__init__(it, r, srcs, maxd); self.max_g = max_g

    def maxprod(self):
        memo = {}
        def f(n, d, gl):
            key = (n, d, gl)
            if key in memo: return memo[key]
            a = self.arr(n); best, path = a, [n]
            if a < 1.0:
                for c in self.ch[n]:
                    e, cost = self.edge(n, c)
                    g2 = gl - (c[0] == "g")
                    if cost > d or g2 < 0: continue
                    v, p2 = f(c, d - cost, g2)
                    if e * v > best: best, path = e * v, [n] + p2
            memo[key] = (best, path)
            return memo[key]
        rs = [f(r, self.maxd, self.max_g - (r[0] == "g")) for r in self.roots if self.max_g - (r[0] == "g") >= 0] or [(0.0, [])]
        b = max(rs, key=lambda x: x[0])
        return b[0], b[1]


VAR = {"N3": ((), 3, 0), "N6": ((), 6, 0), "W1": (("w",), 6, 1), "W": (("w",), 6, 99)}

if __name__ == "__main__":
    r8, r14 = M.jload(sys.argv[1]), M.jload(sys.argv[2])
    old, new = A.labels(); allit = old + new; no = len(old)
    io, inw = np.arange(no), np.arange(no, len(allit))
    y = np.array([int(x["mine"] >= 1) for x in allit]); yg = np.array([int(x["gem"] >= 1) for x in old])
    eo = np.array([x["ent"] for x in old]); en = np.array([x["ent"] for x in new])
    r = {"L": r14["L"], "GR": r14["GR"], "CH": {(w, k): v for w, k, v in r14["choice"]},
         "ST": {(w, k): tuple(v) for w, k, *v in r14["steps"]}, "AR": {(w, k, C): tuple(v) for w, k, C, *v in r14["arrive"]},
         "ARG": {(w, i, C): tuple(v) for w, i, C, *v in r14["arrive_g"]}}
    S, P = {}, {}
    for m, (srcs, d, g) in VAR.items():
        out = [GraphB(it, r, srcs, d, g).maxprod() for it in allit]
        S[m] = np.array([o[0] for o in out]); P[m] = [o[1] for o in out]
    S["1_conf"] = sig(np.array([it["logit"] for it in allit]))
    L8, ST8, AR8, CH8 = A.load(r8)
    R0 = np.array([A.Item(it, "D", *CH8[(it["id"], "D")], L8, ST8, AR8).maxprod()[0] for it in allit])
    def cnt(s, ix):
        return {"yes": int((s[ix] > 0.5).sum()), "wrong_yes": int(((s[ix] > 0.5) & (y[ix] == 0)).sum()),
                "right_no": int(((s[ix] <= 0.5) & (y[ix] == 1)).sum())}
    res = {"check_N3_vs_E8n": float(np.abs(S["N3"] - R0).max()),
           "aurc": {m: {"700": e6.aurc(s[io], y[io]), "700_gemini": e6.aurc(s[io], yg), "300": e6.aurc(s[inw], y[inw]),
                        "c700": cnt(s, io), "c300": cnt(s, inw)} for m, s in S.items()}}
    PR = {"main_700": ("W1", "N3", io), "main_300": ("W1", "N3", inw), "sub_W1_vs_N6_700": ("W1", "N6", io),
          "sub_W1_vs_N6_300": ("W1", "N6", inw), "sub_W_vs_N3_700": ("W", "N3", io), "sub_W1_vs_W_700": ("W1", "W", io)}
    res["cmp"] = {k: judge(S[p][ix], S[q][ix], y[ix], eo if ix is io else en, 500 + n) for n, (k, (p, q, ix)) in enumerate(PR.items())}
    res["cmp"]["sub_W1_vs_N3_700_gemini"] = judge(S["W1"][io], S["N3"][io], yg, eo, 599)
    n3, w1 = S["N3"] > 0.5, S["W1"] > 0.5
    res["flip"] = {part: {"right_no_N3": int((~n3[ix] & (y[ix] == 1)).sum()), "right_no_fixed": int((~n3[ix] & w1[ix] & (y[ix] == 1)).sum()),
                          "wrong_yes_new": int((~n3[ix] & w1[ix] & (y[ix] == 0)).sum()), "right_yes_lost": int((n3[ix] & ~w1[ix] & (y[ix] == 1)).sum())}
                   for part, ix in (("700", io), ("300", inw))}
    GR = r14["GR"]
    res["wiki"] = {"words": len(GR), "with_w": sum(1 for v in GR.values() if any(g["src"] == "w" for g in v)),
                   "ent_with_w": sum(1 for it in allit if any(g["src"] == "w" for g in GR.get(it["ent"], []))), "grow": r14["meta"].get("stat")}
    res["flag_filled"] = bool(res["cmp"]["main_700"]["better"] and res["cmp"]["main_300"]["better"])
    json.dump(res, open(sys.argv[3], "w"), ensure_ascii=False, indent=1)
    if len(sys.argv) > 4:
        cand = [i for i in range(len(allit)) if S["W1"][i] > 0.5 and not n3[i]]
        rng = np.random.default_rng(142); pick = rng.choice(cand, min(50, len(cand)), replace=False) if cand else []
        def txt(n):
            t, w, k = n
            return [w, r["L"][w][k].get("def", "")] if t == "s" else [w, "（文）" + r["GR"][w][k].get("sent", "")]
        json.dump([{"no": j, "id": int(allit[i]["id"]), "stmt": f'{allit[i]["ent"]} -> {allit[i]["catw"]}', "y": int(y[i]),
                    "path": [txt(n) for n in P["W1"][i]]} for j, i in enumerate(pick)], open(sys.argv[4], "w"), ensure_ascii=False, indent=1)
    print("N3=E8n:", res["check_N3_vs_E8n"])
    for m, v in res["aurc"].items():
        print(f"  {m:7s} 700 {v['700']:.3f} Gemini {v['700_gemini']:.3f} 300 {v['300']:.3f}", v["c700"], v["c300"])
    for k, v in res["cmp"].items():
        print(f"  {k:26s} a {v['aurc_a']:.3f} b {v['aurc_b']:.3f} 差 {[round(x, 3) for x in v['diff']]} 良い {v['better']} 悪い {v['worse']}")
    print("数え:", res["flip"]); print("Wikipedia:", res["wiki"]); print("旗:", res["flag_filled"])
