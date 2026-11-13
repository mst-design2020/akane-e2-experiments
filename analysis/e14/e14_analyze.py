"""E14 の比べ（計画 plans/E14_fill_gaps.md、決まりは流す前に固定）。
python3 e14_analyze.py E8n_results.json E11-1_results.json E11-2_results.json E14_results.json(.gz) out.json [drift.json]
⑥N（E8n 方法D、3段）、⑥N6（E11-2 の6段）、⑭g・⑭s・⑭gs（継ぎ足し、6段）、⑪x（同じ文を直接見せる）、学習あり。"""
import json, sys, os, gzip, time, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "e11"))
import e112_analyze as E2
Z, A, e6, sig, judge, GB, REL = E2.Z, E2.A, E2.e6, E2.sig, E2.judge, E2.GB, E2.REL
import importlib.util
_sp = importlib.util.spec_from_file_location("e14_core_mod", os.path.join(HERE, "e14_core.py"))
C14 = importlib.util.module_from_spec(_sp)
C14.genus_candidates = None   # follow しか使わない
_sp.loader.exec_module(C14)

VARIANTS = {"N3": ((), 3), "N6": ((), 6), "G": (("g",), 6), "S": (("s",), 6), "GS": (("g", "s"), 6)}


def jload(p):
    return json.load(gzip.open(p, "rt", encoding="utf-8")) if p.endswith(".gz") else json.load(open(p))


class Graph:
    """1問の鎖（資料を選んで作り直す）。一番強い道の掛け算（E8n と同じ形、つながりごとに強さ）。"""
    def __init__(self, it, r, srcs, maxd):
        self.x, self.C, self.maxd = it["ent"], it["catw"], maxd
        self.L, self.GR = r["L"], r["GR"]
        dep, links = C14.follow(self.L, r["CH"], self.GR, self.x, maxd, srcs)
        self.dep = dep
        self.ch = collections.defaultdict(list)
        for a, b in links:
            self.ch[a].append(b)
        self.ST, self.AR, self.ARG = r["ST"], r["AR"], r["ARG"]
        self.roots = [n for n, d in dep.items() if d == 0 and n[1] == self.x]

    def arr(self, n):
        t, w, k = n
        if t == "s":
            return 1.0 if w == self.C else self.AR.get((w, k, self.C), (0.0, 0.0))[0]
        return self.ARG.get((w, k, self.C), (0.0, 0.0))[0]

    def edge(self, a, b):
        """強さと段の数。"""
        if a[0] == "s" and b[0] == "s":
            return self.ST.get((a[1], a[2]), (0.0, 0.0))[0], 1
        if a[0] == "s" and b[0] == "g":
            return 1.0, 0
        return self.GR[a[1]][a[2]]["step"][0], 1

    def maxprod(self):
        memo = {}
        def f(n, d):
            key = (n, d)
            if key in memo: return memo[key]
            a = self.arr(n); best, path = a, [n]
            if a < 1.0:
                for c in self.ch[n]:
                    e, cost = self.edge(n, c)
                    if cost > d: continue
                    v, p2 = f(c, d - cost)
                    if e * v > best: best, path = e * v, [n] + p2
            memo[key] = (best, path)
            return memo[key]
        rs = [f(r, self.maxd) for r in self.roots] or [(0.0, [])]
        b = max(rs, key=lambda r: r[0])
        steps = sum(self.edge(p, q)[1] for p, q in zip(b[1], b[1][1:]))
        return b[0], steps, b[1], any(n[0] == "g" for n in b[1])


def counts(s, y):
    return {"yes": int((s > 0.5).sum()), "wrong_yes": int(((s > 0.5) & (y == 0)).sum()), "right_no": int(((s <= 0.5) & (y == 1)).sum())}


def run(r8, r11, r12, r14, log=print):
    old, new = A.labels(); allit = old + new; no = len(old)
    io = np.arange(no); inw = np.arange(no, len(allit))
    y = np.array([int(x["mine"] >= 1) for x in allit]); yo, yn = y[io], y[inw]
    yg = np.array([int(x["gem"] >= 1) for x in old])
    eo = np.array([x["ent"] for x in old]); en = np.array([x["ent"] for x in new])
    LG = np.array([it["logit"] for it in allit])
    r = {"L": r14["L"], "GR": r14["GR"],
         "CH": {(w, k): v for w, k, v in r14["choice"]},
         "ST": {(w, k): tuple(v) for w, k, *v in r14["steps"]},
         "AR": {(w, k, C): tuple(v) for w, k, C, *v in r14["arrive"]},
         "ARG": {(w, i, C): tuple(v) for w, i, C, *v in r14["arrive_g"]}}
    S, DEP, PATH, USE = {}, {}, {}, {}
    for m, (srcs, d) in VARIANTS.items():
        out = [Graph(it, r, srcs, d).maxprod() for it in allit]
        S[m] = np.array([o[0] for o in out]); DEP[m] = np.array([o[1] for o in out])
        PATH[m] = [o[2] for o in out]; USE[m] = np.array([o[3] for o in out])
    # 確かめ：N3 は E8n の ⑥D、N6 は E11-2 の D6 と同じ値
    L8, ST8, AR8, CH8 = A.load(r8)
    R0 = np.array([A.Item(it, "D", *CH8[(it["id"], "D")], L8, ST8, AR8).maxprod()[0] for it in allit])
    L12 = r12["L"]; ST12 = {(w, k): (e, c) for w, k, e, c in r12["steps"]}; AR12 = {(w, k, C): (e, c) for w, k, C, e, c in r12["arrive"]}
    its12 = E2.items_of(r12, allit, L12, ST12, AR12)
    D6, _, _ = E2.deep_scores(its12, 6)
    res = {"check_N3_vs_E8n": float(np.abs(S["N3"] - R0).max()), "check_N6_vs_E11-2": float(np.abs(S["N6"] - D6).max())}
    # 直接見せる（⑪x）
    dr = r14["direct"]; pos = {i: j for j, i in enumerate(dr["ids"])}
    S["X_g"] = sig(np.array([dr["g"][pos[it["id"]]] for it in allit]))
    S["X_gs"] = sig(np.array([dr["gs"][pos[it["id"]]] for it in allit]))
    S["1_conf"] = sig(LG)
    res["old"] = {m: {"aurc_claude": e6.aurc(s[io], yo), "aurc_gemini": e6.aurc(s[io], yg)} | counts(s[io], yo) for m, s in S.items()}
    res["new"] = {m: {"aurc": e6.aurc(s[inw], yn)} | counts(s[inw], yn) for m, s in S.items()}
    # 主な比べ・確かめ（先に固定）
    res["main_700"] = judge(S["GS"][io], S["N3"][io], yo, eo, 141)
    res["main_300"] = judge(S["GS"][inw], S["N3"][inw], yn, en, 142)
    res["sub"] = {"0_GS_vs_N6_700": judge(S["GS"][io], S["N6"][io], yo, eo, 143),
                  "0_GS_vs_N6_300": judge(S["GS"][inw], S["N6"][inw], yn, en, 144),
                  "1_G_vs_S_700": judge(S["G"][io], S["S"][io], yo, eo, 145),
                  "1_S_vs_G_700": judge(S["S"][io], S["G"][io], yo, eo, 146),
                  "2_GS_vs_Xg_700": judge(S["GS"][io], S["X_g"][io], yo, eo, 147),
                  "2_GS_vs_Xg_300": judge(S["GS"][inw], S["X_g"][inw], yn, en, 148),
                  "3_GS_vs_N3_700_gemini": judge(S["GS"][io], S["N3"][io], yg, eo, 149),
                  "ref_G_vs_N3_700": judge(S["G"][io], S["N3"][io], yo, eo, 150),
                  "ref_S_vs_N3_700": judge(S["S"][io], S["N3"][io], yo, eo, 151),
                  "ref_GS_vs_conf_300": judge(S["GS"][inw], S["1_conf"][inw], yn, en, 152)}
    # 数え：正しい答えへの「いいえ」（N3）が「はい」に変わった数、間違った「はい」の増え
    n3, gs = S["N3"] > 0.5, S["GS"] > 0.5
    res["flip"] = {part: {"right_no_N3": int((~n3[ix] & (y[ix] == 1)).sum()),
                          "right_no_fixed": int((~n3[ix] & gs[ix] & (y[ix] == 1)).sum()),
                          "wrong_yes_N3": int((n3[ix] & (y[ix] == 0)).sum()),
                          "wrong_yes_new": int((~n3[ix] & gs[ix] & (y[ix] == 0)).sum()),
                          "right_yes_lost": int((n3[ix] & ~gs[ix] & (y[ix] == 1)).sum()),
                          "best_path_uses_graft": int(USE["GS"][ix].sum())}
                   for part, ix in (("700", io), ("300", inw))}
    st = r14["meta"].get("stat", {})
    GRs = r14["GR"]
    res["graft_stats"] = {"words": len(GRs), "with_g": sum(1 for v in GRs.values() if any(g["src"] == "g" for g in v)),
                          "with_s": sum(1 for v in GRs.values() if any(g["src"] == "s" for g in v)),
                          "ent_with_g": sum(1 for it in allit if any(g["src"] == "g" for g in GRs.get(it["ent"], []))),
                          "ent_with_s": sum(1 for it in allit if any(g["src"] == "s" for g in GRs.get(it["ent"], []))),
                          "grow": st}
    # 学習あり：E11-2 のいいとこ取り（E6の特徴＋R0・深い鎖・箱・E11-1の重み）に ⑭gs を足すか
    its8 = [A.Item(it, "D", *CH8[(it["id"], "D")], L8, ST8, AR8) for it in allit]
    dep0 = np.array([x.maxprod()[2] for x in its8])
    S11, _ = Z.scores(allit, its8, r11["ent_p"], r11.get("leak_p", {}))
    t = time.time(); BX, _ = E2.box_scores(allit, its12, L12, ST12, AR12, log); log(f"箱 {time.time() - t:.0f}秒")
    D6s, D6d, _ = E2.deep_scores(its12, 6)
    F = [A.x_feats(it, L8, AR8) for it in allit]
    X5 = np.column_stack([LG, np.array([f[0] for f in F]), np.array([f[1] for f in F])])
    XA = np.column_stack([X5, R0, dep0, D6s, D6d, BX, S11["R3_weight"]])
    XB = np.column_stack([XA, S["GS"], DEP["GS"], S["X_g"]])
    pa, pb = E2.gb_cv(XA[io], yo, eo), E2.gb_cv(XB[io], yo, eo)
    wins = sum(b <= a * (1 - REL) for a, b in zip(pa, pb))
    qa = GB().fit(XA[io], yo).predict_proba(XA[inw])[:, 1]; qb = GB().fit(XB[io], yo).predict_proba(XB[inw])[:, 1]
    res["learn"] = {"old_per_seed": {"all": pa, "all+E14": pb}, "wins_of_5": int(wins), "better_old": bool(wins >= 4),
                    "new_all": e6.aurc(qa, yn), "new_all+E14": e6.aurc(qb, yn), "new_vs": judge(qb, qa, yn, en, 161)}
    res["flags"] = {"filled": bool(res["main_700"]["better"] and res["main_300"]["better"]),
                    "filled_700_only": bool(res["main_700"]["better"] and not res["main_300"]["better"]),
                    "learn": bool(res["learn"]["better_old"] and res["learn"]["new_vs"]["better"])}
    return res, allit, S, DEP, PATH, r


def drift_sample(allit, S, PATH, r, k=50, seed=144):
    """⑭gs で、継ぎ足しを通る道で「はい」になった物から50本（Claude が方法を伏せて取り違えを読む）。"""
    cand = [i for i in range(len(allit)) if S["GS"][i] > 0.5 and any(n[0] == "g" for n in PATH["GS"][i])]
    rng = np.random.default_rng(seed)
    pick = rng.choice(cand, min(k, len(cand)), replace=False) if cand else []
    def txt(n):
        t, w, k_ = n
        return [w, r["L"][w][k_].get("def", "")] if t == "s" else [w, "（文）" + r["GR"][w][k_].get("sent", "")]
    return [{"no": j, "id": int(allit[i]["id"]), "stmt": f'{allit[i]["ent"]} -> {allit[i]["catw"]}',
             "path": [txt(n) for n in PATH["GS"][i]]} for j, i in enumerate(pick)]


if __name__ == "__main__":
    t = time.time()
    r8, r11, r12, r14 = (jload(p) for p in sys.argv[1:5])
    res, allit, S, DEP, PATH, r = run(r8, r11, r12, r14)
    json.dump(res, open(sys.argv[5], "w"), ensure_ascii=False, indent=1)
    if len(sys.argv) > 6:
        json.dump(drift_sample(allit, S, PATH, r), open(sys.argv[6], "w"), ensure_ascii=False, indent=1)
    print(f"{time.time() - t:.0f}秒  確かめ N3=E8n:", res["check_N3_vs_E8n"], " N6=E11-2:", res["check_N6_vs_E11-2"])
    for m in S:
        o, n = res["old"][m], res["new"][m]
        print(f"  {m:7s} 700 常識 {o['aurc_claude']:.3f} Gemini {o['aurc_gemini']:.3f}  300 {n['aurc']:.3f}",
              {k: v for k, v in o.items() if not k.startswith("aurc")})
    for k in ("main_700", "main_300"):
        v = res[k]; print(f"  {k:10s} a {v['aurc_a']:.3f} b {v['aurc_b']:.3f} 差 {[round(x, 3) for x in v['diff']]} 良い {v['better']} 悪い {v['worse']}")
    for k, v in res["sub"].items():
        print(f"  {k:26s} a {v['aurc_a']:.3f} b {v['aurc_b']:.3f} 差 {[round(x, 3) for x in v['diff']]} 良い {v['better']} 悪い {v['worse']}")
    print("数え:", res["flip"]); print("文:", res["graft_stats"])
    b = res["learn"]
    print("学習 700:", [round(x, 3) for x in b["old_per_seed"]["all"]], "→", [round(x, 3) for x in b["old_per_seed"]["all+E14"]],
          "勝ち", b["wins_of_5"], "/5  300:", round(b["new_all"], 3), "→", round(b["new_all+E14"], 3), "良い", b["new_vs"]["better"])
    print("旗:", res["flags"])
