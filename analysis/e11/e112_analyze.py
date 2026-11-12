"""E11-2 の比べ（計画 plans/E11-2_deep_chain_and_box.md、決まりは流す前に固定）。
python3 e112_analyze.py E8n_results.json E11-1_results.json E11-2_results.json out.json [drift_sample.json]
深い鎖 D4〜D6、箱 BX（E8 の組だけで学習）、埋める R0+BX・Dk+BX、いいとこ取り（木の学習に全部入れる）。"""
import json, sys, os, time, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e111_analyze as Z
import e112_box as BXM
A, e6, sig, judge, GB, REL = Z.A, Z.e6, Z.sig, Z.judge, Z.GB, Z.REL
KS = [4, 5, 6]
NFOLD, SEEDS = 5, range(5)


def items_of(r12, allit, L, ST, AR):
    ch = {i: (nd, lk) for i, nd, lk in r12["chains"]}
    return [A.Item(it, "D", [n[:2] for n in ch[it["id"]][0]], ch[it["id"]][1], L, ST, AR) for it in allit]


def deep_scores(its, k):
    A.DEPTH = k
    out = [x.maxprod() for x in its]
    A.DEPTH = 3
    return np.array([v[0] for v in out]), np.array([v[2] for v in out]), [v[3] for v in out]


def box_scores(allit, its, L, ST, AR, log=print):
    """物の語で5つに分け、その組の問題の「物の語の意味→分類」の着きを外して学ぶ。種5つの平均。"""
    nodes = sorted({n for x in its for n in x.nodes}); cats = sorted({it["catw"] for it in allit})
    idx = {n: i for i, n in enumerate(nodes)}; cidx = {c: len(nodes) + i for i, c in enumerate(cats)}; N = len(nodes) + len(cats)
    step = sorted({(a, b) for x in its for a, cs in x.ch.items() for b in cs if a != b})
    sp = np.array([[idx[a], idx[b]] for a, b in step]); sw = np.array([ST.get(a, (0.0, 0.0))[0] for a, b in step])
    arr = sorted((w, k, C) for (w, k, C) in AR if (w, k) in idx and C in cidx)
    ents = np.array(sorted({it["ent"] for it in allit})); rng = np.random.default_rng(0)
    fold = {e: f % NFOLD for f, e in enumerate(rng.permutation(ents))}
    out = np.zeros((len(SEEDS), len(allit)))
    log(f"箱：点 {N}（分類 {len(cats)}）、段の組 {len(step)}、着きの組 {len(arr)}")
    for f in range(NFOLD):
        drop = {(it["ent"], it["catw"]) for it in allit if fold[it["ent"]] == f}
        keep = [(w, k, C) for (w, k, C) in arr if (w, C) not in drop]
        ap = np.array([[idx[(w, k)], cidx[C]] for w, k, C in keep]); aw = np.array([AR[t][0] for t in keep])
        P = np.vstack([sp, ap]); W = np.concatenate([sw, aw])
        te = [i for i, it in enumerate(allit) if fold[it["ent"]] == f]
        for si, s in enumerate(SEEDS):
            m = BXM.train(N, P, W, seed=100 * f + s)
            for i in te:
                rs = allit[i]["ent"]; x = its[i]
                roots = [idx[r] for r in x.roots if r in idx]
                out[si, i] = BXM.score(m, np.array(roots), np.full(len(roots), cidx[allit[i]["catw"]])).max() if roots else 0.0
        log(f"  組{f + 1}/{NFOLD} 済み（外した着き {len(arr) - len(keep)}）")
    return out.mean(0), out


def fill(chain, box):
    return np.where(chain > 0.5, chain, box)


def counts(s, y):
    return {"yes": int((s > 0.5).sum()), "wrong_yes": int(((s > 0.5) & (y == 0)).sum()), "right_no": int(((s <= 0.5) & (y == 1)).sum())}


def gb_cv(X, y, ents):
    per = []
    for seed in range(5):
        rng = np.random.default_rng(600 + seed); ue = np.unique(ents); fo = {e: k % 5 for k, e in enumerate(rng.permutation(ue))}
        fold = np.array([fo[e] for e in ents]); p = np.zeros(len(y))
        for k in range(5):
            p[fold == k] = GB().fit(X[fold != k], y[fold != k]).predict_proba(X[fold == k])[:, 1]
        per.append(e6.aurc(p, y))
    return per


def run(r8, r11, r12, log=print):
    L8, ST8, AR8, CH8 = A.load(r8)
    old, new = A.labels(); allit = old + new; no = len(old)
    io = np.arange(no); inw = np.arange(no, len(allit))
    y = np.array([int(x["mine"] >= 1) for x in allit]); yo, yn = y[io], y[inw]
    yg = np.array([int(x["gem"] >= 1) for x in old])
    eo = np.array([x["ent"] for x in old]); en = np.array([x["ent"] for x in new])
    LG = np.array([it["logit"] for it in allit])
    its8 = [A.Item(it, "D", *CH8[(it["id"], "D")], L8, ST8, AR8) for it in allit]
    R0 = np.array([x.maxprod()[0] for x in its8]); dep0 = np.array([x.maxprod()[2] for x in its8])
    S11, _ = Z.scores(allit, its8, r11["ent_p"], r11.get("leak_p", {}))
    L = r12["L"]; ST = {(w, k): (e, c) for w, k, e, c in r12["steps"]}; AR = {(w, k, C): (e, c) for w, k, C, e, c in r12["arrive"]}
    its = items_of(r12, allit, L, ST, AR)
    S, DEP, PATH = {"R0_all": R0}, {"R0_all": dep0}, {}
    d3, _, _ = deep_scores(its, 3)
    res = {"check_D3_equals_R0": float(np.abs(d3 - R0).max())}
    for k in KS:
        S[f"D{k}"], DEP[f"D{k}"], PATH[f"D{k}"] = deep_scores(its, k)
    t = time.time(); BX, BXs = box_scores(allit, its, L, ST, AR, log); log(f"箱 {time.time() - t:.0f}秒")
    S["BX"] = BX; DEP["BX"] = np.zeros(len(allit))
    kbest = min(KS, key=lambda k: e6.aurc(S[f"D{k}"][io], yo))
    Dk = f"D{kbest}"
    S["R0+BX"] = fill(R0, BX); S[f"{Dk}+BX"] = fill(S[Dk], BX)
    res["k_chosen"] = kbest
    res["old"] = {m: {"aurc_claude": e6.aurc(s[io], yo), "aurc_gemini": e6.aurc(s[io], yg)} | counts(s[io], yo) for m, s in S.items()}
    res["old"]["1_conf"] = {"aurc_claude": e6.aurc(sig(LG[io]), yo), "aurc_gemini": e6.aurc(sig(LG[io]), yg)}
    res["box_seed_spread_old"] = [e6.aurc(b[io], yo) for b in BXs]
    pairs = {"1_deep_vs_R0": (Dk, "R0_all"), "2_R0BX_vs_R0": ("R0+BX", "R0_all"), "3_DkBX_vs_Dk": (f"{Dk}+BX", Dk),
             "ref_BX_vs_R0": ("BX", "R0_all")}
    res["cmp"] = {}
    for lab, yy in (("claude", yo), ("gemini", yg)):
        for k, (a, b) in pairs.items():
            res["cmp"][f"{k}_{lab}"] = judge(S[a][io], S[b][io], yy, eo, 41)
    # 段ごとの「はい」の正しさ（D6 の一番強い道の段の数）
    s6, d6 = S["D6"], DEP["D6"]
    res["yes_by_depth"] = {int(d): {"yes": int(((s6 > 0.5) & (d6 == d)).sum()), "right": int(((s6 > 0.5) & (d6 == d) & (y == 1)).sum())}
                           for d in range(7)}
    # 新しい300問：700問で一番良かった物を、R0 と自信と比べる（1回だけ）
    cands = [Dk, "R0+BX", f"{Dk}+BX"]
    best = min(cands, key=lambda m: e6.aurc(S[m][io], yo))
    res["new"] = {"chosen": best, "aurc": {m: e6.aurc(s[inw], yn) for m, s in S.items()} | {"1_conf": e6.aurc(sig(LG[inw]), yn)},
                  "counts": {m: counts(s[inw], yn) for m, s in S.items()},
                  "vs_R0": judge(S[best][inw], R0[inw], yn, en, 51), "vs_conf": judge(S[best][inw], sig(LG[inw]), yn, en, 52)}
    # いいとこ取り：木の学習に全部入れる（E6の特徴＋R0・深い鎖・箱・E11-1の重み）。相手は E6の特徴＋R0（E8n と同じ形）
    F = [A.x_feats(it, L8, AR8) for it in allit]
    X5 = np.column_stack([LG, np.array([f[0] for f in F]), np.array([f[1] for f in F])])
    X0 = np.column_stack([X5, R0, dep0])
    XA = np.column_stack([X5, R0, dep0, S[Dk], DEP[Dk], BX, S11["R3_weight"]])
    p0, pa = gb_cv(X0[io], yo, eo), gb_cv(XA[io], yo, eo)
    wins = sum(b <= a * (1 - REL) for a, b in zip(p0, pa))
    q0 = GB().fit(X0[io], yo).predict_proba(X0[inw])[:, 1]; qa = GB().fit(XA[io], yo).predict_proba(XA[inw])[:, 1]
    res["best_of_all"] = {"old_per_seed": {"GB_R0": p0, "GB_all": pa}, "wins_of_5": int(wins), "better_old": bool(wins >= 4),
                          "new_GB_R0": e6.aurc(q0, yn), "new_GB_all": e6.aurc(qa, yn), "new_vs": judge(qa, q0, yn, en, 61),
                          "new_vs_conf": judge(qa, sig(LG[inw]), yn, en, 62)}
    res["flags"] = {"deep": bool(res["cmp"]["1_deep_vs_R0_claude"]["better"] and best == Dk and res["new"]["vs_R0"]["better"]),
                    "box": bool((res["cmp"]["2_R0BX_vs_R0_claude"]["better"] or res["cmp"]["3_DkBX_vs_Dk_claude"]["better"])
                                and best != Dk and res["new"]["vs_R0"]["better"]),
                    "best_of_all": bool(res["best_of_all"]["better_old"] and res["best_of_all"]["new_vs"]["better"])}
    return res, allit, S, DEP, PATH, L, Dk


def drift_sample(allit, S, DEP, PATH, L, k=50, seed=88):
    """D6 で4段以上の道で「はい」になった物から50本（Claude が意味の取り違えを読む）。"""
    s, d, P = S["D6"], DEP["D6"], PATH["D6"]
    cand = [i for i in range(len(allit)) if s[i] > 0.5 and d[i] >= 4]
    rng = np.random.default_rng(seed)
    pick = rng.choice(cand, min(k, len(cand)), replace=False) if cand else []
    return [{"no": j, "id": int(allit[i]["id"]), "stmt": f'{allit[i]["ent"]} -> {allit[i]["catw"]}',
             "path": [[w, L[w][kk].get("def")] for w, kk in P[i]]} for j, i in enumerate(pick)]


if __name__ == "__main__":
    t = time.time()
    r8, r11, r12 = (json.load(open(p)) for p in sys.argv[1:4])
    res, allit, S, DEP, PATH, L, Dk = run(r8, r11, r12)
    json.dump(res, open(sys.argv[4], "w"), ensure_ascii=False, indent=1)
    if len(sys.argv) > 5:
        json.dump(drift_sample(allit, S, DEP, PATH, L), open(sys.argv[5], "w"), ensure_ascii=False, indent=1)
    print(f"{time.time() - t:.0f}秒  D3=R0 の確かめ（最大のずれ）:", res["check_D3_equals_R0"], " 選んだ段:", res["k_chosen"])
    for m, d in res["old"].items():
        print(f"  700 {m:10s} 常識 {d['aurc_claude']:.3f} Gemini {d['aurc_gemini']:.3f}", {k: v for k, v in d.items() if not k.startswith("aurc")})
    for k, v in res["cmp"].items():
        print(f"  {k:24s} a {v['aurc_a']:.3f} b {v['aurc_b']:.3f} 差 {[round(x, 3) for x in v['diff']]} 良い {v['better']} 悪い {v['worse']}")
    print("段ごとの「はい」（D6）:", res["yes_by_depth"])
    print("300問:", res["new"]["chosen"], {k: round(v, 3) for k, v in res["new"]["aurc"].items()})
    print("  R0より良い", res["new"]["vs_R0"]["better"], "自信より良い", res["new"]["vs_conf"]["better"])
    b = res["best_of_all"]
    print("いいとこ取り 700:", [round(x, 3) for x in b["old_per_seed"]["GB_R0"]], "→", [round(x, 3) for x in b["old_per_seed"]["GB_all"]],
          "勝ち", b["wins_of_5"], "/5  300:", round(b["new_GB_R0"], 3), "→", round(b["new_GB_all"], 3), "良い", b["new_vs"]["better"])
    print("旗:", res["flags"])
