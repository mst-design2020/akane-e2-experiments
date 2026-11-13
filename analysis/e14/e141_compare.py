"""E14-1 の比べ（PLAN_E14-1.md、決まりは流す前に固定）。学習なしの比べだけ（箱を使わないので速い）。
E8_DIR=../e8 python3 e141_compare.py E8n_results.json E14_for_analysis.json.gz E14-1_for_analysis.json.gz out.json"""
import sys, os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import e14_analyze as M
A, e6, judge, sig = M.A, M.e6, M.judge, M.sig


def scores(r14, allit):
    r = {"L": r14["L"], "GR": r14["GR"], "CH": {(w, k): v for w, k, v in r14["choice"]},
         "ST": {(w, k): tuple(v) for w, k, *v in r14["steps"]}, "AR": {(w, k, C): tuple(v) for w, k, C, *v in r14["arrive"]},
         "ARG": {(w, i, C): tuple(v) for w, i, C, *v in r14["arrive_g"]}}
    return {m: np.array([M.Graph(it, r, s, d).maxprod()[0] for it in allit]) for m, (s, d) in M.VARIANTS.items()}


if __name__ == "__main__":
    r8, a, b = (M.jload(p) for p in sys.argv[1:4])
    old, new = A.labels(); allit = old + new; no = len(old)
    io, inw = np.arange(no), np.arange(no, len(allit))
    y = np.array([int(x["mine"] >= 1) for x in allit]); yg = np.array([int(x["gem"] >= 1) for x in old])
    eo = np.array([x["ent"] for x in old]); en = np.array([x["ent"] for x in new])
    S0, S1 = scores(a, allit), scores(b, allit)
    S = {"N3": S0["N3"], "GS": S0["GS"], "G": S0["G"], "S": S0["S"], "GS'": S1["GS"], "G'": S1["G"], "S'": S1["S"],
         "1_conf": sig(np.array([it["logit"] for it in allit]))}
    def cnt(s, ix):
        return {"yes": int((s[ix] > 0.5).sum()), "wrong_yes": int(((s[ix] > 0.5) & (y[ix] == 0)).sum()),
                "right_no": int(((s[ix] <= 0.5) & (y[ix] == 1)).sum())}
    res = {"check_N3_same": float(np.abs(S0["N3"] - S1["N3"]).max()),
           "aurc": {m: {"700": e6.aurc(s[io], y[io]), "700_gemini": e6.aurc(s[io], yg), "300": e6.aurc(s[inw], y[inw]),
                        "c700": cnt(s, io), "c300": cnt(s, inw)} for m, s in S.items()}}
    P = {"main_700": ("GS'", "N3", io), "main_300": ("GS'", "N3", inw),
         "sub_GS'_vs_GS_700": ("GS'", "GS", io), "sub_GS'_vs_GS_300": ("GS'", "GS", inw),
         "sub_S'_vs_S_700": ("S'", "S", io), "sub_S'_vs_S_300": ("S'", "S", inw),
         "sub_G'_vs_G_700": ("G'", "G", io), "sub_G'_vs_S_700": ("G'", "S", io)}
    res["cmp"] = {k: judge(S[p][ix], S[q][ix], y[ix], eo if ix is io else en, 300 + n) for n, (k, (p, q, ix)) in enumerate(P.items())}
    res["cmp"]["sub_GS'_vs_N3_700_gemini"] = judge(S["GS'"][io], S["N3"][io], yg, eo, 399)
    res["flag_candidate"] = bool(res["cmp"]["main_700"]["better"] and res["cmp"]["main_300"]["better"])
    json.dump(res, open(sys.argv[4], "w"), ensure_ascii=False, indent=1)
    print("N3 同じ:", res["check_N3_same"])
    for m, v in res["aurc"].items():
        print(f"  {m:7s} 700 {v['700']:.3f} Gemini {v['700_gemini']:.3f} 300 {v['300']:.3f}", v["c700"], v["c300"])
    for k, v in res["cmp"].items():
        print(f"  {k:28s} a {v['aurc_a']:.3f} b {v['aurc_b']:.3f} 差 {[round(x, 3) for x in v['diff']]} 良い {v['better']} 悪い {v['worse']}")
    print("旗（候補が見つかった）:", res["flag_candidate"])
