"""E15 の比べ（plans/E15_last_step.md、決まりは流す前に固定）。
E8_DIR=analysis/e8 python3 e15_analyze.py E15_results.json.gz out.json
- 作り方：親の語 n（NLI、今まで）/ g（Gemma）× 最後の段 NLI（今まで）/ path（道を全部見せた Gemma）/ last / ctrl、と none（説明文なし）。
- 点 = 道の強さの掛け算 × 着き の一番大きい物（E8n と同じ形）。点の語が分類の語そのものなら着き 1。
- 確かめ（旗）：E10-3 の「一種か」600問（常識採点 cs_K_claude）。E8 の 1,000問は直し方を見るだけ（旗なし）。"""
import sys, os, json, gzip, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "e14"))
import e15_core as F
import e14_analyze as M
A, e6, judge, sig = M.A, M.e6, M.judge, M.sig
KALL = json.load(open(os.path.join(HERE, "..", "e10", "e103", "k_items.json")))["items"]
KCS = json.load(open(os.path.join(HERE, "..", "e10", "e103", "cs_K_claude.json")))


def scores(res, sn):
    L = res["L"]
    CH = {m: {(w, k): v for w, k, v in res["CH"][m]} for m in "ng"}
    ST = {m: {(w, k): tuple(v) for w, k, *v in res["ST"][m]} for m in "ng"}
    AR = {(w, k, C): v[0] for w, k, C, *v in res["AR"]}
    GA = collections.defaultdict(dict); NONE = {}
    for s, i, mode, var, w, k, v in res["GA"]:
        if s != sn: continue
        p = float(1 / (1 + np.exp(-v)))
        if var == "none": NONE[i] = p
        else: GA[(i, mode, var)][(w, k)] = p
    its = res["items"][sn]
    S = collections.defaultdict(list)
    for it in its:
        x, C, i = it["ent"], it["catw"], it["id"]
        for mode in "ng":
            best = F.reach(L, CH, ST, mode, x)
            one = lambda n: 1.0 if n[0] == C else None
            arr_n = {n: (one(n) or AR.get((n[0], n[1], C), 0.0)) for n in best}
            S[f"{mode}_nli"].append(F.score(best, arr_n, list(best)))
            for var in ("path", "last", "ctrl"):
                a = {n: (one(n) or GA[(i, mode, var)].get(n, 0.0)) for n in best}
                S[f"{mode}_{var}"].append(F.score(best, a, list(best)))
        S["none"].append(NONE.get(i, 0.0))
    return {k: np.array(v) for k, v in S.items()}, its


def cnt(s, y):
    return {"yes": int((s > 0.5).sum()), "wrong_yes": int(((s > 0.5) & (y == 0)).sum()), "right_no": int(((s <= 0.5) & (y == 1)).sum())}


MAIN = {"1_last_step": ("n_path", "n_nli"), "2_parent": ("g_nli", "n_nli"), "3_both": ("g_path", "n_nli")}
SUB = {"ctrl_path_vs_ctrl": ("n_path", "n_ctrl"), "ctrl_path_vs_last": ("n_path", "n_last"), "ctrl_path_vs_none": ("n_path", "none"),
       "g_path_vs_n_path": ("g_path", "n_path"), "last_vs_nli": ("n_last", "n_nli")}

if __name__ == "__main__":
    res = M.jload(sys.argv[1])
    out = {}
    # E8 の 1,000問（直し方を見るだけ）
    S, its = scores(res, "e8")
    old, new = A.labels(); allit = old + new
    assert [x["ent"] for x in allit] == [x["ent"] for x in its], "E8 の問題の並びが違う"
    y = np.array([int(x["mine"] >= 1) for x in allit]); ents = np.array([x["ent"] for x in allit])
    out["e8"] = {"aurc": {m: e6.aurc(s, y) for m, s in S.items()}, "c": {m: cnt(s, y) for m, s in S.items()},
                 "cmp": {k: judge(S[a], S[b], y, ents, 700 + n) for n, (k, (a, b)) in enumerate({**MAIN, **SUB}.items())}}
    nn, np_ = S["n_nli"] > 0.5, S["n_path"] > 0.5
    out["e8"]["flip_path"] = {"right_no_N": int((~nn & (y == 1)).sum()), "fixed": int((~nn & np_ & (y == 1)).sum()),
                              "wrong_yes_new": int((~nn & np_ & (y == 0)).sum()), "right_yes_lost": int((nn & ~np_ & (y == 1)).sum())}
    # E10-3 の「一種か」600問（確かめ）
    S, its = scores(res, "k")
    yk = np.array([KCS[str(it["id"])] for it in its]); yw = np.array([KALL[it["id"]]["gold"] for it in its])
    ek = np.array([it["ent"] for it in its])
    out["k"] = {"aurc": {m: e6.aurc(s, yk) for m, s in S.items()}, "aurc_wordnet": {m: e6.aurc(s, yw) for m, s in S.items()},
                "c": {m: cnt(s, yk) for m, s in S.items()},
                "cmp": {k: judge(S[a], S[b], yk, ek, 800 + n) for n, (k, (a, b)) in enumerate({**MAIN, **SUB}.items())},
                "cmp_wordnet": {k: judge(S[a], S[b], yw, ek, 900 + n) for n, (k, (a, b)) in enumerate(MAIN.items())}}
    ok_ctrl = out["k"]["cmp"]["ctrl_path_vs_ctrl"]["better"]
    out["flags"] = {"last_step_works": bool(out["k"]["cmp"]["1_last_step"]["better"] and ok_ctrl),
                    "parent_works": bool(out["k"]["cmp"]["2_parent"]["better"]),
                    "both_works": bool(out["k"]["cmp"]["3_both"]["better"] and ok_ctrl),
                    "path_read_not_just_knowledge": bool(ok_ctrl)}
    json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
    for part in ("e8", "k"):
        print(f"== {part}")
        for m, v in out[part]["aurc"].items(): print(f"  {m:8s} {v:.3f}", out[part]["c"][m])
        for k, v in out[part]["cmp"].items():
            print(f"  {k:22s} a {v['aurc_a']:.3f} b {v['aurc_b']:.3f} 差 {[round(x, 3) for x in v['diff']]} 良い {v['better']} 悪い {v['worse']}")
    print("E8 path の入れ替わり:", out["e8"]["flip_path"]); print("旗:", out["flags"])
