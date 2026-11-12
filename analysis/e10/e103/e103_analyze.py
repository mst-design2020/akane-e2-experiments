"""E10-3 の比べ（PLAN_E10-3.md、結果の前に固定）。
使い方：python3 e103_analyze.py E10-1_items.json E10-1_RESULTS E10-2_RESULTS E10-3_RESULTS E42_gemma-2-2b-it.json out.json CS_N.json
  K の問題（k_items.json）と常識（cs_K_claude.json）はこのファイルと同じ場所の物を使う。
決まり：物を5つに分け、1つ（2割）で調整・学習、残り8割で評価。5回。差が相手の20%以上かつ5回中4回以上 → 差あり（E10〜E10-2 と同じ）。
"""
import json, sys, os, gzip
import numpy as np
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{HERE}/../e102"); sys.path.insert(0, f"{HERE}/../e101")
import e102_analyze as E2                     # noqa: E402
from e101_analyze import aurc, judge          # noqa: E402
from wordfreq import zipf_frequency as zf     # noqa: E402

lgz = E2.lgz


def folds(ents, seed=0):
    u = sorted(set(ents)); eidx = {e: j for j, e in enumerate(u)}
    rng = np.random.default_rng(seed); perm = rng.permutation(len(u)); f = np.zeros(len(u), int)
    for k in range(5): f[perm[k::5]] = k
    return f[[eidx[e] for e in ents]]


def cal(v, y, tune):
    return LogisticRegression(max_iter=1000).fit(v[tune, None], y[tune]).predict_proba(v[:, None])[:, 1]


def score(P, y, ev, ycs, freq):
    m = ev & (ycs >= 0) & (ycs != 2); r = {"table": {k: aurc(v[ev], y[ev]) for k, v in P.items()}}
    if m.sum() > 20:
        yc = (ycs[m] == 1).astype(int); r["cs"] = {k: aurc(v[m], yc) for k, v in P.items()}
        for nm, sel in (("rare", freq < E2.RARE_Z), ("common", freq >= E2.RARE_Z)):
            mm = m & sel
            if mm.sum() > 20 and len(set((ycs[mm] == 1).astype(int).tolist())) > 1:
                r["cs_" + nm] = {k: aurc(v[mm], (ycs[mm] == 1).astype(int)) for k, v in P.items()}
    return r


def summarize(runs, comp):
    R = {}
    for key in ("table", "cs", "cs_rare", "cs_common"):
        if all(key in r for r in runs):
            R["mean_" + key] = {k: float(np.mean([r[key][k] for r in runs])) for k in runs[0][key]}
            R["main_" + key] = {nm: judge(runs, a, b, key) for nm, (a, b) in comp.items()}
    return R


def part_N(it_n, r101, r102, r103, cs_n):
    items = json.load(open(it_n))["items"]
    _, d101, _ = E2.build101(it_n, r101)
    gN = lgz(f"{r101}/E101_gemma.json.gz")["logit"]
    D = E2.load_set("N", items, gN, r102, {d: d101[d]["F"]["oneN"] for d in "PDG"})["P"]
    selfv = np.array(lgz(f"{r103}/E103_gemma_N_self.json.gz")["logit"], float)
    genF = np.array(lgz(f"{r103}/E103_gemma_N_genF.json.gz")["logit"], float)
    sense = lgz(f"{r103}/E103_sense_N.json.gz")["logit"]; nlip = lgz(f"{r103}/E103_nli_N.json.gz")["ent"]
    assert len(selfv) == len(D["y"]) == len(sense)
    entF = np.array([max([e for e, s in zip(es, ss) if s > 0], default=np.nan) for es, ss in zip(nlip, sense)])
    kept = np.array([sum(s > 0 for s in ss) for ss in sense]); nf = D["nf"]
    cs = {int(k): v for k, v in json.load(open(cs_n)).items()}
    ycs = np.array([cs.get(items[i]["id"], -1) for i in D["idx"]]); y = D["y"]
    fold = folds([it["ent"] for it in D["its"]]); runs = []
    for sp in range(5):
        tune = fold == sp; ev = ~tune
        P, _, freq = E2.methods(D, tune)
        P["self"] = cal(selfv, y, tune); P["genF"] = cal(genF, y, tune)
        P["nliF"] = np.where(~np.isnan(entF), np.nan_to_num(entF), P["A"])
        # X++ に ⑫ を足した版（E10-2 の X++ と同じ木の学習）
        FX = np.column_stack([D["lg"], D["def"], D["gen"], np.nan_to_num(D["ent"]), (nf > 0).astype(float), nf, D["one"], selfv])
        from sklearn.ensemble import HistGradientBoostingClassifier as HG
        P["Xpp_self"] = HG(max_iter=300, learning_rate=0.05, random_state=int(tune.sum()) % 97).fit(FX[tune], y[tune]).predict_proba(FX)[:, 1]
        runs.append(score(P, y, ev, ycs, freq))
    comp = {"1 ⑪g対⑫": ("gen", "self"), "1 ⑫対①": ("self", "A"), "1 ⑪g対①": ("gen", "A"),
            "1参考 X++⑫対X++": ("Xpp_self", "Xpp"), "3 ⑪f対⑪g": ("genF", "gen"), "3 ⑪nf対⑪n": ("nliF", "nli")}
    R = summarize(runs, comp)
    R.update({"n_items": int(len(y)), "facts_rate": float((nf > 0).mean()), "kept_rate_of_facts": float(kept.sum() / max(1, nf.sum())),
              "kept_any_rate": float((kept > 0).mean())})
    return R


def part_K(e42, r103):
    K = json.load(open(f"{HERE}/k_items.json"))["items"]
    lg = np.array(json.load(open(e42))["logit"], float)   # E4-2 で取った ① の値（sims/e4/E42_gemma-2-2b-it.json）
    A = lg[[it["e42"] for it in K]]; y = np.array([it["gold"] for it in K])
    V = {m: np.array(lgz(f"{r103}/E103_gemma_K_{m}.json.gz")["logit"], float) for m in ("def", "defS", "gen", "genS", "self")}
    FK = lgz(f"{r103}/E103_facts_K.json.gz")["items"]; NL = lgz(f"{r103}/E103_nli_K.json.gz")["ent"]
    has = np.array([len(f) > 0 for f in FK]); ent = np.array([max(e) if e else np.nan for e in NL])
    chk = np.array(lgz(f"{r103}/E103_check_plain_K.json.gz")["logit"]);
    cs = {int(k): v for k, v in json.load(open(f"{HERE}/cs_K_claude.json")).items()}
    ycs = np.array([cs.get(i, -1) for i in range(len(K))]); freq = np.array([zf(it["ent"], "en") for it in K])
    fold = folds([it["ent"] for it in K]); runs = []
    for sp in range(5):
        tune = fold == sp; ev = ~tune
        P = {"A": cal(A, y, tune)}
        for m, v in V.items(): P[m] = cal(v, y, tune)
        P["nli"] = np.where(has, np.nan_to_num(ent), P["A"])
        runs.append(score(P, y, ev, ycs, freq))
    comp = {"2 ⑩対①": ("def", "A"), "2 ⑪g対①": ("gen", "A"), "2 ⑫対①": ("self", "A"), "2 ⑪n対①": ("nli", "A"),
            "2 ⑩対⑩s": ("def", "defS"), "2 ⑪g対⑪s": ("gen", "genS"), "2参考 ⑪g対⑫": ("gen", "self")}
    R = summarize(runs, comp)
    R.update({"n_items": len(K), "facts_rate": float(has.mean()), "facts_rate_true": float(has[y == 1].mean()),
              "check_plain_max_diff": float(np.max(np.abs(chk - A[:len(chk)])))})
    return R


def main(it_n, r101, r102, r103, e42, out, cs_n):
    rep = {"N_P": part_N(it_n, r101, r102, r103, cs_n), "K": part_K(e42, r103)}
    P = rep["N_P"]; Kk = rep["K"]
    ok = lambda R, nm, a: bool(R["main_cs"][nm] and R["main_cs"][nm]["判定"] == f"差あり（{a}が良い）")
    rep["overall"] = {
        "外の資料が要る（N の P で ⑪g が ⑫ に差あり）": ok(P, "1 ⑪g対⑫", "gen"),
        "自分の文の方が良い（⑫ が ⑪g に差あり）": ok(P, "1 ⑪g対⑫", "self"),
        "一種かでは辞書が効く（K で ⑩ が ① と ⑩s の両方に差あり）": ok(Kk, "2 ⑩対①", "def") and ok(Kk, "2 ⑩対⑩s", "def"),
        "意味の合う文だけにすると良くなる（⑪f が ⑪g に差あり）": ok(P, "3 ⑪f対⑪g", "genF"),
        "NLI も意味の合う文だけなら良くなる（⑪nf が ⑪n に差あり）": ok(P, "3 ⑪nf対⑪n", "nliF")}
    json.dump(rep, open(out, "w"), ensure_ascii=False, indent=1)
    for nm, R in (("N の P", P), ("K", Kk)):
        print(nm, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R.items() if not isinstance(v, dict)})
        print("  ", {k: round(v, 3) for k, v in R.get("mean_cs", {}).items()})
        for c, v in R.get("main_cs", {}).items(): print("    ", c, v and (v["判定"], v["回"]))
    print(rep["overall"])


if __name__ == "__main__":
    main(*sys.argv[1:])
