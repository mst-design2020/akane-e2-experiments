"""E10-2 の比べ（PLAN_E10-2.md、結果の前に固定）。
使い方：python3 e102_analyze.py E10-1_items.json E10-1_RESULTS E10_items.json E10_gemma.json.gz E10-2_RESULTS out.json CS_N.json CS_O.json
  CS_N＝E10-1 の問題の常識判定（{"id": 0/1/2}）、CS_O＝E10 の問題の常識判定。2 は除く。
決まり（E10・E10-1 と同じ）：物を5つに分け、1つ（2割）で調整・学習、残り8割で評価。5回。差が相手の20%以上かつ5回中4回以上 → 差あり。
Gemma の値（①・⑩・⑩s・⑪g・⑪s）は、どれも同じやり方（調整用の2割でロジスティック回帰）で確からしさに直す。
⑪n は文が見つかった時は NLI の含意の値、見つからない時は ① の値（③ と同じ形）。
"""
import json, sys, os, gzip
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from wordfreq import zipf_frequency as zf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../e101")
from e101_analyze import aurc, judge, build as build101

RARE_Z = 3.5   # 先に固定（PLAN_E10-2.md）


def lgz(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_set(tag, items, glog, rdir, one=None):
    G = {m: lgz(f"{rdir}/E102_gemma_{tag}_{m}.json.gz")["logit"] for m in (["def", "gen", "defS", "genS"] if tag == "N" else ["def", "gen"])}
    NL = lgz(f"{rdir}/E102_nli_{tag}.json.gz"); FA = lgz(f"{rdir}/E102_facts_{tag}.json.gz")
    out = {}
    for dom in "PDG":
        idx = [i for i, it in enumerate(items) if it["dom"] == dom]
        d = dict(idx=idx, its=[items[i] for i in idx], y=np.array([items[i]["gold"] for i in idx]),
                 lg=np.array([glog[i] for i in idx], float),
                 nf=np.array([len(FA["items"][i]) for i in idx]),
                 ent=np.array([NL["ent"][i] if NL["ent"][i] is not None else np.nan for i in idx], float))
        for m, v in G.items():
            d[m] = np.array([v[i] for i in idx], float)
        d["one"] = one[dom] if one is not None else None
        out[dom] = d
    return out


def methods(D, tune):
    y, lg = D["y"], D["lg"]
    cal = lambda v: LogisticRegression(max_iter=1000).fit(v[tune, None], y[tune]).predict_proba(v[:, None])[:, 1]
    P = {"A": cal(lg)}
    for m in ("def", "defS", "gen", "genS"):
        if m in D: P[m] = cal(D[m])
    has = D["nf"] > 0
    P["nli"] = np.where(has, np.nan_to_num(D["ent"]), P["A"])
    its = D["its"]; ents = sorted({it["ent"] for it in its}); eidx = {e: j for j, e in enumerate(ents)}
    ent = np.array([eidx[it["ent"]] for it in its]); classes = sorted({it["cls"] for it in its}); cidx = {c: j for j, c in enumerate(classes)}
    M = np.full((len(ents), len(classes)), np.nan)
    for i, it in enumerate(its): M[ent[i], cidx[it["cls"]]] = lg[i]
    emax, emean, estd = np.nanmax(M, 1)[ent], np.nanmean(M, 1)[ent], np.nanstd(M, 1)[ent]
    rank = np.array([(M[ent[i]] > lg[i]).sum() for i in range(len(its))]); nyes = (M > 0).sum(1)[ent]
    freq = np.array([zf(it["ent"], "en") for it in its]); cls1h = np.eye(len(classes))[[cidx[it["cls"]] for it in its]]
    FG = np.column_stack([lg, emax, emean, estd, rank, nyes, freq, cls1h])   # E10-1 の C++ と同じ（逆向きの文の値はE10の問題にないので両方で外す）
    extra = [D["def"], D["gen"], np.nan_to_num(D["ent"]), has.astype(float), D["nf"]]
    if D.get("one") is not None: extra.append(D["one"])
    FX = np.column_stack([FG] + extra)
    sp = int(tune.sum()) % 97
    gb = lambda X: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=sp).fit(X[tune], y[tune]).predict_proba(X)[:, 1]
    P["Cpp"] = gb(FG); P["Xpp"] = gb(FX)
    return P, ent, freq


def run_set(tag, S, cs, items):
    rep = {}
    for dom in "PDG":
        D = S[dom]; y = D["y"]
        ents = sorted({it["ent"] for it in D["its"]}); eidx = {e: j for j, e in enumerate(ents)}
        ent = np.array([eidx[it["ent"]] for it in D["its"]])
        rng = np.random.default_rng(0); perm = rng.permutation(len(ents)); fold = np.zeros(len(ents), int)
        for f in range(5): fold[perm[f::5]] = f
        ycs = np.array([cs.get(items[i]["id"], -1) for i in D["idx"]])
        runs = []
        for sp in range(5):
            tune = fold[ent] == sp; ev = ~tune
            P, _, freq = methods(D, tune)
            r = {"table": {k: aurc(v[ev], y[ev]) for k, v in P.items()}}
            m = ev & (ycs >= 0) & (ycs != 2)
            if m.sum() > 20:
                yc = (ycs[m] == 1).astype(int)
                r["cs"] = {k: aurc(v[m], yc) for k, v in P.items()}
                for nm_, sel in (("rare", freq < RARE_Z), ("common", freq >= RARE_Z)):
                    mm = m & sel
                    if mm.sum() > 20 and len(set(((ycs[mm] == 1).astype(int)).tolist())) > 1:
                        r["cs_" + nm_] = {k: aurc(v[mm], (ycs[mm] == 1).astype(int)) for k, v in P.items()}
            lg = D["lg"]
            ok = ev & (lg > 0) & (y == 1)
            r["break"] = {k: float(np.mean(v[ok] <= 0.5)) for k, v in P.items()} if ok.sum() else {}
            cw = ev & (((lg > 2) & (y == 0)) | ((lg < -2) & (y == 1)))
            r["fix"] = {k: float(np.mean((v[cw] > 0.5) == y[cw])) for k, v in P.items()} if cw.sum() else {}
            runs.append(r)
        R = {"n_items": int(len(y)), "facts_rate": float((D["nf"] > 0).mean()),
             "facts_rate_true": float((D["nf"][y == 1] > 0).mean()) if y.sum() else None}
        for key in ("table", "cs", "cs_rare", "cs_common"):
            if all(key in r for r in runs):
                R["mean_" + key] = {k: float(np.mean([r[key][k] for r in runs])) for k in runs[0][key]}
                comp = {"1 ⑩対①": ("def", "A"), "1 ⑪g対①": ("gen", "A"), "1 ⑪n対①": ("nli", "A"), "3 X++対C++": ("Xpp", "Cpp")}
                if tag == "N":
                    comp.update({"2 ⑩対⑩s": ("def", "defS"), "2 ⑪g対⑪s": ("gen", "genS")})
                R["main_" + key] = {nm_: judge(runs, a, b, key) for nm_, (a, b) in comp.items()}
        for key in ("break", "fix"):
            R[key] = {k: float(np.mean([r[key][k] for r in runs if r[key]])) for k in runs[0][key]} if runs[0][key] else {}
        rep[dom] = R
    return rep


def main(it_n, r101, it_o, g10, r102, out, cs_n, cs_o):
    items_n = json.load(open(it_n))["items"]; items_o = json.load(open(it_o))["items"]
    _, data101, _ = build101(it_n, r101)
    one = {d: data101[d]["F"]["oneN"] for d in "PDG"}
    gN = lgz(f"{r101}/E101_gemma.json.gz")["logit"]; gO = lgz(g10)["logit"]
    SN = load_set("N", items_n, gN, r102, one); SO = load_set("O", items_o, gO, r102)
    chk = lgz(f"{r102}/E102_check_plain.json.gz")["logit"]
    csN = {int(k): v for k, v in json.load(open(cs_n)).items()}; csO = {int(k): v for k, v in json.load(open(cs_o)).items()}
    report = {"check_plain_max_diff": float(np.max(np.abs(np.array(chk) - np.array(gN[:len(chk)])))),
              "N": run_set("N", SN, csN, items_n), "O": run_set("O", SO, csO, items_o)}
    P = report["N"]["P"]; key = "main_cs" if "main_cs" in P else "main_table"
    ok = lambda nm_: P[key][nm_] and P[key][nm_]["判定"].startswith("差あり（" + {"⑩": "def", "⑪g": "gen"}[nm_.split()[1].split("対")[0]])
    report["overall"] = {"採点": key,
                         "辞書の説明文が効いた（E10-1の部分と全体で 1・2 とも差あり）": bool(ok("1 ⑩対①") and ok("2 ⑩対⑩s")),
                         "一般的な文が効いた（同上）": bool(ok("1 ⑪g対①") and ok("2 ⑪g対⑪s"))}
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=1)
    for tag in "NO":
        for dom in "PDG":
            R = report[tag][dom]
            print(tag, dom, "文あり", round(R["facts_rate"], 3), {k: round(v, 3) for k, v in R.get("mean_cs", R["mean_table"]).items()})
            for nm_, v in R.get("main_cs", R["main_table"]).items():
                print("   ", nm_, v and (v["判定"], v["回"]))
    print(report["overall"], "確かめ（説明文なしの値の差の最大）", report["check_plain_max_diff"])


if __name__ == "__main__":
    main(*sys.argv[1:])
