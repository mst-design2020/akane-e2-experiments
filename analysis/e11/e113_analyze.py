"""E11-3 の比べ（計画 plans/E11-3_combine_on_other_questions.md、決まりは流す前に固定）。
python3 e113_analyze.py out.json [E9n_checker_scores.json]
材料はどれも前の実験で取った物（E10-1・E10-2・E10-3・E9n・E4-2）と、e113_box.py の箱。
決まり（E10 と同じ）：物を5つに分け、1つ（2割）で学習、残り8割で評価。5回。差が相手の20%以上かつ5回中4回以上 → 差あり。"""
import json, sys, os, gzip
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HG
HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(HERE, "..")
sys.path.insert(0, f"{S}/e10/e101"); sys.path.insert(0, f"{S}/e10/e102"); sys.path.insert(0, f"{S}/e10/e103")
import e101_analyze as E1   # noqa: E402
import e102_analyze as E2   # noqa: E402
import e103_analyze as E3   # noqa: E402
from wordfreq import zipf_frequency as zf   # noqa: E402
aurc, judge = E1.aurc, E1.judge
REPO = os.path.join(HERE, "..", "..")   # 公開リポジトリの根
R101, R102, R103 = f"{REPO}/data/E10/E10-1", f"{REPO}/data/E10/E10-2", f"{REPO}/data/E10/E10-3"
CS101 = f"{REPO}/results/E10/E10-1"   # 常識の判定（Claude・Gemini）
IT_N = f"{R101}/e101_items.json"
lgz = lambda p: json.load(gzip.open(p, "rt", encoding="utf-8"))


def gb(X, y, tune):
    return HG(max_iter=300, learning_rate=0.05, random_state=int(tune.sum()) % 97).fit(X[tune], y[tune]).predict_proba(X)[:, 1]


def ent_feats(its, lg, cls_key):
    """E10 の C++ の特徴：Gemma の値と、同じ物の語の他の分類への答えの並び。"""
    ents = sorted({it["ent"] for it in its}); eidx = {e: j for j, e in enumerate(ents)}
    ent = np.array([eidx[it["ent"]] for it in its]); classes = sorted({it[cls_key] for it in its}); cidx = {c: j for j, c in enumerate(classes)}
    M = np.full((len(ents), len(classes)), np.nan)
    for i, it in enumerate(its): M[ent[i], cidx[it[cls_key]]] = lg[i]
    emax, emean, estd = np.nanmax(M, 1)[ent], np.nanmean(M, 1)[ent], np.nanstd(M, 1)[ent]
    rank = np.array([(M[ent[i]] > lg[i]).sum() for i in range(len(its))]); nyes = (M > 0).sum(1)[ent]
    freq = np.array([zf(it["ent"], "en") for it in its]); cls1h = np.eye(len(classes))[[cidx[it[cls_key]] for it in its]]
    return np.column_stack([lg, emax, emean, estd, rank, nyes, freq]), cls1h, ent


def evaluate(groups, base_keys, y, ents, ycs_sets, compare):
    """groups: 名前→特徴の列。方法＝特徴の組。5回の学習で、常識（と Gemini）の AURC。"""
    fold = E3.folds(list(ents)); runs = []
    for sp in range(5):
        tune = fold == sp; ev = ~tune; P = {}
        for name, keys in base_keys.items():
            P[name] = gb(np.column_stack([groups[k] for k in keys]), y, tune)
        r = {"table": {k: aurc(v[ev], y[ev]) for k, v in P.items()}}
        for lab, ycs in ycs_sets.items():
            m = ev & (ycs >= 0) & (ycs != 2)
            if m.sum() > 20:
                r[lab] = {k: aurc(v[m], (ycs[m] == 1).astype(int)) for k, v in P.items()}
        runs.append(r)
    out = {}
    for key in ["table"] + list(ycs_sets):
        if all(key in r for r in runs):
            out["mean_" + key] = {k: float(np.mean([r[key][k] for r in runs])) for k in runs[0][key]}
            out["main_" + key] = {nm: judge(runs, a, b, key) for nm, (a, b) in compare.items()}
    return out


def pdg():
    items = json.load(open(IT_N))["items"]
    _, d101, _ = E1.build(IT_N, R101)
    gN = lgz(f"{R101}/E101_gemma.json.gz")["logit"]
    SN = E2.load_set("N", items, gN, R102, {d: d101[d]["F"]["oneN"] for d in "PDG"})
    selfv = np.array(lgz(f"{R103}/E103_gemma_N_self.json.gz")["logit"], float)
    csC = {int(k): v for k, v in json.load(open(f"{CS101}/cs_labels_claude.json")).items()}
    csG = {int(k): v for k, v in json.load(open(f"{CS101}/cs_labels_gemini.json")).items()}
    rep = {}
    for dom in "PDG":
        D = d101[dom]; F = D["F"]; its = D["its"]; y = D["y"]; lg = D["lg"]; X = SN[dom]
        assert [it["id"] for it in its] == [items[i]["id"] for i in X["idx"]]
        bp = f"{REPO}/data/E11/box_{dom}.json"
        box = json.load(open(bp)) if os.path.exists(bp) else None
        if box: assert box["ids"] == [it["id"] for it in its]
        FG, cls1h, ent = ent_feats(its, lg, "cls")
        g = {"FG": np.column_stack([FG, D["rv"], cls1h]),   # E10-1 の C++ と同じ
             "chain": np.column_stack([F["oneN"], F["sixN"], F["lenN"], F["anyarrN"], F["has"]]),
             "fix": np.column_stack({"P": lambda: [F["fix1"], F["fix2"], F["dir"]], "D": lambda: [F["fix4"], F["fix4_ok"]], "G": lambda: [np.zeros(len(y))]}[dom]()),
             "res": np.column_stack([X["def"], X["gen"], np.nan_to_num(X["ent"]), (X["nf"] > 0).astype(float), X["nf"]]
                                    + ([selfv] if dom == "P" else [])),
             "box": np.array(box["box"])[:, None] if box else None}
        allk = ["FG", "chain", "fix", "res"] + (["box"] if box else [])
        meth = {"Cpp": ["FG"], "N8pp": ["FG", "chain"], "Xpp": ["FG", "res"], "ALL": allk}
        for k in allk[1:]:
            meth["ALL-" + k] = [j for j in allk if j != k]   # 事後の見方（主張に使わない）
        comp = {"主 全部入り対C++": ("ALL", "Cpp"), "副 全部入り対今までの一番": ("ALL", "Xpp" if dom == "P" else "N8pp")}
        ycs = {"cs": np.array([csC.get(it["id"], -1) for it in its]), "cs_gemini": np.array([csG.get(it["id"], -1) for it in its])}
        rep[dom] = evaluate(g, meth, y, [it["ent"] for it in its], ycs, comp) | {"box_seed_spread": box["seed_spread"] if box else "箱なし（まだ）"}
    return rep


def part_K():
    K = json.load(open(f"{S}/e10/e103/k_items.json"))["items"]
    lg = np.array(json.load(open(f"{REPO}/data/E4-2_answers_gemma-2-2b-it.json"))["logit"], float)[[it["e42"] for it in K]]
    y = np.array([it["gold"] for it in K])
    V = {m: np.array(lgz(f"{R103}/E103_gemma_K_{m}.json.gz")["logit"], float) for m in ("def", "gen", "self")}
    FK = lgz(f"{R103}/E103_facts_K.json.gz")["items"]; NL = lgz(f"{R103}/E103_nli_K.json.gz")["ent"]
    nf = np.array([len(f) for f in FK], float); ent_ = np.array([max(e) if e else 0.0 for e in NL])
    E9 = json.load(open(sys.argv[2] if len(sys.argv) > 2 else f"{REPO}/data/E9n_checker_scores.json"))
    e8D = np.array(E9["scores_e8D"], float)[[it["e42"] for it in K]]; e8one = np.array(E9["scores_e8one"], float)[[it["e42"] for it in K]]
    FG, cls1h, ent = ent_feats(K, lg, "cname")
    g = {"FG": np.column_stack([FG, cls1h]), "chain": np.column_stack([e8D, e8one]),
         "res": np.column_stack([V["def"], V["gen"], V["self"], ent_, (nf > 0).astype(float), nf])}
    allk = ["FG", "chain", "res"]
    meth = {"Cpp": ["FG"], "E8pp": ["FG", "chain"], "Rpp": ["FG", "res"], "ALL": allk,
            "ALL-chain": ["FG", "res"], "ALL-res": ["FG", "chain"]}
    comp = {"主 全部入り対C++": ("ALL", "Cpp"), "副 全部入り対C++＋辞書の鎖": ("ALL", "E8pp")}
    cs = {int(k): v for k, v in json.load(open(f"{S}/e10/e103/cs_K_claude.json")).items()}
    ycs = {"cs": np.array([cs.get(i, -1) for i in range(len(K))])}
    return evaluate(g, meth, y, [it["ent"] for it in K], ycs, comp) | {"box": "入れない（E9n は鎖の形を保存していない）"}


if __name__ == "__main__":
    rep = {"K": part_K(), **pdg()}
    win = lambda d: bool(rep[d]["main_cs"]["主 全部入り対C++"] and rep[d]["main_cs"]["主 全部入り対C++"]["判定"] == "差あり（ALLが良い）")
    rep["flags"] = {"K": win("K"), "P": win("P"), "D": win("D"), "G": win("G"), "繰り返せた（K と P の両方）": win("K") and win("P")}
    json.dump(rep, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
    for d in ("K", "P", "D", "G"):
        R = rep[d]
        print(d, "常識", {k: round(v, 3) for k, v in R["mean_cs"].items()})
        if "mean_cs_gemini" in R: print("   Gemini", {k: round(v, 3) for k, v in R["mean_cs_gemini"].items()})
        for key in ("main_cs", "main_cs_gemini", "main_table"):
            for nm, v in R.get(key, {}).items(): print("   ", key, nm, v and (v["判定"], v["回"]))
    print(rep["flags"])
