"""E11-1 の比べ（計画 plans/E11-1_usual_sense.md、決まりは流す前に固定）。
python3 e111_analyze.py E8n_results.json E11-1_results.json out.json
E8_DIR は analysis/e8（e8n_analyze と同じ）。正解：700問は E6 と同じ（Claude の常識判定が主、Gemini も）、新しい300問は e8_new_items_labeled.json。"""
import json, sys, os, numpy as np
os.environ.setdefault("E8_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e8"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nolist"))
import e8n_analyze as A
from sklearn.ensemble import HistGradientBoostingClassifier

e6 = A.e6
sig = A.sig
B = 1000          # 取り直しの回数
D1 = 0.05         # ①の差の基準（5ポイント）
REL = 0.20        # ②の差の基準（相手の20%）
ROOTS = ["R0_all", "R1_first", "R2_gemma", "R3_weight"]


class GB:   # E8n の木の学習（公開 analysis/audit_gb/gblr.py）と同じ設定
    def fit(self, X, y):
        self.m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, min_samples_leaf=20, random_state=0).fit(X, y); return self
    def predict_proba(self, X): return self.m.predict_proba(X)


def ci(v):
    return [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


# ---------- ① WordNet：文なしで普通の意味を選ぶ ----------
def part1(wn):
    n = np.array([len(x["defs"]) for x in wn]); g = np.array([x["gold"] for x in wn])
    hit = {"S0_random": 1.0 / n,
           "S1_similar": np.array([float(np.argmax(x["s1"]) == x["gold"]) for x in wn]),
           "S2_gemma": np.array([float(np.argmax(x["s2"]) == x["gold"]) for x in wn])}
    share = np.array([max(x["counts"]) / sum(x["counts"]) for x in wn])
    subs = {"all": np.ones(len(wn), bool), "senses_2-3": n <= 3, "senses_4+": n >= 4,
            "skew_high": share >= np.median(share), "skew_low": share < np.median(share)}
    out = {"n": len(wn), "pool": wn[0].get("pool"), "acc": {s: {k: float(v[m].mean()) for k, v in hit.items()} | {"n": int(m.sum())} for s, m in subs.items()}}
    rng = np.random.default_rng(11); d = {"S2_vs_S1": [], "S2_vs_S0": []}
    for _ in range(B):
        ii = rng.integers(0, len(wn), len(wn))
        d["S2_vs_S1"].append(hit["S2_gemma"][ii].mean() - hit["S1_similar"][ii].mean())
        d["S2_vs_S0"].append(hit["S2_gemma"][ii].mean() - hit["S0_random"][ii].mean())
    out["boot"] = {k: ci(v) for k, v in d.items()}
    out["sig"] = {k: bool(v[0] >= D1 and v[1] > 0) for k, v in out["boot"].items()}
    out["flag_part1"] = all(out["sig"].values())
    return out


# ---------- ② E8n の1段目 ----------
def per_root(x):
    """根（物の語の意味 k）ごとの一番強い道の掛け算 f_k と段の数。E8n の maxprod を根1つずつで呼ぶ。"""
    keep = x.roots; f = {}
    for r in keep:
        x.roots = [r]; s, ok, dep, path = x.maxprod(); f[r[1]] = (s, dep)
    x.roots = keep
    return f


def scores(items, its, ent_p, leak_p):
    S = {k: np.zeros(len(items)) for k in ROOTS + ["R4_leak_ref"]}; DEP = {k: np.zeros(len(items)) for k in S}
    for i, (it, x) in enumerate(zip(items, its)):
        f = per_root(x)
        if not f: continue
        nk = max(f) + 1
        get = lambda k: f.get(k, (0.0, 0))
        p = np.array(ent_p.get(it["ent"], [1.0] + [0.0] * (nk - 1)))[:nk]
        q = np.array(leak_p.get(str(it["id"]), p))[:nk]
        k0 = max(f, key=lambda k: f[k][0])
        pick = {"R0_all": k0, "R1_first": 0, "R2_gemma": int(np.argmax(p)), "R4_leak_ref": int(np.argmax(q))}
        for m, k in pick.items():
            S[m][i], DEP[m][i] = get(k)
        S["R3_weight"][i] = float(sum(p[k] * get(k)[0] for k in range(nk)))
        DEP["R3_weight"][i] = get(int(np.argmax([p[k] * get(k)[0] for k in range(nk)])))[1]
    return S, DEP


def boot_ent(a, b, y, ents, seed):
    """物の語で取り直した AURC(b) − AURC(a)（正なら a が良い）。"""
    ue = np.unique(ents); idx = {e: np.where(ents == e)[0] for e in ue}; rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        ii = np.concatenate([idx[e] for e in rng.choice(ue, len(ue))])
        v.append(e6.aurc(b[ii], y[ii]) - e6.aurc(a[ii], y[ii]))
    return ci(v)


def judge(a, b, y, ents, seed):
    """a が b より良いか：差が b の20%以上、かつ95%の幅が0をまたがない。"""
    bd = boot_ent(a, b, y, ents, seed); base = e6.aurc(b, y)
    return {"aurc_a": e6.aurc(a, y), "aurc_b": base, "diff": bd, "better": bool(bd[0] >= REL * base and bd[1] > 0),
            "worse": bool(-bd[0] >= REL * e6.aurc(a, y) and bd[2] < 0)}


def part2(r8, r11):
    L, ST, AR, CH = A.load(r8)
    old, new = A.labels(); allit = old + new; no = len(old)
    its = [A.Item(it, "D", *CH[(it["id"], "D")], L, ST, AR) for it in allit]
    S, DEP = scores(allit, its, r11["ent_p"], r11.get("leak_p", {}))
    io = np.arange(no); inw = np.arange(no, len(allit))
    Y = {"claude": np.array([int(x["mine"] >= 1) for x in old]), "claude_2wrong": np.array([int(x["mine"] == 1) for x in old]),
         "gemini": np.array([int(x["gem"] >= 1) for x in old])}
    LG = np.array([it["logit"] for it in allit])
    ents = np.array([x["ent"] for x in old])
    out = {"check_R0_equals_E8n_6D": A.describe(S["R0_all"][io], old, Y), "old": {}, "cmp": {}}
    out["old"]["1_conf"] = A.describe(sig(LG[io]), old, Y)
    for m in S:
        out["old"][m] = A.describe(S[m][io], old, Y) | {"n_yes": int((S[m][io] > 0.5).sum()),
                                                        "wrong_yes": int(((S[m][io] > 0.5) & (Y["claude"] == 0)).sum())}
    pairs = {"1_R3_vs_R0": ("R3_weight", "R0_all"), "2_R3_vs_R1": ("R3_weight", "R1_first"),
             "sub_R2_vs_R1": ("R2_gemma", "R1_first"), "sub_R2_vs_R0": ("R2_gemma", "R0_all"), "ref_R1_vs_R0": ("R1_first", "R0_all")}
    for lab in ("claude", "gemini"):
        for k, (a, b) in pairs.items():
            out["cmp"][f"{k}_{lab}"] = judge(S[a][io], S[b][io], Y[lab], ents, 21)
    # 新しい300問：700問で R1〜R3 から一番良い物を選び、1回だけ
    best = min(ROOTS[1:], key=lambda m: e6.aurc(S[m][io], Y["claude"]))
    yn = np.array([int(x["mine"] >= 1) for x in new]); en = np.array([x["ent"] for x in new])
    out["new"] = {"chosen": best, "aurc": {m: e6.aurc(S[m][inw], yn) for m in ROOTS + ["R4_leak_ref"]} | {"1_conf": e6.aurc(sig(LG[inw]), yn)},
                  "vs_R0": judge(S[best][inw], S["R0_all"][inw], yn, en, 31),
                  "vs_conf": judge(S[best][inw], sig(LG[inw]), yn, en, 32)}
    out["flag_part2"] = bool(out["cmp"]["1_R3_vs_R0_claude"]["better"] and out["new"]["vs_R0"]["better"])
    out["flag_first_sense_enough"] = not out["cmp"]["2_R3_vs_R1_claude"]["better"]
    # 副：木の学習（E8n の特徴の形：E6の特徴＋点数＋段の数）。5回×5分割、物で分ける
    F = [A.x_feats(it, L, AR) for it in allit]
    X5 = np.column_stack([LG, np.array([f[0] for f in F]), np.array([f[1] for f in F])])
    Xa = {m: np.column_stack([X5, S[m], DEP[m]]) for m in ("R0_all", best)}
    per = {m: [] for m in Xa}
    for seed in range(5):
        rng = np.random.default_rng(600 + seed); ue = np.unique(ents); fo = {e: k % 5 for k, e in enumerate(rng.permutation(ue))}
        fold = np.array([fo[e] for e in ents])
        for m, X in Xa.items():
            p = np.zeros(no)
            for k in range(5):
                tr, te = io[fold != k], io[fold == k]
                p[te] = GB().fit(X[tr], Y["claude"][fold != k]).predict_proba(X[te])[:, 1]
            per[m].append(e6.aurc(p, Y["claude"]))
    wins = sum(b <= a * (1 - REL) for a, b in zip(per["R0_all"], per[best]))
    out["learn_gb"] = {"per_seed": per, "wins_of_5": int(wins), "better": bool(wins >= 4)}
    return out


if __name__ == "__main__":
    r8 = json.load(open(sys.argv[1])); r11 = json.load(open(sys.argv[2]))
    res = {"part1": part1(r11["wn"]), "part2": part2(r8, r11)}
    json.dump(res, open(sys.argv[3], "w"), ensure_ascii=False, indent=1)
    p1, p2 = res["part1"], res["part2"]
    print("① 当たり:", {s: {k: round(v, 3) for k, v in d.items()} for s, d in p1["acc"].items()})
    print("① 差:", p1["boot"], " 効いた:", p1["flag_part1"])
    print("② 700問 AURC常識/Gemini:", {m: (round(d["aurc_claude"], 3), round(d["aurc_gemini"], 3), d["wrong_yes"]) for m, d in p2["old"].items() if "aurc_claude" in d and m != "1_conf"})
    for k, v in p2["cmp"].items():
        print(f"  {k:28s} a {v['aurc_a']:.3f} b {v['aurc_b']:.3f} 差 {[round(x, 3) for x in v['diff']]} 良い {v['better']} 悪い {v['worse']}")
    print("② 新しい300問:", p2["new"]["chosen"], {k: round(v, 3) for k, v in p2["new"]["aurc"].items()},
          "R0より良い", p2["new"]["vs_R0"]["better"], "自信より良い", p2["new"]["vs_conf"]["better"])
    print("木の学習:", p2["learn_gb"])
    print("旗: ①", p1["flag_part1"], " ②", p2["flag_part2"], " 最初の意味で足りる", p2["flag_first_sense_enough"])
