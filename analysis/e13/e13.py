"""E13：正解なしで、食い違う手がかり（①⑩⑪g⑫⑪n）をまとめる。多数決・平均・Dawid–Skene・網（分類の決まり入り BP）・p-bit のまね。
python3 e13.py out.json   （データは data/E10・data/E4-2_answers_gemma-2-2b-it.json・results/E10 の既存ファイル。計画 plans/E13_combine_without_labels.md、結果の前に固定）"""
import json, sys, os, gzip
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); S = os.path.join(HERE, ".."); ROOT = os.path.join(S, "..")
D10 = f"{ROOT}/data/E10"; R10 = f"{ROOT}/results/E10"
sys.path.insert(0, f"{S}/e10/e102"); sys.path.insert(0, f"{S}/e10/e101"); sys.path.insert(0, f"{S}/e10/e103")
import e102_analyze as E2                  # noqa: E402
from e101_analyze import aurc              # noqa: E402
from wordfreq import zipf_frequency as zf  # noqa: E402
lgz = E2.lgz
sig = lambda x: 1 / (1 + np.exp(-x))
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
SRC = ["A", "def", "gen", "self", "nli"]
W_MAIN, W_REF = 2.0, [1.0, 3.0]
B = 1000


# ---------- データ ----------
def load_K():
    KJ = json.load(open(f"{S}/e10/e103/k_items.json")); K = KJ["items"]
    e42 = json.load(open(f"{ROOT}/data/E4-2_answers_gemma-2-2b-it.json"))
    r = f"{D10}/E10-3"
    L = {"A": np.array(e42["logit"], float)[[it["e42"] for it in K]]}
    for m in ("def", "gen", "self"):
        L[m] = np.array(lgz(f"{r}/E103_gemma_K_{m}.json.gz")["logit"], float)
    NL = lgz(f"{r}/E103_nli_K.json.gz")["ent"]
    L["nli"] = np.array([logit(max(e)) if e else np.nan for e in NL])
    cs = {int(k): v for k, v in json.load(open(f"{S}/e10/e103/cs_K_claude.json")).items()}
    ycs = np.array([cs.get(i, -1) for i in range(len(K))])
    # 分類どうし（E4-2 で Gemma に聞いた物）：「C は D の一種か」の値が0より大きい組
    names = {v.lower(): v for v in KJ["cats"].values()}
    sub = []
    for s, l in zip(e42["stmt"], e42["logit"]):
        a, _, b = s[:-1].partition(" is a type of ")
        if a.lower() in names and b.lower() in names and a.lower() != b.lower() and l > 0:
            sub.append((names[a.lower()], names[b.lower()]))
    return dict(its=K, L=L, y=np.array([it["gold"] for it in K]), ycs=ycs, ents=[it["ent"] for it in K],
                cls=[it["cname"] for it in K], sub=sub)


def load_P():
    it_n = f"{D10}/E10-1/e101_items.json"; r101 = f"{D10}/E10-1"; r102 = f"{D10}/E10-2"; r103 = f"{D10}/E10-3"
    items = json.load(open(it_n))["items"]
    _, d101, _ = E2.build101(it_n, r101)
    gN = lgz(f"{r101}/E101_gemma.json.gz")["logit"]
    D = E2.load_set("N", items, gN, r102, {d: d101[d]["F"]["oneN"] for d in "PDG"})["P"]
    L = {"A": D["lg"], "def": D["def"], "gen": D["gen"],
         "self": np.array(lgz(f"{r103}/E103_gemma_N_self.json.gz")["logit"], float),
         "nli": np.where(D["nf"] > 0, logit(np.nan_to_num(D["ent"])), np.nan)}
    cs = {int(k): v for k, v in json.load(open(f"{R10}/E10-1/cs_labels_claude.json")).items()}
    ycs = np.array([cs.get(items[i]["id"], -1) for i in D["idx"]])
    return dict(its=D["its"], L=L, y=D["y"], ycs=ycs, ents=[it["ent"] for it in D["its"]], cls=[it["cls"] for it in D["its"]], sub=[])


# ---------- まとめ方（正解を使わない） ----------
def votes(L):
    return np.column_stack([np.where(np.isnan(L[s]), np.nan, (L[s] > 0).astype(float)) for s in SRC])


def m_mean(L):
    return np.nanmean(np.column_stack([sig(L[s]) for s in SRC]), 1)


def m_major(L):
    V = votes(L); return np.nanmean(V, 1) + 1e-3 * (m_mean(L) - 0.5)


def dawid_skene(L, iters=50):
    V = votes(L); obs = ~np.isnan(V); Vz = np.nan_to_num(V)
    q = np.clip(np.nanmean(V, 1), 0.01, 0.99)
    for _ in range(iters):
        pi = q.mean()
        a = ((q[:, None] * Vz * obs).sum(0) / (q[:, None] * obs).sum(0)).clip(0.01, 0.99)
        b = (((1 - q)[:, None] * Vz * obs).sum(0) / ((1 - q)[:, None] * obs).sum(0)).clip(0.01, 0.99)
        W1, W0 = np.log(a / b), np.log((1 - a) / (1 - b))
        z = np.log(pi / (1 - pi)) + ((Vz * W1 + (1 - Vz) * W0) * obs).sum(1)
        q = sig(z)
    return q, z, {"pi": float(pi), "a": dict(zip(SRC, a.round(3).tolist())), "b": dict(zip(SRC, b.round(3).tolist()))}


def net(field, data, w, how="bp", seed=0):
    """物ごとに分類の網。field＝各主張の場（ロジット）。決まり C⊂D を破る（C=1,D=0）と罰 w。"""
    out = sig(field).copy()
    ents = np.array(data["ents"]); cls = np.array(data["cls"]); sub = data["sub"]
    rng = np.random.default_rng(seed)
    for e in np.unique(ents):
        idx = np.flatnonzero(ents == e); pos = {c: k for k, c in enumerate(cls[idx])}
        E = [(pos[c], pos[d]) for c, d in sub if c in pos and d in pos]
        if not E: continue
        h = field[idx]; n = len(idx)
        if how == "bp":
            # 2値の BP。辺ごとの伝言 m[(i→j)] は j が 0/1 の対数の比。
            fac = np.array([[0.0, 0.0], [-w, 0.0]])   # fac[yC][yD]
            msg = {}
            nb = [[] for _ in range(n)]
            for k, (c, d) in enumerate(E):
                nb[c].append((k, d, 0)); nb[d].append((k, c, 1)); msg[(k, 0)] = 0.0; msg[(k, 1)] = 0.0
            for _ in range(50):
                new = {}
                for k, (c, d) in enumerate(E):
                    for src, dst, side in ((c, d, 0), (d, c, 1)):
                        cav = h[src] + sum(msg[(kk, 1 - s2)] for kk, other, s2 in nb[src] if kk != k)
                        # src の確からしさ（0/1）で dst への伝言
                        ps = np.array([0.0, cav])
                        vals = []
                        for yd in (0, 1):
                            t = [ps[ys] + (fac[ys][yd] if side == 0 else fac[yd][ys]) for ys in (0, 1)]
                            vals.append(np.logaddexp(*t))
                        new[(k, side)] = 0.5 * msg[(k, side)] + 0.5 * (vals[1] - vals[0])
                msg = new
            bel = h.copy()
            for k, (c, d) in enumerate(E):
                bel[d] += msg[(k, 0)]; bel[c] += msg[(k, 1)]
            out[idx] = sig(bel)
        else:   # ギブス（p-bit のまね）
            y = (rng.random(n) < sig(h)).astype(int); acc = np.zeros(n); up = [[] for _ in range(n)]; dn = [[] for _ in range(n)]
            for c, d in E: up[c].append(d); dn[d].append(c)
            for t in range(2200):
                for i in rng.permutation(n):
                    f = h[i] - w * sum(1 - y[d] for d in up[i]) + w * sum(y[c] for c in dn[i])
                    y[i] = int(rng.random() < sig(f))
                if t >= 200: acc += y
            out[idx] = acc / 2000
    return out


# ---------- 採点 ----------
def judge(a, b, yv, ents, seed):
    """a が b より良いか（AURC が b の20%以上小さく、物で引き直した95%幅が0をまたがない）。"""
    ents = np.asarray(ents); ue = np.unique(ents); ix = {e: np.flatnonzero(ents == e) for e in ue}
    rng = np.random.default_rng(seed); d = []
    for _ in range(B):
        ii = np.concatenate([ix[e] for e in rng.choice(ue, len(ue))])
        d.append(aurc(b[ii], yv[ii]) - aurc(a[ii], yv[ii]))
    lo, hi = np.percentile(d, [2.5, 97.5]); A_, B_ = aurc(a, yv), aurc(b, yv)
    return {"a": round(A_, 4), "b": round(B_, 4), "diff": round(float(np.mean(d)), 4), "ci95": [round(lo, 4), round(hi, 4)],
            "差あり（aが良い）": bool(np.mean(d) >= 0.2 * B_ and lo > 0), "差あり（bが良い）": bool(-np.mean(d) >= 0.2 * A_ and hi < 0)}


def run(tag, data):
    L, y, ycs = data["L"], data["y"], data["ycs"]
    P = {f"S_{s}": np.where(np.isnan(L[s]), sig(L["A"]), sig(L[s])) for s in SRC}
    P["M1_mean"] = m_mean(L); P["M2_major"] = m_major(L)
    q, z, info = dawid_skene(L); P["M3_DS"] = q
    if data["sub"]:
        P["M4_net"] = net(z, data, W_MAIN)
        P["M4b_net_mean"] = net(logit(P["M1_mean"]), data, W_MAIN)
        for w in W_REF: P[f"ref_net_w{w:g}"] = net(z, data, w)
        P["ref_pbit_gibbs"] = net(z, data, W_MAIN, how="gibbs")
    m = (ycs >= 0) & (ycs != 2); yc = (ycs[m] == 1).astype(int); em = np.array(data["ents"])[m]
    out = {"n": int(len(y)), "n_cs": int(m.sum()), "ds": info, "n_rules": len(data["sub"]),
           "cs": {k: round(aurc(v[m], yc), 4) for k, v in P.items()},
           "wordnet": {k: round(aurc(v, y), 4) for k, v in P.items()}}
    # R1：WordNet の正解20問で一番良い手がかりを選ぶ（くじ5回）
    rng = np.random.default_rng(13); picks = []
    for _ in range(5):
        ii = rng.choice(len(y), 20, replace=False)
        while len(set(y[ii])) < 2: ii = rng.choice(len(y), 20, replace=False)
        picks.append(min(SRC, key=lambda s: aurc(P[f"S_{s}"][ii], y[ii])))
    out["R1_picks"] = picks; out["R1_cs_mean"] = round(float(np.mean([out["cs"][f"S_{s}"] for s in picks])), 4)
    J = {}
    if "M4_net" in P:
        J["主1 網 対 平均"] = judge(P["M4_net"][m], P["M1_mean"][m], yc, em, 1)
        J["主1 網 対 多数決"] = judge(P["M4_net"][m], P["M2_major"][m], yc, em, 2)
        J["副 網 対 DS（決まりの効き目）"] = judge(P["M4_net"][m], P["M3_DS"][m], yc, em, 3)
        J["副 網(平均から) 対 平均"] = judge(P["M4b_net_mean"][m], P["M1_mean"][m], yc, em, 4)
        J["主3 網 対 R1（20問で選ぶ、くじごと）"] = [judge(P["M4_net"][m], P[f"S_{s}"][m], yc, em, 5 + k) for k, s in enumerate(picks)]
        J["参考 p-bitのまね 対 BP"] = judge(P["ref_pbit_gibbs"][m], P["M4_net"][m], yc, em, 11)
    J["主2 DS 対 多数決"] = judge(P["M3_DS"][m], P["M2_major"][m], yc, em, 12)
    J["参考 平均 対 ①"] = judge(P["M1_mean"][m], P["S_A"][m], yc, em, 14)
    out["judge"] = J
    return out


def main(out_p):
    R = {"K": run("K", load_K()), "P": run("P", load_P())}
    K = R["K"]["judge"]; Pj = R["P"]["judge"]
    main1 = K["主1 網 対 平均"]["差あり（aが良い）"] and K["主1 網 対 多数決"]["差あり（aが良い）"]
    main3_lose = sum(j["差あり（bが良い）"] for j in K["主3 網 対 R1（20問で選ぶ、くじごと）"]) >= 3
    R["flags"] = {"主1 正解なしなら網に出番がある": bool(main1),
                  "主2 DSが効く（K）": K["主2 DS 対 多数決"]["差あり（aが良い）"], "主2 DSが効く（P）": Pj["主2 DS 対 多数決"]["差あり（aが良い）"],
                  "主3 20問で選ぶ方に負ける（5回中3回以上）": bool(main3_lose),
                  "結論": ("網の役目は終わり" if not main1 else ("正解が本当に0の時だけの小さな役目" if main3_lose else "正解なしでは網に出番がある"))}
    json.dump(R, open(out_p, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(R, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
