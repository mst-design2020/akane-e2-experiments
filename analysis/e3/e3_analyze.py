# E3 比べの計算（Claudeの作業場所で）。使い方: python3 e3_analyze.py E3_main.json 出力.json
# 各方法は主張ごとに s = P(支える) を出す。答え＝s>0.5、自信＝|s-0.5|。自信の低い順に控える。
# 方法の中身は結果を見る前にここで決める（PLAN.md「比べる方法の細かい決め方」と同じ）。
import json, sys, math, random
import numpy as np

E, N, C = 0, 1, 2
def arr(v): return np.array([x for x in v if x is not None], dtype=float)

# ---------- 部品 ----------
def fact_prob(paras_nli):
    """段落ごとのNLI [含,中,矛] から、その部分の主張が正しい確率。
    段落 j の重み ψ(正)=含+中/2、ψ(誤)=矛+中/2 を掛け合わせる（関係のない段落は五分五分で効かない）"""
    v = arr(paras_nli)
    if len(v) == 0: return 0.5
    lp = np.sum(np.log(v[:, E] + v[:, N] / 2 + 1e-6)); ln = np.sum(np.log(v[:, C] + v[:, N] / 2 + 1e-6))
    return 1 / (1 + math.exp(max(-50, min(50, ln - lp))))

def fact_hard(paras_nli):
    """⑤用：一番はっきりした段落の判定で○×（含の最大 > 矛の最大 なら○）"""
    v = arr(paras_nli)
    return bool(len(v) and v[:, E].max() > v[:, C].max())

def groups(it):
    """組み合わせ（隠れた点の候補の組）ごとに、部分の主張の判定の並びを返す"""
    g = {}
    for k, v in it["nli"].get("facts_paras", {}).items():
        fi, ai = k.split("|"); g.setdefault(int(ai), {})[int(fi)] = v
    return g

def assign_prior(it):
    """候補の組の事前の重み：候補が何回出たか（決まった答え＋揺らした答え）の数え上げは保存していないので、
    同じ重みにする。候補が出なかった点は「分からない」割合を記録しているので、組全体の重みにはしない。"""
    n = len(it["decomp9b"]["assigns"]); return [1 / n] * n

# ---------- 方法 ----------
def m1_gemma(it, drop=None):
    return (it["verdict_drop"] if drop is not None else it["verdict"])["p_true"]

def m2_count(it, drop=None):
    """② 矛盾を数える（Pandian形）：資料の文ごとに元の主張と比べ、含の数−矛の数"""
    v = sent_nli(it, drop); a = v.argmax(1) if len(v) else np.array([])
    ne, nc = int((a == E).sum()), int((a == C).sum())
    return 0.5 + 0.5 * (ne - nc) / (ne + nc + 1)

def m2_rule(it, drop=None):
    """② のPandianそのままの決め方（答える割合は決まってしまう）：片方向だけなら答え、混ざる・無いなら控える"""
    v = sent_nli(it, drop); a = v.argmax(1) if len(v) else np.array([])
    ne, nc = int((a == E).sum()), int((a == C).sum())
    if ne > 0 and nc == 0: return 1
    if nc > 0 and ne == 0: return 0
    return None

def m4_ds(it, drop=None):
    """④ 確率的に合わせる（デンプスター・シェーファー、ヤーガー則：食い違いは「分からない」へ）。段落ごとの元の主張の判定を合わせる"""
    mS, mR, mT = 0.0, 0.0, 1.0
    for j, v in enumerate(it["nli"]["claim_paras"]["_"]):
        if v is None or j == drop: continue
        e, n, c = v
        S = mS * e + mS * n + mT * e; R = mR * c + mR * n + mT * c
        K = mS * c + mR * e                       # 食い違い
        T = mT * n + K
        mS, mR, mT = S, R, T
    return 0.5 + 0.5 * (mS - mR), mT

def m5_hard(it, drop=None):
    """⑤ 決まった論理（ProgramFC・GraphCheck形）：どれかの候補の組で、部分の主張が全部○なら支える"""
    for ai, fs in groups(it).items():
        if all(fact_hard(dropp(v, drop)) for v in fs.values()): return 1.0
    return 0.0

def m5_softmin(it, drop=None):
    """⑤' 網なしの簡単な式：候補の組ごとに部分の主張の確率の一番低いもの、その一番高い組"""
    best = 0.0
    for ai, fs in groups(it).items():
        best = max(best, min(fact_prob(dropp(v, drop)) for v in fs.values()))
    return best if groups(it) else 0.5

def m6_net(it, drop=None):
    """⑥ 網（因子グラフ）：隠れた点の候補の組 A、部分の主張 F（段落の判定が効く）、元の主張 C＝（Aの組の部分の主張が全部正しい）。
    元の主張には段落ごとの直接の判定も効く。この網は小さいので、確率伝搬と同じ答えを厳密に出す（Aで場合分け）。
    P(C=1 | 部分) = Σ_a P(a) Π_f p_f(a)。ただし P(a) は、その組の「説明の部分の主張（#k is 〜）」の確からしさで重み直す。"""
    g = groups(it)
    if not g: return 0.5
    d = it["decomp9b"]; n0 = d["n_orig_facts"]
    pri = assign_prior(it); w, x = [], []
    for ai, fs in g.items():
        ps = {fi: fact_prob(dropp(v, drop)) for fi, v in fs.items()}
        desc = [p for fi, p in ps.items() if fi >= n0]          # 説明の部分＝その候補が本当にその人や物か
        w.append(pri[ai] if ai < len(pri) else 0)
        x.append(float(np.prod(list(ps.values()))))
        # 候補の重み：説明が合うほど重い（説明が全部外れる候補は、主張が誤りというより候補違いの見込み）
        w[-1] *= (float(np.prod(desc)) if desc else 1.0) + 1e-3
    w = np.array(w) / np.sum(w); X = float(np.dot(w, x))
    # 元の主張への段落の直接の判定（ψ）を掛ける
    cp = it["nli"]["claim_paras"]["_"]
    pc = fact_prob([v for j, v in enumerate(cp) if j != drop])
    lr = pc / (1 - pc + 1e-9)
    return X * lr / (X * lr + (1 - X) + 1e-12)

def dropp(v, drop): return [x for j, x in enumerate(v) if j != drop]

def sent_nli(it, drop=None):
    v = it["nli"]["claim_sents"]["_"]
    if drop is None: return arr(v)
    # 抜いた段落の文を外す（文は段落の順に並んでいる）
    from_para = []
    for j, p in enumerate(it["paras"]):
        from_para += [j] * len(split_sents_local(p))
    return arr([x for x, j in zip(v, from_para) if j != drop]) if len(from_para) == len(v) else arr(v)

def split_sents_local(p):
    import re
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=["A-Z0-9])', p.strip()) if len(s.strip()) > 3]

# ③ 特徴を足す（SURE-RAG形）：小さなロジスティック回帰、5分割で学習と評価を分ける
def feats(it):
    s = sent_nli(it); p = arr(it["nli"]["claim_paras"]["_"])
    fp = [fact_prob(v) for fs in groups(it).values() for v in fs.values()] or [0.5]
    unk = [x for k in it["decomp9b"]["unknown"].values() for x in k.values()] or [0.0]
    return [s[:, E].max(), s[:, C].max(), s[:, E].mean(), s[:, C].mean(), s[:, E].max() * s[:, C].max(),
            p[:, E].max(), p[:, C].max(), min(fp), float(np.mean(fp)), max(unk), len(it["paras"])]

def fit_lr(X, y, l2=1.0, it=500, lr=0.1):
    X = np.c_[X, np.ones(len(X))]; w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0] / len(y)
        w -= lr * g * 10
    return w
def pred_lr(w, X): X = np.c_[X, np.ones(len(X))]; return 1 / (1 + np.exp(-X @ w))

# ---------- 測り方 ----------
def sel_curve(y, s, covs=(1.0, 0.9, 0.7, 0.5)):
    y = np.array(y); s = np.array(s); pred = (s > 0.5).astype(int); conf = np.abs(s - 0.5)
    r = random.Random(0); tieb = np.array([r.random() for _ in s]) * 1e-9   # 同点はくじで
    order = np.argsort(-(conf + tieb)); ok = (pred == y)[order]
    out = {f"acc@{int(c*100)}": float(ok[:max(1, int(round(c * len(ok))))].mean()) for c in covs}
    risk = 1 - np.cumsum(ok) / np.arange(1, len(ok) + 1)
    out["AURC"] = float(risk.mean()); return out

def main(path, outp):
    items = json.load(open(path))["items"]
    items = [it for it in items if "nli" in it]
    y = np.array([1 if it["label"] == "supports" else 0 for it in items])
    S = {"①AIの自信": [m1_gemma(it) for it in items],
         "②矛盾を数える": [m2_count(it) for it in items],
         "④確率的に合わせる(DS)": [m4_ds(it)[0] for it in items],
         "⑤決まった論理": [m5_hard(it) for it in items],
         "⑤'網なしの簡単な式": [m5_softmin(it) for it in items],
         "⑥網": [m6_net(it) for it in items]}
    # 5分割（記事数×正解でそろえる）。③はここで学習
    r = random.Random(42); idx = list(range(len(items))); r.shuffle(idx)
    fold = np.zeros(len(items), int)
    for k, i in enumerate(sorted(idx, key=lambda i: (items[i]["num_hops"], items[i]["label"]))): fold[i] = k % 5
    X = np.array([feats(it) for it in items]); mu, sd = X.mean(0), X.std(0) + 1e-9; X = (X - mu) / sd
    s3 = np.zeros(len(items))
    for f in range(5):
        w = fit_lr(X[fold != f], y[fold != f]); s3[fold == f] = pred_lr(w, X[fold == f])
    S["③特徴を足す(LR)"] = s3.tolist()
    res = {"n": len(items), "all": {}, "folds": {}, "hops": {}, "diff": {}}
    for name, s in S.items():
        res["all"][name] = sel_curve(y, s)
        res["folds"][name] = [sel_curve(y[fold == f], np.array(s)[fold == f]) for f in range(5)]
        res["hops"][name] = {h: sel_curve(y[[i for i, it in enumerate(items) if it["num_hops"] == h]],
                                          np.array(s)[[i for i, it in enumerate(items) if it["num_hops"] == h]]) for h in (2, 3, 4)}
    # 主な比べ：⑥対②、⑥対①、⑥対⑤'（AURCは小さいほど良い：差＝相手−⑥、正なら⑥が良い）
    for other in ["②矛盾を数える", "①AIの自信", "⑤'網なしの簡単な式", "③特徴を足す(LR)", "④確率的に合わせる(DS)"]:
        ds = [res["folds"][other][f]["AURC"] - res["folds"]["⑥網"][f]["AURC"] for f in range(5)]
        win = sum(d > 0 for d in ds); lose = sum(d < 0 for d in ds); m = float(np.mean(ds))
        verdict = "⑥が良い" if (m >= 0.02 and win >= 4) else ("⑥が悪い" if (m <= -0.02 and lose >= 4) else "同着")
        res["diff"][f"⑥ 対 {other}"] = {"AURC差(相手−⑥)": round(m, 4), "5回中⑥が良い": win, "判定": verdict}
    # Pandianそのままの決め方
    rule = [m2_rule(it) for it in items]; ans = [(p, t) for p, t in zip(rule, y) if p is not None]
    res["②Pandianの決め方"] = {"答えた割合": len(ans) / len(items), "答えた中の正しさ": float(np.mean([p == t for p, t in ans])) if ans else None}
    # 追加の確かめ：資料を1本抜く（「支える」主張だけ）。控える側に動いたか＝自信|s-0.5|がどれだけ下がったか、答えが「支える」のままの割合
    dr = [it for it in items if "verdict_drop" in it]
    if dr:
        out = {}
        for name, f in [("①AIの自信", m1_gemma), ("②矛盾を数える", m2_count), ("④確率的に合わせる(DS)", lambda it, drop=None: m4_ds(it, drop)[0]),
                        ("⑤'網なしの簡単な式", m5_softmin), ("⑥網", m6_net)]:
            a = np.array([f(it) for it in dr]); b = np.array([f(it, it["verdict_drop"]["drop_para"]) for it in dr])
            out[name] = {"自信 抜く前": round(float(np.abs(a - .5).mean()), 4), "自信 抜いた後": round(float(np.abs(b - .5).mean()), 4),
                         "支える答え 抜く前": round(float((a > .5).mean()), 3), "支える答え 抜いた後": round(float((b > .5).mean()), 3)}
        res["1本抜く"] = {"n": len(dr), "方法": out}
    json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1)
    print("件数", res["n"])
    for k, v in res["all"].items(): print(f"{k:16s}", {a: round(b, 3) for a, b in v.items()})
    for k, v in res["diff"].items(): print(k, v)
    print("②Pandian", res["②Pandianの決め方"])
    if "1本抜く" in res:
        for k, v in res["1本抜く"]["方法"].items(): print("1本抜く", k, v)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
