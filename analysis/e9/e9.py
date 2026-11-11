"""E9：確かめる数に上限がある時、網は少ない確かめで多くを直せるか。計画: PLAN_E9.md
データは E4-2 と同じ（Gemma 2B、8万問、候補選び・網も e42.build と同じ）。
使い方: python3 e9.py [quick]  → E9_results.json
"""
import json, sys, time, os, itertools
import numpy as np
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'e4'))
sys.path.insert(0, os.path.join(_H, '..', 'e4', 'relabel'))
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
STRONG = os.environ.get('E9_STRONG') == '1'   # E9-2：確かめを上手に使う強い学習 C+・C++ を足す
import e42
from e41 import sig, selective, BAD
from my_labels import LAB

E4 = os.path.join(_H, '..', 'e4') + '/'   # e42_data.json は analysis/e4/e42_data.py で作る
RES = os.path.join(_H, '..', '..', 'results', 'E9'); os.makedirs(RES, exist_ok=True)
OUT = os.path.join(RES, 'E9_results.json')
QUICK = len(sys.argv) > 1 and sys.argv[1] == 'quick'
PROBE = len(sys.argv) > 1 and sys.argv[1] in ('probe', 'real')
REAL = len(sys.argv) > 1 and sys.argv[1] == 'real'   # E9-1：本物の辞書の確かめ（e91_checker.py の出力 E91_checker.json）

BUDGETS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
SELECTS = ['rand', 'unc', 'net']
# 確かめ役の間違い方。e6 = E6の辞書の確かめ（NLI、S1）の当たり方に合わせた、種類ごとの言い間違い
NOISES = {'e6': dict(T=0.26, G=0.11, H=0.01, E=0.01), 'zero': dict(T=0, G=0, H=0, E=0),
          'u20': dict(T=0.2, G=0.2, H=0.2, E=0.2)}
SPLITS = [0, 1, 2, 3, 4]
KS = [0.5, 1.0, 2.0, 4.0, 8.0]          # B・B2 のルールの強さ（調整用の物で選ぶ）
WMUL = [0.5, 1.0, 2.0, 4.0]             # D・F の網の強さの倍率（調整用の物で選ぶ）
LCHK = np.log(0.9 / 0.1)                # 確かめた答えの強さ（確かめ役を9割信じる。どの方法も同じ）

t0 = time.time()
D = json.load(open(E4 + 'e42_data.json')); allc = D['claims']; Y = np.array([c['gold'] for c in allc])
L = np.array(json.load(open(os.path.join(_H, '..', '..', 'data', 'E4-2_answers_gemma-2-2b-it.json')))['logit'])
var, factors, ent_x, xtype = e42.build(D, L)
n = len(var); logit = L[var]; y = Y[var].astype(bool); kind = np.array([allc[i]['kind'] for i in var])
T4 = ["trans", "disj", "anti", "cd"]
big = e42.WNet(n, factors); ftypes = np.array([t for t, _ in factors])
ents = list(ent_x)
ent_of = np.full(n, -1);
for k, w in enumerate(ents):
    ent_of[ent_x[w]] = k
xs = np.where(kind == 'x')[0]
typ = np.array([xtype.get(i, '-') for i in range(n)])

CHK = {}
if REAL:
    _c = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'E91_checker.json')))
    key_of = lambda c: f"{c['ent']}|{D['cats'][c['cat']]}"
    for name in _c['checkers']:
        a_ = np.full(n, -1, dtype=int)
        for i in range(n):
            c = allc[var[i]]
            if c['kind'] == 'x':
                a_[i] = _c['checkers'][name].get(key_of(c), -1)
        CHK[name] = a_
    BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20, 1.0]
    print({k: dict(checkable=int((v[kind == 'x'] >= 0).sum()), acc=float(np.mean(v[v >= 0] == y[v >= 0]))) for k, v in CHK.items()}, flush=True)

# ---- E4-2 と同じ特徴 ----
a0 = (logit > 0).astype(np.int8)
SC = big.signed_counts(a0, ftypes, T4)
feat = np.zeros((n, 6)); feat[:, 0] = logit; v1, v0 = big.counts(a0); feat[:, 1] = v1; feat[:, 2] = v0
for w in ents:
    ids = np.array(ent_x[w])
    for i in ids:
        o = logit[ids[ids != i]]; feat[i, 3] = o.max(); feat[i, 4] = o.mean()
cs = np.zeros(n); cc = np.zeros(n)
for t, v in factors:
    if t == "trans": cs[v[0]] += logit[v[1]]; cc[v[0]] += 1
feat[:, 5] = cs / np.maximum(cc, 1); feat3 = np.hstack([feat, SC[:, :2]])

# ---- 常識の判定（E4の見直し 700個） ----
S = json.load(open(E4 + 'relabel/sample.json'))
s_node = np.array([s['node'] for s in S]); s_mine = np.array([LAB.get(s['id'], 0) for s in S])
s_ent = np.array([s['ent'] for s in S])

# ---- ルールの差し込み口（1歩・繰り返し用） ----
fi, si, vi, bi = big.fi, big.si, big.vi, big.bi
sgn = np.where(bi == 1, -1.0, 1.0)      # 他が全部悪い側なら、この口を悪い側から押し出す向き
print('網', n, '主張', len(factors), 'ルール', 'x', len(xs), round(time.time() - t0), 's', flush=True)


def push(P):
    """各主張について、他の口が全部悪い側にある確率の和（向き付き）。P: 主張ごとの「本当」の確率"""
    qb = np.where(bi == 1, P[vi], 1 - P[vi])
    lq = np.log(np.clip(qb, 1e-300, 1))
    Sm = np.zeros((big.F, 3)); Sm[fi, si] = lq
    others = np.exp(Sm.sum(1)[fi] - lq)
    out = np.zeros(n); np.add.at(out, vi, sgn * others)
    return out


def select(kind_, budget, p2, rng, pool=None):
    cand = xs if pool is None else pool          # 確かめられる物の主張（E9-1では説明文のある語だけ）
    k = min(int(round(budget * len(xs))), len(cand))
    if k == 0:
        return np.array([], dtype=int)
    if kind_ == 'rand':
        return rng.choice(cand, size=k, replace=False)
    H = -(p2 * np.log(p2 + 1e-12) + (1 - p2) * np.log(1 - p2 + 1e-12))
    if kind_ == 'unc':
        return cand[np.argsort(-H[cand], kind='stable')[:k]]
    # net：物ごとのあいまいさの合計が大きい物から、1物1つずつ（その物で一番あいまいな主張）。足りなければ2周目
    ok = np.zeros(n, bool); ok[cand] = True
    tot = np.zeros(len(ents)); np.add.at(tot, ent_of[cand], H[cand])
    order = [e for e in np.argsort(-tot, kind='stable') if tot[e] > 0 or ok[ent_x[ents[e]]].any()]
    pools = {e: sorted([i for i in ent_x[ents[e]] if ok[i]], key=lambda i: -H[i]) for e in order}
    chosen = []
    rnd = 0
    while len(chosen) < k:
        for e in order:
            if rnd < len(pools[e]):
                chosen.append(pools[e][rnd])
                if len(chosen) == k:
                    break
        rnd += 1
    return np.array(chosen)


def run_bp(u, Wf, frozen):
    return big.bp(u, Wf, frozen=frozen)[0]


def evaluate(p, mask):
    return selective(p[mask], y[mask])


# ---- E9-2：となりの確かめを特徴にする（C+・C++ 用） ----
_ci = {}
for _k, _c in enumerate(allc):
    if _c['kind'] in ('c', 'd'):
        _ci[(_c['kind'], _c['cat'], _c['cat2'])] = L[_k]
x_cat = {i: allc[var[i]]['cat'] for i in xs}
def lg_c(a, b):   # Gemma「aはbの一種」の自信（なければ強い「いいえ」）
    return _ci.get(('c', a, b), -10.0)
def lg_d(a, b):   # Gemma「aとbは両立しない」の自信。WordNetで上下にある組は問いがない→「いいえ」
    a, b = sorted((a, b)); return _ci.get(('d', a, b), -10.0)
def nb_feats(chk, ans):
    F = np.zeros((n, 10))
    by_e = {}
    for j, a_ in zip(chk, ans):
        by_e.setdefault(ent_of[j], []).append((x_cat[j], bool(a_)))
    for e, lst in by_e.items():
        for i in ent_x[ents[e]]:
            b = x_cat[i]
            for a, yes in lst:
                if a == b:
                    continue
                if yes:
                    v1 = sig(lg_c(a, b)); v2 = sig(lg_d(a, b))
                    F[i, 0] += v1; F[i, 1] = max(F[i, 1], v1)      # aと確かめ・aはbの一種 → bも
                    F[i, 2] += v2; F[i, 3] = max(F[i, 3], v2)      # aと確かめ・両立しない → bでない
                    F[i, 6] += 1
                else:
                    v3 = sig(lg_c(b, a))
                    F[i, 4] += v3; F[i, 5] = max(F[i, 5], v3)      # aでないと確かめ・bはaの一種 → bでない
                    F[i, 7] += 1
    own = logit[:, None]
    return np.hstack([F, F[:, [1, 3, 5]] * own, (F[:, 6:8].sum(1) > 0)[:, None].astype(float)])


results = []
NOISE_LIST = [c for c in CHK if c in os.environ.get('E9_CHECKERS', ','.join(CHK)).split(',')] if REAL else ['e6'] if (QUICK or PROBE) else ['e6', 'zero', 'u20']
SPL = [0] if QUICK else [int(x) for x in os.environ.get('E9_SPLITS', '0,1,2,3,4').split(',')]
for sp in SPL:
    r = np.random.default_rng(1000 + sp); perm = r.permutation(len(ents)); tune = set(ents[i] for i in perm[: len(ents) // 5])
    is_tune = np.array([ent_of[i] >= 0 and ents[ent_of[i]] in tune for i in range(n)])
    xt_all = xs[is_tune[xs]]; xe_all = xs[~is_tune[xs]]
    lr2 = LogisticRegression(max_iter=2000).fit(logit[xt_all, None], y[xt_all])
    cal = lr2.decision_function(logit[:, None]); p2 = sig(cal)
    # E4-2 と同じ重みの学習（確かめなし）
    lrw = LogisticRegression(max_iter=2000, C=1e4).fit(np.hstack([logit[:, None], SC[:, :2]])[xt_all], y[xt_all])
    lam, Wt, Wd = lrw.coef_[0]; b0 = lrw.intercept_[0]; Wt, Wd = max(Wt, 0.0), max(Wd, 0.0); Wo = (Wt + Wd) / 2
    Wf0 = np.array([{"trans": Wt, "disj": Wd, "anti": Wo, "cd": Wo}[t] for t in ftypes])
    for noise in NOISE_LIST:
        for budget in BUDGETS:
            for sel in (SELECTS if budget > 0 else ['none']):
                rng = np.random.default_rng(10_000 * sp + 100 * BUDGETS.index(budget) + 7)
                pool = xs[CHK[noise][xs] >= 0] if REAL else None
                chk = select(sel, budget, p2, rng, pool) if budget > 0 else np.array([], dtype=int)
                if REAL:   # 本物の辞書の確かめ
                    ans = CHK[noise][chk].astype(bool); flip = ans != y[chk]
                else:      # 確かめ役のまね（種類ごとに言い間違い）
                    eps = np.array([NOISES[noise][typ[i]] for i in chk])
                    flip = rng.random(len(chk)) < eps
                    ans = np.where(flip, ~y[chk], y[chk])
                checked = np.zeros(n, bool); checked[chk] = True
                cval = np.full(n, 0.5); cval[chk] = np.where(ans, 0.9, 0.1)
                # 評価：確かめていない物の主張（評価用の物）。調整：確かめていない調整用の物の主張
                me = np.zeros(n, bool); me[xe_all] = True; me &= ~checked
                mt = np.zeros(n, bool); mt[xt_all] = True; mt &= ~checked
                # 分類どうしの主張はGemmaの答えのまま（慎重：動かさない）
                base = p2.copy(); base[kind != 'x'] = sig(logit[kind != 'x'])
                Pk = base.copy(); Pk[chk] = cval[chk]
                # 1歩目：確かめた主張と分類どうしの主張だけを「分かっている」とみなす
                known = checked | (kind != 'x')
                # 1歩の押し：他の口に分かっていない主張があるルールは使わない（自分の口は数えない）
                def push_known(Pfull):
                    qb = np.where(bi == 1, Pfull[vi], 1 - Pfull[vi])
                    qb = np.where(known[vi], qb, 0.0)            # 分かっていない口は押しに使わない
                    lq = np.log(np.clip(qb, 1e-300, 1))
                    Sm = np.zeros((big.F, 3)); Sm[fi, si] = lq
                    others = np.exp(Sm.sum(1)[fi] - lq)
                    out = np.zeros(n); np.add.at(out, vi, sgn * others)
                    return out
                Phard = np.where(known, (Pk > 0.5).astype(float), 0.5)
                v1step = push_known(Phard)
                res = dict(split=sp, noise=noise, budget=budget, select=sel, n_checked=int(len(chk)),
                           n_flip=int(flip.sum()), n_eval=int(me.sum()))
                P = {}
                P['A'] = p2.copy()
                # B：1歩だけ
                best = min(KS, key=lambda K: selective(sig(cal + K * v1step)[mt], y[mt])['aurc']) if budget > 0 else 0
                P['B'] = sig(cal + best * v1step); res['K_B'] = best
                # B2：となりの多数決の繰り返し（確かめた・分類どうしは固定、ほかは今の確率で）
                def relax(K, it=10):
                    Pc = Pk.copy()
                    for _ in range(it):
                        Pn = sig(cal + K * push(Pc)); Pn[known] = Pk[known]
                        Pc = 0.5 * Pc + 0.5 * Pn
                    return Pc
                cand = {K: relax(K) for K in KS}
                bk = min(KS, key=lambda K: selective(cand[K][mt], y[mt])['aurc'])
                P['B2'] = cand[bk]; res['K_B2'] = bk
                # C：特徴＋学習（1歩の押しと「同じ物に確かめがあるか」を足す）
                hasc = np.zeros(len(ents));
                if len(chk): np.add.at(hasc, ent_of[chk], 1)
                fC = np.hstack([feat3, np.maximum(v1step, 0)[:, None], np.maximum(-v1step, 0)[:, None],
                                (hasc[np.maximum(ent_of, 0)] > 0)[:, None].astype(float)])
                lrC = LogisticRegression(max_iter=3000).fit(fC[mt], y[mt])
                zC = lrC.decision_function(fC); P['C'] = sig(zC)
                if PROBE:   # 追加確認：確かめのある物だけで学習した版（確かめが入った物の中での比べ用）
                    hm = (hasc[np.maximum(ent_of, 0)] > 0)
                    if budget > 0 and (mt & hm).sum() > 30 and len(set(y[mt & hm])) == 2:
                        lrS = LogisticRegression(max_iter=3000).fit(fC[mt & hm], y[mt & hm])
                        P['Csub'] = np.where(hm, sig(lrS.decision_function(fC)), P['C'])
                zCp = None
                if STRONG and budget > 0 and budget < 1.0:
                    hm = (hasc[np.maximum(ent_of, 0)] > 0)
                    fP = np.hstack([fC, nb_feats(chk, ans)])
                    lrP = LogisticRegression(max_iter=5000, C=1.0).fit(fP[mt], y[mt])
                    zCp = lrP.decision_function(fP); P['Cp'] = sig(zCp)
                    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=sp).fit(fP[mt], y[mt])
                    P['Cpp'] = gb.predict_proba(fP)[:, 1]
                    if (mt & hm).sum() > 30 and len(set(y[mt & hm])) == 2:
                        lrPs = LogisticRegression(max_iter=5000).fit(fP[mt & hm], y[mt & hm])
                        P['Cp_sub'] = np.where(hm, sig(lrPs.decision_function(fP)), P['Cp'])
                        gbs = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=sp).fit(fP[mt & hm], y[mt & hm])
                        P['Cpp_sub'] = np.where(hm, gbs.predict_proba(fP)[:, 1], P['Cpp'])
                # D：網（BP）。物の主張は λ・b（E4-2の学習）、分類どうしはGemmaのまま、確かめた所は固定
                frozen = (kind != 'x') | checked
                uD = lam * logit + b0; uD[kind != 'x'] = logit[kind != 'x']
                uD[chk] = np.where(ans, LCHK, -LCHK)
                candD = {m: run_bp(uD, m * Wf0, frozen) for m in WMUL}
                bm = min(WMUL, key=lambda m: selective(candD[m][mt], y[mt])['aurc'])
                P['D'] = candD[bm]; res['W_D'] = bm
                # F：学習のあとの仕上げ（Cの答えを網に入れる）
                uF = zC.copy(); uF[kind != 'x'] = logit[kind != 'x']; uF[chk] = np.where(ans, LCHK, -LCHK)
                candF = {m: run_bp(uF, m * Wf0, frozen) for m in WMUL}
                bf = min(WMUL, key=lambda m: selective(candF[m][mt], y[mt])['aurc'])
                P['F'] = candF[bf]; res['W_F'] = bf
                if STRONG and 'Cpp' in P:   # F++：C++ の答えを網で仕上げ
                    zG = np.log(np.clip(P['Cpp'], 1e-4, 1 - 1e-4) / np.clip(1 - P['Cpp'], 1e-4, 1))
                    uG = zG.copy(); uG[kind != 'x'] = logit[kind != 'x']; uG[chk] = np.where(ans, LCHK, -LCHK)
                    candG = {m: run_bp(uG, m * Wf0, frozen) for m in WMUL}
                    bg = min(WMUL, key=lambda m: selective(candG[m][mt], y[mt])['aurc'])
                    P['Fpp'] = candG[bg]; res['W_Fpp'] = bg
                if zCp is not None:   # F+：C+ の答えを網で仕上げ
                    uFp = zCp.copy(); uFp[kind != 'x'] = logit[kind != 'x']; uFp[chk] = np.where(ans, LCHK, -LCHK)
                    candFp = {m: run_bp(uFp, m * Wf0, frozen) for m in WMUL}
                    bfp = min(WMUL, key=lambda m: selective(candFp[m][mt], y[mt])['aurc'])
                    P['Fp'] = candFp[bfp]; res['W_Fp'] = bfp
                # 採点
                ent_has = (hasc[np.maximum(ent_of, 0)] > 0)
                ev_s = np.array([e not in tune for e in s_ent]) & ~checked[s_node]
                for k_, p in P.items():
                    rr = evaluate(p, me)
                    out = dict(aurc=rr['aurc'], acc=float(np.mean((p[me] >= .5) == y[me])))
                    m1 = me & ent_has; m0 = me & ~ent_has
                    if m1.sum() > 50: out['aurc_same_ent_checked'] = selective(p[m1], y[m1])['aurc']
                    if m0.sum() > 50: out['aurc_no_check_in_ent'] = selective(p[m0], y[m0])['aurc']
                    for tt in 'TGHE':
                        mm = me & (typ == tt)
                        out[f'acc_{tt}'] = float(np.mean((p[mm] >= .5) == y[mm]))
                    if budget > 0:   # 参考：確かめた主張も含めた評価用の物の主張すべて（確かめた所は確かめの答え9割）
                        pa = p.copy(); pa[chk] = cval[chk]
                        ma = np.zeros(n, bool); ma[xe_all] = True
                        out['aurc_all_incl_checked'] = selective(pa[ma], y[ma])['aurc']
                    pp = p[s_node[ev_s]]
                    out['aurc_common'] = selective(pp, s_mine[ev_s] >= 1)['aurc']
                    res[k_] = out
                results.append(res)
                print(sp, noise, budget, sel, {k_: round(res[k_]['aurc'], 4) for k_ in P}, 'K', res['K_B'], res['K_B2'], 'W', bm, bf,
                      round(time.time() - t0), 's', flush=True)
                json.dump(dict(meta=dict(n=n, factors=len(factors), x=len(xs), noises=NOISES, lchk=float(LCHK)),
                               results=results), open(OUT.replace('E9_results', ('E92_results' if STRONG else 'E91_results') + os.environ.get('E9_SUFFIX', '')) if REAL else OUT.replace('.json', '_quick.json') if QUICK else OUT.replace('.json', '_probe.json') if PROBE else OUT, 'w'), indent=1)
print('done', round(time.time() - t0), 's')
