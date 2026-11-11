"""E9-3：候補を選ばない 1,681語×46分類（77,326主張）で、網と強い学習を比べ直す。計画: PLAN_E9-3.md（固定）
網のルールは Gemma の答えだけ（「一種」はい・「両立しない」はい）。WordNet の階層は網にも学習にも入れない。
使い方: python3 e93.py [quick]   環境変数 E93_SPLITS=0,1,...  → E93_results(_sN).json
一覧なしの辞書の確かめ（2026-11-11、plans/E9n_nolist_recheck.md）: E93_CHECKER=../../data/E9n_checker_scores.json
"""
import json, sys, os, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'e4'))
from e41 import sig, selective

QUICK = len(sys.argv) > 1 and sys.argv[1] == 'quick'
BUDGETS = [0.0, 0.01, 0.02, 0.05, 0.10]
SELECTS = ['rand', 'unc', 'net']
WMUL = [0.5, 1.0, 2.0, 4.0]
LCHK = np.log(0.9 / 0.1)
SPL = [int(x) for x in os.environ.get('E93_SPLITS', '0' if QUICK else '0,1,2,3,4').split(',')]
t0 = time.time()

# ---------------- データ ----------------
D = json.load(open(os.path.join(HERE, '..', 'e4', 'e42_data.json')))
LALL = np.array(json.load(open(os.path.join(HERE, '..', '..', 'data', 'E4-2_answers_gemma-2-2b-it.json')))['logit'])
CK = json.load(open(os.environ.get('E93_CHECKER') or os.path.join(HERE, '..', '..', 'data', 'E9-1_checker_scores.json')))   # 一覧なし版: E93_CHECKER=../../data/E9n_checker_scores.json
cl = D['claims']; syn = list(D['cats']); name = D['cats']
ENTS = CK['ents']; CATN = CK['cats']; NE, NC = len(ENTS), len(CATN)
cidx = {name[s]: s for s in syn}                     # 名前 → synset
idx = {}
for i, c in enumerate(cl):
    idx[(c['kind'], c.get('ent', c['cat']), c['cat'] if c['kind'] == 'x' else c['cat2'])] = i
X = np.zeros((NE, NC)); Y = np.zeros((NE, NC), bool)
for e, w in enumerate(ENTS):
    for j, cn in enumerate(CATN):
        k = idx[('x', w, cidx[cn])]; X[e, j] = LALL[k]; Y[e, j] = cl[k]['gold']
chs = CK['checkers']['e8D']
CHK = np.array([{'1': 1, '0': 0, '-': -1}[ch] for ch in chs]).reshape(NE, NC)
SCO = np.array(CK['scores_e8D'], float).reshape(NE, NC) / 1000.0   # -0.001 = なし
# Gemma のルール（はい だけ使う）
def rl(kind, a, b):
    k = idx.get((kind, cidx[a], cidx[b]))
    return None if k is None else LALL[k]
IMP = [(a, b) for a in range(NC) for b in range(NC) if a != b and (rl('c', CATN[a], CATN[b]) or -1) > 0]
DIS = []
for a in range(NC):
    for b in range(a + 1, NC):
        s1, s2 = sorted([cidx[CATN[a]], cidx[CATN[b]]])
        k = idx.get(('d', s1, s2))
        if k is not None and LALL[k] > 0:
            DIS.append((a, b))
print('語', NE, '分類', NC, '本当', round(Y.mean(), 4), '一種のルール', len(IMP), '両立しないルール', len(DIS), flush=True)
FA = np.array([a for a, b in IMP] + [a for a, b in DIS]); FB = np.array([b for a, b in IMP] + [b for a, b in DIS])
FT = np.array([0] * len(IMP) + [1] * len(DIS))       # 0: a→b（a=1,b=0 が悪い） 1: a,b 両立しない（両方1が悪い）
NF = len(FA)
IA = np.zeros((NF, NC)); IA[np.arange(NF), FA] = 1
IB = np.zeros((NF, NC)); IB[np.arange(NF), FB] = 1

# ---- 常識の正解（600個、Claude、方法を流す前に判定） ----
SL = json.load(open(os.path.join(HERE, 'labels', 'E9-3_sample_labeled_claude.json')))['items']
eix = {w: e for e, w in enumerate(ENTS)}; jix = {c: j for j, c in enumerate(CATN)}
S_E = np.array([eix[s['ent']] for s in SL]); S_J = np.array([jix[s['cat']] for s in SL]); S_M = np.array([s['mine'] for s in SL])


def bp(U, Wt, Wd, frozen, maxit=100, damp=0.5, tol=1e-5):
    """物ごとの網（ペアのルール）。U: 物×分類の場、frozen: 動かさない主張。返り値 P(本当)"""
    W = np.where(FT == 0, Wt, Wd); c = 1.0 - np.exp(-W)
    ra = np.zeros((NE, NF)); rb = np.zeros((NE, NF))
    fz = frozen.astype(float)
    for it in range(maxit):
        tot = U + ra @ IA + rb @ IB
        qa = sig(tot[:, FA] - ra); qb = sig(tot[:, FB] - rb)
        # a→b：a=1,b=0 が悪い。b へ：-log(1-c qa)、a へ：log(1-c(1-qb))
        # 両立しない：両方1が悪い。b へ：log(1-c qa)、a へ：log(1-c qb)
        nb_ = np.where(FT == 0, -np.log(np.clip(1 - c * qa, 1e-300, 1)), np.log(np.clip(1 - c * qa, 1e-300, 1)))
        na_ = np.where(FT == 0, np.log(np.clip(1 - c * (1 - qb), 1e-300, 1)), np.log(np.clip(1 - c * qb, 1e-300, 1)))
        na_ = na_ * (1 - fz[:, FA]); nb_ = nb_ * (1 - fz[:, FB])
        ra2 = damp * ra + (1 - damp) * na_; rb2 = damp * rb + (1 - damp) * nb_
        d = max(np.abs(ra2 - ra).max(), np.abs(rb2 - rb).max()); ra, rb = ra2, rb2
        if d < tol:
            break
    return sig(U + ra @ IA + rb @ IB)


def signed(A):
    """Gemmaの答え A(0/1) のとき、主張を本当にすると破れる(−)・うそにすると破れる(+)のルール数（種類別）"""
    st = np.zeros((NE, NC)); sd = np.zeros((NE, NC))
    for a, b in IMP:   # a=1 で b=0 なら破れる
        st[:, a] -= (A[:, b] == 0); st[:, b] += (A[:, a] == 1)
    for a, b in DIS:
        sd[:, a] -= (A[:, b] == 1); sd[:, b] -= (A[:, a] == 1)
    return st, sd


A0 = (X > 0).astype(int)
ST, SD = signed(A0)
oth_max = np.zeros((NE, NC)); oth_mean = np.zeros((NE, NC))
for j in range(NC):
    o = np.delete(X, j, axis=1); oth_max[:, j] = o.max(1); oth_mean[:, j] = o.mean(1)
BASEF = np.stack([X, oth_max, oth_mean, ST, SD], -1)          # 物×分類×5
RULE_IMP = np.zeros((NC, NC)); RULE_DIS = np.zeros((NC, NC))
for a, b in IMP: RULE_IMP[a, b] = 1
for a, b in DIS: RULE_DIS[a, b] = RULE_DIS[b, a] = 1


def nb_feats(C):
    """C: 物×分類（1=確かめてはい、0=確かめていいえ、-1=確かめていない）→ 確かめの隣の特徴"""
    yes = (C == 1).astype(float); no = (C == 0).astype(float)
    sup = yes @ RULE_IMP                 # はいの a から a→b で b を支える数
    agt = yes @ RULE_DIS                 # はいの a と両立しない b
    agt2 = no @ RULE_IMP.T               # いいえの a、b→a なら b もいいえ
    ny = yes.sum(1, keepdims=True) * np.ones((1, NC)); nn = no.sum(1, keepdims=True) * np.ones((1, NC))
    has = ((ny + nn) > 0).astype(float)
    F = np.stack([sup, agt, agt2, ny, nn, has, (sup > 0) * X, (agt > 0) * X, (agt2 > 0) * X], -1)
    return F


def select(kind_, k, P2, rng, pool):
    if k == 0:
        return np.zeros((0,), int)
    if kind_ == 'rand':
        return rng.choice(pool, size=k, replace=False)
    H = -(P2 * np.log(P2 + 1e-12) + (1 - P2) * np.log(1 - P2 + 1e-12)).ravel()
    if kind_ == 'unc':
        return pool[np.argsort(-H[pool], kind='stable')[:k]]
    ok = np.zeros(NE * NC, bool); ok[pool] = True
    tot = (H * ok).reshape(NE, NC).sum(1)
    order = np.argsort(-tot, kind='stable')
    lists = [sorted([e * NC + j for j in range(NC) if ok[e * NC + j]], key=lambda i: -H[i]) for e in range(NE)]
    out, rnd = [], 0
    while len(out) < k:
        for e in order:
            if rnd < len(lists[e]):
                out.append(lists[e][rnd])
                if len(out) == k:
                    break
        rnd += 1
    return np.array(out)


def aurc(P, M, T=None):
    T = Y if T is None else T
    return selective(P[M], T[M])['aurc']


pool_all = np.where(CHK.ravel() >= 0)[0]
results = []
OUT = os.path.join(HERE, '..', '..', 'results', 'E9', 'E93_results' + ('_quick' if QUICK else os.environ.get('E93_SUFFIX', '')) + '.json')
for sp in SPL:
    r = np.random.default_rng(1000 + sp); perm = r.permutation(NE)
    is_tune = np.zeros(NE, bool); is_tune[perm[: NE // 5]] = True
    TM = np.repeat(is_tune[:, None], NC, 1); EM = ~TM
    lr2 = LogisticRegression(max_iter=2000).fit(X[TM][:, None], Y[TM])
    CAL = lr2.decision_function(X.reshape(-1, 1)).reshape(NE, NC); P2 = sig(CAL)
    lrw = LogisticRegression(max_iter=2000, C=1e4).fit(np.stack([X[TM], ST[TM], SD[TM]], 1), Y[TM])
    lam, Wt, Wd = lrw.coef_[0]; b0 = lrw.intercept_[0]; Wt, Wd = max(Wt, 0.0), max(Wd, 0.0)
    # K：全部の主張を辞書で見た場合（参考）
    sc = SCO.copy(); miss = sc < 0; sc[miss] = 0
    FK = np.stack([sc, miss.astype(float)], -1)
    lrK = LogisticRegression(max_iter=2000).fit(FK[TM], Y[TM]); PK = lrK.predict_proba(FK.reshape(-1, 2))[:, 1].reshape(NE, NC)
    print('split', sp, 'λ', round(lam, 3), 'Wt', round(Wt, 3), 'Wd', round(Wd, 3), 'b', round(b0, 2), round(time.time() - t0), 's', flush=True)
    for budget in BUDGETS:
        for sel in (SELECTS if budget > 0 else ['none']):
            rng = np.random.default_rng(10_000 * sp + 100 * BUDGETS.index(budget) + 7)
            k = int(round(budget * NE * NC))
            chk = select(sel, k, P2, rng, pool_all)
            C = -np.ones((NE, NC), int); C.ravel()[chk] = CHK.ravel()[chk]
            checked = C >= 0
            ent_has = checked.any(1)
            me = EM & ~checked; mt = TM & ~checked
            m_main = me & ent_has[:, None]
            res = dict(split=sp, budget=budget, select=sel, n_checked=int(len(chk)),
                       check_wrong=int((CHK.ravel()[chk] != Y.ravel()[chk]).sum()), n_eval=int(me.sum()), n_main=int(m_main.sum()))
            P = {'A': P2.copy(), 'K': PK.copy()}
            NB = nb_feats(C)
            F = np.concatenate([BASEF, NB], -1)
            Fm = F.reshape(-1, F.shape[-1]); mtr = mt.ravel()
            mu, sd = Fm[mtr].mean(0), Fm[mtr].std(0) + 1e-9
            lrC = LogisticRegression(max_iter=5000).fit((Fm[mtr] - mu) / sd, Y.ravel()[mtr])
            P['C'] = lrC.predict_proba((Fm - mu) / sd)[:, 1].reshape(NE, NC)
            gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=sp).fit(Fm[mtr], Y.ravel()[mtr])
            P['Cpp'] = gb.predict_proba(Fm)[:, 1].reshape(NE, NC)
            ansU = np.where(C == 1, LCHK, -LCHK)
            # D：網
            U = lam * X + b0; U = np.where(checked, ansU, U)
            cD = {m: bp(U, m * Wt, m * Wd, checked) for m in WMUL}
            bm = min(WMUL, key=lambda m: aurc(cD[m], mt)); P['D'] = cD[bm]; res['W_D'] = bm
            # F++：木の学習の答えを網で仕上げ
            zG = np.log(np.clip(P['Cpp'], 1e-4, 1 - 1e-4) / np.clip(1 - P['Cpp'], 1e-4, 1))
            UG = np.where(checked, ansU, zG)
            cG = {m: bp(UG, m * Wt, m * Wd, checked) for m in WMUL}
            bg = min(WMUL, key=lambda m: aurc(cG[m], mt)); P['Fpp'] = cG[bg]; res['W_Fpp'] = bg
            # 採点
            ev_s = (~is_tune[S_E]) & (~checked[S_E, S_J])
            ev_s_main = ev_s & ent_has[S_E]
            for kk, p in P.items():
                o = dict(aurc_main=aurc(p, m_main) if m_main.sum() > 50 else None, aurc_all=aurc(p, me),
                         acc_all=float(np.mean((p[me] >= .5) == Y[me])),
                         recall_true=float(np.mean(p[me & Y] >= .5)), precision_true=float(np.mean(Y[me & (p >= .5)])) if (me & (p >= .5)).any() else None)
                pp = p[S_E[ev_s], S_J[ev_s]]
                o['aurc_common_all'] = selective(pp, S_M[ev_s] >= 1)['aurc']
                if ev_s_main.sum() > 20:
                    o['aurc_common_main'] = selective(p[S_E[ev_s_main], S_J[ev_s_main]], S_M[ev_s_main] >= 1)['aurc']
                o['n_common_all'] = int(ev_s.sum())
                res[kk] = o
            results.append(res)
            print(sp, budget, sel, 'main', {kk: (round(res[kk]['aurc_main'], 4) if res[kk]['aurc_main'] is not None else None) for kk in P},
                  'all', {kk: round(res[kk]['aurc_all'], 4) for kk in P}, 'W', bm, bg, round(time.time() - t0), 's', flush=True)
            json.dump(dict(meta=dict(n_ents=NE, n_cats=NC, imp=len(IMP), dis=len(DIS), true_rate=float(Y.mean())), results=results),
                      open(OUT, 'w'), indent=1)
print('done', round(time.time() - t0), 's')
