# ===== 設定 =====
USE_MOCK_MOUTH = False      # True: 作り物の口（動作確認用。結果は研究に使わない）
SAVE_TO_DRIVE  = False       # 結果をドライブの AKANE_アーキテクチャ/E2_results に保存
MODEL_ID = "google/gemma-2-2b-it"
SPLITS = [0, 1, 2, 3, 4]    # 調整用（ことばの20%）／評価用の分け方（5通り）
PBIT_SEEDS = [1, 2]         # p-bitの乱数（分け方ごとに2通り → 合計10回）
DATA_SEED = 0               # テストの作り方（固定）

import os, json, time, itertools, collections
import numpy as np
for pkg, mod in [("python-sat", "pysat"), ("nltk", "nltk"), ("wordfreq", "wordfreq")]:
    try: __import__(mod)
    except ImportError: os.system(f"pip -q install {pkg}")
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from sklearn.metrics import roc_auc_score

OUT = "/content/E2_results" if os.path.exists("/content") else "E2_results"
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = "/content/drive/MyDrive/AKANE_アーキテクチャ/E2_results"
os.makedirs(OUT, exist_ok=True)
print("保存先:", OUT)

import nltk
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency
try: nltk.data.find("corpora/wordnet")
except LookupError:
    try: nltk.data.find("corpora/wordnet.zip")
    except LookupError: nltk.download("wordnet", quiet=True)
# 大分類（7つ）と種類（25）。種類は WordNet の見出し、英語の言い方、大分類
CONT = {"animal": "animal", "plant": "plant", "food": "food", "artifact": "man-made object",
        "activity": "activity", "condition": "medical condition", "feeling": "feeling"}
CATS = {"mammal.n.01": ("mammal", "animal"), "bird.n.01": ("bird", "animal"), "fish.n.01": ("fish", "animal"),
        "insect.n.01": ("insect", "animal"), "reptile.n.01": ("reptile", "animal"),
        "tree.n.01": ("tree", "plant"), "flower.n.01": ("flower", "plant"),
        "vegetable.n.01": ("vegetable", "food"), "edible_fruit.n.01": ("fruit", "food"), "beverage.n.01": ("drink", "food"),
        "dish.n.02": ("dish", "food"), "baked_goods.n.01": ("baked good", "food"),
        "tool.n.01": ("tool", "artifact"), "vehicle.n.01": ("vehicle", "artifact"),
        "musical_instrument.n.01": ("musical instrument", "artifact"), "weapon.n.01": ("weapon", "artifact"),
        "furniture.n.01": ("furniture", "artifact"), "garment.n.01": ("clothing", "artifact"),
        "container.n.01": ("container", "artifact"), "building.n.01": ("building", "artifact"),
        "sport.n.01": ("sport", "activity"), "dance.n.01": ("dance", "activity"), "game.n.01": ("game", "activity"),
        "disease.n.01": ("disease", "condition"), "emotion.n.01": ("emotion", "feeling")}
# どちらとも言える大分類の組（偽の問題に使わない）：植物と食べ物、動物と食べ物
AMBIG = {frozenset(("plant", "food")), frozenset(("animal", "food"))}
MASS = {"furniture", "clothing"}
art = lambda w: w if w in MASS else ("an " if w[0] in "aeiou" else "a ") + w
lemma = lambda s: s.lemmas()[0].name()
by_co = collections.defaultdict(list)          # 種類 → ことば
for sid in CATS:
    S = wn.synset(sid)
    for leaf in S.closure(lambda s: s.hyponyms()):
        w = lemma(leaf)
        if "_" in w or not w.isalpha() or not w.islower() or zipf_frequency(w, "en") < 2.5: continue
        if len(wn.synsets(w)) != 1: continue        # 意味が1つしかない語だけ（取り違えを避ける）
        anc = set(leaf.closure(lambda s: s.hypernyms()))
        hits = [c for c in CATS if wn.synset(c) in anc]
        if hits == [sid]: by_co[sid].append(w)
rng = np.random.default_rng(DATA_SEED)
COUNTRIES = [k for k in CATS if len(set(by_co[k])) >= 4]
STMT, GOLD, KIND, CITY = [], [], [], []
idx = {}
def add(key, text, gold, kind, city=None):
    if key in idx: return idx[key]
    idx[key] = len(STMT); STMT.append(text); GOLD.append(gold); KIND.append(kind); CITY.append(city)
    return idx[key]
cont_of = {k: CATS[k][1] for k in COUNTRIES}
same_cont = collections.defaultdict(list)
for k in COUNTRIES: same_cont[cont_of[k]].append(k)
others = lambda z: [w for w in CONT if w != z and frozenset((w, z)) not in AMBIG]
# 種類→大分類
for k in COUNTRIES:
    z = cont_of[k]; fz = rng.choice(others(z))
    for zz in (z, fz):
        add(("CK", k, zz), f"{art(CATS[k][0]).capitalize()} is a kind of {CONT[zz]}.", int(zz == z), "種類→大分類")
# ことば→種類、ことば→大分類（1つの種類から多くても30語）
CITIES = []
for k in COUNTRIES:
    ws = sorted(set(by_co[k]))
    pick = list(rng.choice(ws, min(30, len(ws)), replace=False))
    for name in pick:
        z = cont_of[k]; CITIES.append(name)
        same = [w for w in same_cont[z] if w != k]     # 偽の種類は、なるべく同じ大分類から2つ
        pool = same if len(same) >= 2 else same + [w for w in COUNTRIES if cont_of[w] in others(z)]
        fks = list(rng.choice(pool, 2, replace=False))
        for kk in [k] + fks:
            add(("CC", name, kk), f"{name.capitalize()} is a kind of {CATS[kk][0]}.", int(kk == k), "ことば→種類", name)
        for zz in (z, rng.choice(others(z))):
            add(("CL", name, zz), f"{name.capitalize()} is a kind of {CONT[zz]}.", int(zz == z), "ことば→大分類", name)
GOLD = np.array(GOLD)
DUMMY = len(STMT)   # 「両立しない」ルールの結論に使う、いつも0の素子
# ルール：(前提1, 前提2, 結論)。結論がDUMMYなら「前提が両方1ではいけない」
RULES = set()
for (t, a, b), i in list(idx.items()):
    if t == "CC":
        for zz in CONT:
            if ("CK", b, zz) in idx and ("CL", a, zz) in idx:
                RULES.add((i, idx[("CK", b, zz)], idx[("CL", a, zz)]))     # ことばが種類に入り、種類が大分類に入る → ことばはその大分類
def excl(keys):
    ids = [idx[k] for k in keys if k in idx]
    for p, q in itertools.combinations(sorted(ids), 2): RULES.add((p, q, DUMMY))
for name in set(CITIES):
    excl([("CC", name, k) for k in COUNTRIES]); excl([("CL", name, z) for z in CONT])
for k in COUNTRIES:
    excl([("CK", k, z) for z in CONT])
RULES = sorted(RULES)
# ルールが正解で全部守られているか（守られていなければデータの誤り）
G = np.append(GOLD, 0)
bad = sum(1 for p, q, h in RULES if G[p] == 1 and G[q] == 1 and G[h] == 0)
print(f"種類 {len(COUNTRIES)}、ことば {len(set(CITIES))}、問題 {len(STMT)}（本物 {GOLD.sum()}）、ルール {len(RULES)}本、正解が破るルール {bad}")
print("例:", " | ".join(STMT[i] for i in [0, 1, 60, 61, 62, 63, 64]))
d = json.load(open("../data/wn_mouth_gemma-2-2b-it.json"))
assert d["stmt"] == STMT, "mismatch"
LOGIT = np.array(d["logit"])
def pbit(lg, rules, lam, W, rng, chains=10, burn=100, sweeps=400, col=None):
    R = np.array(rules); n = len(lg); a, b, h = R[:, 0], R[:, 1], R[:, 2]
    col = coloring(n, rules) if col is None else col
    groups = [np.nonzero(col == c)[0] for c in range(col.max() + 1)]
    x = (rng.random((chains, n)) < 0.5).astype(float); x[:, DUMMY] = 0; acc = np.zeros(n)
    for t in range(burn + sweeps):
        for g in groups:
            dE = np.tile(-lam * lg, (chains, 1))
            xa, xb, xh = x[:, a], x[:, b], x[:, h]
            np.add.at(dE.T, a, (W * xb * (1 - xh)).T)     # a を1にすると違反になる
            np.add.at(dE.T, b, (W * xa * (1 - xh)).T)
            np.add.at(dE.T, h, (-W * xa * xb).T)          # h を0にすると違反になる
            p1 = 1 / (1 + np.exp(np.clip(dE[:, g], -50, 50)))
            x[:, g] = (rng.random((chains, len(g))) < p1).astype(float)
            x[:, DUMMY] = 0
        if t >= burn: acc += x.mean(0)
    return acc / sweeps

def coloring(n, rules):
    nb = [set() for _ in range(n)]
    for r in rules:
        for u in r:
            if u != DUMMY: nb[u].update(v for v in r if v != u and v != DUMMY)
    col = -np.ones(n, int)
    for i in np.argsort([-len(s) for s in nb]):
        used = {col[j] for j in nb[i]}; c = 0
        while c in used: c += 1
        col[i] = c
    col[DUMMY] = col.max() + 1
    return col

A8 = ((np.arange(8)[:, None] >> np.arange(3)) & 1)        # 3つ組の全部の組み合わせ
VIOL = (A8[:, 0] == 1) & (A8[:, 1] == 1) & (A8[:, 2] == 0)
def bp(lg, rules, lam, W, damp=0.5, iters=200, tol=1e-6):
    R = np.array(rules); n = len(lg); F = len(R)
    un = np.stack([np.zeros(n), lam * lg], 1)
    lt = -W * VIOL                                          # (8,)
    msg = np.zeros((F, 3, 2)); conv = False
    for it in range(iters):
        tot = un.copy()
        for j in range(3): np.add.at(tot, R[:, j], msg[:, j])
        vin = tot[R] - msg                                  # (F,3,2) 主張→ルール
        vin[R == DUMMY] = (0.0, -1e9)                       # いつも0の印は0に固定（多くのルールから押されても動かさない）
        s = lt[None, :] + sum(vin[:, j, A8[:, j]] for j in range(3))   # (F,8)
        new = np.zeros_like(msg)
        for j in range(3):
            for v in (0, 1):
                sel = A8[:, j] == v
                new[:, j, v] = np.logaddexp.reduce(s[:, sel] - vin[:, j, v][:, None], axis=1)
        new -= new.max(2, keepdims=True)
        new = damp * msg + (1 - damp) * new
        d = np.abs(new - msg).max(); msg = new
        if d < tol: conv = True; break
    tot = un.copy()
    for j in range(3): np.add.at(tot, R[:, j], msg[:, j])
    return 1 / (1 + np.exp(np.clip(tot[:, 0] - tot[:, 1], -50, 50))), conv

def maxsat(lg, rules, lam, W):
    n = DUMMY; wc = WCNF()
    for v in range(n):
        wc.append([(v + 1) if lg[v] > 0 else -(v + 1)], weight=max(1, int(round(100 * lam * abs(lg[v])))))
    for p, q, h in rules:
        wc.append([-(p + 1), -(q + 1)] + ([h + 1] if h != DUMMY else []), weight=max(1, int(round(100 * W))))
    with RC2(wc) as r: m = r.compute()
    return np.append((np.array(m[:n]) > 0).astype(int), 0)

def violations(ans, rules):
    c = np.zeros(len(ans))
    for p, q, h in rules:
        if ans[p] == 1 and ans[q] == 1 and ans[h] == 0:
            c[p] += 1; c[q] += 1; c[h] += 1
    return c

