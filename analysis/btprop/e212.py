"""E2-12: つながり（NLI）を、別のAIが付けたものに替えて、BTProp と ルールを疑うBP を比べる。
cwd=BTPropのclone（eval_dataset/ に Factbench.jsonl・wikibio_preprocess_v2.json）。
python e212.py factcheckgpt|wikibio E2-12_nli.json OUT.json"""
import json, sys, itertools, collections
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '.')
from src.hmm import HMMTreeNode
from src import data_utils as du

ds, nlip, outp = sys.argv[1], sys.argv[2], sys.argv[3]
NLI = json.load(open(nlip))
NAMES = ['entail', 'neutral', 'contradict']
SOURCES = ['元のNLI'] + list(NLI['models'])
TAB = {}
for m, v in NLI['models'].items():
    TAB[m] = {(p, c): f"{NAMES[int(np.argmax(f))]}-{NAMES[int(np.argmax(b))]}"
              for (p, c), f, b in zip(NLI['pairs'], v['fwd'], v['bwd'])}
SRC = {'name': '元のNLI'}
MISS = [0]
def relof(rel, cur, p):
    if SRC['name'] == '元のNLI': return rel
    return TAB[SRC['name']].get((cur, p), rel)
if ds == 'factcheckgpt':
    data = du.load_factcheckgpt_processed_human()
    lab = lambda l: None if isinstance(l, str) else int(not l)       # 1=嘘
else:
    data = du.load_wikibio_processed_human()[120:]
    lab = lambda l: 1 if 'inaccurate' in l else 0
conf = {c['statement']: c['probability'] for c in json.load(open(f'logs/conf_estimation/{ds}_chatgpt.json'))}
prem = {d['statement']: d for d in json.load(open(f'logs/belief_trees/{ds}_chatgpt.json'))}

def build(s):  # BTPropの hmm_forward.py と同じ手順で木を組む
    root = HMMTreeNode('configs/hmm.yaml', s, conf.get(s, 0.5))
    if s not in prem: return None
    seen, todo = [], [s]
    while todo:
        cur = todo.pop()
        if cur not in prem or cur in seen: continue
        seen.append(cur)
        for cd in prem[cur]['child_nodes']:
            kids, rels, m = cd['child_nodes'], cd['mnli_relations'], cd['prem_gen_method']
            if m not in ('decompose', 'logical relation'): kids, rels = kids[:5], rels[:5]
            for i, p in enumerate(kids):
                if p == cur: continue
                rel = rels[i] if m == 'decompose' else relof(rels[i], cur, p)
                if rel == 'neutral-neutral': continue
                if m != 'perturbation' and p not in conf: MISS[0] += 1; continue   # 元で捨てられ自信の無い子
                pc = 1.0 if m == 'perturbation' else conf[p]
                L = root.add_node_via_statement(cur, p, pc, rel, m)
                if L < 2 and m != 'perturbation': todo.append(p)
    return root

def make_items():
    items = []   # (回答番号, ラベル, 自信, BTProp, 木)
    for ri, ex in enumerate(data):
        for s, l in zip(ex['segmented_response'], ex['labels']):
            y = lab(l)
            if y is None or s not in conf: continue
            t = build(s)
            bt = (t.get_posterior() if (t and t.child_nodes) else (1.0 if t is None else conf[s]))
            items.append((ri, y, conf[s], bt, t))
    return items
items = make_items()
print(len(items), 'lies', sum(i[1] for i in items), 'with tree', sum(1 for i in items if i[4] and i[4].child_nodes))

E_T = np.array(items[0][4].emission_probs_true if items[0][4] else HMMTreeNode('configs/hmm.yaml','x',.5).emission_probs_true)
H0 = HMMTreeNode('configs/hmm.yaml', 'x', .5)
def bt_field(c):  # BTPropの表を場に
    return float(np.log(H0.get_p_observation_given_hidden(True, c) / H0.get_p_observation_given_hidden(False, c)))
lgt = lambda c: float(np.log(np.clip(c, 1e-4, 1-1e-4) / (1 - np.clip(c, 1e-4, 1-1e-4))))

def lse(a, b): return np.logaddexp(a, b)
def msg(node, P, pert=False):
    """log m(z), z=0,1：この節と下の木の証拠"""
    if pert: h = P['b']
    elif P.get('bt'): h = bt_field(node.prior_confidence)
    else: h = P['lam'] * (lgt(node.prior_confidence) - P['shift']) * P['scale']
    m = np.array([0.0, h]); W = P['W']
    dec = [c for c, g in zip(node.child_nodes, node.general_relations) if g == 'decompose']
    for c, rel, g in zip(node.child_nodes, node.nli_relations, node.general_relations):
        if g == 'decompose': continue
        mc = msg(c, P, pert=(g == 'perturbation'))
        pen = np.zeros((2, 2))   # [zp, zc]
        if rel == 'entail-entail': pen[1, 0] = pen[0, 1] = W
        elif rel == 'entail-neutral': pen[1, 0] = W
        elif rel == 'neutral-entail': pen[0, 1] = W
        elif 'contradict' in rel: pen[1, 1] = W
        m = m + np.array([lse(mc[0]-pen[zp, 0], mc[1]-pen[zp, 1]) for zp in (0, 1)])
    if dec:
        ms = [msg(c, P) for c in dec]
        A = sum(x[1] for x in ms); T = sum(lse(x[0], x[1]) for x in ms)
        notA = T + np.log(max(1 - np.exp(A - T), 1e-300))
        m = m + np.array([lse(notA, A - W), lse(A, notA - W)])
    return m - m.max()

def score(it, P):   # 嘘らしさ
    ri, y, c, bt, t = it
    if t is None:  # 木がない主張は自分の自信だけ（BTPropの公開コードは確率1にしている）
        h = bt_field(c) if P.get('bt') else P['lam'] * (lgt(c) - P['shift']) * P['scale']
        return 1 - 1 / (1 + np.exp(-h))
    m = msg(t, P); p = 1 / (1 + np.exp(m[0] - m[1]))
    return 1 - p

Y = np.array([i[1] for i in items]); R = np.array([i[0] for i in items])
C = np.array([i[2] for i in items]); BT = np.array([i[3] for i in items])
L = np.array([lgt(c) for c in C])
BTf = np.where([i[4] is None for i in items], C, BT)
print('no tree', sum(i[4] is None for i in items))
resp = np.unique(R); res = []
for src, sp in itertools.product(SOURCES, range(5)):
    SRC['name'] = src; items = make_items()
    BT = np.array([i[3] for i in items])
    rng = np.random.default_rng(sp)
    calr = set(rng.choice(resp, int(round(0.2 * len(resp))), replace=False))
    cal = np.array([R[k] in calr for k in range(len(items))]); test = ~cal
    scale = 2.0 / L[cal].std()
    shift = max(np.quantile(L[cal], np.linspace(.02, .98, 97)), key=lambda b: ((L[cal] > b) == (Y[cal] == 0)).mean())
    def auc(sc, msk): return float(roc_auc_score(Y[msk], np.asarray(sc)[msk]))
    def tune(grid):
        best = None
        for P in grid:
            P = dict(P, shift=shift, scale=scale)
            a = auc([score(items[k], P) if cal[k] else 0 for k in range(len(items))], cal)
            if best is None or a > best[0]: best = (a, P)
        return best[1]
    G = lambda Ws: [dict(lam=l, W=w, b=b) for l, w, b in itertools.product([.5, 1, 2], Ws, [0, .5, 1])]
    Ps = {'疑うBP': tune(G([.5, 1, 2, 4])), '全部信じるBP': tune(G([20])),
          '疑うBP_BTPropの表': tune([dict(bt=True, W=w, b=b, lam=1) for w, b in itertools.product([.5, 1, 2, 4, 20], [0, .5, 1, 2])])}
    row = {'つながり': src, 'split': sp, 'n_test': int(test.sum()), 'lies_test': int(Y[test].sum()),
           'AIの自信': auc(1 - C, test), 'BTProp': auc(1 - BT, test), 'BTProp_木なしは自信': auc(1 - np.where([i[4] is None for i in items], C, BT), test)}
    for k, P in Ps.items():
        row[k] = auc([score(it, P) for it in items], test); row[k + '_設定'] = {a: P[a] for a in P if a not in ('shift', 'scale')}
    print(row); res.append(row)
json.dump(res, open(outp, 'w'), ensure_ascii=False, indent=1)

# まとめ（5通りの平均）
keys = ['AIの自信', 'BTProp', 'BTProp_木なしは自信', '全部信じるBP', '疑うBP']
print('\n平均AUROC', ds)
for src in SOURCES:
    rs = [r for r in res if r['つながり'] == src]
    bt = 'BTProp_木なしは自信' if ds == 'factcheckgpt' else 'BTProp'
    d = [r['疑うBP'] - r[bt] for r in rs]
    print(f"{src:28s}", ' '.join(f"{k}={np.mean([r[k] for r in rs]):.3f}" for k in keys),
          f"| 疑う−BTProp {np.mean(d):+.3f} ({sum(x > 0 for x in d)}/5)")
