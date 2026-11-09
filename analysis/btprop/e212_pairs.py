"""E2-12 判定する組を集める。cwd=BTPropのclone（logs/ と eval_dataset/ があること）。
PAIRS[ds] = [(親, 子)]（分解以外の枝、BTPropと同じく作り変えは5本まで）
ORIG[ds][(親,子)] = 元のNLI（'entail-neutral' の形。前＝親→子、後＝子→親）
HELD = 聞き方選び用の200組（WikiBioの、評価に出る主張から届かない木＝前120人の分）"""
import json, random
TREES = {ds: {d['statement']: d for d in json.load(open(f'logs/belief_trees/{ds}_chatgpt.json'))}
         for ds in ('factcheckgpt', 'wikibio')}

def edges(prem, s):
    for cd in prem[s]['child_nodes']:
        kids, rels, m = cd['child_nodes'], cd['mnli_relations'], cd['prem_gen_method']
        if m not in ('decompose', 'logical relation'): kids, rels = kids[:5], rels[:5]
        for p, r in zip(kids, rels):
            if p != s: yield p, r, m

PAIRS, ORIG = {}, {}
for ds, prem in TREES.items():
    o = {}
    for s in prem:
        for p, r, m in edges(prem, s):
            if m != 'decompose': o.setdefault((s, p), r)
    ORIG[ds] = o; PAIRS[ds] = sorted(o)

def reach(prem, roots):
    seen, todo = set(), list(roots)
    while todo:
        s = todo.pop()
        if s in seen or s not in prem: continue
        seen.add(s)
        todo += [p for p, r, m in edges(prem, s) if m != 'perturbation']
    return seen

def eval_statements(ds):
    if ds == 'factcheckgpt':
        out = []
        for line in open('eval_dataset/Factbench.jsonl', encoding='utf-8'):
            if line.strip():
                d = json.loads(line)
                if d['source'] == 'factcheckgpt': out += d['claims']
        return out
    return [x['statement'] for x in json.load(open('logs/conf_estimation/hmm_posterior.json'))]

_used = reach(TREES['wikibio'], eval_statements('wikibio'))
_cand = [pq for pq in PAIRS['wikibio'] if pq[0] not in _used]
HELD = random.Random(0).sample(_cand, 200)
HELD_SET = set(HELD)
# 実際に聞く組：評価に届く組（FactCheckGPTは全部）＋聞き方選びの200組
RUN = sorted(set(PAIRS['factcheckgpt']) | {pq for pq in PAIRS['wikibio'] if pq[0] in _used} | HELD_SET)

if __name__ == '__main__':
    for ds in PAIRS:
        print(ds, '組', len(PAIRS[ds]))
    print('実際に聞く組', len(RUN))
    print('聞き方選びの候補', len(_cand), '→', len(HELD))
    import collections
    print(collections.Counter(ORIG['wikibio'][pq] for pq in HELD).most_common())
