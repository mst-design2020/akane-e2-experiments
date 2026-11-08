"""Run BTProp's own HMM on its released logs (ChatGPT trees and confidences). Run inside the BTProp clone.
python ../../bt_repro.py factcheckgpt|wikibio  -> logs/conf_estimation/hmm_posterior_<ds>.json"""
import json, sys, contextlib, io
sys.path.insert(0, '.')
from src.hmm import HMMTreeNode
from src import data_utils as du
ds = sys.argv[1]
data = du.load_factcheckgpt_processed_human() if ds == 'factcheckgpt' else du.load_wikibio_processed_human()[120:]
conf = {c['statement']: c['probability'] for c in json.load(open(f'logs/conf_estimation/{ds}_chatgpt.json'))}
prem = {d['statement']: d for d in json.load(open(f'logs/belief_trees/{ds}_chatgpt.json'))}
out = []
for ex in data:
    for s in ex['segmented_response']:
        oc = conf.get(s, 0.5)
        if s not in prem:
            out.append({'statement': s, 'probability': 1.0}); continue
        root = HMMTreeNode('configs/hmm.yaml', s, oc)
        seen, todo = [], [s]
        while todo:
            cur = todo.pop()
            if cur not in prem or cur in seen: continue
            seen.append(cur)
            for cd in prem[cur]['child_nodes']:
                kids, rels, m = cd['child_nodes'], cd['mnli_relations'], cd['prem_gen_method']
                if m not in ('decompose', 'logical relation'):
                    kids, rels = kids[:5], rels[:5]
                for i, p in enumerate(kids):
                    if p == cur or rels[i] == 'neutral-neutral': continue
                    pc = 1.0 if m == 'perturbation' else conf[p]
                    L = root.add_node_via_statement(cur, p, pc, rels[i], m)
                    if L < 2 and m != 'perturbation': todo.append(p)
        nc = root.get_posterior() if root.child_nodes else oc
        out.append({'statement': s, 'probability': nc})
json.dump(out, open(f'logs/conf_estimation/hmm_posterior_{ds}.json', 'w'), indent=1)
print(len(out))
