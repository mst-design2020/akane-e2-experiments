"""E14 の前の数え（plans/E14_count_gaps.md）：E8n 方法D の1,000問で、一番強い道と段の数を1問ずつ出す。
python3 count_gaps.py ../../data/E8n_chains_nli.json   → rows_D.json（行き止まりなどの分け方は、この表を読んで決まりで分けた）"""
import json, sys, os, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("E8_DIR", os.path.join(HERE, "..", "e8"))
sys.path.insert(0, os.path.join(HERE, "..", "nolist"))
import e8n_analyze as A
r = json.load(open(sys.argv[1]))
L, ST, AR, CH = A.load(r)
old = sorted(json.load(open(os.path.join(HERE, "..", "e6", "e6_items.json"))), key=lambda x: x["id"])
new = sorted(json.load(open(os.path.join(HERE, "..", "e8", "e8_new_items_labeled.json"))), key=lambda x: x["id"])
items = old + new
rows = []
for i, it in enumerate(items):
    nd, lk = CH[(it["id"], "D")]
    o = A.Item(it, "D", nd, lk, L, ST, AR)
    s, ok, depth, path = o.maxprod()
    y = int(it["mine"] >= 1)
    rows.append(dict(i=i, x=it["ent"], C=it["catw"], s=s, yes=int(s > 0.5), y=y, depth=depth,
                     path=[list(p) for p in path], links=lk))
json.dump(rows, open("rows_D.json", "w"))
yes = [r_ for r_ in rows if r_["yes"]]
fy = [r_ for r_ in yes if not r_["y"]]; ty = [r_ for r_ in yes if r_["y"]]
fn = [r_ for r_ in rows if not r_["yes"] and r_["y"]]
print("items", len(rows), "yes", len(yes), "wrong yes", len(fy), "right yes", len(ty), "wrong no", len(fn))
print("depth of wrong yes", collections.Counter(r_["depth"] for r_ in fy))
print("depth of right yes", collections.Counter(r_["depth"] for r_ in ty))
def edges(r_):
    p = [tuple(n) for n in r_["path"]]
    return [(p[k], p[k+1]) for k in range(len(p)-1)]
def mids(r_):
    return sorted({n[0] for n in [tuple(n) for n in r_["path"]][1:] if n[0] != r_["C"]})
for name, grp in [("wrong yes", fy), ("right yes", ty)]:
    multi = [r_ for r_ in grp if r_["depth"] >= 1]
    ec = collections.Counter(e for r_ in multi for e in set(edges(r_)))
    wc = collections.Counter(w for r_ in multi for w in mids(r_))
    print(f"\n== {name}: with >=1 step {len(multi)}")
    print(" distinct edges", len(ec), " edges used by >=2 items", sum(1 for v in ec.values() if v >= 2))
    print(" top edges", ec.most_common(12))
    print(" top middle words", wc.most_common(15))
    # share of items explained by the top-k edges
    for k in (1, 3, 5, 10):
        top = {e for e, _ in ec.most_common(k)}
        cov = sum(1 for r_ in multi if set(edges(r_)) & top)
        print(f"  items touching top{k} edges: {cov}/{len(multi)}")
