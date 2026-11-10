"""E8 新しい問題（確かめ用）：E4-2の主張から、E6/E7の700個に出ない物の語だけで300個を、700個と同じ種類の割合で無作為に取る。
python3 e8_newset.py → e8_new_items.json（正解の常識判定は別に付ける）"""
import json, sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
E4 = os.path.join(HERE, "..", "e4")  # e42_data.json は analysis/e4 で e42_data.py から作る
sys.path.insert(0, E4)
os.chdir(E4)
import e42
D = json.load(open("e42_data.json")); allc = D["claims"]
L = np.array(json.load(open("../../data/E4-2_answers_gemma-2-2b-it.json"))["logit"])
var, factors, ent_x, xtype = e42.build(D, L)
used = {x["ent"] for x in json.load(open(os.path.join(HERE, "..", "e6", "e6_items.json")))}
want = {"G": 86, "H": 86, "E": 43, "T": 85}  # 700個（200/200/100/200）と同じ割合
rng = np.random.default_rng(8008)
pool = {t: [] for t in want}
for node, ci in enumerate(var):
    c = allc[ci]
    if c["kind"] != "x" or c["ent"] in used:
        continue
    pool[xtype[node]].append(node)
out = []
for t, k in want.items():
    pick = rng.choice(pool[t], size=k, replace=False)
    for node in sorted(pick):
        c = allc[var[node]]
        out.append({"node": int(node), "type": t, "ent": c["ent"], "syn": c["cat"], "catw": D["cats"][c["cat"]],
                    "stmt": c["stmt"], "logit": float(L[var[node]]), "gold": bool(c["gold"])})
rng.shuffle(out)
for i, x in enumerate(out):
    x["id"] = 1000 + i
json.dump(out, open(os.path.join(HERE, "e8_new_items.json"), "w"), ensure_ascii=False, indent=0)
print({t: len(p) for t, p in pool.items()}, len(out), "語", len({x['ent'] for x in out}))
