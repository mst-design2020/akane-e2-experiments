"""E16-0 の問題を作る（手元）。python3 e160_items.py ../../e10/e103/k_items.json ../../e4/e42_data.json e160_items.json
物と分類は E10-3 の K（501語×46分類、WordNetの正解）。手書きの語の一覧は使わない（分類の木と正解の表だけから作る）。"""
import json, sys, random
K = json.load(open(sys.argv[1])); E = json.load(open(sys.argv[2]))
CN = K["cats"]; ANC = {c: set(E["parent"][c]) for c in CN}          # ANC[c]＝c の先祖の全部
T = {}
for it in K["items"]: T.setdefault(it["ent"], {})[it["cat"]] = it["gold"]
ents = sorted(T); cats = list(CN)
def direct(c):                                                       # すぐ上の親（先祖のうち、他の先祖の先祖でない物）
    return {a for a in ANC[c] if not any(a in ANC[b] for b in ANC[c] if b != a)}
DP = {c: direct(c) for c in cats}
def sibs(c):                                                         # 同じ親を持つ分類（根どうしも兄弟とみる）、先祖・子孫は除く
    return [d for d in cats if d != c and DP[d] == DP[c] and c not in ANC[d] and d not in ANC[c]]
def specific(x):                                                     # x が当てはまる一番細かい分類
    S = [c for c in cats if T[x][c]]
    return [c for c in S if not any(c in ANC[d] for d in S if d != c)]
TGT = {x: specific(x) for x in ents}
rng = random.Random(0)
def pick(pool, k, fill):
    p = list(dict.fromkeys(pool)); rng.shuffle(p); got = p[:k]
    f = [z for z in dict.fromkeys(fill) if z not in got]; rng.shuffle(f)
    return got + f[:k - len(got)], min(len(p), k)
items = []
for x in ents:
    for Y in TGT[x]:
        nf = [c for c in cats if not T[x][c] and Y not in ANC[c]]  # x が当てはまらない分類（Y の子孫は外す）
        ne = [z for z in ents if z != x and not T[z][Y]]           # Y に当てはまらない物
        sy = set(sibs(Y))
        it = dict(ent=x, cat=Y, cname=CN[Y])
        it["f_easy"], _ = pick(nf, 3, nf)
        it["f_hard"], it["f_hard_n"] = pick([c for c in nf if c in sy], 3, nf)
        it["r_easy"], _ = pick(ne, 3, ne)
        k = it["f_hard_n"]                                         # 兄弟の数を前向きとそろえる
        sib_e, _ = pick([z for z in ne if set(TGT[z]) & sy], k, [])
        it["r_hard"], _ = pick(sib_e, 3, ne); it["r_hard_n"] = len(sib_e)
        items.append(it)
json.dump(dict(note="E16-0：一種かを前向き（物→分類）と逆向き（分類→物）の4択で聞く。WordNetの正解（E10-3 K）", cats=CN, ents=ents, items=items), open(sys.argv[3], "w"), ensure_ascii=False)
print("組", len(items), " 物", len(ents), " 分類の種類", len({i['cat'] for i in items}))
print("兄弟で3つそろった割合  前向き", round(sum(i['f_hard_n'] == 3 for i in items) / len(items), 3), " 逆向き", round(sum(i['r_hard_n'] == 3 for i in items) / len(items), 3))
print(items[0])
