"""E4-1 データ作り：WordNet の物と分類から、Gemmaに聞く はい/いいえ の主張と、網のルールを作る。
使い方: NLTK_DATA=<wordnetの場所> python3 -I e41_data.py  → e41_data.json
"""
import json, random, itertools, collections
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

SEED = 0
N_ENT = 1500
MAX_TRUE, N_FALSE = 4, 3
# 分類：WordNetの見出し → 英語の言い方
CATS = {
    "organism.n.01": "living thing", "animal.n.01": "animal", "mammal.n.01": "mammal", "bird.n.01": "bird", "fish.n.01": "fish",
    "insect.n.01": "insect", "reptile.n.01": "reptile", "dog.n.01": "dog", "plant.n.02": "plant",
    "tree.n.01": "tree", "flower.n.01": "flower", "edible_fruit.n.01": "fruit", "vegetable.n.01": "vegetable",
    "food.n.01": "food", "dish.n.02": "dish", "beverage.n.01": "drink",
    "artifact.n.01": "man-made object", "tool.n.01": "tool", "device.n.01": "device", "machine.n.01": "machine",
    "equipment.n.01": "equipment", "vehicle.n.01": "vehicle", "car.n.01": "car", "boat.n.01": "boat",
    "aircraft.n.01": "aircraft", "weapon.n.01": "weapon", "container.n.01": "container",
    "furniture.n.01": "furniture", "clothing.n.01": "clothing", "footwear.n.02": "footwear",
    "plaything.n.01": "toy", "musical_instrument.n.01": "musical instrument",
    "building.n.01": "building", "room.n.01": "room",
    "person.n.01": "person", "worker.n.01": "worker", "artist.n.01": "artist", "athlete.n.01": "athlete",
    "material.n.01": "material", "fabric.n.01": "fabric", "metallic_element.n.01": "metal",
    "body_part.n.01": "body part", "organ.n.01": "organ", "disease.n.01": "disease",
    "game.n.01": "game", "sport.n.01": "sport",
}
MASS = {"food", "equipment", "furniture", "clothing", "footwear", "material", "fabric", "metal"}
art = lambda w: w if w in MASS else ("an " if w[0] in "aeiou" else "a ") + w
cap = lambda s: s[0].upper() + s[1:]

up = lambda s: set(s.closure(lambda z: z.hypernyms())) | {s}
syn = {c: wn.synset(c) for c in CATS}
anc = {c: up(syn[c]) for c in CATS}
desc = {c: set(syn[c].closure(lambda z: z.hyponyms())) | {syn[c]} for c in CATS}
isa = lambda a, b: a != b and syn[b] in anc[a]                    # 分類aは分類bの一種
disjoint = lambda a, b: not isa(a, b) and not isa(b, a) and not (desc[a] & desc[b])

ents = []
for s in wn.all_synsets("n"):
    w = s.lemmas()[0].name()
    if not w.isalpha() or not w.islower() or len(wn.synsets(w, "n")) != 1 or zipf_frequency(w, "en") < 2.5:
        continue
    a = up(s) - {s}
    tc = [c for c in CATS if syn[c] in a]
    if len(tc) >= 2:
        ents.append((w, tc))
rng = random.Random(SEED)
rng.shuffle(ents)
ents = sorted(ents[:N_ENT])

claims, key2id = [], {}
def claim(key, stmt, gold, kind, **kw):
    if key not in key2id:
        key2id[key] = len(claims)
        claims.append(dict(id=len(claims), kind=kind, stmt=stmt, gold=bool(gold), **kw))
    return key2id[key]

factors = []   # (型, [主張id...])  trans: a∧c⇒b ／ disj: a∧b∧d は不可 ／ anti: c_ab∧c_ba は不可 ／ cd: c∧d は不可
for w, tc in ents:
    tc = rng.sample(tc, min(MAX_TRUE, len(tc)))
    false = [c for c in CATS if all(disjoint(c, t) for t in tc)]
    cand = tc + rng.sample(false, min(N_FALSE, len(false)))
    for c in cand:
        claim(("x", w, c), f"{cap(w)} is a type of {CATS[c]}.", c in tc, "x", ent=w, cat=c)
    for a, b in itertools.permutations(cand, 2):
        ci = claim(("c", a, b), f"{cap(CATS[a])} is a type of {CATS[b]}.", isa(a, b), "c", cat=a, cat2=b)
        factors.append(("trans", [key2id[("x", w, a)], ci, key2id[("x", w, b)]]))
    for a, b in itertools.combinations(sorted(cand), 2):
        if isa(a, b) or isa(b, a):
            continue
        di = claim(("d", a, b), f"Nothing can be both {art(CATS[a])} and {art(CATS[b])}.", disjoint(a, b), "d", cat=a, cat2=b)
        factors.append(("disj", [key2id[("x", w, a)], key2id[("x", w, b)], di]))
# 分類どうしのルール（1回ずつ）
seen = set()
for (k, a, b), i in list(key2id.items()):
    if k == "c" and ("c", b, a) in key2id and (b, a) not in seen:
        seen.add((a, b)); factors.append(("anti", [i, key2id[("c", b, a)]]))
    if k == "d":
        for kk in [("c", a, b), ("c", b, a)]:
            if kk in key2id:
                factors.append(("cd", [key2id[kk], i]))
ent_of = {w: [key2id[("x", w, c)] for c in CATS if ("x", w, c) in key2id] for w, _ in ents}
# 正解が破るルールの数（0のはず）
g = [c["gold"] for c in claims]
viol = sum((t == "trans" and g[v[0]] and g[v[1]] and not g[v[2]]) or (t == "disj" and all(g[x] for x in v)) or
           (t in ("anti", "cd") and all(g[x] for x in v)) for t, v in factors)
cnt = collections.Counter(c["kind"] for c in claims)
gold_rate = {k: sum(c["gold"] for c in claims if c["kind"] == k) / n for k, n in cnt.items()}
print("物", len(ents), "主張", len(claims), dict(cnt), "本当の割合", {k: round(v, 3) for k, v in gold_rate.items()})
print("ルール", len(factors), dict(collections.Counter(t for t, _ in factors)), "正解が破るルール", viol)
json.dump(dict(cats=CATS, ents=[w for w, _ in ents], ent_claims=ent_of, claims=claims, factors=factors),
          open("e41_data.json", "w"), ensure_ascii=False)
