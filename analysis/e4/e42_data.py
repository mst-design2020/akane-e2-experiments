"""E4-2 データ作り（1段目）：Gemmaに聞く問いを全部作る。
物×全分類・分類どうしの一種・両立しない。候補選びは Gemma の答えが来てから e42.py で行う（PLAN_E4-2.md）。
使い方: NLTK_DATA=<wordnet> python3 -I e42_data.py → e42_data.json
"""
import json, itertools
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

exec(open(__file__.replace("e42_data.py", "e41_data.py")).read().split("ents = []")[0])  # CATS, syn, isa, disjoint など E4-1と同じ

ents = {}
for s in wn.all_synsets("n"):
    w = s.lemmas()[0].name()
    if not w.isalpha() or not w.islower() or len(wn.synsets(w, "n")) != 1 or zipf_frequency(w, "en") < 2.5:
        continue
    a = up(s) - {s}
    tc = [c for c in CATS if syn[c] in a]
    if len(tc) >= 2:
        ents[w] = tc
ents = dict(sorted(ents.items()))
parent = {c: [p for p in CATS if isa(c, p)] for c in CATS}
claims = []
for w, tc in ents.items():
    for c in CATS:
        claims.append(dict(kind="x", ent=w, cat=c, gold=c in tc, stmt=f"{cap(w)} is a type of {CATS[c]}."))
for a, b in itertools.permutations(CATS, 2):
    claims.append(dict(kind="c", cat=a, cat2=b, gold=isa(a, b), stmt=f"{cap(CATS[a])} is a type of {CATS[b]}."))
for a, b in itertools.combinations(sorted(CATS), 2):
    if not (isa(a, b) or isa(b, a)):
        claims.append(dict(kind="d", cat=a, cat2=b, gold=disjoint(a, b),
                           stmt=f"Nothing can be both {art(CATS[a])} and {art(CATS[b])}."))
dis = {f"{a}|{b}": disjoint(a, b) for a in CATS for b in CATS if a != b}
json.dump(dict(cats=CATS, parent=parent, ents=ents, disjoint=dis, claims=claims),
          open("e42_data.json", "w"), ensure_ascii=False)
import collections
print("物", len(ents), "問", len(claims), collections.Counter(c["kind"] for c in claims))
