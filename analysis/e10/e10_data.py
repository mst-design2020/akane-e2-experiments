"""E10 の問題を作る（3つの場面、全部の組）。python3 e10_data.py OUT_DIR
P 部分と全体：WordNet 3.0 の部品型の「一部」（part meronym）。全体30（先に固定）。
D 病気：Human Disease Ontology（2026-10-29版）、E2-8と同じ病気600（DATA_SEED=0）×分類22。
G 地理：geonamescache の市・国 × 国連の地域分け22（M49、lukes/ISO-3166-Countries-with-Regional-Codes）。
出力：e10_items.json（正解入り、こちら用）と e10_items_nogold.json（Colab用、正解なし）。
"""
import os, re, sys, csv, json, collections, itertools, urllib.request
import numpy as np
from wordfreq import zipf_frequency as zf

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_SEED = 0
cap = lambda s: s[:1].upper() + s[1:]
art = lambda w: ("an " if w[:1].lower() in "aeiou" else "a ") + w

# ---------------- P：部分と全体 ----------------
from nltk.corpus import wordnet as wn
WHOLES = {  # 語 → (WordNetの意味, 文での言い方)
    "body": ("body.n.01", "a human body"), "head": ("head.n.01", "a head"), "face": ("face.n.01", "a face"),
    "hand": ("hand.n.01", "a hand"), "foot": ("foot.n.01", "a foot"), "eye": ("eye.n.01", "an eye"),
    "mouth": ("mouth.n.01", "a mouth"), "arm": ("arm.n.01", "an arm"), "leg": ("leg.n.01", "a leg"),
    "tooth": ("tooth.n.01", "a tooth"), "bird": ("bird.n.01", "a bird"), "fish": ("fish.n.01", "a fish"),
    "horse": ("horse.n.01", "a horse"), "tree": ("tree.n.01", "a tree"), "flower": ("flower.n.01", "a flower"),
    "car": ("car.n.01", "a car"), "bicycle": ("bicycle.n.01", "a bicycle"), "airplane": ("airplane.n.01", "an airplane"),
    "ship": ("ship.n.01", "a ship"), "boat": ("boat.n.01", "a boat"), "house": ("house.n.01", "a house"),
    "church": ("church.n.02", "a church"), "computer": ("computer.n.01", "a computer"), "watch": ("watch.n.01", "a watch"),
    "piano": ("piano.n.01", "a piano"), "gun": ("gun.n.01", "a gun"), "shoe": ("shoe.n.01", "a shoe"),
    "shirt": ("shirt.n.01", "a shirt"), "camera": ("camera.n.01", "a camera"), "knife": ("knife.n.01", "a knife")}
PART_LEX = ("noun.artifact", "noun.body", "noun.plant", "noun.animal", "noun.object", "noun.substance")
GENERIC = {"part", "portion", "section", "segment", "piece", "component", "corpus", "factor", "subdivision"}


def parts_closure(s):
    """全体 s とその上の分類に付いた部品、その部品の部品…（部品型だけ）"""
    out, stack = set(), []
    for h in [s] + list(s.closure(lambda x: x.hypernyms())):
        stack += h.part_meronyms()
    while stack:
        p = stack.pop()
        if p in out:
            continue
        out.add(p); stack += p.part_meronyms()
    return {p for p in out if p.lexname() in PART_LEX}


def ok_word(n):
    return "_" not in n and re.fullmatch(r"[a-z][a-z\-]*", n) and zf(n, "en") >= 3.0 and n not in GENERIC


PC = {w: parts_closure(wn.synset(s)) for w, (s, _) in WHOLES.items()}
part_words = set()
for w, ps in PC.items():
    for p in ps:
        for l in p.lemmas():
            if ok_word(l.name()) and l.name() not in WHOLES:
                part_words.add(l.name())
rng = np.random.default_rng(DATA_SEED)
part_words = sorted(part_words)
P_PARTS = sorted(rng.choice(part_words, min(500, len(part_words)), replace=False).tolist())
# 部品でない物（どの全体の部品にもならない、よく使う物・体・植物の語）100
allp = set().union(*PC.values())
pool = sorted({l.name() for s in wn.all_synsets("n") if s.lexname() in ("noun.artifact", "noun.body", "noun.plant", "noun.animal", "noun.food")
               for l in s.lemmas() if ok_word(l.name()) and zf(l.name(), "en") >= 3.5}
              - set(part_words) - set(WHOLES))
pool = [w for w in pool if not any(s in allp for s in wn.synsets(w, "n"))]
P_OTHER = sorted(rng.choice(pool, 100, replace=False).tolist())
P_ENTS = sorted(set(P_PARTS) | set(P_OTHER))


def p_gold(x, W):
    return int(any(s in PC[W] for s in wn.synsets(x, "n")))


# ---------------- D：病気（E2-8 と同じ選び方） ----------------
DOID_FILE = os.path.join(OUT, "doid_2026-10-29.obo")
if not os.path.exists(DOID_FILE):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/v2026-10-29/src/ontology/doid.obo", DOID_FILE)
TERMS, cur = {}, None
for line in open(DOID_FILE, encoding="utf-8"):
    line = line.rstrip("\n")
    if line == "[Term]": cur = {"isa": [], "obs": False}; continue
    if line.startswith("["): cur = None; continue
    if cur is None: continue
    if line.startswith("id: "): cur["id"] = line[4:]; TERMS[cur["id"]] = cur
    elif line.startswith("name: "): cur["name"] = line[6:]
    elif line.startswith("is_a: "): cur["isa"].append(line[6:].split(" ")[0])
    elif line.startswith("is_obsolete: true"): cur["obs"] = True
TERMS = {k: v for k, v in TERMS.items() if not v["obs"]}
sys.setrecursionlimit(10000)
_anc = {}
def ancestors(k):
    if k not in _anc:
        s = set()
        for p in TERMS[k]["isa"]:
            if p in TERMS: s |= {p} | ancestors(p)
        _anc[k] = s
    return _anc[k]
LABELS = {
 "cardio": ("DOID:1287", "cardiovascular system disease", "a", "cardiovascular system"),
 "resp": ("DOID:1579", "respiratory system disease", "a", "respiratory system"),
 "gi": ("DOID:77", "gastrointestinal system disease", "a", "digestive system"),
 "nerv": ("DOID:863", "nervous system disease", "a", "nervous system"),
 "skin": ("DOID:16", "skin disease", "a", "skin"),
 "musc": ("DOID:17", "musculoskeletal system disease", "a", "musculoskeletal system"),
 "urin": ("DOID:18", "urinary system disease", "a", "urinary system"),
 "endo": ("DOID:28", "endocrine system disease", "an", "endocrine system"),
 "immune": ("DOID:2914", "immune system disease", "an", "immune system"),
 "blood": ("DOID:74", "blood disease", "a", "blood"),
 "repro": ("DOID:15", "reproductive system disease", "a", "reproductive system"),
 "breast": ("DOID:3463", "breast disease", "a", "breast"),
 "chest": ("DOID:0060118", "thoracic disease", "a", "chest"),
 "metab": ("DOID:0014667", "metabolic disease", "a", None), "infect": ("DOID:0050117", "infectious disease", "an", None),
 "tumor": ("DOID:14566", "cancer or other tumor", "a", None), "mental": ("DOID:150", "mental disorder", "a", None),
 "genetic": ("DOID:630", "genetic disease", "a", None), "bact": ("DOID:104", "bacterial infection", "a", None),
 "viral": ("DOID:934", "viral infection", "a", None), "fungal": ("DOID:1564", "fungal infection", "a", None),
 "parasit": ("DOID:1398", "parasitic infection", "a", None)}
LID = {l: v[0] for l, v in LABELS.items()}
lab_of = lambda k: [l for l in LABELS if LID[l] in ancestors(k)]
label_ids = set(LID.values())
cand = sorted(k for k, v in TERMS.items()
              if k not in label_ids and len(v["name"].split()) <= 3 and not re.search(r"\d", v["name"])
              and v["name"].isascii() and zf(v["name"], "en") >= 3.0 and lab_of(k))
names = collections.Counter(TERMS[k]["name"].lower() for k in cand)
cand = [k for k in cand if names[TERMS[k]["name"].lower()] == 1]
rng_d = np.random.default_rng(DATA_SEED)
DIS = sorted(rng_d.choice(cand, min(600, len(cand)), replace=False).tolist())

# ---------------- G：地理 ----------------
import geonamescache
M49_FILE = os.path.join(OUT, "m49_all.csv")
if not os.path.exists(M49_FILE):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv", M49_FILE)
REG = {}
for r in csv.DictReader(open(M49_FILE, encoding="utf-8")):
    g = r["intermediate-region"] or r["sub-region"]
    if g: REG[r["alpha-2"]] = g
REGION_TEXT = {g: ("South-Eastern Asia" if g == "South-eastern Asia" else g) for g in set(REG.values())}
REGIONS = sorted(REGION_TEXT)
gc = geonamescache.GeonamesCache()
SKIP = {"RU", "TR", "EG", "KZ", "AZ", "GE", "AM", "CY", "ID", "PG", "TL"}  # 地域をまたぐ国（E2-2と同じ）
cos = {k: v for k, v in gc.get_countries().items() if k in REG and k not in SKIP}
allc = list(gc.get_cities().values())
name_cnt = collections.Counter(c["name"] for c in allc)
cnames = {v["name"] for v in cos.values()}
by_co = collections.defaultdict(list)
for c in allc:
    if c["countrycode"] in cos and name_cnt[c["name"]] == 1 and c["name"] not in cnames and c["name"].isascii() \
            and re.fullmatch(r"[A-Z][A-Za-z\-' ]+", c["name"]):
        by_co[c["countrycode"]].append(c)
COUNTRIES = [k for k in sorted(by_co) if len(by_co[k]) >= 4 and re.fullmatch(r"[A-Z][A-Za-z\-' ]+", cos[k]["name"])]
rng_g = np.random.default_rng(DATA_SEED)
G_ENTS = []  # (名前, 種類, 国コード, 人口)
for k in COUNTRIES:
    G_ENTS.append((cos[k]["name"], "country", k, int(cos[k].get("population") or 0)))
    cs = sorted(by_co[k], key=lambda c: -c["population"])
    pick = [cs[0]] + list(rng_g.choice(cs[1:], 3, replace=False))
    for c in pick:
        G_ENTS.append((c["name"], "city", k, int(c["population"])))

# ---------------- まとめ ----------------
items = []
def add(dom, ent, cls, stmt, gold, rev=None, extra=None):
    d = {"id": len(items), "dom": dom, "ent": ent, "cls": cls, "stmt": stmt, "gold": int(gold)}
    if rev: d["rev"] = rev
    if extra: d.update(extra)
    items.append(d)
for x in P_ENTS:
    for W, (_, ph) in WHOLES.items():
        add("P", x, W, f"{cap(x)} is a part of {ph}.", p_gold(x, W), rev=f"{cap(ph.split(' ', 1)[1])} is a part of {art(x)}.",
            extra={"other": int(x in P_OTHER)})
for k in DIS:
    name = TERMS[k]["name"]; G = set(lab_of(k))
    for l, (_, lname, a, _) in LABELS.items():
        add("D", name, l, f"{cap(name)} is {a} {lname}.", l in G, extra={"doid": k})
for name, kind, co, pop in G_ENTS:
    for g in REGIONS:
        add("G", name, g, f"{name} is located in {REGION_TEXT[g]}.", REG[co] == g, extra={"kind": kind, "co": co, "pop": pop})
meta = {"wholes": {w: ph for w, (_, ph) in WHOLES.items()},
        "labels": {l: {"name": v[1], "art": v[2], "system": v[3]} for l, v in LABELS.items()},
        "regions": REGION_TEXT, "date": "2026-11-11", "seed": DATA_SEED}
json.dump({"meta": meta, "items": items}, open(os.path.join(OUT, "e10_items.json"), "w"), ensure_ascii=False)
ng = [{k: v for k, v in it.items() if k not in ("gold", "doid", "co", "other")} for it in items]
json.dump({"meta": meta, "items": ng}, open(os.path.join(OUT, "e10_items_nogold.json"), "w"), ensure_ascii=False)
for dom in "PDG":
    its = [it for it in items if it["dom"] == dom]
    print(dom, "物", len({it["ent"] for it in its}), "主張", len(its), "本当", sum(it["gold"] for it in its),
          f"({np.mean([it['gold'] for it in its]):.3f})")
print("P 部品の語候補", len(part_words), "部品でない物", len(P_OTHER), " D 候補", len(cand), " G 国", len(COUNTRIES), "地域", len(REGIONS))
