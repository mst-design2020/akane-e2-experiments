"""E10-1 の問題を作る（E10で使っていない物だけ、手で選ぶ所なし）。NLTK_DATA=... python3 e101_data.py E10_ITEMS OUT_DIR
計画：sims/e10/PLAN_E10-1.md。作り方は e10_data.py と同じで、変えたのは次だけ（先に固定）。
- P 全体30：WordNetの名詞で、その語の一番目の意味が先頭の見出し、よく使う語（wordfreq 4以上）、名詞として使う語（SemCorの名詞の使用数が他の品詞以上）、
  部品の数10以上、物・体・植物などの名詞（PART_LEX）、E10の30個と重ならない物から、種（DATA_SEED=1）で30個。文での言い方は「a/an ＋語」。
  部分の語から外す語の一覧（E10の GENERIC）はやめた。部分の語はE10と重なってよい（組は全部新しい）。部品でない物はE10と重ならない物から。
- D：DOIDの同じ22の分類（分類名と冠詞は問いの定義）。E10と同じ決まりで、E10の600個以外の全部（177個しか残らない）。手で書いた部位名（E10の LABELS の4つ目）はやめ、
  ノートの中で分類名から文法で作る（e101_core.py の system_of）。
- G：同じ国で、E10で使っていない市を国ごとに4つ（国そのものは入れない）。
出力：e101_items.json（正解入り、こちら用）と e101_items_nogold.json（Colab用、正解なし）。
"""
import os, re, sys, csv, json, collections, urllib.request
import numpy as np
from wordfreq import zipf_frequency as zf

E10 = json.load(open(sys.argv[1]))
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
DATA_SEED = 1
cap = lambda s: s[:1].upper() + s[1:]
art = lambda w: ("an " if w[:1].lower() in "aeiou" else "a ") + w
old = collections.defaultdict(set)
for it in E10["items"]:
    old[it["dom"]].add(it["ent"])
OLD_WHOLES = set(E10["meta"]["wholes"])

# ---------------- P：部分と全体 ----------------
from nltk.corpus import wordnet as wn
PART_LEX = ("noun.artifact", "noun.body", "noun.plant", "noun.animal", "noun.object", "noun.substance")


def parts_closure(s):
    """全体 s とその上の分類に付いた部品、その部品の部品…（部品型だけ）。E10と同じ"""
    out, stack = set(), []
    for h in [s] + list(s.closure(lambda x: x.hypernyms())):
        stack += h.part_meronyms()
    while stack:
        p = stack.pop()
        if p in out:
            continue
        out.add(p); stack += p.part_meronyms()
    return {p for p in out if p.lexname() in PART_LEX}


cand_w = []
for s in wn.all_synsets("n"):
    w = s.lemmas()[0].name()
    if not re.fullmatch(r"[a-z]+", w) or w in OLD_WHOLES or zf(w, "en") < 4.0:
        continue
    if wn.synsets(w, "n")[0] != s or s.lexname() not in PART_LEX:
        continue
    # 名詞として使う語に限る：SemCorでの名詞の使用数が、他の品詞の使用数以上（simple＝薬草、electric＝電気自動車 などを外す）
    cnt = lambda pos_ok: sum(l.count() for x in wn.synsets(w) if pos_ok(x.pos()) for l in x.lemmas() if l.name() == w)
    if cnt(lambda p: p == "n") < cnt(lambda p: p != "n"):
        continue
    if len(parts_closure(s)) >= 10:
        cand_w.append((w, s.name()))
cand_w = sorted(set(cand_w))
rng = np.random.default_rng(DATA_SEED)
pick = sorted(rng.choice(len(cand_w), 30, replace=False).tolist())
WHOLES = {cand_w[i][0]: (cand_w[i][1], art(cand_w[i][0])) for i in pick}


def ok_word(n):
    return "_" not in n and re.fullmatch(r"[a-z][a-z\-]*", n) and zf(n, "en") >= 3.0


PC = {w: parts_closure(wn.synset(s)) for w, (s, _) in WHOLES.items()}
part_words = sorted({l.name() for ps in PC.values() for p in ps for l in p.lemmas()
                     if ok_word(l.name()) and l.name() not in WHOLES and l.name() not in OLD_WHOLES})
# 部分の語は E10 の語と重なってよい（全体がすべて新しいので、組はすべて新しい。重ならない語は26語しかなかった、11-11）
P_PARTS = sorted(rng.choice(part_words, min(500, len(part_words)), replace=False).tolist())
allp = set().union(*PC.values())
pool = sorted({l.name() for s in wn.all_synsets("n") if s.lexname() in ("noun.artifact", "noun.body", "noun.plant", "noun.animal", "noun.food")
               for l in s.lemmas() if ok_word(l.name()) and zf(l.name(), "en") >= 3.5}
              - set(part_words) - set(WHOLES) - old["P"] - OLD_WHOLES)
pool = [w for w in pool if not any(s in allp for s in wn.synsets(w, "n"))]
P_OTHER = sorted(rng.choice(pool, 100, replace=False).tolist())
P_ENTS = sorted(set(P_PARTS) | set(P_OTHER))


def p_gold(x, W):
    return int(any(s in PC[W] for s in wn.synsets(x, "n")))


# ---------------- D：病気（E10と同じ選び方、E10の600個を除く） ----------------
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
# 問いの定義（E2-8・E10と同じ22の分類と冠詞）。部位名は持たない
LABELS = {
 "cardio": ("DOID:1287", "cardiovascular system disease", "a"), "resp": ("DOID:1579", "respiratory system disease", "a"),
 "gi": ("DOID:77", "gastrointestinal system disease", "a"), "nerv": ("DOID:863", "nervous system disease", "a"),
 "skin": ("DOID:16", "skin disease", "a"), "musc": ("DOID:17", "musculoskeletal system disease", "a"),
 "urin": ("DOID:18", "urinary system disease", "a"), "endo": ("DOID:28", "endocrine system disease", "an"),
 "immune": ("DOID:2914", "immune system disease", "an"), "blood": ("DOID:74", "blood disease", "a"),
 "repro": ("DOID:15", "reproductive system disease", "a"), "breast": ("DOID:3463", "breast disease", "a"),
 "chest": ("DOID:0060118", "thoracic disease", "a"), "metab": ("DOID:0014667", "metabolic disease", "a"),
 "infect": ("DOID:0050117", "infectious disease", "an"), "tumor": ("DOID:14566", "cancer or other tumor", "a"),
 "mental": ("DOID:150", "mental disorder", "a"), "genetic": ("DOID:630", "genetic disease", "a"),
 "bact": ("DOID:104", "bacterial infection", "a"), "viral": ("DOID:934", "viral infection", "a"),
 "fungal": ("DOID:1564", "fungal infection", "a"), "parasit": ("DOID:1398", "parasitic infection", "a")}
LID = {l: v[0] for l, v in LABELS.items()}
lab_of = lambda k: [l for l in LABELS if LID[l] in ancestors(k)]
label_ids = set(LID.values())
cand = sorted(k for k, v in TERMS.items()
              if k not in label_ids and len(v["name"].split()) <= 3 and not re.search(r"\d", v["name"])
              and v["name"].isascii() and zf(v["name"], "en") >= 3.0 and lab_of(k))
names = collections.Counter(TERMS[k]["name"].lower() for k in cand)
cand = [k for k in cand if names[TERMS[k]["name"].lower()] == 1 and TERMS[k]["name"] not in old["D"]]
rng_d = np.random.default_rng(DATA_SEED)
DIS = sorted(rng_d.choice(cand, min(600, len(cand)), replace=False).tolist())

# ---------------- G：地理（同じ国、E10で使っていない市） ----------------
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
SKIP = {"RU", "TR", "EG", "KZ", "AZ", "GE", "AM", "CY", "ID", "PG", "TL"}  # 地域をまたぐ国（E2-2・E10と同じ、問題の作り方）
cos = {k: v for k, v in gc.get_countries().items() if k in REG and k not in SKIP}
allc = list(gc.get_cities().values())
name_cnt = collections.Counter(c["name"] for c in allc)
cnames = {v["name"] for v in cos.values()}
by_co = collections.defaultdict(list)
for c in allc:
    if c["countrycode"] in cos and name_cnt[c["name"]] == 1 and c["name"] not in cnames and c["name"].isascii() \
            and re.fullmatch(r"[A-Z][A-Za-z\-' ]+", c["name"]) and c["name"] not in old["G"]:
        by_co[c["countrycode"]].append(c)
COUNTRIES = [k for k in sorted(by_co) if len(by_co[k]) >= 4 and re.fullmatch(r"[A-Z][A-Za-z\-' ]+", cos[k]["name"])]
rng_g = np.random.default_rng(DATA_SEED)
G_ENTS = []
for k in COUNTRIES:
    cs = sorted(by_co[k], key=lambda c: -c["population"])
    for i in sorted(rng_g.choice(len(cs), 4, replace=False).tolist()):
        G_ENTS.append((cs[i]["name"], "city", k, int(cs[i]["population"])))

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
    for l, (_, lname, a) in LABELS.items():
        add("D", name, l, f"{cap(name)} is {a} {lname}.", l in G, extra={"doid": k})
for name, kind, co, pop in G_ENTS:
    for g in REGIONS:
        add("G", name, g, f"{name} is located in {REGION_TEXT[g]}.", REG[co] == g, extra={"kind": kind, "co": co, "pop": pop})
meta = {"wholes": {w: ph for w, (_, ph) in WHOLES.items()}, "whole_synsets": {w: s for w, (s, _) in WHOLES.items()},
        "labels": {l: {"name": v[1], "art": v[2]} for l, v in LABELS.items()},
        "regions": REGION_TEXT, "date": "2026-11-11", "seed": DATA_SEED}
json.dump({"meta": meta, "items": items}, open(os.path.join(OUT, "e101_items.json"), "w"), ensure_ascii=False)
ng = [{k: v for k, v in it.items() if k not in ("gold", "doid", "co", "other")} for it in items]
json.dump({"meta": meta, "items": ng}, open(os.path.join(OUT, "e101_items_nogold.json"), "w"), ensure_ascii=False)
for dom in "PDG":
    its = [it for it in items if it["dom"] == dom]
    print(dom, "物", len({it["ent"] for it in its}), "主張", len(its), "本当", sum(it["gold"] for it in its),
          f"({np.mean([it['gold'] for it in its]):.3f})")
print("全体の候補", len(cand_w), "選んだ全体", sorted(WHOLES))
print("P 部品の語候補", len(part_words), "部品でない物", len(P_OTHER), " D 候補", len(cand), " G 国", len(COUNTRIES), "地域", len(REGIONS))
