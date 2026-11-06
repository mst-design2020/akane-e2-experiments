# ===== 病気のテストを作る（Human Disease Ontology 2026-10-29版） =====
import os, re, json, itertools, collections, urllib.request
import numpy as np
try: from wordfreq import zipf_frequency
except ImportError:
    os.system("pip -q install wordfreq"); from wordfreq import zipf_frequency
DOID_URL = "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/v2026-10-29/src/ontology/doid.obo"
DOID_FILE = os.environ.get("DOID_FILE", "doid_2026-10-29.obo")
if not os.path.exists(DOID_FILE): urllib.request.urlretrieve(DOID_URL, DOID_FILE)
TERMS = {}; cur = None
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
assert TERMS["DOID:4"]["name"] == "disease", "分類表の読み込みに失敗"
_anc = {}
def ancestors(k):
    if k not in _anc:
        s = set()
        for p in TERMS[k]["isa"]:
            if p in TERMS: s |= {p} | ancestors(p)
        _anc[k] = s
    return _anc[k]
import sys; sys.setrecursionlimit(10000)
# ラベル22：(分類表のid, 英語の名前（冠詞なし）, 冠詞)
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
SUB = ["bact", "viral", "fungal", "parasit"]            # 感染症の種類（どれも感染症に入る）
TOP = [l for l in LABELS if l not in SUB]
LID = {l: v[0] for l, v in LABELS.items()}
lab_of = lambda k: [l for l in LABELS if LID[l] in ancestors(k)]
# ルールの正解：分類表の全部の病気のどれかが両方を持てば「両立する」
both = collections.Counter()
for k in TERMS:
    L = set(lab_of(k)) | {l for l in LABELS if LID[l] == k}
    for p, q in itertools.combinations(sorted(L), 2): both[(p, q)] += 1
def truth_excl(p, q): return both[tuple(sorted((p, q)))] == 0      # True = 両立しない
# 病気：3語以内・よく使われる名前・数字なし・ラベルそのものではない
label_ids = set(LID.values())
cand = sorted(k for k, v in TERMS.items()
              if k not in label_ids and len(v["name"].split()) <= 3 and not re.search(r"\d", v["name"])
              and v["name"].isascii() and zipf_frequency(v["name"], "en") >= 3.0 and lab_of(k))
names = collections.Counter(TERMS[k]["name"].lower() for k in cand)
cand = [k for k in cand if names[TERMS[k]["name"].lower()] == 1]
rng = np.random.default_rng(DATA_SEED)
DIS = sorted(rng.choice(cand, min(600, len(cand)), replace=False).tolist())
STMT, GOLD, KIND, CITY = [], [], [], []
idx = {}
def add(key, text, gold, kind, city=None):
    if key in idx: return idx[key]
    idx[key] = len(STMT); STMT.append(text); GOLD.append(gold); KIND.append(kind); CITY.append(city)
    return idx[key]
nm = lambda l: LABELS[l][1]; an = lambda l: LABELS[l][2] + " " + LABELS[l][1]
# 種類→大きな分類：「ウイルス感染症は感染症の一種」（本物1つ＋偽3つ）
for s in SUB:
    fz = list(rng.choice([z for z in TOP if z != "infect"], 3, replace=False))
    for z in ["infect"] + fz:
        add(("CK", s, z), f"{an(s)[0].upper() + an(s)[1:]} is a kind of {nm(z)}.", int(z == "infect"), "種類→大分類")
# 病気→ラベル：本物のラベル全部＋偽のラベル4つ
for k in DIS:
    name = TERMS[k]["name"]; G = set(lab_of(k))
    false = [l for l in LABELS if l not in G]
    ask = sorted(G) + list(rng.choice(false, min(4, len(false)), replace=False))
    for l in ask:
        add(("CL", name, l), f"{name[0].upper() + name[1:]} is {an(l)}.", int(l in G), "病気→種類" if l in SUB else "病気→分類", name)
GOLD = np.array(GOLD)
DUMMY = len(STMT)
# つながりのルール（人が書いたもの、E2-3と同じ形）：病気がウイルス感染症で、ウイルス感染症が感染症の一種 → 病気は感染症
RULES = set()
for (t, a, b), i in list(idx.items()):
    if t == "CL" and b in SUB:
        for z in LABELS:
            if ("CK", b, z) in idx and ("CL", a, z) in idx:
                RULES.add((i, idx[("CK", b, z)], idx[("CL", a, z)]))
TRANS = sorted(RULES)
# 両立しないかどうかを確かめるラベルの組（同じ病気の問題に一緒に出てくる組）と、その組が当てはまる問題の対
INST = collections.defaultdict(list)
for name in set(c for c in CITY if c):
    L = [l for l in LABELS if ("CL", name, l) in idx]
    for p, q in itertools.combinations(sorted(L), 2): INST[(p, q)].append((idx[("CL", name, p)], idx[("CL", name, q)]))
for s in SUB:
    L = [z for z in LABELS if ("CK", s, z) in idx]
    for p, q in itertools.combinations(sorted(L), 2): INST[(p, q)].append((idx[("CK", s, p)], idx[("CK", s, q)]))
PAIRS = sorted(INST)
TRUTH = {pq: truth_excl(*pq) for pq in PAIRS}
G0 = np.append(GOLD, 0)
bad_t = sum(1 for p, q, h in TRANS if G0[p] == 1 and G0[q] == 1 and G0[h] == 0)
bad_e = sum(1 for pq in PAIRS if TRUTH[pq] for i, j in INST[pq] if G0[i] == 1 and G0[j] == 1)
print(f"病気 {len(DIS)}（候補 {len(cand)}）、問題 {len(STMT)}（本物 {GOLD.sum()}）、つながりのルール {len(TRANS)}本、"
      f"ラベルの組 {len(PAIRS)}（両立しない {sum(TRUTH.values())}）、正解が破るルール {bad_t}+{bad_e}")
print("例:", " | ".join(STMT[i] for i in [0, 1, 20, 21, 22, 23]))
