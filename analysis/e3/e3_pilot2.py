# E3 試し2回目（Colabのセル本体）。USE_MOCK=True なら作り物のAIで流れだけ確かめる。
# 結果: {OUT}/E3_pilot.json
#   items[i] = {id, claim, label, hops, sents[資料の文], verdict{p_true}, decomp{A:[部分], B:{hidden:{#1:説明}, facts:[...] , cands:{#1:[候補]}}},
#               nli{claim:[[含,中,矛] 文ごと], A:[[...]], B:{"事実の番号|候補の組": [[...]]}}}
import os, re, json, time, random, itertools, collections
import numpy as np

SRC_URL = "https://raw.githubusercontent.com/teacherpeterpan/ProgramFC/main/datasets/HOVER/claims/dev.json"
NAMES = ["entail", "neutral", "contradict"]

def load_hover(path):
    if not os.path.exists(path):
        import urllib.request
        urllib.request.urlretrieve(SRC_URL, path)
    return json.load(open(path))

def pick_pilot(data, per_group=8, seed=0):
    r = random.Random(seed); g = collections.defaultdict(list)
    for e in data: g[(e["num_hops"], e["label"])].append(e)
    out = []
    for k in sorted(g): out += r.sample(g[k], per_group)
    return out

def split_sents(evidence):
    out = []
    for para in evidence.split("\n"):
        para = para.strip()
        if not para: continue
        out += [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=["A-Z0-9])', para) if len(s.strip()) > 3]
    return out

# ---------- Gemmaへの聞き方 ----------
VERDICT_T = ("Evidence:\n{ev}\n\nClaim: {claim}\n\n"
             "Based only on the evidence above, is the claim true or false? Answer with one word: True or False.")

DECOMP_A = """Break the claim into 2 to 4 simple facts. Each fact must be a full sentence that can be checked on its own. Do not add any information that is not in the claim, and do not leave out any part of the claim. Write one fact per line, starting with "- ".

Claim: The author of the novel Kafka on the Shore was born in Kyoto.
- Kafka on the Shore is a novel.
- The author of Kafka on the Shore was born in Kyoto.

Claim: The band that released the album Blue Lines was formed in the same city where the footballer Gary Lineker was born.
- A band released the album Blue Lines.
- The band that released Blue Lines was formed in a city.
- Gary Lineker is a footballer.
- Gary Lineker was born in the city where the band that released Blue Lines was formed.

Claim: {claim}
"""

DECOMP_B = """Rewrite the claim as hidden things and simple facts. A hidden thing is a person, place or thing that the claim describes but does not name. Name hidden things #1, #2, ... and describe each one in a line like "#1 = ...". Use #1, #2 only for things the claim does not name. Then write 2 to 4 simple facts, one per line starting with "- ", using #1, #2 where needed. Do not add any information that is not in the claim, and do not leave out any part of the claim.

Claim: The author of the novel Kafka on the Shore was born in Kyoto.
#1 = the author of the novel Kafka on the Shore
- #1 wrote the novel Kafka on the Shore.
- #1 was born in Kyoto.

Claim: The band that released the album Blue Lines was formed in the same city where the footballer Gary Lineker was born.
#1 = the band that released the album Blue Lines
#2 = the city where #1 was formed
- #1 released the album Blue Lines.
- #1 was formed in #2.
- Gary Lineker was born in #2.

Claim: {claim}
"""

FILL_T = ("Evidence:\n{ev}\n\nUsing only the evidence above, who or what is {desc}? "
          "Answer with its proper name only (the name of the person, place, group or work), not a description. "
          "If the evidence does not say, answer: unknown")
SHOT_WORDS = ["Kafka", "Blue Lines", "Lineker", "Murakami", "Kyoto", "city where #1 was formed"]
UNKNOWN_PAT = re.compile(r"unknown|not (possible|mention|say|provide|tell|relevant)|cannot|can't|does not|doesn't|no mention|impossible|trick question", re.I)

def _leak(line, claim): return any(w in line and w not in claim for w in SHOT_WORDS)
def parse_A(text, claim=""):
    return [l.strip()[2:].strip() for l in text.splitlines() if l.strip().startswith("- ") and len(l.strip()) > 4 and not _leak(l, claim)][:4]

def parse_B(text, claim=""):
    hidden, facts = {}, []
    for l in text.splitlines():
        l = l.strip()
        if _leak(l, claim): continue
        m = re.match(r'^(#\d)\s*=\s*(.+)$', l)
        if m and m.group(1) not in hidden: hidden[m.group(1)] = m.group(2).strip()
        elif l.startswith("- ") and len(l) > 4: facts.append(l[2:].strip())
    facts = [f for f in facts if all(k in hidden for k in re.findall(r'#\d', f))][:4]   # 決めていない #2 などを使う行は捨てる
    used = {k for k in hidden if any(k in f for f in facts)}
    hidden = {k: v for k, v in hidden.items() if k in used}   # 部分の主張で使わない隠れた点は捨てる
    return hidden, facts

def fill(s, assign):
    for k, v in sorted(assign.items(), key=lambda kv: -len(kv[0])): s = s.replace(k, v)
    return s

def clean_name(t):
    t = t.strip().split("\n")[0].strip().strip('."\'*').strip()
    t = re.sub(r'^(the answer is|answer:)\s*', '', t, flags=re.I)
    return "" if UNKNOWN_PAT.search(t) else t[:80]

# ---------- 本体（試し2回目：段階ごと） ----------
def prep(items):
    for it in items:
        it["sents"] = split_sents(it["evidence"])
        it["paras"] = [p.strip() for p in it["evidence"].split("\n") if p.strip()]
    return items

def ev_text(it): return "\n".join(it["sents"])

def do_verdict(items, p_true, log=print):
    pt = p_true([VERDICT_T.format(ev=ev_text(it), claim=it["claim"]) for it in items])
    for it, p in zip(items, pt): it["verdict"] = {"p_true": float(p)}
    log("答えと自信 済み")

def do_decomp(items, gen, tag, log=print, n_samples=3, max_cands=3, max_combos=9):
    """tag: '2b' や '9b'。it['decomp'][tag] に A・B・候補を入れる"""
    ta = gen([DECOMP_A.format(claim=it["claim"]) for it in items])
    tb = gen([DECOMP_B.format(claim=it["claim"]) for it in items])
    for it, a, b in zip(items, ta, tb):
        h, f = parse_B(b, it["claim"])
        it.setdefault("decomp", {})[tag] = {"A_raw": a, "A": parse_A(a, it["claim"]), "B_raw": b,
                                            "B": {"hidden": h, "facts": f, "cands": {}, "unknown": {}}}
    log(f"[{tag}] 分ける 済み")
    for it in items:
        B = it["decomp"][tag]["B"]; assigns = [{}]
        for k in sorted(B["hidden"]):
            new = []
            for a in assigns:
                desc = fill(B["hidden"][k], a)
                pr = [FILL_T.format(ev=ev_text(it), desc=desc)]
                outs = gen(pr, max_new=24) + gen(pr * n_samples, sample=True, max_new=24)
                c, n_unk = [], 0
                for o in outs:
                    o2 = clean_name(o)
                    if not o2: n_unk += 1
                    elif o2 not in c: c.append(o2)
                key = json.dumps(a, ensure_ascii=False)
                B["cands"].setdefault(k, {})[key] = c[:max_cands]
                B["unknown"].setdefault(k, {})[key] = n_unk / len(outs)
                new += [dict(a, **{k: x}) for x in c[:max_cands]]
            assigns = new[:max_combos] or [{}]
        B["assigns"] = assigns
    log(f"[{tag}] 隠れた点の候補 済み")

def do_nli(items, nli, log=print):
    """前提は 文（sents）と 段落（paras）の両方。仮説は 元の主張・各分け方のA・Bに候補を入れたもの"""
    P, H, where = [], [], []
    for i, it in enumerate(items):
        hyps = [("claim", None, it["claim"])]
        for tag, d in it["decomp"].items():
            hyps += [(f"{tag}A", str(fi), f) for fi, f in enumerate(d["A"])]
            for ai, a in enumerate(d["B"]["assigns"]):
                hyps += [(f"{tag}B", f"{fi}|{ai}", fill(f, a)) for fi, f in enumerate(d["B"]["facts"])]
        for unit in ("sents", "paras"):
            for j, s in enumerate(it[unit]):
                for kind, key, h in hyps:
                    P.append(s); H.append(h); where.append((i, unit, kind, key, j))
    log(f"NLIの組 {len(P)}")
    out = nli(P, H)
    for it in items: it["nli"] = {"sents": {}, "paras": {}}
    for (i, unit, kind, key, j), v in zip(where, out):
        d = items[i]["nli"][unit].setdefault(kind, {})
        d.setdefault(key or "_", [None] * len(items[i][unit]))[j] = v
    log("NLI 済み")

# ---------- 作り物のAI（流れの確かめ用。研究には使わない） ----------
def mock_models(seed=0):
    r = np.random.default_rng(seed)
    def gen(prompts, sample=False, max_new=160, **kw):
        out = []
        for p in prompts:
            if p.startswith("Break"):   out.append("- Fact one.\n- Fact two.")
            elif p.startswith("Rewrite"): out.append("#1 = the thing in the claim\n- #1 did something.\n- #1 is somewhere.")
            else: out.append(r.choice(["Alice", "Bob", "Carol"]))
        return out
    def p_true(prompts, **kw): return r.random(len(prompts)).tolist()
    def nli(P, H, **kw): return r.dirichlet([1, 1, 1], len(P)).tolist()
    return gen, p_true, nli
