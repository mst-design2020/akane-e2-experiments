# E3 50件の試し（Colabのセル本体）。USE_MOCK=True なら作り物のAIで流れだけ確かめる。
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

DECOMP_A = """Break the claim into 2 to 4 simple facts. Each fact must be a full sentence that can be checked on its own. Do not add any information that is not in the claim. Write one fact per line, starting with "- ".

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

DECOMP_B = """Rewrite the claim as hidden things and simple facts. A hidden thing is a person, place or thing that the claim describes but does not name. Name hidden things #1, #2, ... and describe each one in a line like "#1 = ...". Then write 2 to 4 simple facts, one per line starting with "- ", using #1, #2 where needed. Do not add any information that is not in the claim.

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
          "Answer with the name only, in a few words.")

def parse_A(text):
    return [l[2:].strip() for l in text.splitlines() if l.strip().startswith("- ") and len(l.strip()) > 4][:4]

def parse_B(text):
    hidden, facts = {}, []
    for l in text.splitlines():
        l = l.strip()
        m = re.match(r'^(#\d)\s*=\s*(.+)$', l)
        if m and m.group(1) not in hidden: hidden[m.group(1)] = m.group(2).strip()
        elif l.startswith("- ") and len(l) > 4: facts.append(l[2:].strip())
    return hidden, facts[:4]

def fill(s, assign):
    for k, v in sorted(assign.items(), key=lambda kv: -len(kv[0])): s = s.replace(k, v)
    return s

def clean_name(t):
    t = t.strip().split("\n")[0].strip().strip('."\'*').strip()
    t = re.sub(r'^(the answer is|answer:)\s*', '', t, flags=re.I)
    return t[:80]

# ---------- 本体 ----------
def run_pilot(items, gen, p_true, nli, log=print, n_samples=3, max_cands=3, max_combos=9):
    """gen(prompts, sample=False)->list[str]; p_true(prompts)->list[float]; nli(prem, hyp)->list[[e,n,c]]"""
    for it in items: it["sents"] = split_sents(it["evidence"])
    ev = lambda it: "\n".join(it["sents"])
    t0 = time.time()
    # 1. 答えと自信
    pt = p_true([VERDICT_T.format(ev=ev(it), claim=it["claim"]) for it in items])
    for it, p in zip(items, pt): it["verdict"] = {"p_true": float(p)}
    log(f"答えと自信 済み {time.time()-t0:.0f}秒")
    # 2. 分ける（A：そのまま、B：隠れた人や物）
    ta = gen([DECOMP_A.format(claim=it["claim"]) for it in items])
    tb = gen([DECOMP_B.format(claim=it["claim"]) for it in items])
    for it, a, b in zip(items, ta, tb):
        h, f = parse_B(b)
        it["decomp"] = {"A_raw": a, "A": parse_A(a), "B_raw": b, "B": {"hidden": h, "facts": f, "cands": {}}}
    log(f"分ける 済み {time.time()-t0:.0f}秒")
    # 3. 隠れた点の候補（前の点の候補を入れてから聞く。決まった答え1つ＋揺らした答え n_samples）
    for it in items:
        B = it["decomp"]["B"]; assigns = [{}]
        for k in sorted(B["hidden"]):
            new = []
            for a in assigns:
                desc = fill(B["hidden"][k], a)
                pr = [FILL_T.format(ev=ev(it), desc=desc)]
                outs = gen(pr) + gen(pr * n_samples, sample=True)
                c = []
                for o in outs:
                    o = clean_name(o)
                    if o and o not in c: c.append(o)
                B["cands"].setdefault(k, {})[json.dumps(a, ensure_ascii=False)] = c[:max_cands]
                new += [dict(a, **{k: x}) for x in c[:max_cands]]
            assigns = new[:max_combos] or [{}]
        B["assigns"] = assigns
    log(f"隠れた点の候補 済み {time.time()-t0:.0f}秒")
    # 4. NLI：資料の文 × (元の主張 / Aの部分 / Bの部分に候補を入れたもの)
    P, H, where = [], [], []
    for i, it in enumerate(items):
        for j, s in enumerate(it["sents"]):
            P.append(s); H.append(it["claim"]); where.append((i, "claim", None, j))
            for fi, f in enumerate(it["decomp"]["A"]):
                P.append(s); H.append(f); where.append((i, "A", str(fi), j))
            B = it["decomp"]["B"]
            for ai, a in enumerate(B["assigns"]):
                for fi, f in enumerate(B["facts"]):
                    P.append(s); H.append(fill(f, a)); where.append((i, "B", f"{fi}|{ai}", j))
    log(f"NLIの組 {len(P)}")
    out = nli(P, H)
    for it in items: it["nli"] = {"claim": [None] * len(it["sents"]), "A": {}, "B": {}}
    for (i, kind, key, j), v in zip(where, out):
        d = items[i]["nli"]
        if kind == "claim": d["claim"][j] = v
        else: d[kind].setdefault(key, [None] * len(items[i]["sents"]))[j] = v
    log(f"NLI 済み {time.time()-t0:.0f}秒")
    return items

# ---------- 作り物のAI（流れの確かめ用。研究には使わない） ----------
def mock_models(seed=0):
    r = np.random.default_rng(seed)
    def gen(prompts, sample=False):
        out = []
        for p in prompts:
            if p.startswith("Break"):   out.append("- Fact one.\n- Fact two.")
            elif p.startswith("Rewrite"): out.append("#1 = the thing in the claim\n- #1 did something.\n- #1 is somewhere.")
            else: out.append(r.choice(["Alice", "Bob", "Carol"]))
        return out
    def p_true(prompts): return r.random(len(prompts)).tolist()
    def nli(P, H): return r.dirichlet([1, 1, 1], len(P)).tolist()
    return gen, p_true, nli
