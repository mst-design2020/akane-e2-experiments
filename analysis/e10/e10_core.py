"""E10の中身（Colabと手元の両方で使う）。E8（sims/e8/e8_core.py）からの変更点：
1. ダンプは「名詞」に加えて「固有名詞」（地名）も読み、空白入り（4語まで）・大文字始まりの見出しも取る。
   地名の説明文によく使われる {{place|en|city|p/Osaka|c/Japan}} を「A city in Osaka, Japan.」の形の文に直す。
2. 鎖の段を2種類にする：
   - 一種の段（E8と同じ、説明文の頭の名詞）
   - 一部・場所の段（新）：説明文の中の「N of Y」「N in Y」で、N が場面ごとの決まった語の一覧に入る時だけ Y をたどる（狭いたどり方）。
     広いたどり方（N を問わない）は部分と全体の参考だけ。
3. 段の確からしさのNLIの言い方：一種の段は E8 と同じ「w is a type of g.」、一部・場所の段は場面ごとに
   P「w is a part of g.」 G「w is located in g.」 D「w affects the g.」
4. 着き（点→分類）のNLIの言い方：P「w is a part of (全体).」 G「w is located in (地域).」
   D「w is a (分類).」と、体の部位の分類なら「w is a part of the (器官系).」の大きい方。
意味の選び方（Gemmaに番号で選ばせる D・印のない最初の意味 B）、3段まで、一番強い道の掛け算は E8 と同じ。
"""
import re

MAX_DEFS_ENT, MAX_DEFS_MID, DEPTH, MAX_LOC = 8, 8, 3, 3
BAD_LABELS = {"obsolete", "archaic", "dated", "rare", "slang", "vulgar", "offensive", "derogatory", "historical",
              "humorous", "euphemistic", "nonstandard", "pejorative", "ethnic slur", "slur", "obsolete outside dialects",
              "rare outside dialects", "nonce word", "jocular", "informal euphemism"}
PERSON_HEADS = ("one who", "someone who", "somebody who", "a person who", "person who", "one that", "anyone who",
                "a person", "someone", "somebody")
OF_HEADS = {"kind", "type", "sort", "variety", "species", "form", "member", "one", "any", "piece", "item", "genus",
            "class", "breed", "style", "group", "family", "unit", "strain", "subspecies", "category", "instance",
            "example", "specimen", "individual", "portion", "quantity"}
META = {"synonym", "ellipsis", "term", "name", "word", "abbreviation", "plural", "spelling", "who", "something",
        "anything", "that", "which", "this", "one", "any", "form", "use", "sense", "variant", "misspelling", "acronym"}

# 一部・場所の段をたどってよい N（結果を見る前に固定、PLAN_E10.md）
LOC_HEADS = {
    "P": {"part", "piece", "section", "portion", "end", "edge", "side", "front", "back", "top", "bottom", "surface",
          "component", "organ", "segment", "bone", "muscle", "joint", "layer", "membrane", "tissue", "cavity", "area",
          "region", "structure", "tip", "rim", "frame", "cover", "covering", "opening", "lining", "wall", "base"},
    "G": {"city", "town", "capital", "port", "village", "municipality", "country", "nation", "state", "province",
          "region", "island", "archipelago", "republic", "kingdom", "territory", "prefecture", "county", "district",
          "part", "area", "emirate", "commune", "settlement", "metropolis", "suburb", "borough", "seaport", "community",
          "governorate", "department", "division", "subdivision", "regency", "oblast", "canton", "parish", "locality"},
    "D": {"disease", "disorder", "inflammation", "infection", "cancer", "tumor", "tumour", "condition", "carcinoma",
          "syndrome", "illness", "cyst", "lesion", "swelling", "injury", "ailment", "malady", "failure", "deficiency",
          "abnormality", "dysfunction", "pain", "enlargement", "narrowing", "hardening", "ulcer", "sarcoma", "lymphoma",
          "leukemia", "leukaemia", "neoplasm", "malformation", "defect", "obstruction", "infestation"}}
LOC_PREPS = {"of", "in", "within", "inside"}


def _first_head(doc):
    chunks = list(doc.noun_chunks)
    return chunks[0].root if chunks else None


def genus(defn, nlp, doc=None):
    t = re.sub(r"\([^)]*\)", " ", defn).strip()
    low = t.lower().lstrip("“\"' ")
    for p in PERSON_HEADS:
        if low.startswith(p + " ") or low == p:
            return "person"
    doc = doc if doc is not None else nlp(t[:200])
    root = _first_head(doc)
    if root is None:
        return None
    lem = root.lemma_.lower()
    if lem in OF_HEADS:
        for ch in root.children:
            if ch.dep_ == "prep" and ch.text.lower() == "of":
                objs = [c for c in ch.children if c.dep_ == "pobj"]
                if objs:
                    lem = objs[0].lemma_.lower()
                    break
    if lem == "being" and "human" in list(doc.noun_chunks)[0].text.lower():
        lem = "human"
    if lem == "specie":
        lem = "species"
    if lem in META or not re.fullmatch(r"[a-z][a-z\-]*", lem):
        return None
    return lem


def _np_text(tok):
    """前置詞の目的語のまとまり。固有名詞なら複合語ごと（"East Asia"）、ふつうの名詞は見出しの形（"lungs"→"lung"）。"""
    if tok.pos_ == "PROPN":
        left = [c for c in tok.lefts if c.dep_ in ("compound", "amod") and c.pos_ in ("PROPN", "ADJ") and c.text[:1].isupper()]
        i0 = min([c.i for c in left] + [tok.i])
        return " ".join(t.text for t in tok.doc[i0:tok.i + 1])
    lem = tok.lemma_.lower()
    return lem if re.fullmatch(r"[a-z][a-z\-]*", lem) and lem not in META else None


def loc_targets(defn, nlp, dom, wide=False, doc=None):
    """「N of Y」「N in Y」の Y（N が場面の一覧にある時、wide なら N を問わない）。P では「part of Y」は N を問わない。"""
    t = re.sub(r"\([^)]*\)", " ", defn).strip()
    doc = doc if doc is not None else nlp(t[:200])
    heads = LOC_HEADS[dom]
    out = []
    for tok in doc:
        if tok.dep_ != "prep" or tok.text.lower() not in LOC_PREPS:
            continue
        n = tok.head
        if n.pos_ not in ("NOUN", "PROPN"):
            continue
        if not (wide or n.lemma_.lower() in heads):
            continue
        for o in [c for c in tok.children if c.dep_ == "pobj"]:
            objs = [o] + [c for c in o.children if c.dep_ in ("conj", "appos") and c.pos_ == o.pos_]
            for y in objs:
                s = _np_text(y)
                if s and s not in out:
                    out.append(s)
    return out[:MAX_LOC]


# --- ダンプの読み取り（E8と同じ書き換え＋{{place}}） ---
DROP_T = {"lb", "lbl", "label", "q", "qual", "qualifier", "i", "gloss", "gl", "sense", "s", "defdate", "rfdef", "rfclarify",
          "senseid", "tlb", "def-date", "c", "C", "topics", "cat", "rfex", "rfquote", "nb...", "ux", "uxi", "quote-book"}
ARG2_T = {"l", "m", "link", "mention", "ll", "l-self", "m-self", "cog"}
ARG1_T = {"w", "taxlink", "taxfmt", "vern", "ws", "pedia", "non-gloss", "n-g", "ngd", "non-gloss definition", "glossary",
          "smallcaps", "taxon", "spelink", "nowrap", "small", "sub", "sup"}
SKIP_LINE = ("plural of", "alternative form of", "alternative spelling of", "obsolete form of", "misspelling of",
             "abbreviation of", "initialism of", "acronym of", "obsolete spelling of", "archaic spelling of", "eye dialect of",
             "nonstandard spelling of", "synonym of", "clipping of", "short for", "ellipsis of", "contraction of")
SKIP_T = re.compile(r"^\{\{\s*(plural of|alternative form of|alt form|alt sp|alternative spelling of|obsolete form of|"
                    r"misspelling of|abbreviation of|abbr of|initialism of|acronym of|obsolete spelling of|archaic spelling of|"
                    r"eye dialect of|nonstandard spelling of|synonym of|syn of|clipping of|short for|ellipsis of|"
                    r"contraction of|form of|inflection of|en-past of)\s*\|", re.I)
LB = re.compile(r"\{\{\s*(?:lb|lbl|label|tlb)\s*\|\s*en\s*\|([^{}]*)\}\}")


def _holo(h):
    h = h.split("/")[-1].strip()
    return re.sub(r"^the ", "", h)


def render_place(args):
    """{{place|en|...}} を文に。新しい書き方（<<city>> in <<p/Osaka>>）と古い書き方（city|p/Osaka|c/Japan）の両方。"""
    pos = [a for a in args if "=" not in a]
    if pos and pos[0] == "en":
        pos = pos[1:]
    if not pos:
        return ""
    if any("<<" in a for a in pos):
        txt = "|".join(pos)
        txt = re.sub(r"<<([^<>]*)>>", lambda m: _holo(m.group(1)) if "/" in m.group(1) else m.group(1), txt)
        return txt.replace("|", " ")
    groups, cur = [], []
    for a in pos:
        if a.strip() == ";":
            if cur: groups.append(cur)
            cur = []
        else:
            cur.append(a.strip())
    if cur: groups.append(cur)
    g = groups[0]
    ptype = g[0].replace("/", " ").strip()
    holos = [_holo(h) for h in g[1:] if h and h not in (",", "in", "of")]
    s = ("an " if ptype[:1].lower() in "aeiou" else "a ") + ptype
    return s + (" in " + ", ".join(holos) if holos else "")


def _tmpl(m):
    parts = [p.strip() for p in m.group(1).split("|")]
    name, args = parts[0].strip(), parts[1:]
    pos = [p for p in args if "=" not in p]
    if name == "place":
        return render_place(args)
    if name in DROP_T:
        return ""
    if name in ARG2_T:
        return pos[1] if len(pos) > 1 else ""
    if name in ARG1_T:
        return pos[0] if pos else ""
    return ""


def strip_wikitext(t):
    for _ in range(6):
        t2 = re.sub(r"\{\{([^{}]*)\}\}", _tmpl, t)
        if t2 == t:
            break
        t = t2
    t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"'{2,}|<[^>]+>", "", t)
    t = re.sub(r"\s+", " ", t).strip(" ;,.")
    return (t[:1].upper() + t[1:] + ".") if t else ""


def labels_of(raw):
    out = []
    for m in LB.finditer(raw):
        for p in m.group(1).split("|"):
            p = p.strip().lower()
            if p and "=" not in p and p not in ("_", "and", "or"):
                out.append(p)
    return out


def noun_senses_from_wikitext(wt, limit=MAX_DEFS_MID):
    """名詞と固有名詞の意味 [[説明文, [印]], ...]（ページに出る順）"""
    m = re.search(r"(?ms)^==English==\s*$(.*?)(?=^==[^=]|\Z)", wt or "")
    if not m:
        return []
    out = []
    for sec in re.finditer(r"(?ms)^===+\s*(?:Noun|Proper noun)\s*===+\s*$(.*?)(?=^===|\Z)", m.group(1)):
        for line in sec.group(1).splitlines():
            if not line.startswith("# "):
                continue
            raw = line[2:].strip()
            if SKIP_T.match(raw):
                continue
            d = strip_wikitext(raw)
            if len(d) > 3 and not d.lower().startswith(SKIP_LINE):
                out.append([d[:220], labels_of(raw)])
    return out[:limit]


TITLE_OK = re.compile(r"[A-Za-z][A-Za-z\-'.]*(?: [A-Za-z][A-Za-z\-'.]*){0,3}")


def parse_dump(fobj, log=print, every=500000):
    import xml.etree.ElementTree as ET
    out, n, title, ns = {}, 0, None, None
    for ev, el in ET.iterparse(fobj, events=("end",)):
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "title":
            title = el.text
        elif tag == "ns":
            ns = el.text
        elif tag == "text":
            if ns == "0" and title and TITLE_OK.fullmatch(title) and el.text and "==English==" in el.text:
                ds = noun_senses_from_wikitext(el.text)
                if ds:
                    out[title] = ds
        elif tag == "page":
            n += 1
            el.clear()
            if n % every == 0:
                log(f"ページ {n:,}  名詞のある語 {len(out):,}")
    return out


def is_bad(labels):
    return any(l in BAD_LABELS for l in labels)


def first_good(senses):
    for k, s in enumerate(senses):
        if not is_bad(s["labels"] if isinstance(s, dict) else s[1]):
            return k
    return 0


def lookup(WKD, w):
    """見出しを探す：そのまま → 先頭だけ大文字/小文字を変えて → 'the ' を外して。"""
    for v in (w, w[:1].upper() + w[1:], w[:1].lower() + w[1:], re.sub(r"^[Tt]he ", "", w)):
        if v in WKD:
            return v
    return None


# --- 鎖 ---
def build_lexicon(ents_by_dom, WKD, nlp, log=print, wide_doms=("P",)):
    """L[w] = [{"def","labels","genus","loc":{dom: [...]}, "locw":{dom: [...]}}]。
    ents_by_dom: {dom: [物の語]}。たどる語は場面ごとに集める（一部・場所の段の決まりが場面ごとに違うため）。"""
    L = {}
    def senses(w):
        if w in L:
            return L[w]
        key = lookup(WKD, w)
        ss = WKD.get(key, [])[:MAX_DEFS_MID] if key else []
        rows = []
        for d, lb in ss:
            t = re.sub(r"\([^)]*\)", " ", d).strip()
            doc = nlp(t[:200])
            rows.append({"def": d, "labels": lb, "genus": genus(d, nlp, doc),
                         "loc": {dm: loc_targets(d, nlp, dm, False, doc) for dm in "PGD"},
                         "locw": {dm: loc_targets(d, nlp, dm, True, doc) for dm in wide_doms}})
        L[w] = rows
        return rows
    for dom, ents in ents_by_dom.items():
        front = set(ents)
        for w in ents:
            senses(w)
        for depth in range(DEPTH):
            nxt = set()
            for w in front:
                for x in L[w]:
                    tg = ([x["genus"]] if x["genus"] else []) + x["loc"][dom] + (x["locw"].get(dom, []) if dom in wide_doms else [])
                    for g in tg:
                        if g not in L:
                            senses(g)
                        nxt.add(g)
            log(f"{dom} 段{depth + 1}: 語の合計 {len(L)}")
            front = nxt
    return L


def next_of(L, w, k, dom, wide=False):
    """(w,k) から出る段：[(g, 'isa'|'loc')]"""
    x = L[w][k]
    out = []
    if x["genus"] and x["genus"] != w and L.get(x["genus"]):
        out.append((x["genus"], "isa"))
    locs = x["locw"].get(dom, []) if wide else x["loc"][dom]
    for g in locs:
        if g != w and L.get(g) and (g, "isa") not in out:
            out.append((g, "loc"))
    return out


def chain_nodes(L, x, dom, method, choice=None, wide=False):
    """物の語 x から3段までの点（語, 意味の番号）と段 ((w,k),(g,j),種類)。
    D：途中の語の意味は choice[(w,k,g)]（Gemmaの選び）。B：印のない最初の意味。物の語は全部の意味。"""
    nodes, links = set(), set()
    front = [(x, k) for k in range(len(L.get(x, [])))]
    nodes.update(front)
    for _ in range(DEPTH):
        nxt = []
        for (w, k) in front:
            for g, typ in next_of(L, w, k, dom, wide):
                if len(L[g]) == 1:
                    j = 0
                elif method == "B":
                    j = first_good(L[g])
                else:
                    j = (choice or {}).get((w, k, g), 0)
                links.add(((w, k), (g, j), typ))
                if (g, j) not in nodes:
                    nodes.add((g, j)); nxt.append((g, j))
        front = nxt
    return nodes, links


def choice_prompt(w, d, g, cands):
    opts = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(cands))
    return (f"Dictionary definition of \"{w}\": {d}\n\nIn this definition, which meaning of the word \"{g}\" is used?\n"
            f"{opts}\n\nAnswer with the number only.")


def premise(w, d):
    return f"\"{w}\" means: {d}"


cap = lambda s: s[:1].upper() + s[1:]


def step_hyp(w, g, typ, dom):
    if typ == "isa":
        return f"{cap(w)} is a type of {g}."
    return {"P": f"{cap(w)} is a part of {g}.", "G": f"{cap(w)} is located in {g}.", "D": f"{cap(w)} affects the {g}."}[dom]


def arrive_hyps(w, cls, dom, meta):
    """点 w が分類 cls に「着く」かを問う文（1つか2つ）"""
    if dom == "P":
        return [f"{cap(w)} is a part of {meta['wholes'][cls]}."]
    if dom == "G":
        return [f"{w} is located in {meta['regions'][cls]}."]
    lb = meta["labels"][cls]
    hs = [f"{cap(w)} is {lb['art']} {lb['name']}."]
    if lb["system"]:
        hs.append(f"{cap(w)} is a part of the {lb['system']}.")
    return hs


def gemma_q(stmt):
    return f"Is the following statement true? Answer Yes or No only.\nStatement: {stmt}"   # E4〜E9と同じ聞き方
