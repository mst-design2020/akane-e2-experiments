"""E7の中身（Colabと手元の両方で使う）：説明文の頭の名詞（genus）を取り出し、辞書をたどって網を作る材料をそろえる。"""
import re

MAX_DEFS_ENT, MAX_DEFS_MID, DEPTH = 8, 4, 3
PERSON_HEADS = ("one who", "someone who", "somebody who", "a person who", "person who", "one that", "anyone who",
                "a person", "someone", "somebody")
OF_HEADS = {"kind", "type", "sort", "variety", "species", "form", "member", "one", "any", "piece", "item", "genus",
            "class", "breed", "style", "group", "family", "unit", "strain", "subspecies", "category", "instance",
            "example", "specimen", "individual", "portion", "quantity"}
META = {"synonym", "ellipsis", "term", "name", "word", "abbreviation", "plural", "spelling", "who", "something",
        "anything", "that", "which", "this", "one", "any", "form", "use", "sense", "variant", "misspelling", "acronym"}
SKIP_DEF = re.compile(r"^(\(|\[)?(obsolete|archaic|dated|rare|slang|vulgar|offensive|derogatory|historical)\b", re.I)


def genus(defn, nlp):
    """説明文の頭の名詞を1語で返す（取れなければ None）。例: 'A young cat' → 'cat'、'One who wears' → 'person'。"""
    t = re.sub(r"\([^)]*\)", " ", defn).strip()
    low = t.lower().lstrip("“\"' ")
    for p in PERSON_HEADS:
        if low.startswith(p + " ") or low == p:
            return "person"
    doc = nlp(t[:200])
    chunks = list(doc.noun_chunks)
    if not chunks:
        return None
    root = chunks[0].root
    lem = root.lemma_.lower()
    if lem in OF_HEADS:  # 'A kind of X' → X、'A member of the cat family' → cat family の頭
        for ch in root.children:
            if ch.dep_ == "prep" and ch.text.lower() == "of":
                objs = [c for c in ch.children if c.dep_ == "pobj"]
                if objs:
                    lem = objs[0].lemma_.lower()
                    break
    if lem == "being" and "human" in chunks[0].text.lower():
        lem = "human"
    if lem == "specie":
        lem = "species"
    if lem in META or not re.fullmatch(r"[a-z][a-z\-]*", lem):
        return None
    return lem


# --- Wiktionaryのダンプ（wikitext）から英語の名詞の意味を取り出す ---
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


def _tmpl(m):
    parts = [p.strip() for p in m.group(1).split("|")]
    name, args = parts[0].strip(), [p for p in parts[1:] if "=" not in p]
    if name in DROP_T:
        return ""
    if name in ARG2_T:
        return args[1] if len(args) > 1 else ""
    if name in ARG1_T:
        return args[0] if args else ""
    return ""


def strip_wikitext(t):
    for _ in range(6):
        t2 = re.sub(r"\{\{([^{}]*)\}\}", _tmpl, t)
        if t2 == t:
            break
        t = t2
    t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"'{2,}|<[^>]+>", "", t)
    return re.sub(r"\s+", " ", t).strip(" ;,.") + "."


def noun_defs_from_wikitext(wt, limit=MAX_DEFS_ENT):
    m = re.search(r"(?ms)^==English==\s*$(.*?)(?=^==[^=]|\Z)", wt or "")
    if not m:
        return []
    out = []
    for sec in re.finditer(r"(?ms)^===+\s*Noun\s*===+\s*$(.*?)(?=^===|\Z)", m.group(1)):
        for line in sec.group(1).splitlines():
            if not line.startswith("# "):
                continue
            raw = line[2:].strip()
            if SKIP_T.match(raw):
                continue
            d = strip_wikitext(raw)
            if len(d) > 3 and not d.lower().startswith(SKIP_LINE):
                out.append(d[:220])
    return out[:limit]


def build_graph(items, defs_of, nlp, log=print):
    """物の語から DEPTH 段まで辞書をたどる。defs_of(word, limit) -> list[str]。
    返り値 G[word] = [{"def": 説明文, "genus": 頭の名詞 or None}]（物の語は8個まで、途中の語は4個まで）。"""
    G = {}
    frontier = sorted(set(it["ent"] for it in items))
    ent_defs = {it["ent"]: it["wk_defs"] for it in items}
    for depth in range(DEPTH + 1):
        nxt = set()
        for w in frontier:
            if w in G:
                continue
            ds = ent_defs.get(w) if (depth == 0 and w in ent_defs) else defs_of(w, MAX_DEFS_MID)
            ds = [d for d in (ds or []) if not SKIP_DEF.match(d)]
            G[w] = [{"def": d, "genus": genus(d, nlp)} for d in ds]
            if depth < DEPTH:
                nxt.update(x["genus"] for x in G[w] if x["genus"] and x["genus"] not in G)
        log(f"段{depth}: 語 {len(frontier)}  次に調べる語 {len(nxt)}")
        frontier = sorted(nxt)
        if not frontier:
            break
    return G


def reach(G, start, depth=DEPTH):
    """start から DEPTH 段までに出てくる語（start含む）。"""
    seen, front = {start}, [start]
    for _ in range(depth):
        nf = []
        for w in front:
            for x in G.get(w, []):
                g = x["genus"]
                if g and g not in seen:
                    seen.add(g); nf.append(g)
        front = nf
    return seen


def nli_jobs(items, G):
    """NLIに聞く組：段（"w" means: 説明文 ⇒ W is a type of genus）と着き（⇒ W is a type of C）。"""
    steps, arrive = set(), set()
    for it in items:
        C = it["catw"]
        for w in reach(G, it["ent"]):
            for k, x in enumerate(G.get(w, [])):
                arrive.add((w, k, C))
                if x["genus"]:
                    steps.add((w, k))
    return sorted(steps), sorted(arrive)


def premise(w, d):
    return f"\"{w}\" means: {d}"


def hyp(w, c):
    return f"{w[:1].upper() + w[1:]} is a type of {c}."


def parse_dump(fobj, log=print, every=500000):
    """Wiktionaryのダンプ（XML）を流し読みし、小文字1語の見出しの英語の名詞の意味を集める。{語: [説明文]}"""
    import xml.etree.ElementTree as ET
    out, n, title, ns = {}, 0, None, None
    for ev, el in ET.iterparse(fobj, events=("end",)):
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "title":
            title = el.text
        elif tag == "ns":
            ns = el.text
        elif tag == "text":
            if ns == "0" and title and re.fullmatch(r"[a-z][a-z\-]*", title) and el.text and "==English==" in el.text:
                ds = noun_defs_from_wikitext(el.text)
                if ds:
                    out[title] = ds
        elif tag == "page":
            n += 1
            el.clear()
            if n % every == 0:
                log(f"ページ {n:,}  名詞のある語 {len(out):,}")
    return out
