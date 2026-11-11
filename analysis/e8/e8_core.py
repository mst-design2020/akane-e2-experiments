"""E8の中身（Colabと手元の両方で使う）。
E7との違い：
- ダンプを読む時に、意味ごとの印（{{lb|en|slang}} など）を残す。途中の語の意味は8個まで持つ。
- 鎖の点を「語」ではなく「語＋意味の番号」にし、途中の語の意味を選ぶ方法を4通り：
  A 選ばない（E7と同じ、途中の語は最初の4つの意味を全部）
  B 印（古い・まれ・俗語など）のない最初の意味（Kovács 2022型）
  C 1つ前の説明文に一番近い意味（文の近さ、McIntosh 2009・OntoLearn型。分類Cは入れない）
  D Gemmaに「この説明文の中の語 g はどの意味か」を選ばせる（Meconi 2025・ConSeC型。分類Cは入れない）
- 各点から、46分類すべてへの「着き」をNLIで測る（打ち消し＝両立しない分類に着くか、に使う）。
"""
import re

MAX_DEFS_ENT, MAX_DEFS_MID, A_MID, DEPTH = 8, 8, 4, 3
# Note (2026-11-11): hand-written list of sense labels, used only by method B (reference). Dropped in analysis/nolist/.
BAD_LABELS = {"obsolete", "archaic", "dated", "rare", "slang", "vulgar", "offensive", "derogatory", "historical",
              "humorous", "euphemistic", "nonstandard", "pejorative", "ethnic slur", "slur", "obsolete outside dialects",
              "rare outside dialects", "nonce word", "jocular", "informal euphemism"}

# 頭の名詞（E7と同じ）
# Note (2026-11-11): PERSON_HEADS, OF_HEADS and META below are hand-written word lists used to pick the head noun.
# The published E7/E8/E9 results were made with them. A version without any hand-written list is in analysis/nolist/
# (e8n_core.py); the recheck and the corrections are in plans/E8n_nolist_recheck.md and plans/E9n_nolist_recheck.md.
# 注（2026-11-11）：下の PERSON_HEADS・OF_HEADS・META は頭の名詞を取るための手書きの語の一覧。一覧なしの版は analysis/nolist/。
PERSON_HEADS = ("one who", "someone who", "somebody who", "a person who", "person who", "one that", "anyone who",
                "a person", "someone", "somebody")
OF_HEADS = {"kind", "type", "sort", "variety", "species", "form", "member", "one", "any", "piece", "item", "genus",
            "class", "breed", "style", "group", "family", "unit", "strain", "subspecies", "category", "instance",
            "example", "specimen", "individual", "portion", "quantity"}
META = {"synonym", "ellipsis", "term", "name", "word", "abbreviation", "plural", "spelling", "who", "something",
        "anything", "that", "which", "this", "one", "any", "form", "use", "sense", "variant", "misspelling", "acronym"}


def genus(defn, nlp):
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
    if lem in OF_HEADS:
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


# --- ダンプの読み取り（E7と同じ書き換え。ただし lb の印は取っておく） ---
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


def labels_of(raw):
    out = []
    for m in LB.finditer(raw):
        for p in m.group(1).split("|"):
            p = p.strip().lower()
            if p and "=" not in p and p not in ("_", "and", "or"):
                out.append(p)
    return out


def noun_senses_from_wikitext(wt, limit=MAX_DEFS_MID):
    """[[説明文, [印]], ...]（E7と同じ説明文の並び。印は {{lb|en|...}} から）"""
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
                out.append([d[:220], labels_of(raw)])
    return out[:limit]


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
            if ns == "0" and title and re.fullmatch(r"[a-z][a-z\-]*", title) and el.text and "==English==" in el.text:
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
    """B：印のない最初の意味の番号（全部印つきなら0）"""
    for k, (_, lb) in enumerate(senses):
        if not is_bad(lb):
            return k
    return 0


# --- 鎖をたどる ---
def build_lexicon(items, WKD, nlp, log=print):
    """L[w] = [{"def", "labels", "genus"}]。物の語は items の wk_defs（印なし）か WKD、途中の語は WKD から8個まで。
    A用の段（途中の語は最初の4つ全部）と、B/C/D用の段（途中の語は8個から1つ選ぶ）の両方で出てくる語をすべて集める。"""
    L = {}
    def senses(w, ent_defs=None):
        if w in L:
            return L[w]
        if ent_defs is not None:
            ss = [[d, []] for d in ent_defs][:MAX_DEFS_ENT]
        else:
            ss = WKD.get(w, [])[:MAX_DEFS_MID]
        L[w] = [{"def": d, "labels": lb, "genus": genus(d, nlp)} for d, lb in ss]
        return L[w]
    front = set()
    for it in items:
        ed = it.get("wk_defs")
        senses(it["ent"], ed if ed else None)
        front.add(it["ent"])
    for depth in range(DEPTH):
        nxt = set()
        for w in front:
            for x in L[w]:
                g = x["genus"]
                if g and g not in L:
                    senses(g)
                    nxt.add(g)
        log(f"段{depth + 1}: 新しい語 {len(nxt)}  語の合計 {len(L)}")
        front = nxt
    return L


def edges(L, ents):
    """(w,k) で genus g を持つ段のうち、たどり得るもの（物の語から3段まで、途中の語は全意味）"""
    seen, front, out = set(ents), set(ents), set()
    for _ in range(DEPTH):
        nxt = set()
        for w in front:
            for k, x in enumerate(L.get(w, [])):
                g = x["genus"]
                if g and g in L and g != w:
                    out.add((w, k))
                    if g not in seen:
                        seen.add(g); nxt.add(g)
        front = nxt
    return sorted(out)


def choice_prompt(w, d, g, cands):
    opts = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(cands))
    return (f"Dictionary definition of \"{w}\": {d}\n\nIn this definition, which meaning of the word \"{g}\" is used?\n"
            f"{opts}\n\nAnswer with the number only.")


def premise(w, d):
    return f"\"{w}\" means: {d}"


def hyp(w, c):
    return f"{w[:1].upper() + w[1:]} is a type of {c}."


# Note (2026-11-11): hand-written broad groups for the negative evidence (a negative result). Dropped in analysis/nolist/.
GROUP_WORDS = ["animal", "plant", "food", "man-made object", "material", "body part", "disease", "activity"]


def chain_nodes(L, x, method, choice=None):
    """物の語 x から3段までの点（語, 意味の番号）と段（(w,k)→(g,j)）。
    A：途中の語は最初の4つの意味を全部（E7と同じ）。B/C/D：途中の語は choice[(w,k)] で1つに決める。物の語は全部の意味。"""
    nodes, links = set(), set()
    front = [(x, k) for k in range(len(L.get(x, [])))]
    nodes.update(front)
    for _ in range(DEPTH):
        nxt = []
        for (w, k) in front:
            g = L[w][k]["genus"]
            if not g or g not in L or g == w or not L[g]:
                continue
            if method == "A":
                js = range(min(A_MID, len(L[g])))
            elif method == "B":
                js = [first_good([[s["def"], s["labels"]] for s in L[g]])]
            else:
                js = [choice.get((w, k), 0) if len(L[g]) > 1 else 0]
            for j in js:
                links.add(((w, k), (g, j)))
                if (g, j) not in nodes:
                    nodes.add((g, j)); nxt.append((g, j))
        front = nxt
    return nodes, links
