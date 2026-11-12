"""E10-1の中身（Colabと手元の両方で使う）。手で書いた語の一覧は使わない（PLAN_E10-1.md、監査_手書き一覧.md）。
E10（e10_core.py）からの変更：
1. 頭の名詞：文法だけで候補を出し（最初の名詞のかたまりの頭と、それに付いた of の目的語。見出し形とそのままの形）、
   どれを使うかはその段のNLIの値で決める（E8nと同じ、set_genus）。PERSON_HEADS・OF_HEADS・META はやめた。
2. 一部・場所の段：「N of Y」「N in Y」は N を問わず全部たどる（LOC_HEADS はやめた）。段が成り立つかはNLIの値で決まる。
   前置詞 of・in は、問いの文の形（is a part of／is located in）から来る物。
3. ダンプから、名詞・固有名詞の意味に加えて、形容詞の意味（病名を分ける直し4で、部位の形容詞 renal などを引くためだけ）と、
   「Meronyms（部分の一覧）」「Holonyms（全体の一覧）」の語（直し2）を読む。
4. 病気の「〜の一部か」の文の器官名は、分類名から文法で作る（system_of）。手書きの部位名はやめた。
書式の読み方（DROP_T など、Wiktionaryのテンプレートの扱い）は E10 と同じ（監査のD：知識ではない）。
"""
import re

MAX_DEFS_ENT, MAX_DEFS_MID, DEPTH, MAX_LOC, MAX_CAND = 8, 8, 3, 3, 4
LOC_PREPS = ("of", "in")   # 問いの文の形（is a part of／is located in）から


# --- 頭の名詞の候補（E8n と同じ） ---
def _clean(defn):
    return re.sub(r"\([^)]*\)", " ", defn).strip()


def genus_candidates(doc, max_c=MAX_CAND):
    chunks = [c for c in doc.noun_chunks if c.root.pos_ in ("NOUN", "PROPN")]
    if not chunks:
        return []
    root = chunks[0].root
    toks = [root] + [o for ch in root.children if ch.dep_ == "prep" and ch.text.lower() == "of"
                     for o in ch.children if o.dep_ == "pobj" and o.pos_ in ("NOUN", "PROPN")]
    out = []
    for tk in toks:
        for w in (tk.lemma_.lower(), tk.text.lower()):
            if w not in out and re.fullmatch(r"[a-z][a-z\-]*", w):
                out.append(w)
    return out[:max_c]


def _np_text(tok):
    """前置詞の目的語のまとまり。固有名詞なら複合語ごと（"East Asia"）、ふつうの名詞は見出しの形。"""
    if tok.pos_ == "PROPN":
        left = [c for c in tok.lefts if c.dep_ in ("compound", "amod") and c.pos_ in ("PROPN", "ADJ") and c.text[:1].isupper()]
        i0 = min([c.i for c in left] + [tok.i])
        return " ".join(t.text for t in tok.doc[i0:tok.i + 1])
    lem = tok.lemma_.lower()
    return lem if re.fullmatch(r"[a-z][a-z\-]*", lem) else None


def loc_targets(doc):
    """「N of Y」「N in Y」の Y を全部（N を問わない）。"""
    out = []
    for tok in doc:
        if tok.dep_ != "prep" or tok.text.lower() not in LOC_PREPS:
            continue
        if tok.head.pos_ not in ("NOUN", "PROPN"):
            continue
        for o in [c for c in tok.children if c.dep_ == "pobj"]:
            for y in [o] + [c for c in o.children if c.dep_ in ("conj", "appos") and c.pos_ == o.pos_]:
                s = _np_text(y)
                if s and s not in out:
                    out.append(s)
    return out[:MAX_LOC]


def lemmas_of(doc):
    return {t.lemma_.lower() for t in doc} | {t.text.lower() for t in doc}


# --- ダンプの読み取り（E10と同じ書き換え＋形容詞・Meronyms・Holonyms） ---
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


def _senses(body, limit):
    out = []
    for line in body.splitlines():
        if not line.startswith("# "):
            continue
        raw = line[2:].strip()
        if SKIP_T.match(raw):
            continue
        d = strip_wikitext(raw)
        if len(d) > 3 and not d.lower().startswith(SKIP_LINE):
            out.append([d[:220], labels_of(raw)])
    return out[:limit]


def _linked_words(body):
    """一覧の節の語：[[X]] と、テンプレート {{名前|en|X|Y…}} の位置引数（= を含まない物）"""
    ws = re.findall(r"\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]", body)
    for m in re.finditer(r"\{\{([^{}]*)\}\}", body):
        ps = [p.strip() for p in m.group(1).split("|")]
        if len(ps) > 2 and ps[1] == "en":
            ws += [p for p in ps[2:] if p and "=" not in p and "<" not in p]
    return [w.strip() for w in ws if re.fullmatch(r"[A-Za-z][A-Za-z\-' ]{0,40}", w.strip()) and not w.startswith(("Thesaurus", "Category", "Appendix"))]


def parse_page(wt):
    """英語の節から {"n": 名詞・固有名詞の意味, "a": 形容詞の意味, "mero": 部分の一覧, "holo": 全体の一覧}"""
    m = re.search(r"(?ms)^==English==\s*$(.*?)(?=^==[^=]|\Z)", wt or "")
    if not m:
        return None
    en = m.group(1)
    out = {"n": [], "a": [], "mero": [], "holo": []}
    for sec in re.finditer(r"(?ms)^(===+)\s*([^=\n]+?)\s*\1\s*$(.*?)(?=^===|\Z)", en):
        head, body = sec.group(2).strip().lower(), sec.group(3)
        if head in ("noun", "proper noun"):
            out["n"] += _senses(body, MAX_DEFS_MID)
        elif head == "adjective":
            out["a"] += _senses(body, MAX_DEFS_MID)
        elif head == "meronyms":
            out["mero"] += _linked_words(body)
        elif head == "holonyms":
            out["holo"] += _linked_words(body)
    out["n"], out["a"] = out["n"][:MAX_DEFS_MID], out["a"][:MAX_DEFS_MID]
    return out if any(out.values()) else None


TITLE_OK = re.compile(r"[A-Za-z][A-Za-z\-'.]*(?: [A-Za-z][A-Za-z\-'.]*){0,3}")


def parse_dump(fobj, log=print, every=500000):
    """WKD（名詞・固有名詞、E10と同じ形）, WKA（形容詞）, MERO, HOLO"""
    import xml.etree.ElementTree as ET
    WKD, WKA, MERO, HOLO = {}, {}, {}, {}
    n, title, ns = 0, None, None
    for ev, el in ET.iterparse(fobj, events=("end",)):
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "title":
            title = el.text
        elif tag == "ns":
            ns = el.text
        elif tag == "text":
            if ns == "0" and title and TITLE_OK.fullmatch(title) and el.text and "==English==" in el.text:
                r = parse_page(el.text)
                if r:
                    if r["n"]: WKD[title] = r["n"]
                    if r["a"]: WKA[title] = r["a"]
                    if r["mero"]: MERO[title] = sorted(set(r["mero"]))
                    if r["holo"]: HOLO[title] = sorted(set(r["holo"]))
        elif tag == "page":
            n += 1
            el.clear()
            if n % every == 0:
                log(f"ページ {n:,}  名詞のある語 {len(WKD):,}")
    return WKD, WKA, MERO, HOLO


def lookup(WKD, w):
    for v in (w, w[:1].upper() + w[1:], w[:1].lower() + w[1:], re.sub(r"^[Tt]he ", "", w)):
        if v in WKD:
            return v
    return None


# --- 語の意味の一覧（一覧なし） ---
def build_lexicon(ents_by_dom, WKD, nlp, WKA=None, adj_ok=(), log=print):
    """L[w] = [{"def","cands","loc","lem"}]。adj_ok の語は名詞の意味がなければ形容詞の意味を使う（直し4の部位の語）。
    たどる語は 頭の名詞の候補 ＋ of/in の目的語（全部）。"""
    L = {}
    def senses(w):
        if w in L:
            return L[w]
        key = lookup(WKD, w)
        ss = WKD.get(key, [])[:MAX_DEFS_MID] if key else []
        if not ss and WKA is not None and w in adj_ok:
            ka = lookup(WKA, w)
            ss = WKA.get(ka, [])[:MAX_DEFS_MID] if ka else []
        rows = []
        for d, lb in ss:
            doc = nlp(_clean(d)[:200])
            rows.append({"def": d, "cands": genus_candidates(doc), "loc": loc_targets(doc), "lem": sorted(lemmas_of(doc))})
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
                    for g in x["cands"] + x["loc"]:
                        if g not in L:
                            senses(g)
                        nxt.add(g)
            log(f"{dom} 段{depth + 1}: 語の合計 {len(L)}")
            front = nxt
    return L


def set_genus(L, SC):
    """SC[(w,k,g)] ＝ 一種の段のNLIの値。候補のうち辞書にあり一番大きい物を頭の名詞にする。"""
    for w, xs in L.items():
        for k, x in enumerate(xs):
            cs = [g for g in x["cands"] if g != w and L.get(g)]
            x["genus"] = max(cs, key=lambda g: SC.get((w, k, g), 0.0)) if cs else None
    return L


def next_of(L, w, k, isa_only=False):
    x = L[w][k]
    out = []
    if x.get("genus") and L.get(x["genus"]):
        out.append((x["genus"], "isa"))
    if not isa_only:
        for g in x["loc"]:
            if g != w and L.get(g) and (g, "isa") not in out:
                out.append((g, "loc"))
    return out


def chain_nodes(L, x, choice, depth=DEPTH, isa_only=False):
    """物の語 x から depth 段までの点と段 ((w,k),(g,j),種類)。途中の語の意味は choice[(w,k,g)]（Gemmaの選び）。"""
    nodes, links = set(), set()
    front = [(x, k) for k in range(len(L.get(x, [])))]
    nodes.update(front)
    for _ in range(depth):
        nxt = []
        for (w, k) in front:
            for g, typ in next_of(L, w, k, isa_only):
                j = 0 if len(L[g]) == 1 else choice.get((w, k, g), 0)
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
art = lambda w: ("an " if w[:1].lower() in "aeiou" else "a ") + w


def isa_hyp(w, g):
    return f"{cap(w)} is a type of {g}."


def step_hyp(w, g, typ, dom):
    if typ == "isa":
        return isa_hyp(w, g)
    return {"P": f"{cap(w)} is a part of {g}.", "G": f"{cap(w)} is located in {g}.", "D": f"{cap(w)} affects the {g}."}[dom]


def system_of(name, WKD, WKA):
    """分類名の頭の名詞（最後の語）を外し、残りの最後の語が辞書に名詞としてあり形容詞としてはない時だけ、残りを器官の名にする
    （gastrointestinal system disease → gastrointestinal system、skin disease → skin、metabolic disease → なし）。"""
    ws = name.split()
    if len(ws) < 2:
        return None
    last = ws[-2]
    return " ".join(ws[:-1]) if lookup(WKD, last) and not lookup(WKA, last) else None


def arrive_hyps(w, cls, dom, meta):
    if dom == "P":
        return [f"{cap(w)} is a part of {meta['wholes'][cls]}."]
    if dom == "G":
        return [f"{w} is located in {meta['regions'][cls]}."]
    lb = meta["labels"][cls]
    hs = [f"{cap(w)} is {lb['art']} {lb['name']}."]
    if lb.get("system"):
        hs.append(f"{cap(w)} is a part of the {lb['system']}.")
    return hs


def gemma_q(stmt):
    return f"Is the following statement true? Answer Yes or No only.\nStatement: {stmt}"   # E4〜E10と同じ聞き方


def dir_prompt(a, b):
    """直し3：向きの2択（A＝1つ目の文）。"""
    return f"Which statement is true? Answer A or B only.\nA: {a}\nB: {b}"


def split_name(name):
    """直し4：病名を最後の語（種類の候補）と残り（部位・形容詞）に分ける。2語以上の時だけ。"""
    ws = name.split()
    if len(ws) < 2:
        return None
    return ws[-1].lower(), " ".join(ws[:-1]).lower()


def disease_hyp(w):
    return f"{cap(w)} is a disease."
