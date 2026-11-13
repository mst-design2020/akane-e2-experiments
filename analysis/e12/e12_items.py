"""E12 問題づくり：WordNet 3.0 説明文コーパス（Princeton Gloss Corpus, noun.xml）から
「名詞の説明文の頭の語」と人が付けた意味（tag="man"）を取り出す。
頭の語＝spaCyで最初の名詞のかたまりの頭（E8n の genus_nolist と同じ取り方）。手書きの語の一覧は使わない。
使い方: python3 e12_items.py noun.xml out.json
"""
import sys, json, re, collections
import xml.etree.ElementTree as ET
import spacy
from spacy.tokens import Doc
from nltk.corpus import wordnet as wn

MIN_S, MAX_S = 2, 8   # 候補の意味の数（先に決めた数）


def defs(path):
    for ev, el in ET.iterparse(path, events=("end",)):
        if el.tag != "synset":
            continue
        sid = el.get("id")
        keys = [k.text for k in el.find("keys")]
        d = el.find(".//def")
        if d is not None:
            toks = []
            for t in d:
                if t.tag not in ("wf", "cf"):
                    continue
                text = "".join(t.itertext()).strip()
                if not text:
                    continue
                ids = [i.get("sk") for i in t.findall("id")]
                toks.append(dict(text=text, kind=t.tag, tag=t.get("tag"), ids=ids))
            yield sid, keys, toks
        el.clear()


def head_index(doc):
    chunks = [c for c in doc.noun_chunks if c.root.pos_ in ("NOUN", "PROPN")]
    if not chunks:
        return None, None
    root = chunks[0].root
    of = None
    for ch in root.children:
        if ch.dep_ == "prep" and ch.text.lower() == "of":
            for o in ch.children:
                if o.dep_ == "pobj" and o.pos_ in ("NOUN", "PROPN"):
                    of = o.i; break
    return root.i, of


def make_item(sid, keys, toks, i, kind):
    t = toks[i]
    st = collections.Counter()
    if t["kind"] != "wf":
        return None, "multiword"
    if t["tag"] != "man":
        return None, "tag_" + str(t["tag"])
    if len(t["ids"]) != 1:
        return None, "multi_key"
    sk = t["ids"][0]
    lemma = sk.split("%")[0]
    if sk.split("%")[1][0] != "1":
        return None, "not_noun"
    if not re.fullmatch(r"[a-z][a-z\-]*", lemma):
        return None, "form"
    syns = wn.synsets(lemma, pos="n")
    syns = [s for s in syns if any(l.name().lower() == lemma for l in s.lemmas())]
    if not (MIN_S <= len(syns) <= MAX_S):
        return None, f"n_senses"
    gold = None
    for k, s in enumerate(syns):
        if any(l.key() == sk for l in s.lemmas()):
            gold = k
    if gold is None:
        return None, "gold_missing"
    wsyn = wn.synset_from_pos_and_offset("n", int(sid[1:]))
    word = keys[0].split("%")[0].replace("_", " ")
    return dict(id=sid, kind=kind, word=word, defn=wsyn.definition(), head=lemma, head_pos=i,
                cands=[s.name() for s in syns], cand_defs=[s.definition() for s in syns], gold=gold,
                n=len(syns)), "ok"


def main(path, out):
    nlp = spacy.load("en_core_web_sm")
    stats = collections.Counter()
    items = []
    for sid, keys, toks in defs(path):
        stats["synsets"] += 1
        if not toks:
            stats["no_def"] += 1; continue
        doc = Doc(nlp.vocab, words=[t["text"] for t in toks])
        for name, proc in nlp.pipeline:
            doc = proc(doc)
        h, of = head_index(doc)
        if h is None:
            stats["no_head"] += 1; continue
        for i, kind in ((h, "head"), (of, "of")):
            if i is None:
                continue
            it, why = make_item(sid, keys, toks, i, kind)
            stats[kind + ":" + why] += 1
            if it:
                items.append(it)
    json.dump(dict(items=items, stats=stats), open(out, "w"))
    print(dict(stats))
    print(len(items))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
