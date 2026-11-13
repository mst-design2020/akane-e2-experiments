"""E12 手元で出せる方法（GPUなし）：S0でたらめ・S1最初の意味・S2 Lesk（説明文の重なり、重みは全説明文から数えたIDF）・
参考T WordNetの木（定義される語の上位にある候補を選ぶ。答えに近い情報を使うので比べ相手ではなく目安）。
手書きの語の一覧（よく出る語を飛ばす一覧など）は使わない。よく出る語はIDFで自然に軽くなる。
使い方: python3 e12_local.py items.json out.json
"""
import sys, json, math, re, collections
import numpy as np
from nltk.corpus import wordnet as wn
import gzip as _gz
jl = lambda p: __import__("json").load(_gz.open(p, "rt") if str(p).endswith(".gz") else open(p))   # .json と .json.gz の両方

TOK = re.compile(r"[a-z][a-z\-]*")


def idf_table():
    df, n = collections.Counter(), 0
    for s in wn.all_synsets("n"):
        n += 1
        df.update(set(TOK.findall(s.definition().lower())))
    return {w: math.log(n / c) for w, c in df.items()}, math.log(n)


def lesk(defn, cands, idf, dmax):
    a = set(TOK.findall(defn.lower()))
    out = []
    for c in cands:
        b = set(TOK.findall(c.lower()))
        out.append(round(sum(idf.get(w, dmax) for w in a & b), 4))
    return out


def tree(item):
    s = wn.synset_from_pos_and_offset("n", int(item["id"][1:]))
    anc = set(x for p in s.closure(lambda y: y.hypernyms() + y.instance_hypernyms()) for x in [p])
    v = []
    for c in item["cands"]:
        cs = wn.synset(c)
        v.append(1.0 if cs in anc else (cs.path_similarity(s) or 0.0) * 0.5)
    return v


def main(src, out):
    items = jl(src)["items"]
    idf, dmax = idf_table()
    res = []
    for it in items:
        res.append(dict(id=it["id"], kind=it["kind"], lesk=lesk(it["defn"], it["cand_defs"], idf, dmax), tree=tree(it)))
    json.dump(res, open(out, "w"))
    print(len(res))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
