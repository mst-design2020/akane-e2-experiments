"""E10-2の中身（Colabと手元の両方で使う）。計画：sims/e10/PLAN_E10-2.md。手で書いた語の一覧は使わない。
⑩ 辞書の説明文を見せてGemmaに聞く／⑩s 関係のない説明文を見せる（対照）
⑪g 一般的な文の資料（GenericsKB）から、物の語と分類の語の両方を含む文を取り、Gemmaに見せて聞く／⑪s 別の主張の文を見せる（対照）
⑪n 同じ文をNLIの前提にして、主張の文が成り立つか（辞書1段の③と同じ形）
語の形（複数形など）は lemminflect（辞書のデータ）で作る。手で書いた語形の一覧は使わない。
"""
import re

K_DEF, K_FACT, MAX_DEF_CHARS = 3, 5, 220
TOKEN = re.compile(r"[a-z0-9][a-z0-9\-']*")


def lookup(WKD, w):
    """E10-1 と同じ探し方：そのまま → 先頭の大文字/小文字を変えて → 'the ' を外して。"""
    for v in (w, w[:1].upper() + w[1:], w[:1].lower() + w[1:], re.sub(r"^[Tt]he ", "", w)):
        if v in WKD:
            return v
    return None


def defs_text(WKD, w, k=K_DEF):
    """ページに出る順に最初の k 個の意味（印による選び分けはしない）。"""
    key = lookup(WKD, w)
    if not key:
        return "(not in the dictionary)"
    ss = [s[0] if isinstance(s, (list, tuple)) else s["def"] for s in WKD[key][:k]]
    return "; ".join(d[:MAX_DEF_CHARS] for d in ss) if ss else "(not in the dictionary)"


def class_term(dom, c, meta):
    """分類の側の語（辞書を引く語・文を探す語）。P は全体の語、D は分類名、G は地域名。"""
    if dom == "P":
        return c
    if dom == "D":
        return meta["labels"][c]["name"]
    return meta["regions"][c]


def def_prompt(x, xd, w, wd, stmt):
    return (f"Dictionary definitions:\n\"{x}\": {xd}\n\"{w}\": {wd}\n\n"
            f"Using these definitions and your own knowledge, is the following statement true? Answer Yes or No only.\n"
            f"Statement: {stmt}")


def fact_prompt(facts, stmt):
    body = "\n".join(f"- {f}" for f in facts) if facts else "(no facts found)"
    return (f"Facts:\n{body}\n\n"
            f"Using these facts and your own knowledge, is the following statement true? Answer Yes or No only.\n"
            f"Statement: {stmt}")


def gemma_q(stmt):
    return f"Is the following statement true? Answer Yes or No only.\nStatement: {stmt}"   # E4〜E10-1と同じ


def tokens(s):
    return TOKEN.findall(s.lower())


def forms(term, infl):
    """語（または語のまとまり）の形の集まり：そのままの形と、最後の語の名詞の変化形（infl＝lemminflect.getAllInflections）。"""
    ws = tokens(term)
    if not ws:
        return set()
    out = {tuple(ws)}
    for vs in infl(ws[-1], upos="NOUN").values():
        for v in vs:
            out.add(tuple(ws[:-1] + [v.lower()]))
    return out


def find_spans(toks, form):
    n = len(form)
    return any(tuple(toks[i:i + n]) == form for i in range(len(toks) - n + 1))


def term_hits(sents, scores, term_forms, log=print, every=500000):
    """一般的な文を score の高い順に並べ、各語（の形のどれか）を含む文の番号の集まりを返す。
    term_forms: {語: {形(語の並び), ...}}。文は一度だけ読む。返す番号は並べ替えた後の番号（小さいほど score が高い）。"""
    order = sorted(range(len(sents)), key=lambda i: -scores[i])
    by_first = {}
    for t, fs in term_forms.items():
        for f in fs:
            by_first.setdefault(f[0], []).append((t, f))
    hits = {t: set() for t in term_forms}
    kept = {}
    for r, i in enumerate(order):
        toks = tokens(sents[i])
        found = False
        for w in set(toks) & by_first.keys():
            for t, f in by_first[w]:
                if len(f) == 1 or find_spans(toks, f):
                    hits[t].add(r); found = True
        if found:
            kept[r] = sents[i]
        if every and r % every == 0:
            log(f"  文 {r:,}/{len(order):,}")
    return hits, kept
