"""E14 の中身：辞書の鎖（E11-2 の6段の鎖＝E8n 方法D を伸ばした物）の各点に、
別の資料の「A は B の一種」を継ぎ足す（計画 plans/E14_fill_gaps.md）。手書きの語の一覧はない。
e8n_core（genus_candidates, choice_prompt, premise, hyp, MAX_DEFS_MID）の後に置いて使う。
NLI・Gemma・文の資料は呼び出しの関数として外から渡す（Colab ではモデル、手元の確かめでは作り物）。

点の種類：
  ("s", w, k)  辞書の意味（E8n と同じ。L[w][k]）
  ("g", w, i)  継ぎ足しの文（GR[w][i]。src は "g"＝一般的な文、"s"＝Gemma が書いた文）
つながり：
  ("s",w,k) → ("s",g,j)   辞書の頭の語（強さ ST[(w,k)]、1段）
  ("s",w,k) → ("g",w,i)   同じ語の文。文の中の w の意味が k の時だけ（Gemma が選ぶ）。物の語の文は根から直接。強さ1、段に数えない
  ("g",w,i) → ("s",p,j)   文の中の親の語 p（強さ＝NLI「文 ⇒ W is a type of p.」、1段）。p の意味 j は Gemma が選ぶ
"""
import re

MAX_DEPTH = 6        # E11-2 と同じ上限
K_GKB = 3            # 一般的な文は1語につき3文まで（親への NLI の値が高い順）
N_GKB_READ = 20      # 1語につき読む一般的な文（資料の score の高い順）
MAX_CANDS = 6        # 1文の親の候補の数
SELF_TOKENS = 40     # 書かせる長さ（30語ほど）
WORD = re.compile(r"[a-z][a-z\-]*")


def self_prompt(w):
    return f"Write one short sentence saying what kind of thing a \"{w}\" is."


def sense_in_sentence_prompt(sent, w, cands):
    opts = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(cands))
    return (f"Sentence: {sent}\n\nIn this sentence, which meaning of the word \"{w}\" is used?\n"
            f"{opts}\n\nAnswer with the number only.")


def clean_self(text):
    """書かせた文：最初の空でない行。行の頭の記号を外す（書式の読み方）。"""
    for line in text.splitlines():
        s = re.sub(r"^[\s\-\*\d\.\)\"]+", "", line).strip()
        if len(s.split()) >= 3:
            return s
    return None


def subject_parents(doc, w, alias=None, pron=False):
    """文の主語（名詞のかたまりで、係りが主語）の頭が w の時、文の中のほかの名詞のかたまりの頭（見出し形）を親の候補にする。
    w が主語でなければ None。どの候補を親にするかは NLI で決める（ここでは決めない）。
    E14-3：pron=True なら代名詞の主語（品詞が PRON）も認める（見出しが w と同じ語の記事の段落の中の文だけで使う）。alias は使っていない（転送先の別の語、例 inventor→Invention を主語にしないため）。"""
    chunks = list(doc.noun_chunks)
    names = {w} | ({alias.lower()} if alias else set())
    subj = [c for c in chunks if c.root.dep_ in ("nsubj", "nsubjpass")
            and (names & {c.root.lemma_.lower(), c.root.text.lower()} or (pron and c.root.pos_ == "PRON"))]
    if not subj:
        return None
    s = subj[0]
    out = []
    for c in chunks:
        if c.start == s.start:
            continue
        if c.root.pos_ != "NOUN":
            continue
        lw = c.root.lemma_.lower()
        if lw != w and WORD.fullmatch(lw) and lw not in out:
            out.append(lw)
    return out[:MAX_CANDS]


def new_word(L, w, WKD, nlp):
    """辞書にまだ無い語を WKD（E10-1 の名詞の辞書）から読む。頭の名詞はまだ決めない。"""
    if w in L:
        return bool(L[w])
    L[w] = [{"def": d, "labels": lb, "cands": genus_candidates(d, nlp), "genus": None}
            for d, lb in WKD.get(w, [])[:MAX_DEFS_MID]]
    return bool(L[w])


def follow(L, choice, GR, x, maxd, srcs=("g", "s")):
    """物の語 x から maxd 段まで。点 → 一番浅い段の数、つながりの組。srcs で使う資料を選ぶ（() なら E11-2 の鎖と同じ）。"""
    depth, links = {}, set()
    front = []
    for k in range(len(L.get(x, []))):
        depth[("s", x, k)] = 0; front.append(("s", x, k))
    for i, g in enumerate(GR.get(x, [])):
        if g["src"] in srcs and g.get("parent"):
            depth[("g", x, i)] = 0; front.append(("g", x, i))
    d = 0
    while front and d < maxd:
        nxt = []
        for n in front:
            outs = []
            if n[0] == "s":
                _, w, k = n
                gw = L[w][k].get("genus")
                if gw and gw in L and gw != w and L[gw]:
                    j = choice.get((w, k), 0) if len(L[gw]) > 1 else 0
                    outs.append((("s", gw, j), 1))
                if w != x:   # 途中の語の文（意味が合う時だけ、段に数えない）
                    for i, g in enumerate(GR.get(w, [])):
                        if g["src"] in srcs and g.get("parent") and g.get("wsense", 0) == k:
                            outs.append((("g", w, i), 0))
            else:
                _, w, i = n
                g = GR[w][i]
                outs.append((("s", g["parent"], g.get("psense", 0)), 1))
            for m, step in outs:
                links.add((n, m))
                dm = depth[n] + step
                if m not in depth or dm < depth[m]:
                    depth[m] = dm
                    if step == 0:
                        front.append(m) if dm == d else nxt.append(m)
                    else:
                        nxt.append(m)
        front = nxt
        d += 1
    return depth, links


def grow(items, L, choice, ST, AR, GR, ARG, WKD, nlp, nli, nli_sent, gemma_choice, gemma_sense, gkb_sents, self_write,
         maxd=MAX_DEPTH, log=print, wiki_sents=None):
    """1段ずつ広げる。（E14-2：wiki_sents([語])→{語: 文 or None} を渡すと、Wikipedia の最初の1文を src "w" として足す）各段で、先の点の語について：
    a. 文を集める（一般的な文 N_GKB_READ 文・書かせた文1文）→ 主語が w の文だけ、親の候補を出す
    b. 親の候補を NLI（文 ⇒ W is a type of 候補）で測り、一番の候補を親に。一般的な文は値の高い順に K_GKB 文
    c. 親の語を辞書から読み、親の意味・文の中の w の意味を Gemma が選ぶ
    d. 辞書の意味の点は E11-2 と同じく 頭の名詞・段の値・意味選び（新しい語だけ）
    e. 新しく着いた点の、問題の分類への着き（辞書の意味は E8n と同じ NLI、文は 文 ⇒ W is a type of C.）
    関数：nli([(w,k,c)])→[(含意,矛盾)]（"w" means: 説明文 ⇒ W is a type of c.）、nli_sent([(文, w, c)])→[(含意,矛盾)]、
    gemma_choice([(w,k)])→[番号]（E8n の方法D）、gemma_sense([(文, 語, [説明文])])→[番号]、
    gkb_sents([語])→{語: [文,...]}（score の高い順）、self_write([語])→{語: 文 or None}。"""
    stat = {"words_grafted": 0, "gkb_sent_read": 0, "gkb_kept": 0, "self_kept": 0, "new_words": 0, "arrive": 0}
    done = set(GR)
    n0 = len(L)
    for lev in range(maxd):
        fr = set()
        for it in items:
            dep, _ = follow(L, choice, GR, it["ent"], lev)
            fr |= {n for n, d in dep.items() if d == lev}
        words = sorted({n[1] for n in fr if n[0] == "s"} - done)
        # a. 文を集める
        G = gkb_sents(words) if words else {}
        S = self_write(words) if words else {}
        Wk = wiki_sents(words) if (words and wiki_sents) else {}
        cand = []   # (w, src, 文, [候補])
        for w in words:
            for s in G.get(w, [])[:N_GKB_READ]:
                ps = subject_parents(nlp(s), w)
                if ps:
                    cand.append((w, "g", s, ps))
            stat["gkb_sent_read"] += len(G.get(w, [])[:N_GKB_READ])
            for src, s in (("s", S.get(w)), ("w", Wk.get(w))):
                # E14-3：wiki_sents が {語: (記事の見出しが w と同じ語か, [文,...])} を返す時は、段落の文を全部見る。
                # 主語は w。代名詞（It・They など）の主語は、記事の見出しが w と同じ語の時だけ認める（転送先の別の語の記事では、代名詞は別の物を指すため）
                if isinstance(s, tuple):
                    same, sl = s
                    for x in sl:
                        ps = subject_parents(nlp(x), w, pron=bool(same))
                        if ps:
                            cand.append((w, src, x, ps))
                elif s:
                    ps = subject_parents(nlp(s), w)
                    if ps:
                        cand.append((w, src, s, ps))
        # 親の候補は辞書にある語だけ
        cand = [(w, src, s, [p for p in ps if new_word(L, p, WKD, nlp)]) for w, src, s, ps in cand]
        cand = [c for c in cand if c[3]]
        # b. 親を NLI で決める
        trip = sorted({(s, w, p) for w, src, s, ps in cand for p in ps})
        sc = dict(zip(trip, nli_sent(trip))) if trip else {}
        per = {}
        for w, src, s, ps in cand:
            best = max(ps, key=lambda p: sc[(s, w, p)][0])
            per.setdefault((w, src), []).append({"src": src, "sent": s, "parent": best, "step": sc[(s, w, best)],
                                                 "cands": ps})
        for w in words:
            gs = sorted(per.get((w, "g"), []), key=lambda g: -g["step"][0])[:K_GKB]
            ss = per.get((w, "s"), [])[:1]
            ws = sorted(per.get((w, "w"), []), key=lambda g: -g["step"][0])[:1]   # E14-2 は1文だけなので並べても同じ
            GR[w] = gs + ss + ws
            done.add(w)
            stat["words_grafted"] += 1; stat["gkb_kept"] += len(gs); stat["self_kept"] += len(ss)
        # c. 親の意味と、文の中の w の意味（意味が2つ以上の時だけ Gemma）
        jobs, keys = [], []
        for w in words:
            for i, g in enumerate(GR[w]):
                p = g["parent"]
                if len(L[p]) > 1:
                    jobs.append((g["sent"], p, [x["def"] for x in L[p]])); keys.append((w, i, "psense"))
                else:
                    g["psense"] = 0
                if len(L.get(w, [])) > 1:
                    jobs.append((g["sent"], w, [x["def"] for x in L[w]])); keys.append((w, i, "wsense"))
                else:
                    g["wsense"] = 0
        if jobs:
            for (w, i, f), v in zip(keys, gemma_sense(jobs)):
                GR[w][i][f] = int(v)
        # d. 辞書の意味の点：新しい語の頭の名詞・段の値・意味選び（E11-2 の extend と同じ手順）
        dep_all = set()
        for it in items:
            dep, _ = follow(L, choice, GR, it["ent"], lev + 1)
            dep_all |= set(dep)
        sn = sorted(n for n in dep_all if n[0] == "s" and L[n[1]][n[2]].get("genus") is None and L[n[1]][n[2]]["cands"]
                    and (n[1], n[2]) not in ST)
        for _, w, k in sn:
            for c in L[w][k]["cands"]:
                if c != w and c not in L:
                    stat["new_words"] += int(new_word(L, c, WKD, nlp))
        pairs = sorted({(w, k, c) for _, w, k in sn for c in L[w][k]["cands"] if c != w and L.get(c)})
        gsc = dict(zip(pairs, nli(pairs))) if pairs else {}
        for _, w, k in sn:
            cs = [c for c in L[w][k]["cands"] if (w, k, c) in gsc]
            if cs:
                L[w][k]["genus"] = max(cs, key=lambda c: gsc[(w, k, c)][0])
                ST[(w, k)] = gsc[(w, k, L[w][k]["genus"])]
        cj = sorted((n[1], n[2]) for n in dep_all if n[0] == "s" and L[n[1]][n[2]].get("genus")
                    and len(L.get(L[n[1]][n[2]]["genus"], [])) > 1 and (n[1], n[2]) not in choice)
        if cj:
            for n, v in zip(cj, gemma_choice(cj)):
                choice[n] = int(v)
        # e. 着き
        dep_all = {}
        arr_s, arr_g = set(), set()
        for it in items:
            dep, _ = follow(L, choice, GR, it["ent"], lev + 1)
            C = it["catw"]
            for n in dep:
                if n[0] == "s" and (n[1], n[2], C) not in AR:
                    arr_s.add((n[1], n[2], C))
                elif n[0] == "g" and (n[1], n[2], C) not in ARG:
                    arr_g.add((n[1], n[2], C))
        arr_s, arr_g = sorted(arr_s), sorted(arr_g)
        if arr_s:
            for n, v in zip(arr_s, nli(arr_s)):
                AR[n] = v
        if arr_g:
            for n, v in zip(arr_g, nli_sent([(GR[w][i]["sent"], w, C) for w, i, C in arr_g])):
                ARG[n] = v
        stat["arrive"] += len(arr_s) + len(arr_g)
        log(f"段{lev + 1}: 先の点 {len(fr)}  文を探した語 {len(words)}  文の候補 {len(cand)}  新しい語 {stat['new_words']}  "
            f"着き {len(arr_s)}+{len(arr_g)}")
    stat["L_words_added"] = len(L) - n0
    return stat


def chains(items, L, choice, GR, maxd=MAX_DEPTH):
    """問題ごとの点（種類, 語, 番号, 段の数）とつながり。分析は資料（g・s）を選んで鎖を作り直す。"""
    out = {}
    for it in items:
        dep, links = follow(L, choice, GR, it["ent"], maxd)
        out[it["id"]] = ([[t, w, k, d] for (t, w, k), d in sorted(dep.items())],
                         sorted([list(a), list(b)] for a, b in links))
    return out
