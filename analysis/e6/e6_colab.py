"""E6 Colabの中身：辞書の説明文を集め、Gemmaにもう一度聞き、NLIで判定する。
items: e6_items.json（700個、正解は入れない）。結果: {"meta":..., "items":[{id, wk_defs, gemma{none,S1,S2,S3}, nli{S1,S2}}]}
nli は意味ごとの [含む, 中立, 矛盾] の確率（S3＝S2の1番目なので別に計算しない）。
"""
import re, json, time, html

MAX_DEFS, MAX_CHARS = 8, 220


def clean(s):
    s = re.sub(r"<[^>]+>", "", s or "")
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()[:MAX_CHARS]


def parse_wiktionary(js):
    """Wiktionary REST（page/definition）の答えから、英語の名詞の意味だけを取り出す。"""
    out = []
    for blk in (js or {}).get("en", []):
        if blk.get("partOfSpeech") != "Noun":
            continue
        for d in blk.get("definitions", []):
            t = clean(d.get("definition", ""))
            if t and not t.lower().startswith(("plural of", "alternative form of", "alternative spelling of", "obsolete form of")):
                out.append(t)
    return out[:MAX_DEFS]


def strip_wikitext(t):
    t = re.sub(r"\{\{(?:lb|lbl|label|q|qualifier|i|gloss)\|[^{}]*\}\}", "", t)        # ラベル類は消す
    for _ in range(3):
        t = re.sub(r"\{\{[^{}|]*\|(?:[^{}|]*\|)*([^{}|=]*)\}\}", r"\1", t)          # {{l|en|bar}} → bar
    t = re.sub(r"\{\{[^{}]*\}\}", "", t)
    t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", t)                           # [[a|b]] → b
    t = re.sub(r"'{2,}|<[^>]+>", "", t)
    return clean(t)


SKIP = ("plural of", "alternative form of", "alternative spelling of", "obsolete form of")


def parse_wikitext(wt):
    """予備：MediaWikiの wikitext から英語の名詞の意味（'# ' の行）を取り出す。"""
    m = re.search(r"(?ms)^==English==\s*$(.*?)(?=^==[^=]|\Z)", wt or "")
    if not m:
        return []
    out = []
    for sec in re.finditer(r"(?ms)^===+\s*Noun\s*===+\s*$(.*?)(?=^===|\Z)", m.group(1)):
        for line in sec.group(1).splitlines():
            if line.startswith("# "):
                raw = line[2:].strip()
                if re.match(r"\{\{(plural of|alternative form of|alt form|alternative spelling of|obsolete form of)\|", raw):
                    continue
                d = strip_wikitext(raw)
                if d and not d.lower().startswith(SKIP):
                    out.append(d)
    return out[:MAX_DEFS]


def fetch_all(words, get_json, get_wikitext=None, log=print, sleep=0.1):
    """get_json(word) -> dict か None。空なら予備として get_wikitext(word) -> str。取れなかった語は空のリスト。"""
    res = {}
    for i, w in enumerate(words):
        try:
            res[w] = parse_wiktionary(get_json(w))
        except Exception as e:  # 取れない語は空
            res[w] = []
        if not res[w] and get_wikitext is not None:
            try:
                res[w] = parse_wikitext(get_wikitext(w))
            except Exception:
                pass
        if sleep:
            time.sleep(sleep)
        if (i + 1) % 100 == 0:
            log(f"Wiktionary {i + 1}/{len(words)}")
    return res


def sources(it):
    wk = it.get("wk_defs") or []
    return {"S1": it["wn_defs"][:MAX_DEFS], "S2": wk, "S3": wk[:1]}


def gemma_question(it, defs):
    q = f"Is the following statement true? Answer Yes or No only.\nStatement: {it['stmt']}"
    if defs is None:
        return q
    if not defs:
        return q  # 説明文がない語は、説明文なしと同じ聞き方（結果に has_defs を残す）
    lines = "\n".join(f"{k + 1}. {d}" for k, d in enumerate(defs))
    return f"Dictionary definitions of the word \"{it['ent']}\":\n{lines}\n\n" + q


def premise(it, d):
    return f"\"{it['ent']}\" means: {d}"


def run(items, gemma_logits, nli_probs, log=print):
    """gemma_logits(list[str]) -> list[float]（はい−いいえ）。nli_probs(premises, hypotheses) -> list[[含む,中立,矛盾]]"""
    # Gemma：説明文なし＋S1〜S3
    for key in ["none", "S1", "S2", "S3"]:
        qs = [gemma_question(it, None if key == "none" else sources(it)[key]) for it in items]
        L = gemma_logits(qs)
        for it, v in zip(items, L):
            it.setdefault("gemma", {})[key] = float(v)
        log(f"Gemma {key} 済み")
    # NLI：意味ごと（S1, S2）
    for key in ["S1", "S2"]:
        P, H, idx = [], [], []
        for i, it in enumerate(items):
            for d in sources(it)[key]:
                P.append(premise(it, d)); H.append(it["stmt"]); idx.append(i)
        out = nli_probs(P, H) if P else []
        for it in items:
            it.setdefault("nli", {})[key] = []
        for i, p in zip(idx, out):
            items[i]["nli"][key].append([float(x) for x in p])
        log(f"NLI {key} 済み（{len(P)}組）")
    return items


def mock(items, seed=0):
    import random
    r = random.Random(seed)
    for it in items:
        it["wk_defs"] = [f"mock sense {k} of {it['ent']}" for k in range(r.randint(0, 4))]
    gl = lambda qs: [r.gauss(0, 3) for _ in qs]
    def nl(P, H):
        out = []
        for _ in P:
            a = [r.random() for _ in range(3)]; s = sum(a); out.append([x / s for x in a])
        return out
    return run(items, gl, nl)


if __name__ == "__main__":
    import sys
    items = [{k: v for k, v in x.items() if k in ("id", "ent", "stmt", "wn_defs")} for x in json.load(open(sys.argv[1]))]
    # 解析の確かめ
    sample = {"en": [{"partOfSpeech": "Noun", "definitions": [{"definition": "A <a href='x'>bar</a> &amp; weights"}, {"definition": "plural of foo"}]},
                     {"partOfSpeech": "Verb", "definitions": [{"definition": "to lift"}]}]}
    assert parse_wiktionary(sample) == ["A bar & weights"], parse_wiktionary(sample)
    wt = ("==English==\n===Etymology===\nx\n===Noun===\n{{en-noun}}\n# {{lb|en|weightlifting}} A [[bar]] with [[weight|weights]] at each end.\n"
          "#: {{ux|en|He lifted the barbell.}}\n# {{plural of|en|foo}}\n===Verb===\n# to lift\n==French==\n===Noun===\n# haltère")
    assert parse_wikitext(wt) == ["A bar with weights at each end."], parse_wikitext(wt)
    res = mock(items)
    print(json.dumps(res[0], ensure_ascii=False)[:500])
    json.dump({"meta": {"mock": True}, "items": res}, open(sys.argv[2], "w"), ensure_ascii=False)
