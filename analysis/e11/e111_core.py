"""E11-1 の中身（計画 plans/E11-1_usual_sense.md）。手書きの語の一覧はない。
- semcor_items：WordNet の名詞から、決まりで問題の語を選ぶ（正解は SemCor の回数が一番多い意味）。
- jobs_usual / jobs_leak：Gemma への聞き方（くじの順と逆順の2通り）。
- merge：2通りの答えを元の順に戻して平均する。"""
import numpy as np

MAX_SENSES = 8   # E8 と同じ（候補の番号が1桁で済む）


def semcor_items(wn, n=2000, seed=0, min_top=3):
    """意味が2〜MAX_SENSES個の名詞で、一番多い意味の回数が min_top 以上、かつ2番目より多い物から n 語をくじで選ぶ。
    defs・counts は WordNet の元の順（＝SemCor の回数順に近い）。見せる時はくじで並べ替える。"""
    pool = []
    for w in sorted({l for l in wn.all_lemma_names(pos="n")}):
        if not (len(w) >= 3 and w.isascii() and w.isalpha() and w.islower()):
            continue   # 語の形の決まり（複合語・固有名・数字・1〜2文字の略語や文字名を外す）
        ss = wn.synsets(w, pos="n")
        if not 2 <= len(ss) <= MAX_SENSES:
            continue
        cnt = []
        for s in ss:
            c = [l.count() for l in s.lemmas() if l.name().lower() == w]
            cnt.append(int(sum(c)))
        top = sorted(cnt, reverse=True)
        if top[0] < min_top or top[0] == top[1]:
            continue
        pool.append({"w": w, "syn": [s.name() for s in ss], "defs": [s.definition() for s in ss], "counts": cnt,
                     "gold": int(np.argmax(cnt))})
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(pool), size=min(n, len(pool)), replace=False).tolist())
    out = [pool[i] for i in pick]
    for it in out:
        it["pool"] = len(pool)
    return out


def _opts(defs, order):
    return "\n".join(f"{j + 1}. {defs[i]}" for j, i in enumerate(order))


def prompt_usual(w, defs, order):
    return (f"Here are the meanings of the English word \"{w}\":\n{_opts(defs, order)}\n\n"
            f"Which meaning is the most common one in everyday English?\nAnswer with the number only.")


def prompt_leak(w, defs, order, c):
    W = w[:1].upper() + w[1:]
    return (f"Here are the meanings of the English word \"{w}\":\n{_opts(defs, order)}\n\n"
            f"In the sentence \"{W} is a type of {c}.\", which meaning of \"{w}\" is used?\nAnswer with the number only.")


def _orders(n, rng):
    o = rng.permutation(n).tolist()
    return [o, o[::-1]]


def jobs_usual(pairs, seed=0):
    """pairs: [(語, 説明文の一覧)]。1語につき2つ（くじの順・逆順）の [聞き方, 見せた順（元の番号）]。"""
    rng = np.random.default_rng(seed)
    out = []
    for w, defs in pairs:
        defs = defs[:MAX_SENSES]
        for o in _orders(len(defs), rng):
            out.append([prompt_usual(w, defs, o), o])
    return out


def jobs_leak(triples, seed=0):
    """triples: [(語, 説明文の一覧, 分類名)]。参考 R4 用（分類を見せる）。"""
    rng = np.random.default_rng(seed)
    out = []
    for w, defs, c in triples:
        defs = defs[:MAX_SENSES]
        for o in _orders(len(defs), rng):
            out.append([prompt_leak(w, defs, o, c), o])
    return out


def merge(jobs, probs, n):
    """jobs[2i], jobs[2i+1] が i 番目の語。見せた順の値を元の順に戻し、2通りを平均する。"""
    assert len(jobs) == len(probs) == 2 * n
    out = []
    for i in range(n):
        acc = None
        for j in (2 * i, 2 * i + 1):
            order, p = jobs[j][1], probs[j]
            v = np.zeros(len(order))
            for pos, orig in enumerate(order):
                v[orig] = p[pos]
            acc = v if acc is None else acc + v
        out.append((acc / 2).round(4).tolist())
    return out
