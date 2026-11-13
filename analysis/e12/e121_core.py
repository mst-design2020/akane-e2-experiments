"""E12-1 の中身：Gemma は WordNet の説明文を覚えていて当てただけか。
B 定義される語の名前を隠す／C 名前を隠し、説明文と候補を別のAI（Qwen 2.5 1.5B）で言い換える／D 説明文の前半を見せて後半を言わせ、覚えている度合いを測る。
手書きの語の一覧はない。"""
import re
import numpy as np

MAX_SENSES = 8
WORD = re.compile(r"[a-z0-9]+")


def choice_prompt_hidden(d, g, cands):
    opts = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(cands))
    return (f"Dictionary definition of a word: {d}\n\nIn this definition, which meaning of the word \"{g}\" is used?\n"
            f"{opts}\n\nAnswer with the number only.")


def jobs_hidden(rows, seed=0):
    """rows: [(説明文, 頭の語, 候補の説明文)]。くじの順と逆順の2通り。"""
    rng = np.random.default_rng(seed)
    out = []
    for d, g, defs in rows:
        defs = defs[:MAX_SENSES]
        o = rng.permutation(len(defs)).tolist()
        for od in (o, o[::-1]):
            out.append([choice_prompt_hidden(d, g, [defs[i] for i in od]), od])
    return out


def para_prompt(d, keep=None):
    k = f" Keep the word \"{keep}\" exactly as it is." if keep else ""
    return ("Rewrite the following dictionary definition in different words, with the same meaning. "
            f"Answer with the new definition only, in one line.{k}\n\nDefinition: {d}")


def has_word(text, w):
    return re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", text.lower()) is not None


def split_half(d, min_words=6):
    ws = d.split()
    if len(ws) < min_words:
        return None
    k = (len(ws) + 1) // 2
    return " ".join(ws[:k]), " ".join(ws[k:])


def recall_prompt(word, first):
    return (f"Complete the WordNet definition of the noun \"{word}\". Continue the text exactly; answer with the rest of the definition only.\n\n"
            f"{word}: {first}")


def lcs_recall(ref, hyp):
    """後半の語のうち、言わせた文に同じ順で出た割合（ROUGE-L の取りこぼしの見方）。"""
    a, b = WORD.findall(ref.lower()), WORD.findall(hyp.lower())
    if not a:
        return None
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1] / len(a)


def overlap(a, b):
    A, B = set(WORD.findall(a.lower())), set(WORD.findall(b.lower()))
    return len(A & B) / max(1, len(A | B))
