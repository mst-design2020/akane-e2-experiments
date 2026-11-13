"""E12 の中身（Colabと手元の両方で使う）。聞き方は E8 の方法D（e8_core.choice_prompt）と同じ文。
候補は WordNet の順だと最初の意味（よく使う順）が先頭に来て答えが漏れるので、くじで並べ替えた順とその逆順の2通りで聞いて平均する（E11-1と同じ）。
手書きの語の一覧はない。"""
import numpy as np

MAX_SENSES = 8


def choice_prompt(w, d, g, cands):
    opts = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(cands))
    return (f"Dictionary definition of \"{w}\": {d}\n\nIn this definition, which meaning of the word \"{g}\" is used?\n"
            f"{opts}\n\nAnswer with the number only.")


def jobs(items, heads, seed=0, shuffle=True):
    """items: [[id, kind, word, defn, head]], heads: {head: [説明文]}。shuffle=True なら1問2通り、False なら WordNet の順で1通り。"""
    rng = np.random.default_rng(seed)
    out = []
    for _id, _k, w, d, g in items:
        defs = heads[g][:MAX_SENSES]
        if shuffle:
            o = rng.permutation(len(defs)).tolist()
            orders = [o, o[::-1]]
        else:
            orders = [list(range(len(defs)))]
        for od in orders:
            out.append([choice_prompt(w, d, g, [defs[i] for i in od]), od])
    return out


def merge(jb, probs, per):
    """per 通りずつ、見せた順の値を元の順に戻して平均する。"""
    out = []
    for i in range(0, len(jb), per):
        acc = None
        for j in range(i, i + per):
            order, p = jb[j][1], probs[j]
            v = np.zeros(len(order))
            for pos, orig in enumerate(order):
                v[orig] = p[pos]
            acc = v if acc is None else acc + v
        out.append((acc / per).round(4).tolist())
    return out
