# E16-0 の中身（ノートと手元の分析で同じ物を使う）。手書きの語の一覧はない。
LET = "ABCD"


def cap(s):
    return s[:1].upper() + s[1:]


def show(ent):
    return ent.replace("_", " ")


def mc_prompt(direction, it, opts):
    body = "\n".join(f"{LET[i]}. {o}" for i, o in enumerate(opts))
    if direction == "f":
        q = f"{cap(show(it['ent']))} is a type of which of the following?"
    else:
        q = f"Which of the following is a type of {it['cname']}?"
    return f"{q}\n{body}\nAnswer with the letter only."


def mc_set(items, cats):
    """4択の問いを全部作る。正解を A〜D の4つの位置に1回ずつ置く（位置の癖を消す）。"""
    out = []
    for n, it in enumerate(items):
        for direction in "fr":
            for level in ("easy", "hard"):
                ds = it[f"{direction}_{level}"]
                if direction == "f":
                    right, wrong = cats[it["cat"]], [cats[c] for c in ds]
                else:
                    right, wrong = show(it["ent"]), [show(z) for z in ds]
                for pos in range(4):
                    opts = wrong[:pos] + [right] + wrong[pos:]
                    out.append(dict(i=n, d=direction, lv=level, pos=pos, q=mc_prompt(direction, it, opts)))
    return out


def open_f_prompt(ent):
    return f"{cap(show(ent))} is a type of what? Answer with one or two words only."


def open_r_prompt(cname, k=20):
    return f"List {k} different types of {cname}, one per line, with no other text."


def parse_list(text):
    out = []
    for ln in text.splitlines():
        s = ln.strip().lstrip("-*•0123456789.) ").strip().strip(".").lower()
        if s and s not in out:
            out.append(s)
    return out
