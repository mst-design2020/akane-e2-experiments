"""E15 の中身（Colab と手元の両方で使う。手書きの語の一覧はありません）。
- 親の語（頭の名詞）：今まで＝NLI の段の値が一番の候補（"genus"）。E15＝Gemma に「説明文のどの語が、w が何の一種かを言っているか」を選ばせる（"ggenus"）。
- 最後の段（分類に着くか）：今まで＝その点の説明文1つで NLI（"w" means: 説明文 ⇒ W is a type of C.）。
  E15＝たどった道の説明文を全部見せて Gemma に「この説明文から、x は C の一種か」を聞く。
  対照：道の最後の説明文だけ（last）、関係のない別の問題の道（ctrl）、説明文なし（none）。
- 道の説明文に分類の語 C を足すことはしない（説明文そのものに C が書いてあるのは辞書の情報なのでそのまま）。
"""

MAXD = 3          # E8n と同じ3段
TOP_M = None      # Gemma に聞く点：全部（上から8つに絞ると、意味が8つある語で根の点だけで埋まり、E8n と 89問ずれた）


def gkey(mode):
    return "genus" if mode == "n" else "ggenus"


def step_from(L, CH, ST, mode, w, k):
    """(w,k) から親の点 (g,j) と段の強さ。無ければ None。"""
    g = L[w][k].get(gkey(mode))
    if not g or g not in L or g == w or not L[g]:
        return None
    j = CH[mode].get((w, k), 0) if len(L[g]) > 1 else 0
    return (g, j), ST[mode].get((w, k), (0.0, 0.0))[0]


def reach(L, CH, ST, mode, x, maxd=MAXD):
    """物の語 x の全部の意味から maxd 段まで。点 → (道の強さの掛け算, 道[点...])。同じ点に着く道は強い方。"""
    best = {}
    front = []
    for k in range(len(L.get(x, []))):
        best[(x, k)] = (1.0, [(x, k)]); front.append((x, k))
    for _ in range(maxd):
        nxt = []
        for n in front:
            s = step_from(L, CH, ST, mode, *n)
            if s is None:
                continue
            m, e = s
            p = best[n][0] * e
            if m not in best or p > best[m][0]:
                best[m] = (p, best[n][1] + [m]); nxt.append(m)
        front = nxt
    return best


def top_nodes(best, m=TOP_M):
    o = sorted(best, key=lambda n: (-best[n][0], len(best[n][1]), n))
    return o if m is None else o[:m]


def path_lines(L, path):
    return [f'- "{w}": {L[w][k]["def"]}' for w, k in path]


def arrive_prompt(x, C, lines):
    """lines が空なら説明文なしで聞く。"""
    if lines:
        return ("Dictionary definitions:\n" + "\n".join(lines) +
                f"\n\nBased on these definitions, is a \"{x}\" a kind of \"{C}\"? Answer Yes or No only.")
    return f"Is a \"{x}\" a kind of \"{C}\"? Answer Yes or No only."


def genus_prompt(w, d, cands):
    opts = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(cands))
    return (f"Dictionary definition of \"{w}\": {d}\n\nWhich of these words from the definition names the kind of thing "
            f"that a \"{w}\" is?\n{opts}\n\nAnswer with the number only.")


def score(best, arr, nodes):
    """道の強さ × 着き の一番大きい物。arr[n] が無い点は 0。"""
    v = [best[n][0] * arr.get(n, 0.0) for n in nodes if n in best]
    return max(v) if v else 0.0
