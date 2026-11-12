"""E11-2 の中身：E8n の方法D の鎖を、3段から6段まで伸ばす（計画 plans/E11-2_deep_chain_and_box.md）。手書きの語の一覧はない。
e8n_core（genus_candidates, choice_prompt, premise, hyp, MAX_DEFS_MID）の後に置いて使う。
NLI と Gemma は呼び出しの関数として外から渡す（Colab ではモデル、手元の確かめでは作り物）。"""

MAX_DEPTH = 6


def follow(L, choice, x, maxd):
    """物の語 x から方法D の鎖を maxd 段まで。点 → 一番浅い段の数、と段の組。E8n の chain_nodes(method="D") と同じたどり方。"""
    depth = {(x, k): 0 for k in range(len(L.get(x, [])))}
    links, front = set(), list(depth)
    for d in range(maxd):
        nxt = []
        for (w, k) in front:
            g = L[w][k].get("genus")
            if not g or g not in L or g == w or not L[g]:
                continue
            j = choice.get((w, k), 0) if len(L[g]) > 1 else 0
            links.add(((w, k), (g, j)))
            if (g, j) not in depth:
                depth[(g, j)] = d + 1
                nxt.append((g, j))
        front = nxt
    return depth, links


def unexpanded(L, w, k):
    """頭の名詞の候補のうち、まだ語の意味の一覧に入っていない物がある（E8n は3段目の語の候補を読んでいない）。"""
    return any(c != w and c not in L for c in L[w][k]["cands"])


def extend(items, L, choice, ST, AR, WKD, nlp, nli, gemma, maxd=MAX_DEPTH, log=print):
    """1段ずつ伸ばす。各段で：
    1. 鎖の先の点のうち、候補が読まれていない物は、候補の語の意味を WKD から読み、全部の候補を NLI で測って頭の名詞を決め直す。
    2. 頭の名詞が決まっていて段の値がない点は、NLI で段の値を取る。
    3. 頭の名詞に意味が2つ以上ある点は、Gemma に意味を選ばせる。
    4. 新しく着いた点について、その問題の分類への着きを NLI で取る。
    nli(pairs: [(w,k,c)]) → [(含意, 矛盾)]、gemma(jobs: [(w,k)]) → [番号]。L・choice・ST・AR を書き足して返す。"""
    added = {"words": 0, "genus_pairs": 0, "steps": 0, "choices": 0, "arrive": 0}
    for lev in range(maxd):
        fr = set()
        for it in items:
            dep, _ = follow(L, choice, it["ent"], lev)
            fr |= {n for n, d in dep.items() if d == lev}
        # 1. 候補を読む → 頭の名詞を決め直す
        todo = sorted(n for n in fr if unexpanded(L, *n))
        for w, k in todo:
            for c in L[w][k]["cands"]:
                if c != w and c not in L:
                    L[c] = [{"def": d, "labels": lb, "cands": genus_candidates(d, nlp), "genus": None}
                            for d, lb in WKD.get(c, [])[:MAX_DEFS_MID]]
                    added["words"] += 1
        pairs = sorted({(w, k, c) for w, k in todo for c in L[w][k]["cands"] if c != w and L.get(c)})
        sc = dict(zip(pairs, nli(pairs))) if pairs else {}
        added["genus_pairs"] += len(pairs)
        for w, k in todo:
            cs = [c for c in L[w][k]["cands"] if (w, k, c) in sc]
            L[w][k]["genus"] = max(cs, key=lambda c: sc[(w, k, c)][0]) if cs else None
            if cs:
                ST[(w, k)] = sc[(w, k, L[w][k]["genus"])]
        # 2. 段の値
        need = sorted((w, k) for w, k in fr if L[w][k].get("genus") and (w, k) not in ST and L.get(L[w][k]["genus"]))
        if need:
            for n, v in zip(need, nli([(w, k, L[w][k]["genus"]) for w, k in need])):
                ST[n] = v
            added["steps"] += len(need)
        # 3. 意味選び
        jobs = sorted((w, k) for w, k in fr if L[w][k].get("genus") and len(L.get(L[w][k]["genus"], [])) > 1 and (w, k) not in choice)
        if jobs:
            for n, v in zip(jobs, gemma(jobs)):
                choice[n] = int(v)
            added["choices"] += len(jobs)
        # 4. 着き
        arr = set()
        for it in items:
            dep, _ = follow(L, choice, it["ent"], lev + 1)
            arr |= {(w, k, it["catw"]) for (w, k), d in dep.items() if (w, k, it["catw"]) not in AR}
        arr = sorted(arr)
        if arr:
            for n, v in zip(arr, nli(arr)):
                AR[n] = v
            added["arrive"] += len(arr)
        log(f"段{lev + 1}: 先の点 {len(fr)}  候補を読んだ点 {len(todo)}  新しい語 {added['words']}  段 {len(need)}  意味選び {len(jobs)}  着き {len(arr)}")
    return added


def chains(items, L, choice, maxd=MAX_DEPTH):
    """問題ごとの点（語, 意味, 段の数）と段の組。分析は段の数で切って 3〜6 段を作る。"""
    out = {}
    for it in items:
        dep, links = follow(L, choice, it["ent"], maxd)
        out[it["id"]] = ([[w, k, d] for (w, k), d in sorted(dep.items())], sorted([list(a), list(b)] for a, b in links))
    return out
