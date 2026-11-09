# E3 本番（Colabのセル本体）。e3_pilot2.py の道具（聞き方・parse・fill など）を使う。
# 途中で切れても、もう一度「すべてのセルを実行」で続きから動く（段階ごと・まとまりごとに保存）。
# 結果: {OUT}/E3_main.json = {"meta":..., "items":[{id, claim, label, num_hops, sents, paras, verdict, decomp9b, nli}]}
import os, json, time

def pick_main(data, exclude_ids, per_group=None, seed=1):
    import random, collections
    rest = [e for e in data if e["id"] not in exclude_ids]
    if per_group is None: return rest
    r = random.Random(seed); g = collections.defaultdict(list)
    for e in rest: g[(e["num_hops"], e["label"])].append(e)
    out = []
    for k in sorted(g): out += r.sample(g[k], per_group)
    return out

def stage_verdict(items, p_true, save, chunk=256, log=print):
    todo = [it for it in items if "verdict" not in it]
    for s in range(0, len(todo), chunk):
        part = todo[s:s + chunk]
        pt = p_true([VERDICT_T.format(ev=ev_text(it), claim=it["claim"]) for it in part])
        for it, p in zip(part, pt): it["verdict"] = {"p_true": float(p)}
        save(); log(f"答えと自信 {s + len(part)}/{len(todo)}")

def stage_decomp(items, gen, save, chunk=256, log=print):
    todo = [it for it in items if "decomp9b" not in it]
    for s in range(0, len(todo), chunk):
        part = todo[s:s + chunk]
        tb = gen([DECOMP_B.format(claim=it["claim"]) for it in part])
        for it, b in zip(part, tb):
            h, f = parse_B(b, it["claim"])
            # 変更2-2：隠れた人や物の説明も確かめる部分の主張にする
            f2 = f + [f"{k} is {v}." for k, v in sorted(h.items())]
            it["decomp9b"] = {"B_raw": b, "hidden": h, "facts": f2, "n_orig_facts": len(f)}
        save(); log(f"分ける {s + len(part)}/{len(todo)}")

def stage_fill(items, gen, save, n_samples=2, max_cands=3, max_assign=4, chunk=256, log=print):
    """隠れた点を順に埋める。まとまりごとに、全部の主張の「次の点」をまとめて聞く"""
    for it in items:
        d = it["decomp9b"]
        if "assigns" not in d:
            d.setdefault("cands", {}); d.setdefault("unknown", {})
            d["_todo"] = sorted(d["hidden"]); d["_assigns"] = [{}]
    def pending(): return [it for it in items if it["decomp9b"].get("_todo")]
    rnd = 0
    while pending():
        rnd += 1; P = pending()
        for s in range(0, len(P), chunk):
            part = P[s:s + chunk]; reqs = []
            for it in part:
                d = it["decomp9b"]; k = d["_todo"][0]
                for a in d["_assigns"]:
                    reqs.append((it, k, a, FILL_T.format(ev=ev_text(it), desc=fill(d["hidden"][k], a))))
            prompts = [q[3] for q in reqs]
            g1 = gen(prompts, max_new=24)
            gs = gen([p for p in prompts for _ in range(n_samples)], sample=True, max_new=24)
            new_assigns = {}
            for qi, (it, k, a, _) in enumerate(reqs):
                outs = [g1[qi]] + gs[qi * n_samples:(qi + 1) * n_samples]
                c, n_unk = [], 0
                for o in outs:
                    o2 = clean_name(o)
                    if not o2: n_unk += 1
                    elif o2 not in c: c.append(o2)
                key = json.dumps(a, ensure_ascii=False); d = it["decomp9b"]
                d["cands"].setdefault(k, {})[key] = c[:max_cands]
                d["unknown"].setdefault(k, {})[key] = n_unk / len(outs)
                new_assigns.setdefault(id(it), []).extend(dict(a, **{k: x}) for x in c[:max_cands])
            for it in part:
                d = it["decomp9b"]; d["_todo"].pop(0)
                d["_assigns"] = new_assigns.get(id(it), [])[:max_assign] or [{}]
            save(); log(f"隠れた点 {rnd}回目 {s + len(part)}/{len(P)}")
    for it in items:
        d = it["decomp9b"]
        if "_assigns" in d: d["assigns"] = d.pop("_assigns"); d.pop("_todo", None)
    save()

def stage_nli(items, nli, save, chunk=128, log=print):
    """変更3：元の主張は文ごと（②数える用）と段落ごと、部分の主張は段落ごと"""
    todo = [it for it in items if "nli" not in it]
    for s in range(0, len(todo), chunk):
        part = todo[s:s + chunk]; P, H, where = [], [], []
        for i, it in enumerate(part):
            d = it["decomp9b"]
            for j, x in enumerate(it["sents"]):
                P.append(x); H.append(it["claim"]); where.append((i, "claim_sents", "_", j))
            for j, x in enumerate(it["paras"]):
                P.append(x); H.append(it["claim"]); where.append((i, "claim_paras", "_", j))
                for ai, a in enumerate(d["assigns"]):
                    for fi, f in enumerate(d["facts"]):
                        P.append(x); H.append(fill(f, a)); where.append((i, "facts_paras", f"{fi}|{ai}", j))
        out = nli(P, H)
        for it in part: it["nli"] = {}
        for (i, kind, key, j), v in zip(where, out):
            it = part[i]; n = len(it["sents"] if kind == "claim_sents" else it["paras"])
            it["nli"].setdefault(kind, {}).setdefault(key, [None] * n)[j] = [round(x, 4) for x in v]
        save(); log(f"NLI {s + len(part)}/{len(todo)}（この回の組 {len(P)}）")

def stage_verdict_drop(items, p_true, save, seed=7, chunk=256, log=print):
    """追加の確かめ：「支える」主張から記事の段落を1つ抜いた資料で、もう一度答えと自信（Gemma 2B）。
    抜いた段落の番号を記録し、NLIの方法は手元で同じ段落の判定を外して計算する。"""
    import random
    r = random.Random(seed)
    for it in items:
        if it["label"] == "supports" and len(it["paras"]) >= 2 and "verdict_drop" not in it:
            it["_drop"] = r.randrange(len(it["paras"]))
    todo = [it for it in items if "_drop" in it and "verdict_drop" not in it]
    for s in range(0, len(todo), chunk):
        part = todo[s:s + chunk]
        ps = []
        for it in part:
            keep = [p for j, p in enumerate(it["paras"]) if j != it["_drop"]]
            ev = "\n".join(x for p in keep for x in split_sents(p))
            ps.append(VERDICT_T.format(ev=ev, claim=it["claim"]))
        pt = p_true(ps)
        for it, p in zip(part, pt): it["verdict_drop"] = {"drop_para": it.pop("_drop"), "p_true": float(p)}
        save(); log(f"1本抜いた答え {s + len(part)}/{len(todo)}")
