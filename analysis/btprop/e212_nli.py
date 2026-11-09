# E2-12 つながりの判定（Colabのセル）。RUN・HELD・ORIG はセル2で作ったもの。
# 結果: E2-12_nli.json = {"pairs": [[親,子]], "models": {名前: {"fwd": [[含,どちらとも,矛]], "bwd": [...]}}, "gemma_prompt": {...}}
if not USE_MOCK_NLI:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
NAMES = ["entail", "neutral", "contradict"]
nli_json = f"{OUT}/E2-12_nli.json"
res = json.load(open(nli_json)) if os.path.exists(nli_json) else {"pairs": [list(x) for x in RUN], "models": {}}
assert res["pairs"] == [list(x) for x in RUN], "保存済みの組と合わない。E2-12_nli.json を消してやり直す"
def save(): json.dump(res, open(nli_json, "w"), ensure_ascii=False); print("  保存:", nli_json, flush=True)

def mock(pairs, seed, acc):   # 作り物の判定：元の判定を acc の割合で当て、残りはでたらめ
    r = np.random.default_rng(seed); F, B = [], []
    for pq in pairs:
        o = ORIG["factcheckgpt"].get(tuple(pq)) or ORIG["wikibio"][tuple(pq)]
        for k, lab in enumerate(o.split("-")):
            v = r.dirichlet([1, 1, 1])
            if r.random() < acc: v = 0.2 * v; v[NAMES.index(lab)] += 0.8
            (F if k == 0 else B).append(v.tolist())
    return F, B

# ---- NLI専用のモデル ----
def run_nli(mid, prem, hyp, bs=64):
    tok = AutoTokenizer.from_pretrained(mid)
    m = AutoModelForSequenceClassification.from_pretrained(mid, torch_dtype=torch.float32).cuda().eval()
    idx = []
    for name in NAMES:
        hit = [i for i, l in m.config.id2label.items() if l.lower().startswith(name[:6])]
        assert len(hit) == 1, (mid, m.config.id2label); idx.append(int(hit[0]))
    out, t = [], time.time()
    for i in range(0, len(prem), bs):
        enc = tok(prem[i:i + bs], hyp[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        with torch.no_grad(): p = m(**enc).logits.float().softmax(-1)[:, idx]
        out += p.cpu().tolist()
        if (i // bs) % 50 == 0: print(f"  {i + bs}/{len(prem)}  {time.time() - t:.0f}秒", flush=True)
    del m; torch.cuda.empty_cache()
    return out

P = [p for p, c in RUN]; C = [c for p, c in RUN]
for name, mid in NLI_MODELS.items():
    if name in res["models"]: print("済み:", name); continue
    print(name, mid)
    if USE_MOCK_NLI: F, B = mock(RUN, len(res["models"]), [.6, .75, .9][len(res["models"]) % 3])
    else:
        both = run_nli(mid, P + C, C + P); F, B = both[:len(P)], both[len(P):]   # 親→子 と 子→親
    res["models"][name] = {"fwd": F, "bwd": B}; save()

# ---- Gemma-2 2B に3択で聞く（聞き方3つから、評価に使わない200組で1つ選ぶ）----
PROMPTS = {
 "P1_前提と仮説": ("Premise: {a}\nHypothesis: {b}\nDoes the premise entail the hypothesis?\n"
                "A. Yes, the hypothesis must be true.\nB. Cannot tell.\nC. No, the hypothesis must be false.\n"
                "Answer with A, B, or C only.", "ABC"),
 "P2_もし本当なら": ("Statement 1: {a}\nStatement 2: {b}\nIf Statement 1 is true, what about Statement 2?\n"
                "A. It must also be true.\nB. It could be true or false.\nC. It must be false.\n"
                "Answer with A, B, or C only.", "ABC"),
 "P3_順番を逆に": ("Statement 1: {a}\nStatement 2: {b}\nIf Statement 1 is true, what about Statement 2?\n"
                "A. It must be false.\nB. It could be true or false.\nC. It must also be true.\n"
                "Answer with A, B, or C only.", "CBA"),   # A=矛盾, C=含まれる（位置の偏りを見る）
}
GNAME = "Gemma-2 2B(聞く)"
def balanced_agree(F, B, pairs):   # 元の判定との一致（3つの答えごとの当たり率の平均）
    hit = collections.defaultdict(list)
    for f, b, pq in zip(F, B, pairs):
        o = ORIG["wikibio"][tuple(pq)].split("-")
        for v, lab in ((f, o[0]), (b, o[1])): hit[lab].append(NAMES[int(np.argmax(v))] == lab)
    return float(np.mean([np.mean(v) for v in hit.values()])), {k: round(float(np.mean(v)), 3) for k, v in hit.items()}

if GNAME in res["models"]: print("済み:", GNAME)
else:
    if USE_MOCK_NLI:
        def ask(key, a, b):
            n = len(a) // 2; F, B = mock(list(zip(a[:n], b[:n])), 7 + len(key), .7); return F + B
    else:
        from transformers import AutoModelForCausalLM
        from google.colab import userdata
        gtok = AutoTokenizer.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), padding_side="left")
        gm = AutoModelForCausalLM.from_pretrained(GEMMA_ID, token=userdata.get("HF_TOKEN"), torch_dtype=torch.float16, device_map="cuda").eval()
        LET = [gtok.encode(x, add_special_tokens=False)[0] for x in "ABC"]
        def ask(key, a, b, bs=32):
            tmpl, order = PROMPTS[key]
            Q = [gtok.apply_chat_template([{"role": "user", "content": tmpl.format(a=x, b=y)}], tokenize=False, add_generation_prompt=True) for x, y in zip(a, b)]
            out, t = [], time.time()
            for i in range(0, len(Q), bs):
                enc = gtok(Q[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.no_grad(): p = gm(**enc).logits[:, -1, LET].float().softmax(-1).cpu().numpy()
                # 並べ替えて [含まれる, どちらとも, 矛盾] に
                for row in p: out.append([float(row[order.index(L)]) for L in "ABC"])
                if (i // bs) % 50 == 0: print(f"  {i + bs}/{len(Q)}  {time.time() - t:.0f}秒", flush=True)
            return out
    hp = [p for p, c in HELD]; hc = [c for p, c in HELD]
    score = {}
    for key in PROMPTS:
        both = ask(key, hp + hc, hc + hp)
        score[key] = balanced_agree(both[:200], both[200:], HELD); print(key, score[key])
    best = max(score, key=lambda k: score[k][0]); print("選んだ聞き方:", best)
    both = ask(best, P + C, C + P)
    res["models"][GNAME] = {"fwd": both[:len(P)], "bwd": both[len(P):]}
    res["gemma_prompt"] = {"chosen": best, "held_out_scores": score}
    save()

# ---- ざっと確かめる：元の判定とのずれ ----
for name, v in res["models"].items():
    agree = n = 0
    for f, b, pq in zip(v["fwd"], v["bwd"], res["pairs"]):
        o = ORIG["factcheckgpt"].get(tuple(pq)) or ORIG["wikibio"][tuple(pq)]
        agree += f"{NAMES[int(np.argmax(f))]}-{NAMES[int(np.argmax(b))]}" == o; n += 1
    print(f"{name:24s} 元の判定と同じ {agree / n:.1%}")
