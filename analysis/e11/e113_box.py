"""E11-3 の箱（E10-1 の部分と全体・病気・地理）。E11-2 と同じ作り方（e112_box.py の数のまま、物の語で5つに分け、種5つの平均）。
python3 e113_box.py E101_RESULTS_DIR e101_items.json DOM out.json
組：一覧なしの鎖（N）の段（意味→次の語の意味、重み＝段の NLI。一種の段と「〜にある」の段の両方）と、着き（点→全体・分類、重み＝着きの NLI、点×全部の分類）。
答え漏れを防ぐ：その組の物の語の、物の語そのものの点からの着きは学習から外す。WordNet も正解も使わない。"""
import json, sys, os, gzip, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e112_box as BXM


def torch_threads():
    import torch; torch.set_num_threads(1)

lgz = lambda d, n: json.load(gzip.open(os.path.join(d, n + ".json.gz"), "rt", encoding="utf-8"))


def main(rdir, items_path, dom, out, only_fold=None):
    torch_threads()
    items = [it for it in json.load(open(items_path))["items"] if it["dom"] == dom]
    res = lgz(rdir, f"E101_res_{dom}")
    ST = {}
    for d, tag, w, k, g, t, e in lgz(rdir, "E101_steps"):
        if d == dom and tag == "N": ST[(w, k, g, t)] = e
    A = np.array(res["arrive"], dtype=np.float32).max(axis=2)
    nid = {tuple(n): i for i, n in enumerate(res["nodes"])}; classes = res["classes"]; cidx = {c: j for j, c in enumerate(classes)}
    ch = {key.split("|", 1)[1]: v for key, v in res["chains"].items() if key.startswith("N|")}
    used = sorted({tuple(n) for nd, lk in ch.values() for n in nd})
    idx = {n: i for i, n in enumerate(used)}; N = len(used) + len(classes)
    links = sorted({(tuple(a), tuple(b), t) for nd, lk in ch.values() for a, b, t in lk if tuple(a) != tuple(b)})
    sp = np.array([[idx[a], idx[b]] for a, b, t in links]); sw = np.array([ST.get((a[0], a[1], b[0], t), 0.0) for a, b, t in links])
    ents = sorted({it["ent"] for it in items}); rng = np.random.default_rng(0)
    fold = {e: f % 5 for f, e in enumerate(rng.permutation(ents))}
    print(dom, "点", N, "段", len(links), "着き", len(used) * len(classes), flush=True)
    roots_of = {}
    for n in used: roots_of.setdefault(n[0], []).append(idx[n])
    out_s = np.zeros((5, len(items)))
    t0 = time.time()
    for f in ([int(only_fold)] if only_fold is not None else range(5)):
        drop = {e for e in ents if fold[e] == f}
        ap, aw = [], []
        for n in used:
            if n[0] in drop: continue
            for c, j in cidx.items():
                ap.append([idx[n], len(used) + j]); aw.append(float(A[nid[n], j]))
        P = np.vstack([sp, np.array(ap)]); W = np.concatenate([sw, np.array(aw)])
        te = [i for i, it in enumerate(items) if fold[it["ent"]] == f]
        for s in range(5):
            m = BXM.train(N, P, W, seed=100 * f + s)
            for i in te:
                x = items[i]["ent"]; roots = roots_of.get(x, [])
                out_s[s, i] = BXM.score(m, np.array(roots), np.full(len(roots), len(used) + cidx[items[i]["cls"]])).max() if roots else 0.0
        print(f"  {dom} 組{f + 1}/5 {time.time() - t0:.0f}秒", flush=True)
    if only_fold is not None:   # 組ごとに別の処理で流した時：その組の問題の値だけ保存（あとで merge でまとめる）
        te = [i for i, it in enumerate(items) if fold[it["ent"]] == int(only_fold)]
        json.dump({"dom": dom, "fold": int(only_fold), "idx": te, "vals": out_s[:, te].round(5).tolist()}, open(out, "w")); return
    json.dump({"dom": dom, "ids": [it["id"] for it in items], "box": out_s.mean(0).round(5).tolist(),
               "seed_spread": out_s.std(0).mean().item()}, open(out, "w"))


def merge(items_path, dom, out, *parts):
    items = [it for it in json.load(open(items_path))["items"] if it["dom"] == dom]
    V = np.full((5, len(items)), np.nan)
    for p in parts:
        d = json.load(open(p)); V[:, d["idx"]] = np.array(d["vals"])
    assert not np.isnan(V).any()
    json.dump({"dom": dom, "ids": [it["id"] for it in items], "box": V.mean(0).round(5).tolist(),
               "seed_spread": V.std(0).mean().item()}, open(out, "w"))


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
