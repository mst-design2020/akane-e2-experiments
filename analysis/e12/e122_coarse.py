"""E12-2 の1（粗い意味で数え直す）と 3a（自信で控える、E12 の中）。
python3 e122_coarse.py items_all.json local.json E12_results.json.gz sensekey2csi_complete.tsv out.json
sensekey2csi_complete.tsv は github.com/SapienzaNLP/csi_code（commit ca30c2f）の data/（CC BY-NC-SA、ここには入れない）。"""
import sys, json, gzip, collections
import numpy as np
from nltk.corpus import wordnet as wn
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from e12_analyze import hit, boot_diff
import gzip as _gz
jl = lambda p: __import__("json").load(_gz.open(p, "rt") if str(p).endswith(".gz") else open(p))   # .json と .json.gz の両方


def labels_of(cands, lemma, csi):
    out_c, out_l = [], []
    for c in cands:
        s = wn.synset(c)
        key = [l.key() for l in s.lemmas() if l.name().lower() == lemma]
        k = key[0] if key else s.lemmas()[0].key()
        out_c.append(csi.get(k)); out_l.append(s.lexname())
    return out_c, out_l


def main(items_p, local_p, res_p, csi_p, out_p):
    it = jl(items_p)["items"]; L = jl(local_p); R = json.load(gzip.open(res_p, "rt"))
    csi = dict(l.rstrip("\n").split("\t")[:2] for l in open(csi_p) if "\t" in l)
    rng = np.random.default_rng(0)
    M = {"Gemma": R["gemma"], "近さ": R["sim"], "Lesk": [l["lesk"] for l in L]}
    out = {}
    head = np.array([x["kind"] == "head" for x in it])
    for tag in ("CSI", "lexname"):
        rows = collections.defaultdict(list); allsame = []; miss = 0; cov = []
        for i, x in enumerate(it):
            lc, ll = labels_of(x["cands"], x["head"], csi)
            lab = lc if tag == "CSI" else ll
            cov.append(not any(v is None for v in lab)); miss += (not cov[-1])
            g = lab[x["gold"]]
            same = np.array([v == g and v is not None for v in lab], float); same[x["gold"]] = 1.0
            allsame.append(bool(same.all()))
            rows["でたらめ"].append(same.mean())
            rows["最初の意味"].append(same[0])
            for m, sc in M.items():
                s = np.asarray(sc[i], float); top = np.flatnonzero(s >= s.max() - 1e-12)
                rows[m].append(same[top].mean())
        A = {k: np.array(v) for k, v in rows.items()}; allsame = np.array(allsame); cov = np.array(cov)
        fine = np.array([hit(R["gemma"][i], x["gold"]) for i, x in enumerate(it)])
        o = {"表に無い意味がある問題": miss, "候補が全部同じまとまり(頭の語)": round(float(allsame[head].mean()), 4)}
        for nm, m in (("頭の語", head & cov), ("頭の語・全部同じを除く", head & cov & ~allsame)):  # 表に無い意味がある問題は外す
            hs = [x["head"] for x, k in zip(it, m) if k]
            d, ci = boot_diff(A["Gemma"][m], A["最初の意味"][m], hs)
            err = m & (fine < 1)
            o[nm] = {"n": int(m.sum()), **{k: round(float(v[m].mean()), 4) for k, v in A.items()},
                     "Gemma − 最初の意味": {"diff": round(d, 4), "ci95": [round(c, 4) for c in ci], "差あり": bool(abs(d) >= .05 and (ci[0] > 0 or ci[1] < 0))},
                     "Gemmaの外れのうち同じまとまり": round(float(((1 - fine[err]) * A["Gemma"][err]).sum() / (1 - fine[err]).sum()), 4)}
        out[tag] = o
    # 3a 自信で控える：自信 ≥ τ なら Gemma、未満なら最初の意味
    TAUS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    hi = np.flatnonzero(head); heads = sorted({it[i]["head"] for i in hi})
    perm = rng.permutation(len(heads)); half = {heads[j]: (k % 2) for k, j in enumerate(perm)}
    conf = np.array([max(p) for p in R["gemma"]])
    G = np.array([hit(p, x["gold"]) for p, x in zip(R["gemma"], it)]); F = np.array([float(x["gold"] == 0) for x in it])
    sel = lambda t: np.where(conf >= t, G, F)
    a = np.array([i for i in hi if half[it[i]["head"]] == 0]); b = np.array([i for i in hi if half[it[i]["head"]] == 1])
    tau = max(TAUS, key=lambda t: sel(t)[a].mean())
    d, ci = boot_diff(sel(tau)[b], G[b], [it[i]["head"] for i in b])
    out["3a 自信で控える"] = {"選んだτ(半分A)": tau, "半分Aでの値": {str(t): round(float(sel(t)[a].mean()), 4) for t in TAUS},
                         "半分B": {"n": int(len(b)), "Gemma": round(float(G[b].mean()), 4), "合わせ技": round(float(sel(tau)[b].mean()), 4),
                                 "合わせ技 − Gemma": {"diff": round(d, 4), "ci95": [round(c, 4) for c in ci], "差あり": bool(abs(d) >= .05 and (ci[0] > 0 or ci[1] < 0))}},
                         "自信τ以上だけ答える(頭の語全部)": {str(t): {"答える割合": round(float((conf[hi] >= t).mean()), 4),
                                                         "正しさ": round(float(G[hi][conf[hi] >= t].mean()), 4)} for t in TAUS}}
    json.dump(out, open(out_p, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:6])
