"""E12-2 の 3b：E8n の鎖で、Gemma の意味選びの自信が τ 未満の段は1つに決めず最初の4つを全部たどる（D'）。
python3 e122_chain.py E8n_results.json(.gz) out.json   （E8_DIR 既定は analysis/e8、e111_analyze の決まりを使う）"""
import sys, os, json, gzip
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "e11"))
import e111_analyze as E   # A（e8n_analyze）・judge（20%・物の語で1000回）
A, e6, sig = E.A, E.e6, E.sig
TAUS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
A_MID = 4


def chains_mixed(L, x, choice, prob, tau):
    nodes, links = set(), set()
    front = [(x, k) for k in range(len(L.get(x, [])))]
    nodes.update(front)
    for _ in range(A.DEPTH):
        nxt = []
        for (w, k) in front:
            g = L[w][k]["genus"]
            if not g or g not in L or g == w or not L[g]:
                continue
            p = prob.get((w, k))
            if len(L[g]) == 1:
                js = [0]
            elif p is not None and max(p) >= tau:
                js = [choice.get((w, k), 0)]
            else:
                js = range(min(A_MID, len(L[g])))
            for j in js:
                links.add(((w, k), (g, j)))
                if (g, j) not in nodes:
                    nodes.add((g, j)); nxt.append((g, j))
        front = nxt
    return nodes, links


def main(src, out_p):
    r = json.load(gzip.open(src, "rt")) if src.endswith(".gz") else json.load(open(src))
    L, ST, AR, CH = A.load(r)
    choice = {(w, k): j for w, k, j in r["choice"]["D"]}
    prob = {(w, k): p for w, k, p in r["dprob"]}
    old, new = A.labels(); allit = old + new; no = len(old)
    io = np.arange(no); inw = np.arange(no, len(allit))
    y = np.array([int(x["mine"] >= 1) for x in old]); yn = np.array([int(x["mine"] >= 1) for x in new])
    ents = np.array([x["ent"] for x in old]); en = np.array([x["ent"] for x in new])
    sD = np.array([A.Item(it, "D", *CH[(it["id"], "D")], L, ST, AR).maxprod()[0] for it in allit])
    sA = np.array([A.Item(it, "A", *CH[(it["id"], "A")], L, ST, AR).maxprod()[0] for it in allit])
    S, miss = {}, {}
    for t in TAUS:
        v, m_, tot = [], 0, 0
        for it in allit:
            nd, lk = chains_mixed(L, it["ent"], choice, prob, t)
            m_ += sum(1 for n in nd if n not in ST and n[0] != it["ent"]); tot += len(nd)
            v.append(A.Item(it, "D", [list(n) for n in nd], [[list(a), list(b)] for a, b in lk], L, ST, AR).maxprod()[0])
        S[t] = np.array(v); miss[t] = round(m_ / max(tot, 1), 4)
    a700 = {str(t): e6.aurc(S[t][io], y) for t in TAUS}
    best = min(TAUS, key=lambda t: a700[str(t)])
    out = {"low_conf_share_steps": round(float(np.mean([max(p) < 0.9 for p in prob.values()])), 4),
           "missing_step_values": {str(t): miss[t] for t in TAUS},
           "aurc_700": {"D": e6.aurc(sD[io], y), "A": e6.aurc(sA[io], y), **{f"D'_{t}": v for t, v in a700.items()}},
           "chosen_tau": best,
           "700_ref_D'_vs_D": E.judge(S[best][io], sD[io], y, ents, 41),
           "300": {"D": e6.aurc(sD[inw], yn), "A": e6.aurc(sA[inw], yn), f"D'_{best}": e6.aurc(S[best][inw], yn),
                   "D'_vs_D": E.judge(S[best][inw], sD[inw], yn, en, 42)},
           "check_D_equals_E8n_6D": e6.aurc(sD[io], y)}
    out["flag_効いた"] = bool(out["300"]["D'_vs_D"]["better"])
    json.dump(out, open(out_p, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
