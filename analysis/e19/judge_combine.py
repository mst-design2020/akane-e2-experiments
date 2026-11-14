"""ChatGPT・Gemini の判定（YES の行番号）を pairs.json に戻し、Claude・WordNet と比べ、e19_analyze を流し直す。
Run from analysis/e19: python judge_combine.py
Reads data/E19/ (c19.json, judge/), writes results/E19/judge/."""
import json, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e19_analyze as A
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "..", "data", "E19")
O = os.path.join(HERE, "..", "..", "results", "E19", "judge")
os.makedirs(O, exist_ok=True)
pairs = json.load(open(f"{D}/judge/pairs.json"))
def load(name):
    yes = set()
    for b in range(10):
        t = open(f"{D}/judge/{name}_{b:02d}.txt").read()
        yes |= {int(x) for x in re.findall(r"\d+", t)}
    return {p: int(i + 1 in yes) for i, p in enumerate(pairs)}
J = {"claude": json.load(open(f"{D}/judge/cs_claude_all.json")), "chatgpt": load("chatgpt"), "gemini": load("gemini")}
J["vote"] = {p: int(J["claude"][p] + J["chatgpt"][p] + J["gemini"][p] >= 2) for p in pairs}
c = json.load(open(f"{D}/c19.json"))
wn = {f"{it[1]}|{it[2]}": g for it, g in zip(c["items"], c["gold"])}
agree = {}
names = ["claude", "chatgpt", "gemini", "vote"]
for a in names:
    agree[a] = {"yes": sum(J[a].values()), "wordnet": round(sum(J[a][p] == wn[p] for p in pairs) / len(pairs), 3)}
    for b in names:
        if a < b: agree[f"{a}~{b}"] = round(sum(J[a][p] == J[b][p] for p in pairs) / len(pairs), 3)
print(agree)
out = {"agree": agree}
for n in names:
    json.dump(J[n], open(f"{O}/cs_{n}.json", "w"))
    r = A.main(f"{D}/c19.json", f"{O}/out_{n}.json", f"{O}/cs_{n}.json")
    out[n] = {bk: {"ba": {k: v["ba"] for k, v in r["common_sense"]["by_bucket"][bk].items() if isinstance(v, dict)},
                   "flags": {k: [v["diff"], v["lo"], v["hi"], v["flag"]] for k, v in r["common_sense"]["flags"][bk].items()}}
              for bk in ("common", "rare")}
json.dump(out, open(f"{O}/summary_judges.json", "w"), ensure_ascii=False, indent=1)
