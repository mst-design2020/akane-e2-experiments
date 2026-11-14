"""ChatGPT・Gemini の判定（YES の行番号）を pairs.json に戻し、Claude・正解表と比べ、e20_analyze を流し直す。
Run from analysis/e20: python judge_combine.py
Reads data/E20/ (c20.json, judge/), writes results/E20/judge/."""
import json, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e20_analyze as A
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "..", "data", "E20")
O = os.path.join(HERE, "..", "..", "results", "E20", "judge")
os.makedirs(O, exist_ok=True)
pairs = json.load(open(f"{D}/judge/pairs.json"))
def load(name):
    yes = set()
    for b in range(5):
        yes |= {int(x) for x in re.findall(r"\d+", open(f"{D}/judge/{name}_{b:02d}.txt").read())}
    return {p: int(i + 1 in yes) for i, p in enumerate(pairs)}
J = {"claude": json.load(open(f"{D}/judge/cs_claude.json")), "chatgpt": load("chatgpt"), "gemini": load("gemini")}
J["vote"] = {p: int(J["claude"][p] + J["chatgpt"][p] + J["gemini"][p] >= 2) for p in pairs}
c = json.load(open(f"{D}/c20.json"))
gold = {f"{it[0]}|{it[2]}|{it[3]}": g for it, g in zip(c["items"], c["gold"])}
names = ["claude", "chatgpt", "gemini", "vote"]
agree = {"gold_yes": sum(gold[p] for p in pairs)}
for a in names:
    agree[a] = {"yes": sum(J[a].values()), "gold": round(sum(J[a][p] == gold[p] for p in pairs) / len(pairs), 3)}
    for b in names:
        if a < b: agree[f"{a}~{b}"] = round(sum(J[a][p] == J[b][p] for p in pairs) / len(pairs), 3)
print(agree)
out = {"agree": agree}
for n in names:
    json.dump(J[n], open(f"{O}/cs_{n}.json", "w"))
    r = A.main(f"{D}/c20.json", f"{O}/out_{n}.json", f"{O}/cs_{n}.json")
    out[n] = {k: v for k, v in r["common_sense"]["flags"].items()} if "flags" in r["common_sense"] else r["common_sense"]
json.dump(out, open(f"{O}/summary_judges.json", "w"), ensure_ascii=False, indent=1)
