# 分け方ごとに別々に回した結果をまとめて判定する。使い方: python3 merge28.py tag  （results/E28_<tag>_s*.json を読む）
import os, sys, glob, json, ast
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "e24.py")).read()
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name == "judge": exec(compile(ast.Module([node], []), "e24", "exec"))
TAG = sys.argv[1]; B, A = [], []; RS = None
for f in sorted(glob.glob(os.path.join(HERE, "..", "results", "E28_runs", f"E28_{TAG}_s*.json"))):
    d = json.load(open(f)); B += d["res"]["B"]; A += d["res"]["A"]; RS = d["rules"]
print("ルール:", json.dumps(RS, ensure_ascii=False, indent=1)); print("回数", len(B), len(A))
out = {"rules": RS, "res": {"B": B, "A": A}, "judge": {}}
def J(name, R, a, b, sub="全体"):
    s = judge(R, a, b, sub); out["judge"][name] = s; print(name + ":", s)
J("問い1 BP対数える", B, "BP", "数える"); J("記録 揺れ対数える", B, "揺れ", "数える")
J("記録 BP対④", B, "BP", "④組み合わせ"); J("記録 BP対③", B, "BP", "③近所の一貫性"); J("記録 BP対①", B, "BP", "口の自信")
J("記録 違反ゼロでBP対数える", B, "BP", "数える", "どのルールも破っていない主張")
AB = [{"auc": {"全体": {"BP(選んだ組)": a["auc"]["全体"]["BP"], "BP(全部の組)": b["auc"]["全体"]["BP"]}}} for a, b in zip(A, B)]
J("問い3 BP(選んだ組)対BP(全部の組)", AB, "BP(選んだ組)", "BP(全部の組)")
J("記録 選んだ組でBP対数える", A, "BP", "数える")
for k, R in (("全部の組", B), ("選んだ組", A)):
    m = {kk: round(float(np.nanmean([r["auc"]["全体"][kk] for r in R])), 3) for kk in R[0]["auc"]["全体"] if isinstance(R[0]["auc"]["全体"][kk], float)}
    acc = {kk: round(float(np.mean([r["acc"][kk] for r in R])), 3) for kk in R[0]["acc"]}
    rule = {kk: round(float(np.nanmean([r["rule"][kk] for r in R])), 3) for kk in R[0]["rule"]}
    out[k] = {"auc": m, "acc": acc, "rule": rule, "収束": sum(r["bp_converged"] for r in R)}
    print(k, "AUROC平均", m, "\n  正解率", acc, "\n  ルール", rule, " BP収束", out[k]["収束"], "/", len(R))
json.dump(out, open(os.path.join(HERE, "..", "results", f"E28_{TAG}.json"), "w"), ensure_ascii=False, indent=1)
