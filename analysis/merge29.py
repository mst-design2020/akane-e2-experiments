import os, glob, json, ast
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
for node in ast.parse(open(os.path.join(HERE, "e24.py")).read()).body:
    if isinstance(node, ast.FunctionDef) and node.name == "judge": exec(compile(ast.Module([node], []), "e24", "exec"))
import sys
SUF = sys.argv[1] if len(sys.argv) > 1 else ""
OUT = {}
for dom in ["dis", "wn"]:
    R = []
    for f in sorted(glob.glob(os.path.join(HERE, "..", "results", "E29_runs", f"E29{SUF}_{dom}_s*.json"))): R += json.load(open(f))
    levels = sorted({r["f"] for r in R}); rows = {}
    for f in levels:
        S = [r for r in R if r["f"] == f]
        m = {k: float(np.mean([r["auc"]["全体"][k] for r in S])) for k in ["口の自信", "数える", "BP", "揺れ", "疑わないBP"]}
        m["差BP-数える"] = m["BP"] - m["数える"]; m["回数"] = len(S); m["外れ率"] = S[0]["wrong_rate"]
        m["判定"] = judge(S, "BP", "数える"); m["揺れ判定"] = judge(S, "揺れ", "数える"); m["疑う対疑わない"] = judge(S, "BP", "疑わないBP")
        rows[f] = m
        print(f"[{dom}{SUF}] 外れ{f:.1f}（実{m['外れ率']:.2f}, {len(S)}回） 口{m['口の自信']:.3f} 数える{m['数える']:.3f} BP{m['BP']:.3f} 揺れ{m['揺れ']:.3f} 疑わないBP{m['疑わないBP']:.3f} 差{m['差BP-数える']:+.3f} | {m['判定']}")
    v = {}
    if all(f in rows for f in [0.0, 0.1, 0.2, 0.3]):
        g = [rows[f]["差BP-数える"] for f in [0.0, 0.1, 0.2, 0.3]]; d = g[3] - g[0]; rho = spearmanr([0, 1, 2, 3], g)[0]
        v = {"差の変化(3割-0割)": d, "順位相関": rho,
             "判定": "外れが多いほど効く" if d >= 0.02 and rho >= 0.8 else ("外れが多いほど効きにくい" if d <= -0.02 else "外れの量で差は変わらない（はっきりしない）")}
        print(f"[{dom}] 判定: {v}")
    OUT[dom] = {"rows": {str(k): x for k, x in rows.items()}, "verdict": v}
json.dump(OUT, open(os.path.join(HERE, "..", "results", f"E29{SUF}_summary.json"), "w"), ensure_ascii=False, indent=1)
