"""Audit (2026-11-11): make a copy of the repository where only the learned baselines use gradient boosting.
Usage (from the repo root): python analysis/audit_gb/make_gb_copy.py <empty_dir>
Then run the usual commands (README, "Reproduce") inside <empty_dir>. Network weight fitting (C=1e4) and
one-feature calibration are left as logistic regression."""
import os, sys, shutil
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DST = os.path.abspath(sys.argv[1]); shutil.copytree(SRC, DST, dirs_exist_ok=True)
GB = os.path.join(DST, "analysis", "audit_gb")
TARGETS = {"e27.py": [55], "e28.py": [88], "e4/e42.py": [210], "e4/relabel/e42_relabel.py": [33],
           "e5/e5.py": [110], "e5/e52.py": [84], "e6/e6_analyze.py": [72], "e6/e7_analyze.py": [161, 162],
           "e8/e8_analyze.py": [217, 224, 242, 243, 246]}
for f, lines in TARGETS.items():
    p = os.path.join(DST, "analysis", f); L = open(p).read().split("\n")
    for n in lines:
        assert "LogisticRegression(" in L[n - 1], (f, n); L[n - 1] = L[n - 1].replace("LogisticRegression(", "GBLR(")
    i = next(i for i, l in enumerate(L) if "from sklearn.linear_model import LogisticRegression" in l)
    L.insert(i + 1, f"import sys as _s; _s.path.insert(0, {GB!r}); from gblr import GBLR")
    open(p, "w").write("\n".join(L)); print("patched", f, lines)
