"""Audit (2026-11-11): a tree-based learner (gradient boosting) with the same calls as LogisticRegression."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
class GBLR:
    def __init__(self, **kw):
        self.m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, min_samples_leaf=20, random_state=0)
    def fit(self, X, y):
        X = np.asarray(X); self.m.fit(X, y); self.coef_ = np.zeros((1, X.shape[1])); self.intercept_ = np.zeros(1); return self
    def predict_proba(self, X): return self.m.predict_proba(np.asarray(X))
    def decision_function(self, X):
        p = np.clip(self.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
    def predict(self, X): return self.m.predict(np.asarray(X))
