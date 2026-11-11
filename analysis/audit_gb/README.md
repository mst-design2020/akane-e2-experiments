# Audit (2026-11-11): are the published comparisons robust to a stronger learner?

Every "features + learning" baseline in E2 to E8 was logistic regression. Here only those lines are swapped for
gradient boosting (`gblr.py`: `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, min_samples_leaf=20, random_state=0)`,
one fixed setting, not tuned). Network weight fitting (C=1e4) and one-feature calibration stay logistic regression.

Reproduce: `python analysis/audit_gb/make_gb_copy.py <empty_dir>`, then run the usual commands (main README, "Reproduce") inside `<empty_dir>`.
With four CPU cores, set `OMP_NUM_THREADS=1` when running several at once.

| Experiment | Published | Logistic regression | Gradient boosting | Verdict (rule fixed beforehand) |
|---|---|---|---|---|
| E2-7 taxonomy (AUROC) | BP beats all other methods | BP 0.844 vs ensemble 0.803 (10/10) | 0.844 vs 0.825 (+0.019, 8/10) | **tie** |
| E2-7 geography (AUROC) | BP ties the ensemble | 0.817 vs 0.804 | 0.817 vs 0.851 (−0.034, BP higher 2/10) | **BP loses** |
| E2-8 disease (AUROC) | (ensemble recorded only) | BP 0.764 vs 0.748 | 0.764 vs 0.721 | unchanged (GB weaker) |
| E4-2 noun claims (AURC) | cautious net ties learning | 0.079 vs 0.086 | 0.079 vs 0.038 | **network loses** (WordNet labels) |
| E4-2, 700 common-sense items | net 0.039 vs 0.056, tie | — | 0.039 vs 0.049 (net lower 5/5) | tie |
| E5 cost of Gemma's rules, learner | 0.044 | 0.087 − 0.043 | 0.041 − 0.020 = 0.021 | smaller, still a cost |
| E5-2 | no cost | ≈0.025 | ≈0.028 | unchanged |
| E6 (S2) | network loses; NLI alone lowest (reference) | net 0.098, learn 0.079, NLI 0.070 | learn 0.067 | network still loses; NLI ≈ learner |
| E7 ⑧ vs E6 features | ⑧ lowest, 0.048 vs 0.079 (5/5) | gap 0.027–0.037 | 0.051 vs 0.067, gap 0.010–0.018 (5/5) | **tie** |
| E8 ⑧D vs E6 features | 0.037 vs 0.079 | gap ≈0.04 | 0.031 vs 0.071, gap 0.033–0.047 (5/5) | holds |
| E8, 300 new items | gain over one-step NLI not shown | ⑧D 0.039 | ⑧D 0.023; gap to one-step 0.034 (95% 0.014–0.058) | reference only (seen after the results) |

Why the tree learner does so well in E4-2: the candidates were built so that true categories are linked in the hierarchy and G is
the false category Gemma believed most, and WordNet uses narrow senses. A tree model picks these regularities up from the features.
The 700 items re-judged by common sense show a much smaller gain. E4-3, E5 and the E9 work (not yet published) use the same candidates.
E6 to E8 use a sample of them, but their features are only Gemma's confidence and NLI scores, so the effect there is small.

Not re-run: E3 (network already loses to logistic regression), E4-1 (logistic regression already best), E2-9 to E2-12 (compared with BTProp, not a learner).
