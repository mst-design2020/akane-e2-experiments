# AKANE Architecture E2: Finding an LLM's mistakes from how its answers connect

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24045800.svg)](https://doi.org/10.5281/zenodo.24045800)

English | [日本語](README_ja.md)

**In short:** A small LLM writes its own consistency rules, some of them wrong. A network then checks the LLM's answers against those rules while also doubting each rule, using belief propagation. This is close to BTProp (Hou et al., NAACL 2025). The difference is that the rules come from the LLM itself and the network estimates which rules to trust, without labels. In E2-10 to E2-12 we ran both on BTProp's own released trees. With the original links BTProp is better or tied. When links are flipped at random (E2-11) the rule-doubting network holds up better, but when the links are made by real models (three NLI models and Gemma, E2-12) BTProp barely degrades and the rule-doubting network is never better. We do not claim our method beats BTProp. E2 is closed here (summary in `plans/E2_summary.md`, Japanese). E3 asked whether a network over retrieved evidence helps the model abstain; it tied with the model's own confidence and lost to a simple learned combination of NLI scores (see E3 below). E4 joined thousands of the model's answers about WordNet categories into one large network with many loops: it did not beat a simple learned combination of features, and BP and the p-bit method never differed in accuracy (see E4 below).

Code, data, and results for a set of small experiments. A small language model (Gemma-2 2B) answers many true/false statements. The statements are linked by rules, and we try to find which of the model's answers are wrong by looking at the whole network of answers.
Every number in the articles below comes from files in this repository.

- Article 1 (Japanese): E2-2 and E2-3, rules written by a person. https://note.com/mst_design2020/n/n28ced8c9549d
- Article 2 (Japanese): E2-4 to E2-7, rules written by the model itself. https://note.com/mst_design2020/n/n5d7de09a97f2
- Article 3 (Japanese): E2-8 and E2-9, a third domain and more wrong rules. https://note.com/mst_design2020/n/n0cddc2e4f209
- Article 4 (Japanese): E2-10 and E2-11, the same test as BTProp. https://note.com/mst_design2020/n/ne9eea6736480

Author: Masaya Kato (M.S.T. Design, Saiki, Oita, Japan) https://mst-design.jp/ · ORCID https://orcid.org/0009-0001-6803-654X · DOI https://doi.org/10.5281/zenodo.24045800

## AKANE Architecture in short

The idea is to use a language model as the "mouth" and to place the checking outside the model. The model answers first. Then an outside system checks whether the answers are consistent with a large memory and with rules. In the long run, we want to hand this checking computation to physical parts such as p-bits (probabilistic bits that flip between 0 and 1).
The experiments here test the smallest version of that outside checker. More about the overall research: https://mst-design.jp/research/ (Japanese)

## What the experiments do

- Example statements: "Paris is a city in France." "France is a country in Europe." "Paris is located in Europe."
- Example rules: "If A is a kind of B and B is a kind of C, then A is a kind of C." "Nothing is both a plant and an animal."
- Signals compared for flagging wrong answers: the model's own confidence; counting rule violations; MaxSAT; loopy belief propagation (BP); a software simulation of p-bits (Gibbs sampling, called "揺れ" in the code). E2-7 adds paraphrase inconsistency, neighbor consistency (a simplified form of Xu et al. 2026), and a logistic-regression ensemble of these signals.
- Metric: AUROC for detecting the model's wrong answers. 20% of the words or cities are used for tuning and 80% for evaluation. There are 5 splits × 2 p-bit seeds, so 10 runs.
- Decision rule, fixed before each experiment: A "wins" over B if the mean AUROC difference is at least 0.02 and A is higher in at least 8 of 10 runs. A difference below 0.02 is a "tie".
- E2-10 and E2-11 differ: they use BTProp's released belief trees and ChatGPT confidences instead of Gemma, detect false statements in FactCheckGPT and WikiBio, split by response (20% tuning), tune by AUROC on the tuning part, compute exactly on the trees (no p-bits), and use 5 runs with the rule "at least 0.02 and higher in at least 4 of 5". Details in `plans/E2-10_btprop.md`.

The plan for each experiment was written and fixed before seeing its results. The plans and results are in `plans/` (Japanese). Ties and losses are reported too.

## How the computation works

This section writes down what the code computes. `analysis/check_formulas.py` checks the code against these formulas (see the end of this section).

**Inputs from the model.** For each statement $i$, the model is asked "Is the following statement true? Answer Yes or No only." and we keep $s_i$ = (logit of "Yes") − (logit of "No") at the first answer token. For each pair of labels $k$ (for example "plant" and "animal"), the model is asked an exclusion question in both orders, and $z_k$ is the mean of the two Yes−No logits. From E2-5 on, the wording is "Is it true that no plant is an animal? Answer Yes or No only." (wording P2). For wordings where "No" means exclusion (E2-4, and P0/P1 in E2-5), the sign is flipped, so that always $z_k > 0$ means the model says the two labels never hold together.

**Calibration (on the tuning 20% only).** $\ell_i = (s_i - \beta)\cdot 2/\sigma$, where $\sigma$ is the standard deviation of $s$ on the tuning statements and the threshold $\beta$ is the one (among 97 quantiles) that gives the best accuracy on them. The model's answer is $m_i = [\ell_i > 0]$ and its confidence is $c_i = |\mathrm{sigmoid}(\ell_i) - 1/2|$. For the rules, $\zeta_k = 2 z_k/\sigma_z$, with $\sigma_z$ the standard deviation of $z$ over all label pairs.

**The network.** Each statement has a variable $x_i \in \{0,1\}$ (1 = true). In the rule-doubting network (E2-6 to E2-9), each label pair also has a variable $u_k \in \{0,1\}$, where $u_k = 1$ means "this exclusion rule does not hold". The network is the probability distribution

$$
P(x,u) \propto \exp\Big( \lambda \sum_i \ell_i x_i \;-\; \lambda r \sum_k \zeta_k u_k \;-\; W \sum_{(a,b,c)\in C} x_a x_b (1-x_c) \;-\; W \sum_k \sum_{(a,b)\in I_k} x_a x_b (1-u_k) \Big)
$$

- $C$: chaining rules written by a person, "if $a$ and $b$ then $c$" (for example "$w$ is a kind of $B$" and "$B$ is a kind of $Z$" give "$w$ is a kind of $Z$"). These are always correct.
- $I_k$: all pairs of statements that put the same word (or the same class) under the two labels of pair $k$. If both are true, the exclusion rule $k$ is broken unless $u_k = 1$.
- The second term makes $u_k = 1$ unlikely when the model was confident that the labels exclude each other ($\zeta_k$ large), and likely when it said they can overlap. So the model's rule answers are a starting belief, and the answers to the statements can overturn a rule.

The network that fully trusts the rules (E2-2 to E2-5, and the "trusting BP" in E2-9) has no $u$: its exclusion term is $-W \sum_{(a,b)\in E} x_a x_b$, where $E$ is the union of the $I_k$ taken as true. In E2-2, E2-3 and the human-rule runs of E2-4 these are the exclusions written by a person (flipped at random in the corrupted runs); otherwise they are the pairs with $z_k > 0$. (The E2-4 run "Bw" instead keeps every pair and multiplies its $W$ by the model's probability of exclusion.)

**Signals for "this answer is wrong" (higher = more suspicious).**
- Model's confidence: $-c_i$.
- Counting: $v_i - c_i/10$, where $v_i$ is the number of rules (chaining rules and exclusions with $z_k>0$, all taken as true) that the model's answers $m$ break and that involve statement $i$. The small $c_i/10$ term only breaks ties.
- BP: loopy sum-product belief propagation on $P$ (damping 0.5, at most 200 iterations, tolerance $10^{-6}$) gives $q_i \approx P(x_i = 1)$. The signal is $1 - q_i$ if $m_i = 1$, and $q_i$ if $m_i = 0$.
- p-bit simulation ("揺れ"): the same $P$ sampled by Gibbs sampling (graph coloring, 10 chains, 100 burn-in sweeps, then 400 sweeps averaged) gives $q_i$; the signal is built in the same way as for BP. Each update sets $x_i = 1$ with probability sigmoid(change in the exponent above), which is how a p-bit behaves. The rules here link three variables at a time, so real p-bit hardware, which usually links pairs, would need extra wiring.
- MaxSAT: the most probable assignment of $P$ (weights rounded to integers: $100\lambda|\ell_i|$ for each statement, $100\lambda r|\zeta_k|$ for each rule variable, $100W$ for each rule), solved with RC2. The signal is $[\text{MaxSAT changes } m_i] - c_i/10$.

**Tuning.** For BP, p-bits and MaxSAT separately, $(\lambda, W, r)$ is chosen from $\{0.5,1,2\}\times\{1,2,4\}\times\{0.5,1,2\}$ ($r$ only when rules are doubted) by the accuracy of the thresholded result on the tuning statements. Counting has no parameters. The tuning statements are all statements about 20% of the words, cities or diseases, and AUROC is measured on the rest.

**E2-9.** The rule answers are replaced by $\pm|z_k|$ with the sign set by the ground truth, and then a random fraction $f$ of the label pairs (seed $10000 + \text{split}$) have their sign flipped.

**Checking the code against these formulas.** `cd analysis; python check_formulas.py` (about 1 minute) builds 40 small random networks, computes the exact probabilities by enumerating every assignment of the formula above, and compares them with the code used in the experiments. Result: BP matches exactly on tree-shaped networks (difference 0.0000) and differs by at most 0.019 on networks with loops, as expected for loopy BP; the p-bit simulation differs by at most 0.003 (sampling noise); MaxSAT always finds the most probable assignment.

## Results at a glance

| Exp. | What | Main result | Plan and results |
|---|---|---|---|
| E2-2 | Geography test (GeoNames, 3,586 statements), human rules | BP 0.83 vs counting 0.81: win, just at the threshold | `plans/E2-2_geo.md` |
| E2-3 | Taxonomy test (WordNet, 2,074 statements), human rules | BP 0.88 vs counting 0.85: win | `plans/E2-3_wordnet.md` |
| E2-4 | Rules written by the model; deliberately corrupted human rules | Model rules were 31% correct, giving a tie. Human rules with 10% flipped still win; at 20% it is a tie | `plans/E2-4_selfrule.md` |
| E2-5 | Choose the question wording on separate labels, then use it once | Rules 87% correct. BP 0.82 vs counting 0.81: tie (BP beats the model's confidence) | `plans/E2-5_wording.md` |
| E2-6 | Let the network also doubt each rule | BP 0.84 vs counting 0.81: win (8 of 10 runs) | `plans/E2-6_doubt_rules.md` |
| E2-7 | Repeat in geography; compare with other methods | Geography: BP 0.82 vs counting 0.80, tie. Taxonomy: BP beats all other methods. Geography: tie with the ensemble | `plans/E2-7_geo_and_baselines.md` |
| E2-8 | Third domain (disease classes, Human Disease Ontology, 3,304 questions); the model also chooses which label pairs to check | Model rules 84% correct. Rule-doubting BP 0.76 vs counting 0.74: win (10/10). Pairs chosen by the model were 80% wrong; BP on them was slightly below BP on all pairs (tie by our criterion, but lower in all 10 runs) | `plans/E2-8_disease.md` |
| E2-9 | Same networks, only the fraction of wrong rules changes (0 to 50%; disease and taxonomy) | The BP-minus-counting gap stays about the same ("no change" by our criterion). BP that fully trusts the rules is better when rules are right and falls behind at 30% wrong. Extra check: without the human-written chaining rules, the gap grows with the wrong fraction in taxonomy (+0.03 to +0.09) | `plans/E2-9_wrong_rate.md` |
| E2-10 | Same test as BTProp (Hou et al., NAACL 2025): BTProp's released ChatGPT belief trees and confidences, scored on FactCheckGPT (631 claims) and WikiBio (921 sentences); only the final probability computation is swapped | BTProp's gain over the model's confidence is reproduced (+0.03 / +0.08). WikiBio: BTProp 0.80 vs our rule-doubting BP 0.77, BTProp wins (5/5). FactCheckGPT: 0.79 vs 0.80, tie | `plans/E2-10_btprop.md` |
| E2-11 | Same trees, a random fraction (0 to 50%) of the NLI links reversed | BTProp falls below the model's confidence at 20–30% wrong links (0.60–0.65 at 50%); rule-doubting BP stays near the confidence. Our BP wins from 20% (FactCheckGPT) and at 50% (WikiBio) | `plans/E2-10_btprop.md` |
| E2-12 | Same trees, links labeled by four real models (NLI small/base/large, Gemma-2 2B) | BTProp barely degrades (still above the confidence with Gemma). FactCheckGPT: all ties; WikiBio: BTProp wins for every link model (by 0.02–0.05). With natural link errors the rule-doubting BP is never better | `plans/E2-12_real_links.md` |

## E3: Can evidence let the model say "I don't know"?

After E2 we asked a different question: if the model is given evidence (retrieved documents), can a network over that evidence tell it when to abstain, better than simpler methods? Plan (fixed before the results, with changes logged): `plans/E3_evidence_abstain.md` (Japanese); prior work: `plans/E3_prior_work.md` (Japanese).

- Data: HoVer dev claims with gold evidence (ProgramFC's release), 2 to 4 Wikipedia articles per claim. 3,952 claims (48 used for pilots are excluded), supports/refutes.
- Models: Gemma-2 2B gives the verdict and its confidence P(True). Gemma-2 9B (4-bit) splits each claim into sub-claims with hidden entities ("#1 = the director of film X") and fills in candidate names from the evidence. A DeBERTa-v3-large NLI model judges each paragraph against the claim and each sub-claim.
- Methods, all scored by how well they rank claims for abstention: (1) Gemma's confidence, (2) counting supporting vs. contradicting sentences (Pandian 2026 style), (3) 11 NLI features + logistic regression (SURE-RAG style, 5-fold), (4) Dempster-Shafer combination (EvidentialRAG style), (5) hard AND logic (ProgramFC/GraphCheck style) and (5') the minimum sub-claim probability, (6) a network over candidates, sub-claims and the claim (exact inference, equivalent to BP on this small graph).

| Method | Accuracy, all answered | 90% answered | 70% | 50% | AURC (lower is better) |
|---|---|---|---|---|---|
| (1) Gemma's confidence | 67.9% | 69.6 | 73.2 | 76.9 | 0.207 |
| (2) Counting | 54.5 | 55.7 | 59.5 | 66.4 | 0.353 |
| (3) NLI features + LR | 71.0 | 73.2 | 77.9 | 82.5 | **0.177** |
| (4) Dempster-Shafer | 67.1 | 68.0 | 70.1 | 73.5 | 0.253 |
| (5) Hard logic | 71.0 | 71.1 | 71.3 | 71.2 | 0.287 |
| (5') Min sub-claim | 70.6 | 71.4 | 74.1 | 76.2 | 0.246 |
| (6) Network | 71.1 | 72.4 | 75.3 | 78.0 | 0.225 |

- With the rule fixed in advance (AURC difference of at least 0.02 in 4 of 5 folds), the network beats counting, Dempster-Shafer and the min sub-claim formula (5/5 each), ties with Gemma's confidence (difference 0.016, network better in 1 of 5 folds), and loses to the NLI features + LR (0/5). By number of articles (AURC for (1)/(6)/(3)): 2: 0.150/0.148/0.144, 3: 0.230/0.245/0.182, 4: 0.243/0.268/0.205.
- Removing one gold paragraph from "supports" claims (1,976 claims): Gemma's "supports" answers fall from 75% to 41%, but its mean confidence |p-0.5| does not move (0.445 to 0.443). The evidence-based scores do become less confident (network 0.405 to 0.349, min sub-claim 0.346 to 0.207).
- So: using evidence helps notice missing evidence, but connecting the evidence in a network did not beat a simple learned combination of NLI scores. We do not claim the network helps. Note that the network as run had no evidence-evidence contradiction links (the plan mentioned them), so it was a small loop-free graph. E3 is closed here.

## E4: A large memory network with many loops, BP and the p-bit method

E4 asks whether many of the model's answers, joined into one large network full of loops, catch the model's mistakes better than simpler methods. It also asks whether the p-bit method (sampling, run in software) differs from BP there. Plans (fixed before the results; checks added afterwards are labeled as such): `plans/E4-0_synthetic.md`, `plans/E4-1_wordnet_net.md`, `plans/E4-2_learned_net.md`; prior work: `plans/E4_prior_work.md`, `plans/E4-2_prior_work.md` (all Japanese). Same verdict rule as before: an AURC difference of at least 0.02 in at least 4 of 5 splits counts as a difference, otherwise a tie.

**E4-0, synthetic networks** (2,000 claims, 2 to 6 links per claim, 1% to 30% wrong rules, rules trusted at the right strength or twice too strong, 5 seeds each). BP and the p-bit method (one chain, 10,000 sweeps) tied in 30 of 32 conditions and BP was better in 2; the p-bit method was never better. Where links are many and wrong rules are over-trusted, BP's probabilities drift from the exact answer on small networks (mean error 0.10 with 6 links, against 0.01 for the p-bit method), but accuracy still tied. With strong, over-trusted rules a single p-bit chain got stuck. Here the network is the true generating model, so BP beat the model's confidence in 29 of 32 conditions.

**E4-1, a real network from WordNet and Gemma.** 1,500 nouns × 46 WordNet categories. Gemma-2 2B answered "X is a type of Y" for each noun (up to 4 true and 3 unrelated categories), for category pairs ("Y is a type of Z"), and for disjointness ("Nothing can be both a Y and a Z"). Only the logical form of the rules is fixed (inheritance, disjointness, and two rules between category claims); the category knowledge itself is the model's answers. 10,864 claims, 51,643 rules, about 80,000 independent loops.

| Method (noun claims, test 80%) | AURC (lower is better) | Accuracy, all answered | 50% answered |
|---|---|---|---|
| (1) Gemma's confidence | 0.142 | 81.9% | 88.3% |
| (2) Counting broken rules | 0.053 | 82.0 | 97.7 |
| (3) Features + logistic regression | **0.009** | 93.6 | 99.8 |
| (4) Small networks, one noun at a time (BP) | 0.059 | 81.6 | 96.1 |
| (5) One large network (BP) | 0.062 | 82.3 | 95.5 |
| (6) Large network, p-bit method (4 chains × 10,000 sweeps) | 0.080 | 82.3 | 93.9 |

The large network beat confidence (5/5) but tied with the small networks and lost to features + learning. BP and the p-bit method tied (in one split the p-bit chains got stuck; in the other four they matched within 0.001). Category claims shared by many nouns got worse inside the network (disjointness AURC 0.009 to 0.063): small doubts from many nouns pile up on them. A weakness: the false candidates were unrelated categories, which rules separate easily.

**E4-2, a network with learned rule weights.** Following Jiang, Lowd and Dou (ICDM 2012), where a Markov logic network with learned weights beat per-fact learning on NELL, we learned the strength of each rule type and a bias correction by logistic regression. All 1,681 nouns × 46 categories were asked (80,377 questions). Harder false candidates: (G) the false category Gemma believed most, (H) a sibling or uncle category (as in TaxoGlimpse, VLDB 2024), (E) an unrelated one. 11,864 claims, 55,136 rules.

| Method (noun claims, test 80%) | AURC | Accuracy, all answered | 50% answered |
|---|---|---|---|
| (1) Gemma's confidence | 0.248 | 69% | 77% |
| (2) Confidence + bias correction | 0.251 | 70 | 78 |
| (3) Features + logistic regression | 0.086 | 77 | 93 |
| (4) Learned small networks (BP) | 0.107 | 79 | 90 |
| (5) Learned large network (BP) | 0.382 | 46 | 61 |
| (5') Cautious large network: category claims held at Gemma's (corrected) values | **0.079** | 79 | 94 |
| (6) Learned large network, p-bit method with parallel tempering (4 temperatures) | 0.382 | 46 | 61 |
| (7) One rule strength chosen on a grid | 0.151 | 70 | 86 |

- The learned large network collapsed: it called almost everything true (false candidates caught: G 0%, H 3 to 11%, E 3 to 13%). It lost to (2), (3) and (4) in 5 of 5 splits. A check added after the results: with the category claims set to Gemma's raw values instead, it still collapsed (AURC 0.18 to 0.21). The shared category claims are pulled over by doubts from many nouns, the same overcounting seen in E4-1.
- The cautious network was lower than (3) in all 5 splits, but by only 0.007, so it is a tie. Jiang et al.'s result did not carry over to this setting, where the rules' content also comes from the model.
- BP and the p-bit method tied again (mean probability difference 0.004). No tempering swap was accepted, so the four temperatures did not help on a network this size.
- On the false category Gemma believed most (G), accuracy was 31% for confidence, 53% for (3), and 54% for the cautious network.
- Of five predictions written down beforehand, three held. Wrong were "the learned large network beats (2)" and "(5) and (5') tie on noun claims".

**Summary of E4.** In these tests, putting all answers into one large loopy network did not beat a simple learned combination of features, and letting shared knowledge change inside the network made results worse. Keeping shared knowledge fixed and inferring only the item-level answers was the best variant, tied with features + learning. BP and the p-bit method never differed in accuracy; any advantage of p-bits would have to be speed or energy, which simulation cannot measure. E4 is closed here.

## Layout

- `colab/`: Google Colab notebooks (T4 GPU). They ask Gemma the questions and save the answers. The E2-2 and E2-3 notebooks also run the analysis and draw the figures.
  - Open in Colab: [E2-2](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-2_geo_gemma.ipynb) / [E2-3](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-3_wordnet_gemma.ipynb) / [E2-4](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-4_selfrule_gemma.ipynb) / [E2-5](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-5_wording_gemma.ipynb) / [E2-7](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-7_geo_baselines_gemma.ipynb) / [E2-8](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-8_disease_gemma.ipynb)
  - The notebooks save to your own Google Drive and need your own Hugging Face token (`HF_TOKEN` in Colab secrets) with access to Gemma.
- `data/`: saved Gemma answers. With these, the analysis can be rerun without a GPU.
  - `geo_mouth_gemma-2-2b-it.json`, `wn_mouth_gemma-2-2b-it.json`: answers to the true/false statements (Yes−No logits)
  - `E2-4_rules_gemma-2-2b-it.json`, `E2-5_rules_gemma-2-2b-it.json`: answers to the rule questions
  - `E2-7_answers_gemma-2-2b-it.json`: geography rule questions and paraphrased questions
  - `E2-8_answers_gemma-2-2b-it.json`: disease questions, rule questions, and the model's pair choices (generated numbers)
- `analysis/`: analysis for E2-4 to E2-9 (CPU only), and `check_formulas.py`
- `analysis/btprop/`: E2-10 to E2-12. BTProp's code has no license file, so it is not copied here; `setup_btprop.sh` clones it at a fixed commit and fetches the labels
- `analysis/e3/`: E3 code (pilot and main Colab cells, notebook builders, `e3_analyze.py`)
- `colab/E3_hover_main.ipynb`: E3 main run (L4 GPU, about 4 hours; resumable). `colab/E3_split_results.ipynb` splits the output for download
- `analysis/e4/`: E4 code. `e40.py` (synthetic networks), `e41_data.py`/`e42_data.py` (build the WordNet questions), `e41.py`/`e42.py` (networks, BP, p-bit method, comparisons), `probe_*.py` (checks added after the results), notebook builders
- `colab/E4-1_isa_gemma.ipynb`, `colab/E4-2_isa_all_gemma.ipynb`: ask Gemma the E4-1 and E4-2 questions (L4 GPU, E4-2 about 30 minutes; resumable)
- `data/E4-1_answers_gemma-2-2b-it.json`, `data/E4-2_answers_gemma-2-2b-it.json`: Gemma's answers for E4 (statements and Yes−No logits)
- `results/`: result JSON files and figures (`results/E3/E3_main_analysis.json` for E3, `results/E4/` for E4)
- `plans/`: plans and results for each experiment (Japanese)

## Rerunning the analysis

You need Python 3.10 or newer and `numpy scipy scikit-learn python-sat nltk wordfreq geonamescache matplotlib`. The taxonomy test also needs the NLTK WordNet data: `python -m nltk.downloader wordnet`.
For figures, set the environment variable `JP_FONT` to a Japanese font file, such as Noto Sans JP.

Run from `analysis/`. Results are written to `results/`.

```
cd analysis
# E2-4: human rules (A), 10/20/30% flipped rules, and the original E2-3 rules
python run_c.py A,F0.1,F0.2,F0.3,E23
# E2-4: Gemma's rules (B) and confidence-weighted rules (Bw)
python run_b.py ../data/E2-4_rules_gemma-2-2b-it.json gemma B,Bw
# E2-5: rules from each question wording
python run_e25.py ../data/E2-5_rules_gemma-2-2b-it.json P2,P1,P12
# E2-6: network that also doubts the rules (wording P2, chosen in E2-5)
python e26.py ../data/E2-5_rules_gemma-2-2b-it.json P2 P2doubt
# E2-7: taxonomy (reuses the E2-6 result and adds the other methods)
python e27.py ../data/E2-7_answers_gemma-2-2b-it.json wn reuse:../results/E26_P2doubt.json wn
# E2-7: geography (runs the rule-doubting network first; about 8 minutes per split)
BASE=base_geo.py python e27.py ../data/E2-7_answers_gemma-2-2b-it.json geo run geo
# summary and figure
python summ.py ../results/C_A.json
python fig27.py
# E2-8: disease (runs the 5 splits in parallel, ~12 min per split). The ontology file is downloaded automatically
bash run28.sh ../data/E2-8_answers_gemma-2-2b-it.json gemma
# E2-9: vary the wrong-rule fraction (run each split, then merge; for the run without chaining rules use NOTRANS=1 LEVELS=0,0.3,0.5 and merge29.py _notrans)
for s in 0 1 2 3 4; do python e29.py dis $s; python e29.py wn $s; done
python merge29.py
# check that the code computes the formulas in "How the computation works"
python check_formulas.py
```

E2-10 and E2-11 (BTProp comparison, CPU only, a few minutes; also needs `pyyaml absl-py`, and `datasets` to download the WikiBio labels from Hugging Face):

```
cd analysis/btprop
bash setup_btprop.sh          # clones BTProp at 6e80da8, gets FactCheckGPT and WikiBio labels
cd work/BTProp
python ../../bt_repro.py factcheckgpt                       # BTProp as released
python ../../e210.py factcheckgpt ../../E210_factcheckgpt.json
python ../../e210.py wikibio ../../E210_wikibio.json
python ../../e211.py wikibio ../../E211_wikibio.json        # reversed links
```

WikiBio note: BTProp evaluates rewritten sentences (pronouns replaced by names) that are not released. `align_wikibio.py` matches the released labels to BTProp's sentences by order and string similarity (924 of 929 sentences). In BTProp's released code, a claim without a belief tree gets probability 1 (true); for FactCheckGPT (48 such claims) we also report BTProp with the model's confidence used for those claims, and compare against that.

E3: run `colab/E3_hover_main.ipynb` (needs access to Gemma-2 2B and 9B on Hugging Face). It writes `E3_main.json`; then `python analysis/e3/e3_analyze.py E3_main.json out.json` (CPU, a few seconds). The per-claim outputs contain HoVer text and are not stored here.

E4: in `analysis/e4/`, `python e40.py` (CPU, about 20 minutes). For E4-1/E4-2, build the questions with `NLTK_DATA=<wordnet> python e41_data.py` and `python e42_data.py` (they reproduce the files the notebooks used), then `python e41.py e41_data.json ../../data/E4-1_answers_gemma-2-2b-it.json out41.json` and `python e42.py e42_data.json ../../data/E4-2_answers_gemma-2-2b-it.json out42.json` (CPU, about 70 minutes each, mostly the p-bit method; `E41_SWEEPS`/`E42_SWEEPS` set the number of sweeps).

## Related work and what is different

The closest work we know is BTProp (Hou et al., NAACL 2025, https://arxiv.org/abs/2406.06950). It builds links between statements with a separate model (NLI) and uses belief propagation to detect an LLM's hallucinations. Here, the model also writes the exclusion rules itself, and the network also estimates whether each rule is true (E2-6). In E2-10 we ran BTProp on its own released trees: with the links as given, BTProp is better on WikiBio and tied on FactCheckGPT, so we do not claim our method is better. When a share of the links is deliberately flipped (E2-11), BTProp, which treats the links as certain, degrades sharply, while the rule-doubting BP does not. With links labeled by real models (E2-12), however, BTProp barely degrades: natural labeling errors seem far less harmful than random flips. The model's own confidence (P(True)), which the BTProp paper treats as a strong baseline, is included here as a baseline. Other related work: Crum et al. 2024 (asking LLMs for disjointness axioms), Xu et al. 2026 (neighbor-consistency belief), Jiang, Lowd and Dou 2012 (cleaning an extracted knowledge base with Markov logic), Maieutic prompting (Jung et al. 2022), BeliefBank (Kassner et al. 2021). The E2-9 pattern (doubting the rules costs a little when the rules are right and helps when many are wrong) matches the analysis in Snorkel (Ratner et al., VLDB 2017, https://arxiv.org/abs/1711.10160): learning how accurate each source is helps most when source accuracies vary. Here we see the same tendency when a model checks its own answers.

## Limitations

- Only one small model (Gemma-2 2B), only English, and three domains: geography, taxonomy and disease classes.
- Up to E2-7 a person chose which label pairs to ask rule questions about. In E2-8 the model chose them, but its choices followed list position and 80% were wrong. The only rule type the model writes is "these two never hold together".
- Disease ground truth comes from the Human Disease Ontology; a class missing from the ontology is not always truly absent (e.g. COVID-19 is not under respiratory system disease there).
- In E2-9 the wrong rules are random flips, which differ from the model's natural errors (mostly missed exclusions). The run without chaining rules was added after seeing the main E2-9 results.
- E2-4 to E2-7 reuse the same data across several experiments, and the 5 splits come from the same data.
- The paraphrase and neighbor-consistency signals in E2-7 are simplified versions of the published methods, adapted to a single yes/no answer.
- E2-10/E2-11 use BTProp's ChatGPT trees; our method was tuned on 20% of the data while BTProp used its published settings (fitted on the first 120 WikiBio passages). The wrong links in E2-11 are random reversals, not natural NLI errors (natural errors are E2-12; the 100-pair re-labeling there was done by Claude, an AI, and is not ground truth), and decomposition links were not changed. The wider grid and the directed decomposition factor in E2-10 were added after seeing the result (they do not change it).
- E3: one dataset (HoVer, gold evidence), small models only. The network ran without evidence-evidence contradiction links, so it was loop-free; whether loops change the picture is not tested here. Method (3) is trained (5-fold) while the network has no tuned parameters. In the drop-one-paragraph check, decompositions and candidates from the full evidence were reused, which slightly favors the NLI-based methods. The pilot checks of decomposition quality were done by Claude, an AI.
- E4: one domain (WordNet categories) and one small model. The truth comes from WordNet, which has its own quirks (we removed four categories for this reason). In E4-1 the false candidates were easy; E4-2 used harder ones. The rule weights in E4-2 are learned by an approximation (neighbors fixed at Gemma's answers), and the cautious network applies the bias correction learned on noun claims to category claims as well, which made its category-pair claims poor. Parallel tempering used only four temperatures and no swap was accepted. Checks marked as added after the results are for reference only.
- The p-bits are simulated in software. These experiments say nothing about speed or energy.

## How this was made

The research direction, the decisions, and the Colab runs are by the author. The experiment design, the code, the analysis, and the write-up were done together with Claude, an AI assistant made by Anthropic.

## License

Code: MIT License (see `LICENSE`). The data in `data/` are outputs of Gemma and follow the Gemma Terms of Use. GeoNames, WordNet, the Human Disease Ontology, Factcheck-GPT (Apache 2.0) and the WikiBio hallucination labels follow their own licenses. BTProp's code and logs are not included. HoVer is CC BY-SA 4.0; its text is not redistributed here.

## Data and model

- GeoNames (through the `geonamescache` package) https://www.geonames.org/
- WordNet (through NLTK) https://wordnet.princeton.edu/ ; word frequencies for E4 from `wordfreq` https://github.com/rspeer/wordfreq
- Human Disease Ontology (release 2026-10-29, CC0) https://disease-ontology.org/ . The file is not stored in this repository; the notebook and analysis download it from the GitHub tag v2026-10-29.
- BTProp code and released logs (commit 6e80da8) https://github.com/UCSB-NLP-Chang/BTProp
- FactCheckGPT labels (Factbench.jsonl, Wang et al. 2024) https://github.com/yuxiaw/Factcheck-GPT
- WikiBio GPT-3 hallucination labels (Manakul et al. 2023) https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination
- HoVer (Jiang et al., Findings of EMNLP 2020, CC BY-SA 4.0) https://hover-nlp.github.io/ , dev claims with gold evidence as released by ProgramFC (Pan et al., ACL 2023) https://github.com/teacherpeterpan/ProgramFC
- Gemma-2 9B instruction-tuned https://huggingface.co/google/gemma-2-9b-it and NLI model https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (E3)
- Gemma-2 2B instruction-tuned https://huggingface.co/google/gemma-2-2b-it . The Gemma outputs in `data/` are used under the Gemma Terms of Use.
