# AKANE Architecture E2: Finding an LLM's mistakes from how its answers connect

English | [日本語](README_ja.md)

Code, data, and results for a set of small experiments. A small language model (Gemma-2 2B) answers many true/false statements. The statements are linked by rules, and we try to find which of the model's answers are wrong by looking at the whole network of answers.
Every number in the two articles below comes from files in this repository.

- Article 1 (Japanese): E2-2 and E2-3, rules written by a person. https://note.com/mst_design2020/n/n28ced8c9549d
- Article 2 (Japanese): E2-4 to E2-7, rules written by the model itself. https://note.com/mst_design2020/n/n5d7de09a97f2

Author: Masaya Kato (M.S.T. Design, Saiki, Oita, Japan) https://mst-design.jp/

## AKANE Architecture in short

The idea is to use a language model as the "mouth" and to place the checking outside the model. The model answers first. Then an outside system checks whether the answers are consistent with a large memory and with rules. In the long run, we want to hand this checking computation to physical parts such as p-bits (probabilistic bits that flip between 0 and 1).
The experiments here test the smallest version of that outside checker. More about the overall research: https://mst-design.jp/research/ (Japanese)

## What the experiments do

- Example statements: "Paris is a city in France." "France is a country in Europe." "Paris is located in Europe."
- Example rules: "If A is a kind of B and B is a kind of C, then A is a kind of C." "Nothing is both a plant and an animal."
- Signals compared for flagging wrong answers: the model's own confidence; counting rule violations; MaxSAT; loopy belief propagation (BP); a software simulation of p-bits (Gibbs sampling, called "揺れ" in the code). E2-7 adds paraphrase inconsistency, neighbor consistency (a simplified form of Xu et al. 2026), and a logistic-regression ensemble of these signals.
- Metric: AUROC for detecting the model's wrong answers. 20% of the words or cities are used for tuning and 80% for evaluation. There are 5 splits × 2 p-bit seeds, so 10 runs.
- Decision rule, fixed before each experiment: A "wins" over B if the mean AUROC difference is at least 0.02 and A is higher in at least 8 of 10 runs. A difference below 0.02 is a "tie".

The plan for each experiment was written and fixed before seeing its results. The plans and results are in `plans/` (Japanese). Ties and losses are reported too.

## Results at a glance

| Exp. | What | Main result | Plan and results |
|---|---|---|---|
| E2-2 | Geography test (GeoNames, 3,586 statements), human rules | BP 0.83 vs counting 0.81: win, just at the threshold | `plans/E2-2_geo.md` |
| E2-3 | Taxonomy test (WordNet, 2,074 statements), human rules | BP 0.88 vs counting 0.85: win | `plans/E2-3_wordnet.md` |
| E2-4 | Rules written by the model; deliberately corrupted human rules | Model rules were 31% correct, giving a tie. Human rules with 10% flipped still win; at 20% it is a tie | `plans/E2-4_selfrule.md` |
| E2-5 | Choose the question wording on separate labels, then use it once | Rules 87% correct. BP 0.82 vs counting 0.81: tie (BP beats the model's confidence) | `plans/E2-5_wording.md` |
| E2-6 | Let the network also doubt each rule | BP 0.84 vs counting 0.81: win (8 of 10 runs) | `plans/E2-6_doubt_rules.md` |
| E2-7 | Repeat in geography; compare with other methods | Geography: BP 0.82 vs counting 0.80, tie. Taxonomy: BP beats all other methods. Geography: tie with the ensemble | `plans/E2-7_geo_and_baselines.md` |

## Layout

- `colab/`: Google Colab notebooks (T4 GPU). They ask Gemma the questions and save the answers. The E2-2 and E2-3 notebooks also run the analysis and draw the figures.
  - Open in Colab: [E2-2](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-2_geo_gemma.ipynb) / [E2-3](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-3_wordnet_gemma.ipynb) / [E2-4](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-4_selfrule_gemma.ipynb) / [E2-5](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-5_wording_gemma.ipynb) / [E2-7](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-7_geo_baselines_gemma.ipynb)
  - The notebooks save to your own Google Drive and need your own Hugging Face token (`HF_TOKEN` in Colab secrets) with access to Gemma.
- `data/`: saved Gemma answers. With these, the analysis can be rerun without a GPU.
  - `geo_mouth_gemma-2-2b-it.json`, `wn_mouth_gemma-2-2b-it.json`: answers to the true/false statements (Yes−No logits)
  - `E2-4_rules_gemma-2-2b-it.json`, `E2-5_rules_gemma-2-2b-it.json`: answers to the rule questions
  - `E2-7_answers_gemma-2-2b-it.json`: geography rule questions and paraphrased questions
- `analysis/`: analysis for E2-4 to E2-7 (CPU only)
- `results/`: result JSON files and figures
- `plans/`: plans and results for each experiment (Japanese)

## Rerunning the analysis

You need Python 3.10 or newer and `numpy scikit-learn python-sat nltk wordfreq geonamescache matplotlib`. The taxonomy test also needs the NLTK WordNet data: `python -m nltk.downloader wordnet`.
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
```

## Related work and what is different

The closest work we know is BTProp (Hou et al., NAACL 2025, https://arxiv.org/abs/2406.06950). It builds links between statements with a separate model (NLI) and uses belief propagation to detect an LLM's hallucinations. Here, the model also writes the exclusion rules itself, and the network also estimates whether each rule is true (E2-6). We did not run BTProp or use its benchmarks, so we cannot say which is better. The model's own confidence (P(True)), which the BTProp paper treats as a strong baseline, is included here as a baseline. Other related work: Crum et al. 2024 (asking LLMs for disjointness axioms), Xu et al. 2026 (neighbor-consistency belief), Jiang, Lowd and Dou 2012 (cleaning an extracted knowledge base with Markov logic), Maieutic prompting (Jung et al. 2022), BeliefBank (Kassner et al. 2021).

## Limitations

- Only one small model (Gemma-2 2B), only English, and only two domains: geography and taxonomy.
- A person chose which label pairs to ask rule questions about. The only rule type the model writes is "these two never hold together".
- E2-4 to E2-7 reuse the same data across several experiments, and the 5 splits come from the same data.
- The paraphrase and neighbor-consistency signals in E2-7 are simplified versions of the published methods, adapted to a single yes/no answer.
- The p-bits are simulated in software. These experiments say nothing about speed or energy.

## How this was made

The research direction, the decisions, and the Colab runs are by the author. The experiment design, the code, the analysis, and the write-up were done together with Claude, an AI assistant made by Anthropic.

## License

Code: MIT License (see `LICENSE`). The data in `data/` are outputs of Gemma and follow the Gemma Terms of Use. GeoNames and WordNet follow their own licenses.

## Data and model

- GeoNames (through the `geonamescache` package) https://www.geonames.org/
- WordNet (through NLTK) https://wordnet.princeton.edu/
- Gemma-2 2B instruction-tuned https://huggingface.co/google/gemma-2-2b-it . The Gemma outputs in `data/` are used under the Gemma Terms of Use.
