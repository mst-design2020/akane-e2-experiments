#!/bin/bash
# Fetch BTProp (no license file, so it is not copied into this repository) and the two label sets.
# Run from analysis/btprop.  Needs git, curl, python with numpy scikit-learn pyyaml absl-py, and `datasets` for WikiBio.
set -e
mkdir -p work && cd work
[ -d BTProp ] || git clone https://github.com/UCSB-NLP-Chang/BTProp
(cd BTProp && git checkout -q 6e80da8afb0acc30aa4555016364291f28d3ebf9)
mkdir -p BTProp/eval_dataset
curl -sSL -o BTProp/eval_dataset/Factbench.jsonl \
  https://raw.githubusercontent.com/yuxiaw/Factcheck-GPT/a9e6a04a953ad880806529c504a679dd1ad06528/Factbench.jsonl
[ -f wikibio_labels.json ] || python ../get_wikibio_labels.py wikibio_labels.json
cd BTProp && python ../../align_wikibio.py ../wikibio_labels.json
echo "ready: cd work/BTProp"
