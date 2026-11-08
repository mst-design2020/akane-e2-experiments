"""Download the WikiBio GPT-3 hallucination labels (Manakul et al. 2023) from Hugging Face.
pip install datasets; python get_wikibio_labels.py OUT.json"""
import json, sys
from datasets import load_dataset
d = load_dataset('potsawee/wiki_bio_gpt3_hallucination')['evaluation']
rows = [{'gpt3_sentences': r['gpt3_sentences'], 'annotation': r['annotation'], 'wiki_bio_test_idx': r['wiki_bio_test_idx']} for r in d]
json.dump(rows, open(sys.argv[1], 'w'), ensure_ascii=False)
print(len(rows))
