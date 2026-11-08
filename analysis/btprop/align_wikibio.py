"""BTPropのWikiBio前処理（代名詞を名前に置き換えた文、非公開）に、公開の正解ラベルを順番で合わせる。
公開の hmm_posterior.json（評価文の順）と、生の gpt3_sentences（121人目以降、126番除く）を文の似かたで並べる。
cwd=clone。python ../../align_wikibio.py ../wikibio_labels.json。出力 eval_dataset/wikibio_preprocess_v2.json（BTPropの形）"""
import json, difflib, numpy as np
import sys
raw = json.load(open(sys.argv[1]))
raw = [raw[i] for i in range(len(raw)) if i != 126][120:]
A = [(k, j, s, r['annotation'][j]) for k, r in enumerate(raw) for j, s in enumerate(r['gpt3_sentences'])]
B = [x['statement'] for x in json.load(open('logs/conf_estimation/hmm_posterior.json'))]
sim = lambda a, b: difflib.SequenceMatcher(None, a, b).ratio()
n, m = len(A), len(B); G = -0.3
S = np.full((n + 1, m + 1), -1e9); S[0, :] = np.arange(m + 1) * G; S[:, 0] = np.arange(n + 1) * G
bk = np.zeros((n + 1, m + 1), int)
band = 60
for i in range(1, n + 1):
    for j in range(max(1, i - band), min(m, i + band) + 1):
        c = [S[i-1, j-1] + sim(A[i-1][2], B[j-1]) - 0.5, S[i-1, j] + G, S[i, j-1] + G]
        bk[i, j] = int(np.argmax(c)); S[i, j] = max(c)
i, j, pairs = n, m, []
while i > 0 and j > 0:
    if bk[i, j] == 0: pairs.append((i-1, j-1, sim(A[i-1][2], B[j-1]))); i -= 1; j -= 1
    elif bk[i, j] == 1: i -= 1
    else: j -= 1
pairs = pairs[::-1]
sims = np.array([p[2] for p in pairs])
print('raw', n, 'btprop', m, 'paired', len(pairs), 'sim<0.5', int((sims < .5).sum()), 'median', float(np.median(sims)))
# 評価で使う形：1人＝1回答。文はBTPropの文、ラベルは生の注釈
by = {}
for a, b, s in pairs:
    if s < 0.5: continue
    k = A[a][0]; by.setdefault(k, ([], []))
    by[k][0].append(B[b]); by[k][1].append(A[a][3])
data = [{'segmented_response': v[0], 'labels': v[1]} for k, v in sorted(by.items())]
# BTPropの読み込み（126番を捨てて[120:]）でちょうど data になるよう空きを入れる
L = [{'segmented_response': [], 'labels': []}] * 120 + data
L.insert(126, {'segmented_response': [], 'labels': []})   # data_utils が126番を捨てる分
json.dump(L, open('eval_dataset/wikibio_preprocess_v2.json', 'w'), ensure_ascii=False)
print('passages', len(data), 'sentences', sum(len(d['labels']) for d in data))
low = [(A[a][2][:70], B[b][:70]) for a, b, s in pairs if s < .6][:6]
for x in low: print(x)
