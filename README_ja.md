# AKANE Architecture E2 実験：AIの間違いを、問題どうしのつながりから見つける

[English](README.md) | 日本語

**ひとことで：** 小さなAIに、答えを確かめるためのルールを自分で作らせます（外れも混ざります）。そのルールでAIの答えを確かめるとき、ルール自体が本当かどうかも一緒に疑いながら、確率伝搬で間違いを探します。BTProp（Hou ほか、NAACL 2025）に近い考え方です。違うのは、ルールをAI自身が作ることと、どのルールを信じるかを正解なしで網が決めることです。BTPropとは直接比べていません。

小さな言葉のAI（Gemma-2 2B）に○×問題をたくさん答えさせ、問題どうしをルールでつないだ網から、どの答えが怪しいかを見つける実験のコード・データ・結果です。
note記事の数字は、すべてこのリポジトリのファイルから出ています。

- 記事1「『数えるだけで足りた』を測り直したら――AIの間違いを、問題どうしのつながりから見つける」（E2-2・E2-3） https://note.com/mst_design2020/n/n28ced8c9549d
- 記事2「ルールまでAIに作らせたら――自分の答えを、自分で作ったルールで確かめられるか」（E2-4〜E2-7） https://note.com/mst_design2020/n/n5d7de09a97f2

著者：加藤雅也（M.S.T.デザイン代表・大分県佐伯市） https://mst-design.jp/

## AKANE Architectureとは（短く）

言葉のAIを「口」として使い、答えを確かめる仕組みをAIの外に置く、という考え方です。AIがまず答え、そのあとで外の仕組みが、答えを大きな記憶やルールに照らして、つじつまが合っているかを確かめます。将来は、この確かめる計算を、p-bit（確率的に0と1を行き来する部品）のような物理の部品に任せることを考えています。
このリポジトリの実験は、その「確かめる仕組み」の、一番小さな形を試したものです。研究全体の説明は https://mst-design.jp/research/ にあります。

## 何をしているか（短く）

- 問題の例：「パリはフランスの都市である」「フランスはヨーロッパの国である」「パリはヨーロッパにある」（実際は英語）
- ルールの例：「AがBの一種で、BがCの一種なら、AはCの一種」「植物であって、しかも動物でもあるものはない」
- 比べる方法：AIの自信／ルールとの矛盾の数を数える／MaxSAT／確率伝搬（BP）／p-bitの動きをパソコンで真似た方法（揺れ）。E2-7ではさらに、言い換えのぶれ・近所の答えとの一致・それらの組み合わせ。
- 物差し：AIの間違いを見分けるAUROC。問題の2割で調整し、残り8割で測る。分け方5通り×p-bitの乱数2通り＝10回。
- 勝ち負けの決め方（各実験の前に固定）：10回の平均でAUROCの差が0.02以上、かつ10回中8回以上で上回ったら「勝ち」。差が0.02未満なら「同着」。

各実験の計画（結果を見る前に書いて固定したもの）と結果は `plans/` にあります。同着や負けの結果もすべて載せています。

## 結果の一覧

| 実験 | 内容 | 主な結果 | 計画と結果 |
|---|---|---|---|
| E2-2 | 地理テスト（GeoNames、3,586問）、人が書いたルール | BP 0.83 対 数える 0.81：基準ぎりぎりで勝ち | `plans/E2-2_geo.md` |
| E2-3 | ものの分類テスト（WordNet、2,074問）、人が書いたルール | BP 0.88 対 数える 0.85：勝ち | `plans/E2-3_wordnet.md` |
| E2-4 | ルールをAIに作らせる／わざと外したルール | AIのルールは31%正しい → 同着。正しいルールを1割外しても勝ち、2割で同着 | `plans/E2-4_selfrule.md` |
| E2-5 | ルールの聞き方を、テストと別の言葉で選んでから使う | ルールは87%正しい。BP 0.82 対 数える 0.81：同着（AIの自信には勝ち） | `plans/E2-5_wording.md` |
| E2-6 | ルール自体も網の中で疑う | BP 0.84 対 数える 0.81：勝ち（10回中8回） | `plans/E2-6_doubt_rules.md` |
| E2-7 | 地理での繰り返し／ほかの方法との比べ | 地理：BP 0.82 対 数える 0.80：同着。分類：BPがほかの方法すべてに勝ち。地理：組み合わせと同着 | `plans/E2-7_geo_and_baselines.md` |
| E2-8 | 3つ目の分野（病気の分類、Human Disease Ontology、3,304問）。確かめる組もAIに選ばせる | ルールは84%正しい。ルールも疑うBP 0.76 対 数える 0.74：勝ち（10回とも）。AIに選ばせた組は8割が外れで、全部の組を聞くより少し下（基準では同着） | `plans/E2-8_disease.md` |
| E2-9 | 同じ網で、ルールの外れの割合だけを0〜5割と変える（病気・分類） | 外れを増やしても、BPと数えるの差はほぼ同じ（基準では「変わらない」）。ルールを全部信じるBPは、ルールが正しいときは上、外れ3割で逆転。追加の確かめ：人のつながりのルールを抜くと、分類では外れが多いほど差が開いた（+0.03→+0.09） | `plans/E2-9_wrong_rate.md` |

## ファイルの置き場所

- `colab/`：Google Colabで動かすノート（T4 GPU）。Gemmaに質問して答えを保存する。E2-2・E2-3はノートの中で計算と図まで行う。
  - Colabで開く：[E2-2](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-2_geo_gemma.ipynb) / [E2-3](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-3_wordnet_gemma.ipynb) / [E2-4](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-4_selfrule_gemma.ipynb) / [E2-5](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-5_wording_gemma.ipynb) / [E2-7](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-7_geo_baselines_gemma.ipynb) / [E2-8](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-8_disease_gemma.ipynb)
  - 答えは自分のGoogleドライブに保存されます。Gemmaを使える自分のHugging Faceの鍵（Colabの鍵マークに `HF_TOKEN`）が必要です。
- `data/`：Gemmaの答え（保存したもの）。これがあれば、GPUなしで計算をやり直せる。
  - `geo_mouth_gemma-2-2b-it.json`、`wn_mouth_gemma-2-2b-it.json`：○×問題への答え（Yes−Noのlogit）
  - `E2-4_rules_gemma-2-2b-it.json`、`E2-5_rules_gemma-2-2b-it.json`：ルールの質問への答え
  - `E2-7_answers_gemma-2-2b-it.json`：地理のルールの質問と、言い換えの質問への答え
  - `E2-8_answers_gemma-2-2b-it.json`：病気の○×問題・ルールの質問・組選び（AIが書いた番号）への答え
- `analysis/`：E2-4〜E2-9の計算（CPUのみ）
- `results/`：計算結果のJSONと図
- `plans/`：各実験の計画と結果

## 計算をやり直す

Python 3.10以上。必要なもの：`numpy scipy scikit-learn python-sat nltk wordfreq geonamescache matplotlib`（ものの分類はNLTKのWordNetのデータも使う：`python -m nltk.downloader wordnet`）。
図を作るときは、日本語のフォント（例：Noto Sans JP）のパスを環境変数 `JP_FONT` に入れてください。

`analysis/` に移動してから実行します。結果は `results/` に書き出されます。

```
cd analysis
# E2-4：人のルール（A）、わざと外したルール（1割・2割・3割）、E2-3のルールそのまま
python run_c.py A,F0.1,F0.2,F0.3,E23
# E2-4：Gemmaのルール（B）と、自信を重みにしたもの（Bw）
python run_b.py ../data/E2-4_rules_gemma-2-2b-it.json gemma B,Bw
# E2-5：聞き方ごとのルール
python run_e25.py ../data/E2-5_rules_gemma-2-2b-it.json P2,P1,P12
# E2-6：ルールも疑う網（E2-5で選んだ聞き方P2のルール）
python e26.py ../data/E2-5_rules_gemma-2-2b-it.json P2 P2doubt
# E2-7：ものの分類（E2-6の結果を使い、ほかの方法と比べる）
python e27.py ../data/E2-7_answers_gemma-2-2b-it.json wn reuse:../results/E26_P2doubt.json wn
# E2-7：地理（ルールも疑う網を計算してから比べる。1つの分け方に8分ほど）
BASE=base_geo.py python e27.py ../data/E2-7_answers_gemma-2-2b-it.json geo run geo
# まとめと図
python summ.py ../results/C_A.json
python fig27.py
# E2-8：病気（分け方5通りを並べて回す。1つの分け方に12分ほど）。病気の分類表は自動で取ってくる
bash run28.sh ../data/E2-8_answers_gemma-2-2b-it.json gemma
# E2-9：外れの割合を変える（分け方ごとに回してからまとめる。つながりのルールを抜くときは NOTRANS=1 LEVELS=0,0.3,0.5）
for s in 0 1 2 3 4; do python e29.py dis $s; python e29.py wn $s; done
python merge29.py
```

## 近い研究と、違うところ

一番近いのはBTProp（Hou ほか、NAACL 2025、https://arxiv.org/abs/2406.06950 ）です。問題どうしのつながりを別のAI（NLI）の判定で作り、確率伝搬でAIの間違いを見つけます。ここでは、両立しないというルールもAI自身に作らせ、そのルールが本当かどうかも網の中で疑わせました（E2-6）。BTPropそのものは動かしておらず、BTPropのテスト問題でも試していないので、どちらが優れているかは言えません。BTPropの論文で強い比べ相手とされる「AIの自信」（P(True)）は、ここでも比べ相手に入れています。ほかに近い研究：Crum ほか 2024（AIに両立しないかを聞いてルールを作る）、Xu ほか 2026（近所の答えとの一致）、Jiang・Lowd・Dou 2012（自動で集めた知識をルールで掃除）、Maieutic prompting（Jung ほか 2022）、BeliefBank（Kassner ほか 2021）。E2-9の「ルールを疑うと、ルールが正しいときは少し損で、外れが多いときに効く」という形は、Snorkel（Ratner ほか、VLDB 2017、https://arxiv.org/abs/1711.10160 ）の分析（ルールの当たり具合を学ぶ利点は、ルールの当たり外れがばらばらなときに大きい）と同じ傾向です。ここでは、それがAIが自分の答えを確かめる場面でも見えた、という位置づけです。

## 言えないこと（弱点）

- AIは小さな1種類（Gemma-2 2B）だけで、問題は英語の地理・分類・病気だけ。
- どの組についてルールを聞くかは、E2-7までは人が決めた。E2-8でAIに選ばせたが、番号の位置に引っぱられて8割が外れた。ルールの形は「両立しない」の1つだけ。
- 病気の正解は分類表（Human Disease Ontology）で、表に載っていない＝本当に違う、とは限らない（例：表では新型コロナが呼吸器の病気に入っていない）。
- E2-9の外れは、くじで選んでわざと逆にしたもので、AIが自然に出す外れ（見逃しが多い）とは形が違う。つながりのルールを抜いた確かめは、結果を見たあとで足したもの。
- 同じデータで実験を重ねている（E2-4〜E2-7）。分け方の5通りは同じデータの使い回し。
- E2-7の「言い換えのぶれ」「近所の答えとの一致」は、元の論文の方法を、はい・いいえの1回の答えに合わせて簡単にしたもの。
- p-bitはパソコンの中で真似たもの（ギブスサンプリング）。速さや電力については、この実験からは何も言えない。

## 作り方について

研究の方向・判断・Colabの実行は著者が行いました。実験の設計、コード、計算、文章は、Anthropic社のAIアシスタント「Claude」と一緒に作りました。

## ライセンス

コードはMITライセンスです（`LICENSE`）。`data/` はGemmaの出力で、Gemmaの利用規約に従います。GeoNames・WordNet・Human Disease Ontologyは、それぞれのライセンスに従います。

## 使ったデータとモデル

- GeoNames（`geonamescache` パッケージ経由） https://www.geonames.org/
- WordNet（NLTK経由） https://wordnet.princeton.edu/
- Human Disease Ontology（2026-10-29版、CC0） https://disease-ontology.org/ 。ファイルはリポジトリに入れず、ノートと計算の中でGitHubのタグ v2026-10-29 から取ってくる。
- Gemma-2 2B（指示調整版） https://huggingface.co/google/gemma-2-2b-it 。`data/` のGemmaの答えは、Gemmaの利用規約に従って使っています。
