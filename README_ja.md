# AKANE Architecture E2 実験：AIの間違いを、問題どうしのつながりから見つける

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24045800.svg)](https://doi.org/10.5281/zenodo.24045800)

[English](README.md) | 日本語

**ひとことで：** 小さなAIに、答えを確かめるためのルールを自分で作らせます（外れも混ざります）。そのルールでAIの答えを確かめるとき、ルール自体が本当かどうかも一緒に疑いながら、確率伝搬で間違いを探します。BTProp（Hou ほか、NAACL 2025）に近い考え方です。違うのは、ルールをAI自身が作ることと、どのルールを信じるかを正解なしで網が決めることです。E2-10・E2-11で、BTPropが公開した木そのものを使って両方を比べました。つながりが正しいときはBTPropが上か同着、つながりの一部が外れているときは、ルールも疑う網のほうが崩れにくい、という結果でした。

小さな言葉のAI（Gemma-2 2B）に○×問題をたくさん答えさせ、問題どうしをルールでつないだ網から、どの答えが怪しいかを見つける実験のコード・データ・結果です。
note記事の数字は、すべてこのリポジトリのファイルから出ています。

- 記事1「『数えるだけで足りた』を測り直したら――AIの間違いを、問題どうしのつながりから見つける」（E2-2・E2-3） https://note.com/mst_design2020/n/n28ced8c9549d
- 記事2「ルールまでAIに作らせたら――自分の答えを、自分で作ったルールで確かめられるか」（E2-4〜E2-7） https://note.com/mst_design2020/n/n5d7de09a97f2
- 記事3「外れたルールでも、疑えば使えるか――AIの間違い探しで、ルールの外れを増やしてみた」（E2-8・E2-9） https://note.com/mst_design2020/n/n0cddc2e4f209
- 記事4「先行研究と同じテストで比べてみた――AIの嘘探しで、負けた所と崩れなかった所」（E2-10・E2-11） https://note.com/mst_design2020/n/ne9eea6736480

著者：加藤雅也（M.S.T.デザイン代表・大分県佐伯市） https://mst-design.jp/ ・ ORCID https://orcid.org/0009-0001-6803-654X ・ DOI https://doi.org/10.5281/zenodo.24045800

## AKANE Architectureとは（短く）

言葉のAIを「口」として使い、答えを確かめる仕組みをAIの外に置く、という考え方です。AIがまず答え、そのあとで外の仕組みが、答えを大きな記憶やルールに照らして、つじつまが合っているかを確かめます。将来は、この確かめる計算を、p-bit（確率的に0と1を行き来する部品）のような物理の部品に任せることを考えています。
このリポジトリの実験は、その「確かめる仕組み」の、一番小さな形を試したものです。研究全体の説明は https://mst-design.jp/research/ にあります。

## 何をしているか（短く）

- 問題の例：「パリはフランスの都市である」「フランスはヨーロッパの国である」「パリはヨーロッパにある」（実際は英語）
- ルールの例：「AがBの一種で、BがCの一種なら、AはCの一種」「植物であって、しかも動物でもあるものはない」
- 比べる方法：AIの自信／ルールとの矛盾の数を数える／MaxSAT／確率伝搬（BP）／p-bitの動きをパソコンで真似た方法（揺れ）。E2-7ではさらに、言い換えのぶれ・近所の答えとの一致・それらの組み合わせ。
- 物差し：AIの間違いを見分けるAUROC。問題の2割で調整し、残り8割で測る。分け方5通り×p-bitの乱数2通り＝10回。
- 勝ち負けの決め方（各実験の前に固定）：10回の平均でAUROCの差が0.02以上、かつ10回中8回以上で上回ったら「勝ち」。差が0.02未満なら「同着」。
- E2-10・E2-11だけはやり方が違います：GemmaではなくBTPropが公開したChatGPTの木と自信を使い、FactCheckGPTとWikiBioで嘘の文を見つけます。回答ごとに2割で調整（AUROCで選ぶ）、木の上で厳密に計算（p-bitなし）、5回で「0.02以上かつ5回中4回以上」が勝ち。詳しくは `plans/E2-10_btprop.md`。

各実験の計画（結果を見る前に書いて固定したもの）と結果は `plans/` にあります。同着や負けの結果もすべて載せています。

## 計算のしかた

コードが実際に何を計算しているかを、式で書きます。コードがこの式どおりかは `analysis/check_formulas.py` で確かめられます（この節の最後）。

**AIから受け取るもの。** 問題 $i$ ごとに「Is the following statement true? Answer Yes or No only.」と聞き、最初の語の「Yes」と「No」のロジットの差を $s_i$ とします。ラベルの組 $k$（例：植物と動物）ごとに「両立しないか」を2つの順番で聞き、Yes−Noのロジットの平均を $z_k$ とします。E2-5からの聞き方は「Is it true that no plant is an animal? Answer Yes or No only.」（聞き方P2）です。「No」が両立しないを意味する聞き方（E2-4、E2-5のP0・P1）では符号を逆にし、いつも $z_k > 0$ が「AIは両立しないと言った」になるようにしています。

**ずれと倍率の調整（調整用の2割だけで決める）。** $\ell_i = (s_i - \beta)\cdot 2/\sigma$。$\sigma$ は調整用の問題での $s$ の標準偏差、$\beta$ は97個の分位点のうち調整用で正解率が一番高くなるしきい値です。AIの答えは $m_i = [\ell_i > 0]$、AIの自信は $c_i = |\mathrm{sigmoid}(\ell_i) - 1/2|$。ルールのほうは $\zeta_k = 2 z_k/\sigma_z$（$\sigma_z$ は全部の組での $z$ の標準偏差）です。

**網。** 問題ごとに $x_i \in \{0,1\}$（1＝正しい）を置きます。ルールも疑う網（E2-6〜E2-9）では、ラベルの組ごとに $u_k \in \{0,1\}$ も置きます。$u_k = 1$ は「この両立しないルールは成り立たない」という意味です。網は次の確率分布です。

$$
P(x,u) \propto \exp\Big( \lambda \sum_i \ell_i x_i \;-\; \lambda r \sum_k \zeta_k u_k \;-\; W \sum_{(a,b,c)\in C} x_a x_b (1-x_c) \;-\; W \sum_k \sum_{(a,b)\in I_k} x_a x_b (1-u_k) \Big)
$$

- $C$：人が書いたつながりのルール「$a$ と $b$ が正しければ $c$ も正しい」（例：「$w$ は $B$ の一種」と「$B$ は $Z$ の一種」なら「$w$ は $Z$ の一種」）。いつも正しいルールです。
- $I_k$：同じことば（または同じ種類）を、組 $k$ の2つのラベルの両方に入れている問題の組。両方が正しいと、$u_k = 1$ でない限り、ルール $k$ を破ります。
- 2つ目の項により、AIが自信を持って「両立しない」と言った組（$\zeta_k$ が大きい）では $u_k = 1$ になりにくく、「両立する」と言った組ではなりやすくなります。つまり、AIのルールの答えは出発点の見込みで、問題への答えのほうからルールがくつがえることもあります。

ルールを全部信じる網（E2-2〜E2-5と、E2-9の「疑わないBP」）には $u$ がなく、両立しないの項は $-W \sum_{(a,b)\in E} x_a x_b$ です。$E$ は正しいとした組の $I_k$ を合わせたもので、E2-2・E2-3とE2-4の人のルールの回では人が書いたもの（わざと外す回はくじで逆にしたもの）、それ以外は $z_k > 0$ の組です。（E2-4の「Bw」だけは、全部の組を入れ、それぞれの $W$ にAIの「両立しない」の確率を掛けています。）

**「この答えは間違い」の合図（大きいほど怪しい）。**
- AIの自信：$-c_i$
- 矛盾の数を数える：$v_i - c_i/10$。$v_i$ は、AIの答え $m$ が破っているルール（つながりのルールと、$z_k>0$ の両立しないルールを全部正しいとしたもの）のうち、問題 $i$ が入っている本数です。$c_i/10$ は同じ本数どうしの順番を決めるだけです。
- 確率伝搬（BP）：$P$ にループのある確率伝搬（sum-product、ダンピング0.5、最大200回、収束の幅 $10^{-6}$）をかけて $q_i \approx P(x_i = 1)$ を出します。合図は $m_i = 1$ なら $1 - q_i$、$m_i = 0$ なら $q_i$ です。
- p-bit方式（コードでは「揺れ」）：同じ $P$ をギブスサンプリング（彩色で同時に更新、10本の鎖、はじめの100回は捨て、そのあと400回の平均）で $q_i$ を出し、BPと同じ形の合図にします。1回の更新で、上の式の指数の変化をsigmoidに通した確率で $x_i = 1$ にするので、p-bitの動きと同じです。ただし、ここのルールは3つの変数を一度につなぐので、ふつう2つずつつなぐp-bitの部品で作るには、配線の工夫が要ります。
- MaxSAT：$P$ で一番確率の高い組み合わせ（重みは整数に丸める：問題ごとに $100\lambda|\ell_i|$、ルールの札ごとに $100\lambda r|\zeta_k|$、ルールごとに $100W$）をRC2で解きます。合図は「MaxSATが $m_i$ を変えたか」$- c_i/10$ です。

**調整。** BP・p-bit方式・MaxSATそれぞれで、$(\lambda, W, r)$ を $\{0.5,1,2\}\times\{1,2,4\}\times\{0.5,1,2\}$ から（$r$ はルールを疑うときだけ）、調整用の問題での正解率で選びます。数えるには調整するものはありません。調整用は、ことば・都市・病気の2割についての問題全部で、AUROCは残りで測ります。

**E2-9。** ルールの答えを $\pm|z_k|$ に置きかえ（符号は正解に合わせる）、そのうち割合 $f$ の組をくじ（seed $10000 + $ 分け方）で選んで符号を逆にします。

**コードが式どおりかの確かめ。** `cd analysis; python check_formulas.py`（1分ほど）は、小さな網を40個くじで作り、上の式の全部の組み合わせを数え上げて正確な確率を出し、実験で使ったコードの結果と比べます。結果：木の形の網ではBPは正確に一致（ずれ0.0000）、ループのある網では最大0.019のずれ（ループのある確率伝搬は近似なので想定どおり）、p-bit方式のずれは最大0.003（サンプリングのばらつき）、MaxSATはいつも一番確率の高い組み合わせを見つけました。

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
| E2-10 | BTProp（Hou ほか、NAACL 2025）と同じテスト。論文の作者が公開したChatGPTの文の木と自信をそのまま使い、最後の確率の計算だけを入れ替える（FactCheckGPT 631主張、WikiBio 921文） | BTPropが自信より良くなる幅は再現（+0.03／+0.08）。WikiBio：BTProp 0.80 対 ルールも疑うBP 0.77でBTPropの勝ち（5回とも）。FactCheckGPT：0.79 対 0.80で同着 | `plans/E2-10_btprop.md` |
| E2-11 | 同じ木で、NLIのつながりを0〜5割くじで逆にする | BTPropは外れ2〜3割でAIの自信より下に落ちる（5割で0.60〜0.65）。ルールも疑うBPは自信の近くで止まる。FactCheckGPTは外れ2割から、WikiBioは5割で疑うBPの勝ち | `plans/E2-10_btprop.md` |

## ファイルの置き場所

- `colab/`：Google Colabで動かすノート（T4 GPU）。Gemmaに質問して答えを保存する。E2-2・E2-3はノートの中で計算と図まで行う。
  - Colabで開く：[E2-2](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-2_geo_gemma.ipynb) / [E2-3](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-3_wordnet_gemma.ipynb) / [E2-4](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-4_selfrule_gemma.ipynb) / [E2-5](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-5_wording_gemma.ipynb) / [E2-7](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-7_geo_baselines_gemma.ipynb) / [E2-8](https://colab.research.google.com/github/mst-design2020/akane-e2-experiments/blob/main/colab/E2-8_disease_gemma.ipynb)
  - 答えは自分のGoogleドライブに保存されます。Gemmaを使える自分のHugging Faceの鍵（Colabの鍵マークに `HF_TOKEN`）が必要です。
- `data/`：Gemmaの答え（保存したもの）。これがあれば、GPUなしで計算をやり直せる。
  - `geo_mouth_gemma-2-2b-it.json`、`wn_mouth_gemma-2-2b-it.json`：○×問題への答え（Yes−Noのlogit）
  - `E2-4_rules_gemma-2-2b-it.json`、`E2-5_rules_gemma-2-2b-it.json`：ルールの質問への答え
  - `E2-7_answers_gemma-2-2b-it.json`：地理のルールの質問と、言い換えの質問への答え
  - `E2-8_answers_gemma-2-2b-it.json`：病気の○×問題・ルールの質問・組選び（AIが書いた番号）への答え
- `analysis/`：E2-4〜E2-9の計算（CPUのみ）と `check_formulas.py`
- `analysis/btprop/`：E2-10・E2-11。BTPropのコードにはライセンス表示がないので、ここには入れず、`setup_btprop.sh` で決まった版を取ってきて正解ラベルもそろえる
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
# コードが「計算のしかた」の式どおりかを確かめる
python check_formulas.py
```

E2-10・E2-11（BTPropとの比べ、CPUのみで数分。`pyyaml absl-py` と、WikiBioの正解をHugging Faceから取るための `datasets` も使う）：

```
cd analysis/btprop
bash setup_btprop.sh          # BTProp（6e80da8）とFactCheckGPT・WikiBioの正解を取ってくる
cd work/BTProp
python ../../bt_repro.py factcheckgpt                       # 公開どおりのBTProp
python ../../e210.py factcheckgpt ../../E210_factcheckgpt.json
python ../../e210.py wikibio ../../E210_wikibio.json
python ../../e211.py wikibio ../../E211_wikibio.json        # つながりを逆にする
```

WikiBioについて：BTPropは「彼」などを名前に置き換えた文で評価していて、その文は公開されていない。`align_wikibio.py` で、公開の正解と BTPropの文を、順番と文の似かたで突き合わせた（929文中924文）。また、公開のBTPropのコードは、木の無い主張を確率1（本当）にする。FactCheckGPT（48主張）では、そこだけAIの自信を使ったBTPropも出し、それと比べた。

## 近い研究と、違うところ

一番近いのはBTProp（Hou ほか、NAACL 2025、https://arxiv.org/abs/2406.06950 ）です。問題どうしのつながりを別のAI（NLI）の判定で作り、確率伝搬でAIの間違いを見つけます。ここでは、両立しないというルールもAI自身に作らせ、そのルールが本当かどうかも網の中で疑わせました（E2-6）。E2-10でBTPropを、論文の作者が公開した木そのもので動かしました。つながりを元のまま使うと、WikiBioでBTPropが上、FactCheckGPTで同着なので、私たちの方法が優れているとは言えません。つながりの一部をわざと反対にすると（E2-11）、つながりを絶対に正しいとするBTPropは大きく崩れ、ルールも疑うBPは崩れませんでした。BTPropの論文で強い比べ相手とされる「AIの自信」（P(True)）は、ここでも比べ相手に入れています。ほかに近い研究：Crum ほか 2024（AIに両立しないかを聞いてルールを作る）、Xu ほか 2026（近所の答えとの一致）、Jiang・Lowd・Dou 2012（自動で集めた知識をルールで掃除）、Maieutic prompting（Jung ほか 2022）、BeliefBank（Kassner ほか 2021）。E2-9の「ルールを疑うと、ルールが正しいときは少し損で、外れが多いときに効く」という形は、Snorkel（Ratner ほか、VLDB 2017、https://arxiv.org/abs/1711.10160 ）の分析（ルールの当たり具合を学ぶ利点は、ルールの当たり外れがばらばらなときに大きい）と同じ傾向です。ここでは、それがAIが自分の答えを確かめる場面でも見えた、という位置づけです。

## 言えないこと（弱点）

- AIは小さな1種類（Gemma-2 2B）だけで、問題は英語の地理・分類・病気だけ。
- どの組についてルールを聞くかは、E2-7までは人が決めた。E2-8でAIに選ばせたが、番号の位置に引っぱられて8割が外れた。ルールの形は「両立しない」の1つだけ。
- 病気の正解は分類表（Human Disease Ontology）で、表に載っていない＝本当に違う、とは限らない（例：表では新型コロナが呼吸器の病気に入っていない）。
- E2-9の外れは、くじで選んでわざと逆にしたもので、AIが自然に出す外れ（見逃しが多い）とは形が違う。つながりのルールを抜いた確かめは、結果を見たあとで足したもの。
- 同じデータで実験を重ねている（E2-4〜E2-7）。分け方の5通りは同じデータの使い回し。
- E2-7の「言い換えのぶれ」「近所の答えとの一致」は、元の論文の方法を、はい・いいえの1回の答えに合わせて簡単にしたもの。
- E2-10・E2-11の木はChatGPTで作られたもの。私たちの方法はデータの2割で調整し、BTPropは公開の設定（WikiBioの前120人で作った表）のまま。E2-11の外れはくじで逆にしたもので、本物のNLIの間違いとは形が違う。分解のつながりは外していない。E2-10の「調整の幅を広げる」「分解を有向にする」は結果を見たあとで足した確かめ（結論は変わらない）。
- p-bitはパソコンの中で真似たもの（ギブスサンプリング）。速さや電力については、この実験からは何も言えない。

## 作り方について

研究の方向・判断・Colabの実行は著者が行いました。実験の設計、コード、計算、文章は、Anthropic社のAIアシスタント「Claude」と一緒に作りました。

## ライセンス

コードはMITライセンスです（`LICENSE`）。`data/` はGemmaの出力で、Gemmaの利用規約に従います。GeoNames・WordNet・Human Disease Ontology・Factcheck-GPT（Apache 2.0）・WikiBioの正解ラベルは、それぞれのライセンスに従います。BTPropのコードと記録は入れていません。

## 使ったデータとモデル

- GeoNames（`geonamescache` パッケージ経由） https://www.geonames.org/
- WordNet（NLTK経由） https://wordnet.princeton.edu/
- Human Disease Ontology（2026-10-29版、CC0） https://disease-ontology.org/ 。ファイルはリポジトリに入れず、ノートと計算の中でGitHubのタグ v2026-10-29 から取ってくる。
- BTPropのコードと公開記録（6e80da8） https://github.com/UCSB-NLP-Chang/BTProp
- FactCheckGPTの正解（Factbench.jsonl、Wang ほか 2024） https://github.com/yuxiaw/Factcheck-GPT
- WikiBioの正解（Manakul ほか 2023） https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination
- Gemma-2 2B（指示調整版） https://huggingface.co/google/gemma-2-2b-it 。`data/` のGemmaの答えは、Gemmaの利用規約に従って使っています。
