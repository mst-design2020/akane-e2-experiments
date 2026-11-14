# E17 先行研究：人が判定した公開データで「一種か」を測る（11-14 朝）

## 使えるデータ（人が判定した物）
| データ | 中身 | 人の判定か | 入手 | 条件 |
|---|---|---|---|---|
| **HyperLex**（Vulić ほか, Computational Linguistics 2017） | 「XはどれだけYの一種か」を0〜6点で、1組あたり約10人が付けた平均。名詞2,163組・動詞453組。WordNet の関係の種類（一種1〜4段・逆向き・同じ仲間・部分・同義・反対・無関係）を混ぜて選んである | **はい（クラウドの人が点を付けた）** | github ivulic/hyperlex（raw.githubusercontent から取れた）。乱数の分け方（学習1,514・確かめ108・試験541、名詞） | 論文を引用。明示のライセンス表記なし→データは再配布せず、Colab で原本から取る |
| BLESS・EVALution・LEDS・WBLESS・BiBLESS・Shwartz | 一種／それ以外の2値など | 主に WordNet・知識ベースから機械で作り人が点検（人が1組ずつ点を付けたのではない） | facebookresearch/hypernymysuite の data/（CC BY-NC） | 参考止まり（WordNet由来で、今回の弱点を消せない） |
→ 主は HyperLex の名詞。人の平均点との順位の合い方（スピアマンの ρ）が普通の物差し。人どうしの一致は ρ 0.864（上限の目安）。

## 近い先行研究
- **Yin & Roth 2018「Term Definitions Help Hypernymy Detection」**（*SEM）：WordNet の説明文を CNN で読んで一種かを当てる。BLESS 等で AP 0.5〜0.9。説明文を使う点は近いが、学習した小さな網で、生成AIの答えを確かめるのではない。HyperLex は使っていない。
- **Greco ほか 2024「Talking the Talk Does Not Entail Walking the Walk」**（EMNLP Findings）：7〜8B の生成AI（Gemma-7B、Llama 等）に動詞の「一種か」を聞く。WordNet の説明文を問いに足す版もある。HyperLex 動詞で F1 最大0.636。3B以下は試していない。名詞は試していない。
- **HyperLex を学習で解く研究**：Pitarch ほか ACL 2023（RoBERTa を学習、ρ 0.79〜0.83）、Sun ほか ACL 2025（DeBERTa・Llama3-8B を学習、ρ 0.87〜0.90）、LEAR 0.686。どれも HyperLex の学習用の組で学習した数字で、学習なしで聞く今回とは比べられない（上の目安として書くだけ）。GPT-4o を学習なしで分類に使うと大きく落ちる例もある（Sun 2025 の関係分類）。
- 古典：Vulić 2017 の論文に、学習なしの方法（単語ベクトルの近さ、WordNet の道の長さ等）の ρ が載っている（WordNet 由来の方法は、組の選び方が WordNet なので有利）。

## 空いている所
- 学習なしの小さい生成AI（2B）の「一種か」の答えを、辞書の説明文の鎖で外から確かめて、人の点数にどれだけ近づくか。
- 2B ＋ TADORI が、同じ聞き方の 9B にどこまで近づくか（大きなAIに聞けば済むのでは、への答え）。
- TADORI を足すと学習ありの組み合わせ（HyperLex の学習用の組を使う）が良くなるか。

## 注意
- HyperLex の組は WordNet の関係で選んである（一種・部分・同じ仲間など）。点数は人。WordNet で答えを引く方法は、組の選び方の分だけ有利になる → 参考で並べ、主の比べには入れない。
- 「部分」「同じ仲間」の組は、人の点が中くらい（例 idol/person 2.57）になる物がある。TADORI は 0/1 寄りの値を出すので、順位の合い方には不利な可能性。
