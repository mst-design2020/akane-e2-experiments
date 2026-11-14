# E18 先行研究：小さいAIの「部分・逆向き・仲間」への思い込みを、辞書の鎖（TADORI）で止められるか

2026-11-14 朝。E17（HyperLex）で、TADORI は全体の順位では 2B に負けたが、「部分」「逆向き」「同じ仲間」への「はい」を大きく減らしていた（結果の後の見方）。ここを、先に決まりを書いて、別の人手のデータで確かめる前に調べた。

## 近い研究
| 研究 | 何をしたか | 今回との違い |
|---|---|---|
| Cao, Yamada, Teufel, Tokunaga（Language Resources and Evaluation 2025, arXiv 2412.01131） | BERT・RoBERTa・Llama-3 8B/70B に、上位語・部分・全体・同義・反対の知識を 10,507問で聞く。人の判定も新しく集めた。 | 向き（上位↔下位、部分↔全体）を区別できるかも測り、反対語以外は全部弱いと報告。人も「上位語」と「部分」を取り違えやすい。**直す方法は出していない**（測るだけ）。 |
| LLM-Hype（VLDB 2025 LLM+Graph ワークショップ） | 大きなAI 12種に「一種か」を聞く。聞き方4つのうち、**説明文を見せる聞き方が一番正しい**。 | 大きなAIだけ。部分・逆向き・仲間との取り違えは調べていない（逆向きは「今後」）。外の辞書をたどることはしない。 |
| Moskvoretskii ほか 2024（Are LLMs good at lexical semantics? 分類の木） | LLM の分類の知識を色々な課題で測る。 | 取り違えの種類ごとの直し方はない。 |
| BLESS（Baroni & Lenci 2011）／BiBLESS・WBLESS（Kiela 2015・Weeds 2014・Nguyen 2017） | 200 の名詞ごとに、上位語・部分・仲間・無関係を人が選んで並べたデータ。向きを入れ替えた版で「向きを分かるか」を測るのが定番。 | 古い「語の近さ」の方法は上位語と仲間・部分を取り違えると何度も報告（Shwartz 2017 など）。小さいAI＋辞書の鎖で測った例は見当たらない。 |
| E17（自分たち） | HyperLex で TADORI は全体では 2B に負け。部分・逆向き・仲間の「はい」は大きく減る（見るだけ）。 | 今回はそれを先に決めた形で確かめる。 |

## 空いていそうな所
- 小さいAIの「一種か」の答えに、外の辞書の鎖で**拒否権**を持たせ、部分・逆向き・仲間の取り違えが減るかを、**AI自身に逆向きも聞く**という安い対抗と並べて測った例は見当たらない（調べた範囲で）。
- LLM-Hype の「説明文を見せると良い」は、TADORI の「説明文をたどる」と同じ向き。ただし相手は大きなAI。

## データの選び方
- **BLESS**（名詞、人が選んだ組）：上位語 1,337・部分 2,943・仲間 3,565・無関係 6,702。逆向きは上位語の組を入れ替えて作る（BiBLESS と同じ作り方）。ライセンスの表示はない→Colab の中で公開の所（Shwartz の UnsupervisedHypernymy の固定版）から取り、リポジトリには入れない。BLESS の語の組は作る時に WordNet・ConceptNet も参考にしている（注意として書く）。
- **EVALution** は今は LDC の有料・契約のデータなので使わない。
- HyperLex は E17 で結果を見てしまったので、**しきい値を決めるだけ**に使う（学習＋確かめの 1,622組）。試験には BLESS だけを使う。HyperLex と同じ組（208組）は BLESS から除く。

## 気をつける所
- 「はい」と言うしきい値を上げるだけでも、取り違えは減る。だから比べる相手は「しきい値をちゃんと決めた 2B」にする（E17 の見え方は、2B のしきい値が 0 のままだった所が混じっている）。
- 2B に「Y は X の一種か」と逆向きにも聞けば、向きの取り違えはそれだけで減るかもしれない。これを対抗に入れる。

出典：
- https://arxiv.org/abs/2412.01131
- https://www.vldb.org/2025/Workshops/VLDB-Workshops-2025/LLM+Graph/LLMGraph-4.pdf
- https://www.inf.uni-hamburg.de/en/inst/ab/lt/publications/2024-moskvoretskii-llmstaxo.pdf
- https://aclanthology.org/W11-2501/ ・ https://sites.google.com/site/geometricalmodels/shared-evaluation
- https://catalog.ldc.upenn.edu/LDC2020T06
