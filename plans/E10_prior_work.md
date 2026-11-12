# E10 先行調査：辞書で確かめるやり方（E8）は、別の種類の問い・別の分野でも効くか（2026-11-11、2回目で深くした版）

読み方の印：【全文】PDFを読んだ（機械で読み取った物を確認）／【要旨】要旨や紹介だけ／【既知】以前の調べ（E6〜E8の先行研究.md）で読んだ物。数字は読んだ物にあった物だけ。2025〜26年の新しい論文は見落としがありうる。
1回目（同じ日の朝）からの追加：部分と全体の古典4本、言葉の関係を当てる研究3本、医療の確かめ5本、地理4本、外の資料が効く場面（PopQA）など4本、手元でWordNetを数えた結果。

---
## 結論（先に）
1. **小さいAIの答えを、辞書の説明文をたどって外から確かめる研究は、「部分と全体」「病気」「地理」のどれでも見つからなかった**（2回調べても同じ）。近いのは、医療の知識グラフで大きなAIの答えを確かめる研究（FAITH 2025）、分類表を検索してでっち上げを消す研究（Born 2026）。どちらも「整った表」を引く方法で、説明文の鎖ではない。
2. **「部分と全体」は、種類をそろえないと鎖が成り立たない**。Winston ほか（1987）は部分と全体を6種類（部品と物、仲間と集まり、ひと切れと全体、材料と物、動作と活動、場所と地域）に分け、**同じ種類どうしなら「AはBの一部、BはCの一部→AはCの一部」が成り立ち、種類が混ざると成り立たない**ことを例で示した（シンプソン氏の腕は彼の一部、彼は哲学科の一部 → 腕は哲学科の一部、とは言えない）。
3. **WordNetの「一部」は種類が混ざっている**（手元で数えた）：「一部」の組9,097の中に、川（Red River → テキサス州）、町（大学町 → 州）、単位（アール → ヘクタール）、果物（りんご → りんごの木）が入っている。また部品は一般的な所に付いている（house の部品は直接4つ、上の分類 building などに38）。→ 正解表は「部品と物」に絞り、上の分類の部品も引き継いで作る必要がある。
4. **説明文に「全体」はよく書かれている**：WordNetの説明文で、よく使う部分の語5,169個のうち59%が全体の語をそのまま含む（"a piece of leather forming the back of a shoe"）。辞書で確かめる手がかりはありそう。ただし「of」は部分以外の意味でもよく使われ、文から部分と全体を取る昔の研究でも「Yの X（of）」の形の当たりは48%しかない（Girju 2006）。
5. **小さいAIは「一部か」の判定で向きを取り違える**：GPT-4でも「XはYの一部か」は69%、LLaMA2は「車輪は車の一部」と「車は車輪の一部」の確からしさを、ほぼ4分の1で区別できない（Proietti & Lenci 2025）。→ 両方の向きを聞く意味がある。
6. **外の資料が一番効くのは、あまり知られていない物**（PopQA, Mallen ほか ACL 2023）：知られていない4,000問ではGPT-3でも19%、小さい2.7Bに検索を足すとGPT-3より上。一方、よく知られた物では検索で10%の問いが壊れた。→ E10でも「よく使う語／まれな語」に分けて測ると、辞書がどこで効くかがはっきりする。
7. **地理は小さいAIほど細かい所と地域の偏りに弱い**：国までは易しいが州・郡で落ちる（Welscher 2026：3Bで平均0.80）、南アジア・東アジアの地名は北米より大きく下がる（GPSBench 2026）。→ 地域別にも結果を出す。
8. **病気の分類表の説明文は答えそのもの**：DOIDの説明文の57%は親の分類名で始まる（手元で数えた、10,070中5,714）。辞書には使えない。

---
## ① 部分と全体（XはYの一部か）

### 古典
- **Winston, Chaffin & Herrmann, 1987, Cognitive Science 11「A Taxonomy of Part-Whole Relations」**【全文】：6種類（上の結論2）。つながりが成り立つのは「両方の段が同じ種類」の時。混ざると崩れる例：腕→シンプソン→哲学科（部品と仲間）、冷蔵庫→台所→家（部品と場所）。「取り付けてある」（指は手に付いている）も「一部」と混同されやすい。数字の実験はない。
- **Berland & Charniak, ACL 1999「Finding Parts in Very Large Corpora」**【全文】：新聞1億語から「Yの X」の言い回しで部分を集めた。上位50語で55%、上位20語で約70%しか正しくない。WordNetにある車の部品（door・engine）が上位に出ない。
- **Girju, Badulescu & Moldovan, CL 2006「Automatic Discovery of Part-Whole Relations」**【全文の8割】：WordNetの組から言い回しを学び、決定木で絞る。全体で正しさ81%・拾える割合76%。ただし言い回しごとでは「Yの X（of・所有）」48%、複合名詞79%。人の判定の一致は0.81〜0.88。「机には4本の脚がある」（部分）と「ケイトには4匹の猫がいる」（持ち物）の取り違えが典型。
- **Bhakthavatsalam ほか, 2020「Do Dogs have Whiskers? A New Knowledge Base of hasPart Relations」**【全文】：一般文から5万の「〜は〜を持つ（部分）」を90%の正しさで集めた。範囲は「部品と物」「材料と物」だけに絞っている。間違いの35%はあいまいな関係、30%は組の取り違え。WordNetの「一部」は9千組だが、つなげて広げると530万に膨らむ（つなぎすぎの心配）。

### AIでの判定
- **Proietti & Lenci, 2025, arXiv 2504.02395**【全文】：上の結論5。AIは部分を挙げるのは得意（GPT-4 97.5%）だが判定は下手。自分で挙げた部分の判定でもGPT-4は81%、LLaMA2-chatは偶然以下。つながり（推移）は「今後」。
- **Pitarch ほか, ACL 2023「No clues, good clues: Out of context Lexical Relation Classification」**【全文】：RoBERTa-largeを練習させると、言葉の関係（CogALex）で「部分」0.784・「上位語」0.718と、部分の方がむしろ当たる。ただし練習ありの小さなモデルで、○×を聞いた小さいAIの話ではない。説明文は使っていない。
- **Cao, Yamada, Teufel, Tokunaga, 2025, Language Resources and Evaluation**【全文の大半】：BERT〜Llama 70Bと人を比べた。挙げた語の正しさは人0.54〜0.90、モデルはふつう0.30前後。部分・下位語では穴埋め型（BERT類）の方が、上位語・全体では文を作る型（Llama類）の方が良い。説明文は使っていない。
- **Zhang, Conia, Rago, 2025, arXiv 2510.13839**【全文】：商品レビューから部品の木をAIで作る。外の辞書での確かめはなし。

## ② 病気
- **OntoLAMA（He ほか 2023, arXiv 2302.06761）**【全文】：病気の分類表（DOID）の「一種か」を、RoBERTa-largeは練習なしで59.8%、128例で93.4%。説明文は「今後」。
- **TaxoGlimpse（Sun ほか 2024, arXiv 2406.11131）**【全文】：難しい問題でGPT-4が健康・生物・地理で0.65前後。深い所ほど悪い。
- **MedLAMA（Meng ほか, ACL 2022「Rewire-then-Probe」）**【要旨】：医学用語集（UMLS）の関係を言葉のモデルに当てさせると、上位10個に正解が入るのは最大3%。医学の関係の知識はモデルの中に薄い。
- **FAITH（Zhou ほか 2025, arXiv 2511.12817）**【要旨】：医療の答えを小さな主張に分け、医学の知識グラフのつながりで点を付けると、医師の判定とよく合う。**外の知識でAIの答えを確かめる、一番近い形**。ただし整った知識グラフを使い、説明文の鎖ではない。小さいAIの自信との比べもない。
- **Born ほか 2026（Clinical NLP @ LREC 2026）**【全文】：大きなAIが医療コードを付けると存在しないコードが1〜5割。分類表を検索して見せるとほぼ0%だが拾える割合が下がる。
- **Hemachandran & Priyadarsini 2026「Asymmetric verification…」（Discover Computing）**【要旨】：医療の主張を、わざと変えた主張と比べて確かめる。うその主張71個中69個を見抜いた。ただし「病名の入れ替え」は見抜きにくい。外の資料は使っていない。
- **Park ほか, EMNLP 2025（ILlama）**【全文の前半】：UMLSの知識グラフ（約2万語・25万組）を検索して医療相談に答える。確かめではなく答えを作る側。
- **E2-8（こちら）**：Gemma 2Bに病気600×分類の○×を聞いた答えがある（3,304問）。ただし「本物の分類全部＋偽4つ」の作り方。

## ③ 地理
- **Welscher ほか, AGILE 2026**【全文】：国 F1 0.99、州0.91、郡0.87。3Bは平均0.80。外の資料で直すことはしていない。
- **GPSBench（Truong, Lau, Qi, 2026, arXiv 2602.16105）**【全文の大半】：14のAI（最小8B）。場所から国を当てる59〜97%、州26〜73%、市1〜23%。南アジア2.6%・東アジア3.8%と北米14.7%で大差。
- **Bhagat ほか 2025（arXiv 2411.07320）**【全文】：貧しい国ほどAIの文の中身が薄い。国の名前が学習データに出る回数と関係。
- **Liu ほか, AGILE 2024**【全文】：Wikipediaの説明文から地名を伏せてGPT-4に当てさせる。国の単位で0.43、地域の単位で0.13。今後としてWikipedia（DBpedia）からの検索を挙げている。
- **GeoHaluBench（Wang ほか, Findings of EMNLP 2025）**【全文】：お店・道の場所。外の資料で確かめることはしていない。
- **E2-2（こちら）**：Gemma 2Bの市→国→大陸の○×の答えがある（3,586問、正しさ0.858）。

## ④ 外の資料・確かめ役についての横断的な研究
- **PopQA（Mallen ほか, ACL 2023「When Not to Trust Language Models」）**【全文】：上の結論6。よく知られた物には検索しない「選んで検索」が一番（GPT-3で46.5%）。
- **MiniCheck（Tang, Laban, Durrett, EMNLP 2024）**【要旨】：7.7億の小さな確かめ役で、GPT-4並みの判定、費用は400分の1。
- **RefChecker（Hu ほか 2024）**【全文】：答えを3つ組に分けて資料とNLIで照らす。NLI型の確かめ役（AlignScore 78.9%）がGPT-4（74.8%）並み。
- **HalluLens（Bang ほか 2025）**【全文の一部】：知られていない物ほどでっち上げが増える（Gemma-2も対象）。
- 既知：TaxoLLaMA（ACL 2024）・Moskvoretskii（SWJ 2026）＝説明文で「一種か」が良くなる。Kovács ほか（NLE 2022/23）＝Wiktionaryの説明文を2段つなぐ。Meconi ほか 2025＝AIに意味を選ばせる。

## ⑤ 手元で確かめた事実（WordNet 3.0、こちらで数えた）
- 「一部」の組：部品型（part）9,097・仲間型（member）12,293・材料型（substance）797。WordNetは種類を3つに分けて持っている。
- 部品型の中にも、場所（川→州、町→州）・単位・果物が混ざる。
- よく使う部分の語（wordfreq 3.0以上）5,169組のうち、説明文が全体の語をそのまま含むのは3,065（59%）、全体の1つ上の分類まで入れると3,369（65%）。仲間型は24%、材料型は43%。
- 部品はよく上の分類に付いている：house（直接4・上の分類に38）、human body（直接0・上に21）、tree（5・10）。
- WordNetには誤りもある（Mississippi州の大学町 Oxford が Georgia州の一部になっている）。→ 常識の採点が要る。
- DOIDの説明文：10,070中5,714（57%）が親の分類名で始まる。

---
## 何が違うか・空いているか（まとめ）
| 近い研究 | していること | E10でする所 |
|---|---|---|
| Proietti 2025 | 「一部か」でAIが向きを間違えると示した | その間違いを外の辞書で直せるかを測る |
| Winston 1987・Girju 2006 | 部分と全体の種類、言い回しの当たりにくさ | 種類が混ざる鎖で確かめがどれだけ崩れるかを数える |
| FAITH 2025・Born 2026 | 整った医療の知識・分類表でAIの答えを確かめる | 分類表と別に作られた辞書の説明文で確かめる（答えを見ない） |
| PopQA 2023 | 知られていない物ほど外の資料が効く | 同じことが辞書の確かめでも言えるかを、語のよく使われ方で分けて測る |
| GPSBench 2026・Welscher 2026 | 地理で小さいAIは細かい所・一部の地域に弱い | 地域ごとに、辞書で直る分と辞書に載っていない分を分ける |
| Kovács 2022・E8 | 「一種か」で説明文をつなぐ | 「一部か」「〜にあるか」へ広げ、設定を変えずに持っていく |

**空き（推測）**：同じ仕組みを設定を変えずに3つ（部分と全体・病気・地理）へ持っていき、効く所と崩れる所（辞書に載っていない／意味の取り違え／部分と全体の種類の混ざり）を並べて測った物は見当たらない。

**言い過ぎないための注意**：部品はどれも昔からある（説明文をたどる・AIに意味を選ばせる・NLIで照らす・外の資料で確かめる）。新しいのは組み方と測り方。

## 参考文献（今回読んだ物）
- Winston, M. E., Chaffin, R., Herrmann, D. (1987). A Taxonomy of Part-Whole Relations. Cognitive Science 11, 417–444.
- Berland, M., Charniak, E. (1999). Finding Parts in Very Large Corpora. ACL 1999, 57–64.
- Girju, R., Badulescu, A., Moldovan, D. (2006). Automatic Discovery of Part-Whole Relations. Computational Linguistics 32(1).
- Bhakthavatsalam, S., Richardson, K., Tandon, N., Clark, P. (2020). Do Dogs have Whiskers? A New Knowledge Base of hasPart Relations. arXiv:2006.07510.
- Proietti, M., Lenci, A. (2025). The quasi-semantic competence of LLMs: a case study on the part-whole relation. arXiv:2504.02395.
- Pitarch, L., Bernad, J., Dranca, L., Bobed Lisbona, C., Gracia, J. (2023). No clues, good clues: Out of context Lexical Relation Classification. ACL 2023, 5607–5625.
- Cao, Z., Yamada, H., Teufel, S., Tokunaga, T. (2025). A comprehensive evaluation of semantic relation knowledge of pretrained language models and humans. Language Resources and Evaluation.
- Zhang, D., Conia, S., Rago, A. (2025). Meronymic Ontology Extraction via Large Language Models. arXiv:2510.13839.
- He, Y., Chen, J., Jiménez-Ruiz, E., Dong, H., Horrocks, I. (2023). Language Model Analysis for Ontology Subsumption Inference. arXiv:2302.06761.
- Sun, Y. ほか (2024). Are Large Language Models a Good Replacement of Taxonomies? arXiv:2406.11131.
- Meng, Z., Liu, F., Shareghi, E., Su, Y., Collins, C., Collier, N. (2022). Rewire-then-Probe: A Contrastive Recipe for Probing Biomedical Knowledge of Pre-trained Language Models. arXiv:2110.08173.
- Zhou, S. ほか (2025). Assessing Automated Fact-Checking for Medical LLM Responses with Knowledge Graphs. arXiv:2511.12817.
- Born, L. ほか (2026). Profiling Hallucinations in Frontier LLMs for Entity Linking to Medical Ontologies. Clinical NLP Workshop @ LREC 2026.
- Hemachandran B., Priyadarsini K. (2026). Asymmetric verification for detecting hallucinated claims in clinical information extraction. Discover Computing 29.
- Park, S.-H. ほか (2025). Leveraging Knowledge Graph-Enhanced LLMs for Context-Aware Medical Consultation. EMNLP 2025.
- Welscher, F., Smith, P., Leppämäki, T., Ilyankou, I. (2026). Should We Ask an LLM? Evaluating Toponym Disambiguation across Administrative Levels. AGILE: GIScience Series 7, 46.
- Truong, T. H., Lau, J. H., Qi, J. (2026). GPSBench: Do Large Language Models Understand GPS Coordinates? arXiv:2602.16105.
- Bhagat, K., Vasisht, K., Pruthi, D. (2025). Richer Output for Richer Countries. arXiv:2411.07320.
- Liu, Z., Janowicz, K., Currier, K., Shi, M. (2024). Measuring Geographic Diversity of Foundation Models with a Natural Language–based Geo-guessing Experiment on GPT-4. AGILE: GIScience Series 5, 38.
- Wang, S. ほか (2025). Mitigating Geospatial Knowledge Hallucination in Large Language Models. Findings of EMNLP 2025.
- Mallen, A., Asai, A., Zhong, V., Das, R., Khashabi, D., Hajishirzi, H. (2023). When Not to Trust Language Models. ACL 2023.
- Tang, L., Laban, P., Durrett, G. (2024). MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents. EMNLP 2024.
- Hu, X. ほか (2024). RefChecker: Reference-based Fine-grained Hallucination Checker and Benchmark for Large Language Models. arXiv:2405.14486.
- Bang, Y. ほか (2025). HalluLens: LLM Hallucination Benchmark. arXiv:2504.17550.
