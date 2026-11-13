# E14 先行研究：資料を足して、辞書の鎖の「段不足」を埋める（2026-11-12 夜）

きっかけ：E8n の正しい答えへの「いいえ」82のうち55は点数ほぼ0。中身は「鎖の行き止まり」48と「頭の名詞がない（One who …）」4が大半（E14_count_gaps.md）。

## 1. 辞書から上下関係を取る昔の研究（一番近い）
- **Chodorow, Byrd, Heidorn 1985**（ACL）：Webster's の説明文から頭の語を取り、分類の木を作る。名詞4万、頭の語の正しさ約98%。「one・kind・class・group …＋of」は**中身のない頭（empty head）**として of の後ろを読み直す。→ E8n の「One who …」問題は40年前から知られている。ただし彼らは語の一覧で処理している（この研究の決まりでは使えない）。
- **Guthrie, Slator, Wilks, Bruce 1990**（COLING）「Is There Content in Empty Heads?」：本当に空なのは one・any・kind・type だけ、集まりの語（group など）は中身がある、と分け直した。LDOCE の分野コードで頭の語の意味を選ぶ。
- **Véronis & Ide 1991**（EACL）：**一番近い**。1つの辞書だけだと上下関係の誤り55〜70%（抜けた上位語・切れたつながりを含む）、**5つの辞書を合わせると約6%**。「別の資料を足して抜けを埋める」はここで既に示されている。ただし台所用品100語・人手の評価。
- **Nazar ほか 2021**（J. Intelligent Systems）：分類の木を剪定し、名詞ごとに一番上までの鎖を作り直す（brie→cheese→food→…）。コーパスと小さな上位の骨組みを使う混合法。作り直しの F1 約0.83（英0.76）（要約から、全文未確認）。

## 2. 文の山から「AはBの一種」を取る
- **Hearst 1992 / Roller ほか 2018**（ACL、Hearst Patterns Revisited）：型（"X such as Y" など）で大きな文の山から取ると精度は高いが抜けが多い。抜けは埋め込みで補う。
- **WebIsA（Seitner ほか 2016）/ WebIsALOD（Hertling & Paulheim 2017）**：Common Crawl から58の型で4億組。WordNet から作っていない。ただし無作為500組で正しいのは7.4%と雑音が多く、大きすぎる → 今回は使わない（言えないことに書く）。
- **GenericsKB（Bhakthavatsalam ほか 2020）**：一般的な文。全部の版（generics_kb、343万文）と Best（102万文、うち24.6万文は **WordNet・ConceptNet から合成**）。→ Best は答え漏れなので使わない。全部の版は E10-2・E10-3 で使った物。source 欄で出どころを数えて記録する。

## 3. 言語モデルに上位語を言わせる
- **Jain & Espinosa-Anke 2022**（arXiv 2202.04876）：BERT・GPT-2 に "X is a type of Y" で上位語を選ばせて木を作る（学習なし）。辺の F1 は25〜38と低い。
- **TaxoLLaMA（Moskvoretskii ほか ACL 2024）**：LLaMA-2-7B を WordNet で学習 → WordNet 採点では答え漏れ。遠い祖先の鎖は測っていない。誤りの75%は「広すぎる」上位語。
- **Chain-of-Layer（2024）、LORex（2025）**：大きいAIで木を層ごとに作る・直接の親を当てる。遠い祖先を埋める話ではない。
- **Talmor ほか 2020（Leap-of-Thought）**：モデルが暗に持つ上下関係を足して推論させる。
- 自分の研究：**E10-3** で「一種か」の問いに一般的な文・自分で書いた文を **Gemma に直接見せても** 何も見せないのと同着だった。**E11-2** で鎖を6段に伸ばしても増えなかった（行き止まりのため）。

## 4. 空いている所（見つけた範囲）
- 小さいAI（2B）の答えの点検で、**辞書の鎖の行き止まりだけを、別の資料（一般的な文・自分で書いた文）の「AはBの一種」で継ぎ足す**。継いだ先の語の意味もGemmaに選ばせる。
- 「直接見せる」（E10-3 で効かなかった形）と「鎖に継ぐ」を同じ問題・同じ資料で比べる。
- 外の資料と自分で書いた文を比べる（E10-3 では同着だった）。
- 新しさは小さい：「資料を足すと抜けが埋まる」は Véronis & Ide 1991 が既に示している。言えるのは「小さいAIの点検の鎖でも同じことが起きるか」まで。

## 出典
- https://aclanthology.org/P85-1037/ （Chodorow 1985）
- https://preview.aclanthology.org/fix_video/C90-3025.pdf （Guthrie 1990）
- https://preview.aclanthology.org/fix_video/E91-1040.pdf （Véronis & Ide 1991）
- https://www.degruyterbrill.com/document/doi/10.1515/jisys-2020-0044/html （Nazar 2021）
- https://arxiv.org/abs/1806.03191 （Roller 2018）
- https://iswc2017.ai.wu.ac.at/wp-content/uploads/papers/MainProceedings/235.pdf （WebIsALOD）
- https://huggingface.co/datasets/community-datasets/generics_kb （GenericsKB）
- https://arxiv.org/pdf/2202.04876 （Jain & Espinosa-Anke 2022）
- https://arxiv.org/html/2403.09207v2 （TaxoLLaMA）
- https://arxiv.org/html/2402.07386v2 （Chain-of-Layer）
- https://arxiv.org/pdf/2006.06609 （Leap-of-Thought）
