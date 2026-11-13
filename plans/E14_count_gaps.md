# 数え：E8n の「段が足りない」は何か（2026-11-12 夜、作る前の確かめ）

材料：E8n 方法D の1,000問（700＋300）、一番強い道の掛け算、常識の判定（Claude）。コード analysis/e14/count_gaps.py → rows_D.json。

- 正しい答えへの「いいえ」82（700で52、300で30）。うち点数がほぼ0（<0.05）が55。
- 55の分類：living thing 25・mammal 12・animal 3＝40が「人や動物→生き物」。残り15はばらばら。
- 55がなぜ止まるか（決まりで分けた）：
  - 鎖が途中で行き止まり 48（例 granddaughter→child→descendant→progeny で止まる、bodybuilder→person→individual→human で止まる）
  - 物の語の説明文に頭の名詞がない 4（zoologist "One who studies or practices zoology." など）
  - 辞書に無い 2、その他 1
- 「One who …」「Someone who …」型の説明文が55問の中に20意味ある。頭の名詞が取れない、または isolationist→isolationism のように行為の語へ行ってしまう。辞書から上下関係を取る研究で昔から知られている「中身のない頭（empty head）」の問題と同じ形（要文献確認）。
- E11-2 で6段まで伸ばしても増えなかったのは、段の数でなく「行き止まり」が原因だったため、と読める。
