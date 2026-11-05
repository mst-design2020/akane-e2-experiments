# E2-5 聞き方を変えて、AIにルールを作らせ直す（WordNet × Gemma-2 2B）
固定：2026-11-05 JST（新しい聞き方の答えを見る前）。E2-4（sims/e2_selfrule/PLAN.md）の続き。

## なぜ
E2-4では「Can something be both a mammal and a bird at the same time?」と聞いたところ、Gemmaはほとんど「はい」と答え、ルールの7割近くが外れた。
先行研究（Crum ほか 2024, arXiv 2410.03235）では、両立しないかをAIに聞くとき、聞き方で結果が大きく変わり、"Can a A be a B?" の形が一番良かった（Gemma 2 9Bで正解率約0.8）。
ただしこの研究は、ルールを作るところまでで、そのルールでAI自身の間違いを見つけることはしていない。

## 問い
聞き方を、今回のテストと関係のない別の組で決めてから使ったとき、Gemmaが作ったルールでつないだ方法（BP・揺れ）は、矛盾を数えるだけの方法や口の自信より、Gemmaの間違いを拾うか。

## 聞き方の候補（4つ、ここで固定）
- P0（E2-4と同じ）："Can something be both a mammal and a bird at the same time? Answer Yes or No only."（はい＝両立する）
- P1（先行研究で一番良かった形）："Can a mammal be a bird? Answer Yes or No only."（はい＝両立する）
- P2（逆向き）："Is it true that no mammal is a bird? Answer Yes or No only."（はい＝両立しない）
- P12：P1とP2の両方が「両立しない」と答えた組だけをルールにする。
どれも、XとYの順番を入れ替えた2回のYes−Noのlogitを平均し、0で区切る（しきい値は0で固定）。

## 聞き方の決め方（テストの正解は使わない）
- テストに出てこない別のラベル18個で、聞き方を決める（決め用の組）。
  - 種類12：amphibian, crustacean, spider, shrub, grass, mushroom, cheese, candy, ship, aircraft, painting, metal
  - 上の分類6：vertebrate, invertebrate, living thing, dairy product, work of art, means of transport
  - 全153組。正解は人が常識で付けた（WordNetでは常識とずれる組があったため）。両立する15組：amphibian–vertebrate, amphibian–living thing, crustacean–invertebrate, crustacean–living thing, spider–invertebrate, spider–living thing, shrub–living thing, grass–living thing, mushroom–living thing, vertebrate–living thing, invertebrate–living thing, cheese–dairy product, ship–means of transport, aircraft–means of transport, painting–work of art。どちらとも言える1組（metal–work of art）は採点から外す。残りは両立しない。
- 決め用の組での「両立する組の正解率」と「両立しない組の正解率」の平均（釣り合い正解率）が一番高い聞き方を、本番の聞き方にする。同点ならP1。
- 本番のテスト（E2-3/E2-4と同じ314組）には、選んだ聞き方を1回だけ使う。

## 比べる条件・合図・物差し・回数
E2-4と同じ（つながりのルールはそのまま、知識のルール＝「両立しない」組だけ変わる）。分け方5通り×p-bitの乱数2通り＝10回。
選ばれなかった聞き方も全部計算して並べる（判定は選んだものだけ）。

## 合格・不合格（先に固定）
- 「AがBに勝つ」＝10回平均のAUROC差が0.02以上、かつ10回中8回以上でAが上。差0.02未満は「同着」。どちらでもなければ「はっきりしない」。
- 主な比べ方：選んだ聞き方のルールで、全体での「BP対数える」と「揺れ対数える」。
  - どちらかが勝てば「聞き方を別の組で決めれば、AIが自分で作ったルールでも、つながりで考えると間違いをよく拾った」と記録する。
  - どちらも勝たなければ「聞き方を変えても、自分で作ったルールでは良さは見えなかった」と記録する。
- 次に見る：BP対口の自信。E2-4（P0）と人のルールとの差。

## 正解表の怪しさへの手当て
WordNetの正解には、常識とずれる組がある（例：dish・drink と man-made object は、WordNetでは両立しないが、普通は料理も飲み物も人が作る）。
そのため、ルールの当たり具合は「全314組」と「誰が見てもはっきりした組」（動物の種類どうし10組、大分類どうし19組、種類とその大分類22組）に分けて出す。

## 言えないこと
- どの組を聞くかは人が決めた（本番ではAIが決める必要がある）。
- ルールの形は「両立しない」だけ、分野はものの分類だけ、口は1種類（2B）、英語。
- E2-4・E2-5と実験を重ねているので、全部の結果を並べて出す。1つの良い結果だけを取り上げない。
- 決め用の組の正解は人が付けた（18ラベル・153組と小さい）。

## 結果（2026-11-05 JST。Gemmaへの質問はColabで実行、残りの計算は手元のPCで実行）
決め用の組（152組）の釣り合い正解率：P0 0.828、P1 0.865、P2 0.971、P12 0.861 → 本番はP2（"Is it true that no X is Y?"）。
本番314組でのP2のルール：正しい判断87%（E2-4のP0は31%）。見逃し38/292、間違ったルール3/22（"no drink is food" "no emotion is a feeling" "no clothing is a man-made object"）。はっきりした組51組で88%。
全体のAUROC（10回平均）：口の自信0.752、数える0.806、MaxSAT0.776、BP0.822、揺れ0.816。
- BP対数える：+0.016（6/10）→ 同着。揺れ対数える：+0.010（4/10）→ 同着。
- BP対口の自信：+0.070（10/10）→ 勝ち（E2-4のP0では同着だった）。
- どのルールも破っていない間違いだけ：BP0.784・揺れ0.780、数える0.697（+0.087・+0.083、10/10で勝ち）。
- 直した後の正解率：口0.879 → BP0.895。
記録のみ：P1 数える0.789・BP0.784・揺れ0.800（同着）、P12 数える0.789・BP0.776（同着）。
→ 先に決めた基準どおり「聞き方を変えても、自分で作ったルールでは（数えるだけに対する）良さは見えなかった」と記録する。
ただし、ルールの正しさは31%→87%に上がり、Gemmaの自信だけには大きく勝つようになった（自分で作ったルールが役に立つようになった）。外れ13%は、わざと外した目盛りの1割（勝ち）と2割（同着）の間で、目盛りどおりの位置。
気づいたこと（判定には使わない）：間違ったルール3本はどれも「本当に両立する組を両立しないとした」もので、飲み物・感情・服のことば全部の正しい答えを押しつぶす向きに働く。ランダムな外れより害が大きい種類の外れ。
図：results/E2-5_図.png
