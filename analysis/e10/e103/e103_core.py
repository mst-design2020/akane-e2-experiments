"""E10-3の中身（Colabと手元の両方で使う）。計画：sims/e10/PLAN_E10-3.md。手で書いた語の一覧は使わない。
⑫ Gemmaに自分で文を書かせ、その文を ⑪g と同じ形で見せて聞く（外の資料が要るかの比べ）
⑪f 一般的な文のうち、Gemmaが「問いと同じ意味で使われている」と答えた文だけを見せる（縁起の案）
K 「一種か」の問題（E4-2 の 77,326 主張のうち、常識の判定がある物 501 語の全部の主張）
"""
import re

K_SELF, SELF_NEW_TOKENS = 5, 100
LEAD = re.compile(r"^[\s\-\*•\d\.\)\(]+")   # 行の頭の記号・番号を外す（書式の読み方）


def self_prompt(x, w):
    return (f"Write up to five short, general, factual sentences that mention both \"{x}\" and \"{w}\". "
            f"Write one sentence per line and nothing else.")


def parse_lines(text, k=K_SELF):
    out = []
    for line in text.split("\n"):
        s = LEAD.sub("", line).strip()
        if len(s.split()) >= 3:
            out.append(s)
        if len(out) >= k:
            break
    return out


def sense_prompt(stmt, fact, x, w):
    return (f"Statement: {stmt}\nSentence: {fact}\n\n"
            f"In the sentence, are \"{x}\" and \"{w}\" used with the same meanings as in the statement? Answer Yes or No only.")


def k_stmt(x, cname):
    return f"{x[:1].upper() + x[1:]} is a type of {cname}."   # E4-2 と同じ書き方
