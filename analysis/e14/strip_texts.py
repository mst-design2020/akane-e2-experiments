"""公開用に、Colab の出力から本文を外す（Wiktionary の説明文と Wikipedia の文は CC BY-SA のため）。
python3 strip_texts.py in.json.gz out.json.gz
- 辞書 L：説明文（def）を外す。頭の語の候補・頭の語・印は残す（比べに使う）。
- 継ぎ足しの文 GR：出どころが Wikipedia（src="w"）の文を外す。GenericsKB の文（CC BY 4.0）と Gemma が書いた文は残す。
比べ（e14_analyze.py・e141_compare.py・e142_compare.py）の数字は、外す前と同じになる（README の「Rerunning」）。"""
import sys, json, gzip


def strip(r):
    for senses in r["L"].values():
        for s in senses:
            s.pop("def", None)
    for v in r.get("GR", {}).values():
        for g in v:
            if g.get("src") == "w":
                g.pop("sent", None)
    return r


if __name__ == "__main__":
    r = json.load(gzip.open(sys.argv[1], "rt", encoding="utf-8"))
    with gzip.open(sys.argv[2], "wt", encoding="utf-8") as f:
        json.dump(strip(r), f, ensure_ascii=False)
