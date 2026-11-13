"""TADORI の模式図（誰が何を受け持つか）。数字は使わない。
python3 fig_tadori.py → figures/TADORI/tadori_en.png, tadori_ja.png
例の説明文は Wiktionary を短くした物（kitten: "A young cat."）。cat の意味の候補は分かりやすく縮めた例。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, INK2, BG = "#0b0b0b", "#52514e", "#fcfcfb"
DICT, GEM, NLI, OUT = "#e9e7e1", "#d6e6fa", "#dff1e3", "#f3efe6"
DICT_E, GEM_E, NLI_E, OUT_E = "#b9b7b0", "#2a78d6", "#3c9a5b", "#b9b7b0"

T = {
 "ja": dict(
  title="TADORI：辞書の説明文をたどり、小さいAIには途中の語の意味だけを選ばせる",
  sub="灰＝辞書（Wiktionary）　青＝小さいAI（Gemma-2 2B）の受け持ち：意味を選ぶだけ　緑＝外の判定（NLI）",
  q="問い\n「X は C の一種？」\n例：kitten / mammal",
  d0="辞書：X の説明文\n\"A young cat.\"\n→ 頭の語 cat",
  g="小さいAI：この説明文での\ncat はどの意味？\n1 動物の猫  2 意地の悪い人\n3 男（俗語） …  → 1 を選ぶ",
  d1="辞書：選んだ意味の\n説明文 → 次の頭の語\n…（3段まで同じことを\nくり返す）",
  n="NLI：各段で\nq ＝ 説明文 ⇒「W は 頭の語 の一種」\na ＝ 説明文 ⇒「W は C の一種」",
  o="点数 ＝ 道ごとの\nq × … × q × a の最大\n（学習に足してもよい）",
  foot="小さいAIは「X は C の一種か」の判定をしない（E15：判定を小さいAIに替えると悪くなった）。分類の木（WordNet の上下関係）は使わない。"),
 "en": dict(
  title="TADORI: follow dictionary definitions; the small LM only picks each word's sense",
  sub="grey = dictionary (Wiktionary)   blue = small LM (Gemma-2 2B): sense choice only   green = external judge (NLI)",
  q="Query\n\"Is X a type of C?\"\ne.g. kitten / mammal",
  d0="Dictionary: definition of X\n\"A young cat.\"\n→ head word: cat",
  g="Small LM: which sense of\n\"cat\" is used here?\n1 the animal  2 a spiteful person\n3 a guy (slang) …  → picks 1",
  d1="Dictionary: definition of\nthe chosen sense → next\nhead word … (repeat,\nup to 3 steps)",
  n="NLI at each step:\nq = definition ⇒ \"W is a type of <head>\"\na = definition ⇒ \"W is a type of C\"",
  o="Score = max over paths\nof q × … × q × a\n(optionally a feature\nfor a learned combiner)",
  foot="The small LM never answers \"Is X a type of C?\" itself (E15: letting it judge made results worse). No WordNet hierarchy is used."),
}

def box(ax, x, y, w, h, text, fc, ec, fs=8.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.35, weight="bold" if bold else "normal")

def arrow(ax, x0, y0, x1, y1, c=INK2, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color=c, lw=1.2, linestyle=ls))

def draw(lang, path):
    if lang == "ja":
        f = "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf"
        font_manager.fontManager.addfont(f); plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    t = T[lang]
    fig = plt.figure(figsize=(11, 4.6), dpi=200); fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.text(0.015, 0.955, t["title"], fontsize=11.5, weight="bold", color=INK, va="top")
    fig.text(0.015, 0.885, t["sub"], fontsize=8, color=INK2, va="top")
    y, h, w = 0.44, 0.30, 0.205
    xs = [0.015, 0.255, 0.495, 0.745]
    box(ax, xs[0], y, 0.20, h, t["q"], OUT, OUT_E)
    box(ax, xs[1], y, w, h, t["d0"], DICT, DICT_E)
    box(ax, xs[2], y, 0.215, h, t["g"], GEM, GEM_E)
    box(ax, xs[3], y, 0.24, h, t["d1"], DICT, DICT_E)
    arrow(ax, xs[0] + 0.20, y + h / 2, xs[1], y + h / 2)
    arrow(ax, xs[1] + w, y + h / 2, xs[2], y + h / 2)
    arrow(ax, xs[2] + 0.215, y + h / 2, xs[3], y + h / 2)
    # NLI と点数
    yb, hb = 0.10, 0.25
    box(ax, 0.255, yb, 0.455, hb, t["n"], NLI, NLI_E)
    box(ax, 0.745, yb, 0.24, hb, t["o"], OUT, OUT_E)
    arrow(ax, xs[1] + w / 2, y, xs[1] + w / 2, yb + hb, c=NLI_E, ls="--")
    arrow(ax, xs[3] + 0.06, y, 0.62, yb + hb, c=NLI_E, ls="--")
    arrow(ax, 0.71, yb + hb / 2, 0.745, yb + hb / 2)
    fig.text(0.015, 0.035, t["foot"], fontsize=7.6, color=INK2, va="bottom")
    fig.savefig(path, facecolor=BG); plt.close(fig)

H = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(H, "..", "..", "figures", "TADORI")
os.makedirs(O, exist_ok=True)
draw("en", f"{O}/tadori_en.png"); draw("ja", f"{O}/tadori_ja.png")
