"""E11-2 の箱の形のベクトル（計画 plans/E11-2_deep_chain_and_box.md「手で決めた所」の数のまま）。
E8 の出した組だけで学ぶ：段（意味→頭の名詞の意味、重み＝段の NLI の含意）と、着き（意味→分類名、重み＝着きの NLI の含意）。
WordNet も正解も使わない。箱は Li ほか 2019（ICLR、softbox）の滑らかな体積、P(A の中に B) = vol(A∩B)/vol(A)。"""
import numpy as np
import torch

DIM, EPOCHS, LR, NEG, BETA = 32, 200, 0.01, 5, 1.0
import os
DEVICE = os.environ.get("BOX_DEVICE", "cpu")   # E11-3 の Colab では "cuda"（数は同じ）


def log_vol(lo, hi):
    return torch.nn.functional.softplus(hi - lo, beta=1 / BETA).clamp_min(1e-10).log().sum(-1)


class Box(torch.nn.Module):
    def __init__(self, n, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.z = torch.nn.Parameter((torch.rand(n, DIM, generator=g) * 0.1).to(DEVICE))
        self.d = torch.nn.Parameter((torch.rand(n, DIM, generator=g) * 0.1 + 1.0).to(DEVICE))

    def logp(self, a, b):
        """log P(a は b の一種) = log vol(a∩b) − log vol(a)"""
        za, Za = self.z[a], self.z[a] + torch.nn.functional.softplus(self.d[a])
        zb, Zb = self.z[b], self.z[b] + torch.nn.functional.softplus(self.d[b])
        return (log_vol(torch.maximum(za, zb), torch.minimum(Za, Zb)) - log_vol(za, Za)).clamp(max=0.0)


def train(n, pairs, w, seed, log=None):
    """pairs: (子, 親) の番号の配列、w: 0〜1 の重み（目標の確からしさ）。負の組は親をでたらめに入れ替えて目標0。"""
    torch.manual_seed(seed)
    m = Box(n, seed); opt = torch.optim.Adam(m.parameters(), lr=LR)
    a = torch.tensor(pairs[:, 0], device=DEVICE); b = torch.tensor(pairs[:, 1], device=DEVICE); t = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    rng = np.random.default_rng(seed)
    for ep in range(EPOCHS):
        nb = torch.tensor(rng.integers(0, n, (len(a), NEG)), device=DEVICE)
        lp = m.logp(a, b); ln = m.logp(a[:, None].expand(-1, NEG), nb)
        p = lp.exp().clamp(1e-6, 1 - 1e-6); q = ln.exp().clamp(1e-6, 1 - 1e-6)
        loss = -(t * p.log() + (1 - t) * (1 - p).log()).mean() - (1 - q).log().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if log and ep % 50 == 0: log(f"  箱 種{seed} 回{ep} 損 {loss.item():.4f}")
    return m


def score(m, a, b):
    with torch.no_grad():
        return m.logp(torch.tensor(a, device=DEVICE), torch.tensor(b, device=DEVICE)).exp().cpu().numpy()
