"""GIST (Gated Information Summary and Transformation) — reference PyTorch forward (forward only).

Shapes (design defaults): B=1536, F=1491, D=192, Q=128.

    X = x_input                       [B, F, D]
    M = mean(X, axis=-1)              [B, F]            gate input  (== LayerNorm mean)
    L = sigmoid(M @ P)                [B, Q*F] -> [B, Q, F]    gate   (P: [F, Q*F])
    N = LayerNorm(X, axis=-1)         [B, F, D]
    O = bmm(L, N)                     [B, Q, D]

This is the readable, un-fused reference. It materializes the big intermediates
(L is [B, Q, F] ~ 0.3G elems; N is [B, F, D]); the fused Triton kernel in
gist_triton.py avoids them.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GIST(nn.Module):
    def __init__(
        self,
        n_features: int,   # F
        dim: int,          # D
        n_queries: int,    # Q
        eps: float = 1e-5,
        ln_affine: bool = True,
    ):
        super().__init__()
        self.F = n_features
        self.D = dim
        self.Q = n_queries
        # proj_params P: maps the per-feature mean (length F) to Q*F gate logits.
        # nn.Linear stores weight as [out, in] = [Q*F, F]; the math "P" is weight.T = [F, Q*F].
        self.proj = nn.Linear(n_features, n_queries * n_features, bias=False)
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=ln_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, D]
        B, Fdim, D = x.shape
        assert Fdim == self.F and D == self.D, (x.shape, (self.F, self.D))

        m = x.mean(dim=-1)                       # [B, F]      gate input (= LN mean)
        l = torch.sigmoid(self.proj(m))          # [B, Q*F]
        l = l.view(B, self.Q, Fdim)              # [B, Q, F]   gate
        n = self.norm(x)                         # [B, F, D]   LayerNorm over channels
        o = torch.bmm(l, n)                      # [B, Q, D]
        return o


if __name__ == "__main__":
    torch.manual_seed(0)
    B, F, D, Q = 4, 64, 16, 8
    m = GIST(F, D, Q)
    x = torch.randn(B, F, D)
    o = m(x)
    print("output:", o.shape)   # [4, 8, 16]
    assert o.shape == (B, Q, D)
