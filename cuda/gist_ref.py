"""GIST reference Model for the cuda_exec harness — ALL-BF16 (like-to-like with gist.cu).

Inputs (order = config input_shapes): X[B,F,D], P[F,Q*F], gamma[D], beta[D]  (bf16)
Output: O[B,Q,D] (bf16)

Mirrors the CUDA kernel's precision path so the harness allclose(1e-2) compares
bf16-vs-bf16 fairly:
  - stats reduction in fp32 (numerically stable), mean stored as bf16 M
  - gate = sigmoid(M_bf16 @ P_bf16): bf16 inputs, fp32 accumulate, L stored bf16
  - LayerNorm value path: (X - M_bf16)*rstd*gamma + beta in fp32 (rstd fp32)
  - pool O = L_bf16 @ N: fp32 accumulate, output bf16
"""

from __future__ import annotations

import torch
from torch import nn

BF16 = torch.bfloat16


class Model(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, X, P, gamma, beta):
        B, F, D = X.shape
        QF = P.shape[1]
        Q = QF // F
        Xf = X.float()

        mean = Xf.mean(dim=-1)               # fp32 reduction
        M_bf16 = mean.to(BF16)               # kernel stores M as bf16
        var = Xf.var(dim=-1, unbiased=False)
        r = torch.rsqrt(var + self.eps)      # fp32 rstd

        logits = (M_bf16 @ P).float()        # bf16 gate (fp32 accumulate)
        L = torch.sigmoid(logits).view(B, Q, F).to(BF16)   # bf16 gate weights

        N = (Xf - M_bf16.float()[:, :, None]) * r[:, :, None]
        N = N * gamma.float()[None, None, :] + beta.float()[None, None, :]   # fp32 values
        N_bf16 = N.to(BF16)                  # kernel materializes N as a bf16 WGMMA operand

        O = torch.bmm(L, N_bf16)             # all-bf16 pool: bf16 operands, fp32 accumulate
        return O.to(X.dtype)


def get_init_inputs():
    return []
