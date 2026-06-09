"""GIST reference Model for the cuda_exec harness — ALL-BF16 (like-to-like with gist.cu).

Inputs (order = config input_shapes): X[B,F,D], P[F,Q*F], W[F,D]  (bf16)
Output: O[B,Q,D] (bf16)

RMSNorm value path (no mean-subtraction, no bias), per-(feature,channel) weight W[F,D]:
  - stats reduction in fp32; mean M (gate input) stored as bf16; rrms = rsqrt(mean_D(X^2)+eps) fp32
  - gate = sigmoid(M_bf16 @ P_bf16): bf16 inputs, fp32 accumulate, L stored bf16
  - N = X * rrms * W  in fp32, materialized bf16 (the WGMMA pool operand)
  - pool O = L_bf16 @ N_bf16: fp32 accumulate, output bf16
"""

from __future__ import annotations

import torch
from torch import nn

BF16 = torch.bfloat16


class Model(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, X, P, W):
        B, F, D = X.shape
        QF = P.shape[1]
        Q = QF // F
        Xf = X.float()

        mean = Xf.mean(dim=-1)               # [B,F]  gate input (mean over D)
        M_bf16 = mean.to(BF16)               # kernel stores M as bf16
        ms = Xf.pow(2).mean(dim=-1)          # [B,F]  mean of squares over D (no centering)
        rrms = torch.rsqrt(ms + self.eps)    # fp32 RMS scale

        logits = (M_bf16 @ P).float()        # bf16 gate (fp32 accumulate)
        L = torch.sigmoid(logits).view(B, Q, F).to(BF16)   # bf16 gate weights

        # RMSNorm: no centering, no bias, per-(feature,channel) weight W[F,D] (bcast over B)
        N = Xf * rrms[:, :, None] * W.float()[None, :, :]   # fp32 values
        N_bf16 = N.to(BF16)                  # kernel materializes N as a bf16 WGMMA operand

        O = torch.bmm(L, N_bf16)             # all-bf16 pool: bf16 operands, fp32 accumulate
        return O.to(X.dtype)


def get_init_inputs():
    return []
