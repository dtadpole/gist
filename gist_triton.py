r"""GIST (Gated Information Summary and Transformation) — fused Triton forward (forward only).

Mirrors gist_pytorch.GIST but fuses the whole module into one kernel:

    O[b,q,d] = sum_g  sigma( sum_f' M[b,f'] P[f', q*F + g] )  *  N[b,g,d]
                       \--------- gate L[b,q,g] ----------/      \-- RMSNorm --/

    M[b,f]   = mean_D(X[b,f,:])              # gate input (mean over D)
    N[b,g,d] = X[b,g,d]*rrms[b,g] * W[g,d]   # RMSNorm over D (no centering, no bias)
               rrms[b,g] = rsqrt(mean_D(X[b,g,:]^2) + eps)

Fusion wins vs the reference:
  * the gate L [B,Q,F] (~0.3G elems) is computed tile-by-tile on-chip and never
    written to HBM; sigmoid is fused into its epilogue;
  * the RMSNorm output N [B,F,D] is recomputed on the fly, never materialized;
  * M (mean, the gate input) and rrms (the RMS scale) come from one pass over X's D axis.
Only X and P are read; only O is written.

Notation map (see the design note):  Xi[beta] = grid over batch ; Phi[rho] = the
F-axis stream.  Mathematically the query axis is whole (tau == Q, not tiled); the
kernel still tiles Q into BLOCK_Q purely so the gate's P sub-tile fits SRAM.

PERFORMANCE / SRAM (read this):
  * The *full* P [F, Q*F] is ~0.57 GB (bf16) / 1.14 GB (fp32) — it NEVER fits
    shared memory (~100-228 KB/SM); it always streams from HBM tile-by-tile.
  * This draft computes the gate as a per-batch GEMV via `tl.sum`, so its P
    sub-tile [BLOCK_K, BLOCK_Q*BLOCK_G] is REGISTER-resident — it must be kept
    small (defaults below) or it spills / overflows. This version is
    correctness-first, NOT speed-optimal (no tensor cores).
  * Fast version: make both stages `tl.dot` GEMMs and tile the BATCH
    (BLOCK_B >= 16 rows of M) so the gate becomes a tensor-core GEMM
    M[BLOCK_B,F] @ P and reuses P across the batch block. See the design note's
    "Optimize the gate".

NOTE: untested on this machine (no GPU here). Run the __main__ self-test on a
CUDA box to verify against the PyTorch reference.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gist_fwd(
    X, P, M, R, W, O,                    # pointers
    B, Fdim, D, Q,                       # sizes (Fdim = F)
    sx_b, sx_f, sx_d,                    # X strides  [B,F,D]
    sp_f, sp_c,                          # P strides  [F, Q*F]   (P[f, q*F+g])
    sm_b, sm_f,                          # M, R strides [B,F]
    sw_f, sw_d,                          # W strides  [F, D]   (RMSNorm weight)
    so_b, so_q, so_d,                    # O strides  [B,Q,D]
    HAS_AFFINE: tl.constexpr,
    BLOCK_Q: tl.constexpr,               # Q tile (hardware tiling of the whole Q axis)
    BLOCK_G: tl.constexpr,               # pooling F-axis tile (g)
    BLOCK_K: tl.constexpr,               # gate-reduction F-axis tile (f')
    BLOCK_D: tl.constexpr,               # >= D
):
    b = tl.program_id(0)
    pid_q = tl.program_id(1)

    d = tl.arange(0, BLOCK_D)
    dm = d < D

    # flattened (q_local, g_local) layout for the gate tile -> [BLOCK_Q, BLOCK_G]
    qg = tl.arange(0, BLOCK_Q * BLOCK_G)
    qg_q = qg // BLOCK_G                  # local q in this block
    qg_g = qg % BLOCK_G                   # local g in this block
    q_abs_flat = pid_q * BLOCK_Q + qg_q   # absolute q

    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)   # O tile [BLOCK_Q, D]

    # Phi[rho]: stream the F axis (the pooling index g)
    for g0 in range(0, Fdim, BLOCK_G):
        g_abs_flat = g0 + qg_g
        col = q_abs_flat * Fdim + g_abs_flat                       # P column = q*F + g
        col_ok = (q_abs_flat < Q) & (g_abs_flat < Fdim)

        # ---- gate L[q,g] = sigmoid( sum_f' M[b,f'] * P[f', q*F+g] ) : full-F reduction ----
        gate = tl.zeros((BLOCK_Q * BLOCK_G,), dtype=tl.float32)
        for k0 in range(0, Fdim, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            km = k < Fdim
            m_k = tl.load(M + b * sm_b + k * sm_f, mask=km, other=0.0).to(tl.float32)        # [BK]
            p = tl.load(
                P + k[:, None] * sp_f + col[None, :] * sp_c,
                mask=km[:, None] & col_ok[None, :], other=0.0,
            ).to(tl.float32)                                                                  # [BK, BQ*BG]
            gate += tl.sum(m_k[:, None] * p, axis=0)                                          # [BQ*BG]
        gate = tl.sigmoid(gate)
        L = tl.reshape(gate, (BLOCK_Q, BLOCK_G))                                              # [BQ, BG]

        # ---- N (RMSNorm on the fly) for this g-block: no centering, no bias ----
        gg = g0 + tl.arange(0, BLOCK_G)
        ggm = gg < Fdim
        x = tl.load(
            X + b * sx_b + gg[:, None] * sx_f + d[None, :] * sx_d,
            mask=ggm[:, None] & dm[None, :], other=0.0,
        ).to(tl.float32)                                                                      # [BG, D]
        r_g = tl.load(R + b * sm_b + gg * sm_f, mask=ggm, other=0.0).to(tl.float32)           # [BG]
        n = x * r_g[:, None]                                                                  # [BG, D]
        if HAS_AFFINE:
            w = tl.load(
                W + gg[:, None] * sw_f + d[None, :] * sw_d,
                mask=ggm[:, None] & dm[None, :], other=0.0,
            ).to(tl.float32)                                                                  # [BG, D]
            n = n * w
        # padded g rows (g >= F) already give n = 0 (x and r load as 0 there); the
        # explicit zero keeps O clean even if a future affine adds a bias term.
        n = tl.where(ggm[:, None], n, 0.0)

        # ---- O += L @ N  (pooling over g) ----
        acc += tl.dot(L, n, out_dtype=tl.float32, allow_tf32=False)                           # [BQ, D]

    # store O[b, q_block, :]
    q_abs = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    o_ptr = O + b * so_b + q_abs[:, None] * so_q + d[None, :] * so_d
    tl.store(o_ptr, acc, mask=(q_abs[:, None] < Q) & dm[None, :])


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def gist_triton_forward(
    x: torch.Tensor,          # [B, F, D]
    proj_P: torch.Tensor,     # [F, Q*F]   (== nn.Linear(F, Q*F).weight.t())
    weight: torch.Tensor | None = None,   # RMSNorm weight [F, D]
    *,
    eps: float = 1e-5,
    BLOCK_Q: int = 16,   # >= 16 for tl.dot; keep BLOCK_Q*BLOCK_G modest (tl.sum gate is register-resident)
    BLOCK_G: int = 16,
    BLOCK_K: int = 32,
) -> torch.Tensor:
    B, Fdim, D = x.shape
    QF = proj_P.shape[1]
    assert proj_P.shape[0] == Fdim and QF % Fdim == 0
    Q = QF // Fdim

    # cheap stats (tiny [B,F]): M is the gate input (mean over D); R is the RMS scale.
    m = x.mean(dim=-1)                              # [B, F]  gate input
    ms = x.pow(2).mean(dim=-1)                      # [B, F]  mean of squares
    r = torch.rsqrt(ms + eps)                      # [B, F]  RMSNorm scale

    has_affine = weight is not None
    if not has_affine:
        weight = x.new_zeros(1, 1)

    o = torch.empty((B, Q, D), device=x.device, dtype=x.dtype)
    grid = (B, triton.cdiv(Q, BLOCK_Q))
    _gist_fwd[grid](
        x, proj_P, m, r, weight, o,
        B, Fdim, D, Q,
        x.stride(0), x.stride(1), x.stride(2),
        proj_P.stride(0), proj_P.stride(1),
        m.stride(0), m.stride(1),
        weight.stride(0), weight.stride(1),
        o.stride(0), o.stride(1), o.stride(2),
        HAS_AFFINE=has_affine,
        BLOCK_Q=BLOCK_Q, BLOCK_G=BLOCK_G, BLOCK_K=BLOCK_K,
        BLOCK_D=_next_pow2(D),
    )
    return o


if __name__ == "__main__":
    # Correctness self-test vs the PyTorch reference (needs CUDA).
    import torch
    from gist_pytorch import GIST

    assert torch.cuda.is_available(), "run on a CUDA box"
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"

    def check(dtype, tol):
        torch.manual_seed(0)
        # F=80 is NOT a multiple of BLOCK_G=32 -> exercises the padded-g masking path
        # (would catch the affine-leak bug); all block dims >= 16 for tl.dot.
        B, F, D, Q = 4, 80, 64, 16
        ref = GIST(F, D, Q).to(dev).to(dtype)
        x = torch.randn(B, F, D, device=dev, dtype=dtype)
        o_ref = ref(x)
        P = ref.proj.weight.t().contiguous()           # [F, Q*F]
        o_tri = gist_triton_forward(
            x, P, ref.norm.weight, eps=ref.norm.eps,
            BLOCK_Q=16, BLOCK_G=32, BLOCK_K=32,
        )
        err = (o_ref.float() - o_tri.float()).abs().max().item()
        status = "OK  " if err < tol else "FAIL"
        print(f"[{status}] {str(dtype):14s} max abs err = {err:.3e}  (tol {tol:.0e})")
        assert err < tol, (dtype, err)

    # fp32: true fp32 match (the pooling tl.dot uses allow_tf32=False).
    # bf16: the kernel accumulates the gate and pool in fp32, so it diverges from
    # the bf16 reference only at bf16's inherent precision (~1e-2), not a bug.
    check(torch.float32, 1e-3)
    check(torch.bfloat16, 2e-2)
    print("OK: Triton matches PyTorch reference")
