"""EXPERIMENT: gate + pool FUSED into one kernel, batch-tiled, 3D accumulator.

Stats (M, rrms) stay a separate kernel (reused from gist_triton_fast) per the plan.
The point of fusing gate+pool is to NOT materialize L [B, Q*F] (~0.56 GB) to HBM.

One program owns a [BLOCK_M batches x BLOCK_Q queries] tile and produces O for all D,
streaming the pooled axis g. Because the gate must be a tensor-core GEMM (>=16 batch
rows), the program carries a 3D accumulator acc[BLOCK_M, BLOCK_Q, D]:

    acc[m,q,d] += sum_g  sigmoid( sum_f' M[m,f'] P[f', q*F+g] ) * N[m,g,d]

    gate:  G[BLOCK_M, BQ*BG] = M_tile @ P_tile         (2D tl.dot, tensor core)
    pool:  acc[BLOCK_M,BQ,D] += L[BLOCK_M,BQ,BG] @ N[BLOCK_M,BG,D]   (batched 3D tl.dot)

Caveat: acc and the N tile both scale with BLOCK_M, so on-chip state is large
(acc[16,16,192] alone = 192 KB) -> low occupancy. This file measures whether the
saved L round-trip beats that penalty vs the split design (gist_triton_fast).

RESULT (H100, B=1536 F=1491 D=192 Q=128, bf16): FUSED = 82.1 ms vs SPLIT = 4.28 ms
-> 19x SLOWER (numerically identical, rel 3.5e-3). The 3D acc + 3D N tile force
occupancy=1 and cap BLOCK_M at 16 (so P is re-read B/16 = 96x, vs the split's
BLOCK_M=64 + GROUP_M with L2 reuse). The ~0.4 ms saved L round-trip is dwarfed.
Conclusion: keep gate and pool as SEPARATE kernels. Fusion does NOT help here.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from gist_triton_fast import _stats_kernel, _next_pow2


@triton.jit
def _fused_kernel(
    M, R, X, P, W, O,
    B, F, D, Q,
    sm_b, sm_f,
    sx_b, sx_f, sx_d,
    sp_f, sp_c,
    sw_f, sw_d,
    so_b, so_q, so_d,
    HAS_AFFINE: tl.constexpr, IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_Q: tl.constexpr,
    BLOCK_G: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_q = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)        # batches
    rq = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)        # queries
    m_mask = rm < B
    q_mask = rq < Q
    d = tl.arange(0, BLOCK_D)
    d_mask = d < D

    acc = tl.zeros((BLOCK_M, BLOCK_Q, BLOCK_D), dtype=tl.float32)   # <-- 3D accumulator

    for g0 in range(0, F, BLOCK_G):
        g = g0 + tl.arange(0, BLOCK_G)
        g_mask = g < F

        # ---- gate G[BLOCK_M, BQ*BG] = sum_f' M[rm,f'] P[f', rq*F+g] (2D tensor-core dot) ----
        cols = (rq[:, None] * F + g[None, :]).reshape(BLOCK_Q * BLOCK_G)             # [BQ*BG]
        col_ok = ((rq[:, None] < Q) & (g[None, :] < F)).reshape(BLOCK_Q * BLOCK_G)
        G = tl.zeros((BLOCK_M, BLOCK_Q * BLOCK_G), dtype=tl.float32)
        for k0 in range(0, F, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            k_mask = k < F
            a = tl.load(M + rm[:, None] * sm_b + k[None, :] * sm_f,
                        mask=m_mask[:, None] & k_mask[None, :], other=0.0)           # [BM,BK] fp32
            b = tl.load(P + k[:, None] * sp_f + cols[None, :] * sp_c,
                        mask=k_mask[:, None] & col_ok[None, :], other=0.0)           # [BK,BQ*BG]
            if IS_BF16:
                G += tl.dot(a.to(tl.bfloat16), b)
            else:
                G += tl.dot(a, b, input_precision="tf32")
        L = tl.sigmoid(G).reshape(BLOCK_M, BLOCK_Q, BLOCK_G)                          # [BM,BQ,BG]

        # ---- N[BLOCK_M, BG, BD] = RMSNorm(X) on the fly (no centering, no bias) ----
        x = tl.load(X + rm[:, None, None] * sx_b + g[None, :, None] * sx_f + d[None, None, :] * sx_d,
                    mask=m_mask[:, None, None] & g_mask[None, :, None] & d_mask[None, None, :],
                    other=0.0).to(tl.float32)                                        # [BM,BG,BD]
        rg = tl.load(R + rm[:, None] * sm_b + g[None, :] * sm_f,
                     mask=m_mask[:, None] & g_mask[None, :], other=0.0)              # [BM,BG]
        n = x * rg[:, :, None]                                                       # RMSNorm scale
        if HAS_AFFINE:
            w = tl.load(W + g[:, None] * sw_f + d[None, :] * sw_d,
                        mask=g_mask[:, None] & d_mask[None, :], other=0.0)           # [BG,BD]
            n = n * w[None, :, :]
        n = tl.where(g_mask[None, :, None], n, 0.0)                                  # padded g -> 0

        # ---- pool acc[BM,BQ,BD] += L[BM,BQ,BG] @ N[BM,BG,BD] (batched 3D dot) ----
        if IS_BF16:
            acc += tl.dot(L.to(tl.bfloat16), n.to(tl.bfloat16))
        else:
            acc += tl.dot(L, n, input_precision="tf32")

    o = acc.to(tl.bfloat16) if IS_BF16 else acc
    tl.store(O + rm[:, None, None] * so_b + rq[None, :, None] * so_q + d[None, None, :] * so_d, o,
             mask=m_mask[:, None, None] & q_mask[None, :, None] & d_mask[None, None, :])


def gist_triton_fused_forward(
    x, proj_P, weight=None, *,
    eps=1e-5, precision="bf16",
    BLOCK_M=16, BLOCK_Q=16, BLOCK_G=16, BLOCK_K=32,
    num_warps=8, num_stages=1,
):
    assert precision in ("bf16", "tf32")
    B, F, D = x.shape
    QF = proj_P.shape[1]; Q = QF // F
    is_bf16 = precision == "bf16"
    BLOCK_D = _next_pow2(D)

    # stats kernel (separate, reused)
    m = torch.empty((B, F), device=x.device, dtype=torch.float32)
    r = torch.empty((B, F), device=x.device, dtype=torch.float32)
    _stats_kernel[lambda META: (B, triton.cdiv(F, META["BLOCK_F"]))](
        x, m, r, B, F, D, x.stride(0), x.stride(1), x.stride(2),
        m.stride(0), m.stride(1), eps, BLOCK_D=BLOCK_D)

    Pg = proj_P.to(torch.bfloat16 if is_bf16 else torch.float32).contiguous()
    has_affine = weight is not None
    if not has_affine:
        weight = x.new_zeros(1, 1)

    o = torch.empty((B, Q, D), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(B, BLOCK_M), triton.cdiv(Q, BLOCK_Q))
    _fused_kernel[grid](
        m, r, x, Pg, weight, o, B, F, D, Q,
        m.stride(0), m.stride(1),
        x.stride(0), x.stride(1), x.stride(2),
        Pg.stride(0), Pg.stride(1),
        weight.stride(0), weight.stride(1),
        o.stride(0), o.stride(1), o.stride(2),
        HAS_AFFINE=has_affine, IS_BF16=is_bf16,
        BLOCK_M=BLOCK_M, BLOCK_Q=BLOCK_Q, BLOCK_G=BLOCK_G, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o


if __name__ == "__main__":
    from gist_pytorch import GIST, init_unit_scale_
    assert torch.cuda.is_available()
    torch.manual_seed(13); dev = "cuda"

    # correctness on a small shape; sweep block sizes to find any config that fits SRAM
    B, F, D, Q = 32, 323, 192, 64
    ref = init_unit_scale_(GIST(F, D, Q).to(dev).float()).eval()
    x = torch.randn(B, F, D, device=dev)
    P = ref.proj.weight.t().contiguous()
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        o32 = ref(x).float()

    CONFIGS = [
        (16, 16, 16, 32), (16, 16, 8, 32), (16, 16, 8, 16),
        (16, 8, 16, 32), (16, 8, 8, 32), (32, 16, 8, 32),
    ]
    for (bm, bq, bg, bk) in CONFIGS:
        try:
            o = gist_triton_fused_forward(
                x.bfloat16(), P.bfloat16(),
                ref.norm.weight.bfloat16(),
                eps=ref.norm.eps, precision="bf16",
                BLOCK_M=bm, BLOCK_Q=bq, BLOCK_G=bg, BLOCK_K=bk)
            rel = ((o.float() - o32).norm() / o32.norm()).item()
            print(f"M{bm} Q{bq} G{bg} K{bk}: FITS  rel-Frob {rel:.3e}  {'OK' if rel < 5e-2 else 'FAIL'}")
        except Exception as e:
            msg = str(e)
            req = ""
            if "Required:" in msg:
                req = " (" + msg.split("Required:")[1].split(',')[0].strip() + " B shared)"
            print(f"M{bm} Q{bq} G{bg} K{bk}: {type(e).__name__}{req}")
