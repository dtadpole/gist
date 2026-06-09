"""GIST fast forward — two-kernel, batch-tiled, tensor-core. BF16 and TF32 variants.

Why two kernels (see Obsidian §4 and the design discussion):
  The gate L = sigmoid(M @ P) only uses tensor cores if we batch-tile (>=16 rows
  of M in the dot's M-dim) — a single batch is a GEMV. But batch-tiling fused with
  the pool forces a 3D accumulator O[BB, BQ, D] (~192 KB) that wrecks occupancy.
  So we split:

    kernel 1 (gate):  L[b, q, g] = sigmoid( sum_f' M[b,f'] P[f', q*F+g] )
                      a plain batch-tiled tensor-core GEMM  M[B,F] @ P[F,Q*F] -> L[B,Q*F],
                      sigmoid epilogue, L written to HBM (bf16 ~0.56 GB; ~0.3 ms traffic,
                      cheap vs the ~0.9 TFLOP gate compute).
    kernel 2 (pool):  O[b, q, d] = sum_g L[b,q,g] * N[b,g,d]
                      per (batch, q-block); streams g; N = RMSNorm(X) recomputed on
                      the fly (never materialized); accumulator is 2D O[BQ, D] (~12 KB).

  M (mean over D, the gate input) and rrms = rsqrt(mean_D(X^2)+eps) ([B,F], tiny) are
  precomputed (the cheap stats step). RMSNorm has no centering and no bias; the value
  weight W is per-(feature, channel) [F, D] (unified with kernel_lab GAttention).

Precision:
  * precision="bf16": X, P, W in bf16; both dots are bf16 tensor-core with
    fp32 accumulation; L stored bf16; O in x.dtype.
  * precision="tf32": fp32 storage; both dots use input_precision="tf32".
  Validate each variant against a *same-precision* PyTorch reference (atol=rtol=1e-2).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# kernel 0 — stats: M[b,f] = mean_D(X) (gate input), R[b,f] = rrms_D(X) for RMSNorm,
#                   one pass over X.  rrms = rsqrt(mean_D(X^2) + eps)  (no centering)
# --------------------------------------------------------------------------- #
@triton.autotune(
    configs=[triton.Config({"BLOCK_F": bf}, num_warps=w)
             for bf in (8, 16, 32, 64) for w in (2, 4, 8)],
    key=["B", "F", "D"],
)
@triton.jit
def _stats_kernel(
    X, M, R,
    B, F, D,
    sx_b, sx_f, sx_d,
    sm_b, sm_f,
    eps,
    BLOCK_F: tl.constexpr,
    BLOCK_D: tl.constexpr,         # >= D
):
    b = tl.program_id(0)
    pid_f = tl.program_id(1)
    f = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
    f_mask = f < F
    d = tl.arange(0, BLOCK_D)
    d_mask = d < D
    x = tl.load(X + b * sx_b + f[:, None] * sx_f + d[None, :] * sx_d,
                mask=f_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)  # [BF, BD]
    mean = tl.sum(x, axis=1) / D                          # gate input M (mean over D)
    # RMSNorm scale: no centering. Masked lanes load as 0, so x*x is 0 there -> excluded.
    ms = tl.sum(x * x, axis=1) / D                        # mean of squares over D
    rrms = 1.0 / tl.sqrt(ms + eps)
    tl.store(M + b * sm_b + f * sm_f, mean, mask=f_mask)
    tl.store(R + b * sm_b + f * sm_f, rrms, mask=f_mask)


# --------------------------------------------------------------------------- #
# kernel 1 — gate GEMM: L = sigmoid(M @ P)   (L2-grouped, autotuned)
# --------------------------------------------------------------------------- #
def _gate_configs():
    # gate GEMM is M=B(1536), K=F(1497), N=Q*F(191616): N huge, M modest.
    # B-operand P (0.56 GB) is re-read once per M-tile, so LARGE BLOCK_M cuts HBM.
    # accumulator [BLOCK_M, BLOCK_N] fp32 must stay <=~128 KB -> cap the product.
    base = [
        (64, 256, 64, 32, 3, 4),   # bf16 sweep winner (~322 TFLOP/s)
        (64, 256, 64, 16, 3, 4),
        (64, 256, 32, 32, 3, 4),
        (64, 128, 64, 32, 3, 4),
        (128, 256, 64, 16, 3, 8),
        (128, 128, 64, 16, 4, 8),
        (256, 128, 32, 8, 3, 8),   # tf32 sweep winner
        (256, 128, 32, 16, 3, 8),
        (128, 256, 32, 16, 4, 8),
    ]
    return [triton.Config({"BLOCK_M": m, "BLOCK_N": n, "BLOCK_K": k, "GROUP_M": gm},
                          num_stages=s, num_warps=w)
            for (m, n, k, gm, s, w) in base]


@triton.autotune(configs=_gate_configs(), key=["B", "F", "QF", "IS_BF16"])
@triton.jit
def _gate_kernel(
    M, P, L,                       # pointers
    B, F, QF,                      # sizes (QF = Q*F)
    sm_b, sm_f,                    # M strides  [B, F]
    sp_f, sp_c,                    # P strides  [F, QF]
    sl_b, sl_c,                    # L strides  [B, QF]
    IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,         # batch tile (>=16 -> tensor core)
    BLOCK_N: tl.constexpr,         # QF tile
    BLOCK_K: tl.constexpr,         # F-reduction tile
    GROUP_M: tl.constexpr,         # L2-locality swizzle
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(B, BLOCK_M)
    num_pid_n = tl.cdiv(QF, BLOCK_N)
    # grouped ordering of program ids for better L2 reuse of P
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)         # batch rows
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)         # QF columns (q*F + g)
    m_mask = rm < B
    n_mask = rn < QF

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, F, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < F
        a = tl.load(M + rm[:, None] * sm_b + k[None, :] * sm_f,
                    mask=m_mask[:, None] & k_mask[None, :], other=0.0)      # [BM, BK]
        b = tl.load(P + k[:, None] * sp_f + rn[None, :] * sp_c,
                    mask=k_mask[:, None] & n_mask[None, :], other=0.0)      # [BK, BN]
        if IS_BF16:
            acc += tl.dot(a, b, out_dtype=tl.float32)
        else:
            acc += tl.dot(a, b, out_dtype=tl.float32, input_precision="tf32")

    l = tl.sigmoid(acc)
    if IS_BF16:
        l = l.to(tl.bfloat16)
    tl.store(L + rm[:, None] * sl_b + rn[None, :] * sl_c, l,
             mask=m_mask[:, None] & n_mask[None, :])


# --------------------------------------------------------------------------- #
# kernel 2 — pool: O = sum_g L[:,:,g] * N[:,g,:], N = RMSNorm(X) on the fly
#                  (N = X * rrms * W; no centering, per-(feature,channel) W [F,D])
# --------------------------------------------------------------------------- #
def _pool_configs():
    # larger BLOCK_Q -> X re-read fewer times (X is reloaded once per q-block).
    base = [
        (64, 32, 3, 4),
        (128, 32, 2, 4),
        (128, 32, 2, 8),
        (128, 32, 3, 8),
        (64, 64, 2, 4),
        (64, 64, 2, 8),
        (128, 64, 2, 8),
        (128, 64, 2, 4),
    ]
    return [triton.Config({"BLOCK_Q": q, "BLOCK_G": g}, num_stages=s, num_warps=w)
            for (q, g, s, w) in base]


@triton.autotune(configs=_pool_configs(), key=["B", "F", "D", "Q", "IS_BF16"])
@triton.jit
def _pool_kernel(
    L, X, R, W, O,                 # pointers
    B, F, D, Q,                    # sizes
    sl_b, sl_q, sl_g,              # L viewed [B, Q, F]
    sx_b, sx_f, sx_d,              # X [B, F, D]
    sr_b, sr_f,                    # R (rrms) [B, F]
    sw_f, sw_d,                    # W (RMSNorm weight) [F, D]
    so_b, so_q, so_d,              # O [B, Q, D]
    HAS_AFFINE: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,         # >= D
):
    b = tl.program_id(0)
    pid_q = tl.program_id(1)
    q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q < Q
    d = tl.arange(0, BLOCK_D)
    d_mask = d < D

    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
    for g0 in range(0, F, BLOCK_G):
        g = g0 + tl.arange(0, BLOCK_G)
        g_mask = g < F
        # gate tile L[b, q-block, g-block] -> [BQ, BG]
        l = tl.load(L + b * sl_b + q[:, None] * sl_q + g[None, :] * sl_g,
                    mask=q_mask[:, None] & g_mask[None, :], other=0.0)
        # value tile N[b, g-block, :] = RMSNorm(X) on the fly -> [BG, BD]
        x = tl.load(X + b * sx_b + g[:, None] * sx_f + d[None, :] * sx_d,
                    mask=g_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        rg = tl.load(R + b * sr_b + g * sr_f, mask=g_mask, other=0.0).to(tl.float32)
        n = x * rg[:, None]                      # RMSNorm: scale by rrms, no centering
        if HAS_AFFINE:
            w = tl.load(W + g[:, None] * sw_f + d[None, :] * sw_d,
                        mask=g_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)  # [BG, BD]
            n = n * w
        n = tl.where(g_mask[:, None], n, 0.0)   # padded g (g>=F) must not contribute

        if IS_BF16:
            acc += tl.dot(l.to(tl.bfloat16), n.to(tl.bfloat16), out_dtype=tl.float32)
        else:
            acc += tl.dot(l, n, out_dtype=tl.float32, input_precision="tf32")

    o = acc.to(tl.bfloat16) if IS_BF16 else acc
    tl.store(O + b * so_b + q[:, None] * so_q + d[None, :] * so_d, o,
             mask=q_mask[:, None] & d_mask[None, :])


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def gist_triton_fast_forward(
    x: torch.Tensor,                      # [B, F, D]
    proj_P: torch.Tensor,                 # [F, Q*F]
    weight: torch.Tensor | None = None,   # RMSNorm weight [F, D]
    *,
    eps: float = 1e-5,
    precision: str = "bf16",              # "bf16" | "tf32"
) -> torch.Tensor:                         # all kernel tiles are autotuned
    assert precision in ("bf16", "tf32"), precision
    B, F, D = x.shape
    QF = proj_P.shape[1]
    assert proj_P.shape[0] == F and QF % F == 0
    Q = QF // F
    is_bf16 = precision == "bf16"
    BLOCK_D = _next_pow2(D)

    # kernel 0 — stats: M = mean over D (gate input), R = rrms (RMSNorm scale); one pass over X
    m = torch.empty((B, F), device=x.device, dtype=torch.float32)
    r = torch.empty((B, F), device=x.device, dtype=torch.float32)
    stats_grid = lambda META: (B, triton.cdiv(F, META["BLOCK_F"]))
    _stats_kernel[stats_grid](
        x, m, r, B, F, D,
        x.stride(0), x.stride(1), x.stride(2),
        m.stride(0), m.stride(1),
        eps, BLOCK_D=BLOCK_D,
    )

    gate_dtype = torch.bfloat16 if is_bf16 else torch.float32
    Mg = m.to(gate_dtype).contiguous()
    Pg = proj_P.to(gate_dtype).contiguous()              # [F, QF], contiguous for coalescing
    L = torch.empty((B, QF), device=x.device, dtype=gate_dtype)

    # kernel 1 — gate GEMM (autotuned, 1D grouped grid)
    gate_grid = lambda META: (triton.cdiv(B, META["BLOCK_M"]) * triton.cdiv(QF, META["BLOCK_N"]),)
    _gate_kernel[gate_grid](
        Mg, Pg, L, B, F, QF,
        Mg.stride(0), Mg.stride(1),
        Pg.stride(0), Pg.stride(1),
        L.stride(0), L.stride(1),
        IS_BF16=is_bf16,
    )

    has_affine = weight is not None
    if not has_affine:
        weight = x.new_zeros(1, 1)

    L3 = L.view(B, Q, F)                                  # L[b, q, g] = Lflat[b, q*F+g]
    o = torch.empty((B, Q, D), device=x.device, dtype=x.dtype)
    pool_grid = lambda META: (B, triton.cdiv(Q, META["BLOCK_Q"]))
    _pool_kernel[pool_grid](
        L3, x, r, weight, o,
        B, F, D, Q,
        L3.stride(0), L3.stride(1), L3.stride(2),
        x.stride(0), x.stride(1), x.stride(2),
        r.stride(0), r.stride(1),
        weight.stride(0), weight.stride(1),
        o.stride(0), o.stride(1), o.stride(2),
        HAS_AFFINE=has_affine, IS_BF16=is_bf16,
        BLOCK_D=BLOCK_D,
    )
    return o


if __name__ == "__main__":
    from gist_pytorch import GIST, init_unit_scale_

    assert torch.cuda.is_available(), "run on a CUDA box"
    torch.manual_seed(0)
    dev = "cuda"

    # unit-scale init (X~N(0,1); fan-in scaled weights) so O ~ O(1) and tolerances mean
    # something. F=320 multiple of BLOCK_G; B=8 < BLOCK_M exercises the batch-edge mask.
    B, F, D, Q = 8, 320, 192, 64
    ref = init_unit_scale_(GIST(F, D, Q).to(dev).to(torch.float32))
    x32 = torch.randn(B, F, D, device=dev, dtype=torch.float32)
    P32 = ref.proj.weight.t().contiguous()

    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        o32 = ref(x32).float()                       # fp32 ground truth
    print(f"O fp32 truth: std {o32.std():.3f}  |max| {o32.abs().max():.3f}")

    REL_BAR = {"tf32": 1e-2, "bf16": 5e-2}           # rel-Frobenius bar per dtype
    for prec in ("tf32", "bf16"):
        if prec == "bf16":
            o = gist_triton_fast_forward(
                x32.bfloat16(), P32.bfloat16(),
                ref.norm.weight.bfloat16(),
                eps=ref.norm.eps, precision="bf16").float()
        else:
            o = gist_triton_fast_forward(
                x32, P32, ref.norm.weight,
                eps=ref.norm.eps, precision="tf32").float()
        rel_fro = ((o - o32).norm() / o32.norm()).item()
        max_abs = (o - o32).abs().max().item()
        allc = torch.allclose(o, o32, atol=1e-2, rtol=1e-2)
        ok = rel_fro < REL_BAR[prec]
        print(f"[{prec:4s}] rel-Frobenius {rel_fro:.3e} (bar {REL_BAR[prec]:.0e})  "
              f"max-abs {max_abs:.3e}  allclose(1e-2) {allc}  -> {'OK' if ok else 'FAIL'}")
        assert ok, (prec, rel_fro)
    print("OK: bf16 and tf32 kernels match fp32 within their relative-error bars")
