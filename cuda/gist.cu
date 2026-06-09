/*
 * GIST forward — optimized CUDA kernel for the cuda_exec harness (v2, RMSNorm).
 *
 * inputs[0]=X[B,F,D], inputs[1]=P[F,Q*F], inputs[2]=W[F,D]  (RMSNorm weight, bf16)
 * outputs[0]=O[B,Q,D].  Dims from env CUDA_EXEC_PARAM_GIST_{B,F,D,Q}.
 *
 *   stats : M[b,f]=mean_d X, R[b,f]=rrms_d X = rsqrt(mean_d(X^2)+eps)  (warp-per-row)
 *   gate  : L[b,q,g]=sigmoid(M[b,:] @ P[:,q*F+g])  (WMMA bf16 tensor-core GEMM)
 *   pool  : O[b,q,d]=sum_g L[b,q,g] * N[b,g,d]   (tiled shared-memory GEMM;
 *           N = RMSNorm(X) = X * rrms * W[f,d] recomputed on the fly; no centering, no bias)
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <cstdlib>
#include <cmath>
#include <cstdint>

#include <cublas_v2.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/arch/mma_sm90_gmma.hpp"   // cute::warpgroup_fence_operand/arrive/commit_batch/wait (small-acc wgmma fix)

using namespace nvcuda;

// ---- CUTLASS 3.x WGMMA GEMM for the gate: L_raw = M_pad @ P_pad (bf16, fp32 acc) ----
// K is padded to a multiple of 8 (TMA/WGMMA alignment). sigmoid is applied later in
// the pool when reading L (free). This is the WGMMA path mma.sync (WMMA) cannot reach.
namespace gategemm {
using cute::Shape; using cute::_1; using cute::_2; using cute::_32; using cute::_64; using cute::_128; using cute::_256;
// LayoutA = ColumnMajor for M[B,F]: its contiguous dim is then M=B=1536 (16B-aligned),
// so K=F=1491 (the stride dim) needs NO padding. P (B-operand, RowMajor [F,QF]) already
// has K=F as its stride dim. => the gate runs unpadded; no k_padM/k_padP, no P copy.
using ElementA = cutlass::bfloat16_t; using LayoutA = cutlass::layout::ColumnMajor;
using ElementB = cutlass::bfloat16_t; using LayoutB = cutlass::layout::RowMajor;
using ElementC = cutlass::bfloat16_t; using LayoutC = cutlass::layout::RowMajor;
using ElementAcc = float;
constexpr int AlignA = 8, AlignB = 8, AlignC = 8;
using TileShape = Shape<_128, _256, _64>;
using ClusterShape = Shape<_1, _1, _1>;
// sigmoid fused into the gate epilogue (free: overlaps the memory-bound mainloop) -> L = sigmoid(logits),
// so the downstream "sigpad" becomes a pure repad (no 293M exp). LinCombEltAct<Sigmoid> needs an EXPLICIT
// (non-Auto) cooperative schedule.
using SigmoidFusion = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::Sigmoid, ElementC, ElementAcc, ElementC, ElementAcc>;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, ElementAcc,
    ElementC, LayoutC, AlignC, ElementC, LayoutC, AlignC,
    cutlass::epilogue::TmaWarpSpecializedCooperative, SigmoidFusion>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB, ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename CollectiveEpilogue::SharedStorage)>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}  // namespace gategemm

// NO-SIGMOID gate (probe + rebalance candidate): identical to gategemm but the epilogue is the default
// LinearCombination (raw logits out, no 293M exp in the epilogue). Used to MEASURE the sigmoid-epilogue
// cost on the gate's tensor pipeline (sigmoid is then applied downstream by k_sigpad).
namespace gategemm_ns {
using cute::Shape; using cute::_1; using cute::_64; using cute::_128; using cute::_256;
using ElementA = cutlass::bfloat16_t; using LayoutA = cutlass::layout::ColumnMajor;
using ElementB = cutlass::bfloat16_t; using LayoutB = cutlass::layout::RowMajor;
using ElementC = cutlass::bfloat16_t; using LayoutC = cutlass::layout::RowMajor;
using ElementAcc = float;
constexpr int AlignA = 8, AlignB = 8, AlignC = 8;
using TileShape = Shape<_128, _256, _64>;
using ClusterShape = Shape<_1, _1, _1>;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, ElementAcc,
    ElementC, LayoutC, AlignC, ElementC, LayoutC, AlignC,
    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB, ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename CollectiveEpilogue::SharedStorage)>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}  // namespace gategemm_ns

// ---- CUTLASS 3.x WGMMA *batched* GEMM for the pool: O[b] = SL[b] @ N[b] ----
// per batch: [M=Q, N=D, K=Fp]; L (batch) = B. SL = sigmoid(gate logits) padded to
// [B,Q,Fp]; N = LayerNorm(X) padded to [B,Fp,D]. Both K=Fp so the F=1491 tail is
// zeroed and does not change the sum. RowMajor everywhere (contiguous dims aligned).
namespace poolgemm {
using cute::Shape; using cute::_1; using cute::_64; using cute::_128; using cute::_192; using cute::_256;
using ElementA = cutlass::bfloat16_t; using LayoutA = cutlass::layout::RowMajor;
using ElementB = cutlass::bfloat16_t; using LayoutB = cutlass::layout::RowMajor;
using ElementC = cutlass::bfloat16_t; using LayoutC = cutlass::layout::RowMajor;
using ElementAcc = float;
constexpr int AlignA = 8, AlignB = 8, AlignC = 8;
using TileShape = Shape<_128, _192, _64>;     // N tile = D = 192 exactly (no waste)
using ClusterShape = Shape<_1, _1, _1>;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, ElementAcc,
    ElementC, LayoutC, AlignC, ElementC, LayoutC, AlignC,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB, ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename CollectiveEpilogue::SharedStorage)>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}  // namespace poolgemm

// NOTE: an SM80 pool reading UNPADDED L (no repad) is impossible for odd F=1491 — SM80 cp.async needs
// >=4-byte (>=2-element) alignment but lda=F=1491 is odd ("Size is not supported"); SM90 TMA needs
// 8-align. So a tensor-core pool MUST use the Fp-padded SL (the repad). The repad is the efficient way
// to pay the odd-F alignment cost once, not waste.

namespace {
__device__ __forceinline__ float b2f(__nv_bfloat16 x) { return __bfloat162float(x); }

// ---- hand-written Hopper WGMMA helpers (inline PTX, verified vs torch: 0 err) ----
// no-swizzle (INTERLEAVE) 64-bit matrix descriptor (cute/arch/mma_sm90_desc.hpp bit layout)
__device__ __forceinline__ uint64_t wg_encode(uint32_t x) { return (uint64_t)((x & 0x3FFFF) >> 4); }
__device__ __forceinline__ uint64_t wg_desc(const void* smem_ptr, uint32_t LBO_bytes, uint32_t SBO_bytes) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t d = 0;
    d |= wg_encode(a);
    d |= wg_encode(LBO_bytes) << 16;
    d |= wg_encode(SBO_bytes) << 32;   // layout_type=0 (no swizzle), base_offset=0
    return d;
}
// D[64,64] += A[64,16]*B[16,64]; A K-major (tnspA=0), B MN-major (tnspB=1); fp32 acc d[32].
__device__ __forceinline__ void wgmma_m64n64k16(float* d, uint64_t da, uint64_t db, int scaleD) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
        " %32, %33, p, %35, %36, %37, %38;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),  "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(da), "l"(db), "r"(scaleD), "n"(1), "n"(1), "n"(0), "n"(1));
}
// D[64,96] += A[64,16]*B[16,96]; A K-major (tnspA=0), B MN-major (tnspB=1); fp32 acc d[48].
__device__ __forceinline__ void wgmma_m64n96k16(float* d, uint64_t da, uint64_t db, int scaleD) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %50, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, "
        " %48, %49, p, %51, %52, %53, %54;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),  "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47])
        : "l"(da), "l"(db), "r"(scaleD), "n"(1), "n"(1), "n"(0), "n"(1));
}
// commit + wait_group 0, binding the 32 fp32 accumulators as "+f" so the compiler keeps them live
// across the async wgmma completion (otherwise it may reuse those regs before the async write lands
// -> silent corruption of part of the tile, seen as warp-3 rows wrong with small accumulators).
__device__ __forceinline__ void wgmma_commit_wait32(float* d) {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),  "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        :: "memory");
}
// cute small-acc[32] wgmma atom + the canonical fence/arrive/commit/wait machinery (cute::warpgroup_*).
// ROOT CAUSE of the long-standing "ptxas warp-3 bug" (rows 48-63 of a small acc corrupted across many
// commit/wait groups): it was NEVER a ptxas register bug. The small-acc tile's smem operands (A,B) are
// produced here with GENERIC-proxy stores (regular __float2bfloat16 writes), but wgmma.mma_async reads
// them through the ASYNC proxy. __syncthreads()/wgmma.fence only order the generic proxy / wgmma register
// accesses — NOT cross-proxy visibility. Without `fence.proxy.async.shared::cta` the wgmma can read STALE
// smem, a relaxed-ordering RACE that intermittently corrupts (warp 3 was just where it surfaced). Verified
// in /tmp/cute_smallacc_test.cu: no proxy fence -> nondeterministic warp-3 errors up to 0.19 (3/12 runs
// corrupt); WITH the proxy fence -> 12/12 runs clean at the fp32 floor (max 0.0005). CUTLASS's collective
// avoids this because TMA already writes smem via the async proxy; a generic-store producer must fence.
// The per-float warpgroup_fence_operand (compiler barrier) + arrive/commit/wait are also retained (the
// canonical accumulator-liveness machinery), but the decisive fix is the proxy fence.
namespace cute_gmma = cute::SM90::GMMA;
using GistGmmaAtom = cute::SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<
    cute_gmma::Major::K, cute_gmma::Major::MN, cute_gmma::ScaleIn::One, cute_gmma::ScaleIn::One>;
__device__ __forceinline__ void wgmma_cute_fence32(float* d) {
    #pragma unroll
    for (int i = 0; i < 32; i++) cute::warpgroup_fence_operand(d[i]);
}
// Make generic-proxy smem writes (the staged A/B operands) visible to wgmma's async-proxy reads.
__device__ __forceinline__ void wgmma_async_proxy_fence() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_cute_m64n64k16(float* d, uint64_t da, uint64_t db) {
    GistGmmaAtom::fma(da, db,
        d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15],
        d[16],d[17],d[18],d[19],d[20],d[21],d[22],d[23],d[24],d[25],d[26],d[27],d[28],d[29],d[30],d[31],
        cute_gmma::ScaleOut::One);   // scaleD=1, always accumulate
}

// stats + fused RMSNorm, one warp per RPW rows over B*Fp rows (one X read).
// Writes M[b,f] (bf16, COLUMN-major for the unpadded gate), R[b,f]=rrms (fp32), and N_pad[b,f,:]
// (bf16, the pool's WGMMA B-operand; pad rows f in [F,Fp) -> 0). NITER=ceil((D/2)/32)=3 for D=192.
constexpr int RPW = 1, NITER = 3;

// stats-only variant: writes ONLY M[b,f] and R[b,f]=rrms, skips N_pad (for fused pool path).
__global__ void k_stats_mr(const __nv_bfloat16* __restrict__ X,
                           __nv_bfloat16* __restrict__ M, float* __restrict__ R,
                           int B, int F, int Fp, int D, float eps) {
    int lane = threadIdx.x & 31;
    long warpId = (long)blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    long total = (long)B * Fp, base = warpId * RPW;
    int D2 = D >> 1;
    const __nv_bfloat162* X2base = reinterpret_cast<const __nv_bfloat162*>(X);
    __nv_bfloat162 xv[RPW][NITER];
    float s[RPW], s2[RPW]; int bb[RPW], ff[RPW]; bool ok[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; r++) { s[r] = 0.f; s2[r] = 0.f; long prow = base + r;
        ok[r] = prow < total; bb[r] = ok[r] ? prow / Fp : 0; ff[r] = ok[r] ? prow % Fp : 0; }
    #pragma unroll
    for (int r = 0; r < RPW; r++) if (ok[r] && ff[r] < F) {
        const __nv_bfloat162* x2 = X2base + ((long)bb[r] * F + ff[r]) * D2;
        #pragma unroll
        for (int i = 0; i < NITER; i++) { int d2 = lane + i * 32;
            if (d2 < D2) { __nv_bfloat162 v = x2[d2]; xv[r][i] = v;
                float a = b2f(v.x), c = b2f(v.y); s[r] += a + c; s2[r] += a * a + c * c; } }
    }
    #pragma unroll
    for (int r = 0; r < RPW; r++) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) { s[r] += __shfl_down_sync(0xffffffff, s[r], o); s2[r] += __shfl_down_sync(0xffffffff, s2[r], o); }
        s[r] = __shfl_sync(0xffffffff, s[r], 0); s2[r] = __shfl_sync(0xffffffff, s2[r], 0);
    }
    #pragma unroll
    for (int r = 0; r < RPW; r++) {
        if (!ok[r] || ff[r] >= F) continue;
        float m = s[r] / D;                          // gate input (mean over D) — unchanged
        float rrms = rsqrtf(s2[r] / D + eps);        // RMSNorm: rsqrt(mean(X^2)+eps), no -m*m centering
        if (lane == 0) { M[(long)ff[r] * B + bb[r]] = __float2bfloat16(m); R[(long)bb[r] * F + ff[r]] = rrms; }
    }
}
// N-only RMSNorm: reads X + R(rrms) + W[F,D] -> Npad[B,Fp,D]. Runs CONCURRENTLY
// with the gate on a 2nd stream (the gate needs only M, and is at 19% DRAM -> N's traffic hides in the
// spare bandwidth). f>=F rows -> 0. One warp per (b,f) row. N = X * rrms * W[f,d] (no centering, no bias).
__global__ void k_stats_n(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ M,
                          const float* __restrict__ R, __nv_bfloat16* __restrict__ Npad,
                          const __nv_bfloat16* __restrict__ W,
                          int B, int F, int Fp, int D) {
    (void)M;
    long warpId = (long)blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    long total = (long)B * Fp;
    if (warpId >= total) return;
    int lane = threadIdx.x & 31, D2 = D >> 1;
    int b = warpId / Fp, f = warpId % Fp;
    __nv_bfloat162* n2 = reinterpret_cast<__nv_bfloat162*>(Npad + warpId * D);
    if (f >= F) {
        for (int i = 0; i < NITER; i++) { int d2 = lane + i * 32; if (d2 < D2) n2[d2] = __float2bfloat162_rn(0.f); }
        return;
    }
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(W + (long)f * D);   // per-(feature,channel) weight row
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(X + ((long)b * F + f) * D);
    float rrms = R[(long)b * F + f];
    for (int i = 0; i < NITER; i++) { int d2 = lane + i * 32;
        if (d2 < D2) { __nv_bfloat162 v = x2[d2], w = w2[d2];
            float n0 = b2f(v.x) * rrms * b2f(w.x);
            float n1 = b2f(v.y) * rrms * b2f(w.y);
            n2[d2] = __halves2bfloat162(__float2bfloat16(n0), __float2bfloat16(n1)); } }
}

__global__ void k_stats(const __nv_bfloat16* __restrict__ X,
                        __nv_bfloat16* __restrict__ M, float* __restrict__ R,
                        __nv_bfloat16* __restrict__ Npad,
                        const __nv_bfloat16* __restrict__ W,
                        int B, int F, int Fp, int D, float eps) {
    int lane = threadIdx.x & 31;
    long warpId = (long)blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    long total = (long)B * Fp, base = warpId * RPW;
    int D2 = D >> 1;
    __nv_bfloat162 xv[RPW][NITER];
    float s[RPW], s2[RPW]; int bb[RPW], ff[RPW]; bool ok[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; r++) { s[r] = 0.f; s2[r] = 0.f; long prow = base + r;
        ok[r] = prow < total; bb[r] = ok[r] ? prow / Fp : 0; ff[r] = ok[r] ? prow % Fp : 0; }
    #pragma unroll
    for (int r = 0; r < RPW; r++) if (ok[r] && ff[r] < F) {
        const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(X + ((long)bb[r] * F + ff[r]) * D);
        #pragma unroll
        for (int i = 0; i < NITER; i++) { int d2 = lane + i * 32;
            if (d2 < D2) { __nv_bfloat162 v = x2[d2]; xv[r][i] = v;
                float a = b2f(v.x), c = b2f(v.y); s[r] += a + c; s2[r] += a * a + c * c; } }
    }
    #pragma unroll
    for (int r = 0; r < RPW; r++) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) { s[r] += __shfl_down_sync(0xffffffff, s[r], o); s2[r] += __shfl_down_sync(0xffffffff, s2[r], o); }
        s[r] = __shfl_sync(0xffffffff, s[r], 0); s2[r] = __shfl_sync(0xffffffff, s2[r], 0);
    }
    #pragma unroll
    for (int r = 0; r < RPW; r++) {
        if (!ok[r]) continue;
        __nv_bfloat162* n2 = reinterpret_cast<__nv_bfloat162*>(Npad + (base + r) * D);
        if (ff[r] >= F) {
            #pragma unroll
            for (int i = 0; i < NITER; i++) { int d2 = lane + i * 32; if (d2 < D2) n2[d2] = __float2bfloat162_rn(0.f); }
            continue;
        }
        float m = s[r] / D;                          // gate input (mean over D) — unchanged
        float rrms = rsqrtf(s2[r] / D + eps);        // RMSNorm: rsqrt(mean(X^2)+eps), no -m*m centering
        if (lane == 0) { M[(long)ff[r] * B + bb[r]] = __float2bfloat16(m); R[(long)bb[r] * F + ff[r]] = rrms; }  // M column-major [B,F]
        const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(W + (long)ff[r] * D);  // per-(feature,channel) weight row
        #pragma unroll
        for (int i = 0; i < NITER; i++) { int d2 = lane + i * 32;
            if (d2 < D2) { __nv_bfloat162 v = xv[r][i], w = w2[d2];
                float n0 = b2f(v.x) * rrms * b2f(w.x);     // RMSNorm: X * rrms * W[f,d], no centering, no bias
                float n1 = b2f(v.y) * rrms * b2f(w.y);
                n2[d2] = __halves2bfloat162(__float2bfloat16(n0), __float2bfloat16(n1)); } }
    }
}

// ---------------- gate: CUTLASS WGMMA GEMM  L_raw = M @ P (K padded for TMA) ----------------
// sigmoid is applied later in the pool when reading L (free). K (F) is padded to Fp
// (multiple of 8) so M_pad/P_pad satisfy TMA/WGMMA 16-byte alignment; the F..Fp-1 tail
// is zero so it does not change M@P.
using StrideA = typename gategemm::Gemm::GemmKernel::StrideA;
using StrideB = typename gategemm::Gemm::GemmKernel::StrideB;
using StrideC = typename gategemm::Gemm::GemmKernel::StrideC;
using StrideD = typename gategemm::Gemm::GemmKernel::StrideD;

__global__ void k_padM(const __nv_bfloat16* __restrict__ M, __nv_bfloat16* __restrict__ Mp,
                       int B, int F, int Fp) {
    long tot = (long)B * Fp;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < tot; idx += (long)gridDim.x * blockDim.x) {
        int b = idx / Fp, f = idx % Fp;
        Mp[idx] = (f < F) ? M[(long)b * F + f] : __float2bfloat16(0.f);
    }
}
__global__ void k_padP(const __nv_bfloat16* __restrict__ P, __nv_bfloat16* __restrict__ Pp,
                       int F, int Fp, long QF) {
    long tot = (long)Fp * QF;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < tot; idx += (long)gridDim.x * blockDim.x) {
        long r = idx / QF, c = idx % QF;
        Pp[idx] = (r < F) ? P[(long)r * QF + c] : __float2bfloat16(0.f);
    }
}

// cuBLAS gate: L[B,QF] = M[B,F] @ P[F,QF] (M,P row-major; bf16 operands, fp32 accumulate).
// Column-major view: C[QF,B] = P^T[QF,F] @ M^T[F,B] with C=L(ldc=QF), A=P(lda=QF), B=M(ldb=F).
static cublasHandle_t g_cublas = nullptr;
static int run_gate_cublas(const __nv_bfloat16* M, const __nv_bfloat16* P, __nv_bfloat16* L,
                           int Bm, int F, long QF, cudaStream_t stream) {
    if (!g_cublas && cublasCreate(&g_cublas) != CUBLAS_STATUS_SUCCESS) return 40;
    cublasSetStream(g_cublas, stream);
    float alpha = 1.f, beta = 0.f; int N = (int)QF;
    cublasStatus_t st = cublasGemmEx(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, Bm, F, &alpha,
        P, CUDA_R_16BF, N, M, CUDA_R_16BF, F, &beta, L, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    return st == CUBLAS_STATUS_SUCCESS ? 0 : 41;
}

static void* g_ws = nullptr; static size_t g_ws_sz = 0;
// FALLBACK: StrideD type is flat Stride<int64,_1,int64>, cannot directly support hierarchical
// (Q,F) indexing. Approach 1 (custom EVT epilogue visitor) is too complex for this iteration.
// Keep the flat g_L output + k_repad for now. Future: investigate TMA descriptor swizzle or
// a minimal scatter-epilogue based on gather_scatter_fusion example.
static int run_gate(const __nv_bfloat16* M, const __nv_bfloat16* P, __nv_bfloat16* L,
                    int Bm, int F, long QF, int Q, int Fp, cudaStream_t stream) {
    using namespace gategemm;
    (void)Q; (void)Fp;  // unused in flat layout path
    int N = (int)QF;
    StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(Bm, F, 1));  // ColumnMajor M[B,F]
    StrideB sb = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, F, 1));   // RowMajor P[F,QF]
    StrideC sc = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(Bm, N, 1));
    StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(Bm, N, 1));
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {Bm, N, F},
        {reinterpret_cast<ElementA const*>(M), sa, reinterpret_cast<ElementB const*>(P), sb},
        {{1.0f, 0.0f}, reinterpret_cast<ElementC const*>(L), sc, reinterpret_cast<ElementC*>(L), sd}};
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 20;
    size_t need = Gemm::get_workspace_size(args);
    if (need > g_ws_sz) { if (g_ws) cudaFree(g_ws); if (cudaMalloc(&g_ws, need) != cudaSuccess) return 23; g_ws_sz = need; }
    if (need) cudaMemsetAsync(g_ws, 0, need, stream);   // zero scheduler atomic counters
    if (gemm.initialize(args, g_ws, stream) != cutlass::Status::kSuccess) return 21;
    if (gemm.run(stream) != cutlass::Status::kSuccess) return 22;
    return 0;
}

// NO-SIGMOID gate: writes RAW logits (sigmoid applied downstream by k_sigpad). Probe for the
// sigmoid-epilogue cost on the gate's tensor pipeline.
static void* g_ws_ns = nullptr; static size_t g_ws_ns_sz = 0;
static int run_gate_ns(const __nv_bfloat16* M, const __nv_bfloat16* P, __nv_bfloat16* L,
                       int Bm, int F, long QF, cudaStream_t stream) {
    using namespace gategemm_ns;
    int N = (int)QF;
    StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(Bm, F, 1));
    StrideB sb = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, F, 1));
    StrideC sc = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(Bm, N, 1));
    StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(Bm, N, 1));
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {Bm, N, F},
        {reinterpret_cast<ElementA const*>(M), sa, reinterpret_cast<ElementB const*>(P), sb},
        {{1.0f, 0.0f}, reinterpret_cast<ElementC const*>(L), sc, reinterpret_cast<ElementC*>(L), sd}};
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 24;
    size_t need = Gemm::get_workspace_size(args);
    if (need > g_ws_ns_sz) { if (g_ws_ns) cudaFree(g_ws_ns); if (cudaMalloc(&g_ws_ns, need) != cudaSuccess) return 27; g_ws_ns_sz = need; }
    if (need) cudaMemsetAsync(g_ws_ns, 0, need, stream);
    if (gemm.initialize(args, g_ws_ns, stream) != cutlass::Status::kSuccess) return 25;
    if (gemm.run(stream) != cutlass::Status::kSuccess) return 26;
    return 0;
}

// -------- sigmoid + pad: SL[b,q,f<F]=sigmoid(L[b,q,f]); SL[..,f>=F]=0 --------
// L (gate logits) is [B,Q,F] (col q*F+f); SL is [B,Q,Fp] (col q*Fp+f), the pool's
// WGMMA A-operand (contiguous Fp aligned). The f in [F,Fp) tail is zero.
// one block per (b,q) row; threads stride over f. No per-element integer division.
__global__ void k_sigpad(const __nv_bfloat16* __restrict__ L, __nv_bfloat16* __restrict__ SL,
                         int B, int Q, int F, int Fp) {
    long nrows = (long)B * Q;
    for (long row = blockIdx.x; row < nrows; row += gridDim.x) {
        const __nv_bfloat16* lr = L + row * F;
        __nv_bfloat16* sr = SL + row * Fp;
        for (int f = threadIdx.x; f < Fp; f += blockDim.x)
            sr[f] = (f < F) ? __float2bfloat16(1.f / (1.f + __expf(-b2f(lr[f])))) : __float2bfloat16(0.f);
    }
}

// VECTORIZED sigmoid+pad (branch C): read 8 consecutive g_L rows as ONE 16B-aligned contiguous block
// (8*F is a multiple of 8, and 8-aligned row groups start 16B-aligned) -> smem; then write each SL row
// with 16B-aligned vectorized stores (SL row stride Fp is 16B-aligned). Both DRAM sides vectorized — the
// odd-F (1491) per-row misalignment that forced scalar 2B access is DODGED by the 8-row grouping. Sigmoid
// applied in registers. Replaces the 0.78ms scalar k_sigpad (46% BW). nrows=B*Q is a multiple of 8.
constexpr int SP_RPB = 8;
__global__ void k_sigpad_v2(const __nv_bfloat16* __restrict__ L, __nv_bfloat16* __restrict__ SL,
                            int B, int Q, int F, int Fp) {
    long nrows = (long)B * Q;
    long ngroups = (nrows + SP_RPB - 1) / SP_RPB;
    extern __shared__ __nv_bfloat16 spsm[];          // SP_RPB * F raw logits
    int tid = threadIdx.x, nt = blockDim.x;
    int nfc = F / 8;                                  // full 16B chunks per row
    for (long g = blockIdx.x; g < ngroups; g += gridDim.x) {
        long r0 = g * SP_RPB;
        int rows_here = (int)min((long)SP_RPB, nrows - r0);
        int nelem = rows_here * F;
        const __nv_bfloat16* Lbase = L + r0 * F;     // 16B-aligned (r0 % 8 == 0)
        const uint4* Lv = (const uint4*)Lbase;
        uint4* smv = (uint4*)spsm;
        int nch = nelem / 8;
        for (int c = tid; c < nch; c += nt) smv[c] = Lv[c];           // aligned 16B reads
        for (int t = nch * 8 + tid; t < nelem; t += nt) spsm[t] = Lbase[t];
        __syncthreads();
        for (int k = 0; k < rows_here; k++) {
            const __nv_bfloat16* src = spsm + (long)k * F;
            __nv_bfloat16* dst = SL + (r0 + k) * Fp;                  // 16B-aligned row
            for (int fc = tid; fc < nfc; fc += nt) {
                __nv_bfloat16 out[8];
                const __nv_bfloat16* s = src + fc * 8;
                #pragma unroll
                for (int j = 0; j < 8; j++) out[j] = __float2bfloat16(1.f / (1.f + __expf(-b2f(s[j]))));
                *(uint4*)(dst + fc * 8) = *(const uint4*)out;          // aligned 16B write
            }
            for (int f = nfc * 8 + tid; f < F; f += nt)
                dst[f] = __float2bfloat16(1.f / (1.f + __expf(-b2f(src[f]))));
        }
        __syncthreads();
    }
}

// repad ONLY (no exp): L is already sigmoid'd (gate epilogue). SL[b,q,f<F]=L[b,q,f]; f>=F -> 0.
// L row stride F (=1491 ODD), SL row stride Fp. Per-row L+row*F is NOT guaranteed 16-byte aligned
// (odd F -> every other row is misaligned). SAFE approach: one block per (b,q) row, threads stride
// over f with SCALAR bf16 loads/stores (no vectorization). Bandwidth-limited but correct.
__global__ void k_repad(const __nv_bfloat16* __restrict__ L, __nv_bfloat16* __restrict__ SL,
                        int B, int Q, int F, int Fp) {
    long nrows = (long)B * Q;
    for (long row = blockIdx.x; row < nrows; row += gridDim.x) {
        const __nv_bfloat16* lr = L + row * F;
        __nv_bfloat16* sr = SL + row * Fp;
        for (int f = threadIdx.x; f < F; f += blockDim.x)
            sr[f] = lr[f];
    }
}

// FUSED sigmoid+repad: read RAW gate logits g_L[B,Q*F] (flat RowMajor from CUTLASS gate
// WITHOUT sigmoid fusion), apply sigmoid, and write directly to g_SL[B,Q,Fp] (padded layout).
// This REPLACES both the sigmoid epilogue AND k_repad, eliminating the 0.79ms repad overhead.
// One block per (b,q) row. Threads stride over F. F=1491, Fp=1536.
__global__ void k_sigpad_direct(const __nv_bfloat16* __restrict__ L, __nv_bfloat16* __restrict__ SL,
                                 int B, int Q, int F, int Fp) {
    long row = blockIdx.x;  // one block per row
    long nrows = (long)B * Q;
    if (row >= nrows) return;
    int b = row / Q, q = row % Q;
    const __nv_bfloat16* lr = L + ((long)b * Q + q) * F;  // flat input: row stride F
    __nv_bfloat16* sr = SL + ((long)b * Q + q) * Fp;       // padded output: row stride Fp
    for (int f = threadIdx.x; f < F; f += blockDim.x) {
        float logit = b2f(lr[f]);
        sr[f] = __float2bfloat16(1.f / (1.f + __expf(-logit)));  // sigmoid
    }
    // pad [F,Fp) is pre-zeroed at alloc; no write needed
}


// ---------------- pool: CUTLASS WGMMA batched GEMM  O[b] = SL[b] @ Npad[b] ----------------
using StrideA2 = typename poolgemm::Gemm::GemmKernel::StrideA;
using StrideB2 = typename poolgemm::Gemm::GemmKernel::StrideB;
using StrideC2 = typename poolgemm::Gemm::GemmKernel::StrideC;
using StrideD2 = typename poolgemm::Gemm::GemmKernel::StrideD;
static void* g_ws2 = nullptr; static size_t g_ws2_sz = 0;
static int run_pool(const __nv_bfloat16* SL, const __nv_bfloat16* Npad, __nv_bfloat16* O,
                    int Bm, int Q, int D, int Fp, cudaStream_t stream) {
    using namespace poolgemm;
    // per-batch problem [M=Q, N=D, K=Fp], batch L=Bm; strides carry the batch stride.
    StrideA2 sa = cutlass::make_cute_packed_stride(StrideA2{}, cute::make_shape(Q, Fp, Bm));
    StrideB2 sb = cutlass::make_cute_packed_stride(StrideB2{}, cute::make_shape(D, Fp, Bm));
    StrideC2 sc = cutlass::make_cute_packed_stride(StrideC2{}, cute::make_shape(Q, D, Bm));
    StrideD2 sd = cutlass::make_cute_packed_stride(StrideD2{}, cute::make_shape(Q, D, Bm));
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {Q, D, Fp, Bm},
        {reinterpret_cast<ElementA const*>(SL), sa, reinterpret_cast<ElementB const*>(Npad), sb},
        {{1.0f, 0.0f}, reinterpret_cast<ElementC const*>(O), sc, reinterpret_cast<ElementC*>(O), sd}};
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 30;
    size_t need = Gemm::get_workspace_size(args);
    if (need > g_ws2_sz) { if (g_ws2) cudaFree(g_ws2); if (cudaMalloc(&g_ws2, need) != cudaSuccess) return 33; g_ws2_sz = need; }
    if (need) cudaMemsetAsync(g_ws2, 0, need, stream);
    if (gemm.initialize(args, g_ws2, stream) != cutlass::Status::kSuccess) return 31;
    if (gemm.run(stream) != cutlass::Status::kSuccess) return 32;
    return 0;
}



// ---------------- FUSED pool: cooperative cp.async WMMA, D-split, KT=64 K-tile ----------------
// O[b, :Q, doff:doff+DPB]. Reads SL=sigmoid(gate) padded [B,Q,Fp] + X (this block's D cols),
// computes N=LayerNorm(X) inline. KT=64-row K-tile -> KSUB=4 WMMA sub-tiles per iter -> 4x
// fewer __syncthreads than a 16-row tile (matches Triton's BLOCK_G). D-split -> 2 blocks/SM.
constexpr int WM = 16, WN = 16, WK = 16, DMAX = 192, FPMAX = 1536, DPW = 3, MAXW = 16;
constexpr int DPB = 96, KT = 64, KSUB = KT / WK, NLDB = DPB + 4, NCOLB = DPB / WN, DGB = NCOLB / DPW;
__global__ void k_pool_fused(const __nv_bfloat16* __restrict__ SL, const __nv_bfloat16* __restrict__ X,
                             const __nv_bfloat16* __restrict__ M, const float* __restrict__ R,
                             const __nv_bfloat16* __restrict__ W,
                             __nv_bfloat16* __restrict__ O, int B, int F, int Fp, int D, int Q) {
    (void)M;
    int b = blockIdx.x, doff = blockIdx.y * DPB, tid = threadIdx.x, nt = blockDim.x, warp = tid >> 5, lane = tid & 31;
    int warp_q = warp / DGB, warp_d = warp % DGB, q0 = warp_q * WM, cbase = warp_d * DPW, qg = Q / WM;
    __shared__ float Rall[FPMAX];
    __shared__ __nv_bfloat16 Xsh[2][KT * DPB], SLsh[2][8 * (WM * KT)], Nsh[KT * NLDB];
    __shared__ float Otmp[MAXW * (WM * WN)];
    for (int f = tid; f < Fp; f += nt) { Rall[f] = (f < F) ? R[(long)b * F + f] : 0.f; }
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc[DPW];
    #pragma unroll
    for (int c = 0; c < DPW; c++) wmma::fill_fragment(acc[c], 0.f);
    int ntiles = (F + KT - 1) / KT, nchunkX = (KT * DPB) / 8, nchunkSL = qg * WM * (KT / 8);
    {   for (int i = tid; i < nchunkX; i += nt) { int fl = (i * 8) / DPB, dl = (i * 8) - fl * DPB;
            if (fl < F) __pipeline_memcpy_async(&Xsh[0][i * 8], &X[((long)b * F + fl) * D + doff + dl], 16); }
        for (int i = tid; i < nchunkSL; i += nt) { int q = i / (KT / 8), c8 = i % (KT / 8);
            __pipeline_memcpy_async(&SLsh[0][q * KT + c8 * 8], &SL[((long)b * Q + q) * Fp + c8 * 8], 16); }
    }
    __pipeline_commit();
    for (int k = 0; k < ntiles; k++) {
        if (k + 1 < ntiles) { int g1 = (k + 1) * KT, buf = (k + 1) & 1;
            for (int i = tid; i < nchunkX; i += nt) { int fl = (i * 8) / DPB, dl = (i * 8) - fl * DPB;
                if (g1 + fl < F) __pipeline_memcpy_async(&Xsh[buf][i * 8], &X[((long)b * F + g1 + fl) * D + doff + dl], 16); }
            for (int i = tid; i < nchunkSL; i += nt) { int q = i / (KT / 8), c8 = i % (KT / 8);
                __pipeline_memcpy_async(&SLsh[buf][q * KT + c8 * 8], &SL[((long)b * Q + q) * Fp + g1 + c8 * 8], 16); }
            __pipeline_commit();
        }
        __pipeline_wait_prior(k + 1 < ntiles ? 1 : 0);
        __syncthreads();
        int cur = k & 1, g0 = k * KT;
        const __nv_bfloat16* xs = Xsh[cur];
        for (int i = tid; i < KT * DPB; i += nt) { int fl = i / DPB, dl = i - fl * DPB; int f = g0 + fl;
            Nsh[fl * NLDB + dl] = (f < F) ? __float2bfloat16(b2f(xs[i]) * Rall[f] * b2f(W[(long)f * D + doff + dl])) : __float2bfloat16(0.f); }
        __syncthreads();
        #pragma unroll
        for (int ks = 0; ks < KSUB; ks++) {
            wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, SLsh[cur] + warp_q * (WM * KT) + ks * WK, KT);
            #pragma unroll
            for (int c = 0; c < DPW; c++) {
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, Nsh + ks * WK * NLDB + (cbase + c) * WN, NLDB);
                wmma::mma_sync(acc[c], a_frag, b_frag, acc[c]);
            }
        }
        __syncthreads();
    }
    float* ot = Otmp + warp * (WM * WN);
    for (int c = 0; c < DPW; c++) {
        wmma::store_matrix_sync(ot, acc[c], WN, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < WM * WN; i += 32) { int ql = i >> 4, dl = i & 15;
            O[((long)b * Q + q0 + ql) * D + doff + (cbase + c) * WN + dl] = __float2bfloat16(ot[i]); }
        __syncwarp();
    }
}

// ---------------- WARP-SPECIALIZED fused pool: producer/consumer, named barriers ----------------
// The cooperative fused pool floors at ~1.9ms / 2 blocks/SM because the N-compute and the
// mma.sync share the same warps with a __syncthreads barrier between them — mma.sync is
// SYNCHRONOUS, so within a warp the N-compute cannot overlap the tensor-core MMA. Triton's
// tl.dot reaches ~0.8ms because its compiler warp-specializes. Here we do it by hand:
//   PRODUCER warps  (8 warps): regular-load X this d-tile, compute N=LayerNorm inline ->
//                              double-buffered Nsh[buf].  Their DRAM-load latency is hidden
//                              by the consumers running on other warps of the same block.
//   CONSUMER warps  (8 warps): one per q-tile.  Load a_frag (sigmoid'd SL, padded) directly
//                              from global, b_frag from Nsh[buf], mma.sync over NCOL d-tiles.
// Handoff via PTX named barriers (no __syncthreads): full[buf] = N(buf) ready (producers
// arrive, consumers wait); empty[buf] = buffer free (consumers arrive, producers wait).
// D-split (DPB=96 -> 2 blocks per b).  16 warps = 512 threads = 1 block/SM.
__device__ __forceinline__ void ws_bar_arrive(int id) { asm volatile("bar.arrive %0, 512;" :: "r"(id)); }
__device__ __forceinline__ void ws_bar_wait(int id)   { asm volatile("bar.sync   %0, 512;" :: "r"(id)); }
__device__ __forceinline__ void ws_bar_prod()         { asm volatile("bar.sync   5, 256;"); }  // producers only
constexpr int WS_DPB = 96, WS_KT = 64, WS_NCONS = 8, WS_NPROD = 8;
constexpr int WS_KSUB = WS_KT / 16, WS_NCOL = WS_DPB / 16, WS_NLDB = WS_DPB + 4;
__global__ __launch_bounds__(512, 1)
void k_pool_warpspec(const __nv_bfloat16* __restrict__ SL, const __nv_bfloat16* __restrict__ X,
                     const __nv_bfloat16* __restrict__ M, const float* __restrict__ R,
                     const __nv_bfloat16* __restrict__ W,
                     __nv_bfloat16* __restrict__ O, int B, int F, int Fp, int D, int Q) {
    (void)M;
    int b = blockIdx.x, doff = blockIdx.y * WS_DPB, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    bool isProd = warp >= WS_NCONS;             // warps 0-7 = consumers, 8-15 = producers
    int pw = warp - WS_NCONS, cw = warp;        // producer idx / consumer idx (= q-tile)
    __shared__ float Rall[FPMAX];
    __shared__ __nv_bfloat16 Xsh[2][WS_KT * WS_DPB], Nsh[2][WS_KT * WS_NLDB];
    __shared__ float Otmp[WS_NCONS * (WM * WN)];
    for (int f = tid; f < Fp; f += 512) { Rall[f] = (f < F) ? R[(long)b * F + f] : 0.f; }
    __syncthreads();                            // Rall ready for all warps
    int ntiles = (F + WS_KT - 1) / WS_KT;

    if (isProd) {
        int ptid = pw * 32 + lane;              // 0..255 over the 8 producer warps
        int nchunkX = (WS_KT * WS_DPB) / 8;     // 8 bf16 (16B) per cp.async chunk
        for (int i = ptid; i < nchunkX; i += 256) {     // prologue: cp.async X(0) -> Xsh[0]
            int fl = (i * 8) / WS_DPB, dl = (i * 8) - fl * WS_DPB;
            if (fl < F) __pipeline_memcpy_async(&Xsh[0][i * 8], &X[((long)b * F + fl) * D + doff + dl], 16);
        }
        __pipeline_commit();
        for (int k = 0; k < ntiles; k++) {
            int buf = k & 1, g0 = k * WS_KT;
            if (k + 1 < ntiles) {               // prefetch X(k+1) -> Xsh[(k+1)&1] (async)
                int g1 = (k + 1) * WS_KT, xn = (k + 1) & 1;
                for (int i = ptid; i < nchunkX; i += 256) {
                    int fl = (i * 8) / WS_DPB, dl = (i * 8) - fl * WS_DPB, f = g1 + fl;
                    if (f < F) __pipeline_memcpy_async(&Xsh[xn][i * 8], &X[((long)b * F + f) * D + doff + dl], 16);
                }
                __pipeline_commit();
            }
            __pipeline_wait_prior(k + 1 < ntiles ? 1 : 0);
            ws_bar_prod();                      // all producers: X(k) fully staged in Xsh[buf]
            if (k >= 2) ws_bar_wait(3 + buf);   // empty[buf]: consumers freed Nsh[buf] (from k-2)
            const __nv_bfloat16* xs = Xsh[buf];
            for (int i = ptid; i < WS_KT * WS_DPB; i += 256) {
                int fl = i / WS_DPB, dl = i - fl * WS_DPB, f = g0 + fl;
                float v = 0.f;
                if (f < F) { float x = b2f(xs[i]); v = x * Rall[f] * b2f(W[(long)f * D + doff + dl]); }
                Nsh[buf][fl * WS_NLDB + dl] = __float2bfloat16(v);
            }
            ws_bar_arrive(1 + buf);             // full[buf]: N(buf) ready
        }
    } else {
        int q0 = cw * 16;
        wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc[WS_NCOL];
        #pragma unroll
        for (int c = 0; c < WS_NCOL; c++) wmma::fill_fragment(acc[c], 0.f);
        for (int k = 0; k < ntiles; k++) {
            int buf = k & 1, g0 = k * WS_KT;
            ws_bar_wait(1 + buf);               // full[buf]: producers computed N
            #pragma unroll
            for (int ks = 0; ks < WS_KSUB; ks++) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag, &SL[((long)b * Q + q0) * Fp + g0 + ks * 16], Fp);
                #pragma unroll
                for (int c = 0; c < WS_NCOL; c++) {
                    wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(b_frag, Nsh[buf] + ks * 16 * WS_NLDB + c * 16, WS_NLDB);
                    wmma::mma_sync(acc[c], a_frag, b_frag, acc[c]);
                }
            }
            ws_bar_arrive(3 + buf);             // empty[buf]: buffer free
        }
        float* ot = Otmp + cw * (WM * WN);
        #pragma unroll
        for (int c = 0; c < WS_NCOL; c++) {
            wmma::store_matrix_sync(ot, acc[c], WN, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane; i < WM * WN; i += 32) { int ql = i >> 4, dl = i & 15;
                O[((long)b * Q + q0 + ql) * D + doff + c * WN + dl] = __float2bfloat16(ot[i]); }
            __syncwarp();
        }
    }
}

// ---------------- WGMMA pool: hand-written inline-PTX wgmma.mma_async, sigmoid FUSED ----------------
// O[b, :Q, :D] = sigmoid(L[b, :Q, :F]) @ N[b, :F, :D].  One warpgroup (128 thr) per (batch, q-tile, n-tile).
// Reads RAW gate logits L[B,Q*F] and applies sigmoid inline while staging the A operand -> NO separate
// sigpad pass. N is read from the Fp-padded buffer (rows >=F are zero); A pad (f>=F) stored 0.
// WORKAROUND ptxas warp-3 accumulator bug: revert to acc[96] (all 3 n64-tiles per warpgroup) which is
// CORRECT but occupancy-limited. Future: investigate async TMA or alternative tiling. grid (B, Q/64).
constexpr int WG_KT = 64;                  // K-tile = 4 k16 wgmma steps
__global__
void k_pool_wgmma(const __nv_bfloat16* __restrict__ L, const __nv_bfloat16* __restrict__ N,
                  __nv_bfloat16* __restrict__ O, int B, int F, int Fp, int D, int Q) {
    int b = blockIdx.x, q0 = blockIdx.y * 64;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;   // 128 threads = 1 warpgroup
    const int NT = D / 64;  // 3 for D=192 (acc[96]; smaller acc hits the ptxas warp-3 bug)
    __shared__ __nv_bfloat16 As[64 * WG_KT];        // A: K-major bricks (8 KB)
    __shared__ __nv_bfloat16 Bs[WG_KT * DMAX];      // B: MN-major bricks, full D (24 KB)
    float acc[(DMAX / 64) * 32];                     // acc[96] = 3 n64-tiles (only correct config)
    #pragma unroll
    for (int i = 0; i < (DMAX / 64) * 32; i++) acc[i] = 0.f;
    int ntilesK = Fp / WG_KT;
    const int Bchunks = (WG_KT * D) / 8, kbStride = (D / 8) * 64;
    for (int t = 0; t < ntilesK; t++) {
        int k0 = t * WG_KT;
        // ---- stage B = N into MN-major bricks (full D; verified descriptor) ----
        for (int i = tid; i < Bchunks; i += 128) {
            int k = (i * 8) / D, nc8 = (i * 8) % D, off = (k / 8) * kbStride + (nc8 / 8) * 64 + (k % 8) * 8;
            const __nv_bfloat16* bsrc = N + ((long)(b * Fp + k0 + k)) * D + nc8;
            #pragma unroll
            for (int j = 0; j < 8; j++) Bs[off + j] = bsrc[j];
        }
        // ---- stage A = sigmoid(L) into K-major bricks ----
        for (int i = tid; i < (64 * WG_KT) / 8; i += 128) {
            int q = (i * 8) / WG_KT, fl = (i * 8) % WG_KT, f = k0 + fl;
            int off = (fl / 16) * 1024 + (q / 8) * 128 + ((fl % 16) / 8) * 64 + (q % 8) * 8;
            const __nv_bfloat16* src = L + ((long)(b * Q + q0 + q)) * F + f;
            if (f + 8 <= F) {
                #pragma unroll
                for (int j = 0; j < 8; j++) As[off + j] = __float2bfloat16(1.f / (1.f + __expf(-b2f(src[j]))));
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) { int ff = f + j;
                    As[off + j] = (ff < F) ? __float2bfloat16(1.f / (1.f + __expf(-b2f(src[j])))) : __float2bfloat16(0.f); }
            }
        }
        __syncthreads();
        // ---- wgmma over the 4 k16 steps of this K-tile (all NT n64-tiles) ----
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
        #pragma unroll
        for (int ks = 0; ks < WG_KT / 16; ks++) {
            int kk = ks * 16;
            uint64_t da = wg_desc(&As[(kk / 16) * 1024], 128, 256);
            for (int n2 = 0; n2 < NT; n2++) {
                uint64_t db = wg_desc(&Bs[(kk / 8) * kbStride + n2 * 512], (uint32_t)(D * 16), 128);
                wgmma_m64n64k16(&acc[n2 * 32], da, db, 1);
            }
        }
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
        __syncthreads();
    }
    // ---- epilogue: acc fragments -> O ----
    for (int nt = 0; nt < NT; nt++) {
        float* d = &acc[nt * 32];
        #pragma unroll
        for (int a = 0; a < 8; a++) {
            int row0 = q0 + 16 * warp + lane / 4, row1 = row0 + 8;
            int col0 = nt * 64 + 8 * a + (lane % 4) * 2, col1 = col0 + 1;
            O[((long)b * Q + row0) * D + col0] = __float2bfloat16(d[4 * a + 0]);
            O[((long)b * Q + row0) * D + col1] = __float2bfloat16(d[4 * a + 1]);
            O[((long)b * Q + row1) * D + col0] = __float2bfloat16(d[4 * a + 2]);
            O[((long)b * Q + row1) * D + col1] = __float2bfloat16(d[4 * a + 3]);
        }
    }
}

// NOTE: an acc[32] one-n64-tile-per-block variant (grid B×Q/64×D/64) for higher occupancy was tried
// with per-tile wait_group 0, wait_group<1> pipelining, scaleD=0, m64n96 acc[48], launch_bounds, and
// single-group double-buffer. ALL corrupt warp 3 (rows 48-63) except acc[96] above and single-group
// (12 ms, occupancy-limited): a ptxas GMMA register-management bug on small accumulators across many
// commit/wait. wait_group<1> reduces it ~30x but not to zero. acc[96] is the only fast-ish correct path.

// ---------------- WGMMA FUSED pool: read sigmoid'd g_L + compute N inline ----------------
// SMALL-ACC version: one m64n64 tile per block (acc[32], 32 fp32 regs) instead of acc[96] (all 3 D
// n-tiles, 96 regs). Grid gains a D-tile dim (blockIdx.z = nt in [0, D/64)) so 3 n64 tiles cover D=192.
// Dropping 96->32 regs + 1/3 the B/X smem raises occupancy. The accumulator path uses the cute GMMA atom
// + cute::warpgroup_fence_operand(EACH of the 32 floats)/arrive/commit/wait machinery — the canonical
// CUTLASS mainloop sequence (fence before arrive, fence after wait, every group) that defeats the ptxas
// warp-3 corruption every raw-asm acc[32] variant hit (verified /tmp/cute_smallacc_test.cu, warp3==floor).
// Reads g_L directly (sigmoid'd by gate epilogue) + computes N inline from X (cp.async double-buffered).
// launch_bounds(128, MINBLK): now SAFE with the async-proxy fence (before the fix, launch_bounds changed
// scheduling enough to expose the proxy-visibility race far more — it was a symptom, not a separate bug).
__global__ __launch_bounds__(128, 4)
void k_pool_wgmma_fused(const __nv_bfloat16* __restrict__ g_L,
                        const __nv_bfloat16* __restrict__ X,
                        const __nv_bfloat16* __restrict__ M, const float* __restrict__ R,
                        const __nv_bfloat16* __restrict__ W,
                        __nv_bfloat16* __restrict__ O, int B, int F, int Fp, int D, int Q) {
    (void)M;
    int b = blockIdx.x, q0 = blockIdx.y * 64, nt = blockIdx.z, dbase = nt * 64;  // this block's n64 tile
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __shared__ __nv_bfloat16 As[64 * WG_KT];         // A: K-major bricks (g_L for this q-tile)
    __shared__ __nv_bfloat16 Bs[WG_KT * 64];         // B: MN-major bricks (N for THIS n64 tile only)
    __shared__ __nv_bfloat16 Xsh[2][WG_KT * 64];     // double-buffer X (64 D-cols of this n-tile)
    __shared__ float Rtile[WG_KT];                   // R(rrms) for the CURRENT K-tile only
    float acc[32];                                    // ONE m64n64 tile = 32 fp32 regs (the occupancy win)
    #pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.f;
    __syncthreads();

    int ntilesK = Fp / WG_KT;
    const int Bchunks = (WG_KT * 64) / 8;            // 8-elem cp.async chunks of this n-tile's X
    const int kbStride = (64 / 8) * 64;              // = 512 for a 64-wide MN-major B tile
    // Descriptors are LOOP-INVARIANT: As/Bs are single-buffered at fixed smem addresses, so the 4
    // k16 (da,db) pairs are constant across all K-tiles. Precompute once to cut register pressure in
    // the wgmma loop (76->fewer regs); high reg count was reintroducing the ptxas warp-3 corruption.
    uint64_t da[WG_KT / 16], db[WG_KT / 16];
    #pragma unroll
    for (int ks = 0; ks < WG_KT / 16; ks++) {
        int kk = ks * 16;
        da[ks] = wg_desc(&As[(kk / 16) * 1024], 128, 256);
        db[ks] = wg_desc(&Bs[(kk / 8) * kbStride], (uint32_t)(64 * 16), 128);
    }

    // Prologue: cp.async X tile 0 (only the 64 D-cols [dbase, dbase+64) of this n-tile)
    for (int i = tid; i < Bchunks; i += 128) {
        int fl = (i * 8) / 64, dl = (i * 8) % 64;
        if (fl < F) {
            __pipeline_memcpy_async(&Xsh[0][fl * 64 + dl], &X[((long)b * F + fl) * D + dbase + dl], 16);
        }
    }
    __pipeline_commit();

    for (int t = 0; t < ntilesK; t++) {
        int k0 = t * WG_KT, buf = t & 1;
        // Prefetch next tile's X if available
        if (t + 1 < ntilesK) {
            int k1 = (t + 1) * WG_KT, nbuf = (t + 1) & 1;
            for (int i = tid; i < Bchunks; i += 128) {
                int fl = (i * 8) / 64, dl = (i * 8) % 64, f = k1 + fl;
                if (f < F) {
                    __pipeline_memcpy_async(&Xsh[nbuf][fl * 64 + dl], &X[((long)b * F + f) * D + dbase + dl], 16);
                }
            }
            __pipeline_commit();
        }
        // Load this K-tile's 64 R(rrms) values (R row-major R[b*F+f])
        for (int fl = tid; fl < WG_KT; fl += 128) {
            int f = k0 + fl;
            Rtile[fl] = (f < F) ? R[(long)b * F + f] : 0.f;
        }
        __pipeline_wait_prior(t + 1 < ntilesK ? 1 : 0);
        __syncthreads();

        // Compute N=RMSNorm from staged X (this n-tile's 64 D-cols), write MN-major B bricks: X*rrms*W[f,d]
        const __nv_bfloat16* xs = Xsh[buf];
        for (int i = tid; i < WG_KT * 64; i += 128) {
            int fl = i / 64, dl = i % 64, f = k0 + fl;
            float val = 0.f;
            if (f < F) {
                float x = b2f(xs[fl * 64 + dl]);
                val = x * Rtile[fl] * b2f(W[(long)f * D + dbase + dl]);
            }
            int k = fl, nc = dl;
            int off = (k / 8) * kbStride + (nc / 8) * 64 + (k % 8) * 8 + (nc % 8);
            Bs[off] = __float2bfloat16(val);
        }
        // Stage A = g_L (regular load, g_L is already cached from gate)
        for (int i = tid; i < (64 * WG_KT) / 8; i += 128) {
            int q = (i * 8) / WG_KT, fl = (i * 8) % WG_KT, f = k0 + fl;
            int off = (fl / 16) * 1024 + (q / 8) * 128 + ((fl % 16) / 8) * 64 + (q % 8) * 8;
            const __nv_bfloat16* src = g_L + ((long)(b * Q + q0 + q)) * F + f;
            if (f + 8 <= F) {
                #pragma unroll
                for (int j = 0; j < 8; j++) As[off + j] = src[j];
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    As[off + j] = (f + j < F) ? src[j] : __float2bfloat16(0.f);
                }
            }
        }
        __syncthreads();

        // wgmma over the 4 k16 steps — cute small-acc[32] machinery. Matches the canonical CUTLASS mainloop:
        // ONE fence+arrive, all 4 k16 wgmmas batched, ONE commit, ONE wait<0>, ONE fence (NOT per-k16).
        wgmma_async_proxy_fence();            // CRITICAL: generic smem writes -> visible to wgmma async reads
        wgmma_cute_fence32(acc);              // fence all 32 acc floats BEFORE the group
        cute::warpgroup_arrive();             // wgmma.fence.sync.aligned
        #pragma unroll
        for (int ks = 0; ks < WG_KT / 16; ks++) {
            wgmma_cute_m64n64k16(acc, da[ks], db[ks]);
        }
        cute::warpgroup_commit_batch();       // wgmma.commit_group.sync.aligned
        cute::warpgroup_wait<0>();            // wgmma.wait_group.sync.aligned 0
        wgmma_cute_fence32(acc);              // fence all 32 acc floats AFTER the wait
        __syncthreads();   // smem reuse barrier (single-buffered As/Bs)
    }
    // Epilogue: this block's n64 tile -> O[:, dbase:dbase+64]
    #pragma unroll
    for (int a = 0; a < 8; a++) {
        int row0 = q0 + 16 * warp + lane / 4, row1 = row0 + 8;
        int col0 = dbase + 8 * a + (lane % 4) * 2, col1 = col0 + 1;
        O[((long)b * Q + row0) * D + col0] = __float2bfloat16(acc[4 * a + 0]);
        O[((long)b * Q + row0) * D + col1] = __float2bfloat16(acc[4 * a + 1]);
        O[((long)b * Q + row1) * D + col0] = __float2bfloat16(acc[4 * a + 2]);
        O[((long)b * Q + row1) * D + col1] = __float2bfloat16(acc[4 * a + 3]);
    }
}

// ---------------- MULTI-WARPGROUP D-split fused pool (direction C) ----------------
// One CTA per (batch b, q-tile of 64). 3 warpgroups (384 thr) split D=192 into 3×64; each warpgroup
// owns acc[32] for its n64 D-tile across the whole K-loop. The A operand (sigmoid'd g_L for the
// q-tile, K-major bricks) is staged ONCE in smem and SHARED by all 3 warpgroups — this kills the 3x
// g_L re-read of the per-D-tile-CTA acc[32] path. Each warpgroup recomputes its 64-col N=LayerNorm(X)
// inline (no repad, no N round-trip). wgmma via the verified cute m64n64 atom + async-proxy fence.
// STEP 2: cp.async double-buffered X (overlap the 880MB X read with wgmma); A (g_L) regular loads.
// Dynamic smem (~82KB): As[64*KT] (shared A) + Bs[3][KT*64] (per-wg N) + Xsh[2][KT*D] (double-buf X)
// + Mtile/Rtile/gsh/bsh. Set via cudaFuncAttributeMaxDynamicSharedMemorySize in kernel_run.
constexpr int MWG_KT = 64, MWG_NWG = 3;                 // K-tile, # warpgroups (D=192=3*64)
__global__ __launch_bounds__(384, 2)
void k_pool_wgmma_mwg(const __nv_bfloat16* __restrict__ g_L,
                      const __nv_bfloat16* __restrict__ X,
                      const __nv_bfloat16* __restrict__ M, const float* __restrict__ R,
                      const __nv_bfloat16* __restrict__ W,
                      __nv_bfloat16* __restrict__ O, int B, int F, int Fp, int D, int Q) {
    (void)M;
    int b = blockIdx.x, q0 = blockIdx.y * 64;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wg = warp >> 2, wgwarp = warp & 3;
    int wtid = tid & 127;                               // thread index within this warpgroup (0..127)
    int dbase = wg * 64;                                // this warpgroup's D-tile [dbase, dbase+64)
    extern __shared__ char smem[];
    __nv_bfloat16* As  = (__nv_bfloat16*)smem;              // 64*KT  (shared A)
    __nv_bfloat16* Bs  = As + 64 * MWG_KT;                  // 3*KT*64 (per-wg N bricks)
    __nv_bfloat16* Xsh = Bs + MWG_NWG * MWG_KT * 64;        // 2*KT*D  (double-buffer X, all D cols)
    float* Rtile = (float*)(Xsh + 2 * MWG_KT * D);         // KT  (rrms for current K-tile)
    __nv_bfloat16* Bswg = Bs + wg * (MWG_KT * 64);          // this wg's N region
    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.f;

    const int kbStride = (64 / 8) * 64;                 // = 512 for a 64-wide MN-major B tile
    uint64_t da[MWG_KT / 16], db[MWG_KT / 16];
    #pragma unroll
    for (int ks = 0; ks < MWG_KT / 16; ks++) {
        int kk = ks * 16;
        da[ks] = wg_desc(&As[(kk / 16) * 1024], 128, 256);
        db[ks] = wg_desc(&Bswg[(kk / 8) * kbStride], (uint32_t)(64 * 16), 128);
    }

    int ntilesK = Fp / MWG_KT;
    const int nchunkX = (MWG_KT * D) / 8;               // 16B (8 bf16) cp.async chunks of the X tile (all D)
    for (int i = tid; i < nchunkX; i += 384) {
        int fl = (i * 8) / D, dl = (i * 8) - fl * D;
        if (fl < F) __pipeline_memcpy_async(&Xsh[fl * D + dl], &X[((long)b * F + fl) * D + dl], 16);
    }
    __pipeline_commit();
    __syncthreads();

    for (int t = 0; t < ntilesK; t++) {
        int k0 = t * MWG_KT, buf = t & 1;
        if (t + 1 < ntilesK) {
            int k1 = (t + 1) * MWG_KT, nbuf = (t + 1) & 1;
            for (int i = tid; i < nchunkX; i += 384) {
                int fl = (i * 8) / D, dl = (i * 8) - fl * D, f = k1 + fl;
                if (f < F) __pipeline_memcpy_async(&Xsh[nbuf * (MWG_KT * D) + fl * D + dl], &X[((long)b * F + f) * D + dl], 16);
            }
            __pipeline_commit();
        }
        for (int fl = tid; fl < MWG_KT; fl += 384) {
            int f = k0 + fl;
            Rtile[fl] = (f < F) ? R[(long)b * F + f] : 0.f;
        }
        for (int i = tid; i < (64 * MWG_KT) / 8; i += 384) {
            int q = (i * 8) / MWG_KT, fl = (i * 8) % MWG_KT, f = k0 + fl;
            int off = (fl / 16) * 1024 + (q / 8) * 128 + ((fl % 16) / 8) * 64 + (q % 8) * 8;
            const __nv_bfloat16* src = g_L + ((long)(b * Q + q0 + q)) * F + f;
            if (f + 8 <= F) {
                #pragma unroll
                for (int j = 0; j < 8; j++) As[off + j] = src[j];
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) As[off + j] = (f + j < F) ? src[j] : __float2bfloat16(0.f);
            }
        }
        __pipeline_wait_prior(t + 1 < ntilesK ? 1 : 0);
        __syncthreads();   // Xsh[buf], Rtile, As ready
        const __nv_bfloat16* xs = Xsh + buf * (MWG_KT * D);
        for (int i = wtid; i < MWG_KT * 64; i += 128) {
            int fl = i / 64, dl = i % 64, f = k0 + fl;
            float val = 0.f;
            if (f < F) {
                float x = b2f(xs[fl * D + dbase + dl]);
                val = x * Rtile[fl] * b2f(W[(long)f * D + dbase + dl]);   // RMSNorm: X*rrms*W[f,d]
            }
            int off = (fl / 8) * kbStride + (dl / 8) * 64 + (fl % 8) * 8 + (dl % 8);
            Bswg[off] = __float2bfloat16(val);
        }
        __syncthreads();   // A and all Bs ready for wgmma
        wgmma_async_proxy_fence();
        wgmma_cute_fence32(acc);
        cute::warpgroup_arrive();
        #pragma unroll
        for (int ks = 0; ks < MWG_KT / 16; ks++) wgmma_cute_m64n64k16(acc, da[ks], db[ks]);
        cute::warpgroup_commit_batch();
        cute::warpgroup_wait<0>();
        wgmma_cute_fence32(acc);
        __syncthreads();   // smem reuse barrier (single-buffered As/Bs)
    }
    // Epilogue: this warpgroup's n64 tile -> O[:, dbase:dbase+64]
    #pragma unroll
    for (int a = 0; a < 8; a++) {
        int row0 = q0 + 16 * wgwarp + lane / 4, row1 = row0 + 8;
        int col0 = dbase + 8 * a + (lane % 4) * 2, col1 = col0 + 1;
        O[((long)b * Q + row0) * D + col0] = __float2bfloat16(acc[4 * a + 0]);
        O[((long)b * Q + row0) * D + col1] = __float2bfloat16(acc[4 * a + 1]);
        O[((long)b * Q + row1) * D + col0] = __float2bfloat16(acc[4 * a + 2]);
        O[((long)b * Q + row1) * D + col1] = __float2bfloat16(acc[4 * a + 3]);
    }
}

// ---------------- MULTI-WARPGROUP FACTORED fused pool (direction C) ----------------
// NOTE (RMSNorm): the LayerNorm algebraic factorization (B = raw X, gamma/beta pulled out of the GEMM,
// mLR/sL corrections) DOES NOT hold for RMSNorm — N[f,d] = X[f,d]*rrms[f]*W[f,d] has W INSIDE the f-sum
// and W is per-(f,d), so it cannot be pulled out of the contraction. This env-gated variant is therefore
// kept as a correct-but-simple RMSNorm pool (B = materialized N = X*rrms*W, plain epilogue), mirroring
// k_pool_wgmma_mwg. It is not the factored fast path anymore; retained only so GIST_WGMMA_MWGF compiles+runs.
__global__ __launch_bounds__(384, 2)
void k_pool_wgmma_mwgf(const __nv_bfloat16* __restrict__ g_L,
                       const __nv_bfloat16* __restrict__ X,
                       const __nv_bfloat16* __restrict__ M, const float* __restrict__ R,
                       const __nv_bfloat16* __restrict__ W,
                       __nv_bfloat16* __restrict__ O, int B, int F, int Fp, int D, int Q) {
    (void)M;
    int b = blockIdx.x, q0 = blockIdx.y * 64;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wg = warp >> 2, wgwarp = warp & 3;
    int wtid = tid & 127, dbase = wg * 64;
    extern __shared__ char smem[];
    __nv_bfloat16* As  = (__nv_bfloat16*)smem;             // 64*KT  (shared A = g_L bricks)
    __nv_bfloat16* Bs  = As + 64 * MWG_KT;                 // 3*KT*64 (per-wg N bricks)
    __nv_bfloat16* Xsh = Bs + MWG_NWG * MWG_KT * 64;       // 2*KT*D  (double-buffer X, all D cols)
    float* Rtile = (float*)(Xsh + 2 * MWG_KT * D);         // KT  (rrms for current K-tile)
    __nv_bfloat16* Bswg = Bs + wg * (MWG_KT * 64);         // this wg's N region
    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.f;

    const int kbStride = (64 / 8) * 64;
    uint64_t da[MWG_KT/16], db[MWG_KT/16];
    #pragma unroll
    for (int ks = 0; ks < MWG_KT/16; ks++) {
        int kk = ks * 16;
        da[ks] = wg_desc(&As[(kk/16)*1024], 128, 256);
        db[ks] = wg_desc(&Bswg[(kk/8)*kbStride], (uint32_t)(64*16), 128);
    }

    int ntilesK = Fp / MWG_KT;
    const int nchunkX = (MWG_KT * D) / 8;
    for (int i = tid; i < nchunkX; i += 384) {
        int fl = (i * 8) / D, dl = (i * 8) - fl * D;
        if (fl < F) __pipeline_memcpy_async(&Xsh[fl * D + dl], &X[((long)b * F + fl) * D + dl], 16);
    }
    __pipeline_commit();
    __syncthreads();

    for (int t = 0; t < ntilesK; t++) {
        int k0 = t * MWG_KT, buf = t & 1;
        if (t + 1 < ntilesK) {
            int k1 = (t+1)*MWG_KT, nbuf = (t+1)&1;
            for (int i = tid; i < nchunkX; i += 384) {
                int fl = (i * 8) / D, dl = (i * 8) - fl * D, f = k1 + fl;
                if (f < F) __pipeline_memcpy_async(&Xsh[nbuf * (MWG_KT * D) + fl * D + dl], &X[((long)b * F + f) * D + dl], 16);
            }
            __pipeline_commit();
        }
        for (int fl = tid; fl < MWG_KT; fl += 384) {
            int f = k0 + fl;
            Rtile[fl] = (f<F) ? R[(long)b*F + f] : 0.f;
        }
        // Stage A = g_L (sigmoid'd by gate) bricks, shared across the 3 D-warpgroups
        for (int i = tid; i < (64 * MWG_KT) / 8; i += 384) {
            int q = (i * 8) / MWG_KT, fl = (i * 8) % MWG_KT, f = k0 + fl;
            int off = (fl / 16) * 1024 + (q / 8) * 128 + ((fl % 16) / 8) * 64 + (q % 8) * 8;
            const __nv_bfloat16* src = g_L + ((long)(b * Q + q0 + q)) * F + f;
            if (f + 8 <= F) {
                #pragma unroll
                for (int j = 0; j < 8; j++) As[off + j] = src[j];
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) As[off + j] = (f + j < F) ? src[j] : __float2bfloat16(0.f);
            }
        }
        __pipeline_wait_prior(t + 1 < ntilesK ? 1 : 0);
        __syncthreads();   // Xsh[buf], Rtile, As ready
        // Compute N = X*rrms*W[f,d] into this wg's B bricks
        const __nv_bfloat16* xs = Xsh + buf * (MWG_KT * D);
        for (int i = wtid; i < MWG_KT * 64; i += 128) {
            int fl = i / 64, dl = i % 64, f = k0 + fl;
            float val = 0.f;
            if (f < F) {
                float x = b2f(xs[fl * D + dbase + dl]);
                val = x * Rtile[fl] * b2f(W[(long)f * D + dbase + dl]);
            }
            int off = (fl / 8) * kbStride + (dl / 8) * 64 + (fl % 8) * 8 + (dl % 8);
            Bswg[off] = __float2bfloat16(val);
        }
        __syncthreads();   // A and all Bs ready for wgmma
        wgmma_async_proxy_fence();
        wgmma_cute_fence32(acc);
        cute::warpgroup_arrive();
        #pragma unroll
        for (int ks = 0; ks < MWG_KT/16; ks++) wgmma_cute_m64n64k16(acc, da[ks], db[ks]);
        cute::warpgroup_commit_batch();
        cute::warpgroup_wait<0>();
        wgmma_cute_fence32(acc);
        __syncthreads();   // before next tile overwrites As (shared)
    }
    // Epilogue: this warpgroup's n64 tile -> O[:, dbase:dbase+64]
    #pragma unroll
    for (int a = 0; a < 8; a++) {
        int row0 = q0 + 16 * wgwarp + lane / 4, row1 = row0 + 8;
        int col0 = dbase + 8 * a + (lane % 4) * 2, col1 = col0 + 1;
        O[((long)b * Q + row0) * D + col0] = __float2bfloat16(acc[4 * a + 0]);
        O[((long)b * Q + row0) * D + col1] = __float2bfloat16(acc[4 * a + 1]);
        O[((long)b * Q + row1) * D + col0] = __float2bfloat16(acc[4 * a + 2]);
        O[((long)b * Q + row1) * D + col1] = __float2bfloat16(acc[4 * a + 3]);
    }
}

// ---------------- FUSED MPF pool: read UNPADDED sigmoid'd g_L + compute N inline (WMMA) ----------------
// Matches Triton's fused pool: eliminates both the repad (0.79ms) AND the N round-trip (stats->0.44ms M,R-only).
// A = g_L (already sigmoid'd by gate; UNPADDED [B,Q,F] row-major, F=1491, NO alignment). Load with REGULAR
// masked loads (cp.async can't handle odd F). N computed INLINE from X (cp.async, D-contiguous=aligned).
// D-split (DPB=96 -> 2 blocks/b) + Q-split (QPB=64 -> 2 blocks/b) to raise occupancy: each warp does
// ONLY 1 q-tile (acc[1][3]=24 regs instead of 2 q-tiles=48 regs). Grid: (B, D/DPB, Q/QPB).
// Uses WMMA (mma.sync m16n16k16). 256 threads = 8 warps, layout [QPB/16, DPB/16/3] = [4, 2].
// K-tile=32: balance smem and K-loop iterations. Pad Nsh (stride +8).
constexpr int MPF_DPB = 96, MPF_QPB = 64, MPF_KT = 32, MPF_KSUB = 2, MPF_NLDB = MPF_DPB + 8;
__global__ __launch_bounds__(256, 5)
void k_pool_mpf(const __nv_bfloat16* __restrict__ g_L,
                const __nv_bfloat16* __restrict__ X,
                const __nv_bfloat16* __restrict__ M, const float* __restrict__ R,
                const __nv_bfloat16* __restrict__ W,
                __nv_bfloat16* __restrict__ O, int B, int F, int Fp, int D, int Q) {
    (void)M;
    int b = blockIdx.x, doff = blockIdx.y * MPF_DPB, qoff = blockIdx.z * MPF_QPB;
    int tid = threadIdx.x, nt = blockDim.x, warp = tid >> 5, lane = tid & 31;
    // 8 warps: 4 q-tiles (16 rows each) × 2 d-groups (3 n16-cols each). Each warp does 1 q-tile now.
    // warp_d=warp%2, warp_q=warp/2 (0..3). acc[1][3] → 24 regs (vs 48 before).
    int warp_d = warp % 2, warp_q = warp / 2, cbase = warp_d * 3;
    __shared__ float Rall[FPMAX];
    __shared__ __nv_bfloat16 Xsh[2][MPF_KT * MPF_DPB], Ash[2][MPF_QPB * MPF_KT], Nsh[MPF_KT * MPF_NLDB];
    __shared__ float Otmp[8 * (16 * 16)];
    for (int f = tid; f < Fp; f += nt) { Rall[f] = (f < F) ? R[(long)b * F + f] : 0.f; }
    int ntiles = (F + MPF_KT - 1) / MPF_KT, nchunkX = (MPF_KT * MPF_DPB) / 8, nchunkA = (MPF_QPB * MPF_KT);
    // accumulators PERSIST across the whole K-loop (1 q-tile x 3 d-tiles per warp) -> the reduction
    // over all F accumulates here; store to O ONCE after the loop.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[1][3];
    #pragma unroll
    for (int c = 0; c < 3; c++) wmma::fill_fragment(acc[0][c], 0.f);
    // Prefetch first K-tile of X (cp.async) and A (regular loads into Ash[0])
    {   for (int i = tid; i < nchunkX; i += nt) { int fl = (i * 8) / MPF_DPB, dl = (i * 8) - fl * MPF_DPB;
            if (fl < F) __pipeline_memcpy_async(&Xsh[0][i * 8], &X[((long)b * F + fl) * D + doff + dl], 16); }
    }
    __pipeline_commit();
    {   for (int i = tid; i < nchunkA; i += nt) { int ql = i / MPF_KT, fl = i % MPF_KT, q = qoff + ql;
            Ash[0][i] = (q < Q && fl < F) ? g_L[((long)b * Q + q) * F + fl] : __float2bfloat16(0.f); }
    }
    for (int k = 0; k < ntiles; k++) {
        int cur = k & 1, nxt = 1 - cur, g0 = k * MPF_KT;
        // Prefetch next K-tile's X and A into the OTHER buffer (overlapped with WMMA below)
        if (k + 1 < ntiles) { int g1 = (k + 1) * MPF_KT;
            for (int i = tid; i < nchunkX; i += nt) { int fl = (i * 8) / MPF_DPB, dl = (i * 8) - fl * MPF_DPB;
                if (g1 + fl < F) __pipeline_memcpy_async(&Xsh[nxt][i * 8], &X[((long)b * F + g1 + fl) * D + doff + dl], 16); }
            __pipeline_commit();
        }
        __pipeline_wait_prior(k + 1 < ntiles ? 1 : 0);
        __syncthreads();
        const __nv_bfloat16* xs = Xsh[cur];
        // Compute N=RMSNorm (X*rrms*W[f,d]) from X (this K-tile) while the NEXT A is being loaded in the background
        for (int i = tid; i < MPF_KT * MPF_DPB; i += nt) { int fl = i / MPF_DPB, dl = i - fl * MPF_DPB; int f = g0 + fl;
            Nsh[fl * MPF_NLDB + dl] = (f < F) ? __float2bfloat16(b2f(xs[i]) * Rall[f] * b2f(W[(long)f * D + doff + dl])) : __float2bfloat16(0.f); }
        __syncthreads();
        // WMMA on current K-tile (reads Ash[cur], Nsh). Overlap: next A loads happen during this compute.
        #pragma unroll
        for (int ks = 0; ks < MPF_KSUB; ks++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, Ash[cur] + warp_q * (16 * MPF_KT) + ks * 16, MPF_KT);
            #pragma unroll
            for (int c = 0; c < 3; c++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, Nsh + ks * 16 * MPF_NLDB + (cbase + c) * 16, MPF_NLDB);
                wmma::mma_sync(acc[0][c], a_frag, b_frag, acc[0][c]);
            }
        }
        // Load next A into Ash[nxt] (overlaps with above WMMA via warp scheduling if occupancy high enough)
        if (k + 1 < ntiles) { int g1 = (k + 1) * MPF_KT;
            for (int i = tid; i < nchunkA; i += nt) { int ql = i / MPF_KT, fl = i % MPF_KT, q = qoff + ql, f = g1 + fl;
                Ash[nxt][i] = (q < Q && f < F) ? g_L[((long)b * Q + q) * F + f] : __float2bfloat16(0.f); }
        }
        __syncthreads();
    }
    // epilogue: store the fully-reduced accumulators to O
    int q_base = qoff + warp_q * 16;
    float* ot = Otmp + warp * (16 * 16);
    for (int c = 0; c < 3; c++) {
        wmma::store_matrix_sync(ot, acc[0][c], 16, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 16 * 16; i += 32) { int ql = i >> 4, dl = i & 15;
            O[((long)b * Q + q_base + ql) * D + doff + (cbase + c) * 16 + dl] = __float2bfloat16(ot[i]); }
        __syncwarp();
    }
}

// ---------------- (legacy) pool: thread-per-d, stream g ONCE (X read once) ----------------
// Block = (q-tile of PQ, b). Threads = one per d (D padded to PBD). Each thread
// keeps acc[PQ] for its channel d, streaming g so X[b,:,d] and L[b,q-tile,:] are
// each read exactly once (the old version re-read X per d-tile/q-tile).
constexpr int PQ = 16, PG = 32, PBD = 256;   // PBD >= D
__global__ void k_pool(const __nv_bfloat16* __restrict__ L, const __nv_bfloat16* __restrict__ X,
                       const __nv_bfloat16* __restrict__ M, const float* __restrict__ R,
                       const __nv_bfloat16* __restrict__ W,
                       __nv_bfloat16* __restrict__ O, int B, int F, int D, int Q) {
    (void)M;
    __shared__ float Ls[PQ][PG];
    __shared__ float Rs[PG];
    int b = blockIdx.y, q0 = blockIdx.x * PQ;
    int d = threadIdx.x;                       // one channel per thread
    float acc[PQ];
    #pragma unroll
    for (int i = 0; i < PQ; i++) acc[i] = 0.f;

    for (int g0 = 0; g0 < F; g0 += PG) {
        for (int idx = threadIdx.x; idx < PQ * PG; idx += blockDim.x) {
            int i = idx / PG, j = idx % PG; int qq = q0 + i, gg = g0 + j;
            // L holds raw gate logits (CUTLASS gate); sigmoid here (free on the load)
            Ls[i][j] = (qq < Q && gg < F) ? 1.f / (1.f + expf(-b2f(L[((long)b * Q + qq) * F + gg]))) : 0.f;
        }
        for (int idx = threadIdx.x; idx < PG; idx += blockDim.x) {
            int gg = g0 + idx;
            Rs[idx] = (gg < F) ? R[(long)b * F + gg] : 0.f;
        }
        __syncthreads();
        if (d < D) {
            for (int j = 0; j < PG; j++) {
                int gg = g0 + j;
                // RMSNorm: N = X * rrms * W[f,d] (no centering, no bias)
                float nv = (gg < F) ? b2f(X[((long)b * F + gg) * D + d]) * Rs[j] * b2f(W[(long)gg * D + d]) : 0.f;
                #pragma unroll
                for (int i = 0; i < PQ; i++) acc[i] += Ls[i][j] * nv;
            }
        }
        __syncthreads();
    }
    if (d < D) {
        for (int i = 0; i < PQ; i++) {
            int qq = q0 + i;
            if (qq < Q) O[((long)b * Q + qq) * D + d] = __float2bfloat16(acc[i]);
        }
    }
}

int g_B = 0, g_F = 0, g_D = 0, g_Q = 0;
__nv_bfloat16 *g_M = nullptr;            // mean, COLUMN-major [B,F] (for the unpadded gate)
float *g_R = nullptr;                    // rstd [B,F]
__nv_bfloat16 *g_L = nullptr;            // raw gate logits [B, Q*F]
__nv_bfloat16 *g_N = nullptr;            // LayerNorm(X) padded to [B, Fp, D] (pool B-operand)
__nv_bfloat16 *g_SL = nullptr;           // sigmoid(L) padded to [B, Q, Fp] (pool A-operand)
cudaStream_t g_stream2 = nullptr;        // 2nd stream: N-compute overlaps the gate (gate needs only M)
cudaEvent_t g_ev_mr = nullptr, g_ev_n = nullptr;
int env_i(const char* n, int def) { const char* v = getenv(n); return v ? atoi(v) : def; }
}  // namespace

extern "C" int kernel_run(__nv_bfloat16** inputs, int num_inputs,
                          __nv_bfloat16** outputs, int num_outputs,
                          int n, cudaStream_t stream) {
    (void)num_inputs; (void)num_outputs; (void)n;
    int B = env_i("CUDA_EXEC_PARAM_GIST_B", 0), F = env_i("CUDA_EXEC_PARAM_GIST_F", 0);
    int D = env_i("CUDA_EXEC_PARAM_GIST_D", 0), Q = env_i("CUDA_EXEC_PARAM_GIST_Q", 0);
    if (B <= 0 || F <= 0 || D <= 0 || Q <= 0) return 10;
    long QF = (long)Q * F;
    int Fp = (F + 63) / 64 * 64;              // pad K to a multiple of 64 (fused pool uses a 64-row K-tile)
    if (B != g_B || F != g_F || D != g_D || Q != g_Q) {
        if (g_M) cudaFree(g_M); if (g_R) cudaFree(g_R); if (g_L) cudaFree(g_L);
        if (g_N) cudaFree(g_N); if (g_SL) cudaFree(g_SL);
        if (cudaMalloc(&g_M, (size_t)B * F * sizeof(__nv_bfloat16)) != cudaSuccess) return 11;
        if (cudaMalloc(&g_R, (size_t)B * F * 4) != cudaSuccess) return 11;
        if (cudaMalloc(&g_L, (size_t)B * QF * sizeof(__nv_bfloat16)) != cudaSuccess) return 11;
        if (cudaMalloc(&g_N, (size_t)B * Fp * D * sizeof(__nv_bfloat16)) != cudaSuccess) return 11;
        if (cudaMalloc(&g_SL, (size_t)B * Q * Fp * sizeof(__nv_bfloat16)) != cudaSuccess) return 11;
        cudaMemset(g_SL, 0, (size_t)B * Q * Fp * sizeof(__nv_bfloat16));  // zero ONCE: the f in [F,Fp) pad stays 0
        g_B = B; g_F = F; g_D = D; g_Q = Q;
    }
    const __nv_bfloat16 *X = inputs[0], *P = inputs[1], *W = inputs[2];   // RMSNorm: 3 inputs, W[F,D] weight (no beta)
    __nv_bfloat16* O = outputs[0];

    int warps_per_blk = 8;
    long stat_warps = ((long)B * Fp + RPW - 1) / RPW;
    long stat_blocks = (stat_warps + warps_per_blk - 1) / warps_per_blk;

    // (PADP — gate writes SL_pad directly via column-padded P — was measured SLOWER: 4.27 (Fpad8) /
    //  4.36 (Fp); the P-pad copy + padded-P gate read cost more than the repad it removes. Removed.)

    // MPF path: WMMA fused pool, reads UNPADDED sigmoid'd g_L + computes N inline (stats writes M,R only)
    if (getenv("GIST_MPF")) {
        k_stats_mr<<<stat_blocks, warps_per_blk * 32, 0, stream>>>(X, g_M, g_R, B, F, Fp, D, 1e-5f);
        if (getenv("GIST_SKIP_GATE") == nullptr) {
            int rc = run_gate(g_M, P, g_L, B, F, QF, Q, Fp, stream);
            if (rc != 0) return rc;
        }
        if (!getenv("GIST_SKIP_POOL")) {
            dim3 mpfgrid(B, D / MPF_DPB, Q / MPF_QPB);  // D-split (2) × Q-split (2) = 4 blocks per b
            k_pool_mpf<<<mpfgrid, 256, 0, stream>>>(g_L, X, g_M, g_R, W, O, B, F, Fp, D, Q);
        }
        return 0;
    }

    // FUSED WGMMA pool path: stats writes ONLY M,R; pool computes N inline + reads sigmoid'd g_L
    if (getenv("GIST_WGMMA_FUSED")) {
        k_stats_mr<<<stat_blocks, warps_per_blk * 32, 0, stream>>>(X, g_M, g_R, B, F, Fp, D, 1e-5f);
        if (getenv("GIST_SKIP_GATE") == nullptr) {
            int rc = run_gate(g_M, P, g_L, B, F, QF, Q, Fp, stream);
            if (rc != 0) return rc;
        }
        if (!getenv("GIST_SKIP_POOL")) {
            dim3 wgrid(B, Q / 64, D / 64);   // + D-tile dim: one n64 tile per block (acc[32], high occ)
            k_pool_wgmma_fused<<<wgrid, 128, 0, stream>>>(g_L, X, g_M, g_R, W, O, B, F, Fp, D, Q);
        }
        return 0;
    }

    // MULTI-WARPGROUP D-split fused pool (direction C): stats writes ONLY M,R; pool computes N inline,
    // reads sigmoid'd g_L ONCE per q-tile (shared A across 3 warpgroups splitting D).
    if (getenv("GIST_WGMMA_MWG")) {
        k_stats_mr<<<stat_blocks, warps_per_blk * 32, 0, stream>>>(X, g_M, g_R, B, F, Fp, D, 1e-5f);
        if (getenv("GIST_SKIP_GATE") == nullptr) {
            int rc = run_gate(g_M, P, g_L, B, F, QF, Q, Fp, stream);
            if (rc != 0) return rc;
        }
        if (!getenv("GIST_SKIP_POOL")) {
            dim3 mwgrid(B, Q / 64);   // one CTA per (batch, q-tile of 64); 3 warpgroups split D
            size_t smem = (size_t)(64 * MWG_KT + MWG_NWG * MWG_KT * 64 + 2 * MWG_KT * D) * sizeof(__nv_bfloat16)
                        + (size_t)(MWG_KT) * sizeof(float);   // Rtile only (RMSNorm: no Mtile/gsh/bsh)
            static bool mwg_attr_set = false;
            if (!mwg_attr_set) {
                cudaFuncSetAttribute(k_pool_wgmma_mwg, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
                mwg_attr_set = true;
            }
            k_pool_wgmma_mwg<<<mwgrid, 384, smem, stream>>>(g_L, X, g_M, g_R, W, O, B, F, Fp, D, Q);
        }
        return 0;
    }

    // MULTI-WARPGROUP FACTORED pool (RMSNorm): the LayerNorm factorization doesn't hold for W[F,D];
    // this variant now materializes N=X*rrms*W (same smem layout as MWG). Kept so the env path compiles+runs.
    if (getenv("GIST_WGMMA_MWGF")) {
        k_stats_mr<<<stat_blocks, warps_per_blk * 32, 0, stream>>>(X, g_M, g_R, B, F, Fp, D, 1e-5f);
        if (getenv("GIST_SKIP_GATE") == nullptr) {
            int rc = run_gate(g_M, P, g_L, B, F, QF, Q, Fp, stream);
            if (rc != 0) return rc;
        }
        if (!getenv("GIST_SKIP_POOL")) {
            dim3 mwgrid(B, Q / 64);
            size_t smem = (size_t)(64 * MWG_KT + MWG_NWG * MWG_KT * 64 + 2 * MWG_KT * D) * sizeof(__nv_bfloat16)
                        + (size_t)(MWG_KT) * sizeof(float);   // Rtile only
            static bool mwgf_attr_set = false;
            if (!mwgf_attr_set) {
                cudaFuncSetAttribute(k_pool_wgmma_mwgf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
                mwgf_attr_set = true;
            }
            k_pool_wgmma_mwgf<<<mwgrid, 384, smem, stream>>>(g_L, X, g_M, g_R, W, O, B, F, Fp, D, Q);
        }
        return 0;
    }

    // OLD path (kept as fallback behind GIST_GATE_SIG): sigmoid fused in the gate epilogue + plain k_repad.
    // Branch-C measured this is SLOWER (4.02ms): the Sigmoid epilogue's 293M exp competes with the
    // compute-bound gate's tensor pipeline (gate 1.66ms). Also hosts the experimental hand-rolled pools.
    if (getenv("GIST_GATE_SIG")) {
        k_stats<<<stat_blocks, warps_per_blk * 32, 0, stream>>>(X, g_M, g_R, g_N, W, B, F, Fp, D, 1e-5f);
        if (getenv("GIST_SKIP_GATE") == nullptr) {
            int rc = run_gate(g_M, P, g_L, B, F, QF, Q, Fp, stream);   // sigmoid fused in epilogue
            if (rc != 0) return rc;
        }
        if (getenv("GIST_SKIP_POOL")) return 0;
        if (getenv("GIST_WGMMA_POOL")) {
            dim3 wgrid(B, Q / 64);
            k_pool_wgmma<<<wgrid, 128, 0, stream>>>(g_L, g_N, O, B, F, Fp, D, Q);
            return 0;
        }
        if (getenv("GIST_WARPSPEC")) {
            k_sigpad<<<32768, 256, 0, stream>>>(g_L, g_SL, B, Q, F, Fp);
            dim3 wgrid(B, D / WS_DPB);
            k_pool_warpspec<<<wgrid, 512, 0, stream>>>(g_SL, X, g_M, g_R, W, O, B, F, Fp, D, Q);
            return 0;
        }
        if (getenv("GIST_FUSED_POOL")) {
            k_sigpad<<<32768, 256, 0, stream>>>(g_L, g_SL, B, Q, F, Fp);
            dim3 fgrid(B, D / DPB);
            k_pool_fused<<<fgrid, (Q / WM) * DGB * 32, 0, stream>>>(g_SL, X, g_M, g_R, W, O, B, F, Fp, D, Q);
            return 0;
        }
        k_repad<<<32768, 256, 0, stream>>>(g_L, g_SL, B, Q, F, Fp);   // L already sigmoid'd; pure repad
        if (getenv("GIST_SKIP_POOLGEMM")) return 0;
        return run_pool(g_SL, g_N, O, B, Q, D, Fp, stream);
    }

    // DEFAULT path (branch-C gate-sigmoid-relocate, 3.67ms, beats the 4.02 sig path): run the gate with a
    // plain LinearCombination epilogue (RAW logits — no 293M exp competing with the compute-bound tensor
    // pipeline, gate 1.66->1.26ms), then apply sigmoid in k_sigpad (sigmoid+pad in one memory-bound pass,
    // exp ~free behind DRAM latency: +0.05ms vs k_repad). stats writes M,R,N (one X read).
    k_stats<<<stat_blocks, warps_per_blk * 32, 0, stream>>>(X, g_M, g_R, g_N, W, B, F, Fp, D, 1e-5f);
    if (getenv("GIST_SKIP_GATE") == nullptr) {
        int rc = run_gate_ns(g_M, P, g_L, B, F, QF, stream);   // CUTLASS gate, raw logits (no sigmoid)
        if (rc != 0) return rc;
    }
    if (getenv("GIST_SKIP_POOL")) return 0;
    if (getenv("GIST_SIGPAD1")) {     // fallback: scalar sigmoid+pad (A/B reference)
        k_sigpad<<<32768, 256, 0, stream>>>(g_L, g_SL, B, Q, F, Fp);
    } else {                           // vectorized 8-row-aligned sigmoid+pad (default)
        size_t spsmem = (size_t)SP_RPB * F * sizeof(__nv_bfloat16);
        k_sigpad_v2<<<32768, 256, spsmem, stream>>>(g_L, g_SL, B, Q, F, Fp);
    }
    if (getenv("GIST_SKIP_POOLGEMM")) return 0;
    return run_pool(g_SL, g_N, O, B, Q, D, Fp, stream);
}
