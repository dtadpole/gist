/*
 * GIST forward — optimized CUDA kernel for the cuda_exec harness (v2, RMSNorm / F=1497).
 *
 * inputs[0]=X[B,F,D], inputs[1]=P[F,Q*F], inputs[2]=W[F,D]  (RMSNorm weight, per-(feature,channel); no beta)
 * outputs[0]=O[B,Q,D].  Dims from env CUDA_EXEC_PARAM_GIST_{B,F,D,Q}.
 *
 *   stats : M[b,f]=mean_d X, R[b,f]=rrms_d X = rsqrt(mean_d(X^2)+eps)  (warp-per-row, coalesced)
 *   gate  : L[b,q,g]=sigmoid(M[b,:] @ P[:,q*F+g])  (hand WGMMA bf16 GEMM, σ fused in epilogue via tanh.approx)
 *   pool  : O[b,q,d]=sum_g L[b,q,g] * N[b,g,d]   (N = RMSNorm value path = X * rrms * W[f,d], no centering/bias)
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>            // CUtensorMap, cuTensorMapEncodeTiled (TMA, driver API)
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
// σ(x) = 0.5*tanh(0.5x)+0.5 via tanh.approx.f32 (ONE hardware SFU op) — shorter critical-path chain than
// __expf's 2-SFU EX2->RCP chain (~28 vs ~48 cyc), which is what the serial-epilogue σ latency needs.
__device__ __forceinline__ float sigmoidf(float x) {
    float t; asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * x));
    return fmaf(0.5f, t, 0.5f);
}
__device__ __forceinline__ float sigmoidf_exp(float x) { return 1.f / (1.f + __expf(-x)); }  // ref (2-SFU)

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
// B128 canonical SWIZZLE (shared finding from branch C, validated max_err 0): conflict-free cp.async+wgmma.
// 64-wide MN-major canonical tile: for logical (k, mn) the u128 index u=(mn/8)+(k%8)*8+(k/8)*64, swizzled
// su=u^((u>>3)&7), bf16 element offset = su*8 + (mn%8). Descriptor: layout_type=1 (128B), SBO=1024 bytes,
// per-k16 start advance = ks*1024 elements (swizzle-invariant). Smem tile base must be 128B-aligned.
__device__ __forceinline__ int gh_sw_off(int k, int mn) {   // bf16 element offset in a 64-wide canon tile
    int u = (mn >> 3) + (k & 7) * 8 + (k >> 3) * 64;
    int su = u ^ ((u >> 3) & 7);
    return su * 8 + (mn & 7);
}
__device__ __forceinline__ int gh_sw_chunk(int k, int mn8) { // 16B-chunk (8 mn) base offset (mn%8==0)
    int u = mn8 + (k & 7) * 8 + (k >> 3) * 64;
    int su = u ^ ((u >> 3) & 7);
    return su * 8;
}
__device__ __forceinline__ uint64_t wg_desc_sw(const void* smem_ptr, uint32_t SBO_bytes) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t d = 0;
    d |= wg_encode(a);
    d |= wg_encode(SBO_bytes) << 32;     // SBO; LBO irrelevant for a 64-wide (n=1) tile
    d |= (uint64_t)1 << 62;              // layout_type=1 (128B swizzle)
    return d;
}
// 3-arg descriptor: also sets LBO (leading byte offset). Needed for the m64n256 atom whose
// B operand spans 4 contiguous 64x64 canonical bricks: LBO=8192 (=64*64*2, brick stride),
// SBO=1024. A operand (64-wide) uses LBO=128,SBO=1024. Validated in /tmp/swz256.cu (max_err 0).
__device__ __forceinline__ uint64_t wg_desc_sw3(const void* smem_ptr, uint32_t LBO_bytes, uint32_t SBO_bytes) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t d = 0;
    d |= wg_encode(a);
    d |= wg_encode(LBO_bytes) << 16;
    d |= wg_encode(SBO_bytes) << 32;
    d |= (uint64_t)1 << 62;              // layout_type=1 (128B swizzle)
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
// POOL atom: m64n192k16, A K-major (trans-a=0), B MN-major (trans-b=1), fp32 acc[96]. Validated in
// /tmp/pooltest192.cu (max_err 0). The K-major A smem tile is the canonical Layout_K_SW128_Atom
// ([64x64] contiguous bf16); A desc = wg_desc_sw3(&As[ks*16], LBO=16, SBO=1024) (per-k16 +16 bf16,
// since K is contiguous in K-major). B (3 contiguous 64-bricks) = wg_desc_sw3(&Bs[ks*1024], 8192, 1024).
using GistPoolAtom = cute::SM90::GMMA::MMA_64x192x16_F32BF16BF16_SS<
    cute_gmma::Major::K, cute_gmma::Major::MN, cute_gmma::ScaleIn::One, cute_gmma::ScaleIn::One>;
__device__ __forceinline__ void wgmma_cute_m64n192k16(float* d, uint64_t da, uint64_t db) {
    GistPoolAtom::fma(da, db,
        d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15],
        d[16],d[17],d[18],d[19],d[20],d[21],d[22],d[23],d[24],d[25],d[26],d[27],d[28],d[29],d[30],d[31],
        d[32],d[33],d[34],d[35],d[36],d[37],d[38],d[39],d[40],d[41],d[42],d[43],d[44],d[45],d[46],d[47],
        d[48],d[49],d[50],d[51],d[52],d[53],d[54],d[55],d[56],d[57],d[58],d[59],d[60],d[61],d[62],d[63],
        d[64],d[65],d[66],d[67],d[68],d[69],d[70],d[71],d[72],d[73],d[74],d[75],d[76],d[77],d[78],d[79],
        d[80],d[81],d[82],d[83],d[84],d[85],d[86],d[87],d[88],d[89],d[90],d[91],d[92],d[93],d[94],d[95],
        cute_gmma::ScaleOut::One);
}
__device__ __forceinline__ void wgmma_cute_fence32(float* d) {
    #pragma unroll
    for (int i = 0; i < 32; i++) cute::warpgroup_fence_operand(d[i]);
}
// Make generic-proxy smem writes (the staged A/B operands) visible to wgmma's async-proxy reads.
__device__ __forceinline__ void wgmma_async_proxy_fence() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
// Per-warpgroup register reallocation (Hopper setmaxnreg) — must be called by all 128 threads of a warpgroup,
// before they use the (de)allocated registers. Lets compute WGs take many regs while producer/epilogue WGs
// give them up, so a 4-WG persistent kernel fits 1 CTA/SM.
template <int R> __device__ __forceinline__ void wg_reg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(R)); }
template <int R> __device__ __forceinline__ void wg_reg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(R)); }
// mbarrier producer/consumer handoff (phase parity + release/acquire) — shared by the warp-spec gate and pool.
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(a), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar); uint64_t s;
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 %0, [%1];" : "=l"(s) : "r"(a));
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, int parity) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("{\n .reg .pred P;\n LAB_W_%=:\n"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], %1;\n"
                 "@P bra DONE_W_%=;\n bra LAB_W_%=;\n DONE_W_%=:\n}\n" :: "r"(a), "r"(parity));
}
__device__ __forceinline__ void wgmma_fenceN(float* d, int n) {
    for (int i = 0; i < n; i++) cute::warpgroup_fence_operand(d[i]);
}
// Named-barrier sync over the 256 CONSUMER threads only (barrier id 1). The warp-spec gate's
// producer WG (tid>=256) has exited by the epilogue, so __syncthreads (all 384) would hang;
// bar.sync over a private id with the consumer count avoids the producer.
__device__ __forceinline__ void bar_sync_consumers() {
    asm volatile("bar.sync 1, 256;" ::: "memory");
}
// TMA (cp.async.bulk.tensor) helpers — validated by branch C (/tmp/tmatest.cu, max_err 0). TMA SW128 writes
// the SAME canonical B128 swizzle that wg_desc_sw reads, and OOB box elements are zero-filled (handles K-pad
// + N-tail for free). One thread issues the bulk copy; the mbarrier tracks byte completion (expect_tx).
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, unsigned bytes) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(a), "r"(bytes));
}
__device__ __forceinline__ void tma_load_2d(void* smem, const CUtensorMap* tm, int x, int y, uint64_t* bar) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t m = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];"
                 :: "r"(s), "l"(tm), "r"(x), "r"(y), "r"(m) : "memory");
}
// 3D TMA load — for the FUSED gate's B-operand P viewed [f(gate col), q, k(=f')]: the f-coordinate
// (fb*256 + n2*64) stays 64-aligned (SWIZZLE_128B legal) while q is a separate non-swizzled index, so
// per-q tiling works WITHOUT padding P (the flat-2D coord q*F+... is not %64 since F=1491 -> illegal inst).
__device__ __forceinline__ void tma_load_3d(void* smem, const CUtensorMap* tm, int c0, int c1, int c2, uint64_t* bar) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t m = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3,%4}], [%5];"
                 :: "r"(s), "l"(tm), "r"(c0), "r"(c1), "r"(c2), "r"(m) : "memory");
}
// TMA STORE (SM90_TMA_STORE): async bulk copy smem tile -> global, hardware-coalesced, OOB-clipped to
// globalDim (handles the QF tail for free). Issued by one thread; the warps are freed (no per-thread
// store). tma_store_fence makes the generic-proxy staging writes visible to the async-proxy store.
__device__ __forceinline__ void tma_store_2d(const CUtensorMap* tm, const void* smem, int x, int y) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1,%2}], [%3];"
                 :: "l"(tm), "r"(x), "r"(y), "r"(s) : "memory");
}
__device__ __forceinline__ void tma_store_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(N) : "memory");
}
__device__ __forceinline__ void wgmma_cute_m64n64k16(float* d, uint64_t da, uint64_t db) {
    GistGmmaAtom::fma(da, db,
        d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15],
        d[16],d[17],d[18],d[19],d[20],d[21],d[22],d[23],d[24],d[25],d[26],d[27],d[28],d[29],d[30],d[31],
        cute_gmma::ScaleOut::One);   // scaleD=1, always accumulate
}
// A MN-major variant (trans-a=1): for the hand gate, A=M is col-major (m-contiguous = MN-major natural),
// so staging A MN-major avoids the K-major scatter (C measured that scatter as the dominant cost).
using GistGmmaAtomAN = cute::SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS<
    cute_gmma::Major::MN, cute_gmma::Major::MN, cute_gmma::ScaleIn::One, cute_gmma::ScaleIn::One>;
__device__ __forceinline__ void wgmma_cute_m64n64k16_AN(float* d, uint64_t da, uint64_t db) {
    GistGmmaAtomAN::fma(da, db,
        d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15],
        d[16],d[17],d[18],d[19],d[20],d[21],d[22],d[23],d[24],d[25],d[26],d[27],d[28],d[29],d[30],d[31],
        cute_gmma::ScaleOut::One);
}
// Single m64n256k16 wgmma (MN-major A trans-a=1, MN-major B trans-b=1), fp32 acc d[128]. Replaces
// 4x m64n64 per k16 (1 issue vs 4) reading the SAME 4-brick Bs layout via LBO=8192. The acc[128]
// layout is BIT-IDENTICAL to 4 concatenated m64n64 acc[32] (group g=n2*8+a -> acc idx 4*g, col 8*g),
// so the existing epilogue is unchanged. Validated in /tmp/swz256.cu (max_err 0).
__device__ __forceinline__ void wgmma_m64n256k16_AN(float* d, uint64_t da, uint64_t db, int scaleD) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %130, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
        "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, "
        "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, "
        "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, "
        "%128, %129, p, %131, %132, %133, %134;\n}\n"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
          "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
          "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
          "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),
          "+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),
          "+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),
          "+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),
          "+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]),
          "+f"(d[64]),"+f"(d[65]),"+f"(d[66]),"+f"(d[67]),"+f"(d[68]),"+f"(d[69]),"+f"(d[70]),"+f"(d[71]),
          "+f"(d[72]),"+f"(d[73]),"+f"(d[74]),"+f"(d[75]),"+f"(d[76]),"+f"(d[77]),"+f"(d[78]),"+f"(d[79]),
          "+f"(d[80]),"+f"(d[81]),"+f"(d[82]),"+f"(d[83]),"+f"(d[84]),"+f"(d[85]),"+f"(d[86]),"+f"(d[87]),
          "+f"(d[88]),"+f"(d[89]),"+f"(d[90]),"+f"(d[91]),"+f"(d[92]),"+f"(d[93]),"+f"(d[94]),"+f"(d[95]),
          "+f"(d[96]),"+f"(d[97]),"+f"(d[98]),"+f"(d[99]),"+f"(d[100]),"+f"(d[101]),"+f"(d[102]),"+f"(d[103]),
          "+f"(d[104]),"+f"(d[105]),"+f"(d[106]),"+f"(d[107]),"+f"(d[108]),"+f"(d[109]),"+f"(d[110]),"+f"(d[111]),
          "+f"(d[112]),"+f"(d[113]),"+f"(d[114]),"+f"(d[115]),"+f"(d[116]),"+f"(d[117]),"+f"(d[118]),"+f"(d[119]),
          "+f"(d[120]),"+f"(d[121]),"+f"(d[122]),"+f"(d[123]),"+f"(d[124]),"+f"(d[125]),"+f"(d[126]),"+f"(d[127])
        : "l"(da), "l"(db), "r"(scaleD), "n"(1), "n"(1), "n"(1), "n"(1));
}

// stats + fused LayerNorm, one warp per RPW rows over B*Fp rows (one X read).
// Writes M[b,f] (bf16, COLUMN-major for the unpadded gate), R[b,f] (fp32), and N_pad[b,f,:]
// (bf16, the pool's WGMMA B-operand; pad rows f in [F,Fp) -> 0). NITER=ceil((D/2)/32)=3 for D=192.
constexpr int RPW = 1, NITER = 3;

// stats-only (M,R=rrms; no N). VECTORIZED: int4 (16B) X loads + an SG-lane sub-group reduction over D (each
// warp does 32/SG rows). The old RPW=1/4-byte-load version was SM-issue-bound (NCU: sm 85.6%, DRAM 54% — too
// many small loads); int4 + SG=8 makes it BW-bound at ~92% of this GPU's real ~2.3 TB/s read ceiling (~0.42ms).
// (Ported from GEMS/2026-06-08-stats-mr-vec128; final math swapped LayerNorm rstd -> RMSNorm rrms.)
#ifndef STATS_SG
#define STATS_SG 8
#endif
constexpr int STATS_WPB = 8;                         // warps/block for k_stats_mr (block = 256 threads)
template <int SG>
__device__ __forceinline__ void stats_mr_body(const __nv_bfloat16* __restrict__ X,
                             __nv_bfloat16* __restrict__ M, float* __restrict__ R,
                             int B, int F, int Fp, int D, float eps) {
    constexpr int ROWS_PER_WARP = 32 / SG;
    const int tid = threadIdx.x, lane = tid & 31;
    const int warpInBlk = tid >> 5, warpsPerBlk = blockDim.x >> 5;
    const int sub = lane / SG, sl = lane % SG;       // row within warp / lane within sub-group
    const long warpId = (long)blockIdx.x * warpsPerBlk + warpInBlk;
    const long prow = warpId * ROWS_PER_WARP + sub;
    const long total = (long)B * Fp;
    const int D8 = D >> 3;                            // # int4 (8 bf16) along D
    const unsigned sgmask = ((SG == 32) ? 0xffffffffu : ((1u << SG) - 1u)) << (sub * SG);
    bool ok = (prow < total);
    int b = ok ? (int)(prow / Fp) : 0, f = ok ? (int)(prow % Fp) : 0;
    bool valid = ok && (f < F);
    float s = 0.f, s2 = 0.f;
    if (valid) {
        const int4* x4 = reinterpret_cast<const int4*>(X + ((long)b * F + f) * D);
        #pragma unroll
        for (int i = sl; i < D8; i += SG) {
            int4 raw = x4[i];
            const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
            #pragma unroll
            for (int k = 0; k < 4; k++) { float a = b2f(v2[k].x), c = b2f(v2[k].y); s += a + c; s2 += a * a + c * c; }
        }
    }
    #pragma unroll
    for (int o = SG >> 1; o > 0; o >>= 1) { s += __shfl_down_sync(sgmask, s, o, SG); s2 += __shfl_down_sync(sgmask, s2, o, SG); }
    if (valid && sl == 0) {
        float m = s / D;                             // mean (gate input)
        float rrms = rsqrtf(s2 / D + eps);           // RMSNorm: rsqrt(mean(X^2)+eps), no centering
        M[(long)f * B + b] = __float2bfloat16(m);
        R[(long)b * F + f] = rrms;
    }
}
__global__ void k_stats_mr(const __nv_bfloat16* __restrict__ X,
                           __nv_bfloat16* __restrict__ M, float* __restrict__ R,
                           int B, int F, int Fp, int D, float eps) {
    stats_mr_body<STATS_SG>(X, M, R, B, F, Fp, D, eps);
}
// N-only LayerNorm: reads X + M (col-major bf16) + R + gamma/beta -> Npad[B,Fp,D]. Runs CONCURRENTLY
// with the gate on a 2nd stream (the gate needs only M, and is at 19% DRAM -> N's traffic hides in the
// spare bandwidth). f>=F rows -> 0. One warp per (b,f) row.
__global__ void k_stats_n(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ M,
                          const float* __restrict__ R, __nv_bfloat16* __restrict__ Npad,
                          const __nv_bfloat16* __restrict__ GAMMA, const __nv_bfloat16* __restrict__ BETA,
                          int B, int F, int Fp, int D) {
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
    const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(GAMMA + (long)f * D);  // W[f,:] per-feature row
    const __nv_bfloat162* be2 = reinterpret_cast<const __nv_bfloat162*>(BETA);
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(X + ((long)b * F + f) * D);
    (void)M; (void)be2;
    float rrms = R[(long)b * F + f];
    for (int i = 0; i < NITER; i++) { int d2 = lane + i * 32;
        if (d2 < D2) { __nv_bfloat162 v = x2[d2], g = g2[d2];
            float n0 = b2f(v.x) * rrms * b2f(g.x);     // RMSNorm: X * rrms * W (W passed as GAMMA), no centering/bias
            float n1 = b2f(v.y) * rrms * b2f(g.y);
            n2[d2] = __halves2bfloat162(__float2bfloat16(n0), __float2bfloat16(n1)); } }
}

__global__ void k_stats(const __nv_bfloat16* __restrict__ X,
                        __nv_bfloat16* __restrict__ M, float* __restrict__ R,
                        __nv_bfloat16* __restrict__ Npad,
                        const __nv_bfloat16* __restrict__ GAMMA, const __nv_bfloat16* __restrict__ BETA,
                        int B, int F, int Fp, int D, float eps) {
    int lane = threadIdx.x & 31;
    long warpId = (long)blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    long total = (long)B * Fp, base = warpId * RPW;
    int D2 = D >> 1;
    const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(GAMMA);
    const __nv_bfloat162* be2 = reinterpret_cast<const __nv_bfloat162*>(BETA);
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
        float rrms = rsqrtf(s2[r] / D + eps);        // RMSNorm: rsqrt(mean(X^2)+eps), no centering
        (void)be2;
        if (lane == 0) { M[(long)ff[r] * B + bb[r]] = __float2bfloat16(m); R[(long)bb[r] * F + ff[r]] = rrms; }  // M column-major [B,F]
        #pragma unroll
        for (int i = 0; i < NITER; i++) { int d2 = lane + i * 32;
            if (d2 < D2) { __nv_bfloat162 v = xv[r][i], g = g2[(long)ff[r] * D2 + d2];  // W[ff,:] per-feature row
                float n0 = b2f(v.x) * rrms * b2f(g.x);     // RMSNorm: X * rrms * W (W passed as GAMMA), no centering/bias
                float n1 = b2f(v.y) * rrms * b2f(g.y);
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

// ================= HAND-WRITTEN CUDA+PTX GATE GEMM (no CUTLASS) — Phase 1 baseline =================
// L = M @ P (RAW logits, no sigmoid). M col-major [B,F] (M[m+k*B]), P row-major [F,QF] (P[k*QF+n]),
// L row-major [B,QF]. One CTA per (m64-tile, n64-tile), acc[32], reuses the VERIFIED no-swizzle wgmma
// machinery (wg_desc + cute m64n64 atom + proxy fence). v0 = correctness baseline (single-buffer, A staged
// K-major via coalesced-read/scatter-store — the path C measured as A-transpose-bound; fixed in later revs).
constexpr int HG_BK = 64, HG_BN = 256, HG_NSUB = HG_BN / 64, HG_BM = 128, HG_MSUB = HG_BM / 64, HG_S = 4;
// v14: TRUE 3-WARPGROUP WARP-SPEC (CUTLASS TmaWarpSpecializedCooperative structure). 384 threads:
//   WG2 (tid>=256) = PRODUCER: leader streams TMA (2 A m64 + 4 B n64) far ahead into an HG_S-stage pipeline,
//     gated only by empt[]; holds NO accumulator. WG0/WG1 (tid<256) = CONSUMERS: each owns one m64-tile,
//     streams wgmma (acc[128]) reading its A subtile + the SHARED B; sync ONLY via mbarrier full/empt (NO
//     __syncthreads). full[buf]=TMA complete (expect_tx); empt[buf]=both consumers done with shared B[buf].
__global__ __launch_bounds__(384, 1)
void k_gate_hand_b(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
                   const __grid_constant__ CUtensorMap tmL,
                   __nv_bfloat16* __restrict__ L, int B, int F, long QF) {
    int m0 = blockIdx.x * HG_BM; long n0 = (long)blockIdx.y * HG_BN;
    int tid = threadIdx.x;
    const int TILE = 64 * 64;
    extern __shared__ __align__(128) char hgsm[];
    __nv_bfloat16* As = (__nv_bfloat16*)hgsm;
    __nv_bfloat16* Bs = As + HG_S * HG_MSUB * TILE;
    uint64_t* full = (uint64_t*)(Bs + HG_S * HG_NSUB * TILE);  // [HG_S]
    uint64_t* empt = full + HG_S;                             // [HG_S]
    const int Asb = HG_MSUB * TILE, Bsb = HG_NSUB * TILE;
    const unsigned txBytes = (unsigned)((HG_MSUB + HG_NSUB) * 64 * 64 * 2);
    int Kp = ((F + 63) / 64) * 64, ntK = Kp / 64;
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < HG_S; s++) { mbar_init(&full[s], 1); mbar_init(&empt[s], 256); }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();

    if (tid >= 256) {
        // ===== PRODUCER warpgroup: leader streams TMA far ahead (gated by empt), no acc =====
        if (tid == 256) {
            int pe[HG_S];
            #pragma unroll
            for (int s = 0; s < HG_S; s++) pe[s] = 0;
            for (int t = 0; t < ntK; t++) {
                int buf = t % HG_S, k0 = t * 64;
                if (t >= HG_S) { mbar_wait(&empt[buf], pe[buf]); pe[buf] ^= 1; }   // shared B[buf] free
                mbar_expect_tx(&full[buf], txBytes);
                #pragma unroll
                for (int ms = 0; ms < HG_MSUB; ms++)
                    tma_load_2d(&As[buf * Asb + ms * TILE], &tmA, m0 + ms * 64, k0, &full[buf]);
                #pragma unroll
                for (int n2 = 0; n2 < HG_NSUB; n2++)
                    tma_load_2d(&Bs[buf * Bsb + n2 * TILE], &tmB, (int)(n0 + (long)n2 * 64), k0, &full[buf]);
            }
        }
    } else {
        // ===== CONSUMER warpgroups: each owns one m64-tile, streams wgmma (no __syncthreads) =====
        int wg = tid >> 7, warp = (tid >> 5) & 3, lane = tid & 31;   // wg in {0,1} = m64-tile
        float acc[HG_NSUB * 32];
        #pragma unroll
        for (int i = 0; i < HG_NSUB * 32; i++) acc[i] = 0.f;
        int pf[HG_S];
        #pragma unroll
        for (int s = 0; s < HG_S; s++) pf[s] = 0;
        // NOTE: the kernel is TENSOR-PIPE-THROUGHPUT-bound (NCU: sm__pipe_tensor_op_hmma 72.5%,
        // ~= CUTLASS MFU). A wgmma software pipeline (warpgroup_wait<1> + delayed empt) was MEASURED:
        // it cut stalled_barrier 4.82->2.84 but left latency unchanged (stalls are off the critical
        // path), so the simpler per-tile drain (wait<0>) is kept. The residual ~8% vs CUTLASS 1.26ms
        // is skinny-K tensor-idle (per-CTA epilogue + fill/drain), which needs a persistent overlap.
        static_assert(HG_BN == 256, "m64n256 atom requires HG_BN==256");
        for (int t = 0; t < ntK; t++) {
            int buf = t % HG_S;
            mbar_wait(&full[buf], pf[buf]); pf[buf] ^= 1;        // tile t's TMA landed
            // NOTE: no fence.proxy.async here -- TMA (cp.async.bulk) writes via the ASYNC proxy and
            // the mbarrier completion already orders it visible to wgmma's async-proxy read. The proxy
            // fence is only needed for GENERIC-proxy producers (old cp.async/__float2bfloat16 path).
            wgmma_fenceN(acc, HG_NSUB * 32);
            cute::warpgroup_arrive();
            // single m64n256 per k16 (1 wgmma issue vs 4x m64n64): B spans the 4 contiguous
            // 64x64 bricks via LBO=8192. acc[128] layout is identical, epilogue unchanged.
            #pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                uint64_t da = wg_desc_sw3(&As[buf * Asb + wg * TILE + ks * 1024], 128, 1024);
                uint64_t db = wg_desc_sw3(&Bs[buf * Bsb + ks * 1024], 8192, 1024);
                wgmma_m64n256k16_AN(acc, da, db, 1);
            }
            cute::warpgroup_commit_batch();
            cute::warpgroup_wait<0>();
            wgmma_fenceN(acc, HG_NSUB * 32);
            mbar_arrive(&empt[buf]);                             // shared B[buf] free (256 consumer arrivals)
        }
        // ===== TMA-STORE epilogue: stage acc -> smem (contiguous [HG_BM][HG_BN]), then ONE thread =====
        // issues a bulk tensor store smem -> L (hardware-coalesced, OOB-clips the QF tail; frees the
        // warps from per-thread stores -- the CUTLASS SM90_TMA_STORE recipe). STG must be the contiguous
        // HG_BN (the 2D store box is row-major), so the bank-pad is dropped here.
        const int STG = HG_BN;
        __nv_bfloat16* stg = (__nv_bfloat16*)hgsm;       // [HG_BM rows][HG_BN cols]; reuses As/Bs (free now)
        bar_sync_consumers();                            // all consumers done reading As/Bs in the mainloop
        #pragma unroll
        for (int g = 0; g < 32; g++) {
            int r0 = wg * 64 + 16 * warp + lane / 4, r1 = r0 + 8;
            int c0 = 8 * g + (lane % 4) * 2;
            *(__nv_bfloat162*)&stg[r0 * STG + c0] = __floats2bfloat162_rn(acc[4*g+0], acc[4*g+1]);
            *(__nv_bfloat162*)&stg[r1 * STG + c0] = __floats2bfloat162_rn(acc[4*g+2], acc[4*g+3]);
        }
        bar_sync_consumers();                            // staging complete
        if (tid == 0) {
            wgmma_async_proxy_fence();                   // smem stores visible to the async-proxy store
            tma_store_2d(&tmL, stg, (int)n0, m0);        // L tile at (col=n0, row=m0); OOB-clips QF tail
            tma_store_commit();
            tma_store_wait<0>();                         // ensure landed before the CTA frees smem
        }
    }
}

// ===== PERSISTENT warp-spec gate: overlap tile T's epilogue with tile T+1's mainloop =====
// v17 is tensor-pipe-bound (NCU 72.5%); the residual ~8% vs CUTLASS 1.26 is per-CTA tensor-idle
// (epilogue + pipeline fill/drain, no wgmma running). Fix: ONE persistent CTA per SM streams a
// grid-strided set of output tiles through ONE CONTINUOUS TMA->wgmma pipeline -- the buffer index
// gk is continuous ACROSS output-tile boundaries, so the producer WG prefetches tile T+1's first
// k-tiles into the pipeline WHILE the consumers run tile T's epilogue. The tensor pipe never drains
// between tiles (no per-tile fill bubble), and the epilogue tensor-idle is hidden. Tile order
// (m=ti%nM fastest, n=ti/nM) reproduces the non-persistent scheduler's wave => same L2 reuse of P.
// Staging is a SEPARATE smem region (the mainloop buffers stay live for the producer's prefetch),
// so HGP_S=3 stages to fit smem (210KB).
// DECOUPLED A/B PRODUCERS + asymmetric ring depths: A 2-deep (M is L2-resident — M=4.6MB fits L2,
// reused across all n in the m-fastest schedule => A loads mostly hit L2, depth 2 hides L2 latency),
// B 4-deep (P=569MB DRAM-streamed => deep pipe for TMA latency). A and B are streamed by TWO SEPARATE
// producer threads (tid256=B 4-ahead, tid288=A 2-ahead) with SEPARATE full_A/full_B barriers, so A's
// shallow ring does NOT bound B's lookahead (the single-producer asym version capped lookahead at 2 =>
// NCU long_scoreboard 5.76, tensor 52%). This lets the deep B pipe coexist with the 66KB separate
// staging inside 227KB smem, while persistent overlap hides the per-CTA epilogue/fill tensor-idle.
// FINAL persistent recipe: DEEP single-ring A4/B4 pipe (the v17 pipe that hides TMA, NCU long_scoreboard
// 1.2) + CHUNKED TMA-STORE epilogue. The 128x256 output is stored in 4 chunks of 64 cols via async bulk
// TMA store (double-buffered, wait<1>), so the epilogue staging is SMALL (2x128x64=32KB) and the deep S=4
// ring fits in 227KB smem -- avoiding the asymmetric-ring regression (A2 starvation, long_scoreboard 6.8).
// The async stores of tile T overlap tile T+1's mainloop; the continuous pipe (gk across tiles) removes
// the per-tile fill bubble. Tile order m=ti%nM fast => same L2 reuse of P as the non-persistent wave.
constexpr int HGP_S = 4, HGP_NCH = 4, HGP_CW = HG_BN / HGP_NCH;
__global__ __launch_bounds__(384, 1)
void k_gate_hand_persist(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
                         const __grid_constant__ CUtensorMap tmLc,
                         __nv_bfloat16* __restrict__ L, int B, int F, long QF, int dosig, int padfp) {
    const int TILE = 64 * 64;
    const int nM = (B + HG_BM - 1) / HG_BM;   // ceil: B<128 (e.g. B=8) must still make 1 m-tile (TMA clamps OOB rows)
    const int nN = (int)((QF + HG_BN - 1) / HG_BN);
    const int ntiles = nM * nN;
    const int ntK = (F + 63) / 64;
    extern __shared__ __align__(128) char hgsm[];
    __nv_bfloat16* As = (__nv_bfloat16*)hgsm;
    __nv_bfloat16* Bs = As + HGP_S * HG_MSUB * TILE;
    uint64_t* full = (uint64_t*)(Bs + HGP_S * HG_NSUB * TILE);   // [HGP_S]
    uint64_t* empt = full + HGP_S;                               // [HGP_S]
    __nv_bfloat16* cstg = (__nv_bfloat16*)(((uintptr_t)(empt + HGP_S) + 127) & ~(uintptr_t)127); // [2][HG_BM][HGP_CW]
    const int Asb = HG_MSUB * TILE, Bsb = HG_NSUB * TILE;
    const unsigned txBytes = (unsigned)((HG_MSUB + HG_NSUB) * 64 * 64 * 2);
    int tid = threadIdx.x;
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < HGP_S; s++) { mbar_init(&full[s], 1); mbar_init(&empt[s], 256); }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();

    if (tid >= 256) {
        // ===== PRODUCER: tid256 streams A+B into the single S=4 ring, continuous gk across tiles =====
        if (tid == 256) {
            long gk = 0;
            int pe[HGP_S];
            #pragma unroll
            for (int s = 0; s < HGP_S; s++) pe[s] = 0;
            for (int ti = blockIdx.x; ti < ntiles; ti += gridDim.x) {
                int m0 = (ti % nM) * HG_BM; long n0 = (long)(ti / nM) * HG_BN;
                for (int kt = 0; kt < ntK; kt++) {
                    int buf = (int)(gk % HGP_S), k0 = kt * 64;
                    if (gk >= HGP_S) { mbar_wait(&empt[buf], pe[buf]); pe[buf] ^= 1; }
                    mbar_expect_tx(&full[buf], txBytes);
                    #pragma unroll
                    for (int ms = 0; ms < HG_MSUB; ms++)
                        tma_load_2d(&As[buf * Asb + ms * TILE], &tmA, m0 + ms * 64, k0, &full[buf]);
                    #pragma unroll
                    for (int n2 = 0; n2 < HG_NSUB; n2++)
                        tma_load_2d(&Bs[buf * Bsb + n2 * TILE], &tmB, (int)(n0 + (long)n2 * 64), k0, &full[buf]);
                    gk++;
                }
            }
        }
    } else {
        // ===== CONSUMER: per-tile mainloop (acc reset) + CHUNKED TMA-store epilogue (overlapped) =====
        int wg = tid >> 7, warp = (tid >> 5) & 3, lane = tid & 31;
        long gk = 0;
        int pf[HGP_S];
        #pragma unroll
        for (int s = 0; s < HGP_S; s++) pf[s] = 0;
        for (int ti = blockIdx.x; ti < ntiles; ti += gridDim.x) {
            int m0 = (ti % nM) * HG_BM; long n0 = (long)(ti / nM) * HG_BN;
            float acc[HG_NSUB * 32];
            #pragma unroll
            for (int i = 0; i < HG_NSUB * 32; i++) acc[i] = 0.f;
            for (int kt = 0; kt < ntK; kt++) {
                int buf = (int)(gk % HGP_S);
                mbar_wait(&full[buf], pf[buf]); pf[buf] ^= 1;
                wgmma_fenceN(acc, HG_NSUB * 32);
                cute::warpgroup_arrive();
                #pragma unroll
                for (int ks = 0; ks < 4; ks++) {
                    uint64_t da = wg_desc_sw3(&As[buf * Asb + wg * TILE + ks * 1024], 128, 1024);
                    uint64_t db = wg_desc_sw3(&Bs[buf * Bsb + ks * 1024], 8192, 1024);
                    wgmma_m64n256k16_AN(acc, da, db, 1);
                }
                cute::warpgroup_commit_batch();
                cute::warpgroup_wait<0>();
                wgmma_fenceN(acc, HG_NSUB * 32);
                mbar_arrive(&empt[buf]);
                gk++;
            }
            // chunked TMA-store epilogue: 4 chunks of HGP_CW=64 cols, double-buffered (cb=c&1), async.
            #pragma unroll
            for (int c = 0; c < HGP_NCH; c++) {
                int cb = c & 1;
                __nv_bfloat16* ch = cstg + cb * (HG_BM * HGP_CW);
                if (c >= 2 && tid == 0) tma_store_wait<1>();   // chunk c-2's store done -> ch[cb] free
                bar_sync_consumers();                          // all wait for tid0's store-wait; ch free
                #pragma unroll
                for (int gg = 0; gg < HGP_CW / 8; gg++) {
                    int g = c * (HGP_CW / 8) + gg;
                    int r0 = wg * 64 + 16 * warp + lane / 4, r1 = r0 + 8;
                    int lc = 8 * gg + (lane % 4) * 2;
                    float a0 = acc[4*g+0], a1 = acc[4*g+1], b0 = acc[4*g+2], b1 = acc[4*g+3];
                    if (dosig) { a0 = sigmoidf(a0); a1 = sigmoidf(a1); b0 = sigmoidf(b0); b1 = sigmoidf(b1); }
                    if (padfp > 0) {   // PADDED output to g_SL[B,Q,Fp]: write 0 for g>=F pad cols.
                        // (padfp<0 = NOPADMASK A/B: skip — SL[pad] is don't-care since the fused pool's N[g>=F]=0,
                        //  so SL[pad]*N[pad]=0 regardless. Saves the per-col compare in the epilogue.)
                        int g0 = (int)(n0 % padfp) + c * HGP_CW + lc;   // tile is within one q (Fp%256==0)
                        if (g0     >= F) { a0 = 0.f; b0 = 0.f; }
                        if (g0 + 1 >= F) { a1 = 0.f; b1 = 0.f; }
                    }
                    *(__nv_bfloat162*)&ch[r0 * HGP_CW + lc] = __floats2bfloat162_rn(a0, a1);
                    *(__nv_bfloat162*)&ch[r1 * HGP_CW + lc] = __floats2bfloat162_rn(b0, b1);
                }
                bar_sync_consumers();                          // chunk staging visible
                wgmma_async_proxy_fence();                     // generic stores -> async-proxy visible
                if (tid == 0) {
                    tma_store_2d(&tmLc, ch, (int)(n0 + (long)c * HGP_CW), m0);
                    tma_store_commit();
                }
            }
            if (tid == 0) tma_store_wait<0>();                 // drain stores before next tile reuses cstg
        }
    }
}

// ===== PHASE 2.s: σ FUSED via a DEDICATED EPILOGUE WARPGROUP (the "one more WG" overlap) — OPTIMIZED =====
// NCU root-cause of inline-σ (+0.30ms): σ in the consumer epilogue drains the tensor pipe (hmma 78->57%,
// XU/SFU only 9.6% -- NOT SFU-bound, it's serialization). The store-offload EWG proved gate 1.258 = PARITY
// (the WG structure overlaps perfectly). The σ-on-E stall was the UNDER-OPTIMIZED 2-buffer cstg: when E's
// σ slowed it, consumers stalled MID-DUMP (≤1 chunk ahead). FIX: shrink the ring to S=3 (144KB) to free
// smem for a 4-buffer cstg (64KB = a FULL TILE of slack) -> consumers dump the whole tile and never wait E;
// E does σ + pipelined async stores entirely in the shadow of the next mainloop. 352 threads (no setmaxnreg).
constexpr int HGE_RING = 4, HGE_NCB = 2;                    // POWER-OF-2 ring+cstg: gk&3 / gc&1 stay in registers
// (S=3/NCB=4 forced gk%3 -> magic ÷3 + pf[3]/pce[] in LOCAL memory -> LDL/STL = long_scoreboard every iter; the
// power-of-2 sizes keep the parity arrays in registers, exactly why the original S=4 gate had long_scoreboard 2.0).
constexpr int HGE_NCON = 256, HGE_NE = 96, HGE_TPB = 384;   // consumer / epilogue-WG (3 warps) / total threads
constexpr int HGE_KEEP = HGE_NCB - 1;                       // max stores E keeps in flight (= free 1-behind for NCB=2)
// 384 threads => ptxas base 65536/384=170 >= wgmma's 154 need, so it compiles WITHOUT setmaxnreg. E=96 (3 warps,
// vs the 2-warp E that was LATENCY-bound on σ: σ-no-store==σ+store=3.25ms proved store is free, σ-on-2warp-E
// couldn't hide the EX2->FADD->RCP chain). More warps => more independent σ chains in flight to hide latency.
__global__ __launch_bounds__(HGE_TPB, 1)
void k_gate_hand_persist_ewg(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
                             const __grid_constant__ CUtensorMap tmLc,
                             __nv_bfloat16* __restrict__ L, int B, int F, long QF, int dosig) {
    const int TILE = 64 * 64;
    const int nM = (B + HG_BM - 1) / HG_BM;   // ceil: B<128 (e.g. B=8) must still make 1 m-tile (TMA clamps OOB rows)
    const int nN = (int)((QF + HG_BN - 1) / HG_BN);
    const int ntiles = nM * nN;
    const int ntK = (F + 63) / 64;
    extern __shared__ __align__(128) char hgsm[];
    __nv_bfloat16* As = (__nv_bfloat16*)hgsm;
    __nv_bfloat16* Bs = As + HGE_RING * HG_MSUB * TILE;
    uint64_t* full = (uint64_t*)(Bs + HGE_RING * HG_NSUB * TILE);   // [HGE_RING]
    uint64_t* empt = full + HGE_RING;                               // [HGE_RING]
    __nv_bfloat16* cstg = (__nv_bfloat16*)(((uintptr_t)(empt + HGE_RING) + 127) & ~(uintptr_t)127); // [HGE_NCB][HG_BM][HGP_CW]
    uint64_t* cfull = (uint64_t*)(((uintptr_t)(cstg + HGE_NCB * HG_BM * HGP_CW) + 127) & ~(uintptr_t)127); // [HGE_NCB]
    uint64_t* cempt = cfull + HGE_NCB;                              // [HGE_NCB] E->consumer (buffer free)
    const int Asb = HG_MSUB * TILE, Bsb = HG_NSUB * TILE;
    const unsigned txBytes = (unsigned)((HG_MSUB + HG_NSUB) * 64 * 64 * 2);
    const int CB = HG_BM * HGP_CW;                                  // one cstg chunk (128x64)
    int tid = threadIdx.x;
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < HGE_RING; s++) { mbar_init(&full[s], 1); mbar_init(&empt[s], HGE_NCON); }
        #pragma unroll
        for (int c = 0; c < HGE_NCB; c++) { mbar_init(&cfull[c], HGE_NCON); mbar_init(&cempt[c], 1); }  // cempt: tid256 only
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();

    if (tid >= HGE_NCON + HGE_NE) {
        // ===== PRODUCER (tid 352): stream A+B into the S=3 ring; only tid352 issues TMA =====
        if (tid == HGE_NCON + HGE_NE) {
            long gk = 0;
            int pe[HGE_RING];
            #pragma unroll
            for (int s = 0; s < HGE_RING; s++) pe[s] = 0;
            for (int ti = blockIdx.x; ti < ntiles; ti += gridDim.x) {
                int m0 = (ti % nM) * HG_BM; long n0 = (long)(ti / nM) * HG_BN;
                for (int kt = 0; kt < ntK; kt++) {
                    int buf = (int)(gk % HGE_RING), k0 = kt * 64;
                    if (gk >= HGE_RING) { mbar_wait(&empt[buf], pe[buf]); pe[buf] ^= 1; }
                    mbar_expect_tx(&full[buf], txBytes);
                    #pragma unroll
                    for (int ms = 0; ms < HG_MSUB; ms++)
                        tma_load_2d(&As[buf * Asb + ms * TILE], &tmA, m0 + ms * 64, k0, &full[buf]);
                    #pragma unroll
                    for (int n2 = 0; n2 < HG_NSUB; n2++)
                        tma_load_2d(&Bs[buf * Bsb + n2 * TILE], &tmB, (int)(n0 + (long)n2 * 64), k0, &full[buf]);
                    gk++;
                }
            }
        }
    } else if (tid >= HGE_NCON) {
        // ===== EPILOGUE WG (tid 256-351, 3 warps): σ in-place on the dumped chunk + PIPELINED async TMA-store. With a
        //       4-buffer cstg E can work a full tile behind the consumers (never throttles their dump), and
        //       keeps up to 2 stores in flight, so σ + store run entirely under the next tile's wgmma.
        int et = tid - HGE_NCON;            // 0..63
        int pf[HGE_NCB];                    // cfull parity (chunk ready)
        #pragma unroll
        for (int c = 0; c < HGE_NCB; c++) pf[c] = 0;
        long gc = 0;
        for (int ti = blockIdx.x; ti < ntiles; ti += gridDim.x) {
            int m0 = (ti % nM) * HG_BM; long n0 = (long)(ti / nM) * HG_BN;
            #pragma unroll
            for (int c = 0; c < HGP_NCH; c++) {
                int cb = (int)(gc % HGE_NCB);
                mbar_wait(&cfull[cb], pf[cb]); pf[cb] ^= 1;     // consumers dumped this chunk (RAW logits) into cstg[cb]
                __nv_bfloat16* ch = cstg + cb * CB;
                if (dosig) {                                    // σ in-place (packed bf16x2). Batch the smem LOADS
                    // into registers (UF-deep ILP) so many loads are in flight -> hides the smem-load latency
                    // that otherwise stalls E's 2 warps on short-scoreboard (NCU: 56.7% of cycles).
                    __nv_bfloat162* p = (__nv_bfloat162*)ch;
                    constexpr int UF = 8;
                    #pragma unroll
                    for (int base = et; base < CB / 2; base += HGE_NE * UF) {
                        float2 r[UF];
                        #pragma unroll
                        for (int u = 0; u < UF; u++) { int i = base + u * HGE_NE; if (i < CB / 2) r[u] = __bfloat1622float2(p[i]); }
                        #pragma unroll
                        for (int u = 0; u < UF; u++) { r[u].x = sigmoidf(r[u].x); r[u].y = sigmoidf(r[u].y); }
                        #pragma unroll
                        for (int u = 0; u < UF; u++) { int i = base + u * HGE_NE; if (i < CB / 2) p[i] = __floats2bfloat162_rn(r[u].x, r[u].y); }
                    }
                }
                asm volatile("bar.sync 2, %0;" :: "n"(HGE_NE) : "memory");   // all E σ-writes for this chunk done
                if (et == 0) {                                  // ONLY tid256 stores + waits + frees (cempt count=1):
                    if (dosig != 2) {                           // dosig==2 = DEBUG: σ but skip store (isolate σ vs store)
                        wgmma_async_proxy_fence();              // generic σ stores -> async-proxy visible
                        tma_store_2d(&tmLc, ch, (int)(n0 + (long)c * HGP_CW), m0);
                        tma_store_commit();
                        if (gc >= HGE_KEEP) { tma_store_wait<HGE_KEEP>(); mbar_arrive(&cempt[(int)((gc - HGE_KEEP) & (HGE_NCB - 1))]); }
                    } else {
                        if (gc >= HGE_KEEP) mbar_arrive(&cempt[(int)((gc - HGE_KEEP) & (HGE_NCB - 1))]);  // free immediately (no store)
                    }
                }
                gc++;
            }
        }
        if (et == 0) {
            tma_store_wait<0>();                                // drain the final in-flight stores
            #pragma unroll
            for (int j = HGE_KEEP; j >= 1; j--) mbar_arrive(&cempt[(int)((gc - j) & (HGE_NCB - 1))]);  // free last KEEP buffers (tail)
        }
    } else {
        // ===== CONSUMERS (tid 0-255): per-tile wgmma mainloop + a FAST RAW acc->smem dump (NO σ, NO store).
        //       4-buffer cstg = a whole tile of slack: the consumer dumps all 4 chunks without waiting E, then
        //       returns straight to the next tile's wgmma. The tensor pipe is never held by σ or the store.
        int wg = tid >> 7, warp = (tid >> 5) & 3, lane = tid & 31;
        long gk = 0;
        int pf[HGE_RING], pce[HGE_NCB];     // pf=ring full parity, pce=cempt parity (buffer free)
        #pragma unroll
        for (int s = 0; s < HGE_RING; s++) pf[s] = 0;
        #pragma unroll
        for (int c = 0; c < HGE_NCB; c++) pce[c] = 0;
        long gc = 0;
        for (int ti = blockIdx.x; ti < ntiles; ti += gridDim.x) {
            float acc[HG_NSUB * 32];
            #pragma unroll
            for (int i = 0; i < HG_NSUB * 32; i++) acc[i] = 0.f;
            for (int kt = 0; kt < ntK; kt++) {
                int buf = (int)(gk % HGE_RING);
                mbar_wait(&full[buf], pf[buf]); pf[buf] ^= 1;
                wgmma_fenceN(acc, HG_NSUB * 32);
                cute::warpgroup_arrive();
                #pragma unroll
                for (int ks = 0; ks < 4; ks++) {
                    uint64_t da = wg_desc_sw3(&As[buf * Asb + wg * TILE + ks * 1024], 128, 1024);
                    uint64_t db = wg_desc_sw3(&Bs[buf * Bsb + ks * 1024], 8192, 1024);
                    wgmma_m64n256k16_AN(acc, da, db, 1);
                }
                cute::warpgroup_commit_batch();
                cute::warpgroup_wait<0>();
                wgmma_fenceN(acc, HG_NSUB * 32);
                mbar_arrive(&empt[buf]);
                gk++;
            }
            // FAST RAW dump: stage acc -> cstg (4 chunks into 4 buffers), hand each to E via cfull. NO σ here.
            #pragma unroll
            for (int c = 0; c < HGP_NCH; c++) {
                int cb = (int)(gc % HGE_NCB);
                if (gc >= HGE_NCB) { mbar_wait(&cempt[cb], pce[cb]); pce[cb] ^= 1; }   // E freed buffer cb
                __nv_bfloat16* ch = cstg + cb * CB;
                #pragma unroll
                for (int gg = 0; gg < HGP_CW / 8; gg++) {
                    int g = c * (HGP_CW / 8) + gg;
                    int r0 = wg * 64 + 16 * warp + lane / 4, r1 = r0 + 8;
                    int lc = 8 * gg + (lane % 4) * 2;
                    *(__nv_bfloat162*)&ch[r0 * HGP_CW + lc] = __floats2bfloat162_rn(acc[4*g+0], acc[4*g+1]);
                    *(__nv_bfloat162*)&ch[r1 * HGP_CW + lc] = __floats2bfloat162_rn(acc[4*g+2], acc[4*g+3]);
                }
                bar_sync_consumers();                            // all consumer dumps for chunk c visible
                mbar_arrive(&cfull[cb]);                         // signal E: chunk cb ready (256 arrivals)
                gc++;
            }
        }
    }
}

// PHASE 2 one-time STATIC weight repack: P[F, Q*F] -> Pp[F, Q*Fp] (per-q F->Fp pad; pad region pre-zeroed).
// This makes the per-q gate B-read at column q*Fp+fb*256 64-aligned (SWIZZLE_128B legal) -- the only way to
// fuse the padded SL output, since F=1491 is odd (q*F not %64, and a 3D map's q-stride F*2 not %16). Cached
// on the P pointer: runs ONCE (P is a static weight), amortized to ~0 over the timed loop. NOT the removed
// repad (that was the DATA-dependent output sigmoid+pad, recomputed every forward).
__global__ void k_ppad(const __nv_bfloat16* __restrict__ P, __nv_bfloat16* __restrict__ Pp,
                       int F, int Q, long QF, long QFp, int Fp) {
    long blk = (long)blockIdx.x;                 // grid = (long)F * Q blocks; one (f', q) row
    long fpp = blk / Q, q = blk % Q;
    const __nv_bfloat16* src = P + fpp * QF + q * F;
    __nv_bfloat16* dst = Pp + fpp * QFp + q * Fp;
    for (int f = threadIdx.x; f < F; f += blockDim.x) dst[f] = src[f];
}

// ===== PHASE 2: FUSED gate — same persistent WGMMA GEMM as k_gate_hand_persist, but the epilogue =====
// applies sigmoid AND writes the PADDED SL[B, Q, Fp] DIRECTLY (f in [F,Fp) -> 0), eliminating k_sigpad and
// the repad entirely; run_pool reads SL straight from here. N is re-tiled per (q, f-block): Fp=1536=6*256
// so each 256-col output tile lives inside ONE q's padded Fp region => the chunked TMA store stays
// contiguous (col q*Fp+fb*256 .. , one q, no straddle). The gate B-operand P column for tile (q,fb) is
// q*F + fb*256 (F=1491 real); the fb=5 tail (f>=1491) reads wrapped/OOB P (computed, then masked to 0).
constexpr int HGF_S = 4;   // fused ring depth: S=4 (deep, hides TMA) + CHUNKED double-buffer fits 227KB smem
// PHASE 2 (fast): PER-Q tiling reads PADDED Pp (aligned) so the SL store is the FAST async-TMA store; the
// SIGMOID is OVERLAPPED with consumer wgmma by moving it onto the idle producer warpgroup: consumers stage
// the RAW bf16 logit to smem, then producer threads sigmoid it IN PLACE and async-TMA-store to padded SL —
// all while the consumers run the next tile's wgmma. (Pp is a one-time weight prepack, refreshed on a side
// stream hidden under k_stats; the harness's in-place P re-fill makes a pointer cache stale otherwise.)
__global__ __launch_bounds__(384, 1)
void k_gate_hand_fused(const __grid_constant__ CUtensorMap tmBp, const __grid_constant__ CUtensorMap tmA,
                       const __grid_constant__ CUtensorMap tmSL,
                       __nv_bfloat16* __restrict__ SL, int B, int F, int Q, int Fp) {
    const int TILE = 64 * 64;
    const int nM = (B + HG_BM - 1) / HG_BM;   // ceil: B<128 (e.g. B=8) must still make 1 m-tile (TMA clamps OOB rows)
    const int nFB = Fp / HG_BN;                       // f-blocks per q (1536/256=6)
    const int nN = Q * nFB;
    const int ntiles = nM * nN;
    const int ntK = (F + 63) / 64;
    extern __shared__ __align__(128) char hgsm[];
    __nv_bfloat16* As = (__nv_bfloat16*)hgsm;
    __nv_bfloat16* Bs = As + HGF_S * HG_MSUB * TILE;
    uint64_t* full   = (uint64_t*)(Bs + HGF_S * HG_NSUB * TILE);   // [HGF_S] mainloop ring
    uint64_t* empt   = full + HGF_S;                               // [HGF_S]
    uint64_t* ep_full = empt + HGF_S;                              // [2] consumers staged a tile (double-buffered)
    uint64_t* ep_empt = ep_full + 2;                             // [2] producer drained the staging buffer
    __nv_bfloat16* estg = (__nv_bfloat16*)(((uintptr_t)(ep_empt + 2) + 127) & ~(uintptr_t)127);  // [2][HG_BM][HGP_CW]
    const int ESB = HG_BM * HGP_CW;                                // estg per-buffer stride (one 64-col chunk)
    const int Asb = HG_MSUB * TILE, Bsb = HG_NSUB * TILE;
    const unsigned txBytes = (unsigned)((HG_MSUB + HG_NSUB) * 64 * 64 * 2);
    const int NEP = 96;                               // epilogue threads = producer warps 9,10,11 (tid 288..383)
    int tid = threadIdx.x;
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < HGF_S; s++) { mbar_init(&full[s], 1); mbar_init(&empt[s], 256); }
        #pragma unroll
        for (int e = 0; e < 2; e++) { mbar_init(&ep_full[e], 256); mbar_init(&ep_empt[e], NEP); }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    if (tid >= 256) {
        if (tid == 256) {
            // PRODUCER-TMA: per-q col q*Fp+fb*256 into PADDED Pp (aligned), continuous S=3 ring.
            long gk = 0;
            int pe[HGF_S];
            #pragma unroll
            for (int s = 0; s < HGF_S; s++) pe[s] = 0;
            for (int ti = blockIdx.x; ti < ntiles; ti += gridDim.x) {
                int i = ti % nM, nu = ti / nM, q = nu / nFB, fb = nu % nFB;
                int m0 = i * HG_BM, n0 = q * Fp + fb * HG_BN;
                for (int kt = 0; kt < ntK; kt++) {
                    int buf = (int)(gk % HGF_S), k0 = kt * 64;
                    if (gk >= HGF_S) { mbar_wait(&empt[buf], pe[buf]); pe[buf] ^= 1; }
                    mbar_expect_tx(&full[buf], txBytes);
                    #pragma unroll
                    for (int ms = 0; ms < HG_MSUB; ms++)
                        tma_load_2d(&As[buf * Asb + ms * TILE], &tmA, m0 + ms * 64, k0, &full[buf]);
                    #pragma unroll
                    for (int n2 = 0; n2 < HG_NSUB; n2++)
                        tma_load_2d(&Bs[buf * Bsb + n2 * TILE], &tmBp, n0 + n2 * 64, k0, &full[buf]);
                    gk++;
                }
            }
        } else if (tid >= 288) {
            // PRODUCER-EPILOGUE (96 threads): per 64-col CHUNK, sigmoid IN PLACE on estg[cb] + async-TMA store,
            // OVERLAPS consumer wgmma. Chunk-level double-buffer (cb = global_chunk&1) -> no stall, deep S=4 ring.
            int et = tid - 288; long gc = 0; int pe[2] = {0, 0}, prev_cb = -1;
            for (int ti = blockIdx.x; ti < ntiles; ti += gridDim.x) {
                int i = ti % nM, nu = ti / nM, q = nu / nFB, fb = nu % nFB;
                int m0 = i * HG_BM, fbase = fb * HG_BN;
                long slq = (long)q * Fp + (long)fb * HG_BN;
                #pragma unroll
                for (int c = 0; c < HG_BN / HGP_CW; c++) {
                    int cb = (int)(gc & 1);
                    __nv_bfloat16* eb = estg + cb * ESB;
                    mbar_wait(&ep_full[cb], pe[cb]); pe[cb] ^= 1;     // consumers staged chunk gc into cb
                    for (int idx = et; idx < HG_BM * HGP_CW; idx += NEP) {
                        int clc = idx & (HGP_CW - 1);
                        int f = fbase + c * HGP_CW + clc;
                        eb[idx] = (f < F) ? __float2bfloat16(sigmoidf(b2f(eb[idx]))) : __float2bfloat16(0.f);
                    }
                    asm volatile("bar.sync 2, 96;" ::: "memory");
                    if (et == 0) {
                        wgmma_async_proxy_fence();
                        tma_store_2d(&tmSL, eb, (int)(slq + (long)c * HGP_CW), m0);  // aligned 64x128 chunk store
                        tma_store_commit();
                        if (prev_cb >= 0) tma_store_wait<1>();   // PIPELINED: drain only the PREVIOUS chunk's store
                    }
                    asm volatile("bar.sync 2, 96;" ::: "memory");
                    if (prev_cb >= 0) mbar_arrive(&ep_empt[prev_cb]);  // prev buffer free (its store drained)
                    prev_cb = cb;
                    gc++;
                }
            }
            if (et == 0) tma_store_wait<0>();          // drain the final in-flight store
            asm volatile("bar.sync 2, 96;" ::: "memory");
            if (prev_cb >= 0) mbar_arrive(&ep_empt[prev_cb]);
        }
    } else {
        // CONSUMERS: wgmma mainloop -> stage RAW bf16 logit in 4 CHUNKS to estg[cb] -> hand to producer-epilogue.
        int wg = tid >> 7, warp = (tid >> 5) & 3, lane = tid & 31;
        long gk = 0, gc = 0;
        int pf[HGF_S], ce[2] = {0, 0};
        #pragma unroll
        for (int s = 0; s < HGF_S; s++) pf[s] = 0;
        for (int ti = blockIdx.x; ti < ntiles; ti += gridDim.x) {
            float acc[HG_NSUB * 32];
            #pragma unroll
            for (int z = 0; z < HG_NSUB * 32; z++) acc[z] = 0.f;
            for (int kt = 0; kt < ntK; kt++) {
                int buf = (int)(gk % HGF_S);
                mbar_wait(&full[buf], pf[buf]); pf[buf] ^= 1;
                wgmma_fenceN(acc, HG_NSUB * 32);
                cute::warpgroup_arrive();
                #pragma unroll
                for (int ks = 0; ks < 4; ks++) {
                    uint64_t da = wg_desc_sw3(&As[buf * Asb + wg * TILE + ks * 1024], 128, 1024);
                    uint64_t db = wg_desc_sw3(&Bs[buf * Bsb + ks * 1024], 8192, 1024);
                    wgmma_m64n256k16_AN(acc, da, db, 1);
                }
                cute::warpgroup_commit_batch();
                cute::warpgroup_wait<0>();
                wgmma_fenceN(acc, HG_NSUB * 32);
                mbar_arrive(&empt[buf]);
                gk++;
            }
            // stage RAW logit in 4 chunks; chunk-level double-buffer (cb=gc&1) -> hand each to the producer
            #pragma unroll
            for (int c = 0; c < HG_BN / HGP_CW; c++) {
                int cb = (int)(gc & 1);
                __nv_bfloat16* eb = estg + cb * ESB;
                if (gc >= 2) { mbar_wait(&ep_empt[cb], ce[cb]); ce[cb] ^= 1; }
                #pragma unroll
                for (int gg = 0; gg < HGP_CW / 8; gg++) {
                    int g = c * (HGP_CW / 8) + gg;
                    int r0 = wg * 64 + 16 * warp + lane / 4, r1 = r0 + 8;
                    int clc = 8 * gg + (lane % 4) * 2;
                    *(__nv_bfloat162*)&eb[r0 * HGP_CW + clc] = __floats2bfloat162_rn(acc[4*g+0], acc[4*g+1]);
                    *(__nv_bfloat162*)&eb[r1 * HGP_CW + clc] = __floats2bfloat162_rn(acc[4*g+2], acc[4*g+3]);
                }
                bar_sync_consumers();
                mbar_arrive(&ep_full[cb]);
                gc++;
            }
        }
    }
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

// ===== PHASE 3a: HAND-WRITTEN persistent warp-spec WGMMA pool (replaces CUTLASS run_pool) =====
// O[b] = SL[b] @ N[b], per batch [M=Q=128, N=D=192, K=Fp=1536], batch=B=1536. One batch = one
// output tile [128x192]. Mirrors k_gate_hand_persist: 132 persistent CTAs grid-stride over batches,
// 3 WG (producer tid>=256 issues TMA; 2 consumers each own 64 Q-rows), S-stage ring, continuous gk.
//   A = SL [B,Q,Fp] row-major -> K-major operand (Fp contiguous). TMA globalDim {Fp, B*Q}, box 64x64
//       SW128; load 2 M-halves at (k0, b*Q + ms*64). desc = wg_desc_sw3(&As[ks*16], 16, 1024).
//   B = N  [B,Fp,D] row-major -> MN-major operand (D contiguous). TMA globalDim {D, B*Fp}, box 64x64
//       SW128; load 3 D-bricks at (n2*64, b*Fp + k0). desc = wg_desc_sw3(&Bs[ks*1024], 8192, 1024).
//   wgmma.m64n192k16 (acc[96]/thread) x ntK=24 k-blocks. Epilogue: stage acc -> smem [128x192],
//   TMA-store O tile (box {192,128}, no swizzle). Validated K-major recipe: /tmp/pooltest192.cu max_err 0.
constexpr int HP_BM = 128, HP_BN = 192, HP_BK = 64, HP_MSUB = 2, HP_NSUB = 3, HP_S = 4;
__global__ __launch_bounds__(384, 1)
void k_pool_hand_persist(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
                         const __grid_constant__ CUtensorMap tmO,
                         __nv_bfloat16* __restrict__ O, int B, int Q, int D, int Fp) {
    const int TILE = HP_BK * HP_BK;                  // 64x64 brick
    const int ntiles = B;                            // one tile per batch
    const int ntK = Fp / HP_BK;                      // 24 k-blocks
    extern __shared__ __align__(128) char psm[];
    __nv_bfloat16* As = (__nv_bfloat16*)psm;                 // [HP_S][HP_MSUB][TILE]
    __nv_bfloat16* Bs = As + HP_S * HP_MSUB * TILE;         // [HP_S][HP_NSUB][TILE]
    uint64_t* full = (uint64_t*)(Bs + HP_S * HP_NSUB * TILE);// [HP_S]
    uint64_t* empt = full + HP_S;                            // [HP_S]
    __nv_bfloat16* ostg = (__nv_bfloat16*)(((uintptr_t)(empt + HP_S) + 127) & ~(uintptr_t)127); // [HP_BM][HP_BN]
    const int Asb = HP_MSUB * TILE, Bsb = HP_NSUB * TILE;
    const unsigned txBytes = (unsigned)((HP_MSUB + HP_NSUB) * TILE * 2);
    int tid = threadIdx.x;
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < HP_S; s++) { mbar_init(&full[s], 1); mbar_init(&empt[s], 256); }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();

    if (tid >= 256) {
        // ===== PRODUCER: tid256 streams A+B into the S-stage ring, continuous gk across batches =====
        if (tid == 256) {
            long gk = 0;
            int pe[HP_S];
            #pragma unroll
            for (int s = 0; s < HP_S; s++) pe[s] = 0;
            for (int ti = blockIdx.x; ti < ntiles; ti += gridDim.x) {
                int b = ti;
                int mrow = b * Q;            // flat M-row base for A (SL [B*Q, Fp])
                long brow = (long)b * Fp;    // flat row base for B (N [B*Fp, D])
                for (int kt = 0; kt < ntK; kt++) {
                    int buf = (int)(gk % HP_S), k0 = kt * HP_BK;
                    if (gk >= HP_S) { mbar_wait(&empt[buf], pe[buf]); pe[buf] ^= 1; }
                    mbar_expect_tx(&full[buf], txBytes);
                    #pragma unroll
                    for (int ms = 0; ms < HP_MSUB; ms++)
                        tma_load_2d(&As[buf * Asb + ms * TILE], &tmA, k0, mrow + ms * HP_BK, &full[buf]);
                    #pragma unroll
                    for (int n2 = 0; n2 < HP_NSUB; n2++)
                        tma_load_2d(&Bs[buf * Bsb + n2 * TILE], &tmB, n2 * HP_BK, (int)(brow + k0), &full[buf]);
                    gk++;
                }
            }
        }
    } else {
        // ===== CONSUMER: each wg owns 64 Q-rows; acc[96] (m64n192) per batch tile =====
        int wg = tid >> 7, warp = (tid >> 5) & 3, lane = tid & 31;
        long gk = 0;
        int pf[HP_S];
        #pragma unroll
        for (int s = 0; s < HP_S; s++) pf[s] = 0;
        for (int ti = blockIdx.x; ti < ntiles; ti += gridDim.x) {
            int b = ti;
            float acc[96];
            #pragma unroll
            for (int i = 0; i < 96; i++) acc[i] = 0.f;
            for (int kt = 0; kt < ntK; kt++) {
                int buf = (int)(gk % HP_S);
                mbar_wait(&full[buf], pf[buf]); pf[buf] ^= 1;
                wgmma_fenceN(acc, 96);
                cute::warpgroup_arrive();
                #pragma unroll
                for (int ks = 0; ks < 4; ks++) {
                    uint64_t da = wg_desc_sw3(&As[buf * Asb + wg * TILE + ks * 16], 16, 1024);
                    uint64_t db = wg_desc_sw3(&Bs[buf * Bsb + ks * 1024], 8192, 1024);
                    wgmma_cute_m64n192k16(acc, da, db);
                }
                cute::warpgroup_commit_batch();
                cute::warpgroup_wait<0>();
                wgmma_fenceN(acc, 96);
                mbar_arrive(&empt[buf]);
                gk++;
            }
            // ===== epilogue: stage acc -> ostg [HP_BM][HP_BN], one TMA store of the [128x192] O tile =====
            bar_sync_consumers();                            // all consumers done reading As/Bs ring
            #pragma unroll
            for (int g = 0; g < 24; g++) {
                int r0 = wg * 64 + 16 * warp + lane / 4, r1 = r0 + 8;
                int c0 = 8 * g + (lane % 4) * 2;
                *(__nv_bfloat162*)&ostg[r0 * HP_BN + c0] = __floats2bfloat162_rn(acc[4*g+0], acc[4*g+1]);
                *(__nv_bfloat162*)&ostg[r1 * HP_BN + c0] = __floats2bfloat162_rn(acc[4*g+2], acc[4*g+3]);
            }
            bar_sync_consumers();                            // staging complete
            if (tid == 0) {
                wgmma_async_proxy_fence();                   // generic stores -> async-proxy store visible
                tma_store_2d(&tmO, ostg, 0, b * Q);          // O tile [128x192] at (col=0, row=b*Q); box {192,128}
                tma_store_commit();
                tma_store_wait<0>();                         // landed before ostg reused next tile
            }
        }
    }
    (void)D;
}

// ===== PHASE 3b: FUSED pool — computes N = X*rrms*W INLINE (no materialized g_N). stats writes only M,R. =====
// Same persistent warp-spec WGMMA pool as k_pool_hand_persist, but the B-operand is built on the fly:
//   producer TMA-loads A=SL (2 bricks) + X (3 bricks, 3D TMA over [D,F,B], OOB-fill 0 for f>=F) + W (3 bricks)
//   consumers load rrms[b,k0:k0+64], compute N[g,d]=X[g,d]*rrms[g]*W[g,d] IN-PLACE over the X brick (rrms[g]
//   per-K-row via gh_sw_off(k,mn)), then wgmma. Mirrors Triton _pool_kernel (N=x*rg[:,None]*w, on the fly).
// Memory win: replaces the 0.906GB g_N read with a 0.441GB X read (+ W L2-cached) AND deletes the N write in stats.
constexpr int HPF_S = 3;   // fused ring depth (8 bricks/stage As2+Xs3+Ws3 -> S=3=192KB + chunked O-store fits 227KB)
__global__ __launch_bounds__(384, 1)
void k_pool_hand_fused(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmX,
                       const __grid_constant__ CUtensorMap tmW, const __grid_constant__ CUtensorMap tmO,
                       __nv_bfloat16* __restrict__ O, const float* __restrict__ R,
                       int B, int Q, int D, int F, int Fp) {
    const int TILE = HP_BK * HP_BK;                  // 64x64 brick
    const int ntiles = B;
    const int ntK = Fp / HP_BK;                      // 24 k-blocks
    extern __shared__ __align__(128) char psm[];
    __nv_bfloat16* As = (__nv_bfloat16*)psm;                  // [HPF_S][HP_MSUB][TILE]  (SL, K-major)
    __nv_bfloat16* Xs = As + HPF_S * HP_MSUB * TILE;         // [HPF_S][HP_NSUB][TILE]  (X -> N in place)
    __nv_bfloat16* Ws = Xs + HPF_S * HP_NSUB * TILE;         // [HPF_S][HP_NSUB][TILE]  (W, TMA-staged)
    uint64_t* full = (uint64_t*)(Ws + HPF_S * HP_NSUB * TILE);// [HPF_S]
    uint64_t* empt = full + HPF_S;                            // [HPF_S]
    float* rsm = (float*)(empt + HPF_S);                     // [64] rrms for the current k-block
    __nv_bfloat16* ostg = (__nv_bfloat16*)(((uintptr_t)(rsm + 64) + 127) & ~(uintptr_t)127); // [2][HP_BM][64] chunked
    const int Asb = HP_MSUB * TILE, Xsb = HP_NSUB * TILE, Wsb = HP_NSUB * TILE;
    const unsigned txBytes = (unsigned)((HP_MSUB + 2 * HP_NSUB) * TILE * 2);   // SL2 + X3 + W3 bricks
    int tid = threadIdx.x;
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < HPF_S; s++) { mbar_init(&full[s], 1); mbar_init(&empt[s], 256); }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();

    if (tid >= 256) {
        // PRODUCER: stream A=SL + X + W into the ring (continuous gk across batches).
        if (tid == 256) {
            long gk = 0;
            int pe[HPF_S];
            #pragma unroll
            for (int s = 0; s < HPF_S; s++) pe[s] = 0;
            for (int ti = blockIdx.x; ti < ntiles; ti += gridDim.x) {
                int b = ti, mrow = b * Q;
                for (int kt = 0; kt < ntK; kt++) {
                    int buf = (int)(gk % HPF_S), k0 = kt * HP_BK;
                    if (gk >= HPF_S) { mbar_wait(&empt[buf], pe[buf]); pe[buf] ^= 1; }
                    mbar_expect_tx(&full[buf], txBytes);
                    #pragma unroll
                    for (int ms = 0; ms < HP_MSUB; ms++)
                        tma_load_2d(&As[buf * Asb + ms * TILE], &tmA, k0, mrow + ms * HP_BK, &full[buf]);
                    #pragma unroll
                    for (int n2 = 0; n2 < HP_NSUB; n2++)
                        tma_load_3d(&Xs[buf * Xsb + n2 * TILE], &tmX, n2 * HP_BK, k0, b, &full[buf]);  // X[D,F,B] OOB-fill 0
                    #pragma unroll
                    for (int n2 = 0; n2 < HP_NSUB; n2++)
                        tma_load_2d(&Ws[buf * Wsb + n2 * TILE], &tmW, n2 * HP_BK, k0, &full[buf]);     // W[D,F] OOB-fill 0
                    gk++;
                }
            }
        }
    } else {
        int wg = tid >> 7, warp = (tid >> 5) & 3, lane = tid & 31;
        long gk = 0;
        int pf[HPF_S];
        #pragma unroll
        for (int s = 0; s < HPF_S; s++) pf[s] = 0;
        for (int ti = blockIdx.x; ti < ntiles; ti += gridDim.x) {
            int b = ti;
            float acc[96];
            #pragma unroll
            for (int i = 0; i < 96; i++) acc[i] = 0.f;
            // PIPELINE: compute N one k-block AHEAD so the N-compute (CUDA cores) overlaps the wgmma (tensor).
            // Prologue computes N(kt=0); the loop issues wgmma(kt) [async] then computes N(kt+1) while it runs.
            #define POOL_COMPUTE_N(BUF, K0) do {                                                              \
                mbar_wait(&full[(BUF)], pf[(BUF)]); pf[(BUF)] ^= 1;                                            \
                wgmma_async_proxy_fence();                       /* TMA X/W -> generic reads */               \
                for (int k = tid; k < 64; k += 256) rsm[k] = ((K0) + k < F) ? R[(long)b * F + (K0) + k] : 0.f;\
                bar_sync_consumers();                                                                          \
                _Pragma("unroll") for (int C = tid; C < HP_NSUB * 512; C += 256) {  /* 512=64k*8chunks per brick */ \
                    int brick = C / 512, rem = C - brick * 512, kk = rem >> 3, mn8 = rem & 7;                  \
                    int so = brick * TILE + gh_sw_chunk(kk, mn8);  /* 16B chunk: 8 bf16, all share rsm[kk] */  \
                    uint4 xv = *(uint4*)&Xs[(BUF) * Xsb + so]; uint4 wv = *(uint4*)&Ws[(BUF) * Wsb + so];       \
                    float rr = rsm[kk];                                                                        \
                    __nv_bfloat162* xp = (__nv_bfloat162*)&xv; __nv_bfloat162* wp = (__nv_bfloat162*)&wv;       \
                    _Pragma("unroll") for (int j = 0; j < 4; j++) {                                            \
                        float2 fx = __bfloat1622float2(xp[j]), fw = __bfloat1622float2(wp[j]);                 \
                        xp[j] = __floats2bfloat162_rn(fx.x * rr * fw.x, fx.y * rr * fw.y);                     \
                    }                                                                                          \
                    *(uint4*)&Xs[(BUF) * Xsb + so] = xv;                                                       \
                }                                                                                              \
                bar_sync_consumers();                            /* N fully written */                        \
                wgmma_async_proxy_fence();                       /* generic N -> wgmma async-proxy */          \
            } while (0)
            POOL_COMPUTE_N((int)(gk % HPF_S), 0);                // prologue: N(kt=0)
            for (int kt = 0; kt < ntK; kt++) {
                int buf = (int)(gk % HPF_S);
                wgmma_fenceN(acc, 96);
                cute::warpgroup_arrive();
                #pragma unroll
                for (int ks = 0; ks < 4; ks++) {
                    uint64_t da = wg_desc_sw3(&As[buf * Asb + wg * TILE + ks * 16], 16, 1024);
                    uint64_t db = wg_desc_sw3(&Xs[buf * Xsb + ks * 1024], 8192, 1024);
                    wgmma_cute_m64n192k16(acc, da, db);
                }
                cute::warpgroup_commit_batch();
                if (kt + 1 < ntK)                                // compute N(kt+1) overlapping the async wgmma(kt)
                    POOL_COMPUTE_N((int)((gk + 1) % HPF_S), (kt + 1) * HP_BK);
                cute::warpgroup_wait<0>();
                wgmma_fenceN(acc, 96);
                mbar_arrive(&empt[buf]);
                gk++;
            }
            #undef POOL_COMPUTE_N
            // chunked O store: 3 column-chunks of 64, double-buffered ostg[2][128][64] (fits S=3 smem).
            #pragma unroll
            for (int c = 0; c < 3; c++) {
                int cb = c & 1;
                __nv_bfloat16* ch = ostg + cb * (HP_BM * 64);
                if (c >= 2 && tid == 0) tma_store_wait<1>();
                bar_sync_consumers();
                #pragma unroll
                for (int gg = 0; gg < 8; gg++) {
                    int g = c * 8 + gg;
                    int r0 = wg * 64 + 16 * warp + lane / 4, r1 = r0 + 8;
                    int c0 = 8 * gg + (lane % 4) * 2;
                    *(__nv_bfloat162*)&ch[r0 * 64 + c0] = __floats2bfloat162_rn(acc[4*g+0], acc[4*g+1]);
                    *(__nv_bfloat162*)&ch[r1 * 64 + c0] = __floats2bfloat162_rn(acc[4*g+2], acc[4*g+3]);
                }
                bar_sync_consumers();
                if (tid == 0) {
                    wgmma_async_proxy_fence();
                    tma_store_2d(&tmO, ch, c * 64, b * Q);   // chunk c at (col c*64, row b*Q); box {64,128}
                    tma_store_commit();
                }
            }
            if (tid == 0) tma_store_wait<0>();
        }
    }
    (void)D;
}

// Launcher for the hand pool: cached TMA maps on (SL,N,O) ptrs, smem attr once, persistent grid.
static int launch_pool_hand(const __nv_bfloat16* SL, const __nv_bfloat16* N, __nv_bfloat16* O,
                            int B, int Q, int D, int Fp, cudaStream_t stream) {
    static CUtensorMap s_tmPA, s_tmPB, s_tmPO;
    static const void* s_pSL = nullptr; static const void* s_pN = nullptr; static const void* s_pO = nullptr;
    if (SL != s_pSL || N != s_pN || O != s_pO) {
        // A = SL [B*Q, Fp] row-major (K-major operand): globalDim {Fp(fast), B*Q}, box {64,64}, SW128.
        cuuint64_t gdA[2] = {(cuuint64_t)Fp, (cuuint64_t)B * Q}, gsA[1] = {(cuuint64_t)Fp * 2};
        cuuint32_t bxA[2] = {64, 64}, es[2] = {1, 1};
        cuTensorMapEncodeTiled(&s_tmPA, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)SL, gdA, gsA, bxA, es,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        // B = N [B*Fp, D] row-major (MN-major operand): globalDim {D(fast), B*Fp}, box {64,64}, SW128.
        cuuint64_t gdB[2] = {(cuuint64_t)D, (cuuint64_t)B * Fp}, gsB[1] = {(cuuint64_t)D * 2};
        cuuint32_t bxB[2] = {64, 64};
        cuTensorMapEncodeTiled(&s_tmPB, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)N, gdB, gsB, bxB, es,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        // O = [B*Q, D] row-major store: globalDim {D(fast), B*Q}, box {HP_BN=192, HP_BM=128}, no swizzle.
        cuuint64_t gdO[2] = {(cuuint64_t)D, (cuuint64_t)B * Q}, gsO[1] = {(cuuint64_t)D * 2};
        cuuint32_t bxO[2] = {HP_BN, HP_BM};
        cuTensorMapEncodeTiled(&s_tmPO, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)O, gdO, gsO, bxO, es,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        s_pSL = SL; s_pN = N; s_pO = O;
    }
    const int TILE = HP_BK * HP_BK;
    size_t mb = (size_t)(HP_S * (HP_MSUB + HP_NSUB)) * TILE * 2 + (size_t)(2 * HP_S) * 8;
    mb = (mb + 127) & ~(size_t)127;
    size_t smemB = mb + (size_t)HP_BM * HP_BN * 2;          // + full-tile O staging [128x192]
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(k_pool_hand_persist, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemB); attr = true; }
    int numSM = 132; cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);
    k_pool_hand_persist<<<numSM, 384, smemB, stream>>>(s_tmPA, s_tmPB, s_tmPO, O, B, Q, D, Fp);
    return 0;
}

// Launcher for the FUSED hand pool (Phase 3b): N computed inline from X,rrms,W (no g_N). Maps: SL(K-major),
// X(3D [D,F,B] OOB-0), W(2D [D,F]), O. rrms=R[B,F] passed as a pointer (not a map).
static int launch_pool_hand_fused(const __nv_bfloat16* SL, const __nv_bfloat16* X, const __nv_bfloat16* W,
                                  __nv_bfloat16* O, const float* R, int B, int Q, int D, int F, int Fp,
                                  cudaStream_t stream) {
    static CUtensorMap f_tmA, f_tmX, f_tmW, f_tmO;
    static const void* f_pSL = nullptr; static const void* f_pX = nullptr; static const void* f_pW = nullptr; static const void* f_pO = nullptr;
    if (SL != f_pSL || X != f_pX || W != f_pW || O != f_pO) {
        cuuint32_t es2[2] = {1, 1}, es3[3] = {1, 1, 1};
        // A = SL [B*Q, Fp] K-major: {Fp(fast), B*Q}, box {64,64}, SW128.
        cuuint64_t gdA[2] = {(cuuint64_t)Fp, (cuuint64_t)B * Q}, gsA[1] = {(cuuint64_t)Fp * 2};
        cuuint32_t bxA[2] = {64, 64};
        cuTensorMapEncodeTiled(&f_tmA, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)SL, gdA, gsA, bxA, es2,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        // X = [B,F,D] viewed 3D {D(fast), F, B}: strides dim1(F)=D*2, dim2(B)=F*D*2 (bytes). box {64,64,1} SW128, OOB-fill 0.
        cuuint64_t gdX[3] = {(cuuint64_t)D, (cuuint64_t)F, (cuuint64_t)B};
        cuuint64_t gsX[2] = {(cuuint64_t)D * 2, (cuuint64_t)F * D * 2};
        cuuint32_t bxX[3] = {64, 64, 1};
        cuTensorMapEncodeTiled(&f_tmX, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, (void*)X, gdX, gsX, bxX, es3,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        // W = [F,D] viewed {D(fast), F}, box {64,64}, SW128 (shared across batches; OOB-fill 0 for f>=F).
        cuuint64_t gdW[2] = {(cuuint64_t)D, (cuuint64_t)F}, gsW[1] = {(cuuint64_t)D * 2};
        cuuint32_t bxW[2] = {64, 64};
        cuTensorMapEncodeTiled(&f_tmW, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)W, gdW, gsW, bxW, es2,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        // O = [B*Q, D] store, CHUNKED: {D(fast), B*Q}, box {64,128}, no swizzle (store 3 col-chunks of 64).
        cuuint64_t gdO[2] = {(cuuint64_t)D, (cuuint64_t)B * Q}, gsO[1] = {(cuuint64_t)D * 2};
        cuuint32_t bxO[2] = {64, (cuuint32_t)Q};   // store only this batch's Q rows (Q<HP_BM=128: wg1 computes the
                                                   // next batch's queries with wrong N -> must NOT be stored, else race)
        cuTensorMapEncodeTiled(&f_tmO, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)O, gdO, gsO, bxO, es2,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        f_pSL = SL; f_pX = X; f_pW = W; f_pO = O;
    }
    const int TILE = HP_BK * HP_BK;
    size_t mb = (size_t)(HPF_S * (HP_MSUB + 2 * HP_NSUB)) * TILE * 2 + (size_t)(2 * HPF_S) * 8 + 64 * 4;
    mb = (mb + 127) & ~(size_t)127;
    size_t smemB = mb + (size_t)2 * HP_BM * 64 * 2;        // + double-buffered chunked O staging [2][128][64]
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(k_pool_hand_fused, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemB); attr = true; }
    int numSM = 132; cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);
    k_pool_hand_fused<<<numSM, 384, smemB, stream>>>(f_tmA, f_tmX, f_tmW, f_tmO, O, R, B, Q, D, F, Fp);
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
                             const __nv_bfloat16* __restrict__ GAMMA, const __nv_bfloat16* __restrict__ BETA,
                             __nv_bfloat16* __restrict__ O, int B, int F, int Fp, int D, int Q) {
    int b = blockIdx.x, doff = blockIdx.y * DPB, tid = threadIdx.x, nt = blockDim.x, warp = tid >> 5, lane = tid & 31;
    int warp_q = warp / DGB, warp_d = warp % DGB, q0 = warp_q * WM, cbase = warp_d * DPW, qg = Q / WM;
    __shared__ float gsh[DPB], bsh[DPB], Mall[FPMAX], Rall[FPMAX];
    __shared__ __nv_bfloat16 Xsh[2][KT * DPB], SLsh[2][8 * (WM * KT)], Nsh[KT * NLDB];
    __shared__ float Otmp[MAXW * (WM * WN)];
    for (int d = tid; d < DPB; d += nt) { gsh[d] = b2f(GAMMA[doff + d]); bsh[d] = b2f(BETA[doff + d]); }
    for (int f = tid; f < Fp; f += nt) { Mall[f] = (f < F) ? b2f(M[(long)f * B + b]) : 0.f; Rall[f] = (f < F) ? R[(long)b * F + f] : 0.f; }
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
            Nsh[fl * NLDB + dl] = (f < F) ? __float2bfloat16((b2f(xs[i]) - Mall[f]) * Rall[f] * gsh[dl] + bsh[dl]) : __float2bfloat16(0.f); }
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
                     const __nv_bfloat16* __restrict__ GAMMA, const __nv_bfloat16* __restrict__ BETA,
                     __nv_bfloat16* __restrict__ O, int B, int F, int Fp, int D, int Q) {
    int b = blockIdx.x, doff = blockIdx.y * WS_DPB, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    bool isProd = warp >= WS_NCONS;             // warps 0-7 = consumers, 8-15 = producers
    int pw = warp - WS_NCONS, cw = warp;        // producer idx / consumer idx (= q-tile)
    __shared__ float gsh[WS_DPB], bsh[WS_DPB], Mall[FPMAX], Rall[FPMAX];
    __shared__ __nv_bfloat16 Xsh[2][WS_KT * WS_DPB], Nsh[2][WS_KT * WS_NLDB];
    __shared__ float Otmp[WS_NCONS * (WM * WN)];
    for (int d = tid; d < WS_DPB; d += 512) { gsh[d] = b2f(GAMMA[doff + d]); bsh[d] = b2f(BETA[doff + d]); }
    for (int f = tid; f < Fp; f += 512) { Mall[f] = (f < F) ? b2f(M[(long)f * B + b]) : 0.f; Rall[f] = (f < F) ? R[(long)b * F + f] : 0.f; }
    __syncthreads();                            // Mall/Rall/gsh/bsh ready for all warps
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
                if (f < F) { float x = b2f(xs[i]); v = (x - Mall[f]) * Rall[f] * gsh[dl] + bsh[dl]; }
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
                        const __nv_bfloat16* __restrict__ GAMMA, const __nv_bfloat16* __restrict__ BETA,
                        __nv_bfloat16* __restrict__ O, int B, int F, int Fp, int D, int Q) {
    int b = blockIdx.x, q0 = blockIdx.y * 64, nt = blockIdx.z, dbase = nt * 64;  // this block's n64 tile
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __shared__ __nv_bfloat16 As[64 * WG_KT];         // A: K-major bricks (g_L for this q-tile)
    __shared__ __nv_bfloat16 Bs[WG_KT * 64];         // B: MN-major bricks (N for THIS n64 tile only)
    __shared__ __nv_bfloat16 Xsh[2][WG_KT * 64];     // double-buffer X (64 D-cols of this n-tile)
    __shared__ float Mtile[WG_KT], Rtile[WG_KT];     // M,R for the CURRENT K-tile only (was full Fp -> 12KB)
    __shared__ float gsh[64], bsh[64];               // gamma, beta for this n-tile's 64 cols
    float acc[32];                                    // ONE m64n64 tile = 32 fp32 regs (the occupancy win)
    #pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.f;

    for (int d = tid; d < 64; d += 128) {
        gsh[d] = b2f(GAMMA[dbase + d]);
        bsh[d] = b2f(BETA[dbase + d]);
    }
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
        // Load this K-tile's 64 M/R values (M col-major M[f*B+b], R row-major R[b*F+f])
        for (int fl = tid; fl < WG_KT; fl += 128) {
            int f = k0 + fl;
            Mtile[fl] = (f < F) ? b2f(M[(long)f * B + b]) : 0.f;
            Rtile[fl] = (f < F) ? R[(long)b * F + f] : 0.f;
        }
        __pipeline_wait_prior(t + 1 < ntilesK ? 1 : 0);
        __syncthreads();

        // Compute N from staged X (this n-tile's 64 D-cols), write MN-major B bricks
        const __nv_bfloat16* xs = Xsh[buf];
        for (int i = tid; i < WG_KT * 64; i += 128) {
            int fl = i / 64, dl = i % 64, f = k0 + fl;
            float val = 0.f;
            if (f < F) {
                float x = b2f(xs[fl * 64 + dl]);
                val = (x - Mtile[fl]) * Rtile[fl] * gsh[dl] + bsh[dl];
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

// ================= Direction B: PRODUCER/CONSUMER WARP-SPECIALIZED fused wgmma pool =================
// Goal (NCU-grounded): the acc[32] fused pool is 4.4ms iso @ only 34% DRAM / 30% warps_active — it is
// latency/occupancy-bound, and its grid (B,Q/64,D/64) reads A=L 3x (D-split) + X 2x = 3.5GB.  Here:
//   * grid (B, Q/64): one CTA per (batch, q64-tile), FULL D=192 per CTA  -> A=L read ONCE per (b,q) (no 3x).
//   * 2 warpgroups: CONSUMER (tid 0..127) issues wgmma.mma_async (acc[96]=3 n64 tiles, full D); PRODUCER
//     (tid 128..255) does cp.async X (double-buffered), computes N=LayerNorm(X) inline -> Bs bricks, and
//     stages A=sigmoid(L) -> As bricks.  Multi-stage (WSB_S) smem pipeline so the producer runs AHEAD and
//     the wgmma overlaps the next tile's loads+N-compute (the win mma.sync/single-WG can't get).
//   * handoff via PTX named barriers (256-count 2-party handshake): full[s]=N/A ready, empty[s]=buffer free.
//   * generic-proxy stores (N compute + A sigmoid) -> wgmma async-proxy reads need fence.proxy.async.shared.
// Reuses the VERIFIED machinery: wg_desc, As/Bs brick layouts, cute m64n64 atom, acc->O epilogue map.
constexpr int WSB_KT = 64;        // K-tile rows (4 k16 wgmma steps)
constexpr int WSB_S  = 2;         // pipeline stages (As/Bs multi-buffered)
// (mbar_init/arrive/wait + wgmma_fenceN moved earlier — before the hand gate — so both kernels use them.)
// D-SPLIT acc[32] warp-spec (BEST config, 5.19ms): grid (B, Q/64, D/64). One CTA = one (batch, q64-tile,
// n64-tile). Consumer holds ONE m64n64 acc[32] (proven config) -> small smem -> 4 blocks/SM (the occupancy
// that matters for this latency/barrier-bound pool). Producer reads X DIRECTLY from global as vectorized 16B
// (no cp.async — that raced with mbarrier + cost a smem round-trip), computes N inline -> Bs bricks, stages
// A=g_L (already sigmoid'd) -> As bricks. mbarrier handoff (phase parity + rel/acq + cross-proxy fence).
// VERDICT: correct + best-tuned, but 5.19ms > 4.03ms default (consumer idles ~half the time => warp-spec is
// structurally worse than cooperative here). Kept as the documented branch-B result. See DIRECTION.md/LEDGER.
__global__ __launch_bounds__(256, 3)
void k_pool_ws(const __nv_bfloat16* __restrict__ g_L,   // gate output [B,Q,F] (already sigmoid'd by gate epilogue)
               const __nv_bfloat16* __restrict__ X,     // [B,F,D]
               const __nv_bfloat16* __restrict__ M, const float* __restrict__ R,
               const __nv_bfloat16* __restrict__ GAMMA, const __nv_bfloat16* __restrict__ BETA,
               __nv_bfloat16* __restrict__ O, int B, int F, int Fp, int D, int Q) {
    const int b = blockIdx.x, q0 = blockIdx.y * 64, dbase = blockIdx.z * 64;  // this CTA's n64 tile
    const int tid = threadIdx.x, lane = tid & 31;
    const bool isCons = tid < 128;
    const int ntilesK = Fp / WSB_KT;
    const int kbStride = (64 / 8) * 64;    // =512: MN-major B-brick stride for a 64-wide tile

    // ---- dynamic smem partition (single n64 tile of B) ----
    extern __shared__ char smem[];
    uint64_t* full = (uint64_t*)smem;      // [WSB_S]
    uint64_t* empt = full + WSB_S;         // [WSB_S]
    float* Mall = (float*)(empt + WSB_S);  // [Fp]
    float* Rall = Mall + Fp;               // [Fp]
    float* gsh  = Rall + Fp;               // [64]
    float* bsh  = gsh + 64;                // [64]
    __nv_bfloat16* As  = (__nv_bfloat16*)(bsh + 64);  // [WSB_S][64*WSB_KT]
    __nv_bfloat16* Bs  = As + WSB_S * 64 * WSB_KT;    // [WSB_S][WSB_KT*64]
    const int Astage = 64 * WSB_KT, Bstage = WSB_KT * 64;

    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < WSB_S; s++) { mbar_init(&full[s], 128); mbar_init(&empt[s], 128); }
    }
    for (int d = tid; d < 64; d += 256) { gsh[d] = b2f(GAMMA[dbase + d]); bsh[d] = b2f(BETA[dbase + d]); }
    for (int f = tid; f < Fp; f += 256) { Mall[f] = (f < F) ? b2f(M[(long)f * B + b]) : 0.f;
                                          Rall[f] = (f < F) ? R[(long)b * F + f] : 0.f; }
    __syncthreads();

    if (!isCons) {
        // ============================ PRODUCER warpgroup ============================
        const int ptid = tid - 128;
        int pe[WSB_S];
        #pragma unroll
        for (int s = 0; s < WSB_S; s++) pe[s] = 0;
        for (int k = 0; k < ntilesK; k++) {
            int pbuf = k % WSB_S, g0 = k * WSB_KT;
            if (k >= WSB_S) { mbar_wait(&empt[pbuf], pe[pbuf]); pe[pbuf] ^= 1; }
            // --- N(k) for this n-tile's 64 cols, X read DIRECTLY from global as VECTORIZED 16B (bf162x4) ---
            __nv_bfloat162* Bs2 = (__nv_bfloat162*)(Bs + pbuf * Bstage);
            for (int i = ptid; i < (WSB_KT * 64) / 8; i += 128) {
                int fl = (i * 8) >> 6, dl0 = (i * 8) & 63, f = g0 + fl;
                bool ok = (f < F);
                float mf = ok ? Mall[f] : 0.f, rf = ok ? Rall[f] : 0.f;
                const __nv_bfloat162* xp = (const __nv_bfloat162*)&X[((long)b * F + f) * D + dbase + dl0];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int dl = dl0 + 2 * j;
                    __nv_bfloat162 xv = ok ? xp[j] : __halves2bfloat162(__float2bfloat16(0.f), __float2bfloat16(0.f));
                    float v0 = ok ? (b2f(xv.x) - mf) * rf * gsh[dl]     + bsh[dl]     : 0.f;
                    float v1 = ok ? (b2f(xv.y) - mf) * rf * gsh[dl + 1] + bsh[dl + 1] : 0.f;
                    int off = (fl / 8) * kbStride + (dl / 8) * 64 + (fl % 8) * 8 + (dl % 8);
                    Bs2[off >> 1] = __halves2bfloat162(__float2bfloat16(v0), __float2bfloat16(v1));
                }
            }
            // --- stage A(k) -> As[pbuf] (K-major bricks). g_L already sigmoid'd -> copy. Mask f>=F -> 0. ---
            for (int i = ptid; i < (64 * WSB_KT) / 8; i += 128) {
                int q = (i * 8) / WSB_KT, fl = (i * 8) % WSB_KT, f = g0 + fl;
                int off = (fl / 16) * 1024 + (q / 8) * 128 + ((fl % 16) / 8) * 64 + (q % 8) * 8;
                const __nv_bfloat16* src = g_L + ((long)(b * Q + q0 + q)) * F + f;
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    As[pbuf * Astage + off + j] = (f + j < F) ? src[j] : __float2bfloat16(0.f);
            }
            wgmma_async_proxy_fence();
            mbar_arrive(&full[pbuf]);
        }
    } else {
        // ============================ CONSUMER warpgroup ============================
        const int warp = tid >> 5;
        float acc[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) acc[i] = 0.f;
        int pf[WSB_S];
        #pragma unroll
        for (int s = 0; s < WSB_S; s++) pf[s] = 0;
        // (Tested wgmma.wait_group<1> software-pipelining here — CORRECT but 5.33ms > 5.19 wait<0>: the
        //  consumer idles on the producer-bound pipeline, so pipelining the consumer's wgmma doesn't help.
        //  Reverted to wait<0>. See LEDGER 2026-06-08T14:55Z.)
        for (int k = 0; k < ntilesK; k++) {
            int pbuf = k % WSB_S;
            mbar_wait(&full[pbuf], pf[pbuf]); pf[pbuf] ^= 1;
            wgmma_async_proxy_fence();
            wgmma_cute_fence32(acc);
            cute::warpgroup_arrive();
            #pragma unroll
            for (int ks = 0; ks < WSB_KT / 16; ks++) {
                int kk = ks * 16;
                uint64_t da = wg_desc(&As[pbuf * Astage + (kk / 16) * 1024], 128, 256);
                uint64_t db = wg_desc(&Bs[pbuf * Bstage + (kk / 8) * kbStride], (uint32_t)(64 * 16), 128);
                wgmma_cute_m64n64k16(acc, da, db);
            }
            cute::warpgroup_commit_batch();
            cute::warpgroup_wait<0>();
            wgmma_cute_fence32(acc);
            mbar_arrive(&empt[pbuf]);
        }
        // ---- epilogue: acc[32] -> O[b, q0:q0+64, dbase:dbase+64] ----
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
                const __nv_bfloat16* __restrict__ GAMMA, const __nv_bfloat16* __restrict__ BETA,
                __nv_bfloat16* __restrict__ O, int B, int F, int Fp, int D, int Q) {
    int b = blockIdx.x, doff = blockIdx.y * MPF_DPB, qoff = blockIdx.z * MPF_QPB;
    int tid = threadIdx.x, nt = blockDim.x, warp = tid >> 5, lane = tid & 31;
    // 8 warps: 4 q-tiles (16 rows each) × 2 d-groups (3 n16-cols each). Each warp does 1 q-tile now.
    // warp_d=warp%2, warp_q=warp/2 (0..3). acc[1][3] → 24 regs (vs 48 before).
    int warp_d = warp % 2, warp_q = warp / 2, cbase = warp_d * 3;
    __shared__ float gsh[MPF_DPB], bsh[MPF_DPB], Mall[FPMAX], Rall[FPMAX];
    __shared__ __nv_bfloat16 Xsh[2][MPF_KT * MPF_DPB], Ash[2][MPF_QPB * MPF_KT], Nsh[MPF_KT * MPF_NLDB];
    __shared__ float Otmp[8 * (16 * 16)];
    for (int d = tid; d < MPF_DPB; d += nt) { gsh[d] = b2f(GAMMA[doff + d]); bsh[d] = b2f(BETA[doff + d]); }
    for (int f = tid; f < Fp; f += nt) { Mall[f] = (f < F) ? b2f(M[(long)f * B + b]) : 0.f; Rall[f] = (f < F) ? R[(long)b * F + f] : 0.f; }
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
        // Compute N from X (this K-tile) while the NEXT A is being loaded in the background
        for (int i = tid; i < MPF_KT * MPF_DPB; i += nt) { int fl = i / MPF_DPB, dl = i - fl * MPF_DPB; int f = g0 + fl;
            Nsh[fl * MPF_NLDB + dl] = (f < F) ? __float2bfloat16((b2f(xs[i]) - Mall[f]) * Rall[f] * gsh[dl] + bsh[dl]) : __float2bfloat16(0.f); }
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
                       const __nv_bfloat16* __restrict__ GAMMA, const __nv_bfloat16* __restrict__ BETA,
                       __nv_bfloat16* __restrict__ O, int B, int F, int D, int Q) {
    __shared__ float Ls[PQ][PG];
    __shared__ float Ms[PG];
    __shared__ float Rs[PG];
    int b = blockIdx.y, q0 = blockIdx.x * PQ;
    int d = threadIdx.x;                       // one channel per thread
    float acc[PQ];
    #pragma unroll
    for (int i = 0; i < PQ; i++) acc[i] = 0.f;
    float gamma = (d < D) ? b2f(GAMMA[d]) : 0.f;
    float beta  = (d < D) ? b2f(BETA[d]) : 0.f;

    for (int g0 = 0; g0 < F; g0 += PG) {
        for (int idx = threadIdx.x; idx < PQ * PG; idx += blockDim.x) {
            int i = idx / PG, j = idx % PG; int qq = q0 + i, gg = g0 + j;
            // L holds raw gate logits (CUTLASS gate); sigmoid here (free on the load)
            Ls[i][j] = (qq < Q && gg < F) ? 1.f / (1.f + expf(-b2f(L[((long)b * Q + qq) * F + gg]))) : 0.f;
        }
        for (int idx = threadIdx.x; idx < PG; idx += blockDim.x) {
            int gg = g0 + idx;
            Ms[idx] = (gg < F) ? b2f(M[(long)b * F + gg]) : 0.f;
            Rs[idx] = (gg < F) ? R[(long)b * F + gg] : 0.f;
        }
        __syncthreads();
        if (d < D) {
            for (int j = 0; j < PG; j++) {
                int gg = g0 + j;
                float nv = (gg < F) ? (b2f(X[((long)b * F + gg) * D + d]) - Ms[j]) * Rs[j] * gamma + beta : 0.f;
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
__nv_bfloat16 *g_Ppad = nullptr;         // Phase 2: P repacked per-q-padded [F, Q*Fp] (one-time, cached on P ptr)
const void *g_Ppad_src = nullptr;        // the P ptr g_Ppad was built from
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
    int Fp = (F + 255) / 256 * 256;           // pad features to a multiple of 256: pool K-tile is 64 AND the
                                              // padded gate's 256-wide N-tiles must align to q (Fp%256==0 ->
                                              // no q-boundary crossing). Big F=1497 -> 1536 (unchanged).
    long QFp = (long)Q * Fp;
    if (B != g_B || F != g_F || D != g_D || Q != g_Q) {
        if (g_M) cudaFree(g_M); if (g_R) cudaFree(g_R); if (g_SL) cudaFree(g_SL); if (g_Ppad) cudaFree(g_Ppad);
        if (cudaMalloc(&g_M, (size_t)B * F * sizeof(__nv_bfloat16)) != cudaSuccess) return 11;
        if (cudaMalloc(&g_R, (size_t)B * F * 4) != cudaSuccess) return 11;
        if (cudaMalloc(&g_SL, (size_t)B * Q * Fp * sizeof(__nv_bfloat16)) != cudaSuccess) return 11;
        if (cudaMalloc(&g_Ppad, (size_t)F * QFp * sizeof(__nv_bfloat16)) != cudaSuccess) return 11;
        cudaMemset(g_SL, 0, (size_t)B * Q * Fp * sizeof(__nv_bfloat16));  // pad cols [F,Fp) stay 0
        cudaMemset(g_Ppad, 0, (size_t)F * QFp * sizeof(__nv_bfloat16));   // Pp pad cols stay 0
        g_B = B; g_F = F; g_D = D; g_Q = Q;
    }
    // RMSNorm: 3 inputs X[B,F,D], P[F,Q*F], W[F,D] (per-(feature,channel) weight). No beta. GAMMA carries W
    // into the stats kernels (they now do N=X*rrms*W, no centering/bias); BETA aliased to inputs[2] (unread).
    const __nv_bfloat16 *X = inputs[0], *P = inputs[1], *GAMMA = inputs[2], *BETA = inputs[2];
    __nv_bfloat16* O = outputs[0];

    // ===== GIST forward — fully fused 3-kernel pipeline (the default; no env selectors). =====
    // stats(M,R) -> gate(sigma(M.Pp) -> padded g_SL) -> pool(O = SL . N, N=X*rrms*W inline).
    // The repad / separate-sigmoid / materialized-N kernels are all deleted. Only GIST_SKIP_GATE /
    // GIST_SKIP_POOL remain (component differential timing); everything else is unconditional.
    long mr_warps  = ((long)B * Fp + (32 / STATS_SG) - 1) / (32 / STATS_SG);
    long mr_blocks = (mr_warps + STATS_WPB - 1) / STATS_WPB;

    // (1) stats: M = mean_d X, R = rrms (vectorized int4). N is NOT materialized (pool computes it inline).
    k_stats_mr<<<mr_blocks, STATS_WPB * 32, 0, stream>>>(X, g_M, g_R, B, F, Fp, D, 1e-5f);

    // TMA tensor maps (cached on the g_M / P pointers, stable across the L2-flush timed loop):
    //   A = g_M [B(m,fast), F(k)] col-major;  Bp = g_Ppad [QFp(n,fast), F(k)] row-major;  SL store = g_SL chunk.
    static CUtensorMap s_tmA, s_tmBp, s_tmSL; static const void* s_pM = nullptr; static const void* s_pP = nullptr;
    if (g_M != s_pM || (const void*)P != s_pP) {
        cuuint64_t gdA[2] = {(cuuint64_t)B, (cuuint64_t)F}, gsA[1] = {(cuuint64_t)B * 2};
        cuuint32_t bxA[2] = {64, 64}, es2[2] = {1, 1};
        cuTensorMapEncodeTiled(&s_tmA, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)g_M, gdA, gsA, bxA, es2,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        cuuint64_t gdBp[2] = {(cuuint64_t)QFp, (cuuint64_t)F}, gsBp[1] = {(cuuint64_t)QFp * 2};
        cuuint32_t bxBp[2] = {64, 64};
        cuTensorMapEncodeTiled(&s_tmBp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)g_Ppad, gdBp, gsBp, bxBp, es2,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        cuuint64_t gdSL[2] = {(cuuint64_t)QFp, (cuuint64_t)B}, gsSL[1] = {(cuuint64_t)QFp * 2};
        cuuint32_t bxSL[2] = {HGP_CW, HG_BM};
        cuTensorMapEncodeTiled(&s_tmSL, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)g_SL, gdSL, gsSL, bxSL, es2,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        s_pM = g_M; s_pP = (const void*)P;
    }

    // (0) one-time weight prepack P[F,QF] -> Pp[F,Q*Fp] (cached on the P pointer; the harness holds weights
    //     constant via CUDA_EXEC_PARAM_HARNESS_WEIGHT_INPUTS, so this runs once and is amortized to ~0).
    static const void* s_ppsrc = nullptr;
    if ((const void*)P != s_ppsrc) {
        k_ppad<<<(unsigned)((long)F * Q), 256, 0, stream>>>(P, g_Ppad, F, Q, QF, QFp, Fp);
        s_ppsrc = (const void*)P;
    }

    // (2) gate: SL = sigma(M . Pp) -> g_SL[B,Q,Fp] directly. sigma fused (tanh.approx, overlaps the chunked
    //     store); padded write needs NO remap (Fp=1536 | 256) and NO pad-mask (pool's N[g>=F]=0 makes SL[pad]
    //     a don't-care, so SL[pad]*N[pad]=0). Persistent warp-specialized WGMMA.
    size_t mb = (size_t)(HGP_S * (HG_MSUB + HG_NSUB)) * 64 * 64 * 2 + (size_t)(2 * HGP_S) * 8;
    mb = (mb + 127) & ~(size_t)127;
    size_t hgsmem_p = mb + (size_t)2 * HG_BM * HGP_CW * 2;   // ring + double-buffered chunk staging
    int numSM = 132; cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);
    // gate (always run; the GIST_SKIP_GATE/SKIP_POOL guards are stripped from the shipped kernel — a
    // skip-gate env at measure-time would fake ~1.23ms, so they must not exist in the deliverable).
    static bool ap = false;
    if (!ap) { cudaFuncSetAttribute(k_gate_hand_persist, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)hgsmem_p); ap = true; }
    k_gate_hand_persist<<<numSM, 384, hgsmem_p, stream>>>(s_tmA, s_tmBp, s_tmSL, g_SL, B, F, QFp, /*dosig=*/1, /*padarg=*/-Fp);

    // (3) pool: O = SL . N, with N = X * rrms * W computed INLINE (N never materialized).
    return launch_pool_hand_fused(g_SL, X, GAMMA, O, g_R, B, Q, D, F, Fp, stream);
}
