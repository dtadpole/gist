// =============================================================================
// GIST gate GEMM — hand-written CUDA + inline PTX (NO CUTLASS).  PHASE 1: bare
// GEMM, raw logits, NO sigmoid.   L = M @ P   (L = gate logits G in the notation)
//
//   M : x_input_mean   [B, F]   column-major (B-contiguous)   <- A operand (trans-a=1, MN-major)
//   P : proj_params    [F, QF]  row-major   (QF-contiguous)   <- B operand (trans-b=1, MN-major)
//   L : gate logits    [B, QF]  row-major,  QF = Q*F          <- output (raw, no sigma)
//
// Design shape: B=1536, F=1491, Fp=1536 (pad F to x64), D=192, Q=128, bf16.
// As a GEMM:  M_dim = B = 1536,  N_dim = QF = 190848,  K_dim = F (Kpad = Fp = 1536).
// This is the skinny-K problem (huge N, modest K).
//
// MEASURED (GPU2, design shape, bf16, L2-flush harness):
//   warm (SKIP_GATE differential) = 1.2745 ms   == CUTLASS gate_ns ~1.26 ms
//   NCU cold = 1.66 ms / Tensor 77.6%  ==  CUTLASS gate_ns 1.66 ms / 77.8%   (PARITY)
//
// This file is a READABLE EXTRACT of the kernel that ships behind env GIST_GATE_WS
// in ~/gist-opt/dir-c-multiwg/gist.cu (full authoritative copy: gist_full_snapshot.cu).
// It is the FUSE=false (no-sigmoid) instantiation, inlined for reference.
//
// Architecture (matches CUTLASS KernelTmaWarpSpecializedCooperative, MMA_64x256x16):
//   TileShape 128(M) x 256(N) x 64(K), NS=4-stage async pipeline, 3 warpgroups
//   (1 producer issues TMA loads, 2 consumers run pure wgmma cooperatively on the
//   128-M tile, 64 rows each), ClusterShape <1,1,1> (NO multicast).
//   Build: nvcc -gencode arch=compute_90a,code=sm_90a   (plain -arch=sm_90a
//   silently downgrades -> "wgmma not supported on sm_90").
// =============================================================================
#include <cuda.h>            // CUtensorMap + cuTensorMapEncodeTiled (TMA)
#include <cuda_bf16.h>

// ---- Hopper WGMMA shared-memory matrix descriptor (cute bit layout) ----------
__device__ __forceinline__ uint64_t wg_encode(uint32_t x) { return (uint64_t)((x & 0x3FFFF) >> 4); }
// Swizzled B128 descriptor: layout_type=1 (bit 62). Pair with the cute canonical
// B128 smem layout that TMA(SWIZZLE_128B) produces (validated /tmp/swz256.cu: max_err 0).
__device__ __forceinline__ uint64_t wg_desc_sw(const void* smem_ptr, uint32_t LBO_bytes, uint32_t SBO_bytes) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t d = 0;
    d |= wg_encode(a);
    d |= wg_encode(LBO_bytes) << 16;
    d |= wg_encode(SBO_bytes) << 32;
    d |= (uint64_t)1 << 62;            // layout_type = B128 (128-byte swizzle)
    return d;
}

// ---- single wgmma.m64n256k16 (CUTLASS MMA_64x256 equivalent) ------------------
// D[64,256] += A[64,16] * B[16,256], fp32 acc d[128]. trans-a=1, trans-b=1.
// Reads A ONCE and B once (4x m64n64k16 would re-read A 4x -> +37% smem traffic).
// B is the 4 contiguous 64x64 canonical-B128 bricks; the descriptor auto-walks
// N=256 across them via LBO=8192 (brick stride 4096 elem * 2 bytes), SBO=1024.
__device__ __forceinline__ void wgmma_m64n256k16(float* d, uint64_t da, uint64_t db, int scaleD) {
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
        " %128, %129, p, %131, %132, %133, %134;\n}\n"
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
__device__ __forceinline__ void wgmma_commit_group() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
// wait_group N, binding the 32 fp32 accumulators "+f" so the compiler keeps them live across the
// async completion (call per acc[32] chunk; extra calls after one commit are no-ops).
__device__ __forceinline__ void wgmma_wait_g1_32(float* d) {  // keep 1 group in flight (pipeline)
    asm volatile("wgmma.wait_group.sync.aligned 1;\n"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
          "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
          "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
          "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]) :: "memory");
}
__device__ __forceinline__ void wgmma_wait_g0_32(float* d) {  // full drain
    asm volatile("wgmma.wait_group.sync.aligned 0;\n"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
          "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
          "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
          "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]) :: "memory");
}

// ---- tile shape: matches CUTLASS 128x256x64, NS=4 -----------------------------
constexpr int WS_BM = 128, WS_BN = 256, WS_BK = 64, WS_NS = 4, WS_NH = 2, WS_NSUB = WS_BN / 64;

// =============================================================================
//  k_gate_ws  —  warp-specialized cooperative TMA wgmma gate (bare, no sigmoid)
//  grid = ( ceil(B/128), ceil(QF/256) ),  block = 384 threads (3 warpgroups).
// =============================================================================
__global__ __launch_bounds__(384, 1)
void k_gate_ws(const __grid_constant__ CUtensorMap tmA,    // A = M [B,F] col-major, box{64,64} SW128
               const __grid_constant__ CUtensorMap tmB,    // B = P [F,QF] row-major, box{64,64} SW128
               __nv_bfloat16* __restrict__ L, int B, int F, int Fp, long QF) {
    int m0 = blockIdx.x * WS_BM; long n0 = (long)blockIdx.y * WS_BN;
    int tid = threadIdx.x, warp = tid >> 5, wg = warp >> 2, lane = tid & 31, wgwarp = warp & 3;
    const int ASZ = 64 * 64, kaStride = (64 / 8) * 64;     // 4096 elem/brick, 512 elem per-k16 advance
    extern __shared__ char sm[];
    __nv_bfloat16* As = (__nv_bfloat16*)(((uintptr_t)sm + 127) & ~(uintptr_t)127);    // [NS][NH bricks]
    __nv_bfloat16* Bs = As + WS_NS * WS_NH * ASZ;                                     // [NS][NSUB bricks]
    uint64_t* full  = (uint64_t*)(((uintptr_t)(Bs + WS_NS * WS_NSUB * ASZ) + 127) & ~(uintptr_t)127);
    uint64_t* empty = full + WS_NS;
    if (tid < WS_NS) {                                       // mbarriers: full(count 1 producer), empty(count 2 consumers)
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"((uint32_t)__cvta_generic_to_shared(&full[tid])));
        asm volatile("mbarrier.init.shared.b64 [%0], 2;" :: "r"((uint32_t)__cvta_generic_to_shared(&empty[tid])));
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    int ntilesK = Fp / WS_BK;                                // K-tiles over the F (gate) reduction = Phi[tau]

    if (wg == 2) {                                           // ===== PRODUCER warpgroup (TMA) =====
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;\n" ::: "memory");   // free regs for consumers
        int ptid = tid - 256;
        for (int k = 0; k < ntilesK; k++) {
            int s = k % WS_NS, k0 = k * WS_BK;
            if (ptid == 0) {
                uint32_t me = (uint32_t)__cvta_generic_to_shared(&empty[s]);
                if (k >= WS_NS) {                            // wait until consumers freed buffer s
                    unsigned par = (unsigned)(((k / WS_NS) - 1) & 1);
                    asm volatile("{\n.reg .pred p;\nLE: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n@!p bra LE;\n}\n" :: "r"(me), "r"(par));
                }
                uint32_t mf = (uint32_t)__cvta_generic_to_shared(&full[s]);
                asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(mf), "r"((unsigned)((WS_NH + WS_NSUB) * ASZ * 2)));
                #pragma unroll                                // 2 A-bricks (128 M x 64 K)
                for (int h = 0; h < WS_NH; h++)
                    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                        :: "r"((uint32_t)__cvta_generic_to_shared(&As[s * WS_NH * ASZ + h * ASZ])), "l"(&tmA), "r"(m0 + h * 64), "r"(k0), "r"(mf) : "memory");
                #pragma unroll                                // 4 B-bricks (64 K x 256 N) -> auto OOB-zero of K/N tails
                for (int n2 = 0; n2 < WS_NSUB; n2++)
                    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                        :: "r"((uint32_t)__cvta_generic_to_shared(&Bs[s * WS_NSUB * ASZ + n2 * ASZ])), "l"(&tmB), "r"((int)(n0 + n2 * 64)), "r"(k0), "r"(mf) : "memory");
            }
        }
    } else {                                                 // ===== CONSUMER warpgroups (wg 0,1) =====
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");  // take producer's regs (wgmma pipeline room)
        int mh = m0 + wg * 64;                               // this wg owns 64 M-rows (cooperative split of 128)
        float acc[128];                                      // m64n256 accumulator
        #pragma unroll
        for (int i = 0; i < 128; i++) acc[i] = 0.f;
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");   // once (acc never written by non-wgmma after init)
        for (int k = 0; k < ntilesK; k++) {                  // Phi[tau]: stream K-tiles, NS-stage pipeline
            int s = k % WS_NS; unsigned par = (unsigned)((k / WS_NS) & 1);
            uint32_t mf = (uint32_t)__cvta_generic_to_shared(&full[s]);
            asm volatile("{\n.reg .pred p;\nLF: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n@!p bra LF;\n}\n" :: "r"(mf), "r"(par));
            #pragma unroll
            for (int ks = 0; ks < WS_BK / 16; ks++) {        // 4 k16 wgmmas per K-tile (BK=64)
                int kk = ks * 16;
                uint64_t da = wg_desc_sw(&As[s * WS_NH * ASZ + wg * ASZ + (kk / 8) * kaStride], 128, 1024);
                uint64_t db = wg_desc_sw(&Bs[s * WS_NSUB * ASZ + (kk / 8) * kaStride], 8192, 1024);  // LBO=8192 walks N=256
                wgmma_m64n256k16(acc, da, db, 1);
            }
            wgmma_commit_group();
            #pragma unroll
            for (int n2 = 0; n2 < WS_NSUB; n2++) wgmma_wait_g1_32(&acc[n2 * 32]);  // keep wgmma(k) in flight; drain k-1
            if (k >= 1 && wgwarp == 0 && lane == 0)          // free buffer (k-1) for the producer
                asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&empty[(k - 1) % WS_NS])));
        }
        #pragma unroll
        for (int n2 = 0; n2 < WS_NSUB; n2++) wgmma_wait_g0_32(&acc[n2 * 32]);      // drain last wgmma

        // ---- coalesced epilogue: stage acc -> smem (conflict-free, row stride 264) then 16B (uint4)
        // coalesced global stores. Scalar 2B stores were 41% of stall samples (lg_throttle) and pinned
        // tensor to 47.7%; this recovers CUTLASS's 77.6% (its STSM+TMA-store epilogue, same traffic shape).
        const int SST = 264;                                 // 256 + 8 pad: bank = const + lane (no conflict), keeps 16B align
        __nv_bfloat16* St = As + wg * (64 * SST);            // reuse mainloop smem (free post-drain), disjoint per wg
        int lrow = 16 * wgwarp + lane / 4;
        #pragma unroll
        for (int n2 = 0; n2 < WS_NSUB; n2++) {
            float* d = &acc[n2 * 32];
            #pragma unroll
            for (int a = 0; a < 8; a++) {
                int col0 = n2 * 64 + 8 * a + (lane % 4) * 2;
                *(__nv_bfloat162*)&St[lrow * SST + col0]       = __floats2bfloat162_rn(d[4 * a + 0], d[4 * a + 1]);
                *(__nv_bfloat162*)&St[(lrow + 8) * SST + col0] = __floats2bfloat162_rn(d[4 * a + 2], d[4 * a + 3]);
            }
        }
        asm volatile("bar.sync %0, %1;" :: "r"(wg + 1), "r"(128));   // warpgroup barrier (producer not involved)
        int twg = tid & 127; long mbase = (long)mh;
        #pragma unroll
        for (int c = twg; c < 64 * 256 / 8; c += 128) {      // 2048 uint4 chunks (8 bf16 each), 16/thread
            int row = c >> 5, colb = (c & 31) * 8;
            long gcol = n0 + colb;
            if (mbase + row < B && gcol < QF)
                *(uint4*)&L[(mbase + row) * QF + gcol] = *(uint4*)&St[row * SST + colb];
        }
    }
}

// ---- host launch (TMA descriptor setup; A=M col-major, B=P row-major) ---------
// (excerpt of kernel_run's GIST_GATE_WS path; g_M,g_R from k_stats; harness links -lcuda)
//
//   uint64_t gdA[2]={B,F}, gsA[1]={B*2};  uint32_t bxA[2]={64,64}, esA[2]={1,1};
//   cuTensorMapEncodeTiled(&tmA, BFLOAT16, 2, g_M, gdA, gsA, bxA, esA,
//       INTERLEAVE_NONE, SWIZZLE_128B, L2_PROMOTION_NONE, FLOAT_OOB_FILL_NONE);
//   uint64_t gdB[2]={QF,F}, gsB[1]={QF*2}; uint32_t bxB[2]={64,64}, esB[2]={1,1};
//   cuTensorMapEncodeTiled(&tmB, BFLOAT16, 2, P, gdB, gsB, bxB, esB,
//       INTERLEAVE_NONE, SWIZZLE_128B, L2_PROMOTION_NONE, FLOAT_OOB_FILL_NONE);
//   dim3 grid(ceil(B/128), ceil(QF/256));
//   size_t smem = 128 + NS*(NH+NSUB)*64*64*2 + 128 + 2*NS*8;   // ~197 KB
//   cudaFuncSetAttribute(k_gate_ws, MaxDynamicSharedMemorySize, smem);
//   k_gate_ws<<<grid, 384, smem, stream>>>(tmA, tmB, g_L, B, F, Fp, QF);
