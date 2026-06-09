// Pool micro-test: K-major A (trans-a=0) + MN-major B (trans-b=1) wgmma via cute make_gmma_desc.
// C[M,N] = A[M,K] @ B[K,N]. A row-major (K contig) -> K-major operand. B row-major [K,N] (N contig)
// -> MN-major operand (its leading global dim is N). Both TMA SW128. Validate vs CPU ref.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "cute/tensor.hpp"
#include "cute/arch/mma_sm90_gmma.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"

using namespace cute;
namespace cg = cute::SM90::GMMA;

using AtomN64 = cg::MMA_64x64x16_F32BF16BF16_SS<cg::Major::K, cg::Major::MN, cg::ScaleIn::One, cg::ScaleIn::One>;
#ifndef ADV_A
#define ADV_A 2
#endif
#ifndef ADV_B
#define ADV_B 128
#endif

__device__ __forceinline__ uint64_t wg_encode(uint32_t x){ return (uint64_t)((x&0x3FFFF)>>4); }
__device__ __forceinline__ uint64_t wg_desc_sw3(const void* p, uint32_t LBO, uint32_t SBO){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(p); uint64_t d=0;
    d|=wg_encode(a); d|=wg_encode(LBO)<<16; d|=wg_encode(SBO)<<32; d|=(uint64_t)1<<62; return d;
}
__device__ __forceinline__ void commit_wait32(float*d){
    asm volatile("wgmma.commit_group.sync.aligned;\n":::"memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n"
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
         "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
         "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
         "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
        ::"memory");
}

constexpr int M=64, N=64, K=64;
__global__ void mm(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB, float* C){
    int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
    __shared__ alignas(128) __nv_bfloat16 As[M*K];
    __shared__ alignas(128) __nv_bfloat16 Bs[K*N];
    __shared__ alignas(8) uint64_t mbar;
    uint32_t ma=(uint32_t)__cvta_generic_to_shared(&mbar);
    uint32_t sa=(uint32_t)__cvta_generic_to_shared(As), sb=(uint32_t)__cvta_generic_to_shared(Bs);
    if(tid==0){
        asm volatile("mbarrier.init.shared.b64 [%0], 1;"::"r"(ma));
        asm volatile("fence.proxy.async.shared::cta;":::"memory");
    }
    __syncthreads();
    if(tid==0){
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"::"r"(ma),"r"((unsigned)((M*K+K*N)*2)));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];"
            ::"r"(sa),"l"(&tmA),"r"(0),"r"(0),"r"(ma):"memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];"
            ::"r"(sb),"l"(&tmB),"r"(0),"r"(0),"r"(ma):"memory");
    }
    asm volatile("{\n.reg .pred p;\nLW: mbarrier.try_wait.parity.shared.b64 p, [%0], 0;\n@!p bra LW;\n}\n"::"r"(ma));
    __syncthreads();
    float acc[32];
    #pragma unroll
    for(int i=0;i<32;i++)acc[i]=0.f;
    asm volatile("wgmma.fence.sync.aligned;\n":::"memory");

    // cute canonical smem layouts (K-major A SW128, MN-major B SW128) tiled to [64,64].
    auto sA_layout = tile_to_shape(cg::Layout_K_SW128_Atom<__nv_bfloat16>{},  Shape<Int<M>,Int<K>>{});
    auto sB_layout = tile_to_shape(cg::Layout_MN_SW128_Atom<__nv_bfloat16>{}, Shape<Int<N>,Int<K>>{}); // MN-major: (N,K)
    Tensor sA = make_tensor(make_smem_ptr(As), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(Bs), sB_layout);
    // Split K(=64) into (16, 4) so we can slice each k16 step: tile [MN, 16] per ks.
    Tensor sAk = zipped_divide(sA, Shape<Int<M>,_16>{});   // ((M,16),(1,4))
    Tensor sBk = zipped_divide(sB, Shape<Int<N>,_16>{});   // ((N,16),(1,4))
    #pragma unroll
    for(int ks=0;ks<4;ks++){
        Tensor a_k = sAk(make_coord(_,_), make_coord(0,ks));   // [M,16] K-major slice
        Tensor b_k = sBk(make_coord(_,_), make_coord(0,ks));   // [N,16] MN-major slice
        // HAND descriptors (the in-kernel recipe): A K-major LBO=16B,SBO=1024B, advance 16 bf16/ks.
        //                                          B MN-major LBO=0,SBO=1024B, advance 1024 bf16/ks.
        uint64_t da = wg_desc_sw3(&As[ks*16],   16, 1024);
        uint64_t db = wg_desc_sw3(&Bs[ks*1024],  0, 1024);
        (void)a_k; (void)b_k;
        AtomN64::fma(da,db,
            acc[0],acc[1],acc[2],acc[3],acc[4],acc[5],acc[6],acc[7],acc[8],acc[9],acc[10],acc[11],acc[12],acc[13],acc[14],acc[15],
            acc[16],acc[17],acc[18],acc[19],acc[20],acc[21],acc[22],acc[23],acc[24],acc[25],acc[26],acc[27],acc[28],acc[29],acc[30],acc[31],
            cg::ScaleOut::One);
    }
    commit_wait32(acc);
    #pragma unroll
    for(int a=0;a<8;a++){ int r0=16*warp+lane/4,r1=r0+8,c0=8*a+(lane%4)*2,c1=c0+1;
        C[r0*N+c0]=acc[4*a+0];C[r0*N+c1]=acc[4*a+1];C[r1*N+c0]=acc[4*a+2];C[r1*N+c1]=acc[4*a+3];
    }
}
#define CK(x) do{ CUresult e=(x); if(e){const char*s;cuGetErrorString(e,&s);printf("CU err %d: %s\n",e,s);return 1;} }while(0)
int main(){
    cudaFree(0); cuInit(0);
    __nv_bfloat16 hA[M*K],hB[K*N]; float hC[M*N],ref[M*N];
    for(int m=0;m<M;m++)for(int k=0;k<K;k++) hA[m*K+k]=__float2bfloat16(((m*7+k*3)%5-2)*0.5f);
    for(int k=0;k<K;k++)for(int n=0;n<N;n++) hB[k*N+n]=__float2bfloat16(((k*3+n)%7-3)*0.25f);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){float s=0;for(int k=0;k<K;k++)s+=__bfloat162float(hA[m*K+k])*__bfloat162float(hB[k*N+n]);ref[m*N+n]=s;}
    __nv_bfloat16 *dA,*dB; float* dC;
    cudaMalloc(&dA,sizeof hA);cudaMalloc(&dB,sizeof hB);cudaMalloc(&dC,sizeof hC);
    cudaMemcpy(dA,hA,sizeof hA,cudaMemcpyHostToDevice);cudaMemcpy(dB,hB,sizeof hB,cudaMemcpyHostToDevice);
    CUtensorMap tmA,tmB;
    uint64_t gdimA[2]={K,M}, gstrA[1]={(uint64_t)K*2};
    uint32_t boxA[2]={64,64}, estA[2]={1,1};
    CK(cuTensorMapEncodeTiled(&tmA,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,dA,gdimA,gstrA,boxA,estA,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    uint64_t gdimB[2]={N,K}, gstrB[1]={(uint64_t)N*2};
    uint32_t boxB[2]={64,64}, estB[2]={1,1};
    CK(cuTensorMapEncodeTiled(&tmB,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,dB,gdimB,gstrB,boxB,estB,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    mm<<<1,128>>>(tmA,tmB,dC);
    cudaError_t e=cudaDeviceSynchronize();
    if(e){printf("kernel err: %s\n",cudaGetErrorString(e));return 1;}
    cudaMemcpy(hC,dC,sizeof hC,cudaMemcpyDeviceToHost);
    float maxerr=0;int nbad=0;
    for(int i=0;i<M*N;i++){float d=fabsf(hC[i]-ref[i]);if(d>maxerr)maxerr=d;if(d>0.5f)nbad++;}
    printf("POOL K-major A: max_err=%.4f nbad=%d/%d  C[0,0]=%.2f ref=%.2f  C[5,9]=%.2f ref=%.2f  C[33,40]=%.2f ref=%.2f\n",
           maxerr,nbad,M*N,hC[0],ref[0],hC[5*N+9],ref[5*N+9],hC[33*N+40],ref[33*N+40]);
    return 0;
}
