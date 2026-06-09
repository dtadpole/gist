// Pool micro-test m64n192k16: C[64,192]=A[64,64]@B[64,192]. A K-major (trans-a=0), B MN-major
// (trans-b=1, 3 contiguous 64-bricks). Hand descriptors. acc[96]. Validate vs CPU ref (max_err 0).
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "cute/arch/mma_sm90_gmma.hpp"
namespace cg = cute::SM90::GMMA;
using AtomN192 = cg::MMA_64x192x16_F32BF16BF16_SS<cg::Major::K, cg::Major::MN, cg::ScaleIn::One, cg::ScaleIn::One>;

__device__ __forceinline__ uint64_t wg_encode(uint32_t x){ return (uint64_t)((x&0x3FFFF)>>4); }
__device__ __forceinline__ uint64_t wg_desc_sw3(const void* p, uint32_t LBO, uint32_t SBO){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(p); uint64_t d=0;
    d|=wg_encode(a); d|=wg_encode(LBO)<<16; d|=wg_encode(SBO)<<32; d|=(uint64_t)1<<62; return d;
}
__device__ __forceinline__ void commit_wait96(float*d){
    asm volatile("wgmma.commit_group.sync.aligned;\n":::"memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n":::"memory");
}
constexpr int M=64,N=192,K=64;
__global__ void mm(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB, float* C){
    int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
    extern __shared__ __align__(128) char sm[];
    __nv_bfloat16* As=(__nv_bfloat16*)sm;                  // [64*64]
    __nv_bfloat16* Bs=As+M*K;                              // [3*64*64] (3 contiguous 64-bricks)
    __shared__ alignas(8) uint64_t mbar;
    uint32_t ma=(uint32_t)__cvta_generic_to_shared(&mbar);
    uint32_t sa=(uint32_t)__cvta_generic_to_shared(As), sb=(uint32_t)__cvta_generic_to_shared(Bs);
    if(tid==0){ asm volatile("mbarrier.init.shared.b64 [%0], 1;"::"r"(ma)); asm volatile("fence.proxy.async.shared::cta;":::"memory"); }
    __syncthreads();
    if(tid==0){
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"::"r"(ma),"r"((unsigned)((M*K+K*N)*2)));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];"
            ::"r"(sa),"l"(&tmA),"r"(0),"r"(0),"r"(ma):"memory");
        // B: 3 boxes of 64 cols (n2*64), one brick each (contiguous in smem)
        for(int n2=0;n2<3;n2++){ uint32_t sbn=sb+n2*64*64*2;
            asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];"
                ::"r"(sbn),"l"(&tmB),"r"(n2*64),"r"(0),"r"(ma):"memory"); }
    }
    asm volatile("{\n.reg .pred p;\nLW: mbarrier.try_wait.parity.shared.b64 p, [%0], 0;\n@!p bra LW;\n}\n"::"r"(ma));
    __syncthreads();
    float acc[96];
    #pragma unroll
    for(int i=0;i<96;i++)acc[i]=0.f;
    asm volatile("wgmma.fence.sync.aligned;\n":::"memory");
    #pragma unroll
    for(int ks=0;ks<4;ks++){
        uint64_t da=wg_desc_sw3(&As[ks*16],16,1024);
        // B 3-brick: LBO walks the brick stride (=64*64=4096 bf16 = 8192B). per-ks +1024 bf16.
        uint64_t db=wg_desc_sw3(&Bs[ks*1024],8192,1024);
        AtomN192::fma(da,db,
            acc[0],acc[1],acc[2],acc[3],acc[4],acc[5],acc[6],acc[7],acc[8],acc[9],acc[10],acc[11],acc[12],acc[13],acc[14],acc[15],
            acc[16],acc[17],acc[18],acc[19],acc[20],acc[21],acc[22],acc[23],acc[24],acc[25],acc[26],acc[27],acc[28],acc[29],acc[30],acc[31],
            acc[32],acc[33],acc[34],acc[35],acc[36],acc[37],acc[38],acc[39],acc[40],acc[41],acc[42],acc[43],acc[44],acc[45],acc[46],acc[47],
            acc[48],acc[49],acc[50],acc[51],acc[52],acc[53],acc[54],acc[55],acc[56],acc[57],acc[58],acc[59],acc[60],acc[61],acc[62],acc[63],
            acc[64],acc[65],acc[66],acc[67],acc[68],acc[69],acc[70],acc[71],acc[72],acc[73],acc[74],acc[75],acc[76],acc[77],acc[78],acc[79],
            acc[80],acc[81],acc[82],acc[83],acc[84],acc[85],acc[86],acc[87],acc[88],acc[89],acc[90],acc[91],acc[92],acc[93],acc[94],acc[95],
            cg::ScaleOut::One);
    }
    commit_wait96(acc);
    // acc[96] layout: group g in [0,48), col c = 8*g + (lane%4)*2, rows r0/r1. acc 4*g..4*g+3.
    #pragma unroll
    for(int g=0;g<24;g++){ int r0=16*warp+lane/4,r1=r0+8,c0=8*g+(lane%4)*2,c1=c0+1;
        C[r0*N+c0]=acc[4*g+0];C[r0*N+c1]=acc[4*g+1];C[r1*N+c0]=acc[4*g+2];C[r1*N+c1]=acc[4*g+3];
    }
}
#define CK(x) do{ CUresult e=(x); if(e){const char*s;cuGetErrorString(e,&s);printf("CU err %d: %s\n",e,s);return 1;} }while(0)
int main(){
    cudaFree(0); cuInit(0);
    static __nv_bfloat16 hA[M*K],hB[K*N]; static float hC[M*N],ref[M*N];
    for(int m=0;m<M;m++)for(int k=0;k<K;k++) hA[m*K+k]=__float2bfloat16(((m*7+k*3)%5-2)*0.5f);
    for(int k=0;k<K;k++)for(int n=0;n<N;n++) hB[k*N+n]=__float2bfloat16(((k*3+n)%7-3)*0.25f);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){float s=0;for(int k=0;k<K;k++)s+=__bfloat162float(hA[m*K+k])*__bfloat162float(hB[k*N+n]);ref[m*N+n]=s;}
    __nv_bfloat16 *dA,*dB; float* dC;
    cudaMalloc(&dA,sizeof hA);cudaMalloc(&dB,sizeof hB);cudaMalloc(&dC,sizeof hC);
    cudaMemcpy(dA,hA,sizeof hA,cudaMemcpyHostToDevice);cudaMemcpy(dB,hB,sizeof hB,cudaMemcpyHostToDevice);
    CUtensorMap tmA,tmB;
    uint64_t gdimA[2]={K,M}, gstrA[1]={(uint64_t)K*2}; uint32_t boxA[2]={64,64}, estA[2]={1,1};
    CK(cuTensorMapEncodeTiled(&tmA,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,dA,gdimA,gstrA,boxA,estA,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    // B row-major [K,N], N contiguous. globalDim {N(fast),K}, box {64,64}; SW128.
    uint64_t gdimB[2]={N,K}, gstrB[1]={(uint64_t)N*2}; uint32_t boxB[2]={64,64}, estB[2]={1,1};
    CK(cuTensorMapEncodeTiled(&tmB,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,dB,gdimB,gstrB,boxB,estB,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    int smem=(M*K+K*N)*2;
    cudaFuncSetAttribute(mm,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
    mm<<<1,128,smem>>>(tmA,tmB,dC);
    cudaError_t e=cudaDeviceSynchronize();
    if(e){printf("kernel err: %s\n",cudaGetErrorString(e));return 1;}
    cudaMemcpy(hC,dC,sizeof hC,cudaMemcpyDeviceToHost);
    float maxerr=0;int nbad=0;
    for(int i=0;i<M*N;i++){float d=fabsf(hC[i]-ref[i]);if(d>maxerr)maxerr=d;if(d>0.5f)nbad++;}
    printf("POOL m64n192 K-major: max_err=%.4f nbad=%d/%d  C[0,0]=%.2f/%.2f C[5,100]=%.2f/%.2f C[40,190]=%.2f/%.2f\n",
           maxerr,nbad,M*N,hC[0],ref[0],hC[5*N+100],ref[5*N+100],hC[40*N+190],ref[40*N+190]);
    return 0;
}
