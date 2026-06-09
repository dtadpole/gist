// Bench the PADDED cuBLAS gate GEMM + the Mp/Pp pad kernels SEPARATELY at the exact GIST gate shape.
// Honest: warmup, 100 timed iters, L2 flush (256MB) before each, CUDA events. bf16 IO, fp32 accumulate.
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CK(x) do{cudaError_t e=(x); if(e){printf("cuda err %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
#define CB(x) do{cublasStatus_t e=(x); if(e){printf("cublas err %d line %d\n",(int)e,__LINE__);exit(1);}}while(0)

__global__ void k_padMcm(const __nv_bfloat16* __restrict__ M, __nv_bfloat16* __restrict__ Mp, int B, int F){
    long tot=(long)B*F;
    for(long i=(long)blockIdx.x*blockDim.x+threadIdx.x;i<tot;i+=(long)gridDim.x*blockDim.x) Mp[i]=M[i];
}
__global__ void k_padProw(const __nv_bfloat16* __restrict__ P, __nv_bfloat16* __restrict__ Pp, int F, long QF){
    long tot8=((long)F*QF)>>3; const uint4* Pv=(const uint4*)P; uint4* Ppv=(uint4*)Pp;
    for(long i=(long)blockIdx.x*blockDim.x+threadIdx.x;i<tot8;i+=(long)gridDim.x*blockDim.x) Ppv[i]=Pv[i];
}

static int FLUSH_N = 64*1024*1024;
float bench(void(*fn)(void*), void* ctx, int iters, int* flush){
    cudaEvent_t evs,eve; CK(cudaEventCreate(&evs)); CK(cudaEventCreate(&eve));
    for(int i=0;i<15;i++) fn(ctx);
    CK(cudaDeviceSynchronize());
    std::vector<float> t(iters);
    for(int i=0;i<iters;i++){
        CK(cudaMemset(flush,0,(size_t)FLUSH_N*4));
        CK(cudaEventRecord(evs)); fn(ctx); CK(cudaEventRecord(eve)); CK(cudaEventSynchronize(eve));
        CK(cudaEventElapsedTime(&t[i],evs,eve));
    }
    std::sort(t.begin(),t.end());
    return t[iters/2];   // median
}

struct Ctx { cublasHandle_t h; __nv_bfloat16 *Mp,*Pp,*L,*M,*P; int B,F,Fp; long QF; };
static void do_gemm(void* p){ Ctx*c=(Ctx*)p; float a=1.f,b=0.f; int N=(int)c->QF;
    cublasGemmEx(c->h,CUBLAS_OP_N,CUBLAS_OP_T,N,c->B,c->Fp,&a,c->Pp,CUDA_R_16BF,N,c->Mp,CUDA_R_16BF,c->B,&b,
        c->L,CUDA_R_16BF,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT); }
static void do_padM(void* p){ Ctx*c=(Ctx*)p; k_padMcm<<<512,256>>>(c->M,c->Mp,c->B,c->F); }
static void do_padP(void* p){ Ctx*c=(Ctx*)p; k_padProw<<<2048,256>>>(c->P,c->Pp,c->F,c->QF); }
static void do_all(void* p){ do_padM(p); do_padP(p); do_gemm(p); }

int main(){
    Ctx c; c.B=1536; c.F=1497; c.Fp=1536; int Q=128; c.QF=(long)Q*c.F;
    CB(cublasCreate(&c.h));
    CK(cudaMalloc(&c.M,(size_t)c.B*c.F*2)); CK(cudaMalloc(&c.P,(size_t)c.F*c.QF*2));
    CK(cudaMalloc(&c.Mp,(size_t)c.B*c.Fp*2)); CK(cudaMalloc(&c.Pp,(size_t)c.Fp*c.QF*2));
    CK(cudaMalloc(&c.L,(size_t)c.B*c.QF*2));
    CK(cudaMemset(c.M,1,(size_t)c.B*c.F*2)); CK(cudaMemset(c.P,1,(size_t)c.F*c.QF*2));
    CK(cudaMemset(c.Mp,0,(size_t)c.B*c.Fp*2)); CK(cudaMemset(c.Pp,0,(size_t)c.Fp*c.QF*2));
    int* flush; CK(cudaMalloc(&flush,(size_t)FLUSH_N*4));
    do_padM(&c); do_padP(&c); CK(cudaDeviceSynchronize());
    double flops = 2.0*c.B*c.QF*c.Fp;
    float gemm = bench(do_gemm,&c,100,flush);
    float pm   = bench(do_padM,&c,100,flush);
    float pp   = bench(do_padP,&c,100,flush);
    float all  = bench(do_all,&c,100,flush);
    printf("gate shape M=%d N=%ld K(pad)=%d  (valid F=%d)\n",c.B,c.QF,c.Fp,c.F);
    printf("  cuBLAS GEMM only : %.4f ms  (%.0f TFLOP/s)\n",gemm,flops/(gemm*1e-3)/1e12);
    printf("  padM only        : %.4f ms\n",pm);
    printf("  padP only        : %.4f ms\n",pp);
    printf("  padM+padP+GEMM   : %.4f ms  (full padded-cuBLAS gate)\n",all);
    return 0;
}
