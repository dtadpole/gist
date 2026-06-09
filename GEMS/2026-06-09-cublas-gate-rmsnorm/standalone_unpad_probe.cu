// Probe: can cuBLAS/cublasLt do the gate GEMM at UNPADDED K=F=1497 fast (avoid the 0.56ms P-pad)?
// If any algo gets < 1.27ms at K=1497 with NO pad, the pad copy is avoidable. bf16 IO, fp32 acc.
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#define CK(x) do{cudaError_t e=(x); if(e){printf("cuda err %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
#define CB(x) do{cublasStatus_t e=(x); if(e){printf("cublas err %d line %d\n",(int)e,__LINE__);}}while(0)
static int FLUSH_N=64*1024*1024;
int main(){
    int B=1536,F=1497,Q=128; long QF=(long)Q*F;
    cublasHandle_t h; cublasCreate(&h);
    __nv_bfloat16 *M,*P,*L; CK(cudaMalloc(&M,(size_t)B*F*2)); CK(cudaMalloc(&P,(size_t)F*QF*2)); CK(cudaMalloc(&L,(size_t)B*QF*2));
    CK(cudaMemset(M,0,(size_t)B*F*2)); CK(cudaMemset(P,0,(size_t)F*QF*2));
    int* flush; CK(cudaMalloc(&flush,(size_t)FLUSH_N*4));
    float a=1.f,b=0.f; int N=(int)QF;
    // UNPADDED K=F (same op flags as padded, but k=F, Mp=M ld=B, Pp=P ld=QF)
    auto run=[&](){ cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,N,B,F,&a,P,CUDA_R_16BF,N,M,CUDA_R_16BF,B,&b,
        L,CUDA_R_16BF,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT); };
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    for(int i=0;i<15;i++) run(); CK(cudaDeviceSynchronize());
    std::vector<float> t(50);
    for(int i=0;i<50;i++){ CK(cudaMemset(flush,0,(size_t)FLUSH_N*4));
        cudaEventRecord(s); run(); cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&t[i],s,e); }
    std::sort(t.begin(),t.end());
    double flops=2.0*B*QF*F;
    printf("UNPADDED cuBLAS GEMM K=F=%d : %.4f ms (%.0f TFLOP/s)  [default heuristic]\n",F,t[25],flops/(t[25]*1e-3)/1e12);
    // try a few explicit algos
    for(int alg=CUBLAS_GEMM_DEFAULT_TENSOR_OP; alg<=CUBLAS_GEMM_DEFAULT_TENSOR_OP+15; alg++){
        auto run2=[&](){ cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,N,B,F,&a,P,CUDA_R_16BF,N,M,CUDA_R_16BF,B,&b,
            L,CUDA_R_16BF,N,CUBLAS_COMPUTE_32F,(cublasGemmAlgo_t)alg); };
        cublasStatus_t st=cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_T,N,B,F,&a,P,CUDA_R_16BF,N,M,CUDA_R_16BF,B,&b,
            L,CUDA_R_16BF,N,CUBLAS_COMPUTE_32F,(cublasGemmAlgo_t)alg);
        if(st!=CUBLAS_STATUS_SUCCESS) continue;
        for(int i=0;i<10;i++) run2(); CK(cudaDeviceSynchronize());
        std::vector<float> tt(20);
        for(int i=0;i<20;i++){ CK(cudaMemset(flush,0,(size_t)FLUSH_N*4));
            cudaEventRecord(s); run2(); cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&tt[i],s,e); }
        std::sort(tt.begin(),tt.end());
        printf("  algo %d : %.4f ms\n",alg,tt[10]);
    }
    return 0;
}
