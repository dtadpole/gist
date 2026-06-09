"""Profile Triton's _pool_kernel in isolation: autotuner choice, do_bench MIN, IR/PTX dump.

Run: CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python cuda/profile_triton_pool.py
Set TRITON_KERNEL_DUMP=1 TRITON_CACHE_DIR=/tmp/ttcache to capture IR/PTX.
"""
import os, sys
import torch
import triton

sys.path.insert(0, "/home/zhenc/gist")
from gist_pytorch import GIST, init_unit_scale_
import gist_triton_fast as gtf

B, F, D, Q = 1536, 1497, 192, 128
dev = "cuda"
torch.manual_seed(0)

ref = init_unit_scale_(GIST(F, D, Q).to(dev).to(torch.float32))
x32 = torch.randn(B, F, D, device=dev, dtype=torch.float32)
P32 = ref.proj.weight.t().contiguous()
w32 = ref.norm.weight

x16 = x32.bfloat16()
P16 = P32.bfloat16()
w16 = w32.bfloat16()

# Warm up the full pipeline so the autotuner runs for all kernels.
o = gtf.gist_triton_fast_forward(x16, P16, w16, eps=ref.norm.eps, precision="bf16")
torch.cuda.synchronize()

# ---- reconstruct the exact pool-kernel inputs the pipeline builds ----
QF = P32.shape[1]
is_bf16 = True
BLOCK_D = gtf._next_pow2(D)
m = torch.empty((B, F), device=dev, dtype=torch.float32)
r = torch.empty((B, F), device=dev, dtype=torch.float32)
stats_grid = lambda META: (B, triton.cdiv(F, META["BLOCK_F"]))
gtf._stats_kernel[stats_grid](x16, m, r, B, F, D,
    x16.stride(0), x16.stride(1), x16.stride(2), m.stride(0), m.stride(1),
    ref.norm.eps, BLOCK_D=BLOCK_D)
Mg = m.to(torch.bfloat16).contiguous()
Pg = P16.contiguous()
L = torch.empty((B, QF), device=dev, dtype=torch.bfloat16)
gate_grid = lambda META: (triton.cdiv(B, META["BLOCK_M"]) * triton.cdiv(QF, META["BLOCK_N"]),)
gtf._gate_kernel[gate_grid](Mg, Pg, L, B, F, QF,
    Mg.stride(0), Mg.stride(1), Pg.stride(0), Pg.stride(1),
    L.stride(0), L.stride(1), IS_BF16=True)
L3 = L.view(B, Q, F)
o = torch.empty((B, Q, D), device=dev, dtype=x16.dtype)
d0, d1 = gtf._split_d(D)
weight = w16
pool_grid = lambda META: (B, triton.cdiv(Q, META["BLOCK_Q"]))

def run_pool():
    gtf._pool_kernel[pool_grid](L3, x16, r, weight, o, B, F, D, Q,
        L3.stride(0), L3.stride(1), L3.stride(2),
        x16.stride(0), x16.stride(1), x16.stride(2),
        r.stride(0), r.stride(1), weight.stride(0), weight.stride(1),
        o.stride(0), o.stride(1), o.stride(2),
        HAS_AFFINE=True, IS_BF16=True, D0=d0, D1=d1)

run_pool(); torch.cuda.synchronize()

# ---- autotuner choice ----
best = gtf._pool_kernel.best_config
print("=== Triton _pool_kernel BEST CONFIG ===")
print(best)
print("D0,D1 =", d0, d1)

# ---- do_bench MIN ----
ms = triton.testing.do_bench(run_pool, warmup=50, rep=200, return_mode="min")
print(f"=== Triton pool do_bench MIN = {ms:.4f} ms ===")

# also time gate and stats and full
def run_full():
    gtf.gist_triton_fast_forward(x16, P16, w16, eps=ref.norm.eps, precision="bf16")
ms_full = triton.testing.do_bench(run_full, warmup=50, rep=200, return_mode="min")
print(f"=== Triton FULL do_bench MIN = {ms_full:.4f} ms ===")

# traffic estimate for the pool (bytes)
L_bytes = B * Q * F * 2           # gate tile reads (each g once per q-block; full L per b reread per q-block)
X_bytes = B * F * D * 2           # X read once per q-block
print(f"pool min-traffic (L once + X once): L={L_bytes/1e9:.2f}GB X={X_bytes/1e9:.2f}GB")
print(f"  L is reread per q-block (Q/BLOCK_Q blocks per b); X reread per q-block too")
