"""Isolate the big-shape correctness bug: build gist.cu as a .so, run the real
kernel via ctypes on harness-identical inputs, and show WHERE O diverges from the
reference (per-b / per-q / per-d error structure) to pinpoint the failing stage."""
import os, sys, ctypes, subprocess, math
import torch

B, F, D, Q = 1536, 1491, 192, 128
QF = Q * F
_su = (1.0 / 12.0) ** 0.5
SCALES = [1.0/_su, 1.0/_su, (1.0/math.sqrt(F))/_su, (1.0/F)/_su]

def fill(count, seed, scale):  # bit-identical to harness _harness_fill_random_bf16
    idx = torch.arange(count, dtype=torch.int64)
    h = idx ^ seed
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h = ((h ^ (h >> 16)) * 0x45d9f3b) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    return (((h & 0xFFFF).float() / 65536.0 - 0.5) * scale).to(torch.bfloat16)

# ---- build .so ----
GV = os.path.expanduser("~/gist/.venv/lib/python3.12/site-packages")
LIB, INC = f"{GV}/nvidia/cu13/lib", f"{GV}/nvidia/cu13/include"
CCCL = os.path.expanduser("~/cccl/libcudacxx/include")
NVML = f"{GV}/nvidia/nvml_dev/include"
CUT, CUTU = os.path.expanduser("~/cutlass/include"), os.path.expanduser("~/cutlass/tools/util/include")
STUB = os.path.expanduser("~/gist/cuda/linkstubs")
so = "/tmp/gist_dbg.so"
cmd = ["/usr/local/cuda-13.0/bin/nvcc", "-shared", "-Xcompiler", "-fPIC", "-O3",
       "-gencode", "arch=compute_90a,code=sm_90a", "-o", so, os.path.expanduser("~/gist/cuda/gist.cu"),
       f"-I{CUT}", f"-I{CUTU}", f"-I{INC}", f"-I{CCCL}", f"-I{NVML}",
       f"-L{LIB}", f"-L{STUB}", "-L/usr/local/cuda-13.0/lib64", "-Xlinker", f"-rpath={LIB}",
       "--expt-relaxed-constexpr", "-DCUTLASS_ENABLE_GDC_FOR_SM90", "-lcudart"]
print("building...", flush=True)
r = subprocess.run(cmd, env={**os.environ, "LD_LIBRARY_PATH": LIB}, capture_output=True, text=True)
if r.returncode != 0:
    print(r.stderr[-3000:]); sys.exit(1)
print("built", flush=True)

for k, v in [("GIST_B", B), ("GIST_F", F), ("GIST_D", D), ("GIST_Q", Q)]:
    os.environ[f"CUDA_EXEC_PARAM_{k}"] = str(v)
lib = ctypes.CDLL(so)
lib.kernel_run.restype = ctypes.c_int

dev = torch.device("cuda:0")
X = fill(B*F*D, 1, SCALES[0]).to(dev).reshape(B, F, D)
P = fill(F*QF, 2, SCALES[1]).to(dev).reshape(F, QF)
G = fill(D, 3, SCALES[2]).to(dev)
Be = fill(D, 4, SCALES[3]).to(dev)
O = torch.zeros(B, Q, D, dtype=torch.bfloat16, device=dev)

ptrs = (ctypes.c_void_p * 4)(X.data_ptr(), P.data_ptr(), G.data_ptr(), Be.data_ptr())
outp = (ctypes.c_void_p * 1)(O.data_ptr())
rc = lib.kernel_run(ptrs, 4, outp, 1, 0, None)
torch.cuda.synchronize()
print("kernel_run rc =", rc, flush=True)

# ---- reference (all-bf16, mirrors gist_ref.py) ----
Xf = X.float()
Mbf = Xf.mean(-1).to(torch.bfloat16)                       # [B,F]
L = torch.sigmoid((Mbf @ P).float()).view(B, Q, F).to(torch.bfloat16)  # [B,Q,F]
var = Xf.var(-1, unbiased=False)
N = (Xf - Mbf.float().unsqueeze(-1)) * torch.rsqrt(var + 1e-5).unsqueeze(-1) * G.float() + Be.float()
Oref = torch.bmm(L, N.to(torch.bfloat16)).to(torch.bfloat16)   # [B,Q,D] all-bf16 pool

Oc, Or = O.float(), Oref.float()
err = (Oc - Or).abs()
print(f"max_abs={err.max():.4f} mean_abs={err.mean():.4f} | Oref std={Or.std():.3f} mean={Or.mean():.3f}")
print("err by q-block (mean over b,d):", [round(err[:, q0:q0+16].mean().item(), 3) for q0 in range(0, Q, 16)])
print("err by d-block (mean over b,q):", [round(err[:, :, d0:d0+32].mean().item(), 3) for d0 in range(0, D, 32)])
bb = err.mean(dim=(1, 2))  # per-b
print(f"err by b: first8={[round(x,3) for x in bb[:8].tolist()]} ... b1024..1031={[round(x,3) for x in bb[1024:1032].tolist()]}")
print(f"  #b with err>0.1: {(bb>0.1).sum().item()}/{B}; worst b={bb.argmax().item()} ({bb.max():.3f})")
# ratio test: is O_cuda ~ k * O_ref ? (scale bug)  or ~ O_ref + c ? (bias)
m = err > 0.1
if m.any():
    print(f"  on bad elems: Oc/Or median={torch.median(Oc[m]/Or[m].clamp(min=1e-3)):.3f}")
