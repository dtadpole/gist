"""Hardened multi-seed correctness test for the fast GIST kernels (bf16 + tf32).

For 5 seeds x 2 precisions x several shapes (design shape + a masking edge case:
B < BLOCK_M and F not a tile multiple), each kernel output must match
  (a) the fp32 ground truth, and
  (b) the same-precision PyTorch reference
in relative-Frobenius norm, under per-dtype bars. Exits nonzero on any failure.

Run on a CUDA box:  .venv/bin/python test_gist.py   (pin an idle GPU via CUDA_VISIBLE_DEVICES)
"""

import sys
import torch

from gist_pytorch import GIST, init_unit_scale_
from gist_triton_fast import gist_triton_fast_forward

dev = "cuda"
SEEDS = [13, 97, 233, 769, 1597]   # spread-out distinct primes (avoid trivial 0..4)
SHAPES = [
    (1536, 1497, 192, 128),   # design shape
    (8, 323, 192, 64),        # edge: B < BLOCK_M; F not a multiple of the gate/pool tiles
]
# relative-Frobenius bars
BAR_VS_FP32 = {"tf32": 1e-2, "bf16": 5e-2}   # low-precision rounding vs the fp32 truth
BAR_VS_SAME = {"tf32": 5e-3, "bf16": 1e-2}   # kernel vs SAME-precision PyTorch (tiling vs cuBLAS)


def relf(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def main():
    assert torch.cuda.is_available(), "run on a CUDA box"
    fails = 0
    print(f"{'shape':>22} {'prec':>5} {'seed':>4} {'rel/fp32':>10} {'rel/same':>10}  res")
    print("-" * 66)
    for (B, F, D, Q) in SHAPES:
        for prec in ("tf32", "bf16"):
            for s in SEEDS:
                torch.manual_seed(s)
                ref = init_unit_scale_(GIST(F, D, Q).to(dev).to(torch.float32)).eval()
                x = torch.randn(B, F, D, device=dev)
                P = ref.proj.weight.t().contiguous()

                torch.backends.cuda.matmul.allow_tf32 = False
                with torch.no_grad():
                    o_fp32 = ref(x)

                if prec == "bf16":
                    ref16 = GIST(F, D, Q).to(dev).to(torch.bfloat16).eval()
                    ref16.load_state_dict({k: v.to(torch.bfloat16) for k, v in ref.state_dict().items()})
                    with torch.no_grad():
                        o_same = ref16(x.to(torch.bfloat16))
                    o_tri = gist_triton_fast_forward(
                        x.to(torch.bfloat16), P.to(torch.bfloat16),
                        ref.norm.weight.to(torch.bfloat16),
                        eps=ref.norm.eps, precision="bf16")
                else:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.set_float32_matmul_precision("high")
                    with torch.no_grad():
                        o_same = ref(x)
                    o_tri = gist_triton_fast_forward(
                        x, P, ref.norm.weight, eps=ref.norm.eps, precision="tf32")
                    torch.backends.cuda.matmul.allow_tf32 = False

                r_fp32 = relf(o_tri, o_fp32)
                r_same = relf(o_tri, o_same)
                ok = r_fp32 < BAR_VS_FP32[prec] and r_same < BAR_VS_SAME[prec]
                fails += not ok
                print(f"{str((B, F, D, Q)):>22} {prec:>5} {s:>4} "
                      f"{r_fp32:>10.2e} {r_same:>10.2e}  {'OK' if ok else 'FAIL'}")

    n = len(SHAPES) * 2 * len(SEEDS)
    print("-" * 66)
    print(f"ALL {n} CHECKS PASSED" if not fails else f"{fails}/{n} CHECKS FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
