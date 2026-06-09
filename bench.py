"""Benchmark GIST forward: PyTorch eager vs torch.compile vs Triton fused kernel.

Reports median latency and achieved TFLOP/s at the design-default large shape
(B=1536, F=1491, D=192, Q=128). FLOPs counted as the two matmuls:
  gate  M[B,F] @ P[F, Q*F]   -> 2*B*F*(Q*F)   (~0.9 TFLOP, GEMV-shaped: P not reused across batch)
  pool  L[B,Q,F] @ N[B,F,D]  -> 2*B*Q*F*D     (~0.1 TFLOP)

Run on a CUDA box:  .venv/bin/python bench.py
"""

from __future__ import annotations

import torch

from gist_pytorch import GIST, init_unit_scale_
from gist_triton import gist_triton_forward
from gist_triton_fast import gist_triton_fast_forward


def cuda_time(fn, warmup: int = 5, iters: int = 20) -> float:
    """Median wall time (ms) of fn() on the GPU, via CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def main():
    assert torch.cuda.is_available(), "run on a CUDA box"
    torch.manual_seed(0)
    dev = "cuda"

    B, F, D, Q = 1536, 1497, 192, 128
    gate_flops = 2 * B * F * (Q * F)
    pool_flops = 2 * B * Q * F * D
    total_flops = gate_flops + pool_flops

    print(f"shape B={B} F={F} D={D} Q={Q}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"work: gate ~{gate_flops/1e12:.2f} TFLOP + pool ~{pool_flops/1e12:.2f} TFLOP "
          f"= total ~{total_flops/1e12:.2f} TFLOP")
    print("init: unit-scale (O~O(1)); error = rel-Frobenius vs fp32-ieee eager\n")

    # fp32-ieee eager is the correctness reference; unit-scale init so O ~ O(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    ref32 = init_unit_scale_(GIST(F, D, Q).to(dev).to(torch.float32)).eval()
    x32 = torch.randn(B, F, D, device=dev, dtype=torch.float32)
    with torch.no_grad():
        o_ref = ref32(x32)

    rows = []  # (label, group, latency_ms, rel_frobenius, abs_err)

    def measure(label, group, fn, iters=20, warmup=5):
        with torch.no_grad():
            o = fn().float()
            ms = cuda_time(fn, warmup=warmup, iters=iters)
        rel = ((o - o_ref).norm() / o_ref.norm()).item()
        abserr = (o - o_ref).abs().max().item()
        rows.append((label, group, ms, rel, abserr))

    # ---- PyTorch eager + compile, precision sweep ----
    # speedup is computed within each precision group vs that group's eager baseline;
    # error is always vs the fp32-ieee eager ground truth.
    torch.backends.cuda.matmul.allow_tf32 = False
    measure("eager", "fp32", lambda: ref32(x32))
    cref32 = torch.compile(ref32); cref32(x32)
    measure("compile", "fp32", lambda: cref32(x32))

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    measure("eager", "tf32", lambda: ref32(x32))
    cref32_tf = torch.compile(ref32); cref32_tf(x32)
    measure("compile", "tf32", lambda: cref32_tf(x32))

    ref16 = GIST(F, D, Q).to(dev).to(torch.bfloat16).eval()
    ref16.load_state_dict({k: v.to(torch.bfloat16) for k, v in ref32.state_dict().items()})
    x16 = x32.to(torch.bfloat16)
    measure("eager", "bf16", lambda: ref16(x16))
    cref16 = torch.compile(ref16); cref16(x16)
    measure("compile", "bf16", lambda: cref16(x16))

    # ---- our fast two-kernel Triton (batch-tiled, tensor core) ----
    P32 = ref32.proj.weight.t().contiguous()
    w32 = ref32.norm.weight                                   # RMSNorm weight [F, D]
    measure("triton-fast", "tf32",
            lambda: gist_triton_fast_forward(x32, P32, w32, eps=ref32.norm.eps, precision="tf32"))
    x16b, P16b, w16 = x16, P32.bfloat16(), w32.bfloat16()
    measure("triton-fast", "bf16",
            lambda: gist_triton_fast_forward(x16b, P16b, w16, eps=ref32.norm.eps, precision="bf16"))

    # ---- v1 correctness-first fused kernel (fp32 group; for reference, slow) ----
    torch.backends.cuda.matmul.allow_tf32 = False
    measure("triton v1 (slow)", "fp32",
            lambda: gist_triton_forward(x32, P32, w32, eps=ref32.norm.eps,
                                        BLOCK_Q=16, BLOCK_G=16, BLOCK_K=32),
            iters=10, warmup=3)

    # per-group eager baseline for same-precision speedup
    base = {grp: ms for label, grp, ms, _, _ in rows if label == "eager"}
    print(f"work per call: {total_flops/1e9:.1f} GFLOP "
          f"(gate {gate_flops/1e9:.1f} + pool {pool_flops/1e9:.1f})")
    print("speedup = vs same-precision eager;  error = rel-Frobenius / max-abs vs fp32 truth\n")
    print(f"{'method':16s} {'prec':5s} {'latency':>10s} {'speedup':>8s} {'TFLOP/s':>9s} "
          f"{'rel-err':>9s} {'abs-err':>9s}")
    print("-" * 74)
    last = None
    for label, grp, ms, rel, abserr in rows:
        if last is not None and grp != last:
            print()
        last = grp
        tflops = total_flops / (ms * 1e-3) / 1e12
        print(f"{label:16s} {grp:5s} {ms:7.3f} ms {base[grp]/ms:7.2f}x {tflops:8.1f} "
              f"{rel:9.1e} {abserr:9.1e}")


if __name__ == "__main__":
    main()
