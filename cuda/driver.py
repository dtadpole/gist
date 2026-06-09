"""Drive the cuda_exec harness for the GIST kernel — code lives here in ~/gist,
the harness (compile.sh + eval_harness.cu + trial.py) is used as a tool.

Run with the worktree's cuda_exec on PYTHONPATH and the pip CUDA toolkit on the
NVCC_* / LD_LIBRARY_PATH env (see run_gist.sh):

    PYTHONPATH=~/kernel_lab/.worktrees/gist \
    NVCC_INCLUDE_DIRS=<pip include> NVCC_LIB_DIRS=<pip lib> \
    LD_LIBRARY_PATH=<pip lib> \
    ~/kernel_lab/.venv/bin/python driver.py [revision]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cuda_exec
from cuda_exec.models import Metadata
from cuda_exec.tasks import run_compile_task, run_trial_task, _primary_artifact_from_manifest
from cuda_exec.runner import resolve_workspace_bundle

print("cuda_exec from:", cuda_exec.__file__)   # must be the worktree copy

HERE = Path(__file__).resolve().parent
# GIST_CU lets a parallel exploration point the shared driver/harness at a per-direction
# gist.cu (CUDA source only) while keeping driver/ref/toolchain shared. Defaults to local.
import os as _os
CU = Path(_os.environ.get("GIST_CU", str(HERE / "gist.cu"))).read_text()
REF = (HERE / "gist_ref.py").read_text()

rev = int(sys.argv[1]) if len(sys.argv) > 1 else 1
shape_mode = sys.argv[2] if len(sys.argv) > 2 else "small"

# small = quick correctness; big = design shape (matches the Triton benchmark)
if shape_mode == "big":
    B, F, D, Q = 1536, 1491, 192, 128
else:
    B, F, D, Q = 8, 320, 192, 64
QF = Q * F
X_n, P_n, O_n = B * F * D, F * QF, B * Q * D
buf = max(X_n, P_n, O_n)

import math
# unit-scale init (match gist_pytorch.init_unit_scale_): X,P ~ N(0,1); gamma ~ N(0,1/sqrt(F));
# beta ~ N(0,1/F) -> O ~ O(1), gates span (0,1). Harness fills uniform[-0.5,0.5) (std 1/sqrt(12)),
# so per-input scale = target_std / uniform_std makes the bf16 kernel testable at allclose(1e-2).
_su = (1.0 / 12.0) ** 0.5
config = {
    "family": "gist",
    "shape": [B, Q, D], "rank": 3, "shape_kind": "3d", "input_size": O_n,
    "input_shapes": [[B, F, D], [F, QF], [D], [D]],
    "harness_num_inputs": 4, "harness_num_outputs": 1,
    "harness_buf_size": buf, "harness_output_size": O_n,
    "gist_b": B, "gist_f": F, "gist_d": D, "gist_q": Q,
    "harness_scale_0": 1.0 / _su,                       # X ~ N(0,1)
    "harness_scale_1": 1.0 / _su,                       # P ~ N(0,1)
    "harness_scale_2": (1.0 / math.sqrt(F)) / _su,      # gamma ~ N(0,1/sqrt(F))
    "harness_scale_3": (1.0 / F) / _su,                 # beta ~ N(0,1/F)
}

md = Metadata(run_tag="run_h8_3_gist", version="v1", direction_id=0,
              direction_slug="gist", revision=rev)

print("=== compile (rev %d) ===" % rev)
cres = run_compile_task(
    metadata=md, timeout_seconds=300,
    impls={"ref-pytorch": {"pytorch.py": REF}, "gen-cuda": {"cuda.cu": CU}},
)
print("compile keys:", list(cres.keys()))
print("compile all_ok:", cres.get("all_ok"), cres.get("status"))
# surface compile errors/logs if any
for k in ("logs", "error", "errors"):
    if cres.get(k):
        print(f"compile.{k}:", json.dumps(cres[k])[:2000])

ws = resolve_workspace_bundle(**md.model_dump())
try:
    binpath, _art = _primary_artifact_from_manifest(ws)
    print("binary:", binpath, "exists:", Path(binpath).exists())
except Exception as e:
    print("binary resolve failed:", e)
    binpath = ""

print("=== trial ===")
tres = run_trial_task(
    metadata=md, timeout_seconds=300,
    configs={f"gist-{shape_mode}": config}, gpu_index=0,
    binary_map=f"gen-cuda={binpath}" if binpath else "",
)
print(json.dumps(tres, indent=2)[:4000])
