#!/usr/bin/env bash
# Run the GIST CUDA kernel through the cuda_exec harness (worktree) with the
# pip-assembled CUDA-13 toolchain. Usage: ./run_gist.sh <revision> [big]
#
# Toolchain pieces (this box has only a runtime-only system CUDA):
#   - nvcc/ptxas/nvdisasm : system /usr/local/cuda-13.0 (13.2 symlinked -> 13.0)
#   - headers/libs        : pip nvidia-cu13 (in ~/gist/.venv)
#   - nv/target (CCCL)    : ~/cccl/libcudacxx/include
#   - nvml.h              : pip nvidia-nvml-dev-cu12
#   - libcublas/.so etc.  : unversioned link stubs in ./linkstubs
# Harness gist-family support lives in the kernel_lab worktree (PYTHONPATH).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$HOME/kernel_lab/.worktrees/gist"
GVENV="$HOME/gist/.venv/lib/python3.12/site-packages"
LIB="$GVENV/nvidia/cu13/lib"
INC="$GVENV/nvidia/cu13/include"
CCCL="$HOME/cccl/libcudacxx/include"
NVML="$GVENV/nvidia/nvml_dev/include"
STUB="$HERE/linkstubs"

CUTLASS="$HOME/cutlass/include"
CUTLASS_UTIL="$HOME/cutlass/tools/util/include"

PYTHONPATH="$WT" \
  NVCC_INCLUDE_DIRS="$CUTLASS $CUTLASS_UTIL $INC $CCCL $NVML" \
  NVCC_LIB_DIRS="$LIB" \
  NVCC_EXTRA_FLAGS="-L$LIB -L$STUB -Xlinker -rpath=$LIB -lcublas --expt-relaxed-constexpr -DCUTLASS_ENABLE_GDC_FOR_SM90 ${GIST_NVCC_APPEND:-}" \
  LD_LIBRARY_PATH="$LIB" \
  "$HOME/kernel_lab/.venv/bin/python" "$HERE/driver.py" "$@"
