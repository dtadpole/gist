#!/usr/bin/env bash
# Assemble a working CUDA-13 toolchain on this devserver (system CUDA is
# runtime-only: nvcc/ptxas/nvdisasm present, but no headers/libs/cuobjdump).
# Idempotent. Network steps go through `ssh localhost` (run ssh_fix.sh first).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GVENV="$HOME/gist/.venv/lib/python3.12/site-packages"
LIB="$GVENV/nvidia/cu13/lib"

# 1) host_env maps this box to CUDA_HOME=/usr/local/cuda-13.2 (not installed);
#    symlink it to the installed 13.0 (nvcc resolves internals via realpath).
if [[ ! -e /usr/local/cuda-13.2 ]]; then
  sudo ln -sfn /usr/local/cuda-13.0 /usr/local/cuda-13.2
  echo "[toolchain] symlinked cuda-13.2 -> cuda-13.0"
fi

# 2) CCCL headers (nv/target, cuda/std) pulled in by cuda_bf16.h — header-only clone.
if [[ ! -e "$HOME/cccl/libcudacxx/include/nv/target" ]]; then
  ssh localhost "cd ~ && git -c url.'https://github.com/'.insteadOf='git@github.com:' clone --depth 1 https://github.com/NVIDIA/cccl.git ~/cccl" 2>&1 | tail -2
  echo "[toolchain] cloned CCCL"
fi

# 3) nvml.h (eval_harness.cu uses NVML); pip cu13 lacks it, cu12 wheel has it.
if [[ ! -e "$GVENV/nvidia/nvml_dev/include/nvml.h" ]]; then
  ssh localhost "cd ~/gist && uv pip install nvidia-nvml-dev-cu12" 2>&1 | tail -2
  echo "[toolchain] installed nvml.h (nvidia-nvml-dev-cu12)"
fi

# 4) Unversioned link stubs — pip ships libcublas.so.13 etc. but ld wants libX.so.
mkdir -p "$HERE/linkstubs"
ln -sf "$LIB/libcublas.so.13"   "$HERE/linkstubs/libcublas.so"
ln -sf "$LIB/libcublasLt.so.13" "$HERE/linkstubs/libcublasLt.so"
ln -sf /usr/lib64/libnvidia-ml.so.1 "$HERE/linkstubs/libnvidia-ml.so"
echo "[toolchain] link stubs ready"

# 5) kernel_lab worktree with the gist-family harness support.
if [[ ! -d "$HOME/kernel_lab/.worktrees/gist" ]]; then
  ( cd "$HOME/kernel_lab" && mkdir -p .worktrees && git worktree add .worktrees/gist -b worktree-h8_3-gist )
  echo "[toolchain] created kernel_lab worktree"
fi
echo "[toolchain] OK"
