#!/usr/bin/env bash
# Profile the cached gen-cuda binary for a given rev with nsys (GPU metrics + cuda trace).
# Usage: ./nsys_gate.sh <rev> [extra_env_KEY=VAL ...]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REV="$1"; shift || true
WS="$HOME/.cuda_exec/run_h8_3_gist/v1/0_gist/rev_${REV}"
BIN="$WS/artifacts/compile.attempt_001.generated.bin"
OUT="$WS/workspace/_output_gen-cuda_gist-big"
mkdir -p "$OUT"
GVENV="$HOME/gist/.venv/lib/python3.12/site-packages"
LIB="$GVENV/nvidia/cu13/lib"
# design shape env
export CUDA_EXEC_PARAM_GIST_B=1536 CUDA_EXEC_PARAM_GIST_F=1497 \
       CUDA_EXEC_PARAM_GIST_D=192 CUDA_EXEC_PARAM_GIST_Q=128
# harness buffer sizing (B*F*D, F*Q*F, B*Q*D -> max; out=B*Q*D) — MUST match driver.py config (RMSNorm/F=1497)
# B*F*D=1536*1497*192=441452544, F*Q*F=1497*128*1497=286794048, B*Q*D=37748736 -> buf=max=441452544
export CUDA_EXEC_PARAM_INPUT_SIZE=37748736
export CUDA_EXEC_PARAM_RANK=3
export CUDA_EXEC_PARAM_SHAPE_KIND=3d
export CUDA_EXEC_PARAM_SHAPE="1536,128,192"
export CUDA_EXEC_PARAM_HARNESS_NUM_INPUTS=3
export CUDA_EXEC_PARAM_HARNESS_NUM_OUTPUTS=1
export CUDA_EXEC_PARAM_HARNESS_BUF_SIZE=441452544
export CUDA_EXEC_PARAM_HARNESS_OUTPUT_SIZE=37748736
# RMSNorm scales (driver.py): X,P=1/sqrt(1/12)=3.4641; W=(1/sqrt(F))/sqrt(1/12)=0.089528 (F=1497)
export CUDA_EXEC_PARAM_HARNESS_SCALE_0=3.464101615137755
export CUDA_EXEC_PARAM_HARNESS_SCALE_1=3.464101615137755
export CUDA_EXEC_PARAM_HARNESS_SCALE_2=0.08952812
export CUDA_EXEC_OUTPUT_DIR="$OUT"
export LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH:-}"
# extra env overrides (e.g. GIST_SKIP_POOL=1)
for kv in "$@"; do export "$kv"; done
cd "$WS/workspace"
echo "BIN=$BIN"
"$BIN"
