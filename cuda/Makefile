# GIST CUDA kernel — build & run through the cuda_exec harness.
#
# This devserver has only a runtime-only system CUDA, so the toolchain is
# assembled from pip wheels + a CCCL clone + link stubs (see setup_toolchain.sh).
# All kernel code lives here in ~/gist/cuda; the kernel_lab cuda_exec harness
# (worktree, with gist-family support) is used only as the compile/test tool.
#
# Quick start on a fresh box:   make setup && make run
# Big shape (vs Triton):        make run-big
# Triton baseline to beat:      make triton

SHELL := /bin/bash
HERE  := $(CURDIR)
REVF  := $(HERE)/.rev

# next revision (compile may run only once per revision in cuda_exec)
define next_rev
$(shell r=$$(cat $(REVF) 2>/dev/null || echo 0); r=$$((r+1)); echo $$r > $(REVF); echo $$r)
endef

.DEFAULT_GOAL := help
.PHONY: help setup ssh toolchain run run-big triton clean print-env

help:
	@echo "GIST CUDA via cuda_exec harness:"
	@echo "  make setup      ssh egress + full CUDA toolchain + worktree (run once / after reprovision)"
	@echo "  make ssh        (re)establish ssh localhost (cert + sshd bypass; Chef wipes it ~hourly)"
	@echo "  make toolchain  assemble CUDA-13 toolchain (cuda symlink, CCCL, nvml.h, link stubs, worktree)"
	@echo "  make run        compile+trial, SMALL shape (B=8 F=320 D=192 Q=64) — quick correctness"
	@echo "  make run-big    compile+trial, BIG shape (B=1536 F=1491 D=192 Q=128) — vs Triton 4.22ms"
	@echo "  make triton     run the Triton bf16/tf32 benchmark (the number to beat)"
	@echo "  make clean      remove link stubs and the revision counter"

# --- one-time / after-reprovision setup ---
setup: ssh toolchain
	@echo "[setup] done — try: make run"

ssh:
	@bash $(HERE)/ssh_fix.sh

toolchain:
	@bash $(HERE)/setup_toolchain.sh

# --- build + run the kernel through the harness ---
# Each run uses a fresh revision (cuda_exec forbids recompiling a revision).
run:
	@bash $(HERE)/run_gist.sh $(call next_rev) small

run-big:
	@bash $(HERE)/run_gist.sh $(call next_rev) big

# --- comparison baseline (the Triton implementation we must beat) ---
triton:
	@cd $(HERE)/.. && CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench.py 2>&1 | grep -vE "UserWarning|warnings.warn|set_float32|utotun"

clean:
	@rm -rf $(HERE)/linkstubs $(REVF)
	@echo "[clean] removed link stubs + revision counter"

print-env:
	@echo "worktree : $(HOME)/kernel_lab/.worktrees/gist"
	@echo "gist venv: $(HOME)/gist/.venv"
	@echo "kl venv  : $(HOME)/kernel_lab/.venv"
	@echo "cccl     : $(HOME)/cccl/libcudacxx/include"
