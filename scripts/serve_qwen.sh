#!/usr/bin/env bash
# Launch the vLLM OpenAI-compatible server for the Qwen3-4B planner (RESEARCH_PLAN
# Phase 1). Serves on :8000; the harness's VLLMPolicy is an HTTP client to it.
#
#   bash scripts/serve_qwen.sh            # foreground (Ctrl-C to stop)
#
# Model weights are pre-cached on the volume (HF_HOME below); HF_HUB_OFFLINE avoids
# the sandbox's huggingface.co egress gate. The venv lives at /workspace/envs/vllm.
set -eu
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_OFFLINE=1          # model is pre-cached on the volume
export TRANSFORMERS_OFFLINE=1    # ditto — no hub round-trips
export VLLM_NO_USAGE_STATS=1     # no telemetry POST to stats.vllm.ai
export DO_NOT_TRACK=1            # belt-and-suspenders for telemetry
export PYTHONUNBUFFERED=1        # so the startup log is visible in real time
export VLLM_LOGGING_LEVEL=INFO
# FlashInfer's prebuilt kernels can't detect Blackwell sm_120 (wants CUDA>=12.9 cubins)
# and wrongly throw "requires sm75" — for attention AND the top-k/top-p sampler. We
# uninstalled flashinfer entirely; these force vLLM's native/Triton paths regardless.
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
PORT=${PORT:-8000}

exec /workspace/envs/vllm/bin/vllm serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16 \
  --enforce-eager
