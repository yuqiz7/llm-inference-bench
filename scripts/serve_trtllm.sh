#!/usr/bin/env bash
# Launch trtllm-serve (OpenAI-compatible, PyTorch backend — no offline engine
# build) and wait for health. Runs against the NGC TensorRT-LLM container's
# system install; do NOT activate a venv here.
# Usage: serve_trtllm.sh [MODEL] [extra trtllm-serve serve flags...]
set -euo pipefail
source /workspace/env.sh

# The container's OpenMPI (hpcx) is relocated; without OPAL_PREFIX MPI_Init
# aborts on import of tensorrt_llm (PID 1 has these, login shells do not).
export OPAL_PREFIX=/opt/hpcx/ompi
export OMPI_MCA_coll_hcoll_enable=0

MODEL="${1:-Qwen/Qwen3-8B}"
[ $# -gt 0 ] && shift
PORT="${PORT:-8000}"
LOG="${LOG:-/workspace/logs/trtllm_server.log}"
mkdir -p "$(dirname "$LOG")"

nohup trtllm-serve serve "$MODEL" --backend pytorch \
  --host 127.0.0.1 --port "$PORT" "$@" > "$LOG" 2>&1 &
PID=$!
echo "trtllm-serve launching: model=$MODEL port=$PORT pid=$PID log=$LOG"

# First startup can JIT-compile kernels; allow up to 20 min.
for _ in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
    echo "trtllm-serve ready: pid=$PID"
    exit 0
  fi
  if ! kill -0 "$PID" 2> /dev/null; then
    echo "trtllm-serve process died; last log lines:" >&2
    tail -n 40 "$LOG" >&2
    exit 1
  fi
  sleep 5
done
echo "trtllm-serve health timeout after 20 min" >&2
exit 1
