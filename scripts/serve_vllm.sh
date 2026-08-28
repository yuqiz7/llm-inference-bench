#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server and wait for health.
# Usage: serve_vllm.sh [MODEL] [extra vllm serve flags...]
# (DeepSeek-V2-Lite-Chat later needs --trust-remote-code.)
set -euo pipefail
source /workspace/env.sh

MODEL="${1:-Qwen/Qwen3-8B}"
[ $# -gt 0 ] && shift
PORT="${PORT:-8000}"
LOG="${LOG:-/workspace/logs/vllm_server.log}"
mkdir -p "$(dirname "$LOG")"

source /workspace/venvs/vllm/bin/activate
nohup vllm serve "$MODEL" --port "$PORT" "$@" > "$LOG" 2>&1 &
PID=$!
echo "vLLM launching: model=$MODEL port=$PORT pid=$PID log=$LOG"

for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
    echo "vLLM ready: pid=$PID"
    exit 0
  fi
  if ! kill -0 "$PID" 2> /dev/null; then
    echo "vLLM server process died; last log lines:" >&2
    tail -n 40 "$LOG" >&2
    exit 1
  fi
  sleep 5
done
echo "vLLM health timeout after 15 min" >&2
exit 1
