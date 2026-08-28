#!/usr/bin/env bash
# Launch an SGLang OpenAI-compatible server and wait for health.
# Usage: serve_sglang.sh [MODEL] [extra launch_server flags...]
# (DeepSeek-V2-Lite-Chat later needs --trust-remote-code.)
set -euo pipefail
source /workspace/env.sh

MODEL="${1:-Qwen/Qwen3-8B}"
[ $# -gt 0 ] && shift
# Not 8001: the RunPod template's nginx (README proxy) already listens there
# and answers health checks, masking a failed bind.
PORT="${PORT:-30000}"
LOG="${LOG:-/workspace/logs/sglang_server.log}"
mkdir -p "$(dirname "$LOG")"

source /workspace/venvs/sglang/bin/activate
nohup python -m sglang.launch_server --model-path "$MODEL" --port "$PORT" "$@" \
  > "$LOG" 2>&1 &
PID=$!
echo "SGLang launching: model=$MODEL port=$PORT pid=$PID log=$LOG"

for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
    echo "SGLang ready: pid=$PID"
    exit 0
  fi
  if ! kill -0 "$PID" 2> /dev/null; then
    echo "SGLang server process died; last log lines:" >&2
    tail -n 40 "$LOG" >&2
    exit 1
  fi
  sleep 5
done
echo "SGLang health timeout after 15 min" >&2
exit 1
