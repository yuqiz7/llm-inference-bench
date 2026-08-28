#!/usr/bin/env bash
# Stop the running engine server and verify GPU memory is released.
# Usage: stop_server.sh <vllm|sglang>
set -euo pipefail

ENGINE="$1"
case "$ENGINE" in
  vllm)   PATTERN="vllm serve" ;;
  sglang) PATTERN="sglang.launch_server" ;;
  *) echo "unknown engine: $ENGINE" >&2; exit 2 ;;
esac

pkill -f "$PATTERN" 2>/dev/null || true
for i in $(seq 1 60); do
  # any engine process left?
  if ! pgrep -f "$PATTERN" >/dev/null 2>&1; then
    USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
    if [ "$USED" -lt 500 ]; then
      echo "stop_server.sh: $ENGINE stopped, GPU memory ${USED} MiB"
      exit 0
    fi
  fi
  [ "$i" -eq 20 ] && pkill -9 -f "$PATTERN" 2>/dev/null || true
  sleep 2
done
echo "stop_server.sh: $ENGINE did not release GPU memory in 120 s" >&2
nvidia-smi >&2
exit 1
