#!/usr/bin/env bash
# Capture the run environment for one benchmark cell.
# Usage: env_capture.sh <cell_id> <engine: vllm|sglang> <launch command...>
set -euo pipefail
source /workspace/env.sh

CELL_ID="$1"
ENGINE="$2"
shift 2
LAUNCH_CMD="$*"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$REPO_ROOT/results/manifests"
mkdir -p "$OUT_DIR"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
ENGINE_VERSION="$(/workspace/venvs/$ENGINE/bin/python - "$ENGINE" <<'EOF'
import importlib, sys
print(importlib.import_module(sys.argv[1]).__version__)
EOF
)"

jq -n \
  --arg cell_id "$CELL_ID" \
  --arg gpu "$GPU_NAME" \
  --arg driver "$DRIVER" \
  --arg engine "$ENGINE" \
  --arg engine_version "$ENGINE_VERSION" \
  --arg launch_cmd "$LAUNCH_CMD" \
  --arg captured_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{cell_id: $cell_id, gpu: $gpu, driver: $driver, engine: $engine,
    engine_version: $engine_version, launch_cmd: $launch_cmd,
    captured_at: $captured_at}' \
  > "$OUT_DIR/$CELL_ID.json"
echo "wrote $OUT_DIR/$CELL_ID.json"
