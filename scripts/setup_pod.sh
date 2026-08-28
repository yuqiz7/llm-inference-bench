#!/usr/bin/env bash
# Idempotent provisioning for a fresh RunPod pod (H100, runpod/pytorch template).
# Re-run safe: skips work that is already done.
set -euo pipefail

VENVS=/workspace/venvs
ENV_SH=/workspace/env.sh

echo "=== [1/6] apt packages ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git jq tmux curl ca-certificates

echo "=== [2/6] nsys ==="
# Nsight Systems from the preconfigured CUDA apt repo (see docs/env/g0_notes.md).
if ! command -v nsys >/dev/null 2>&1; then
  apt-get install -y -qq nsight-systems-2026.1.3
fi
nsys --version

echo "=== [3/6] env.sh ==="
if [ ! -f "$ENV_SH" ]; then
  cat > "$ENV_SH" <<'EOF'
# Shared environment for llm-inference-bench. Source this from all scripts.
export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
export PATH="$HOME/.local/bin:$PATH"
# Package caches are disposable: keep them on the container overlay disk, not
# the quota-limited /workspace network volume (~80 GB quota).
export UV_CACHE_DIR=/root/.cache/uv
export PIP_CACHE_DIR=/root/.cache/pip
EOF
fi
# shellcheck source=/dev/null
source "$ENV_SH"
mkdir -p "$HF_HOME"

echo "=== [4/6] uv ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
source "$ENV_SH"
uv --version

echo "=== [5/6] venvs ==="
mkdir -p "$VENVS"
for v in vllm sglang bench; do
  if [ ! -x "$VENVS/$v/bin/python" ]; then
    uv venv "$VENVS/$v" --python 3.12
  fi
done

echo "=== [6/6] packages ==="
# Engine versions frozen at the G0-installed versions (docs/env/g0_versions.txt).
uv pip install --python "$VENVS/vllm/bin/python" "vllm==0.28.0"
uv pip install --python "$VENVS/sglang/bin/python" "sglang[all]==0.5.9"
uv pip install --python "$VENVS/bench/bin/python" \
  aiohttp numpy pandas matplotlib transformers datasets \
  "huggingface_hub[cli,hf_transfer]" lm_eval ruff

echo "setup_pod.sh: done"
