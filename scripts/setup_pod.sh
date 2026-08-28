#!/usr/bin/env bash
# Idempotent provisioning for a fresh RunPod pod (H100, runpod/pytorch template).
# Re-run safe: skips work that is already done.
set -euo pipefail

VENVS=/workspace/venvs
ENV_SH=/workspace/env.sh

echo "=== [1/5] apt packages ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git jq tmux curl ca-certificates

echo "=== [2/5] env.sh ==="
if [ ! -f "$ENV_SH" ]; then
  cat > "$ENV_SH" <<'EOF'
# Shared environment for llm-inference-bench. Source this from all scripts.
export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
export PATH="$HOME/.local/bin:$PATH"
EOF
fi
# shellcheck source=/dev/null
source "$ENV_SH"
mkdir -p "$HF_HOME"

echo "=== [3/5] uv ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
source "$ENV_SH"
uv --version

echo "=== [4/5] venvs ==="
mkdir -p "$VENVS"
for v in vllm sglang bench; do
  if [ ! -x "$VENVS/$v/bin/python" ]; then
    uv venv "$VENVS/$v" --python 3.12
  fi
done

echo "=== [5/5] packages ==="
uv pip install --python "$VENVS/vllm/bin/python" vllm
uv pip install --python "$VENVS/sglang/bin/python" "sglang[all]"
uv pip install --python "$VENVS/bench/bin/python" \
  aiohttp numpy pandas matplotlib transformers \
  "huggingface_hub[cli,hf_transfer]" lm_eval ruff

echo "setup_pod.sh: done"
