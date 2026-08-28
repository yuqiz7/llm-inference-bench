#!/usr/bin/env bash
# Accuracy regression via lm-eval against a running OpenAI-compatible server.
# Usage: run_accuracy.sh <config_name e.g. vllm_bf16> [base_url]
# Writes results/accuracy/w1_acc_<config>_{gsm8k,mmlu}.json plus the exact
# command lines to results/accuracy/w1_acc_<config>.cmd.txt.
set -euo pipefail
source /workspace/env.sh

CONFIG="$1"
BASE_URL="${2:-http://127.0.0.1:8000}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO_ROOT/results/accuracy"
mkdir -p "$OUT"
LM_EVAL=/workspace/venvs/bench/bin/lm_eval
CMDFILE="$OUT/w1_acc_${CONFIG}.cmd.txt"
: > "$CMDFILE"

run_task () {  # <task> <extra args...>
  local task="$1"; shift
  local outdir="$OUT/raw_${CONFIG}_${task}"
  local cmd=("$LM_EVAL" --model local-completions
    --model_args "model=$MODEL,base_url=$BASE_URL/v1/completions,num_concurrent=32,max_retries=3,tokenizer=$MODEL"
    --tasks "$task" "$@" --output_path "$outdir" --seed 1234)
  echo "${cmd[*]}" >> "$CMDFILE"
  "${cmd[@]}"
  # lm-eval writes <outdir>/<sanitized model>/results_<timestamp>.json
  local latest
  latest="$(find "$outdir" -name 'results_*.json' | sort | tail -1)"
  cp "$latest" "$OUT/w1_acc_${CONFIG}_${task%%_*}.json"
  echo "copied $latest -> $OUT/w1_acc_${CONFIG}_${task%%_*}.json"
}

run_task gsm8k --num_fewshot 5 --limit 500
run_task mmlu --limit 20
