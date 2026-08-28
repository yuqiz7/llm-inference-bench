# W1 environment / decision notes (2026-08-28)

## Task 0 — re-provision

- `setup_pod.sh` now installs `nsight-systems-2026.1.3` (step 2/6) and pins
  `vllm==0.28.0`, `sglang[all]==0.5.9` — the G0-installed versions.
- `docs/env/w1_versions.txt` matches `g0_versions.txt` exactly for all pinned
  extracts; the only addition is `datasets==5.0.1` in the bench venv, added
  for the new "nat" workload (GSM8K question texts). No drift.
- Models re-downloaded to /workspace/hf: Qwen3-8B (16G), DeepSeek-V2-Lite-Chat
  (30G, for the W2 MoE pod — downloaded now while the GPU was idle).

## Harness

- `bench/sweep.py`: per concurrency C in {1,2,4,8,16,32,64,128,256},
  N = min(512, max(64, 2C)), 8 unrecorded warmups, cell ids `<prefix>_c<C:03d>`.
- Workload `nat` = GSM8K test-split questions via the `datasets` library
  (openai/gsm8k, main, 1319 questions), cycled deterministically by seed.
  `random` workload unchanged from G0.

## M2 FP8 — route and observations

- `lm_eval` needed the `[api]` extra (`tenacity`) to use `local-completions`;
  fixed permanently in setup_pod.sh (`lm_eval[api]`). Not a version change.
- MMLU loglikelihood DID work against the server API (vLLM /v1/completions
  with echo+logprobs): 57 subtasks x 20 = 1140 samples per config. No
  gsm8k-only fallback needed.
- SGLang FP8 weights: online quantization (`--quantization fp8`) works on the
  pinned 0.5.9 (server_args confirms `quantization='fp8'`), so the official
  Qwen3-8B-FP8 checkpoint route was not needed and nothing extra was
  downloaded.
- Fairness note: both engines run with their defaults, which include prefix
  caching (vLLM `enable_prefix_caching=True`, SGLang radix cache on). The
  random workload has no cross-request prefix overlap, so this does not
  inflate M1/M2 numbers.

## M3 speculative decoding — configuration decisions

- vLLM ngram: vLLM 0.28.0 exposes no built-in default knobs
  (`prompt_lookup_max/min` and `num_speculative_tokens` are `None`/required),
  so we use the documented example values from vLLM's speculative-decoding
  docs: `{"method": "ngram", "num_speculative_tokens": 3,
  "prompt_lookup_max": 4, "prompt_lookup_min": 2}`. Depth 3 matches the
  EAGLE-3 arm for comparability.
- SGLang EAGLE3 chain config (conservative, for cross-batch comparability):
  `--speculative-num-steps 3 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4` with draft
  `Tengyunw/qwen3_8b_eagle3`. The model card's aggressive bs=1 config
  (steps 6, topk 10, draft tokens 32) was NOT run — it targets single-request
  latency and would confound the batch sweep.
- SGLang 0.5.9 *does* expose an ngram mode (`--speculative-algorithm NGRAM`
  per `--help`), so an SGLang ngram arm is included in M3.
