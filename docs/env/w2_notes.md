# W2 environment / decision notes (2026-08-28)

## Task 0 — provisioning (fresh 2x H100 SXM pod)

- Fresh pod, fresh /workspace volume (nothing survived from the W1 pod —
  the W1-downloaded DeepSeek-V2-Lite-Chat cache was on the other volume).
  `setup_pod.sh` re-run from scratch; DeepSeek-V2-Lite-Chat re-downloaded
  via hf_transfer.
- `docs/env/w2_versions.txt` matches `w1_versions.txt` exactly for every
  pinned extract (vllm/sglang/bench venvs, driver 580.126.09, CUDA 12.8,
  nsys 2026.1.3). No drift.
- This pod has 2x H100 80GB (TP=2 for M5); W1 pods had 1x. M4 runs with
  `CUDA_VISIBLE_DEVICES=0` so its TP=1 cells are single-GPU like W1.

## M4 — MoE (DeepSeek-V2-Lite-Chat, TP=1)

- Both engines serve the model on the pinned versions with only
  `--trust-remote-code` added; otherwise defaults, same fairness rules as
  W1 M1 (random 1024in/256out, greedy, ignore_eos, N = min(512, max(64, 2C))).
- vLLM 0.28.0 uses MLA attention for this model
  (`vllm::unified_mla_attention_with_output` in the compile splitting ops)
  and reports dtype=torch.bfloat16, max_seq_len=163840.
- Note on TPOT at C=1: DeepSeek-V2-Lite has ~2.4B active params (16B total),
  so low-concurrency decode is much faster than the dense Qwen3-8B baseline
  (vLLM TPOT p50 3.30 ms at C=1 vs 6.77 ms for Qwen3-8B in W1).

### FP8 KV cache on MLA (design-doc answer, vLLM-only probe)

- **Supported on vLLM 0.28.0.** `--kv-cache-dtype fp8` with
  DeepSeek-V2-Lite-Chat starts cleanly and serves; server log:
  `Using fp8 data type to store kv cache. It reduces the GPU memory
  footprint and boosts the performance. Meanwhile, it may cause accuracy
  drop without a proper scaling factor` and the engine config records
  `kv_cache_dtype=fp8`.
- Single probe cell `w2_m4_vllm_fp8kv_c032`: 2826.7 output tok/s
  (TTFT p50 453.3 ms, TPOT p50 9.46 ms) vs BF16-KV `w2_m4_vllm_c032`
  3383.2 tok/s (TTFT p50 79.1 ms, TPOT p50 9.03 ms) — the probe cell ran
  *slower* than BF16 KV at this level, with a much higher TTFT. Probe only
  (one cell, one run, boxed at 15 min per plan); not swept, no accuracy
  check, so treat as "works, no free win at C=32", not as a tuned result.

## M5 — TP=2 scaling

(to be filled during the run)
