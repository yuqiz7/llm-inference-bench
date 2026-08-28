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

(to be filled during the run)

## M5 — TP=2 scaling

(to be filled during the run)
