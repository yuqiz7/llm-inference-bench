# P3-S-001: Project scope and benchmark matrix

Status: accepted · Date: 2026-08-28

## Scope

Benchmark LLM inference engines — vLLM, SGLang, and TensorRT-LLM — on a single
H100 SXM (plus one TP=2 configuration), measuring serving throughput and
latency (TTFT/TPOT percentiles, output tok/s) under a closed-loop streaming
client, and the accuracy cost of quantization. Models: Qwen/Qwen3-8B (dense)
and deepseek-ai/DeepSeek-V2-Lite-Chat (MoE / MLA). Out of scope for v1:
multi-node serving, engines beyond the three above, models beyond the two
above, and cost modeling.

## Matrix families

- **M1 — Baseline throughput/latency.** BF16, engine defaults, both models,
  concurrency sweep, fixed random workload (input/output lengths controlled).
  All three engines.
- **M2 — FP8 weight quantization.** M1's best cells re-run with FP8 weights;
  throughput/latency deltas.
- **M3 — FP8 KV-cache quantization + accuracy regression.** KV-cache FP8 on
  top of M2; GSM8K and MMLU via lm_eval against each served engine, compared
  to the BF16 baseline.
- **M4 — Speculative decoding.** n-gram and EAGLE-3 drafting on supported
  engines; acceptance-length analysis vs. workload.
- **M5 — TP=2 scaling.** M1 subset on 2x H100; scaling efficiency per engine.
- **M6 — Kernel attribution.** Nsight Systems traces of representative cells;
  time attribution to attention / GEMM / comms / host gaps.

## v1 freeze

v1 is frozen when: M1–M6 each have at least one completed, manifest-backed
cell per applicable engine; every README number regenerates from results/ via
`make_report.py --check` in CI; and each family has a short written finding.
After freeze, new work goes to v2; v1 numbers are only corrected, not extended.

## Fairness rules

- Same model weights, workload generator, sampling parameters, and
  `max_tokens` across engines in any compared cell.
- Engine defaults; no per-engine tuning. Every launch flag is recorded in
  `results/manifests/<cell_id>.json` by `scripts/env_capture.sh`.
- One engine process per run; server restarted between cells; warmup requests
  excluded from measurement.
- Comparisons only within identical (model, workload, concurrency) cells.
