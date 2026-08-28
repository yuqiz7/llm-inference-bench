# llm-inference-bench

End-to-end benchmark of LLM inference engines (vLLM / SGLang / TensorRT-LLM) on
NVIDIA H100: throughput and latency under continuous batching and
PagedAttention, FP8 weight and KV-cache quantization with accuracy regression
(GSM8K / MMLU), speculative decoding (n-gram / EAGLE-3) acceptance analysis,
tensor-parallel (TP=2) scaling, and kernel-level attribution with Nsight
Systems. Fairness rules and the full matrix are defined in
[docs/decisions/p3s-001-scope.md](docs/decisions/p3s-001-scope.md).

## Status

### DONE (G0, 2026-08-28)

- Pod provisioning script ([scripts/setup_pod.sh](scripts/setup_pod.sh)):
  uv-managed venvs for vLLM, SGLang, and the bench client.
- Models downloaded: Qwen/Qwen3-8B and deepseek-ai/DeepSeek-V2-Lite-Chat.
- Closed-loop streaming benchmark client ([bench/client.py](bench/client.py)).
- First smoke benchmark: vLLM + Qwen3-8B, one cell (results below).
- SGLang sanity check: server launched, one completion returned HTTP 200.

### PLANNED

- Benchmark matrix M1–M6 (baselines, FP8 weights, FP8 KV cache + accuracy,
  speculative decoding, TP=2 scaling, Nsight attribution) — see the scope doc.
- TensorRT-LLM engine integration.
- Nsight Systems profiling (nsys) runs.

## Environment

<!-- GEN:env -->
| Cell | GPU | Driver | Engine | Version |
|---|---|---|---|---|
| g0_sanity_sglang_qwen3-8b | NVIDIA H100 80GB HBM3 | 580.126.09 | sglang | 0.5.9 |
| g0_smoke_vllm_qwen3-8b | NVIDIA H100 80GB HBM3 | 580.126.09 | vllm | 0.28.0 |

Source: `results/manifests/*.json`, rendered by `analysis/report/make_report.py`.
<!-- /GEN:env -->

## G0 smoke benchmark

Single cell, engine defaults, config in
[configs/g0_smoke.yaml](configs/g0_smoke.yaml). Not a tuned or comparative
result — a pipeline check only.

<!-- GEN:g0_smoke -->
Engine: **vllm**, model: **Qwen/Qwen3-8B**, random workload 1024 in / 256 out, greedy, ignore_eos.

| Concurrency | N | TTFT p50 (ms) | TTFT p95 (ms) | TPOT p50 (ms) | TPOT p95 (ms) | Output tok/s |
|---|---|---|---|---|---|---|
| 8 | 32 | 245.4 | 271.7 | 7.1 | 7.6 | 991.6 |

Source: `results/raw/g0_smoke_vllm_qwen3-8b.summary.json` (per-request data: `results/raw/g0_smoke_vllm_qwen3-8b.jsonl`), rendered by `analysis/report/make_report.py`.
<!-- /GEN:g0_smoke -->

## Reproducing

```bash
bash scripts/setup_pod.sh
bash scripts/serve_vllm.sh Qwen/Qwen3-8B
source /workspace/venvs/bench/bin/activate
python bench/client.py --config configs/g0_smoke.yaml
python analysis/report/make_report.py
```
