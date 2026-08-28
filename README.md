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
_No manifests captured yet._
<!-- /GEN:env -->

## G0 smoke benchmark

Single cell, engine defaults, config in
[configs/g0_smoke.yaml](configs/g0_smoke.yaml). Not a tuned or comparative
result — a pipeline check only.

<!-- GEN:g0_smoke -->
_No G0 smoke results yet._
<!-- /GEN:g0_smoke -->

## Reproducing

```bash
bash scripts/setup_pod.sh
bash scripts/serve_vllm.sh Qwen/Qwen3-8B
source /workspace/venvs/bench/bin/activate
python bench/client.py --config configs/g0_smoke.yaml
python analysis/report/make_report.py
```
