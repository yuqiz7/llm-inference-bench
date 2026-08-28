# G0 environment notes (2026-08-28)

## nsys check: PASS

- Not present on the fresh pod; installed `nsight-systems-2026.1.3` from the
  preconfigured CUDA apt repo (developer.download.nvidia.com, ubuntu2404).
  `nsys --version` → 2026.1.3.425.
- Validation: 5 s CUDA trace (`nsys profile -t cuda`) of a bf16 4096x4096
  torch matmul loop on the H100, then `nsys stats --report cuda_gpu_kern_sum`.
  Trace captured ~25k `nvjet_tst_256x128_...` GEMM kernel instances; stats
  report rendered correctly. PASS.
- `nsys` binary: /usr/local/bin/nsys (alternatives-managed). Re-provisioning a
  fresh pod requires re-running `apt-get install nsight-systems-2026.1.3`
  (not yet in setup_pod.sh since profiling starts 2026-08-31).

## /workspace quota incident (resolved)

- First `setup_pod.sh` run failed installing SGLang: `Quota exceeded (os
  error 122)` extracting wheels into `/workspace/.cache/uv` (the RunPod
  template points all package caches at /workspace). The network volume has a
  per-volume quota (~80 GB; not visible via statvfs) and models (46 GB) +
  venvs + a 23 GB wheel cache exceeded it.
- Fix (permanent, in setup_pod.sh + env.sh): `UV_CACHE_DIR` and
  `PIP_CACHE_DIR` moved to the container overlay disk (`/root/.cache`, 49 GB,
  disposable), old /workspace uv cache deleted, install rerun. Caches are
  rebuildable, so losing them on pod recycle is the intended trade-off.
- Budget on the quota volume: models 46 GB + venvs ~17 GB + repo ≈ 64 GB,
  leaving >8 GB spare under the ~80 GB quota.

## Port 8001 conflict (resolved)

- The planned SGLang port 8001 is already bound by the RunPod template's
  nginx (README proxy), which answers `/health` with 200 and masked the
  failed launch. SGLang moved to port 30000 (default in
  `scripts/serve_sglang.sh`); vLLM keeps 8000, which is free.

## Filesystem layout

- /workspace is a large shared network filesystem (MooseFS). Venvs
  (/workspace/venvs) and HF cache (/workspace/hf) live there; container
  overlay / has only 50 GB.
