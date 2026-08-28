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

## Filesystem layout

- /workspace is a large shared network filesystem (MooseFS). Venvs
  (/workspace/venvs) and HF cache (/workspace/hf) live there; container
  overlay / has only 50 GB.
