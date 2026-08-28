# P3-S-002 (TEMP): /workspace volume quota leaves <8 GB headroom

Status: TEMP workaround, follow-up due **2026-08-31** (before M6/nsys work)

## Problem

The RunPod network volume mounted at /workspace has a ~80 GiB quota (probed
empirically on 2026-08-28; `df`/`statvfs` report cluster-wide space, not the
quota, so the G0 disk guard did not catch it up front). After G0 provisioning
the volume holds: HF models 45 GB + venvs 34 GB (vLLM 15 GB, SGLang 16 GB,
bench 3 GB) ≈ 78 GB used, leaving only ~2 GB spare — below the project's 8 GB
headroom rule.

## What was done (permanent part)

Package caches (`UV_CACHE_DIR`, `PIP_CACHE_DIR`) were moved off the quota
volume to the disposable container overlay disk (`/root/.cache`); the stale
23 GB uv cache on /workspace was deleted. This is a keeper, but it is not
enough headroom on its own.

## TEMP status and follow-up

G0's remaining writes (results, manifests) are KB-scale, so G0 completed on
~2 GB spare. This is NOT viable for later milestones: nsys traces
(multi-GB), FP8 checkpoints, and TensorRT-LLM engines will not fit.

Follow-up (due 2026-08-31, owner: user): resize the RunPod volume to
>= 150 GB (user action — pod/volume changes are billed and out of agent
scope), or approve rebuilding both engine venvs with a shared hardlinked uv
cache on /workspace (~10 GB saving, more fragile). Until resolved, no new
model downloads and no nsys traces written to /workspace.
