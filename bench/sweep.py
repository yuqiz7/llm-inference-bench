"""Concurrency sweep driver: runs bench/client.py cells against a running server.

For each concurrency C in the list: N = max(64, 2*C) capped at 512, with 8
unrecorded warmup requests. Per cell it writes raw jsonl + summary
(results/raw/<cell_id>.jsonl / .summary.json), and an environment manifest
(results/manifests/<cell_id>.json via scripts/env_capture.sh).

Cell ids: <prefix>_c<C, 3 digits>, e.g. w1_m1_vllm_c032.

Acceptance capture for speculative-decoding runs (--capture-accept):
- vllm: snapshot GET <base_url>/metrics before and after the measured phase,
  save every line whose metric name mentions spec/draft/accept to
  results/raw/<cell_id>.metrics.txt, and parse numeric counters into
  results/raw/<cell_id>.accept.json (before/after/delta per counter, plus
  derived ratios where the exposed counters allow — never invented).
- sglang: the server exposes no per-request spec counters on this path;
  capture new server-log lines (--server-log) mentioning accept length written
  during the cell, save them to results/raw/<cell_id>.metrics.txt, and parse
  the reported accept lengths (mean over log lines) into <cell_id>.accept.json.
"""

import argparse
import asyncio
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

import client
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "results" / "raw"

DEFAULT_CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

METRIC_NAME_RE = re.compile(r"(spec|draft|accept)", re.IGNORECASE)
METRIC_LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9.eE+-]+)$")
SGLANG_ACCEPT_RE = re.compile(r"accept[ _-]?len[a-z]*[:=]\s*([0-9.]+)", re.IGNORECASE)


def n_requests_for(concurrency: int) -> int:
    return min(512, max(64, 2 * concurrency))


def fetch_metrics(base_url: str) -> str:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as resp:
        return resp.read().decode("utf-8")


def spec_lines(metrics_text: str) -> list[str]:
    """All lines (incl. HELP/TYPE) whose metric name mentions spec/draft/accept."""
    out = []
    for line in metrics_text.splitlines():
        name = line.split()[2] if line.startswith("#") and len(line.split()) > 2 else (
            line.split("{")[0].split()[0] if line.strip() else ""
        )
        if name and METRIC_NAME_RE.search(name):
            out.append(line)
    return out


def parse_counters(metrics_text: str) -> dict[str, float]:
    """Sum spec/draft/accept-named numeric series across label sets."""
    counters: dict[str, float] = {}
    for line in metrics_text.splitlines():
        if line.startswith("#"):
            continue
        m = METRIC_LINE_RE.match(line.strip())
        if not m or not METRIC_NAME_RE.search(m.group(1)):
            continue
        counters[m.group(1)] = counters.get(m.group(1), 0.0) + float(m.group(3))
    return counters


def derive_acceptance(delta: dict[str, float]) -> dict[str, float]:
    """Ratios derivable from the exposed counter deltas. Missing counters ->
    the corresponding ratio is simply absent (recorded fact, not invented)."""
    derived: dict[str, float] = {}

    def find(*subs: str) -> float | None:
        for name, val in delta.items():
            low = name.lower()
            if all(s in low for s in subs):
                return val
        return None

    accepted = find("accept", "token")
    draft_tokens = find("draft", "token")
    drafts = find("num_drafts")
    emitted = find("emitted", "token")
    if accepted is not None and drafts:
        derived["mean_accepted_tokens_per_step"] = accepted / drafts
    if accepted is not None and draft_tokens:
        derived["acceptance_rate_per_draft_token"] = accepted / draft_tokens
    if emitted is not None and drafts:
        derived["mean_emitted_tokens_per_step"] = emitted / drafts
    return derived


async def run_cell(cfg: dict, tokenizer, nat_prompts) -> dict:
    warmup = cfg.get("warmup_requests", 8)
    print(f"[sweep] {cfg['cell_id']}: warmup {warmup} @ C={cfg['concurrency']}")
    await client.run_phase(cfg, tokenizer, warmup, seed_offset=1_000_000,
                           nat_prompts=nat_prompts)
    print(f"[sweep] {cfg['cell_id']}: measuring {cfg['num_requests']} requests")
    t0 = time.perf_counter()
    records = await client.run_phase(cfg, tokenizer, cfg["num_requests"],
                                     seed_offset=0, nat_prompts=nat_prompts)
    wall_s = time.perf_counter() - t0

    RAW.mkdir(parents=True, exist_ok=True)
    with (RAW / f"{cfg['cell_id']}.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    summary = client.summarize(cfg, records, wall_s)
    (RAW / f"{cfg['cell_id']}.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def capture_accept_vllm(cell_id: str, base_url: str, before_text: str) -> None:
    after_text = fetch_metrics(base_url)
    (RAW / f"{cell_id}.metrics.txt").write_text("\n".join(spec_lines(after_text)) + "\n")
    before = parse_counters(before_text)
    after = parse_counters(after_text)
    delta = {k: after[k] - before.get(k, 0.0) for k in after}
    out = {
        "cell_id": cell_id,
        "source": "vllm /metrics counter deltas (before vs after measured phase; "
                  "warmup requests are included in 'before')",
        "counters_before": before,
        "counters_after": after,
        "counters_delta": delta,
        "derived": derive_acceptance(delta),
    }
    if not after:
        out["note"] = "no spec/draft/accept counters exposed by this server"
    (RAW / f"{cell_id}.accept.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"[sweep] {cell_id}: acceptance capture (vllm): {out.get('derived') or out.get('note')}")


def capture_accept_sglang(cell_id: str, log_path: Path, offset: int) -> None:
    text = log_path.read_bytes()[offset:].decode("utf-8", errors="replace")
    lines = [l for l in text.splitlines() if SGLANG_ACCEPT_RE.search(l)]
    (RAW / f"{cell_id}.metrics.txt").write_text("\n".join(lines) + "\n")
    vals = [float(m.group(1)) for l in lines for m in [SGLANG_ACCEPT_RE.search(l)] if m]
    out: dict = {
        "cell_id": cell_id,
        "source": f"sglang server log lines mentioning accept length "
                  f"({log_path}), window = this cell (warmup + measured)",
        "n_log_lines": len(lines),
    }
    if vals:
        out["mean_accept_length_reported"] = sum(vals) / len(vals)
        out["min_accept_length_reported"] = min(vals)
        out["max_accept_length_reported"] = max(vals)
        out["note"] = ("mean over per-batch log-reported accept lengths; "
                       "not request-weighted")
    else:
        out["note"] = "no accept-length lines appeared in the server log for this cell"
    (RAW / f"{cell_id}.accept.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"[sweep] {cell_id}: acceptance capture (sglang): {out['note']}, "
          f"n={len(lines)}")


def write_manifest(cell_id: str, engine: str, launch_cmd: str) -> None:
    subprocess.run(
        [str(REPO_ROOT / "scripts" / "env_capture.sh"), cell_id, engine, launch_cmd],
        check=True,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="base cell config yaml")
    parser.add_argument("--cell-prefix", required=True, help="e.g. w1_m1_vllm")
    parser.add_argument(
        "--concurrencies",
        default=",".join(str(c) for c in DEFAULT_CONCURRENCIES),
        help="comma-separated concurrency list",
    )
    parser.add_argument("--launch-cmd", default="",
                        help="server launch command recorded in manifests")
    parser.add_argument("--capture-accept", action="store_true",
                        help="capture speculative-decoding acceptance per cell")
    parser.add_argument("--server-log", default=None,
                        help="sglang server log path (for --capture-accept)")
    args = parser.parse_args()

    base = yaml.safe_load(Path(args.config).read_text())
    engine = base["engine"]
    concurrencies = [int(c) for c in args.concurrencies.split(",")]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base["model"])
    nat_prompts = (
        client.load_nat_prompts() if base.get("workload") == "nat" else None
    )

    for c in concurrencies:
        cfg = dict(base)
        cfg["concurrency"] = c
        cfg["num_requests"] = n_requests_for(c)
        cfg["cell_id"] = f"{args.cell_prefix}_c{c:03d}"

        before_text = ""
        log_offset = 0
        if args.capture_accept and engine == "vllm":
            before_text = fetch_metrics(cfg["base_url"])
        if args.capture_accept and engine == "sglang" and args.server_log:
            log_offset = Path(args.server_log).stat().st_size

        summary = await run_cell(cfg, tokenizer, nat_prompts)
        print(f"[sweep] {cfg['cell_id']}: {summary['output_tok_per_s']:.1f} tok/s, "
              f"TTFT p50 {summary['ttft_p50_ms']:.1f} ms, "
              f"TPOT p50 {summary['tpot_p50_ms']:.2f} ms")

        if args.capture_accept and engine == "vllm":
            capture_accept_vllm(cfg["cell_id"], cfg["base_url"], before_text)
        if args.capture_accept and engine == "sglang" and args.server_log:
            capture_accept_sglang(cfg["cell_id"], Path(args.server_log), log_offset)

        write_manifest(cfg["cell_id"], engine, args.launch_cmd)

    print(f"[sweep] done: {len(concurrencies)} cells, prefix {args.cell_prefix}")


if __name__ == "__main__":
    asyncio.run(main())
