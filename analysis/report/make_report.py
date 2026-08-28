"""Regenerate the GEN blocks in README.md from results/ files.

Every number in the README comes from results/raw and results/manifests via
this script — never hand-typed. --check regenerates to a temporary copy and
exits nonzero if the committed README has drifted.

Stdlib-only so CI can run it without installing project dependencies.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
README = REPO_ROOT / "README.md"
MANIFESTS = REPO_ROOT / "results" / "manifests"
RAW = REPO_ROOT / "results" / "raw"

G0_SMOKE_CELL = "g0_smoke_vllm_qwen3-8b"


def gen_env() -> str:
    manifests = sorted(MANIFESTS.glob("*.json"))
    if not manifests:
        return "_No manifests captured yet._\n"
    lines = [
        "| Cell | GPU | Driver | Engine | Version |",
        "|---|---|---|---|---|",
    ]
    for path in manifests:
        m = json.loads(path.read_text())
        lines.append(
            f"| {m['cell_id']} | {m['gpu']} | {m['driver']} "
            f"| {m['engine']} | {m['engine_version']} |"
        )
    lines.append("")
    lines.append(
        "Source: `results/manifests/*.json`, "
        "rendered by `analysis/report/make_report.py`."
    )
    return "\n".join(lines) + "\n"


def gen_g0_smoke() -> str:
    summary_path = RAW / f"{G0_SMOKE_CELL}.summary.json"
    if not summary_path.exists():
        return "_No G0 smoke results yet._\n"
    s = json.loads(summary_path.read_text())
    lines = [
        (
            f"Engine: **{s['engine']}**, model: **{s['model']}**, "
            f"random workload {s['input_len']} in / {s['output_len']} out, "
            "greedy, ignore_eos."
        ),
        "",
        (
            "| Concurrency | N | TTFT p50 (ms) | TTFT p95 (ms) "
            "| TPOT p50 (ms) | TPOT p95 (ms) | Output tok/s |"
        ),
        "|---|---|---|---|---|---|---|",
        (
            f"| {s['concurrency']} | {s['num_requests']} "
            f"| {s['ttft_p50_ms']:.1f} | {s['ttft_p95_ms']:.1f} "
            f"| {s['tpot_p50_ms']:.1f} | {s['tpot_p95_ms']:.1f} "
            f"| {s['output_tok_per_s']:.1f} |"
        ),
        "",
        (
            f"Source: `results/raw/{G0_SMOKE_CELL}.summary.json` "
            f"(per-request data: `results/raw/{G0_SMOKE_CELL}.jsonl`), "
            "rendered by `analysis/report/make_report.py`."
        ),
    ]
    return "\n".join(lines) + "\n"


def replace_block(text: str, name: str, content: str) -> str:
    start = f"<!-- GEN:{name} -->"
    end = f"<!-- /GEN:{name} -->"
    i = text.index(start) + len(start)
    j = text.index(end)
    return text[:i] + "\n" + content + text[j:]


def render(text: str) -> str:
    text = replace_block(text, "env", gen_env())
    text = replace_block(text, "g0_smoke", gen_g0_smoke())
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail (exit 1) if README.md is out of date instead of writing it",
    )
    args = parser.parse_args()

    current = README.read_text()
    rendered = render(current)
    if args.check:
        if rendered != current:
            print("README.md is out of date; run analysis/report/make_report.py")
            return 1
        print("README.md is up to date")
        return 0
    README.write_text(rendered)
    print("README.md regenerated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
