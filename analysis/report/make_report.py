"""Regenerate the GEN blocks in README.md (and figures) from results/ files.

Every number in the README comes from results/raw, results/manifests, and
results/accuracy via this script — never hand-typed. --check regenerates to a
temporary copy and exits nonzero if the committed README has drifted.

Stdlib-only for the README text so CI can run --check without project deps.
Figures (docs/figures/*.png) are regenerated only when matplotlib is
importable (run in the bench venv); --check does not depend on figures.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
README = REPO_ROOT / "README.md"
MANIFESTS = REPO_ROOT / "results" / "manifests"
RAW = REPO_ROOT / "results" / "raw"
ACCURACY = REPO_ROOT / "results" / "accuracy"
FIGURES = REPO_ROOT / "docs" / "figures"

G0_SMOKE_CELL = "g0_smoke_vllm_qwen3-8b"

M1_SERIES = [("vllm", "w1_m1_vllm"), ("sglang", "w1_m1_sglang")]
M2_SERIES = [
    ("vLLM BF16 (M1)", "w1_m1_vllm"),
    ("vLLM FP8 weights", "w1_m2_vllm_fp8w"),
    ("vLLM FP8 KV cache", "w1_m2_vllm_fp8kv"),
    ("vLLM FP8 weights+KV", "w1_m2_vllm_fp8wkv"),
    ("SGLang FP8 weights", "w1_m2_sglang_fp8w"),
]
ACC_CONFIGS = [
    ("vLLM BF16", "w1_acc_vllm_bf16"),
    ("vLLM FP8 weights", "w1_acc_vllm_fp8w"),
    ("vLLM FP8 weights+KV", "w1_acc_vllm_fp8wkv"),
]
M3_SERIES = {
    "vllm": [("baseline", "w1_m3_vllm_base"), ("ngram", "w1_m3_vllm_ngram"),
             ("EAGLE-3", "w1_m3_vllm_eagle3")],
    "sglang": [("baseline", "w1_m3_sglang_base"), ("EAGLE-3", "w1_m3_sglang_eagle3")],
}


def load_series(prefix: str) -> dict[int, dict]:
    """{concurrency: summary} for all results/raw/<prefix>_c*.summary.json."""
    out: dict[int, dict] = {}
    for path in sorted(RAW.glob(f"{prefix}_c*.summary.json")):
        s = json.loads(path.read_text())
        out[int(s["concurrency"])] = s
    return out


def load_accept(prefix: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in sorted(RAW.glob(f"{prefix}_c*.accept.json")):
        a = json.loads(path.read_text())
        c = int(a["cell_id"].rsplit("_c", 1)[1])
        out[c] = a
    return out


def accept_value(a: dict | None) -> str:
    """Mean accepted tokens per step, only where the capture exposed it."""
    if a is None:
        return "—"
    derived = a.get("derived") or {}
    if "mean_accepted_tokens_per_step" in derived:
        return f"{derived['mean_accepted_tokens_per_step']:.2f}"
    if "mean_accept_length_reported" in a:
        return f"{a['mean_accept_length_reported']:.2f}"
    return "not exposed"


def gen_env() -> str:
    manifests = sorted(MANIFESTS.glob("*.json"))
    if not manifests:
        return "_No manifests captured yet._\n"
    # Grouped: one row per (gpu, driver, engine, version); per-cell manifests
    # (incl. exact launch commands) live in results/manifests/.
    groups: dict[tuple, int] = {}
    for path in manifests:
        m = json.loads(path.read_text())
        key = (m["gpu"], m["driver"], m["engine"], m["engine_version"])
        groups[key] = groups.get(key, 0) + 1
    lines = [
        "| GPU | Driver | Engine | Version | Cells |",
        "|---|---|---|---|---|",
    ]
    for (gpu, driver, engine, ver), n in sorted(groups.items(), key=lambda kv: kv[0][2:]):
        lines.append(f"| {gpu} | {driver} | {engine} | {ver} | {n} |")
    lines.append("")
    lines.append(
        "Source: `results/manifests/*.json` (per-cell launch commands there), "
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


def series_table(series: dict[int, dict]) -> list[str]:
    lines = [
        (
            "| Concurrency | N | TTFT p50 (ms) | TTFT p95 (ms) "
            "| TPOT p50 (ms) | TPOT p95 (ms) | Output tok/s |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for c in sorted(series):
        s = series[c]
        lines.append(
            f"| {c} | {s['num_requests']} "
            f"| {s['ttft_p50_ms']:.1f} | {s['ttft_p95_ms']:.1f} "
            f"| {s['tpot_p50_ms']:.2f} | {s['tpot_p95_ms']:.2f} "
            f"| {s['output_tok_per_s']:.1f} |"
        )
    return lines


def gen_w1_m1() -> str:
    blocks = []
    any_data = False
    for engine, prefix in M1_SERIES:
        series = load_series(prefix)
        if not series:
            continue
        any_data = True
        blocks.append(f"**{engine}** (defaults):")
        blocks.append("")
        blocks.extend(series_table(series))
        blocks.append("")
    if not any_data:
        return "_No M1 results yet._\n"
    blocks.append("![Throughput vs concurrency](docs/figures/fig1_throughput_concurrency.png)")
    blocks.append("")
    blocks.append("![Latency vs concurrency](docs/figures/fig2_latency_concurrency.png)")
    blocks.append("")
    blocks.append(
        "Source: `results/raw/w1_m1_*_c*.summary.json`, "
        "rendered by `analysis/report/make_report.py`."
    )
    return "\n".join(blocks) + "\n"


def gen_w1_m2() -> str:
    series = {label: load_series(prefix) for label, prefix in M2_SERIES}
    if not any(series[label] for label, _ in M2_SERIES):
        return "_No M2 results yet._\n"
    present = [(label, prefix) for label, prefix in M2_SERIES if series[label]]
    concs = sorted({c for label, _ in present for c in series[label]})
    lines = ["Output tok/s by concurrency:", ""]
    lines.append("| Concurrency | " + " | ".join(label for label, _ in present) + " |")
    lines.append("|---" * (len(present) + 1) + "|")
    for c in concs:
        row = [str(c)]
        for label, _ in present:
            s = series[label].get(c)
            row.append(f"{s['output_tok_per_s']:.1f}" if s else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("![FP8 throughput](docs/figures/fig3_fp8.png)")
    lines.append("")
    lines.extend(gen_accuracy_table())
    lines.append(
        "Source: `results/raw/w1_m2_*_c*.summary.json`, `results/accuracy/*.json`, "
        "rendered by `analysis/report/make_report.py`."
    )
    return "\n".join(lines) + "\n"


def accuracy_metrics(stem: str) -> dict[str, float]:
    """Extract task accuracies from an lm-eval results json in results/accuracy/."""
    path = ACCURACY / f"{stem}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    results = data.get("results", {})
    out: dict[str, float] = {}
    gsm = results.get("gsm8k", {})
    for key in ("exact_match,strict-match", "exact_match,flexible-extract"):
        if key in gsm:
            out["gsm8k"] = float(gsm[key])
            out["gsm8k_metric"] = key  # type: ignore[assignment]
            break
    mmlu = results.get("mmlu", {})
    for key in ("acc,none", "acc"):
        if key in mmlu:
            out["mmlu"] = float(mmlu[key])
            break
    return out


def gen_accuracy_table() -> list[str]:
    rows = [(label, accuracy_metrics(stem)) for label, stem in ACC_CONFIGS]
    if not any(m for _, m in rows):
        return ["_No accuracy results yet._", ""]
    base = rows[0][1]
    has_mmlu = any("mmlu" in m for _, m in rows)
    header = "| Config | GSM8K (strict) | Δ vs BF16 |"
    sep = "|---|---|---|"
    if has_mmlu:
        header += " MMLU | Δ vs BF16 |"
        sep += "---|---|"
    lines = ["Accuracy (lm-eval via the live server API; GSM8K 5-shot limit 500"
             + (", MMLU limit 20/subtask" if has_mmlu else "") + "):", "",
             header, sep]
    for label, m in rows:
        if not m:
            continue
        row = f"| {label} | {m['gsm8k']:.4f} | "
        row += ("baseline" if label == rows[0][0]
                else f"{m['gsm8k'] - base['gsm8k']:+.4f}") + " |"
        if has_mmlu:
            if "mmlu" in m:
                row += f" {m['mmlu']:.4f} | "
                row += ("baseline" if label == rows[0][0]
                        else f"{m['mmlu'] - base.get('mmlu', 0.0):+.4f}") + " |"
            else:
                row += " — | — |"
        lines.append(row)
    lines.append("")
    return lines


def gen_w1_m3() -> str:
    any_data = False
    lines: list[str] = []
    for engine, modes in M3_SERIES.items():
        present = [(label, p) for label, p in modes if load_series(p)]
        if not present:
            continue
        any_data = True
        base_series = load_series(modes[0][1])
        concs = sorted({c for _, p in present for c in load_series(p)})
        lines.append(f"**{engine}** (nat workload = GSM8K questions, greedy, "
                     "256 out, ignore_eos):")
        lines.append("")
        header = "| Concurrency |"
        sep = "|---|"
        for label, _ in present:
            header += f" {label} tok/s |"
            sep += "---|"
            if label != present[0][0]:
                header += " speedup | accepted/step |"
                sep += "---|---|"
        lines.append(header)
        lines.append(sep)
        for c in concs:
            row = f"| {c} |"
            for label, prefix in present:
                s = load_series(prefix).get(c)
                row += f" {s['output_tok_per_s']:.1f} |" if s else " — |"
                if label != present[0][0]:
                    b = base_series.get(c)
                    if s and b:
                        row += f" {s['output_tok_per_s'] / b['output_tok_per_s']:.2f}x |"
                    else:
                        row += " — |"
                    row += f" {accept_value(load_accept(prefix).get(c))} |"
            lines.append(row)
        lines.append("")
    if not any_data:
        return "_No M3 results yet._\n"
    lines.append("![Speculative decoding](docs/figures/fig4_spec_decode.png)")
    lines.append("")
    lines.append(
        "Source: `results/raw/w1_m3_*_c*.summary.json` and "
        "`results/raw/w1_m3_*_c*.accept.json` (acceptance shown only where the "
        "engine exposed counters/logs), rendered by `analysis/report/make_report.py`."
    )
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
    text = replace_block(text, "w1_m1", gen_w1_m1())
    text = replace_block(text, "w1_m2", gen_w1_m2())
    text = replace_block(text, "w1_m3", gen_w1_m3())
    return text


def make_figures() -> None:
    try:
        import matplotlib
    except ImportError:
        print("figures skipped: matplotlib not importable (run in bench venv)")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)

    # fig1: throughput vs concurrency (M1)
    m1 = {eng: load_series(p) for eng, p in M1_SERIES}
    if any(m1.values()):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for eng, series in m1.items():
            if not series:
                continue
            cs = sorted(series)
            ax.plot(cs, [series[c]["output_tok_per_s"] for c in cs],
                    marker="o", label=eng)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Output tok/s")
        ax.set_title("Qwen3-8B BF16, random 1024in/256out (H100)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / "fig1_throughput_concurrency.png", dpi=150)
        plt.close(fig)

        # fig2: TTFT & TPOT, p50 + p95
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for eng, series in m1.items():
            if not series:
                continue
            cs = sorted(series)
            axes[0].plot(cs, [series[c]["ttft_p50_ms"] for c in cs],
                         marker="o", label=f"{eng} p50")
            axes[0].plot(cs, [series[c]["ttft_p95_ms"] for c in cs],
                         marker="^", linestyle="--", label=f"{eng} p95")
            axes[1].plot(cs, [series[c]["tpot_p50_ms"] for c in cs],
                         marker="o", label=f"{eng} p50")
            axes[1].plot(cs, [series[c]["tpot_p95_ms"] for c in cs],
                         marker="^", linestyle="--", label=f"{eng} p95")
        for ax, name in zip(axes, ["TTFT (ms)", "TPOT (ms)"]):
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("Concurrency")
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("Qwen3-8B BF16, random 1024in/256out (H100)")
        fig.tight_layout()
        fig.savefig(FIGURES / "fig2_latency_concurrency.png", dpi=150)
        plt.close(fig)

    # fig3: FP8 variants
    m2 = [(label, load_series(p)) for label, p in M2_SERIES]
    if any(series for _, series in m2):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for label, series in m2:
            if not series:
                continue
            cs = sorted(series)
            ax.plot(cs, [series[c]["output_tok_per_s"] for c in cs],
                    marker="o", label=label)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Output tok/s")
        ax.set_title("FP8 variants, Qwen3-8B, random 1024in/256out (H100)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "fig3_fp8.png", dpi=150)
        plt.close(fig)

    # fig4: speculative decoding speedup + acceptance
    have_m3 = False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for engine, modes in M3_SERIES.items():
        base_series = load_series(modes[0][1])
        for label, prefix in modes[1:]:
            series = load_series(prefix)
            if not series or not base_series:
                continue
            have_m3 = True
            cs = sorted(set(series) & set(base_series))
            axes[0].plot(
                cs,
                [series[c]["output_tok_per_s"] / base_series[c]["output_tok_per_s"]
                 for c in cs],
                marker="o", label=f"{engine} {label}",
            )
            acc = load_accept(prefix)
            acs = [c for c in cs if accept_value(acc.get(c)) not in ("—", "not exposed")]
            if acs:
                axes[1].plot(acs, [float(accept_value(acc[c])) for c in acs],
                             marker="o", label=f"{engine} {label}")
    if have_m3:
        axes[0].axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
        axes[0].set_ylabel("Speedup vs baseline (output tok/s)")
        axes[1].set_ylabel("Mean accepted tokens / step")
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Concurrency")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("Speculative decoding, Qwen3-8B, nat workload (H100)")
        fig.tight_layout()
        fig.savefig(FIGURES / "fig4_spec_decode.png", dpi=150)
    plt.close(fig)
    print("figures regenerated in docs/figures/")


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
    make_figures()
    return 0


if __name__ == "__main__":
    sys.exit(main())
