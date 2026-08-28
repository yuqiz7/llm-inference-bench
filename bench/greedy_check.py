"""Greedy-consistency spot check for speculative decoding (M3).

Sends the first 3 nat-workload prompts (GSM8K test questions, dataset order —
the same prompts the C=1 sweep cell measures first) as non-streaming greedy
completions and saves the outputs. Run once against the baseline server and
once against the speculative server, then compare with --compare A.json B.json:
speculative decoding should be token-identical under greedy; any divergence is
recorded as-is (batched-verify float nondeterminism is possible).
"""

import argparse
import json
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "results" / "raw"


def capture(base_url: str, model: str, out: Path, n: int = 3) -> None:
    import client

    prompts = client.load_nat_prompts()[:n]
    results = []
    for i, prompt in enumerate(prompts):
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0,
            "ignore_eos": True,
        }).encode()
        req = urllib.request.Request(
            f"{base_url}/v1/completions", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        results.append({
            "prompt_index": i,
            "prompt": prompt,
            "text": data["choices"][0]["text"],
        })
        print(f"[greedy_check] prompt {i}: {len(data['choices'][0]['text'])} chars")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"base_url": base_url, "model": model,
                               "outputs": results}, indent=2) + "\n")
    print(f"[greedy_check] wrote {out}")


def compare(path_a: Path, path_b: Path, out: Path) -> None:
    a = json.loads(path_a.read_text())["outputs"]
    b = json.loads(path_b.read_text())["outputs"]
    rows = []
    for ra, rb in zip(a, b):
        identical = ra["text"] == rb["text"]
        row = {"prompt_index": ra["prompt_index"], "identical": identical}
        if not identical:
            # first divergence position (character level, on the raw text)
            pos = next(
                (k for k, (x, y) in enumerate(zip(ra["text"], rb["text"])) if x != y),
                min(len(ra["text"]), len(rb["text"])),
            )
            row["first_divergence_char"] = pos
            row["a_context"] = ra["text"][max(0, pos - 40):pos + 40]
            row["b_context"] = rb["text"][max(0, pos - 40):pos + 40]
        rows.append(row)
        status = ("IDENTICAL" if identical
                  else f"DIVERGED at char {row['first_divergence_char']}")
        print(f"[greedy_check] prompt {row['prompt_index']}: {status}")
    out.write_text(json.dumps({"a": str(path_a), "b": str(path_b),
                               "comparison": rows}, indent=2) + "\n")
    print(f"[greedy_check] wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_cap = sub.add_parser("capture")
    p_cap.add_argument("--base-url", required=True)
    p_cap.add_argument("--model", default="Qwen/Qwen3-8B")
    p_cap.add_argument("--out", required=True)
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("a")
    p_cmp.add_argument("b")
    p_cmp.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.cmd == "capture":
        capture(args.base_url, args.model, Path(args.out))
    else:
        compare(Path(args.a), Path(args.b), Path(args.out))


if __name__ == "__main__":
    main()
