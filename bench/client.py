"""Closed-loop streaming benchmark client for OpenAI-compatible /v1/completions.

Keeps exactly C requests in flight. Prompts are per-request random token ids
sampled with the model tokenizer (per-request seed) and decoded to text.
Writes per-request records to results/raw/<cell_id>.jsonl and an aggregate
summary to results/raw/<cell_id>.summary.json.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
import numpy as np
import yaml
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_prompt(tokenizer, input_len: int, seed: int) -> str:
    """Sample ~input_len random token ids (special tokens excluded) and decode."""
    rng = np.random.default_rng(seed)
    special = set(tokenizer.all_special_ids)
    ids = [
        int(t)
        for t in rng.integers(0, tokenizer.vocab_size, size=input_len)
        if int(t) not in special
    ]
    return tokenizer.decode(ids, skip_special_tokens=True)


async def run_one(session: aiohttp.ClientSession, cfg: dict, prompt: str) -> dict:
    payload = {
        "model": cfg["model"],
        "prompt": prompt,
        "max_tokens": cfg["output_len"],
        "temperature": cfg["temperature"],
        "ignore_eos": cfg["ignore_eos"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t_send = time.perf_counter()
    t_first_token = None
    n_chunks = 0
    usage_completion_tokens = None
    async with session.post(f"{cfg['base_url']}/v1/completions", json=payload) as resp:
        resp.raise_for_status()
        async for raw_line in resp.content:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage_completion_tokens = chunk["usage"]["completion_tokens"]
            choices = chunk.get("choices") or []
            if choices and choices[0].get("text"):
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                n_chunks += 1
    t_end = time.perf_counter()
    return {
        "t_send": t_send,
        "t_first_token": t_first_token,
        "t_end": t_end,
        "n_output_tokens": (
            usage_completion_tokens if usage_completion_tokens is not None else n_chunks
        ),
    }


async def run_phase(cfg: dict, tokenizer, n_requests: int, seed_offset: int) -> list[dict]:
    """Closed loop: C workers each pull the next request index until exhausted."""
    results: list[dict | None] = [None] * n_requests
    next_idx = 0
    lock = asyncio.Lock()
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)

    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def worker() -> None:
            nonlocal next_idx
            while True:
                async with lock:
                    if next_idx >= n_requests:
                        return
                    idx = next_idx
                    next_idx += 1
                prompt = make_prompt(
                    tokenizer, cfg["input_len"], cfg["seed"] + seed_offset + idx
                )
                results[idx] = await run_one(session, cfg, prompt)

        await asyncio.gather(*[worker() for _ in range(cfg["concurrency"])])
    return results  # type: ignore[return-value]


def summarize(cfg: dict, records: list[dict], wall_s: float) -> dict:
    ttft_ms = [
        (r["t_first_token"] - r["t_send"]) * 1e3
        for r in records
        if r["t_first_token"] is not None
    ]
    tpot_ms = [
        (r["t_end"] - r["t_first_token"]) * 1e3 / (r["n_output_tokens"] - 1)
        for r in records
        if r["t_first_token"] is not None and r["n_output_tokens"] > 1
    ]
    total_output_tokens = sum(r["n_output_tokens"] for r in records)
    return {
        "cell_id": cfg["cell_id"],
        "engine": cfg["engine"],
        "model": cfg["model"],
        "concurrency": cfg["concurrency"],
        "num_requests": len(records),
        "input_len": cfg["input_len"],
        "output_len": cfg["output_len"],
        "ttft_p50_ms": float(np.percentile(ttft_ms, 50)),
        "ttft_p95_ms": float(np.percentile(ttft_ms, 95)),
        "tpot_p50_ms": float(np.percentile(tpot_ms, 50)),
        "tpot_p95_ms": float(np.percentile(tpot_ms, 95)),
        "output_tok_per_s": total_output_tokens / wall_s,
        "total_output_tokens": total_output_tokens,
        "wall_time_s": wall_s,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to cell config yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])

    warmup = cfg.get("warmup_requests", 8)
    print(f"[client] warmup: {warmup} requests at concurrency {cfg['concurrency']}")
    await run_phase(cfg, tokenizer, warmup, seed_offset=1_000_000)

    print(f"[client] measuring: {cfg['num_requests']} requests")
    t0 = time.perf_counter()
    records = await run_phase(cfg, tokenizer, cfg["num_requests"], seed_offset=0)
    wall_s = time.perf_counter() - t0

    raw_dir = REPO_ROOT / "results" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = raw_dir / f"{cfg['cell_id']}.jsonl"
    with jsonl_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    summary = summarize(cfg, records, wall_s)
    summary_path = raw_dir / f"{cfg['cell_id']}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[client] wrote {jsonl_path} and {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
