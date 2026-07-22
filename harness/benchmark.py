"""
@brief  Perf benchmark runner. Measures decode throughput, TTFT, and peak GPU
        memory against ANY OpenAI-compatible endpoint — so the identical code
        path measures the vLLM baseline and the candidate engine.

        Hygiene built in: warmup rounds, median-of-N, fixed seed/prompt set,
        concurrency sweep, and run-to-run stddev reported for the merge gate.
"""
import json
import time
import argparse
import asyncio
import statistics
import subprocess

import aiohttp

WORKLOAD_PATH = "configs/bench_prompts.json"   # fixed ShareGPT-style mix
WARMUP_ROUNDS = 2
MEASURE_ROUNDS = 5


async def one_request(session, endpoint, model, prompt, max_tokens):
    """
    @brief  Streamed completion; returns (ttft_s, n_tokens, total_s).
    """
    t0 = time.perf_counter()
    ttft, n_tok = None, 0
    async with session.post(
        f"{endpoint}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "seed": 0,
        },
    ) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            n_tok += 1
    return ttft, n_tok, time.perf_counter() - t0


async def one_round(endpoint, model, prompts, concurrency, max_tokens):
    """
    @brief  Fire `concurrency` simultaneous streams; aggregate throughput.
    """
    conn = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=conn) as session:
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *[
                one_request(session, endpoint, model, prompts[i % len(prompts)], max_tokens)
                for i in range(concurrency)
            ]
        )
        wall = time.perf_counter() - t0
    total_tokens = sum(r[1] for r in results)
    ttfts = [r[0] for r in results if r[0] is not None]
    return {
        "decode_tok_s": total_tokens / wall,
        "ttft_ms": 1000 * statistics.median(ttfts),
    }


def peak_gpu_mem_gb(gpu_index: int) -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
         "-i", str(gpu_index)],
        capture_output=True, text=True,
    ).stdout.strip()
    return float(out) / 1024 if out else -1.0


def run(endpoint, model, gpu_index, concurrency, max_tokens):
    prompts = json.load(open(WORKLOAD_PATH))

    #1. warmup: fills caches, triggers JIT/autotune paths, stabilizes clocks
    for _ in range(WARMUP_ROUNDS):
        asyncio.run(one_round(endpoint, model, prompts, concurrency, max_tokens))

    #2. measure: N independent rounds, medians + stddev reported
    rounds = [
        asyncio.run(one_round(endpoint, model, prompts, concurrency, max_tokens))
        for _ in range(MEASURE_ROUNDS)
    ]
    toks = [r["decode_tok_s"] for r in rounds]
    return {
        "concurrency": concurrency,
        "decode_tok_s": statistics.median(toks),
        "run_stddev_tok_s": statistics.stdev(toks),
        "ttft_ms": statistics.median(r["ttft_ms"] for r in rounds),
        "peak_mem_gb": peak_gpu_mem_gb(gpu_index),
        "rounds": rounds,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32, 128])
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    #1. sweep concurrency levels; the merge gate keys off the highest level
    report = {str(c): run(args.endpoint, args.model, args.gpu, c, args.max_tokens)
              for c in args.concurrency}
    print(json.dumps(report, indent=2))
