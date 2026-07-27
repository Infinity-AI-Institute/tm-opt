"""
@brief  Canonical perf benchmark. Measures total throughput against ANY
        OpenAI-compatible endpoint under a FROZEN canonical config — the
        identical code path measures the vLLM baseline and the candidate
        engine (fairness rule: same workload generator, same counting, same
        timing for both).

        Methodology is IDENTICAL to the P1 freeze sweep (freeze_canonical.py):
        seeded synthetic prompts (range_ratio applied), non-streamed
        completions, tokens counted from usage.completion_tokens, wall-clock
        over the measured block. The sweep's numbers and these are therefore
        directly comparable.

        Protocol per run (from the canonical config):
          warmup:   num_warmup_requests (= 2 x concurrency), untimed
          measure:  --repeats independent blocks of num_bench_requests
                    (= 5 x concurrency); median = the number,
                    stddev/median = the noise floor input
        Ledger: with --ledger-iteration N --engine E --label/--mechanism,
        appends a row to experiments/ledger.jsonl carrying every
        docs/LEDGER_SCHEMA.md field.

        TIME-BOXED protocol (--max-seconds N; HUMAN RULING 2026-07-27):
        the canonical request-count protocol cannot terminate below ~1000
        tok/s aggregate (one OSL-8192 request alone exceeds the per-request
        clock; full math in PROGRESS.md B4.1). For such engines the honest
        iteration-0 form is a fixed-duration closed-loop steady-state
        measurement: the SAME seeded prompt generator (identical ISL
        distribution), OSL capped so individual requests complete inside
        the box, effective concurrency capped to the engine's admission
        width, N seconds of continuous issuance, throughput = completed
        tokens / elapsed (in-flight requests at deadline are drained and
        counted). Rows carry protocol="timeboxed(...)" and are NEVER
        compared against protocol="canonical" rows as like-for-like; the
        pct_vs_baseline on a timeboxed row is an orientation number and is
        labeled by its protocol field.
"""
import argparse
import hashlib
import json
import pathlib
import random
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

LEDGER = "experiments/ledger.jsonl"


def make_prompts(isl: int, range_ratio: float, seed: int, n: int) -> list[str]:
    """
    @brief  Seeded synthetic prompts — MUST stay byte-identical to
            freeze_canonical.make_prompts (same generator = same contract).
    """
    rng = random.Random(seed)
    out = []
    for i in range(n):
        L = rng.randint(int(isl * range_ratio), isl)
        out.append(f"[req {i}] " + "word " * max(L - 4, 1))
    return out


def one_request(endpoint: str, model: str, prompt: str, osl: int, ignore_eos: bool) -> int:
    r = requests.post(
        f"{endpoint}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": osl,
              "temperature": 0, "seed": 0, "ignore_eos": ignore_eos},
        timeout=3600,
    )
    r.raise_for_status()
    return r.json()["usage"]["completion_tokens"]


def run_block(endpoint, model, prompts, conc, osl, ignore_eos):
    """
    @brief  One timed block: len(prompts) requests at fixed concurrency.
    @return (tok_per_s, gen_tokens, wall_s)
    """
    with ThreadPoolExecutor(max_workers=conc) as pool:
        t0 = time.time()
        toks = list(pool.map(
            lambda p: one_request(endpoint, model, p, osl, ignore_eos), prompts))
        dt = time.time() - t0
    return sum(toks) / dt, sum(toks), dt


def run_canonical(endpoint: str, model: str, cfg: dict, repeats: int) -> dict:
    """
    @brief  Frozen protocol for one workload config; returns the measurement
            record (median tok/s, noise floor, evidence per block).
    """
    conc = cfg["concurrency"]
    n_warm = cfg["num_warmup_requests"]
    n_bench = cfg["num_bench_requests"]
    total = n_warm + repeats * n_bench
    prompts = make_prompts(cfg["isl"], cfg["range_ratio"], cfg["seed"], total)

    #1. warmup: fills caches/JIT/clock stabilization; untimed, but counted
    #   toward warm server state (fairness: identical for both engines)
    print(f"[bench] warmup {n_warm} reqs @ conc {conc}", flush=True)
    run_block(endpoint, model, prompts[:n_warm], conc, cfg["osl"], cfg["ignore_eos"])

    #2. measured blocks
    blocks, off = [], n_warm
    for b in range(repeats):
        tps, toks, wall = run_block(
            endpoint, model, prompts[off:off + n_bench], conc,
            cfg["osl"], cfg["ignore_eos"])
        off += n_bench
        blocks.append({"tok_per_s": round(tps, 1), "gen_tokens": toks,
                       "wall_s": round(wall, 1)})
        print(f"[bench] block {b+1}/{repeats}: {tps:,.1f} tok/s "
              f"({toks} tok, {wall:.0f}s)", flush=True)

    toks = [b["tok_per_s"] for b in blocks]
    med = statistics.median(toks)
    noise_pct = (100 * statistics.stdev(toks) / med) if len(toks) > 1 else 0.0
    return {"tok_per_s": med, "noise_floor_pct": round(noise_pct, 2),
            "blocks": blocks, "concurrency": conc,
            "protocol": "canonical"}


def run_timeboxed(endpoint: str, model: str, cfg: dict, max_seconds: int,
                  tb_conc: int, tb_osl: int) -> dict:
    """
    @brief  Time-boxed closed-loop steady-state protocol (see module header;
            HUMAN RULING 2026-07-27). Same seeded prompt generator, OSL
            capped to tb_osl, tb_conc worker threads issuing back-to-back
            until the deadline; in-flight requests at deadline drain and
            count. Throughput = all completed tokens / total elapsed.
    """
    conc = min(cfg["concurrency"], tb_conc)
    osl = min(cfg["osl"], tb_osl)
    #1. generous prompt pool from the SAME generator (identical ISL dist);
    #   pool size is an upper bound, threads walk it by shared index
    pool_n = 100000
    prompts = make_prompts(cfg["isl"], cfg["range_ratio"], cfg["seed"], pool_n)

    #2. short warmup: conc requests, untimed
    print(f"[bench] TIMEBOXED warmup {conc} reqs @ conc {conc} osl {osl}",
          flush=True)
    run_block(endpoint, model, prompts[:conc], conc, osl, cfg["ignore_eos"])

    #3. closed loop until deadline; drain counts
    lock = threading.Lock()
    state = {"next": conc, "toks": 0, "done": 0}
    t0 = time.time()
    deadline = t0 + max_seconds

    def worker():
        while True:
            with lock:
                if time.time() >= deadline or state["next"] >= pool_n:
                    return
                i = state["next"]; state["next"] += 1
            n = one_request(endpoint, model, prompts[i], osl, cfg["ignore_eos"])
            with lock:
                state["toks"] += n; state["done"] += 1

    threads = [threading.Thread(target=worker) for _ in range(conc)]
    for t in threads: t.start()
    for t in threads: t.join()          # drain: in-flight at deadline finish
    wall = time.time() - t0

    tps = state["toks"] / wall if wall > 0 else 0.0
    print(f"[bench] timeboxed: {tps:,.1f} tok/s ({state['toks']} tok, "
          f"{state['done']} reqs, {wall:.0f}s incl. drain)", flush=True)
    return {"tok_per_s": round(tps, 1), "noise_floor_pct": 0.0,
            "blocks": [{"tok_per_s": round(tps, 1),
                        "gen_tokens": state["toks"],
                        "wall_s": round(wall, 1),
                        "requests": state["done"]}],
            "concurrency": conc,
            "protocol": f"timeboxed({max_seconds}s, conc={conc}, osl_cap={osl})"}


def gpu_mem_snapshot() -> str:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader"], capture_output=True, text=True).stdout
    return "; ".join(l.strip() for l in out.splitlines())


def append_ledger_row(row: dict):
    #1. append-only by convention AND by never opening with truncate
    pathlib.Path("experiments").mkdir(exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[bench] ledger row appended: iter={row['iteration']} "
          f"engine={row['engine']} workload={row['workload']} "
          f"tok_per_s={row['tok_per_s']:,}")


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="/workspace/models/inkling-nvfp4")
    ap.add_argument("--config", required=True,
                    help="configs/canonical_<workload>.json (frozen)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="independent measured blocks (median reported)")
    ap.add_argument("--max-seconds", type=int, default=None,
                    help="TIME-BOXED protocol (HUMAN RULING 2026-07-27): "
                         "closed-loop steady-state for N seconds instead of "
                         "the canonical request-count protocol; for engines "
                         "below ~1000 tok/s where canonical cannot terminate")
    ap.add_argument("--timebox-conc", type=int, default=8,
                    help="effective concurrency cap in timeboxed mode "
                         "(default 8 = pyengine server admission width)")
    ap.add_argument("--timebox-osl", type=int, default=128,
                    help="per-request max_tokens cap in timeboxed mode so "
                         "individual requests complete inside the box")
    ap.add_argument("--ledger-iteration", type=int, default=None,
                    help="if set, append a ledger row for this iteration")
    ap.add_argument("--engine", default="vllm",
                    help="ledger engine tag: vllm | pyengine")
    ap.add_argument("--baseline-id", default="vllm_mtp_off")
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--mechanism", default="vLLM pinned build, canonical config")
    ap.add_argument("--log-path", default="")
    args = ap.parse_args()

    cfg = json.loads(pathlib.Path(args.config).read_text())
    workload = pathlib.Path(args.config).stem.replace("canonical_", "")
    proto = f"TIMEBOXED {args.max_seconds}s" if args.max_seconds else f"repeats={args.repeats}"
    print(f"[bench] {workload}: ISL~{cfg['isl']} OSL={cfg['osl']} "
          f"conc={cfg['concurrency']} cache_key={cfg['cache_key']} {proto}",
          flush=True)

    if args.max_seconds:
        rec = run_timeboxed(args.endpoint, args.model, cfg, args.max_seconds,
                            args.timebox_conc, args.timebox_osl)
    else:
        rec = run_canonical(args.endpoint, args.model, cfg, args.repeats)
    rec.update({"workload": workload, "cache_key": cfg["cache_key"],
                "gpu_mem": gpu_mem_snapshot(),
                "measured_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(json.dumps(rec, indent=1))

    if args.ledger_iteration is not None:
        #2. pct_vs_baseline: baseline rows are their own denominator (100.0);
        #   candidate rows compute vs the committed baseline row. NOTE: a
        #   timeboxed row's pct is an ORIENTATION number (protocol differs
        #   from the canonical baseline row) — the protocol field labels it.
        pct = 100.0
        if args.engine != "vllm":
            base = None
            for line in pathlib.Path(LEDGER).read_text().splitlines():
                r = json.loads(line)
                if (r.get("engine") == "vllm" and r.get("workload") == workload
                        and r.get("baseline_id") == args.baseline_id):
                    base = r["tok_per_s"]
            if base:
                pct = round(100 * rec["tok_per_s"] / base, 1)
        append_ledger_row({
            "iteration": args.ledger_iteration, "engine": args.engine,
            "workload": workload, "label": args.label,
            "mechanism": args.mechanism, "tok_per_s": rec["tok_per_s"],
            "pct_vs_baseline": pct, "baseline_id": args.baseline_id,
            "cache_key": cfg["cache_key"], "commit": git_head(),
            "accepted": True, "noise_floor_pct": rec["noise_floor_pct"],
            "protocol": rec["protocol"],
            "log_path": args.log_path, "blocks": rec["blocks"],
        })


if __name__ == "__main__":
    main()
    