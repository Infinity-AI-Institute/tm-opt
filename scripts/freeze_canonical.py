#!/usr/bin/env python3
"""
P1: freeze the canonical measurement contract.

Runs a concurrency sweep for BOTH workloads (decision D6) against the
benchmark-mode vLLM server (scripts/serve_vllm_bench.sh must be the one
serving), finds the throughput knee, and emits:

  configs/canonical_decode_heavy.json    (headline workload, D6)
  configs/canonical_prefill_heavy.json
  configs/canonical.lock.json            (both + sweep evidence + CACHE KEY)

The cache key = sha256 over the sorted canonical parameters. Every ledger row
must carry it; rows with a different key are not comparable (fairness rule).

Graph-readiness (user requirement 2026-07-20): the lock file also carries the
`report` block — the exact footnote string and baseline-line labels the
trajectory chart needs, so plot_trajectory.py never invents metadata.

Usage:
  python scripts/freeze_canonical.py --endpoint http://localhost:8106 \
      [--concs 8,16,32,64,128] [--per-conc-prompts 3] [--dry]

Notes:
- This script MEASURES to choose `concurrency`; everything else is declared
  here, reviewed by a human, then frozen by committing the emitted configs.
- Sweep methodology: per concurrency c, one warmup wave of c requests, then
  timed waves totalling max(3c, 24) requests; throughput = total generated
  tokens / wall time. Knee = smallest c whose throughput is within 5% of the
  best observed (prefer lower concurrency at equal throughput: lower latency,
  less scheduler luck).
- Prompts are seeded synthetic (seed in the contract) with range_ratio 0.5:
  lengths uniform in [ISL/2, ISL]. ignore_eos pins OSL exactly.
"""
import argparse, hashlib, json, os, random, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor

import requests

MODEL = "/workspace/models/inkling-nvfp4"

#1. the declared (non-measured) half of the contract — review before freezing
WORKLOADS = {
    "decode_heavy": {  # headline (D6): where serving is throughput-constrained
        "isl": 1024, "osl": 8192, "range_ratio": 0.5, "seed": 1001,
    },
    "prefill_heavy": {
        "isl": 8192, "osl": 1024, "range_ratio": 0.5, "seed": 1002,
    },
}
COMMON = {
    "model": MODEL,
    "max_model_len": 16384,
    "temperature": 0.0,
    "ignore_eos": True,
    "prefix_caching": False,          # serve_vllm_bench.sh enforces
    "kv_cache_memory": 111231943107,  # pinned bytes (BASELINE_NOTES.md)
    "tensor_parallel": 4,
    "mtp": False,                     # D7: canonical is MTP-OFF both engines
    "gpu_arch": "B300 SXM6 (sm_103a)",
}


def make_prompts(isl: int, range_ratio: float, seed: int, n: int) -> list[str]:
    """Seeded synthetic prompts, token-length ~ U[isl*range_ratio, isl].
    'word ' ~= 1 token for this tokenizer family; exactness is not required —
    the SAME generator + seed is the contract, applied to both engines."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        L = rng.randint(int(isl * range_ratio), isl)
        out.append(f"[req {i}] " + "word " * max(L - 4, 1))
    return out


def one_request(endpoint: str, prompt: str, osl: int) -> int:
    r = requests.post(
        f"{endpoint}/v1/completions",
        json={"model": MODEL, "prompt": prompt, "max_tokens": osl,
              "temperature": 0, "ignore_eos": True},
        timeout=3600,
    )
    r.raise_for_status()
    return r.json()["usage"]["completion_tokens"]


def measure(endpoint: str, wl: dict, conc: int, per_conc: int) -> dict:
    n = max(per_conc * conc, 24)
    prompts = make_prompts(wl["isl"], wl["range_ratio"], wl["seed"], n + conc)
    with ThreadPoolExecutor(max_workers=conc) as pool:
        #2. warmup wave (untimed): one full wave at this concurrency
        list(pool.map(lambda p: one_request(endpoint, p, wl["osl"]),
                      prompts[:conc]))
        #3. timed waves
        t0 = time.time()
        toks = list(pool.map(lambda p: one_request(endpoint, p, wl["osl"]),
                             prompts[conc:conc + n]))
        dt = time.time() - t0
    return {"concurrency": conc, "requests": n,
            "gen_tokens": sum(toks), "wall_s": round(dt, 2),
            "tok_per_s": round(sum(toks) / dt, 1)}


def pick_knee(rows: list[dict]) -> int:
    best = max(r["tok_per_s"] for r in rows)
    for r in rows:  # rows are in ascending concurrency order
        if r["tok_per_s"] >= 0.95 * best:
            return r["concurrency"]
    return rows[-1]["concurrency"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8106")
    ap.add_argument("--concs", default="8,16,32,64,128")
    ap.add_argument("--per-conc-prompts", type=int, default=3)
    ap.add_argument("--dry", action="store_true",
                    help="tiny lengths + 2 concs; pipeline check only")
    args = ap.parse_args()
    concs = [int(c) for c in args.concs.split(",")]

    if args.dry:  #4. fast path to validate plumbing before the real evening run
        for wl in WORKLOADS.values():
            wl["isl"], wl["osl"] = 256, 128
        concs = concs[:2]

    #5. refuse to run against a prefix-caching server (wrong config = wrong key)
    metrics = requests.get(f"{args.endpoint}/metrics", timeout=30).text
    if args.dry is False and "prefix_cache" in metrics:
        q = [l for l in metrics.splitlines()
             if l.startswith("vllm:prefix_cache_queries_total")]
        if q and float(q[0].rsplit(" ", 1)[1]) > 0:
            sys.exit("[freeze] server has served prefix-cached traffic — "
                     "restart with scripts/serve_vllm_bench.sh first")

    sweeps, frozen = {}, {}
    for name, wl in WORKLOADS.items():
        print(f"[freeze] sweeping {name} ISL={wl['isl']} OSL={wl['osl']} "
              f"concs={concs}", flush=True)
        rows = []
        for c in concs:
            r = measure(args.endpoint, wl, c, args.per_conc_prompts)
            rows.append(r)
            print(f"[freeze]   conc={c:<4d} {r['tok_per_s']:>10.1f} tok/s "
                  f"({r['requests']} reqs, {r['wall_s']}s)", flush=True)
        knee = pick_knee(rows)
        print(f"[freeze] {name}: knee -> concurrency {knee}")
        sweeps[name] = rows
        frozen[name] = {**COMMON, **wl, "concurrency": knee,
                        "num_warmup_requests": 2 * knee,
                        "num_bench_requests": 5 * knee}

    #6. cache key over the frozen parameters only (not the sweep evidence)
    key = hashlib.sha256(
        json.dumps(frozen, sort_keys=True).encode()).hexdigest()[:16]

    os.makedirs("configs", exist_ok=True)
    for name, cfg in frozen.items():
        cfg["cache_key"] = key
        with open(f"configs/canonical_{name}.json", "w") as f:
            json.dump(cfg, f, indent=2)

    lock = {
        "cache_key": key,
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workloads": frozen,
        "sweep_evidence": sweeps,
        #7. graph-readiness block: plot_trajectory.py reads labels/footnote here
        "report": {
            "headline_workload": "decode_heavy",
            "baseline_lines": [
                {"id": "vllm_mtp_off", "label": "vLLM (canonical, MTP off)"},
                {"id": "vllm_mtp_on", "label": "vLLM + MTP (tracked pair)"},
            ],
            "footnote": (f"Inkling-NVFP4, 4xB300 TP=4, decode-heavy ISL~1k/"
                         f"OSL=8k & prefill-heavy ISL~8k/OSL=1k, range_ratio "
                         f"0.5, ignore_eos, prefix cache off, KV pinned, "
                         f"no speculative decoding (canonical); build "
                         f"g9243e0124; cache_key {key}"),
        },
    }
    with open("configs/canonical.lock.json", "w") as f:
        json.dump(lock, f, indent=2)
    print(f"[freeze] WROTE configs/canonical_*.json + canonical.lock.json\n"
          f"[freeze] cache_key = {key}\n"
          f"[freeze] review, then commit to FREEZE. Changing any parameter "
          f"afterwards changes the key and invalidates comparisons.")


if __name__ == "__main__":
    main()