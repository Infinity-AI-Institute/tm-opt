#!/usr/bin/env python3
"""Diagnose greedy nondeterminism: re-run parity prompts against the live
server and report first-divergence position + logprob gap vs the committed
goldens. Near-tie flips (small gaps, varying positions) = kernel reduction
nondeterminism; systematic divergence at position 0 = config mismatch.
Usage: python scripts/diag_determinism.py [n_prompts]"""
import json, sys, requests

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
items = json.load(open("goldens/goldens_parity.json"))["items"][:N]

def run(p):
    r = requests.post("http://localhost:8106/v1/completions", json={
        "model": "/workspace/models/inkling-nvfp4", "prompt": p,
        "max_tokens": 64, "temperature": 0, "logprobs": 1,
        "return_token_ids": True}, timeout=600).json()["choices"][0]
    return r["token_ids"], r["logprobs"]["token_logprobs"]

for i, it in enumerate(items):
    ids2, lp2 = run(it["prompt"])
    for k, (x, y) in enumerate(zip(it["token_ids"], ids2)):
        if x != y:
            gap = abs(it["token_logprobs"][k] - lp2[k])
            print(f"prompt {i:2d}: diverges at pos {k:2d}, "
                  f"lp_golden={it['token_logprobs'][k]:.4f} lp_rerun={lp2[k]:.4f} "
                  f"(gap {gap:.4f})")
            break
    else:
        print(f"prompt {i:2d}: identical")