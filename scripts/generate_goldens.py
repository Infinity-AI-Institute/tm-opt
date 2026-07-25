#!/usr/bin/env python3
"""
P2: generate correctness goldens — PATH C (decision 2026-07-25):
reference = the PINNED vLLM build serving the NVFP4 checkpoint.

Why vLLM is the reference (audit note, keep in sync with BASELINE_NOTES.md):
- transformers 5.14.1 was DEMONSTRATED to load this checkpoint incorrectly:
  grouped MoE expert weights hit "Reinit due to size mismatch" (packed-fp4
  half-width tensors vs dequantized module shapes) -> layers 3-65 experts
  randomly re-initialized. Evidence: docs/logs/ probe run 3 (2026-07-24/25).
- vLLM's Inkling integration is the model authors' day-0 code path on the
  identical weights both engines serve; the parity gate's real requirement is
  "identical computation semantics on identical weights".
- Independence hardening = PATH B (bf16 transformers spot-check on a small
  prompt subset), scheduled separately; see plan.

What this writes:
  goldens/goldens_parity.json   {"meta": {...}, "items": [per-prompt records]}
Each item: prompt, token_ids (list[int]), tokens (list[str]),
token_logprobs (list[float]) for max_new greedy tokens.

Run (server must be the BENCH config on :8106; this is correctness work —
vLLM serving is exactly what we want here):
  python scripts/generate_goldens.py
  python scripts/generate_goldens.py --check   # re-query and diff vs file
"""
import argparse, hashlib, json, pathlib, sys, time

import requests

ENDPOINT = "http://localhost:8106"
MODEL = "/workspace/models/inkling-nvfp4"
PARITY = "configs/parity_prompts.json"
OUT = "goldens/goldens_parity.json"
MAX_NEW = 64          # tokens per prompt; matches harness parity depth
TIMEOUT = 600


def greedy(prompt: str) -> dict:
    #1. exact request shape the parity gate uses (BASELINE_NOTES format
    #   contract): greedy, logprobs, choice-level token ids
    r = requests.post(f"{ENDPOINT}/v1/completions", json={
        "model": MODEL, "prompt": prompt, "max_tokens": MAX_NEW,
        "temperature": 0, "seed": 0, "logprobs": 1,
        "return_token_ids": True, "ignore_eos": False,
    }, timeout=TIMEOUT)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    ids = ch.get("token_ids")
    if ids is None:
        raise RuntimeError("server did not return token_ids — wrong build/config?")
    return {
        "prompt": prompt,
        "token_ids": ids,
        "tokens": ch["logprobs"]["tokens"],
        "token_logprobs": ch["logprobs"]["token_logprobs"],
        "finish_reason": ch.get("finish_reason"),
    }


def server_meta() -> dict:
    #2. provenance: pin exactly which server produced these goldens
    r = requests.post(f"{ENDPOINT}/v1/completions", json={
        "model": MODEL, "prompt": "x", "max_tokens": 1, "temperature": 0,
    }, timeout=TIMEOUT)
    r.raise_for_status()
    fp = r.json().get("system_fingerprint", "unknown")
    lock = json.loads(pathlib.Path("configs/canonical.lock.json").read_text())
    return {
        "reference": "vllm-pinned (PATH C; transformers loader demonstrated "
                     "incorrect on grouped fp4 experts — see docs/logs/)",
        "system_fingerprint": fp,
        "model_dir": MODEL,
        "cache_key": lock["cache_key"],
        "max_new_tokens": MAX_NEW,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "independence_hardening": "PATH B pending: bf16 transformers "
                                  "spot-check on >=5 prompts",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="re-query the server and diff against existing goldens")
    a = ap.parse_args()

    prompts = json.loads(pathlib.Path(PARITY).read_text())
    if isinstance(prompts, dict):
        prompts = prompts["prompts"]
    print(f"[goldens] {len(prompts)} parity prompts, {MAX_NEW} tokens each")

    items, t0 = [], time.time()
    for i, p in enumerate(prompts):
        items.append(greedy(p))
        if (i + 1) % 10 == 0:
            print(f"[goldens] {i+1}/{len(prompts)} ({time.time()-t0:.0f}s)", flush=True)

    record = {"meta": server_meta(), "items": items}
    blob = json.dumps(record, indent=1, sort_keys=True)
    digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
    record["meta"]["goldens_sha"] = digest

    if a.check:
        old = json.loads(pathlib.Path(OUT).read_text())
        mism = sum(1 for x, y in zip(old["items"], items)
                   if x["token_ids"] != y["token_ids"])
        print(f"[goldens] CHECK: {mism} / {len(items)} prompts differ vs {OUT}")
        sys.exit(0 if mism == 0 else 1)

    pathlib.Path("goldens").mkdir(exist_ok=True)
    pathlib.Path(OUT).write_text(json.dumps(record, indent=1, sort_keys=True))
    print(f"[goldens] wrote {OUT} (sha {digest}) in {time.time()-t0:.0f}s\n"
          f"[goldens] next: point harness/correctness.py reference loader at "
          f"this file, run the gate against the live server (must be green), "
          f"commit goldens/ + the gate run log.")


if __name__ == "__main__":
    main()
