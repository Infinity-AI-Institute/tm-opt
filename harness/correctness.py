"""
@brief  Correctness gate: compares the candidate engine (OpenAI-compatible
        endpoint) against the HuggingFace transformers reference for
        Inkling. An experiment that fails here never reaches perf.

        Two checks, in order of strictness:
          1. Greedy token match  — temperature 0, exact token-id agreement
                                   for the first N generated tokens.
          2. Logprob parity      — max |delta| of top-1 logprobs within a
                                   bf16-appropriate tolerance.
"""
import json
import argparse
import requests

# tolerance chosen for bf16 accumulation differences; tighten to 5e-3 once
# the engine is stable, loosen ONLY with a written justification in the ledger
LOGPROB_TOL = 2e-2
GREEDY_TOKENS = 64

PROMPTS_PATH = "configs/parity_prompts.json"   # ~50 diverse prompts, fixed forever


def reference_greedy(model_dir: str, prompts: list[str], max_new: int, device: str):
    """
    @brief  Greedy reference generation + per-step top-1 logprobs via HF.
    @param  model_dir : local path of the downloaded Inkling checkpoint
    @return list of (token_ids, logprobs) per prompt
    """
    #1. import lazily so perf-only runs never pay the transformers import cost
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=device
    ).eval()

    out = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(model.device)
        gen_ids, logprobs = [], []
        #2. step-by-step greedy so we capture the chosen token's logprob
        with torch.no_grad():
            cur = ids
            past = None
            for _ in range(max_new):
                res = model(cur, past_key_values=past, use_cache=True)
                past = res.past_key_values
                logits = res.logits[:, -1, :].float()
                lp = torch.log_softmax(logits, dim=-1)
                nxt = int(logits.argmax(dim=-1))
                gen_ids.append(nxt)
                logprobs.append(float(lp[0, nxt]))
                cur = torch.tensor([[nxt]], device=model.device)
        out.append((gen_ids, logprobs))
    return out


def candidate_greedy(endpoint: str, model: str, prompts: list[str], max_new: int):
    """
    @brief  Same generation through the engine under test via /v1/completions.
    """
    out = []
    for p in prompts:
        r = requests.post(
            f"{endpoint}/v1/completions",
            json={
                "model": model,
                "prompt": p,
                "max_tokens": max_new,
                "temperature": 0,
                "logprobs": 1,
                "seed": 0,
                "return_token_ids": True,
            },
            timeout=300,
        )
        r.raise_for_status()
        choice = r.json()["choices"][0]
        ids = choice.get("token_ids")
        if ids is None:
            raise RuntimeError(
                "engine did not return token_ids — check return_token_ids support")
        lps = choice["logprobs"]["token_logprobs"]
        out.append((ids, lps))
    return out


def run_gate(endpoint: str, model: str, model_dir: str, device: str = "cuda:0"):
    """
    @return (parity_pass: bool, max_logprob_delta: float, detail: dict)
    """
    prompts = json.load(open(PROMPTS_PATH))

    ref = reference_greedy(model_dir, prompts, GREEDY_TOKENS, device)
    cand = candidate_greedy(endpoint, model, prompts, GREEDY_TOKENS)

    max_delta, mismatches = 0.0, 0
    for (r_ids, r_lp), (c_ids, c_lp) in zip(ref, cand):
        #1. token-id agreement is the hard requirement
        if list(r_ids) != list(c_ids)[: len(r_ids)]:
            mismatches += 1
            continue
        #2. logprob drift is the soft requirement, tracked as max |delta|
        for a, b in zip(r_lp, c_lp):
            if a is None or b is None:
                continue
            max_delta = max(max_delta, abs(a - b))

    passed = mismatches == 0 and max_delta <= LOGPROB_TOL
    return passed, max_delta, {"prompt_mismatches": mismatches, "n_prompts": len(prompts)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)      
    ap.add_argument("--model", required=True)         # served model name
    ap.add_argument("--model-dir", required=True)     # local HF checkpoint path
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ok, delta, detail = run_gate(args.endpoint, args.model, args.model_dir, args.device)
    print(json.dumps({"parity_pass": ok, "max_logprob_delta": delta, **detail}))
    raise SystemExit(0 if ok else 1)
