#!/usr/bin/env python3
"""
P2 step 1: PROBE — can HF transformers load the NVFP4 checkpoint directly on
GPUs 4-7? Decides the goldens path (CONTEXT_AND_PLAN P2):

  PROBE PASSES -> goldens generate on GPUs 4-7 while vLLM stays up (idle);
                  no server downtime, reference precision = NVFP4 (same as
                  both engines -> ideal).
  PROBE FAILS  -> fallback: scheduled vLLM downtime, all-8-GPU bf16 reference
                  staged via /dev/shm (~2 TB model; slower, one-off).

This script only PROBES (load + one tiny greedy generation + timing). Golden
generation proper is generate_goldens.py, run after the path is chosen.

Run (vLLM may stay up — this is correctness work, not measurement; §4.3
forbids concurrent MEASUREMENT only):
  CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/probe_goldens_path.py
"""
import json, sys, time

MODEL_DIR = "/workspace/models/inkling-nvfp4"


def main():
    #1. version gate first: Inkling needs a recent transformers; fail informatively
    import transformers, torch
    print(f"[probe] transformers {transformers.__version__}, torch {torch.__version__}, "
          f"visible GPUs: {torch.cuda.device_count()}")
    if torch.cuda.device_count() != 4:
        sys.exit("[probe] expected exactly 4 visible GPUs — set CUDA_VISIBLE_DEVICES=4,5,6,7")

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    #2. config + tokenizer load (cheap; catches trust_remote_code / class issues)
    cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    print(f"[probe] config class: {type(cfg).__name__}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    print(f"[probe] tokenizer ok, vocab {tok.vocab_size if hasattr(tok,'vocab_size') else 'n/a'}")

    #3. the real question: does from_pretrained understand the ModelOpt NVFP4
    #   checkpoint (hf_quant_config.json)? Load spread over the 4 visible GPUs.
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            device_map="auto",        # shard across visible GPUs 4-7
            torch_dtype="auto",
        )
    except Exception as e:
        print(f"[probe] LOAD FAILED after {time.time()-t0:.0f}s:\n  {type(e).__name__}: {e}")
        print("[probe] VERDICT: transformers cannot load NVFP4 directly -> "
              "fallback path (bf16 reference, all 8 GPUs, vLLM down). "
              "If the error is quant-format related, also try: pip install -U "
              "transformers nvidia-modelopt")
        sys.exit(2)
    print(f"[probe] load OK in {time.time()-t0:.0f}s")
    for i in range(4):
        print(f"[probe]   gpu{i}: {torch.cuda.memory_allocated(i)/2**30:.1f} GiB")

    #4. one tiny greedy generation — proves the forward path end to end
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").to(model.device)
    t0 = time.time()
    out = model.generate(**ids, max_new_tokens=8, do_sample=False)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:])
    dt = time.time() - t0
    print(f"[probe] greedy 8 tokens in {dt:.1f}s ({8/dt:.2f} tok/s): {text!r}")

    #5. verdict + speed note for planning golden generation (50 prompts × 64 tok)
    est = 50 * 64 * dt / 8 / 60
    print(f"[probe] VERDICT: PASS — goldens can generate on GPUs 4-7 with vLLM up.")
    print(f"[probe] speed estimate for the parity set: ~{est:.0f} min "
          f"(fine if <120; else consider fewer max_new tokens)")
    #6. sanity: reference should agree with vLLM on this trivial prompt
    if "Paris" not in text:
        print("[probe] WARNING: expected 'Paris' in output — investigate before "
              "trusting this path for goldens")


if __name__ == "__main__":
    main()