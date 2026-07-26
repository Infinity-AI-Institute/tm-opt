# Plan Checklist

## Stage 0 — Access & Environment ✅
- [x] Node sanity check (8×B300 sm_103a, 267.7 GiB/GPU, NV18 mesh, 2 NUMA
      domains, 3.9 TiB RAM, /workspace 1.5 TB persistent)
- [x] SSH/VS Code direct-TCP stable; tmux discipline; git identity + token
- [x] /workspace/maverick quarantined (never opened); SECURITY_LEDGER.md started
- [x] pod_init.sh accumulating every restart-lost step (CUDA apt, flashinfer
      symlink, git creds, sshd keepalives, numactl)

## Stage 1 — Baseline & Ground Truth
**Exit criteria: committed baseline number + goldens + frozen configs**

### Setup
- [x] Download weights (inkling-nvfp4, 551.35 GiB, 33 shards + mtp.safetensors)
- [x] Toolchain setup (venv; pinned vLLM g9243e0124 + scipy fix; PyTorch cu130;
      **CUDA 13.0** — found to be a serve-time BLOCKER for sm_103a FlashInfer JIT)
- [x] Establish the canonical benchmark contract ✅ **FROZEN 2026-07-23**
    - [x] bench-mode server (prefix cache OFF, KV pinned, numactl capability-probed)
    - [x] sweep tooling (freeze_canonical.py: two workloads per D6, seeded prompts,
          knee selection, cache-key emission, graph `report` block)
    - [x] first sweep {8..128} — no knee found (still scaling 1.5×/doubling at 128;
          correct catch, false knee avoided)
    - [x] extended sweep {128,256,512} — decode still +24%/doubling → extended again
    - [x] final sweep {512,768,1024} — TRUE PEAK found: decode 13,374 tok/s @512,
          collapse to 4,959 @768 (KV-capacity edge: 512×0.64 GiB fits pool, 768 doesn't)
    - [x] 4 review checks passed (peak shape; 512 plausible; noise decode 0.68% /
          prefill 2.2%; absolutes sane vs per-stream orientation)
    - [x] committed configs/canonical_*.json + canonical.lock.json,
          cache_key 8451a604a8849296 (D10) + three-stage sweep evidence in docs/logs/

### Context & design docs
- [x] Define CLAUDE.md (auto-loaded router; synced to rev-3 truth 2026-07-20)
- [x] Define docs/ARCHITECTURE.md (rev 3: KV lever RETIRED after investigation;
      "Where the wins must come from" ranking)
    - [x] scheduler section updated (rev 4: sustain 512 seqs, KV cliff,
          preemption grace, launch-overhead ×512 rationale)
- [x] Define/generate config.h based on checkpoint (not ARCHITECTURE — checkpoint
      is the source of truth; ARCHITECTURE derives from it)
- [x] CONTEXT_AND_PLAN.md — decision record D1–D9 + phases P1–P6 (added later;
      supersedes older docs on conflict)

### Kernel scaffolds (C++ side = CUDA porting roadmap per D5)
- [x] Normalizing activations (fused_rmsnorm.cu — renamed, no RoPE in Inkling)
- [x] sconv scaffold (sconv.cu — decode ring / prefill conv design comments)
- [x] decode_attention.cu (two shapes: paged-global 8KV / ring-512 16KV, rel-bias)
- [x] moe_grouped_gemm.cu (sigmoid gate + top-6 + norm_after_topk contract)
- [x] Initial main.cpp (skeleton serve entry)
- [x] pyengine scaffold (D5 Triton-first: 7 kernel stubs + 5 module contracts,
      populated by the build loop, NOT by hand)

### Review
- [x] Add JSON key per-field comment (//cfg: //top: //mtp: //quant: //derived:
      //choice: provenance tags on every constant)
- [x] Add a startup assertion block (load_and_verify_model_config + standalone
      verify_config binary; caught float-representation mismatch on first run)
- [x] Have agent review (reconciliation pass: fixed softmax→sigmoid gate comment,
      decode_attention signature drift, Inkling-Small→Inkling retarget)
- [x] Build sanity check (verify_config GREEN vs checkpoint; full CUDA stub
      build links; inkling_serve runs)
- [x] Push (repo synced; evidence logs curated into docs/logs/)

### Ground Truth Establishment
- [x] Precision pairing decided BY TEST: vLLM serves NVFP4 on B300 →
      canonical = NVFP4 vs NVFP4 (first token " Paris.", fingerprint
      g9243e0124-tp4)
- [x] KV anomaly investigated & CLOSED: vLLM allocation is window-aware
      (spec + empirical 0.21% gauge); 10× lever retired; win burden re-ranked
- [x] API format contract verified live (return_token_ids → choice-level IDs);
      parity gate patched to fail loudly
- [ ] Correctness Golden Generation (P2: transformers-NVFP4 probe on GPUs 4–7
      first; else all-8-GPU bf16 with vLLM down; commit goldens/)
- [ ] Capture vLLM baseline (P3: benchmark.py under both frozen configs →
      iteration-0 vllm ledger rows + experiments/baseline_vllm.json — the
      dashed lines on the final graph)
- [ ] Validate Parity gate vis-à-vis vLLM (correctness.py must pass against
      vLLM itself before it ever judges our engine)
- [ ] Full pipeline end-to-end dry run (spec → queue → dispatcher → worker →
      parity → bench → ledger, using vLLM as the candidate engine)
- [ ] Tamper-proof the gates (chmod -R a-w harness/ configs/ goldens/)
- [ ] 1 supervised Ralph-build end-to-end (ralph_build.sh 3, watched; sane
      first commit → overnight runs authorized)

## Stage 2 — Engine Bring-up (P4; executed BY the build loop)
- [x] Ralph kit ready: PROMPT_BUILD/PROMPT_EXPERIMENT, capped loop scripts
      (stop-file, no-progress halt, backpressure), PROGRESS.md seeded with
      27 tested items, plot_trajectory.py validated on synthetic data
- [ ] B0 scaffolding sanity (imports, Triton smoke on GPUs 4–7)
- [ ] B1 loader (shard index → census → dtype map → NVFP4 dequant → TP=4 plan
      → full load ≤150 GiB/GPU)
- [ ] B2 model graph (embed → rmsnorm → rel-bias → sconv → both attention
      shapes → MoE gate/experts → dense layers 0–1 → full-forward logits match
      transformers greedy)
- [ ] B3 KV + decode + scheduler + server (ring-512 oracle, batch invariance,
      OpenAI endpoint with return_token_ids)
- [ ] **B3.5 PARITY GATE GREEN — human-verified milestone**
- [ ] B4 first honest benchmark number in ledger (losing expected; case-study
      iteration 0 was 13.6% of vLLM)

## Stage 3 — Optimization Loop (P6)
- [ ] Dispatcher live; ralph_experiment.sh with queue backpressure
- [ ] Attack surface per D4 (launch-overhead deletion → skinny-expert MoE →
      sconv fusion → two-shape attention → scheduler specialization → MTP
      pair → comm overlap → Triton→CUDA ports as experiments → topology A/B)
- [ ] Every merge: one variable, parity green, > best + 2×noise (min 0.3%),
      ledger row with LEDGER_SCHEMA fields (label/mechanism/cache_key/...)
- [ ] **Crossing: ledger best > vLLM canonical baseline**

## Stage 4 — Validation & Write-up
- [ ] Same-hour sequential A/B (vLLM → ours → vLLM), NUMA note included
- [ ] MTP-ON tracked pair measured (D7)
- [ ] Fairness audit doc from BASELINE_NOTES + ledger
- [ ] plot_trajectory.py → trajectory graphs + ITERATION_LOG.md (generated
      from ledger only)
- [ ] Security ledger revocations (git token, HF token, doc password, root pw)
