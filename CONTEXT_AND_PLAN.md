# tm-opt — Context & Execution Plan (handoff for pod-side Claude)

> Paste-into-context document. State as of 2026-07-20. Everything here is
> either verified on this pod (evidence path given) or an explicit decision
> with rationale. When this document conflicts with older docs, this wins;
> update the older doc in the same commit.

## 0. Mission

Replicate Infinity's "Surpassing vLLM with a Generated Inference Stack"
(https://infinity.inc/research/qwen3-optimization) for **Thinking Machines
Inkling** (full model; Inkling-Small unreleased) on **one Runpod 8×B300 node**:

1. Establish vLLM benchmark performance (canonical, frozen, audit-grade).
2. Generate our own inference library — **Triton-first** for bring-up speed,
   with CUDA replacing Triton kernel-by-kernel during optimization.
3. Optimize in the style of an iteration log: one hypothesis per experiment,
   parity gate + noise-aware merge gate decide what lands.
4. Report as trajectory graph (tok/s vs iteration, vLLM baseline line) +
   per-iteration log — generated FROM the ledger, never hand-written.
5. Use all 8 GPUs well (2×TP=4 replicas baseline shape; 8×1 as tracked A/B).

Reference trajectory from the case study (calibrate expectations): their
iteration 0 was **13.6% of vLLM** (1,408 tok/s, from scratch); parity came
via iterations (memory layout → attention rewrite → batched prefill → …);
final +15.9% prefill-heavy / +34.3% decode-heavy at iteration 111 ("KV write
fusion"). A slow, correct first number is the expected starting point.

## 1. Hard facts (verified on this pod — do not re-derive, do re-verify if suspicious)

**Node:** 8×B300 SXM6 (sm_103a), 275,040 MiB (~267.7 GiB usable) each, NV18
NVSwitch full mesh, 2 NUMA domains (GPU0–3 ↔ cores 0–85, GPU4–7 ↔ 86–171),
3.9 TiB RAM, /dev/shm 1.9 TB. `/workspace` = 1.5 TB persistent volume;
everything else is wiped on pod restart (see `scripts/pod_init.sh`).
`/workspace/maverick` = prior occupant's data: **never open, grep, or delete.**

**Model:** `/workspace/models/inkling-nvfp4` (551.35 GiB, 34 shards +
mtp.safetensors). Config is checkpoint-verified by `./build/verify_config`
(green). Key shapes (full citations in `engine/include/tmopt/config.h`):
66 layers; hidden 6144; vocab 201024 (unpadded 200058, eos 200006);
**11 global attention layers** {5,11,…,65} (64Q/8KV/hd128) + **55 SWA layers,
window 512** (64Q/**16KV**/hd128); **no RoPE** — relative attention (d_rel 16,
extent 1024, log scaling α=0.1 floor 128000); **sconv** window-4 convs on
attn K/V/out + MoE out every layer; MoE 256 experts top-6, **sigmoid gate +
bias, norm_after_topk, route_scale 8.0**, 2 shared experts (sink), expert FFN
**3072** (skinny); layer 2 = dense MLP (24576); **MTP: 8 chained draft
layers**; ctx 1,048,576 (we serve at 16,384). NVFP4 weights, **bf16 KV**
(kv_cache_quant_algo "none"), exclude list stays bf16.

**Baseline:** vLLM pinned `0.23.1rc1.dev1270+g9243e0124` (wheel archived
/workspace/wheels). Serves via `scripts/serve_vllm_baseline.sh` (TP=4,
GPUs 0–3, port 8106, CUDA_HOME=/usr/local/cuda-13.0 — 12.8 cannot compile
sm_103a; FlashInfer JIT requires 13.0). First tokens correct. Cold init
1748 s (KV profiling ~18 min dominant); caches persisted at
/workspace/.flashinfer. Per-GPU: 137.46 weights + 106.27 KV + 2.53 graphs +
~2.5 other of 267.69. Full anatomy: `docs/BASELINE_NOTES.md`.

**KV verdict (closed investigation):** vLLM's Inkling integration is
window-aware in ALLOCATION, not just compute (`vllm/models/inkling/nvidia/`
`attention.py:200–211` returns SlidingWindowSpec/FullAttentionSpec per layer
type; sconv emits window-4 spec). Startup "10.48× concurrency" is a
worst-case reservation convention (171,639/16,384 exactly). Empirical: one
12K+4K request peaked KV gauge at **0.21%**. → **No ~10× KV lever exists.
Hybrid KV is parity work.** KV is not binding at 16K; canonical concurrency
is compute/scheduler-bound, set empirically. Evidence: BASELINE_NOTES.md.

**API format contract (verified live; our engine must match):** greedy via
/v1/completions, `temperature:0`, `logprobs:1`, **`"return_token_ids": true`**
→ token IDs at **choice level** (`choice["token_ids"]`), strings in
`logprobs.tokens`, floats in `logprobs.token_logprobs`. Parity gate
(`harness/correctness.py`) fails loudly if `token_ids` absent.

**Orientation (non-canonical):** single-stream decode ≳170 tok/s on vLLM.

## 2. Decisions log (each with the why)

| # | Decision | Rationale / evidence |
|---|---|---|
| D1 | Canonical precision: NVFP4 vs NVFP4 | vLLM serves NVFP4 on B300 (tested); BF16 (~2 TB) doesn't fit a 4-GPU replica |
| D2 | Baseline topology TP=4 (GPUs 0–3); 2-replica node shape | Official recipe; 137 GiB/GPU measured; GPUs 4–7 free |
| D3 | KV "concurrency lever" RETIRED | Spec + empirical verdict above |
| D4 | Win burden: launch-overhead deletion, MoE grouped-GEMM (skinny N=3072, fused sigmoid/top6/norm), sconv fusion, two-shape attention w/ fused rel-bias, scheduler specialization, MTP tuning, TP comm overlap | ARCHITECTURE.md rev 3; matches where the case study's verified wins came from |
| D5 | **Stage 2 re-scope: Triton-first Python engine** (`engine/pyengine/`), C++ skeleton retained as CUDA-port roadmap + verify_config provenance gate | Trial spec says "likely in Triton for ease… expect most Triton replaced by CUDA"; case study started at 13.6% and iterated; C++-first bring-up is a parity-schedule risk on a 975B MoE |
| D6 | Two frozen workloads, like the case study: prefill-heavy ISL≈8K/OSL≈1K and decode-heavy ISL≈1K/OSL≈8K; headline = decode-heavy | Mirrors their published protocol (BS=88, no spec decode footnote) |
| D7 | Canonical configs run **MTP OFF both engines**; MTP-ON tracked as separate config pair | Their footnote: "no speculative decoding"; vLLM gets ~2.7× from MTP — excluding it from headline while reporting both is the defensible move |
| D8 | Engines never run simultaneously during measurement; experiments serialize | Fairness constitution §4.3 |
| D9 | Multi-node out of scope (one pod); multi-GPU TP=4 NCCL is our version of the "multi-node challenge" | README scope note |

## 3. Repo state (Infinity-AI-Institute/tm-opt, working copy /workspace/tm-opt)

Done & green: engine config provenance (config.h cited per-field +
verify_config runtime check), CUDA stub build (sm_100 under CMake; 13.0
toolchain installed for sm_103a), harness (correctness/benchmark/ledger/
dispatcher/worker) committed with token_ids gate fix, serve script
self-logging to /workspace/logs (evidence logs curated into docs/logs/),
docs current (ARCHITECTURE rev 3, BASELINE_NOTES, SECURITY_LEDGER, PLAN).

Not yet: canonical freeze, goldens, committed baseline numbers, pyengine,
Ralph kit, trajectory plot script.

## 4. Execution plan from here (strict order; each step has an acceptance test)

### P1. Freeze the measurement contract  → `configs/canonical.json`
- Benchmark serve variant: add `--no-enable-prefix-caching`; pin KV via
  `--kv-cache-memory` (log-suggested 111231943107) rather than util fraction.
- Two workloads (D6), fixed seeds, `ignore_eos:true`, range_ratio 0.5.
- Concurrency: sweep {8,16,32,64,128} per workload on vLLM, pick the
  throughput knee, freeze. NUMA-pin the load generator to the replica's node.
- Emit cache-key hash (SHA of canonical.json) — every ledger row carries it.
- **Accept:** two frozen JSON configs + hash committed; sweep data in docs/logs/.

### P2. Goldens + gate validation
- Try NVFP4-in-transformers on GPUs 4–7 first (vLLM may stay up but idle —
  correctness work, not measurement). If transformers can't load NVFP4:
  tear vLLM down, all-8-GPU bf16 reference from /dev/shm staging.
- Greedy tokens + logprobs for configs/parity_prompts.json → goldens/
  committed. Then run the gate against live vLLM: must PASS.
- **Accept:** `python harness/correctness.py --endpoint http://localhost:8106`
  green against goldens.

### P3. Committed baseline
- benchmark.py full protocol (warmups 2×conc, prompts 5×conc) both workloads
  → `experiments/baseline_vllm.json` committed. This is the wall target.
- **Accept:** baseline numbers in repo + BASELINE_NOTES updated.

### P4. pyengine bring-up (the Ralph BUILD loop's job)
Target layout `engine/pyengine/`:
- `loader.py` (safetensors → NVFP4 tensors on 4 GPUs, expert-major),
- `model.py` (66-layer graph: rel-attn two shapes, sconv, MoE sigmoid-top6,
  dense layer 2, embed-norm, unembed),
- `kernels/` (Triton: rmsnorm, sconv, moe_gather_gemm, decode_attn_global,
  decode_attn_swa, rel_bias),
- `kv.py` (paged global + ring-512 SWA + sconv state — parity semantics),
- `scheduler.py` (continuous batching, chunked prefill),
- `server.py` (OpenAI /v1/completions incl. return_token_ids, /health).
Bring-up milestones the loop must hit IN ORDER, each with a test:
B1 loader loads + layer-0 forward matches reference slice (tolerance);
B2 single-token greedy matches vLLM on 5 prompts; B3 parity gate green on
full parity set (**human-verified milestone**); B4 first honest benchmark
number in ledger (losing is expected — case-study iteration 0 was 13.6%).
- **Accept:** B3 green + B4 row in ledger.

### P5. Ralph kit (write before P4 runs; the loop executes P4)
- `agent/PROMPT_BUILD.md` — bring-up brief: read PROGRESS.md, pick next
  unchecked item, small diffs, run that item's test, update PROGRESS.md,
  commit. Never touch harness/, configs/, goldens/.
- `agent/PROMPT_EXPERIMENT.md` — one hypothesis per iteration: read ledger
  tail + ROOFLINE.md, propose spec, implement patch in worktree, run parity
  locally, submit to queue; dispatcher/worker run canonical gates.
- `scripts/ralph_build.sh` / `ralph_experiment.sh` — capped-iteration loops
  invoking `claude -p` with the prompt file; PROGRESS.md is the loop's memory.
- Tamper-proofing: `chmod -R a-w harness/ configs/ goldens/`; worker rejects
  any candidate whose diff touches them.
- **Accept:** one supervised build iteration observed end-to-end.

### P6. Optimization loop → crossing → report
- Attack surface order per D4; Triton→CUDA ports are themselves experiments.
- MTP-ON config pair measured once engine has MTP (mandatory for the
  "beats vLLM at its best" claim; separate from headline per D7).
- `scripts/plot_trajectory.py`: ledger → case-study-style chart (tok/s vs
  iteration, dashed vLLM baseline, annotation per merged experiment) +
  ITERATION_LOG.md generation.
- Stage 4 close: same-hour A/B (vLLM → ours → vLLM), fairness audit doc,
  write-up from ledger, security-ledger revocations.

## 5. Standing rules (unchanged, binding on every iteration)

1. /workspace/maverick untouched. 2. Public sources only. 3. harness/,
configs/, goldens/, ledger.jsonl, vLLM reference cache: read-only/append-only;
tampering auto-invalidates. 4. One variable per experiment; correctness
before performance; too-good results → suspect the benchmark first.
5. Engines never measured concurrently; NUMA-pin; same-hour A/B for claims.
6. Every merged number carries: commit hash, canonical cache-key, noise
floor, validation checklist, log path in docs/logs/.
