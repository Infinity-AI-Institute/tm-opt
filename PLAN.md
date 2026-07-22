# tm-opt Trial Plan — Surpassing vLLM for Inkling on 8×B300

**Goal:** Replicate Infinity's "Surpassing vLLM with a Generated Inference Stack"
for Thinking Machines **Inkling** (975B MoE, 41B active; Inkling-Small unreleased,
so full model), on a Runpod 8×B300 node, with an agent-driven (Ralph loop)
optimization process and audit-grade fairness discipline.

**Strategy in one line:** Specialize everything vLLM must keep general (one model,
one chip, static expert placement, comm/compute overlap, exact-semantics hybrid KV,
MTP speculative decode) and let infrastructure — parity gate + noise-aware merge
gate + append-only ledger — decide what merges, so every claimed win survives audit.

**Fairness constitution (from OPTIMIZATION_INSTRUCTIONS.md, adapted):**
identical checkpoint + identical KV dtype on both engines; canonical workload frozen;
no computation reduction beyond the model's trained semantics; engines never run
simultaneously; ledger and vLLM reference cache never hand-edited; improvements
below noise floor (≥0.3% minimum) don't merge.

**Verified facts:** HF repos `thinkingmachines/Inkling` (BF16, ~2 TB) and
`thinkingmachines/Inkling-NVFP4` (~600 GB, Blackwell); vLLM has launch support
for Inkling. Node: 8×B300 SXM6 (275,040 MiB ≈ 269 GB usable each), full NV18
NVSwitch mesh, 2 NUMA domains (GPU0–3 / GPU4–7), CUDA 12.8 + 13.0 toolkits (13.0 at /usr/local/cuda-13.0, required for
sm_103a FlashInfer JIT and our kernel work), 3.9 TiB RAM (`/dev/shm` 1.9 TB, remountable larger),
`/workspace` = 1.5 TB volume (1.3 TB free), `/workspace/maverick` = prior occupant's
work, DO NOT ACCESS.

---

## Stage 0 — Access & Environment  `[mostly done]`

- [x] SSH access to pod via Runpod proxy (CLI)
- [x] GitHub repo `Infinity-AI-Institute/tm-opt` created; push working via
      fine-grained token (Contents: R/W); credential store on `/workspace`
- [x] Git identity fixed (commits attribute to adity-om, not pod owner);
      history rebased; README UTF-8 fix
- [x] Volume expanded 400 GB → 1.5 TB
- [x] Node sanity verified: 8×B300, NV18 mesh, driver 580.126.09, disk, RAM
- [x] Direct TCP SSH working in VS Code (pubkey appended to pod
      `~/.ssh/authorized_keys`; durable fix still open = key registered in
      owning Runpod account settings)
- [ ] Runpod team access OR contact handles pod edits (message sent re:
      maverick dir + access)
- [ ] `/workspace/maverick` disposition decided by Infinity (never opened by us)
- [x] Security ledger started: `docs/SECURITY_LEDGER.md` (git token, HF
      read-only token, root password if set, exposed doc password rotated)
- [ ] Research/agentic-inference access via research@infinity.inc confirmed

**Exit criteria:** stable dev loop (VS Code remote or CLI+tmux), push/pull working,
no unresolved access blockers.

---

## Stage 1 — Weights, Baseline & Measurement Contract  `[in progress]`

### 1A. Weights
- [x] `hf download thinkingmachines/Inkling-NVFP4 → /workspace/models/inkling-nvfp4`
      (551.35 GiB on volume; survives restarts)
- [x] Model card + `config.json` read and transcribed into `docs/ARCHITECTURE.md`
      and engine `ModelConfig` — and runtime-ENFORCED: `verify_config` binary
      checks every header constant against the checkpoint (green 2026-07-19)
- [ ] Decide BF16 handling for goldens: `/dev/shm` staging (remount larger if
      needed) vs NVFP4-reference-only (if calibration is documented as
      output-faithful, goldens can come from NVFP4 reference in Transformers/SGLang)

### 1B. Toolchain
- [x] Python venv on `/workspace` (survives restarts); pin versions in repo
- [x] vLLM pinned: `0.23.1rc1.dev1270+g9243e0124` (nightly wheel, archived in
      /workspace/wheels); Inkling arch registered; scipy dep hole patched
- [x] PyTorch cu130 confirmed
- [x] CUDA 13.0 toolkit installed alongside 12.8 — turned out to be a Stage-1
      BLOCKER, not a Stage-3 nicety: FlashInfer JITs fp4 kernels for sm_103a at
      serve time (`nvcc fatal: compute_103a` under 12.8). Serve script exports
      CUDA_HOME=/usr/local/cuda-13.0. apt lines added to pod_init.sh (container
      disk = ephemeral). FlashInfer/autotune caches persisted to
      /workspace/.flashinfer

### 1C. Canonical benchmark contract (freeze before any engine work)
- [x] Precision pairing DECIDED BY TEST 2026-07-19: vLLM serves Inkling-NVFP4
      on 4×B300, first greedy completion correct (" Paris.") →
      **canonical = NVFP4 vs NVFP4**. Fingerprint
      `vllm-0.23.1rc1.dev1270+g9243e0124-tp4`. Evidence: docs/BASELINE_NOTES.md
      + docs/logs/
- [x] Topology for vLLM baseline: TP=4 per official recipe (GPUs 0-3, GPUs 4-7
      idle → 4-GPU×2-replica shape viable). Flags in
      scripts/serve_vllm_baseline.sh. Benchmark variant must add
      --no-enable-prefix-caching (prefix cache is ON by default)
- [ ] Canonical workload frozen: ISL/OSL targets (long-context, KV-bandwidth-
      stressing), `ignore_eos=True`, range_ratio=0.5-style variable lengths,
      fixed seeds, prefix cache OFF both engines, matched gpu_memory_utilization,
      matched max_model_len
- [ ] Canonical concurrency measured empirically and frozen. NOTE (KV verdict,
      BASELINE_NOTES.md): KV memory is NOT binding at 16K — vLLM allocation is
      window-aware; concurrency is compute/scheduler-bound. Find the knee via
      throughput sweep at conc {8,16,32,64,128}
- [ ] `configs/parity_prompts.json` + `configs/bench_prompts.json` committed —
      FROZEN thereafter
- [ ] Canonical-parameter hash (cache key) implemented in harness; ledger
      records carry it; comparisons invalid on mismatch
- [ ] NUMA pinning added to benchmark/serve runners (`numactl` node 0 / node 1)
- [ ] Clock locking attempted (`nvidia-smi -lgc`); if blocked in container,
      documented + compensated with longer warmups

### 1D. Ground truth
- [ ] Correctness goldens generated (greedy tokens + logprobs on parity set),
      committed. Reference path decision first: BF16 won't fit 4 GPUs (~1.95 TB
      vs ~1.07 TB) → either all-8-GPU run with vLLM DOWN, or probe whether
      transformers loads the NVFP4 checkpoint directly on GPUs 4-7.
      Gate format facts already verified on live server: request needs
      `"return_token_ids": true`; IDs at CHOICE level (choice["token_ids"]);
      correctness.py patched to fail loudly if absent
- [ ] vLLM baseline captured: warmups = 2× conc, prompts = 5× conc, full
      report → `experiments/baseline_vllm.json`, committed BEFORE engine work
- [ ] Parity gate validated against vLLM itself (field-name mismatches fixed now)
- [ ] Full pipeline dry run: dummy spec through dispatcher → worker → gates →
      ledger, with vLLM as the serve command
- [ ] One supervised Ralph-build iteration observed end-to-end (prompt/permission
      issues fixed while watching)

**Exit criteria:** committed baseline number + goldens + frozen configs; the loop
demonstrated end-to-end; canonical cache key stable.

---

## Stage 2 — Engine Bring-up  `[the long pole]`

Human-owned design calls (write into ARCHITECTURE.md before Ralph runs):
- [ ] EP topology decision: NVFP4 across 4 GPUs × 2 replicas vs 8 GPUs × 1
      (269 GB/GPU real budget; KV headroom vs replica count trade-off)
- [ ] All-to-all / compute overlap scheme (NV18 mesh = topology-flat; overlap
      shared-expert GEMM with dispatch)
- [ ] Scope declaration in README: text-only serving v1; multimodal
      (patch encoder / dMel audio paths) explicitly out of scope

Ralph-build loop (nights, capped iterations, PROGRESS.md checklist as memory):
- [ ] `scripts/ralph_build.sh` + `agent/PROMPT_BUILD.md` + `PROGRESS.md` committed
- [ ] Gates untamperable: `chmod -R a-w harness/ configs/`; dispatcher
      auto-rejects candidates diffing harness/ or configs/; `main` protected
- [x] config.json parse → ModelConfig (fail-loud verify_config; build green)
- [ ] Safetensors loader: mmap staging, NVFP4 expert-major device layout,
      sharded across EP ranks
- [ ] KV cache: paged global + ring-buffer local layers (exact trained
      semantics — fairness-audit entry written)
- [ ] Continuous-batching scheduler: decode-priority, chunked prefill
- [ ] Reference kernels (correct-first): CUTLASS grouped GEMM MoE, reference
      attention, NCCL all-to-all
- [ ] OpenAI-compatible server (/v1/completions streaming, /health) —
      MUST implement `return_token_ids` + choice-level `token_ids` exactly as
      the pinned vLLM build does (verified format in BASELINE_NOTES.md)
- [ ] Morning review ritual: commit trail reviewed, checklist reordered,
      stuck items unstuck

**Milestone (human-verified, hand-committed):**
- [ ] **Parity gate passes on our engine against goldens — at any speed**

**Exit criteria:** engine serves the canonical workload correctly; first honest
(probably losing) benchmark number in the ledger.

---

## Stage 3 — Optimization Loop  `[rest of trial]`

Mechanics:
- [ ] Dispatcher running in tmux; experiments SERIALIZE on the node (depth-1
      queue; 2-wide only if 4-GPU×2 topology proven AND canonical A/Bs stay
      exclusive); engines never run simultaneously
- [ ] 2–3 `ralph_experiment.sh` loops live with queue backpressure; one
      hypothesis per patch; agents never run canonical benchmarks themselves
- [ ] Validation checklist (adapted §7) embedded in every ledger entry —
      worker auto-fills verifiable items, agent completes the rest
- [ ] ITERATION_LOG.md discipline: before/after numbers, commit hash, checklist,
      dead-ends documented with root cause
- [ ] Merge gate live: > best + 2× noise floor AND > 0.3% absolute

Attack surface (priority order per ARCHITECTURE.md rev 3 — KV lever RETIRED
after baseline investigation; re-rank weekly from ledger):
- [ ] CUDA graph coverage + Python/launch-overhead deletion on decode path
      (the original project's +38%-class win)
- [ ] MoE grouped-GEMM on skinny experts (N=3072): fused sigmoid gate → top-6 →
      norm_after_topk → permute; device-side work queue; NVFP4 tensor-core paths
      (CUDA 13 toolchain now in place)
- [ ] sconv fusion into producer/consumer kernels (4 extra ops/layer otherwise)
- [ ] Decode attention: two shapes only (paged-global 8KV / ring-512 16KV,
      head_dim 128, rel-bias fused), split-KV sizing, deterministic reductions
- [ ] Scheduler specialization: chunked-prefill sizing, token budget, decode
      priority tuned to the frozen canonical workload
- [ ] MTP speculative decode (draft-len sweep 1..8; vLLM shows ~2.7x from MTP —
      implementing it is mandatory for a fair fight, tuning it is the edge)
- [ ] All-to-all overlap with shared-expert compute (TP=4 comm on critical path)
- [ ] Topology A/B: 4×2 replicas vs 8×1 (whole-node exclusive experiment)
- [ ] KV parity + allocator characterization: hybrid ring/paged KV matches
      vLLM's window-aware allocation (DEMOTED from "concurrency lever" —
      vLLM already allocates window-aware; see BASELINE_NOTES.md)
- [ ] Per accepted experiment: roofline appended to docs/ROOFLINE.md
      (bytes moved, FLOPs, AI, ceiling, % of roofline, named limiter)

**Exit criteria:** ledger best exceeds vLLM baseline on the canonical workload,
sustained across reruns.

---

## Stage 4 — Validation & Write-up

- [ ] All loops stopped; node quiesced
- [ ] Same-hour sequential A/B: vLLM then ours (then vLLM again), same GPUs,
      same clocks — headline number
- [ ] Parity re-run on final build; long-context spot checks; sustained-load soak
- [ ] Fairness audit doc finalized (attention semantics, precision parity,
      workload equality — the three criticisms the original post received)
- [ ] Write-up generated FROM the ledger: baseline vs final, per-experiment
      attribution, proposed/rejected/merged counts (the "agents iteratively
      refine" evidence), roofline positioning, invalid-result disclosures if any
- [ ] Repo hygiene: maverick untouched (statement in write-up), credentials
      revoked per security ledger, repo visibility per Infinity's preference
- [ ] Deliverable review pass with fresh eyes (or Codex audit) before submission

**Exit criteria:** a reviewer can reproduce the headline claim from the repo alone.

---

## Standing Rules (apply to every stage)

1. `/workspace/maverick` is never opened, grepped, or deleted without Infinity's
   instruction.
2. Public sources only: vLLM docs, HF/Transformers, model card, CUTLASS. No other
   Infinity code.
3. harness/, configs/*.json, ledger.jsonl, vLLM reference cache: append-only or
   read-only; tampering auto-invalidates.
4. Everything of value lives in `/workspace` or the git repo; the pod is
   disposable; anything credential-like is on the end-of-trial revoke list.
5. One variable per experiment; correctness before performance; when a result
   looks too good, suspect the benchmark first (§4.4 history).
