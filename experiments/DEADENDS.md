# Dead ends — post-mortems for rejected experiments

Every rejected ledger row gets one paragraph here before the next hypothesis
is proposed (PROMPT_EXPERIMENT rule). A rejection is data: record what the
pipeline said, why, and what the next iteration should do differently.

## warmup-0000 — pipeline warm-up (rejected 2026-07-28, iteration 1)

Designed rejection, and the design worked: an unmodified engine (commit
HEAD) was pushed through the full dispatcher→worker pipeline to prove the
Stage-3 machinery before any real hypothesis spent GPU time on it. Every
stage behaved: D13 teacher-forced parity ran and PASSED inside the envelope
(agree 0.9587 = 767/800 vs floor 0.93, delta_mean 0.0453 vs cap 0.0583),
the protocol-aware bench auto-selected timeboxed(1800s, conc=8, osl_cap=128)
below the canonical floor, reproduced iteration-0 throughput exactly
(1.6 tok/s, 24 reqs × 128 tok, zero failures), and the merge gate REJECTED
it with the correct arithmetic (`tok/s 1.6 <= threshold 1.6 = best +0.3%`)
— a full-schema rejected row is now chartable in the ledger. Lessons worth
money: (1) rejection rows carry parity + protocol fields, so a rejected
mechanism still yields calibration data; (2) worker wall was ~75 min
end-to-end, of which the 800-request tf gate took ~30 min at ~2.3 s/request
under the worker's own server launch — that pace is the budget anchor for
every future pre-gate estimate (and the 3600 s parity timeout has ~2×
headroom at it); (3) nothing about the engine itself was learned, by
design — decode serialization remains the whole iteration-0 gap. Next
iteration: attack surface #1 (batched decode), through this now-proven
pipeline.

## CORRECTION 2026-07-29 — warmup-0000 lesson (2) is WRONG; gate pace is ~5.5 s/req everywhere

Lesson (2) above mis-derived the worker's gate window by assuming the
ledger row's `measured_at` is stamped at bench END. It is stamped BEFORE
the bench: worker.py builds `row_common` (including `measured_at`)
immediately after `run_parity` returns, then calls `run_timeboxed`
in-process (worker.py:126-139; benchmark.py's own end-stamp in its
`main()` never runs under the worker). Corrected warm-up timeline, every
edge independently pinned: dispatcher claim 01:51 (dispatcher_20260728.log
mtime) + ~2 min load → gate 01:53 → 03:06:13 (`measured_at`) = **73 min
for 800 requests = 5.5 s/req**; bench 03:06 → 03:49 (≈666 s warmup wave +
1939.4 s box+drain = 43.4 min) lands exactly on the worker log's
final-flush mtime 03:49 (queue.rootowned.bak/warmup-0000.log; 758 bytes
< one pipe buffer, flushed at exit). Worker end-to-end was ~118 min, not
~75. Corroboration — every directly-timestamped gate run agrees, across
builds and launch contexts: main build 800 reqs 13:39→14:51 on 07-27 =
5.4 s/req (tf_pyengine_d13_20260727_1339.log name→mtime window); exp-0001
build 708 reqs @ 5.45 s avg (prior session's serve-log trace); exp-0001
K=4 200 reqs ≈ 18 min (tf_pyengine_exp0001_k4 + serve exp0001b logs,
03:13→03:31). Consequences: (a) the "unexplained 2.4× delta" flagged in
exp-0001's spec NEVER EXISTED — there is no worker-vs-session pace gap and
no TF-path regression in the batched-decode build (its gate requests are
max_tokens=1 = the untouched per-sequence prefill path; the diff only ADDS
decode-batch functions); (b) the true budget anchor is ~5.4–5.5 s/req
(K=4 ≈ 18 min, full K=16 ≈ 73 min ≈ 4,380 s), and worker wall per
experiment is ~2 h; (c) the 3600 s parity timeout added in 952c6f8 —
sized from this entry's wrong ~30-min figure — is BELOW the true full-gate
wall, so once the known sh() one-liner lands, EVERY worker gate (exp-0001's
rerun first) dies at 60 min with run_parity's "treat as parity red"
RuntimeError and vanishes without a ledger row. Human package updated in
the 2026-07-29 PROGRESS.md loop note: the worker.py fix needs a SECOND
one-liner, timeout 3600 → ≥5400 s (7200 recommended).

## exp-0002 abandoned variant: pair-bmm MoE dispatch (2026-07-29, in-session — never queued)
Hypothesis was that moe_experts' per-hit-expert Python loop (~15 eager ops
per expert + a unique().tolist() sync per MoE layer, ~44 hits at decode
batch 8) was launch-bound; a pair-parallel bmm dispatch (one batched
dequant + fixed ~30-kernel schedule per layer over all T*top_k pairs)
would delete it. Implementation was numerically sound (synthetic vs loop:
max|d| 2e-3 bf16 accumulation-drift class; PackedExperts.gather bitwise
vs __getitem__; deterministic) but t_batched measured decode speedup
DOWN, 1.60x -> 1.39x (decode wall 55.4 -> 64.0 s, log
/workspace/logs/t_batched_exp0002_2026-07-29.log). Micro profile at real
shapes explains it: the loop was never launch-bound — moe_experts at T=8
costs 51.9 ms/layer of which ~49.8 ms is the on-demand NVFP4 dequant
itself (eager dequant_nvfp4 materializes int64 LUT indices + fp32
temporaries, ~750 MB traffic per expert w13 for a 75 MB result, 0.689
ms/expert); pair-bmm keeps that dequant cost, adds pair-duplication and
multi-GB transient allocations under real memory pressure. LESSON: the
batched decode step wall (~2.9 s at B=8) is ~launch-amortized already;
its limiter is dequant TRAFFIC (~44 hits x 64 layers x ~1.1 ms ~ 3.1 s).
That insight became exp-0002-triton-dequant (bit-exact one-pass kernel,
0.689 -> 0.089 ms/expert). A future grouped/fused dispatch should be
re-tested only AFTER the dequant floor is gone — and measured against
the then-current step profile, not launch-count intuition.

## 2026-07-30 — batched group prefill (exp-0009 groundwork; refuted in local pre-gate, never queued)

Hypothesis: after exp-0008 (45.3 tok/s) the bench runs in lockstep cohorts
(osl_cap=128 exactly -> all 64 co-residents retire together), so the 64
replacement prefills serialize at 1.92 s each = 122.9 s of the measured
180.8 s cycle (68%); prefilling co-admitted prompts as ONE right-padded
batched 66-layer traversal was expected to amortize what exp-0005 called a
dispatch-bound engine. Implementation was sound (per-row state seeding via
the per-seq append/seed calls on [:len_i] slices; CPU synthetic A/B
bitwise 24/24 state tensors, scheduler-level token identity vs a
serial-fallback arm; GPU determinism arm bitwise; E2E 8x8 tokens 64/64)
but the TIMING was flat: batched-8 17.08 s vs serial 18.45 s (1.1x),
32+32-group cohort 145.3 vs ~148 s serial (1.0x); seeding is free (2.54
vs 2.53 s with/without states). Logs: t_prefill_batch_{cpu,gpu}_exp0009,
t_prefill_debug_exp0009 (all 2026-07-30, /workspace/logs/). CAUSE (torch
.profiler on the resident model, t_prefill_profile_exp0009 log): prefill
is not dispatch-bound at all — 93.4% of prefill CUDA time is the exp-0003
grouped packed-NVFP4 GEMM (_gate_up_silu 16.0 + _down_scale 5.9 = 21.9 of
each 24.3 ms MoE layer at T=574; attention is 1.1 ms), whose cost scales
with ceil(rows_per_expert/BLOCK_M) full-matrix in-register dequants per
hit expert — expert-count-bound, not token- or launch-bound. Batching
prompts moves pairs between calls without reducing that dequant total
(batched-8 at BLOCK_M 16: ceil(8x13.5/16)=7 m-chunks vs 8 serial 1-chunk
calls = the measured 1.1x). LESSON: exp-0005's "host enqueue = 100% of
step wall" was a DECODE-step fact; it never licensed conclusions about
prefill. The insight became exp-0009-prefill-gemm-blockm (prefill-scale
BLOCK_M 32: 1.5-1.9x on the dominant term). Batched group prefill is
worth RE-TESTING only after that merges — with dequant amortized by
BLOCK_M 32, a 32-row group does ~17 m-chunks vs 32 serial full dequants
(~1.5-2x further on the prefill term), and the groundwork (implemented,
validated, bitwise gate path preserved via singleton fall-through) is
saved at /workspace/logs/prefill_batch_groundwork_exp0009.diff against
tree eab6676. The endgame for the dequant ALU floor itself is native-FP4
tensor-core MMA (tl.dot_scaled) — a separate, larger experiment.

## 2026-07-30 — tl.dot_scaled native-NVFP4 MMA: dead on Triton 3.6 / sm_103a (exp-0010 iteration)
The exp-0009 ranked-next attack "native-FP4 tensor-core MMA to delete the
dequant ALU floor" was tried first this iteration and is unimplementable on
the installed toolchain: Triton 3.6's dot_scaled API accepts NVFP4 exactly
(e2m1 operand + float8_e4m3fn scales at group 16 — semantic.py
verify_scaled_shape), but EVERY bf16-lhs x e2m1+e4m3-scale-rhs shape tried
((16..128) x (64..128) x (64..128)) dies in the same place: an MLIR
internal assert (`isIntOrIndex` in DenseElementsAttr::get) inside the
TritonGPUAccelerateMatmul pass — a compiler bug in the cuda:103 backend,
same family as the BLOCK_M-64 shared-layout crash exp-0009 logged. Repro
scripts + full dumps: /workspace/logs/t_ds_probe_exp0010.py,
/workspace/logs/t_moe_ds_exp0010.py. Do NOT re-try dot_scaled on Triton
3.6; the attack stays live only as (a) a Triton-upgrade experiment in a
separate venv or (b) the CUTLASS/CUDA port of moe_gemm (D5 roadmap). What
IS available on this toolchain — the hardware convert instruction
cvt.rn.f16x2.e2m1x2 via inline asm, bit-identical to the _e2m1 chain —
shipped as exp-0010 (1.21x both kernels; the residual floor is now the
convert pipe at ~2.5 cvt-class ops/element, which only native MMA removes).
One trap for the record: tl.inline_asm_elementwise with pack=4 on uint8
inputs SILENTLY mispacks whenever a thread owns fewer than 4 elements
(passthrough probe in /workspace/logs/t_asm_map_probe2.py) — use pack=1
with a widened b32 input, or guard with a full-tile bitwise check.

## exp-0011-insitu-phase-profile (rejected 2026-07-30, 61.4 vs bar 70.2 — by design)
Post-mortem (control row, rejection expected): the experiment delivered its
deliverable — /workspace/logs/insitu_phase_exp0011_2026-07-30.log, in-situ at
conc 64 on the exp-0009 engine: dec_enq 107.1 s of dec_wall 108.4 s over 256
steps (host enqueue 98.8% of step wall, exposed GPU tail 1.2%), glue+idle
small — which adjudicated fork option (a): host-dispatch deletion. The
INTERPRETIVE LESSON that must outlive it: host-enqueue share of WALL cannot
distinguish "host-bound with idle GPU" from "host and GPU near-parity" —
when the GPU trails the enqueue closely, deleting dispatch yields ~nothing.
exp-0012's A/B resolved exactly this: the graphed step (host ~free) ran
473 ms vs eager 476.6 ms — the decode step was ALREADY GPU-bound (58% grouped
NVFP4 GEMMs, 28% attention gather copies; t_graph_profile_exp0012 log). A
phase profiler that wants to rank dispatch-deletion must also measure GPU
busy time per phase (CUPTI/events), not just host walls and sync tails.

## exp-0012 implementation note: multi-GPU torch.cuda.graph capture
torch.cuda.graph without an explicit `stream=` uses a CLASS-LEVEL
default_capture_stream created on whichever device is current at the FIRST
capture in the process (torch/cuda/graphs.py). Every later capture on a
DIFFERENT device then records on the wrong device's stream: small probes
"succeed" with an EMPTY graph (UserWarning "graph is empty ... wrong device
or stream"), real segments die with cudaErrorStreamCaptureInvalidated. Fix:
keep one torch.cuda.Stream() per device and pass it explicitly (pool sharing
also wants the same stream per pool). Also for the record: torch.bincount is
capture-illegal (data-dependent output size -> hidden device sync);
torch.searchsorted on the pre-sorted list yields bit-identical segment
offsets and is capture-safe.

## 2026-07-30 — exp-0015 GROUNDWORK (no spec this iteration): native NVFP4 MMA is AVAILABLE — torch._scaled_mm FP4xFP4 works on sm_103a
Supersedes the exp-0010 dot_scaled entry's "stays live only as (a) Triton
upgrade or (b) CUTLASS/CUDA port": there is a THIRD, already-installed path,
probed end-to-end this iteration (script + full output:
/workspace/logs/t_fp4mma_probe_exp0015.py, t_fp4mma_probe_exp0015_out.log;
run on GPU 4 while the worker held 4-7 for exp-0014).
FACTS: (1) torch 2.11.0+cu130 `_scaled_mm(fp4, fp4.t(), sfa, sfb)` runs
native FP4xFP4 tensor-core MMA on sm_103a: checkpoint packed bytes usable
AS-IS (low-nibble-first confirmed by one-hot probe, arm B), scales need a
one-time load-side repack to the cuBLAS 128x4 blocked swizzle (to_blocked in
the probe), outputs match fp32 emulation at small AND prefill shapes (rel
~2.6e-3 = bf16-out class), bitwise-deterministic across reruns, CUDA-graph
capture-safe (explicit-stream capture+replay verified, arm F). (2) PERF:
dense 4416x6144x6144 (a T=736 prefill's w13 pair-rows folded dense) = 0.053
ms = 6.2 PFLOP/s vs ~13.5 ms for the same layer-class work in the Triton
W4A16 kernel — the exp-0010 convert-pipe ALU floor (~2.5 cvt-class
ops/element) deleted, confirming the decode GEMM's 5.7x-off-roofline gap
(266.4 ms/step -> ~47 ms traffic-bound) is recoverable. (3) torch
`_scaled_grouped_mm` REJECTS fp4 (fp8 rowwise/tensorwise + mxfp8 scales
only, arm C) — no torch grouped path for the graphed ragged-M decode MoE;
a per-expert dense loop is host-enqueue-bound (5.6 ms/256 calls, arm H:
only ~2x on prefill w13, not the 100x). (4) CUTLASS DSL 4.5.2 IS installed
with blockscaled layout utils (arm I) — the decode-capable kernel (ragged M
from device-side problem sizes, capture-safe, FP4xFP4 grouped blockscaled)
is a DSL port, i.e. exactly D5's Triton->CUDA roadmap, now de-risked to
"port one example kernel".
FAIRNESS: W4A4 on routed experts is the REFERENCE'S OWN numeric recipe, not
a computation reduction — the pinned vLLM serves these layers through the
ModelOpt NVFP4 fused-MoE method with input_scale = input_amax/(448*6)
(site-packages vllm/models/inkling/nvidia/moe.py:562; the checkpoint ships
`.input_amax` per quantized weight — our loader currently ignores it), and
the D11 goldens + D13 envelope were generated by that path. Our W4A16
engine runs MORE precision than the reference today.
RISKS for the implementing iteration: (a) model-level TF-envelope drift
from activation quant — mitigate with recipe-exact device-side quant
(hardware cvt.rn.satfinite.e2m1x2.f32; the probe's bucketize emulation may
differ on exact ties); headroom is delta_mean 0.05065 now vs 0.0583 bar;
(b) the activation-quant kernel must emit the swizzled scale layout
directly (an eager to_blocked repack per call would eat the decode win);
(c) `_scaled_mm` exposes no alpha — apply the per-expert scale2 x
input_scale product in the existing routing-weight epilogue (envelope-class
rounding change only).
PROPOSED SPLIT (one variable each, both citing this note): exp-0015a =
prefill-side W4A4 grouped blockscaled GEMM (attacks the ~61 s prefill term,
~47 s if exp-0014's batched prefill lands); exp-0015b = graphed decode step
W4A4 (attacks 266.4 ms of the 322.7 ms step -> ~103 ms, decode term 30 s ->
~13 s). Combined ceiling ~2-3x on the cycle (91.4 s -> ~38 s, ~215 tok/s
class) — the largest single lever left on the board.
WHY NO SPEC: session hard-limit (~90 min) minus GPUs 4-7 being held by the
worker's exp-0014 run (its bench-warmup crashed mid-session — see
queue/exp-0014-batched-prefill.log; server left wedged, next iteration
likely owes that post-mortem once the ledger row lands) left no window for
engine integration plus the ~25-min local TF pre-gate; per the budget rule
the probe evidence is committed instead of a red or rushed spec.
ADDENDUM (same iteration, arms J+K; logs t_fp4mma_realwt_exp0015_out.log,
t_cutedsl_probe_exp0015_out.log in /workspace/logs): J — real-weight risk-(a)
quantification on layer-3's real experts with the real input_amax (bf16
scalar 4.375 -> input_scale 1.6276e-3; NOTE for the implementer: checkpoint
`.scale` is ALREADY float8_e4m3fn [E,R,K/16], scale2 fp32 [256]): per-GEMM
W4A4-vs-W4A16 delta rel_mean 9.5e-2, cos 0.9955, uniform 8/8 experts. This
LOWERS the gate risk rather than raising it: the D11 goldens/D13 envelope
come from vLLM running W4A4, so today's PASSING W4A16 engine (delta_mean
0.05065 vs bar 0.0583) already carries the full A16-vs-A4 systematic
difference through all 64 MoE layers — adopting the reference's A4 recipe
moves the engine TOWARD the goldens, provided the quant recipe is
vLLM-exact (block-16 e4m3 through input_scale = amax/(448*6), RTNE e2m1 via
the hardware cvt). K — decode-port toolchain: CUTLASS DSL JITs AND runs on
sm_103a (file-based only, no REPL), cutlass.utils.blockscaled_layout ships
Sm103BlockScaledBasicChunk, and cute.nvgpu.tcgen05 exposes MmaMXF4NVF4Op /
BlockScaledMmaOp — the grouped blockscaled decode kernel (exp-0015b) is
toolchain-supported end to end.

## exp-0014-batched-prefill (rejected 2026-07-30, 86.0 vs bar 90.0)
Post-mortem: parity PASSED bit-identically (0.955/0.05065/0.600919 — the
singleton fall-through preserved the gate path exactly as designed), but the
mechanism delivered ~ZERO in situ: per-bench-thread pace was 7.05 reqs/600s
(430 reqs / 61 live threads) vs exp-0013's 7.00 (448/64) — +0.7%, noise —
and the whole -4.1% headline deficit is exactly the THREE bench worker
threads that died at warmup (three ConnectionResetError tracebacks in the
worker log; 89.7 x 61/64 = 85.5 ~ the measured 86.0). Two lessons. (1) The
engine-level 1.30x / HTTP 1.15x grouping gains evaporate under the bench's
real arrival pattern: replacement prefills arrive per-completion, spread
over the retiring cohort's drain, so the 150 ms idle-edge window still saw
mostly singleton groups — grouping must be triggered by the SCHEDULER'S OWN
cohort-retire signal (it knows 64 seqs just freed), not by a wall-clock
accumulation window racing HTTP arrivals; re-attack only as part of
scheduler specialization (#8) AFTER the GEMM native port (exp-0015a/b),
where prefill GEMM time shrinks ~4x and grouping matters less anyway.
(2) The resets happened BEFORE any status line was written (client died in
_read_status), i.e. accepted-then-killed in the handler or accept-backlog
overflow (server.py uses ThreadingHTTPServer with the socketserver default
request_queue_size=5) under the 64-simultaneous warmup blast; the server's
own stderr is not captured by the worker anywhere we can read, so the
server-side trace is LOST — any future engine change should raise
request_queue_size alongside, and a worker-side server-log capture would
have named the culprit (harness is read-only for the loop; noting, not
fixing). The exp-0014 branch's measured engine-level 1.30x and its
length-sorted packing survive as reusable groundwork on the
exp-0014-batched-prefill ref.
ADDENDUM 2 (arm L): the shared prerequisite for BOTH exp-0015a/b is built
and validated — /workspace/logs/fp4_quant_groundwork_exp0015.py (self-test
output fp4_quant_groundwork_exp0015_out.log): a Triton NVFP4
activation-quant kernel (hardware cvt.rn.satfinite.e2m1x2.f32 with pack=1,
asm operand order $1=high/$2=low probe-verified; fp32->e4m3 via Triton's
native cast; scales written DIRECTLY in the cuBLAS 128x4 blocked swizzle,
off = ((G*32 + (m%128)%32)*4 + (m%128)//32)*4 + kb%4, G = (m//128)*nc +
kb//4). Bitwise-matches the torch reference on packed bytes AND swizzled
scales at all tested shapes incl. partial 128-row blocks; _scaled_mm fed by
kernel vs reference is bitwise identical; deterministic across reruns;
25.5 us at 4416x6144 (2.7 TB/s) vs the 53 us GEMM it feeds. Constraint for
the integrator: the k-loop is unmasked — BLOCK_K must divide K (engine
shapes 6144/3072 are fine; the wrapper auto-picks). Copy into
engine/pyengine/kernels/ in the exp-0015a worktree; do NOT land on main
outside the dispatcher path.
ADDENDUM 3 (arm M — the decode vehicle EXISTS and RUNS): the public CUTLASS
repo ships a purpose-built MoE example, examples/python/CuTeDSL/cute/
blackwell/kernel/moe/torch_scaled_grouped_mm.py — "Scaled Grouped GEMM for
MoE operations with block scaling (MXFP8, MXFP4, NVFP4)", torch 2Dx3D offs
interface (tokens_sum,K)x(E,K,N), warp-specialized (scheduler warp +
tcgen05.mma.block_scale + epilogue with GLOBAL_SCALE multiply built in —
risk (c) solved in-kernel), with an sm103-specific variant next to it
(sm103_grouped_blockscaled_gemm.py; "SM103 only supports Float4E2M1FN" —
NVFP4 is THE native dtype on our arch). Fetched to
/workspace/logs/cutlass-dsl-examples (BSD-3, package tree reconstructed
under cute/). RAN it on GPU 4: `--kind nvfp4 --tokens 384 --experts 256
--top_k_select 6 --hidden 6144 --intermediate 6144` -> "Validation PASSED
(exact match)", DSL kernel 408 us at default (128,128,128) tiles vs the
current Triton W4A16's ~4.2 ms/layer decode share. exp-0015b is now a
WIRING job (adapt tensor prep to our PackedExperts + pooled step, verify
capture-safety + determinism arms, tune tiles), not kernel R&D. Constraint
flagged in source for the integrator: divisibility_ab=32 for fp4 (K=6144,
N=6144/3072 all satisfy); check per-group M handling ("scheduler handles
fake dimensions by computing token_offset from offs") against ragged 1-2
row decode segments early.

## exp-0018-pipelined-prefill (rejected 2026-07-30 — INFRASTRUCTURE, not the
## hypothesis; NO ledger row was written)
Post-mortem (exp-0019's iteration, which found and cleared the cause). The
worker never measured anything: its log stops after `git worktree add`, and
600 s later `RuntimeError: engine never became healthy; see serve.log`. The
engine server it launched died instantly — the worker's only child was an
already-defunct `sh`, no CUDA context ever appeared on GPUs 4-7, and the
verdict archived as `{"accepted": false, "reason": null}`. CAUSE: the
PREVIOUS iteration's local D13 pre-gate server (PID 331687, `python -m
engine.pyengine.server --port 8218`, started 16:45:21) was never killed when
that iteration ended. It was reparented to init (PPID 1) and sat on GPUs 4-7
holding 171-183 GiB of each of the four cards. The dispatcher handed
exp-0018 to those same GPUs at 16:52 (worker pins CUDA_VISIBLE_DEVICES=
4,5,6,7, harness/worker.py:43); the candidate server needs ~137 GiB/GPU of
weights and got ~85-95 GiB of free memory, so it OOMed before writing a
health endpoint. Killing the orphan at 16:56 returned all four cards to
4 MiB, but wait_healthy's 600 s window had already been spent losing.
THREE lessons, in order of cost. (1) A local pre-gate server is the loop's
own property and MUST be killed inside the same foreground Bash call that
starts it — a trap/kill pair, not a hope. exp-0019 ran its pre-gate that way
and verified 4 MiB/GPU afterwards. Any iteration that leaks one silently
destroys the NEXT experiment the dispatcher schedules, and the failure is
maximally confusing because it appears as a fault in the innocent
experiment's own code. (2) The failure mode is invisible from the queue log:
serve.log lives inside the worker's root-owned worktree, which is removed by
`git worktree remove --force` on the way out, so the OOM traceback is
destroyed with it — exactly the "server-side trace is LOST" gap exp-0014's
post-mortem already flagged. Diagnosis was only possible from live process
state (PPID 1, zombie child, idle GPUs) because it was caught while running.
(3) exp-0018's MECHANISM IS UNJUDGED and should be resubmitted unchanged:
its D13 gate was green and bit-identical to the accepted rows, its serial-vs-
pipelined token streams were exactly equal, and its two determinism arms
passed — none of that was contradicted, and none of it was measured.

## exp-0020-pipelined-prefill-rerun (rejected 2026-07-30, 222.6 vs bar 223.6)
Post-mortem, written by exp-0021's iteration, which then measured the cause.
The resubmission worked as infrastructure — the run completed, the ledger row
exists (iter 21), and the mechanism was finally judged: exp-0018's
sync-elimination is worth NOTHING in situ (222.6 against a 222.9 best, i.e.
-0.13%, inside noise). Its own isolated arm was honest (a mixed step went
1.409 -> 1.373 s, and 36 ms x ~180 mixed steps is ~6 s of a 620 s box = 1%,
which is simply too thin to clear a 0.3% bar reliably); what was wrong was
the MODEL underneath it, and the model was wrong in a way that mattered for
the two experiments queued behind it. exp-0018 assumed the traversal was
GPU-bound and merely sync-pinned, so that deleting the 23 device->host syncs
would let successive traversals overlap across the layer-split GPUs. The
syncs really were deleted (verified below), but no overlap followed, because
a prefill traversal is not GPU-bound at all. LESSON, and it is the useful
one: "I removed the thing that blocks overlap" is not evidence of overlap.
The arm that would have caught it costs one line — time the enqueue call's
RETURN separately from the sync that follows it — and exp-0018 never ran it.
Its sync-free traversal is still in the tree (exp-0021 builds on it and needs
it) and its kill switch still works; it just is not, by itself, worth a row.

## exp-0021 arm 1: prefill MICROBATCHING (measured, negative, not submitted)
The hypothesis this iteration started with: now that the traversal is
sync-free (exp-0018), feed the layer-split pipeline more traversals by
cutting a released cohort into consecutive sub-groups, since exp-0017's
two-point fit (per-group wall = A + B*G, A = 183.7 ms flat, B = 196.8 ms/seq)
says (N+3)/4 * (A + 6B/N) beats 4*(A+6B)/4 for N = 2..6 at any overlap
efficiency above ~32%. Implemented as a scheduler-only knob
(_prefill_groups cuts each budget-packed group to _MICROBATCH rows) and
measured on the real resident model, interleaved arms, one load
(engine/pyengine/tests/t_microbatch_prefill.py; log
/workspace/logs/t_microbatch_prefill_exp0021_out.log). RESULT, mixed step
(6-prompt cohort landing on 8 already-decoding residents): one group
1.354 s -> two groups of 3 1.544 s -> three groups of 2 1.710 s; the
prefill-only step moves the same way (1.529 -> 1.859 -> 2.091 s). The cost
of each extra traversal is ~180 ms, flat, and it is EXACTLY the model's A
term with ZERO of the pipeline gain: overlap efficiency measured 0, not
0.32, not 0.29. Kept in the tree at _MICROBATCH=0 (a no-op branch) with its
evidence script, so the negative result stays reproducible. Nobody should
re-propose splitting a traversal to fill the layer-split pipeline until the
finding below is fixed.

## exp-0021 arm 2: WHY nothing overlaps — the prefill traversal is HOST-BOUND
The measurement both exp-0018 and the microbatch arm needed
(engine/pyengine/tests/t_traversal_sync_probe.py; log
/workspace/logs/t_traversal_sync_probe_exp0021_out.log), on the real
resident model, after warm-up, with every implicit device->host sync entry
point (Tensor.item/tolist/cpu/numpy/nonzero, torch.nonzero/unique/bincount,
cuda.synchronize) counted by monkeypatch:
  arm a  syncs inside one grouped 3-row prefill traversal: **NONE**. exp-0018
         did exactly what it claimed.
  arm b  that traversal costs **696 ms of HOST time and 0.1 ms of device
         tail** (call -> return, then sync). The four GPUs finish everything
         the instant the host stops issuing. Repeated: 695 ms / 0.3 ms.
  arm c  two 3-row traversals back to back cost 1355 ms whether or not a
         full device sync is placed between them (1368 / 1355 / 1355 / 1355).
         There is no run-ahead to give away.
So prefill is PYTHON-limited, not pipeline-limited, and the whole
"traversals should overlap across the layer split" family (exp-0018,
exp-0020, the microbatch arm above) was attacking a bubble that is not the
binding constraint. The prefill term is ~40% of the canonical box and its
GPUs are idle for essentially all of it. Two consequences for the ranking.
(1) The BIG prefill lever is deleting Python from the traversal — the
exp-0012 treatment (CUDA-graphed, tensorized per-row state handling) applied
to prefill; note the host cost scales with rows as well as traversals
(A + B*G with both terms host-side), which points at the per-row Python
inside the grouped walk, not just the 66-layer loop. (2) The cheap lever,
which exp-0021 submits, is to spend that idle device time on the OTHER
traversal: issue the graphed decode step FIRST and read it back after the
prefill Python.

## exp-0021 REJECTED at 223.5 (bar 223.6) — the mixed-step share is real
## but small, and the prefill host-bound reading is NOT established
Ledger iter 22: 223.5 tok/s vs best 222.9 + 0.3% = 223.6. It missed by
0.1 tok/s — 0.04% — so the mechanism is not wrong, it is thin: the +124 ms
it measured on a MIXED step (t_decode_first, interleaved arms, repeats
agreeing to 1 ms) bought +0.27% in the box instead of the predicted +3.7%,
which prices the mixed-step share of the canonical box at roughly a tenth
of the cohort model's ~180 steps. Two lessons, and the second is the
load-bearing one.
(1) Three consecutive experiments (exp-0018, exp-0020, exp-0021) have now
    been rejected attacking the ORDER in which prefill and decode work is
    handed to the GPUs. That line is closed: the box does not have enough
    mixed steps for issue-order to pay. Do not re-open it.
(2) The diagnosis those experiments leaned on — "the prefill traversal is
    HOST-bound, the GPUs are idle for essentially all of it" (arm b of
    t_traversal_sync_probe: 696 ms host enqueue, 0.1 ms device tail) — does
    NOT follow from that arm. A host thread that outruns the CUDA launch
    queue BLOCKS INSIDE cudaLaunchKernel, so "time from call to return"
    counts device time it was waiting on, and a 0.1 ms tail only says the
    LAST device in the layer-split pipeline had caught up by the end, which
    is exactly what a serial 4-stage pipeline does. Counting the host ops a
    prefill layer actually issues (~170 + 15 per row, ~14k per 3-row
    traversal) puts pure dispatch at ~200-300 ms of that 696 ms, not 696.
    The arm that would settle it is per-device GPU-BUSY time (CUDA events
    bracketing each device's segment, or a profiler kernel-time total), and
    it has still not been run. Until it is, "delete Python from the prefill
    traversal" is an unsized lever, and the exp-0012 treatment applied to
    prefill (CUDA-graphing a shape-bucketed traversal, a large and
    memory-hungry change) should NOT be started on the strength of arm b
    alone. exp-0022 goes at prefill and decode from the other side —
    deleting DEVICE work that is there under either reading.
Also killed cheaply this iteration, before it cost a submission: the CUTE
DSL marshalling hypothesis (that _run_gemm's five cutlass_torch.from_dlpack
conversions per GEMM were a large part of the per-layer host cost).
Measured on CPU tensors of the prefill shapes: 5.7-6.8 us per conversion,
i.e. ~60 us per layer, ~4 ms per traversal. Not the fat. Do not re-propose.

## Process note (exp-0024): merge commits need the [ralph] prefix too
The engine lineage lives on branches — main has carried only spec JSONs and
docs since 99f6bbc — so every iteration since exp-0016 builds on the accepted
tip and merges current main in to satisfy the spec's
`git merge-base --is-ancestor main <commit>` rule. `git merge` writes its own
default message ("Merge branch 'main' into HEAD"), and when that merge is the
commit the spec pins, the worker logs `WARNING: candidate commit <sha> lacks
[ralph] prefix` (seen on exp-0022's b1a28ba and exp-0024's b6f8831; it is
non-blocking, both ran). Use `git merge --no-edit -m "[ralph] exp(<id>): merge
main"` so the pinned commit carries the prefix like every other one.

## exp-0027-fused-moe-out REJECTED 2026-07-30 — the kernel takes an ILLEGAL
## MEMORY ACCESS on the first conc-64 prefill group; the trace is captured
Post-mortem written by exp-0029's iteration. There is NO ledger row: the
worker died before it could write one, so `experiments/queue/
exp-0027-fused-moe-out.log` is the whole record. Its sequence is
worktree(691c6a6) -> D13 teacher-forced gate (GREEN, the worker printed no
failure) -> `[bench] TIMEBOXED warmup 64 reqs @ conc 64 osl 128` ->
`HTTPError: 500` out of /v1/completions -> `accepted: false, reason: null`.
A 500 from this server means the ENGINE THREAD died (server.py:193-198
records the traceback, wakes every waiter; handlers then answer
`engine died:\n<traceback>`, :390-398), and that traceback goes to stderr ->
serve.log INSIDE the worker's worktree -> destroyed by `git worktree remove
--force` on the way out. That is exp-0020's lesson 2 verbatim, unaddressed,
costing its second experiment.

WHAT THE TRACE SAYS. It was recoverable this time only because exp-0028's
pinned commit e14d829 has 691c6a6 as an ancestor (`git merge-base
--is-ancestor` YES), so the dispatcher was re-running the same code while
this iteration was reading; the live worktree's serve.log was snapshotted
before the worker exited and is committed evidence at
`/workspace/logs/serve_exp0028_WORKER_CRASH.log`. It reproduces exactly:

    server.py:257 sched.step
    scheduler.py:552  engine.prefill_batch_enqueue(group)
    scheduler.py:344  pmodel.layer_prefill(...)
    model.py:634 -> :435 moe_experts -> :339 moe_experts_w4a4
    moe_gemm_w4a4.py:283  moe_out_reduce(c2, rows, topk_weights, K)
    moe_out.py:111  _moe_out_kernel[(T, K // block)](...)
    RuntimeError: Triton Error [CUDA]: an illegal memory access

So: PREFILL, not decode; the GROUPED cohort path (`_prefill_batch_rows`),
not the singleton path; and it fires on the FIRST bench group — the server
came ready at 22:23:30, served the gate's 402 requests clean, and died at
22:24:52, ~80 s later. Both engines' D13 gates pass because the gate is
teacher-forced and SEQUENTIAL: it only ever forms singleton prefill groups
(scheduler.prefill_batch's own docstring says so), so the shape that fails
is one the gate cannot construct. exp-0027's isolated arm validated
T=4,800 (a 6-row cohort) and its in-situ arm ran 8-row groups; the bench's
first group is up to max_batch=64 length-sorted rows, i.e. T = B*Tmax
~= 51,200 tokens and P = T*6 ~= 307,200 pair rows — 10.7x the largest
shape either arm ever launched. NOTHING in the evidence chain covered it.

CONSEQUENCES, in the order they bite.
(1) exp-0028 IS ALREADY LOST — it pins e14d829, which contains the faulting
    kernel, and it died the same way at 22:24:52. Its own mechanism (the
    fused shared-expert epilogues) is UNJUDGED and, like exp-0018 before
    it, should be resubmitted once the base is fixed: its arms were bitwise
    (0 of 29.5M / 0 of 393K), its gate was green and its in-situ deltas
    were measured against everything else fixed.
(2) The accepted lineage tip is 94f81e5 (exp-0026, 514.2 tok/s). 691c6a6
    and e14d829 are NOT accepted bases. Do not build on them until the
    illegal access is fixed and reproduced-green at cohort scale.
(3) LESSON, and it generalizes past this kernel: an experiment's local
    evidence must include the shape the CANONICAL BOX will hand it, not
    the shape that was convenient to test. The canonical config is conc 64
    and the scheduler packs one length-sorted prefill group per step, so
    the prefill shape any prefill-path kernel must survive is B up to 64,
    T = B*Tmax, P = 6*T — not the 6-8 row cohort the in-situ harness
    builds. Every fused-prefill kernel merged since exp-0022 was validated
    only at 6-8 rows; they have all SURVIVED the box, so this is a gap in
    the evidence, not a claim about them, but exp-0027 is proof the gap is
    load-bearing.
(4) The D13 teacher-forced gate is NOT a crash gate for grouped prefill.
    It is sequential by construction. Passing it says nothing about
    conc-64 group shapes, and two iterations in a row have read a green
    gate as "the tree runs".

### exp-0027 addendum — the epilogue KERNEL is exonerated; look at the layer
Written the same iteration, after the repro. `_moe_out_kernel` is the site
the RuntimeError names, but a CUDA illegal access is sticky and asynchronous:
`cuLaunchKernel` returns the error left by an EARLIER kernel, so the launch
that reports it need not be the launch that caused it. Tested directly
(engine/pyengine/tests/t_moe_out_cohort on the exp-0028 tree, GPU 7,
CUDA_LAUNCH_BLOCKING=1, log /workspace/logs/t_moe_out_cohort_exp0029_out.log):
the exact pair bookkeeping moe_experts_w4a4 builds, launched at every group
shape the scheduler can form — B = min(64, PREFILL_SCORES_BUDGET/(128*Tmax^2))
with Tmax over the canonical decode_heavy prompt range [512, 1536], i.e.
T from 4,800 to 46,336, P to 278,016, m_cap to 310,528, c2 to 0.888 x 2^31
elements, grids to (46336, 6). NO FAULT at any shape, and the fused-vs-eager
value arm stays in the one-bf16-ulp class throughout (27 of 284,688,384 at
the largest). So the kernel is clean at every shape the box can hand it, and
the int32/int64 addressing worry is closed: it carries `row` in int64
already, and the c2 element count peaks at 89% of 2^31 rather than over it.
WHAT THAT LEAVES. The fault is upstream in the same MoE layer —
nvfp4_quant_scatter, the two CUTE DSL grouped GEMMs, or nvfp4_quant_silu —
and exp-0027 did not introduce it so much as EXPOSE it: the eager chain it
deleted allocated ~26 GB of [P, K] fp32/bf16 transients per MoE layer, and
removing them changes what the caching allocator hands out, so an
out-of-bounds access that used to land inside a live cached block now lands
on unmapped memory. That is a hypothesis, not a result; the arm that would
settle it is `moe_experts_w4a4` end-to-end at cohort scale against real
PackedExperts weights for ONE layer (~10 GB, one GPU, no server), with
CUDA_LAUNCH_BLOCKING=1 and compute-sanitizer if the fault reproduces. It was
not run here for budget: the session had ~20 minutes left and the exp-0028
worker was squatting GPUs 4-7. NEXT ITERATION SHOULD RUN IT FIRST — it is
cheap, it needs no model server, and until it lands there is a latent
illegal access in the ACCEPTED tree that only the eager epilogue's
allocation pressure is hiding.
OPERATIONAL NOTE, verified live: when the engine thread dies the worker does
NOT exit promptly — its 64 bench threads sit on requests the dead loop will
never retire, so the worker hangs (exp-0027: 62 minutes wall against a
~13-minute norm for exp-0024/0025/0026) while its server keeps ~175 GiB/GPU
resident on GPUs 4-7. An iteration that finds the dispatcher "busy" for far
longer than 13 minutes should read the worktree's serve.log BEFORE assuming
the run is healthy; that file is the only copy of the trace and it dies with
the worktree.
