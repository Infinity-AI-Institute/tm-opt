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

## exp-0023 finding 1: the prefill traversal's ACTIVE GPU is SATURATED —
## exp-0021 arm 2's "prefill is PYTHON-limited" is WRONG (existing evidence)
The exp-0021 post-mortem left this open and told the next iteration what to
run ("per-device GPU-BUSY time ... has still not been run"). It did not need
a new run: the arm already exists in the tree, from exp-0018.
/workspace/logs/t_pipeline_bubble_exp0018_util.log samples
`nvidia-smi --query-gpu=utilization.gpu` every 0.4 s on GPUs 4-7 DURING the
worker's own canonical bench of exp-0017 (accepted-lineage engine, conc 64).
The prefill phase is the marching wave, and every sample in it reads
92-100% on exactly ONE device and 0% on the other three (99/0/0/0,
0/100/0/0, 0/0/100/0, 0/0/0/99, ...). utilization.gpu is the fraction of the
sample window in which at least one kernel was resident, so a device being
fed by a host that cannot keep up would read FAR below 100% — a traversal
issuing ~14k ops in 696 ms with pure-dispatch gaps would show tens of
percent, not 99%. The tail of the same log (35-40% on all four at once) is
the decode phase and agrees from the other side: a 168 ms graphed step with
one device active at a time is 25-42% of a 0.4 s window per device.
So arm b's "696 ms host enqueue / 0.1 ms device tail" is exactly what the
post-mortem's caveat predicted — the host was BLOCKING INSIDE the launch
path of a saturated device, and 696 ms is device time seen from the host.
CONSEQUENCES, and they re-rank the surface:
(1) Deleting DEVICE work from the prefill traversal PAYS at face value.
    "Delete Python from prefill" (CUDA-graphing a shape-bucketed traversal,
    a large and memory-hungry change) is NOT the lever exp-0021 thought it
    was and should stay unbuilt.
(2) The binding structural loss is the same in BOTH phases: the 66 layers
    are split [16,17,16,17] across 4 devices and walked serially, so at
    every instant THREE of four GPUs are idle. The engine is running on
    ~1/4 of the node's HBM bandwidth and ~1/4 of its FLOPs. That is the
    D4 #7/#8 lever (TP / topology), it is the largest one left, and
    nothing has attacked it yet. Microbatching the layer-split pipeline
    is NOT the way in (exp-0021 arm 1 measured zero overlap, and the
    reason is now visible: it issued each sub-group as a whole 4-device
    traversal back to back, which is the serial schedule by construction —
    a real pipeline needs per-stage issue, and under (1) the stages are
    saturated anyway, so the honest fix is intra-layer parallelism, not
    inter-microbatch).
Fairness caveat, stated so nobody over-reads this: utilization.gpu is a
kernel-resident fraction, not occupancy. A device can read 99% while its
kernels use a fraction of its SMs. The claim this evidence supports is the
narrow one that matters here — there is no host-side gap to reclaim in a
prefill traversal — not that every prefill kernel is efficient.

## exp-0023 groundwork: fused two-shape PREFILL attention (the prefill twin
## of exp-0013), and why this iteration submitted no spec
Under finding 1 the ranking question becomes "which device term in the
prefill traversal is fat?", and prefill attention is the one part of the
engine still running the bring-up eager chain over a materialized
[B, heads, T, T] tensor. model.attn_prefill #3-#8 at the canonical cohort
shape (B=6 rows, T=1536 padded, 64 heads, bf16 scores = 1.81 GB per
tensor): rel_bias materializes the [B,H,T,E] logit mix (604 MB), gathers it
into a [B,H,T,T] bias (1.81 GB) and masked_fills that (read+write); then
matmul writes scores, `* scale` reads+writes them, `+ bias` reads+writes,
`+ mask` reads+writes, softmax(dtype=fp32) writes a 3.62 GB fp32 copy and
`.to(bf16)` reads it back, and the PV matmul reads the result — plus the
GQA repeat_kv materializing K and V at 64 heads (302 MB/layer). That is
~26 GB of HBM traffic per layer to produce a [B,T,H,D] output worth 19 MB,
~1.7 TB per traversal over 66 layers, on devices finding 1 says are
saturated. A flash-style kernel that reads K/V in place, gathers the rel
bias in-kernel by distance and never materializes a score tile is ~1 GB per
layer, and on the 55 SWA layers it also drops the out-of-window half of the
QK/PV work (window 512 vs T=1536).
The causal half is free too: the eager chain computes the whole T x T grid
and then adds finfo.min to the upper triangle, so even the 11 global layers
do ~2x the QK/PV work they need, and the 55 SWA layers do ~3x at T=1536
(window 512). The kernel simply never visits those tiles.
FOUND BY THE CPU PRE-GATE, and it is load-bearing for the kernel: K and V
do NOT arrive at attention in a [.., .., T, head_dim]-contiguous layout.
sconv_prefill returns `conv.transpose(1,2) + xf`, and that add keeps the
channel-major layout of its non-contiguous operand, so post-sconv K/V carry
strides (96, 12288, 1, 96) at [B,HK,T,D] — stride 1 along T, stride T along
head_dim (t_fused_prefill_attn arm b, run on the engine's own ops). The
first draft of the kernel asserted last-dim contiguity and tripped on it;
it now takes explicit head_dim strides for q/k/v and requires contiguity
only of the rel mix (which comes from a matmul and has it). The same fact
says the eager chain's matmul is paying a contiguity copy on K and V that
the fused path deletes as well — the prefill twin of the 128 ms/step of
`direct_copy` exp-0013 found on the decode side.
NO SPEC WAS SUBMITTED THIS ITERATION, and the reason is budget, not doubt:
the dispatcher started exp-0022 on GPUs 4-7 at 18:37 (queue log) and held
203-213 GiB per device for the whole session, while GPUs 0-3 hold the
frozen vLLM baseline server (254 GiB, 4-day resident) and are off-limits by
the hard rule. No local D13 gate, no numerics arm and no timing arm was
possible, and submitting a NEW Triton kernel with zero GPU evidence would
have burned a canonical slot on a coin flip. What is committed instead is
the kernel, its call-site wiring behind a kill switch
(PYENGINE_FUSED_PREFILL_ATTN=0), and a validation script whose algebra arms
run on CPU with no CUDA context at all. Those were run and are GREEN
(/workspace/logs/t_fused_prefill_attn_exp0023_sim.log): a literal torch
transcription of the kernel — same block bounds, same distance/band/mask
expressions, same online-softmax update — matches the engine's own eager
chain in fp32 to 4.8e-07 on both layer shapes (arm a), padded rows are
bit-exact against the same rows run alone with garbage pads left in place
(arm c), tau-on-the-mix equals tau-on-the-gathered-bias with the floor
lowered so tau actually varies (arm d, 1.000-1.249), the kernel's mask
predicate is bitwise additive_causal_mask's for both window forms (arm e),
and the eager call site still runs end to end with the switch off (arm d2).
What that does NOT cover, stated plainly: Triton codegen, register
pressure/tile choice, the real bf16 drift against the D13 envelope, and
every timing claim. NEXT ITERATION RUNBOOK, ~35 min total: (1)
`python -m engine.pyengine.tests.t_fused_prefill_attn --gpu --dev 4`
(~5 min; arm f numerics, arm g determinism, arm h sweeps BLOCK_M/BLOCK_N
64x64 / 64x32 / 128x64 / 32x64 — pin the winner in the attn_prefill_fused
call site, arm i peak memory), (2) the D13 teacher-forced gate on a loaded
server (~20 min) — THIS patch moves prefill numerics, so unlike exp-0013
the gate is genuinely at risk and must be run before submitting, (3)
submit. If the gate goes red, the fallback that keeps most of the win is
to reproduce the eager rounding chain inside the kernel (bf16-round the
QK product, then scale/add bias in bf16) instead of carrying fp32.
