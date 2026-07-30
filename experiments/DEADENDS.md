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
