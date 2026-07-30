# exp-0018 — prefill/decode traversals pipelined across the layer-split GPUs

Nothing below is predicted except the clearly-labelled cycle model; every
other number was measured on THIS tree, on GPUs 4-7, 2026-07-30.

## The finding

`server.build_resident` shards the 66 layers into four contiguous
per-device segments (`layer split [16, 17, 16, 17] over 4 GPUs`). That
makes ONE traversal — a prefill, or a decode step — a strictly serial walk
across four GPUs: while it is on device d, the other three are idle. The
engine has never had two traversals in flight, so the machine has been
running at ~1/4 of its width for the whole Stage-3 trajectory.

It is visible without instrumentation. Sampling `nvidia-smi
--query-gpu=utilization.gpu` at 0.4 s while the WORKER's own canonical
bench of exp-0017 was running (log
`/workspace/logs/t_pipeline_bubble_exp0018_util.log`) gives a marching
wave:

    96 %   0 %   0 %   0 %
    100 %  0 %   0 %   0 %
    0 %   99 %   0 %   0 %
    0 %    0 %  99 %   0 %
    0 %    0 % 100 %   0 %
    0 %    0 %   0 %  95 %

interleaved with 4-way ~35-40 % rows (decode steps, several per sample
window). The 100/0/0/0 rows are the cohort prefill walking the pipeline.

What pinned it there is host synchronisation, not data dependency. Every
device->host sync inside a traversal blocks the host until THAT device
catches up, so the next traversal cannot be handed to the GPUs. The
prefill traversal had 23 of them: `kv.PagedKV._flat_slots` does a
`.tolist()` on the page table for `append` and again for `gather` on each
of the 11 global layers (the sync exp-0005 flagged and the batched decode
core already routes around), plus the final `int(lg.argmax())`.

## Diff (one mechanism)

- `kv.py _flat_slots`: pool-backed caches index the device mirror of the
  page table (`table_dev_row`, which `ensure_pages` already maintains for
  the batched decode core) instead of round-tripping it to the host.
  Value-identical integers; the non-pooled path (batch-1 tests) is
  untouched.
- `scheduler.py`: `Engine.prefill` / `prefill_batch` split into an
  enqueue half (returns the fp32 logits rows still on device) and a
  readback half; both keep their original synchronous form as the
  composition, so every caller outside the scheduler is unchanged.
  `Scheduler.step` enqueues the step's prefill groups, then the decode
  traversal, and only then reads the prefill tokens back.

Nothing numeric moves: identical kernels, identical shapes, identical
per-sequence op order. Only *when the host hands work over* changes.
Kill switch `PYENGINE_PIPELINE_PREFILL=0`.

## Measured (`engine/pyengine/tests/t_pipeline_prefill.py`, log
`/workspace/logs/t_pipeline_prefill_exp0018_out.log`; one resident load,
8 residents + a cohort of 6 at ISL 690-800, warm-up pinned to the same
1024 key-length bucket so no capture lands in a timed arm)

- **arm a — exact equality**: serial vs pipelined token streams
  `True` over 14 sequences / 56 tokens. Not envelope-close: EQUAL. That
  is the whole correctness argument for this patch.
- **arm c — determinism**: two identical pipelined runs bitwise equal
  (the t_b3-adapted worker arm).
- **arm b — wall**: the mixed step (a grouped prefill AND a decode
  traversal) **1.409 s -> 1.373 s = 1.026x**, i.e. 36 ms. Whole 5-step
  schedule 3.509 -> 3.405 s. Prefill-only steps and decode-only steps are
  unchanged (1.593/1.539 and 0.168/0.168) — as they must be, since a step
  with one traversal has nothing to overlap.

## D13 teacher-forced pre-gate — GREEN

    parity_pass true | agree 379/400 = 0.9475 (floor 0.945)
    delta_mean 0.043856 (bar 0.054281) | delta_max 0.396472

Bit-identical to the accepted exp-0016 / exp-0015b / exp-0017 gate
numbers, which is the expected result for a change that moves no
arithmetic. Server load 120 s, gate wall ~100 s, port 8218, GPUs 4-7.

## Cycle model (the only predicted part)

The measured pieces reproduce the accepted exp-0017 row (221.8 tok/s,
138.6k tok / 1083 reqs / 625 s) to within 2 %: 1083 reqs at cohort 6 =
~180 prefill traversals x 1.373 s = 247 s, plus 1083 x 128 / 64 = 2166
decode steps x 0.168 s = 364 s, total 611 s vs the row's 625 s. Applying
the measured 36 ms to the ~180 mixed steps gives 6.5 s back: **~224.2
tok/s against a merge bar of 222.5**.

## What this does NOT recover, and why it matters more than the win

The pipelining ceiling for one prefill + one decode traversal is
`3 x decode_stage`: serial costs 4 prefill segments + 4 decode segments,
perfectly pipelined it costs 4 prefill segments + 1. At the measured
0.168 s decode step that ceiling is ~126 ms, and only 36 ms was realized.
The residual is on device 0: the decode step's first segment plus the
EAGER LAYER-2 ISLAND (graphrun `_run_island` — layer 2's bf16-unquantized
experts still run the reference `unique().tolist()` + per-hit-expert loop,
the one region exp-0012 could not capture) both sit on the device the
prefill traversal occupies FIRST, so they are exactly the part that cannot
overlap. That points at the next two attacks, in order:

1. **Fold layer 2's bf16 experts into the captured graph** (a fixed-shape
   grouped/bmm form over all 256 experts is capture-legal; the current
   loop is ~400 launches at M~1-2 rows). It deletes the island from the
   critical path AND unblocks the remaining ~90 ms of this experiment's
   ceiling. Size it first: instrument `_run_island` and the four segment
   replays separately — if the island is the large share of the 168 ms
   step that this experiment's shortfall implies, it is worth far more
   than the fold itself (decode is 364 s of the 611 s cycle).
2. **Microbatched decode.** A layer-split model can only overlap as many
   traversals as it has independent ones; today that is at most two (one
   prefill group + one decode step), which is why this win is small.
   Splitting the 64-row pool into 4 independent 16-row microbatches makes
   four. The MoE cost per traversal is sublinear in rows (it is set by the
   number of DISTINCT hit experts, ~200 at 64 rows vs ~80 at 16), so four
   pipelined 16-row traversals should cost ~(4+3) x 0.4 of a segment
   against 4 x 1.0 today — order 1.4x on the decode term. It needs
   per-microbatch graph captures and pool slots, and it needs exactly the
   sync-free traversal this experiment built.
