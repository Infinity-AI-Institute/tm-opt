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
