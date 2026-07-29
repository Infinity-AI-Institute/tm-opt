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
