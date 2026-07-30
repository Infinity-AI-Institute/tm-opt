# exp-0017-cohort-prefill — IN-PROGRESS hedge note (2026-07-30 ~16:25 UTC)

State of the world at start: exp-0015b ACCEPTED (ledger iter 18, 194.5 tok/s
timeboxed decode_heavy) — both prefill (exp-0016) and decode (exp-0015b)
routed-expert GEMMs now run the native NVFP4xNVFP4 grouped MMA. Queue empty.

Hypothesis (one variable): the native MMA pads every expert group to a
128-row M tile, so an expert's GEMM cost is FLAT from 1 to 128 rows. A
single canonical prompt (T~736, top-6 over 256 experts) routes only ~17
rows to each expert — the tile is 87% empty, and prefill walks it once PER
SEQUENCE. Co-prefilling 6 sequences fills the same tile (~124 rows) and
costs the same tiles. exp-0014 built this (batched group prefill,
right-padded, length-sorted, budget-partitioned) and measured 1.30x at
engine level under W4A16 — where cost still scaled with rows, so grouping
only amortized launch overhead. Under W4A4 the flat-tile regime should make
it multiplicative.

Why exp-0014 lost in situ (its DEADENDS post-mortem) and what is different
here: its 150 ms wall-clock accumulation window raced per-completion HTTP
arrivals and admitted mostly singletons. This diff replaces the window with
a cohort-forming pool in EngineLoop keyed on the scheduler's own state —
release when PREFILL_COHORT prompts have gathered AND that many slots have
actually freed (a retired cohort), never more than the free slots, with an
idle bypass (D13 gate/cold start keep the batch-1-exact per-seq path) and a
4 s valve. Released together they retire together, so the cohort is
self-sustaining after one round.

Also carried (measurement hygiene, requested by the exp-0014 post-mortem,
cannot change engine speed): listen backlog 5 -> 256, the artifact that
dropped 3 of 64 bench threads and -4% off exp-0014's headline.

Budget/status at this stamp: diff written and syntax-clean; base is
f09315b (accepted exp-0015b tip) merged with main ac8e8a9. NEXT: GPU
evidence arm t_prefill_batch (S serial vs G grouped wall + G==G2 bitwise
determinism) ~8 min on GPUs 4-7, then the D13 teacher-forced gate ~8 min,
then spec. If the session ends before those, this note + the diff stand and
the next iteration re-runs from here.
