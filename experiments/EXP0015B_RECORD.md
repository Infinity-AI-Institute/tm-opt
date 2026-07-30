# exp-0015b — W4A4 in the CUDA-graphed decode step (local record)

The second half of the exp-0015 split plan committed in DEADENDS
(2026-07-30): exp-0015a/0016 put the native FP4xFP4 grouped MMA on the
PREFILL routed experts; this puts the same kernel, same recipe, on the
DECODE ones — including inside the exp-0012 CUDA graphs. After this there
is no phase distinction left in the routed-expert path.

## Why here
exp-0012's graphed-step kernel table (the current ground truth, since the
step is GPU-bound after the dispatch floor was locked) puts the routed
expert GEMMs at **266.4 ms of the 322.7 ms** decode step. They run the
W4A16 Triton kernel, which pays the exp-0010 in-register convert-ALU floor
(~2.5 cvt-class ops per weight element) on every weight tile. Native FP4
MMA deletes that pipe outright.

## Diff (one variable)
`model.py`: the routed-expert dispatch no longer restricts the W4A4 kernel
to `phase == "prefill"` (kill switch `PYENGINE_W4A4_DECODE=0`, read once at
import like exp-0013's `_FUSED_ATTN`, so captured graphs match eager
warmups). `moe_gemm_w4a4.py`: one capture-safety fix, below. Plus two test
files. Nothing else.

## The capture blocker, found and fixed early (as the ruling demanded)
`torch.bincount` is **not CUDA-graph-capturable**: its CUDA path sizes the
histogram from `self.max().cpu()`, a device->host copy, and capture aborts
with "Cannot copy between CPU and CUDA tensors during CUDA graph capture"
(tests/t_w4a4_capture, arm b, pre-fix). Replaced with a `searchsorted`
boundary scan over `0..E` on the already-sorted pair list — same exact
integers, no sync, no atomics, and it also subsumes the second `cumsum`.

**This does not perturb the accepted prefill path.** `offs`/`dst`/`src` are
the only outputs of the changed block, and t_w4a4_decode arm a shows them
bitwise identical to the bincount form at T = 736 / 64 / 1 / 3000. So
exp-0016's green D13 TF gate (parity_pass true, agree 379/400 = 0.9475,
delta_mean 0.043856 vs bar 0.067851) transfers to this tree unchanged.

## Local evidence (GPU 4, real layer-3 experts; log
/workspace/logs/t_w4a4_decode_exp0015b_out.log)
- **a — prefill bit-identity**: PASS at all four shapes (above).
- **b — decode numerics, with the prefill-shape CONTROL** (the argument
  that matters): W4A4-vs-W4A16 at the decode shape T=64 is `rel_mean
  1.934e0, cos 0.98856`; at the accepted prefill shape T=736 it is
  `1.901e0, cos 0.98873`. Ratio 0.9998 — decode W4A4 is **the same
  numeric class as the already-accepted prefill W4A4**, not a new risk.
  (The full-chain rel_mean is naturally larger than DEADENDS arm J's
  per-GEMM 9.5e-2: two GEMMs, a silu, a routing epilogue and a 6-slot sum
  compound. Being shape-invariant is the point.) Finite everywhere.
- **d — capture arms**: capture+replay bitwise == eager; same-schedule
  determinism (two replays bitwise identical — the t_b3-adapted worker
  arm); and replay under a **different routing** bitwise == eager on that
  routing. The last is load-bearing: routing changes every step under a
  fixed graph, so the captured kernel must be a pure function of buffer
  contents, not of the routing it was captured with. Also validated at
  small shapes in tests/t_w4a4_capture (arms b/c/d).
- **e — the honest speed number**: both paths *captured*, replayed, matched
  routing, same weights: **11.197 ms/layer W4A16 -> 2.874 ms/layer W4A4,
  3.90x** (3.70x on the prior run; eager arm c agrees at 3.79x).

### Reading the speed number correctly
The **absolute** walls above are on uniform-random routing, which hits
201/256 experts at T=64; both kernels are dominated by per-hit-expert
weight traffic, so the trained router's concentration makes both sides
smaller in situ — which is exactly why W4A16 measures 11.2 ms/layer here
against the in-situ table's 266.4/63 = 4.23 ms. The **ratio** is what
transfers (matched routing, both captured). Applied to the in-situ table:
routed-expert term **266.4 -> ~68 ms**, graphed step **322.7 -> ~125 ms**.

## Ragged decode segments (the flagged risk)
Handled by construction and exercised: every expert group is padded to a
128-row multiple, so the 1-3-row decode segments never touch the example's
unexercised non-16-multiple group path — they ride the same 128-aligned
layout the prefill path validated, pad rows exact zeros. The cost is real
and honest: ~200 nonzero groups x 128 rows for ~384 real rows, i.e. the
MMA does far more work than the tokens require. It still wins 3.9x. A
tighter per-group granularity is future work, not this experiment.

## Untested here, stated plainly
The local D13 TF gate could **not** be run this session: GPUs 4-7 were held
for the whole session by the worker's own exp-0016 benchmark (server pid
47423, 168-180 GiB/GPU) and GPUs 0-3 by the vLLM baseline, so a 4-GPU
engine server did not fit. What that gate measures is prefill (max_tokens
= 1), which arm a proves is bitwise unchanged from exp-0016's green run;
the decode path this patch changes is not exercised by it at all. The
worker runs the full canonical gate before benching regardless.
Consequently untested in situ: graph-pool growth with the m_cap=32896-row
transient buffers live in every captured segment across 4 devices (they
are freed and reused within a capture, and exp-0016 measured ~90 GiB
headroom, but this is the one place a surprise could live). No new JIT
compiles: decode reuses prefill's `(device, K, N)` kernel-cache keys, so
the ~33 s first-request compile storm does not grow.
