# exp-0015b IN-PROGRESS hedge note (2026-07-30)

Mechanism: the committed exp-0015 split plan's SECOND half — route the
CUDA-graphed decode step's routed experts through the native W4A4 grouped
MMA (kernels/moe_gemm_w4a4.py), which exp-0015a/0016 already wired for
prefill. exp-0012's kernel table puts the routed-expert GEMMs at 266.4 ms
of the 322.7 ms graphed step.

Base: 41e827e (exp-0016, in flight) + main — so the measurement is the
COMBINATION of both halves, not a stale disjoint half (the exp-0015a
lesson recorded in exp-0016's spec).

Done so far:
- capture-safety probe (tests/t_w4a4_capture, GPU 4): found and fixed the
  one capture blocker — torch.bincount's CUDA path syncs on max().cpu().
  Replaced with a searchsorted boundary scan (same exact integers, no
  sync, no atomics). Post-fix: capture+replay bitwise == eager,
  same-schedule determinism, and replay under a DIFFERENT routing bitwise
  == eager on that routing (required: routing changes every step under a
  fixed graph). PASS.
- decode wiring in model.py behind PYENGINE_W4A4_DECODE (default on).

BUDGET / GATE STATUS: GPUs 4-7 are held by the worker's own exp-0016
benchmark (server pid 47423, 168-180 GiB per GPU); GPUs 0-3 hold the vLLM
baseline. A 4-GPU engine server for the local D13 TF gate therefore does
NOT fit this session. Mitigation being run instead: real-layer-3-weight
decode-shape probe (tests/t_w4a4_decode) covering prefill-bit-identity,
decode numerics vs W4A16, per-layer wall, and capture/determinism at the
real shape. If that comes back green the spec ships with the gate status
stated plainly; if it comes back red the spec is not written and this note
becomes the record.
