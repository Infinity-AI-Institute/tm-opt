# exp-0016-prefill-w4a4 — submission record (iteration 17)

This file started life as the budget-edge hedge note committed BEFORE the
local pre-gate run (commit `3ef2344`); it is rewritten here with the run's
results. Nothing below is predicted — every number was measured on THIS
tree.

## Why this experiment exists (the stale-base finding)

exp-0015a was ACCEPTED (ledger iteration 16, 93.0 tok/s decode_heavy,
commit `09ae5fa`) — but that commit is NOT a descendant of main:

    git merge-base --is-ancestor 09ae5fa main                 -> NO
    git ls-tree -r 09ae5fa engine/pyengine/ | grep graphrun   -> (empty)

`09ae5fa` branches from `42b5457`, which predates merge `99f6bbc` ("brings
the chained accepted lineage 0008-0012; ff refused"). Its bench engine
therefore lacks the CUDA-graphed decode step (0012 — `graphrun.py` is
absent from its tree, verified above); `git diff 09ae5fa main -- engine/`
also separates `attn_global.py`, `attn_swa.py`, `moe_gemm.py`, `kv.py`,
`scheduler.py`, i.e. some-to-all of pooled sconv (0008), prefill BLOCK_M 32
(0009), fp4cvt dequant (0010) and fused decode attention (0013).

So **93.0 is W4A4 prefill on a pre-0012 decode**, and **89.7 is the
0008-0013 decode on a W4A16 prefill**. Neither number is the combination,
and the dispatcher cannot fast-forward `09ae5fa` onto main. This branch is
the combination: the exp-0015a mechanism rebased onto current main
`e1666ee`.

## One-variable audit vs main

- `git rebase e1666ee` of the two 0015a commits — conflict-free (the
  0008-0013 lineage rebuilt decode; the prefill call site was untouched).
  `git merge-base --is-ancestor main HEAD` passes.
- Mechanism files byte-identical to the accepted tree:
  `git diff --quiet 09ae5fa HEAD --` on `kernels/moe_gemm_w4a4.py`,
  `kernels/fp4_quant.py`, `loader.py` — all IDENTICAL. `model.py` differs
  only because main's carries 0008-0013.
- Diff vs main touches: `kernels/fp4_quant.py` (new),
  `kernels/moe_gemm_w4a4.py` (new), `vendor/cutlass_moe/*` (new, BSD-3),
  `loader.py` (+`input_amax` field), `model.py` (+`phase=` kwarg
  threading), `tests/*`. No other engine file.
- Only `model.py:432` (`layer_prefill`) passes `phase="prefill"`.
  `model.py:538` (`layer_decode`), `model.py:872` (`layer_decode_batch`)
  and `graphrun.py:248` all take the default `phase="decode"` — the
  CUDA-graphed step stays bit-identical W4A16.
- Serving really does get `input_amax`: `server.py:91/119` loads through
  `t_b2.load_layer_weights`, the site the loader patch touches.

## Measured on this tree (GPUs 4-7, 2026-07-30)

**D13 teacher-forced pre-gate — GREEN.** `--tf-positions 8` (matches the
envelope's `k_per_prompt` and the worker's own invocation),
`--envelope experiments/tf_envelope.json`:

    parity_pass true | agree 379/400 = 0.9475 (floor 0.945)
    delta_mean 0.043856 (bar 0.067851; prior engine 0.05065)
    delta_max 0.396472

Identical to the stale-base gate, as expected: teacher-forced parity is
`max_tokens=1`, so it exercises the prefill path this patch changes and
barely touches the decode kernels the rebase brought in. Precision moves
TOWARD the goldens because W4A4 is the reference's own ModelOpt recipe.

Server load 110 s; first request 33 s incl. the 8 DSL compiles + 63 layer
scale repacks; greedy smoke correct (" Paris."); gate wall 103 s.

**Batched + graphed decode co-residency — the one risk the stale-base run
could NOT cover** (that tree has no `graphrun.py`, so nobody had ever run
the W4A4 scale tensors alongside the graph's static buffers). Server at
`--max-batch 64`, 6 concurrent 24-token requests, two identical rounds:

    round 1 vs round 2 token ids : IDENTICAL (bitwise-deterministic)
    per-GPU memory               : 163-175 GiB of 267.7 GiB
    traceback/error/OOM in log   : 0

The ~+0.9 GB/layer resident blocked scales are already inside that
163-175 GiB, leaving ~90 GiB of headroom under the bench's max_batch-64
graph buffers.

## Submitted

`experiments/queue/exp-0016-prefill-w4a4.json`. Merge bar is over the best
same-protocol accepted row, which is now exp-0015a's 93.0, not 89.7:
93.0 x 1.003 = **93.3**.
