# exp-0016-prefill-w4a4 — IN-PROGRESS hedge note (iteration 17)

Committed BEFORE the local D13 pre-gate run, per the budget-edge rule. If
this file is the last thing on the branch, the gate run is what killed the
iteration; everything below is the state a successor inherits.

## Why this experiment exists (the stale-base finding)

exp-0015a was ACCEPTED (ledger iteration 16, 93.0 tok/s decode_heavy,
commit `09ae5fa`) — but that commit is NOT a descendant of main:

    git merge-base --is-ancestor 09ae5fa main   -> NO
    git ls-tree -r 09ae5fa engine/pyengine/ | grep graphrun  -> (empty)

`09ae5fa` branches from `42b5457`, which predates merge `99f6bbc` ("brings
the chained accepted lineage 0008-0012; ff refused"). Its bench engine
therefore lacks pooled sconv (0008), prefill BLOCK_M 32 (0009), fp4cvt
dequant (0010), the CUDA-graphed decode step (0012 — `graphrun.py` is
absent from its tree, verified above) and fused decode attention (0013).
So 93.0 is **W4A4 prefill on an exp-0007-era decode**, and 89.7 is **the
0008-0013 decode on a W4A16 prefill**. Neither number is the combination,
and the dispatcher cannot fast-forward `09ae5fa` onto main.

This branch is the combination: the exp-0015a mechanism rebased onto
current main `e1666ee`.

## What is already done on this branch

- `git rebase e1666ee` of the two 0015a commits — conflict-free (the
  0008-0013 lineage rebuilt decode; the prefill call site was untouched).
  `git merge-base --is-ancestor main HEAD` now passes.
- Mechanism files verified byte-identical to the accepted tree:
  `git diff --quiet 09ae5fa HEAD -- kernels/moe_gemm_w4a4.py`,
  `kernels/fp4_quant.py`, `loader.py` all IDENTICAL. `model.py` differs
  only because main's carries 0008-0013.
- One-variable audit vs main: diff touches `kernels/fp4_quant.py` (new),
  `kernels/moe_gemm_w4a4.py` (new), `vendor/cutlass_moe/*` (new, BSD-3),
  `loader.py` (+`input_amax` field), `model.py` (+`phase=` kwarg threading),
  `tests/*` — no other engine file. Only `model.py:432` (`layer_prefill`)
  passes `phase="prefill"`; `model.py:538` (`layer_decode`), `model.py:872`
  (`layer_decode_batch`) and `graphrun.py:248` all take the default
  `phase="decode"`, so the CUDA-graphed step stays bit-identical W4A16.
  Serving really does get `input_amax`: `server.py:91/119` loads through
  `t_b2.load_layer_weights`, which is the site the loader patch touches.

## What remains

1. Local D13 teacher-forced pre-gate on THIS tree, GPUs 4-7:
   `--tf-positions 8 --envelope experiments/tf_envelope.json` (k=8 matches
   the envelope and the worker's own invocation). The gate numbers quoted
   in the two cherry-picked commit messages were measured on the STALE
   tree and do not certify this one.
2. Fill `GATE_RESULT_PLACEHOLDER` in `experiments/exp-0016-prefill-w4a4.json.draft`,
   `mv` it into `experiments/queue/`, commit.

## Budget estimate for the run this note hedges

server load (4 GPUs, 66 layers) ~6-12 min + first-request DSL compile ~32 s
+ 400 teacher-forced comparisons at ~0.3-0.9 s each ~4-7 min => ~15-25 min,
inside the ~40 min ceiling as ONE foreground call. If a successor finds the
run did not finish: split it — run the gate against an already-loaded
server in a separate iteration, or lower `--tf-positions` for a smoke pass
and note the deviation (the SUBMITTED gate must be k=8).
