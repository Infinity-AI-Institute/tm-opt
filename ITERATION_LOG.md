# ITERATION_LOG.md — generated from experiments/ledger.jsonl. Do not hand-edit.

## Baselines (dashed lines)

| workload | baseline_id | tok/s | noise | measured commit |
|---|---|---|---|---|
| decode_heavy | vllm_mtp_off | 13,409.5 | 0.12% | 476ee98 |
| prefill_heavy | vllm_mtp_off | 4,999.3 | 0.82% | 06ef2ae |

## Iterations

| iter | workload | label | mechanism | tok/s | vs vLLM | commit | log |
|---|---|---|---|---|---|---|---|
| 0 | decode_heavy | iteration 0 | per-sequence eager engine, Triton pending | 2 | 0.0% | 8c29167 |  |
| 0 | prefill_heavy | iteration 0 | per-sequence eager engine, Triton pending | 2 | 0.0% | 759c5ac |  |
