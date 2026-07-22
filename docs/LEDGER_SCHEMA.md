# Ledger schema — graph-required fields (append to harness/ledger.py docstring
# and enforce in the worker when writing accepted rows)

Requirement (2026-07-20): the final deliverable must reproduce a trajectory
chart like infinity.inc/research/qwen3-optimization. Every accepted-experiment
ledger row therefore carries, from day one:

| field | example | chart element |
|---|---|---|
| iteration | 46 | x-axis |
| label | "Memory fusion" | hover title |
| mechanism | "Eliminate HBM round-trips" | hover subtitle |
| workload | "decode_heavy" | which series |
| tok_per_s | 19618.0 | y-axis |
| pct_vs_baseline | 98.1 | hover "VS VLLM" |
| baseline_id | "vllm_mtp_off" | which dashed line it's measured against |
| cache_key | "<16-hex>" | validity guard (must match canonical.lock.json) |
| commit | "abc1234" | reproducibility |
| accepted | true/false | plotted vs dead-end log |
| noise_floor_pct | 0.4 | merge-gate audit |
| log_path | "docs/logs/exp-0046.log" | evidence |

Baseline rows: iteration 0 entries per (engine=vllm, workload, mtp on/off)
written by P3's baseline capture — these define the dashed lines and the
denominators for pct_vs_baseline. plot_trajectory.py consumes ONLY the ledger
+ canonical.lock.json's `report` block; if the chart needs a datum not in
those two places, the fix is to add the field, never to hand-edit the chart.