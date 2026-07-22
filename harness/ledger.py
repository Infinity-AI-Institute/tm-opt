"""
@brief  Append-only experiment ledger (JSONL). Single source of truth for the
        dispatcher's accept/reject decisions and the final trial write-up.
"""
import json
import time
import fcntl
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict, field

LEDGER_PATH = Path(__file__).resolve().parent.parent / "experiments" / "ledger.jsonl"


@dataclass
class ExperimentRecord:
    """
    @brief  One row per completed experiment. Every field is filled by the
            worker; nothing is written until the experiment fully finishes.
    """
    experiment_id: str            # e.g. "exp-0042"
    hypothesis: str               # one sentence, one variable changed
    commit: str                   # git commit hash of the candidate build
    base_commit: str              # main commit the worktree branched from
    gpu_slot: int                 # which physical GPU ran the perf suite
    # -- correctness gate --
    parity_pass: bool             # greedy token match vs HF reference
    parity_max_logprob_delta: float
    # -- perf (medians over N runs) --
    decode_tok_s: float           # aggregate decode throughput
    ttft_ms: float                # median time-to-first-token
    peak_mem_gb: float
    run_stddev_tok_s: float       # run-to-run stddev, feeds accept threshold
    # -- verdict --
    accepted: bool = False
    reject_reason: str = ""
    timestamp: float = field(default_factory=time.time)
    extra: dict = field(default_factory=dict)   # batch sweep, kernel configs, etc.


def append(record: ExperimentRecord) -> None:
    #1. lock the file so six concurrent workers never interleave writes
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(asdict(record)) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def load() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    with open(LEDGER_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def current_best() -> dict | None:
    """
    @brief  Best accepted result so far (highest decode throughput among
            parity-passing, accepted experiments).
    """
    accepted = [r for r in load() if r["accepted"] and r["parity_pass"]]
    return max(accepted, key=lambda r: r["decode_tok_s"], default=None)


def noise_floor() -> float:
    """
    @brief  Estimate of benchmark noise: median run-to-run stddev across all
            parity-passing experiments. Used by the merge gate; falls back to
            1% of best throughput when the ledger is young.
    """
    stds = [r["run_stddev_tok_s"] for r in load() if r["parity_pass"]]
    if len(stds) >= 3:
        return statistics.median(stds)
    best = current_best()
    return 0.01 * best["decode_tok_s"] if best else 0.0
