"""
@brief  Dispatcher: drains a queue of experiment specs across the six
        experiment GPUs, one experiment per GPU, and merges winners.

        GPU layout (8x B300):
          slots 0-5  -> experiment workers
          slot  6    -> vLLM baseline, kept warm for same-hour A/B numbers
          slot  7    -> full validation of merge candidates

        Queue = directory of JSON spec files in experiments/queue/.
        Agents (or you) drop specs in; the dispatcher does the rest.
"""
import json
import time
import subprocess
from pathlib import Path

from ledger import load, current_best

REPO_ROOT = Path(__file__).resolve().parent.parent
QUEUE_DIR = REPO_ROOT / "experiments" / "queue"
DONE_DIR = REPO_ROOT / "experiments" / "done"
EXPERIMENT_SLOTS = [0, 1, 2, 3, 4, 5]
POLL_S = 15


def launch(spec_path: Path, gpu: int) -> subprocess.Popen:
    #1. pin the worker to its slot; logs land next to the spec for auditing
    log = open(spec_path.with_suffix(".log"), "w")
    return subprocess.Popen(
        ["python", str(REPO_ROOT / "harness" / "worker.py"), "--spec", str(spec_path)],
        env={"CUDA_VISIBLE_DEVICES": str(gpu), "PATH": "/usr/local/cuda/bin:/usr/bin:/bin"},
        stdout=log, stderr=subprocess.STDOUT,
    )


def merge_if_accepted(spec_path: Path) -> None:
    """
    @brief  If the just-finished experiment was accepted, fast-forward main to
            its commit so every subsequent worktree starts from the new best.
    """
    spec = json.load(open(spec_path))
    records = {r["experiment_id"]: r for r in load()}
    rec = records.get(spec["experiment_id"])
    if rec and rec["accepted"]:
        #1. merge the winning commit into main
        subprocess.run(f"git merge --ff-only {rec['commit']}",
                       shell=True, cwd=REPO_ROOT, check=False)
        best = current_best()
        print(f"[dispatcher] merged {rec['experiment_id']} "
              f"-> best now {best['decode_tok_s']:.1f} tok/s", flush=True)


def main():
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    DONE_DIR.mkdir(parents=True, exist_ok=True)
    running: dict[int, tuple[subprocess.Popen, Path]] = {}

    print(f"[dispatcher] slots {EXPERIMENT_SLOTS}, watching {QUEUE_DIR}", flush=True)
    while True:
        #1. reap finished workers, merge winners, archive their specs
        for gpu in list(running):
            proc, spec_path = running[gpu]
            if proc.poll() is not None:
                merge_if_accepted(spec_path)
                spec_path.rename(DONE_DIR / spec_path.name)
                del running[gpu]

        #2. fill free slots from the queue, oldest spec first
        pending = sorted(QUEUE_DIR.glob("*.json"))
        for gpu in EXPERIMENT_SLOTS:
            if gpu in running or not pending:
                continue
            spec_path = pending.pop(0)
            running[gpu] = (launch(spec_path, gpu), spec_path)
            print(f"[dispatcher] {spec_path.name} -> GPU {gpu}", flush=True)

        time.sleep(POLL_S)


if __name__ == "__main__":
    main()