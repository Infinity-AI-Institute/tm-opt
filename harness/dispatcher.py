"""
@brief  Dispatcher: drains experiment specs SERIALLY through the engine's
        single 4-GPU slot, and fast-forwards main on accepted winners.

        GPU layout (8x B300) — REWRITTEN 2026-07-28 for Stage-3 reality:
          GPUs 0-3 -> vLLM baseline, resident-idle (D8: never measured
                      concurrently with experiments; kept warm for the
                      same-hour A/B at Stage 4)
          GPUs 4-7 -> THE experiment slot (engine is TP=4; one experiment
                      at a time, by construction)

        Queue = directory of JSON spec files in experiments/queue/.
        The experiment loop (or you) drops specs in; the dispatcher does
        the rest: launch worker -> await verdict -> merge if accepted ->
        archive spec. Serial by design — no slot pool.
"""
import json
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUEUE_DIR = REPO_ROOT / "experiments" / "queue"
DONE_DIR = REPO_ROOT / "experiments" / "done"
LEDGER = REPO_ROOT / "experiments" / "ledger.jsonl"
POLL_S = 15


def launch(spec_path: Path) -> subprocess.Popen:
    #1. one worker at a time; worker pins CUDA_VISIBLE_DEVICES=4,5,6,7 itself;
    #   log lands next to the spec (and is the ledger row's log_path)
    log = open(spec_path.with_suffix(".log"), "w")
    return subprocess.Popen(
        ["python", str(REPO_ROOT / "harness" / "worker.py"),
         "--spec", str(spec_path)],
        cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)


def merge_if_accepted(spec_path: Path) -> None:
    """
    @brief  If the just-finished experiment's ledger row is accepted,
            fast-forward main to its commit so every subsequent worktree
            starts from the new best. Guards: [ralph]-prefixed commit,
            ff-only (a non-ff candidate means the agent branched stale —
            reject loudly, human rebases).
    """
    spec = json.load(open(spec_path))
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()] \
        if LEDGER.exists() else []
    mine = [r for r in rows if r.get("commit") == spec.get("commit")]
    if not mine or not mine[-1].get("accepted"):
        return
    commit = spec["commit"]
    subject = subprocess.run(f"git log --format=%s -1 {commit}", shell=True,
                             cwd=REPO_ROOT, capture_output=True, text=True).stdout
    if not subject.startswith("[ralph]"):
        print(f"[dispatcher] REFUSING merge of {commit[:8]}: no [ralph] prefix",
              flush=True)
        return
    r = subprocess.run(f"git merge --ff-only {commit}", shell=True,
                       cwd=REPO_ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[dispatcher] ff-only merge FAILED for {commit[:8]} "
              f"(stale base? human review): {r.stderr.strip()}", flush=True)
        return
    print(f"[dispatcher] merged {spec['experiment_id']} ({commit[:8]}) "
          f"-> best now {mine[-1]['tok_per_s']:.1f} tok/s "
          f"[{mine[-1].get('protocol')}]", flush=True)


def main():
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    DONE_DIR.mkdir(parents=True, exist_ok=True)
    running = None  # (proc, spec_path) — SINGLE slot

    print(f"[dispatcher] single 4-GPU slot (CUDA 4-7), watching {QUEUE_DIR}",
          flush=True)
    while True:
        #1. reap the finished worker, merge winner, archive its spec
        if running:
            proc, spec_path = running
            if proc.poll() is not None:
                merge_if_accepted(spec_path)
                print(f"[dispatcher] {spec_path.name} finished — archived to done/", flush=True)
                spec_path.rename(DONE_DIR / spec_path.name)
                running = None

        #2. slot free -> take the oldest spec
        if running is None:
            pending = sorted(QUEUE_DIR.glob("*.json"))
            if pending:
                spec_path = pending[0]
                running = (launch(spec_path), spec_path)
                print(f"[dispatcher] {spec_path.name} -> GPUs 4-7", flush=True)

        time.sleep(POLL_S)


if __name__ == "__main__":
    main()