"""
@brief  Worker: runs ONE experiment end-to-end on the engine's 4-GPU slot.

        Lifecycle:
          worktree checkout (agent's committed ref) -> launch engine (GPUs
          4-7, TP=4 resident) -> D13 teacher-forced parity gate -> perf
          benchmark (timeboxed until the engine clears the canonical floor,
          canonical after) -> per-workload merge-gate verdict -> ledger row
          (accepted OR rejected — every measured attempt is chartable data)
          -> teardown.

        Invoked by the dispatcher as:
          python harness/worker.py --spec experiments/queue/<id>.json

        REALITY NOTES (rewritten 2026-07-28 for Stage 3):
        - The engine is TP=4: exactly ONE experiment runs at a time, on
          GPUs 4-7 (vLLM baseline stays resident-idle on 0-3, D8).
        - Parity = D13 teacher-forced envelope (harness/correctness.py
          --teacher-forced), NOT free-run exact-token (that form is
          deterministic self-consistency; see D13).
        - Bench protocol auto-selects: timeboxed(1800) below the canonical
          feasibility floor (~1,165 tok/s aggregate, B4.1 math), canonical
          above it. Rows compare like-for-like by protocol.
        - Merge bars are PER-WORKLOAD (D12): max(2 x baseline noise, 0.3%).
"""
import json
import os
import signal
import subprocess
import sys
import time
import argparse
import pathlib

import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from benchmark import run_canonical, run_timeboxed, append_ledger_row, git_head  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LEDGER = REPO_ROOT / "experiments" / "ledger.jsonl"
ENVELOPE = REPO_ROOT / "experiments" / "tf_envelope.json"
ENGINE_GPUS = "4,5,6,7"
PORT = 8200
CANONICAL_FLOOR_TOKS = 1165.0   # B4.1: below this, canonical cannot terminate
MERGE_BAR_MIN_PCT = 0.3         # D12


def sh(cmd, cwd=None, check=True, env=None):
    """@brief thin subprocess wrapper with echo, so worker logs are auditable"""
    print(f"[worker] $ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, cwd=cwd, check=check,
                          capture_output=True, text=True, env=env)


def ledger_rows():
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def wait_healthy(endpoint: str, timeout_s: int = 600) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(f"{endpoint}/health", timeout=5).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    return False


def run_parity(endpoint: str) -> dict:
    """
    @brief  D13 teacher-forced envelope gate via the frozen harness CLI.
    @return the gate's verdict JSON (parity_pass, agree_rate, deltas, ...)
    """
    try:
        r = sh(f"python {REPO_ROOT}/harness/correctness.py --endpoint {endpoint} "
               f"--teacher-forced --envelope {ENVELOPE}", check=False,
               timeout=3600)
    except subprocess.TimeoutExpired:
        raise RuntimeError("parity gate exceeded 3600 s — engine wedged or "
                           "unresponsive; treat as parity red")
    #1. last stdout line is the verdict JSON (progress lines precede it)
    for line in reversed(r.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"parity gate produced no verdict; stderr: {r.stderr[-500:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    spec = json.load(open(args.spec))
    exp_id = spec["experiment_id"]
    workload = spec.get("workload", "decode_heavy")
    cfg_path = REPO_ROOT / "configs" / f"canonical_{workload}.json"
    cfg = json.loads(cfg_path.read_text())
    endpoint = f"http://localhost:{PORT}"

    #1. worktree at the agent's COMMITTED ref (the agent commits in its own
    #   worktree per PROMPT_EXPERIMENT; the spec carries that commit)
    commit = spec["commit"]
    if not sh(f"git log --format=%s -1 {commit}", cwd=REPO_ROOT).stdout.startswith("[ralph]"):
        print(f"[worker] WARNING: candidate commit {commit[:8]} lacks [ralph] prefix")
    wt = REPO_ROOT / "experiments" / f"wtrun-{exp_id}"
    sh(f"git worktree add --detach {wt} {commit}", cwd=REPO_ROOT)

    engine_proc = None
    verdict = {"experiment_id": exp_id, "accepted": False, "reason": None}
    try:
        #2. launch the candidate engine on the 4-GPU slot
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ENGINE_GPUS}
        serve_cmd = spec.get("serve_cmd",
                             f"python -m engine.pyengine.server --port {PORT}")
        engine_proc = subprocess.Popen(
            serve_cmd, shell=True, cwd=wt, env=env, preexec_fn=os.setsid,
            stdout=open(wt / "serve.log", "w"), stderr=subprocess.STDOUT)
        if not wait_healthy(endpoint):
            raise RuntimeError("engine never became healthy; see serve.log")

        #3. D13 parity gate — hard stop on failure, perf never runs
        parity = run_parity(endpoint)
        row_common = {
            "engine": "pyengine", "workload": workload,
            "label": spec.get("label", exp_id),
            "mechanism": spec.get("mechanism", ""),
            "hypothesis": spec.get("hypothesis", ""),
            "predicted_delta": spec.get("predicted_delta"),
            "baseline_id": spec.get("baseline_id", "vllm_mtp_off"),
            "cache_key": cfg["cache_key"], "commit": commit,
            "parity": {k: parity.get(k) for k in
                       ("parity_pass", "agree_rate", "delta_mean", "delta_max")},
            "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "log_path": str(pathlib.Path(args.spec).with_suffix(".log")),
        }
        if not parity.get("parity_pass"):
            verdict["reason"] = f"parity failure ({row_common['parity']})"
            append_ledger_row({**row_common,
                               "iteration": _next_iteration(workload),
                               "tok_per_s": 0.0, "pct_vs_baseline": 0.0,
                               "accepted": False, "noise_floor_pct": 0.0,
                               "protocol": "none (parity red)",
                               "reject_reason": verdict["reason"], "blocks": []})
            return

        #4. perf: protocol auto-selection (like-for-like comparisons only)
        best = _current_best(workload)
        use_canonical = best is not None and best >= CANONICAL_FLOOR_TOKS
        if use_canonical:
            rec = run_canonical(endpoint, spec.get("model",
                  "/workspace/models/inkling-nvfp4"), cfg, repeats=3)
        else:
            rec = run_timeboxed(endpoint, spec.get("model",
                  "/workspace/models/inkling-nvfp4"), cfg,
                  max_seconds=1800, tb_conc=8, tb_osl=128)

        #5. merge gate: per-workload bar over the best SAME-PROTOCOL row
        base_noise = _baseline_noise(workload)
        bar_pct = max(2 * base_noise, MERGE_BAR_MIN_PCT)
        best_same = _current_best(workload, protocol=rec["protocol"])
        threshold = (best_same or 0.0) * (1 + bar_pct / 100.0)
        accepted = rec["tok_per_s"] > threshold
        if not accepted:
            verdict["reason"] = (f"tok/s {rec['tok_per_s']:.1f} <= threshold "
                                 f"{threshold:.1f} (best {best_same} +{bar_pct}%)")
        verdict["accepted"] = accepted

        base_tok = _baseline_tok(workload)
        append_ledger_row({**row_common,
                           "iteration": _next_iteration(workload),
                           "tok_per_s": rec["tok_per_s"],
                           "pct_vs_baseline": round(100 * rec["tok_per_s"] /
                                                    base_tok, 3) if base_tok else 0.0,
                           "accepted": accepted,
                           "reject_reason": verdict["reason"],
                           "noise_floor_pct": rec["noise_floor_pct"],
                           "protocol": rec["protocol"],
                           "blocks": rec["blocks"]})
    finally:
        if engine_proc:
            try:
                os.killpg(engine_proc.pid, signal.SIGTERM)
                engine_proc.wait(timeout=30)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(engine_proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        sh(f"git worktree remove --force {wt}", cwd=REPO_ROOT, check=False)
        print(json.dumps(verdict), flush=True)


def _rows_for(workload, engine="pyengine"):
    return [r for r in ledger_rows()
            if r.get("workload") == workload and r.get("engine") == engine]


def _next_iteration(workload) -> int:
    rows = _rows_for(workload)
    return max((r.get("iteration", 0) for r in rows), default=-1) + 1


def _current_best(workload, protocol=None):
    rows = [r for r in _rows_for(workload) if r.get("accepted")]
    if protocol:
        rows = [r for r in rows if r.get("protocol") == protocol]
    return max((r["tok_per_s"] for r in rows), default=None)


def _baseline_noise(workload) -> float:
    vs = [r for r in ledger_rows()
          if r.get("engine") == "vllm" and r.get("workload") == workload]
    return float(vs[-1].get("noise_floor_pct", 0.0)) if vs else 0.0


def _baseline_tok(workload):
    vs = [r for r in ledger_rows()
          if r.get("engine") == "vllm" and r.get("workload") == workload]
    return vs[-1]["tok_per_s"] if vs else None


if __name__ == "__main__":
    main()