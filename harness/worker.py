"""
@brief  Worker: runs ONE experiment end-to-end on ONE pinned GPU.

        Lifecycle:
          worktree checkout -> build -> launch engine -> correctness gate
          -> perf benchmark -> merge-gate verdict -> ledger append -> teardown

        Invoked by the dispatcher as:
          CUDA_VISIBLE_DEVICES=<slot> python harness/worker.py --spec exp.json
"""
import os
import json
import time
import argparse
import subprocess
from pathlib import Path

import requests

from ledger import ExperimentRecord, append, current_best, noise_floor
from correctness import run_gate

REPO_ROOT = Path(__file__).resolve().parent.parent
ACCEPT_MULTIPLIER = 2.0     # must beat best by > 2x noise floor to merge


def sh(cmd, cwd=None, check=True):
    """@brief thin subprocess wrapper with echo, so worker logs are auditable"""
    print(f"[worker] $ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, cwd=cwd, check=check,
                          capture_output=True, text=True)


def make_worktree(exp_id: str, base_branch: str) -> Path:
    #1. isolated checkout so six agents never collide on files
    wt = REPO_ROOT / "experiments" / f"wt-{exp_id}"
    sh(f"git worktree add {wt} {base_branch}", cwd=REPO_ROOT)
    return wt


def wait_healthy(endpoint: str, timeout_s: int = 600) -> bool:
    #1. poll until the engine's health endpoint answers or we give up
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(f"{endpoint}/health", timeout=5).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="experiment spec JSON")
    args = ap.parse_args()
    spec = json.load(open(args.spec))

    exp_id = spec["experiment_id"]
    gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    port = 8100 + gpu
    endpoint = f"http://localhost:{port}"

    wt = make_worktree(exp_id, spec.get("base_branch", "main"))
    engine_proc = None
    try:
        #1. apply the candidate patch (agent-authored) and record commits
        base_commit = sh("git rev-parse HEAD", cwd=wt).stdout.strip()
        if spec.get("patch_file"):
            sh(f"git apply {spec['patch_file']}", cwd=wt)
            sh(f"git commit -am 'exp {exp_id}: {spec['hypothesis']}'", cwd=wt)
        commit = sh("git rev-parse HEAD", cwd=wt).stdout.strip()

        #2. build (engine-specific; kept in the spec so the harness is generic)
        sh(spec["build_cmd"], cwd=wt)

        #3. launch the engine pinned to this GPU
        engine_proc = subprocess.Popen(
            spec["serve_cmd"].format(port=port), shell=True, cwd=wt,
            stdout=open(wt / "serve.log", "w"), stderr=subprocess.STDOUT,
        )
        if not wait_healthy(endpoint):
            raise RuntimeError("engine never became healthy; see serve.log")

        #4. correctness gate — hard stop on failure, perf never runs
        parity, delta, detail = run_gate(
            endpoint, spec["model_name"], spec["model_dir"], device="cuda:0"
        )

        rec = ExperimentRecord(
            experiment_id=exp_id, hypothesis=spec["hypothesis"],
            commit=commit, base_commit=base_commit, gpu_slot=gpu,
            parity_pass=parity, parity_max_logprob_delta=delta,
            decode_tok_s=0.0, ttft_ms=0.0, peak_mem_gb=0.0,
            run_stddev_tok_s=0.0, extra={"parity_detail": detail},
        )

        if parity:
            #5. perf benchmark at the contract concurrency level
            from benchmark import run as bench_run
            perf = bench_run(endpoint, spec["model_name"], gpu,
                             spec.get("concurrency", 128),
                             spec.get("max_tokens", 256))
            rec.decode_tok_s = perf["decode_tok_s"]
            rec.ttft_ms = perf["ttft_ms"]
            rec.peak_mem_gb = perf["peak_mem_gb"]
            rec.run_stddev_tok_s = perf["run_stddev_tok_s"]
            rec.extra["perf_rounds"] = perf["rounds"]

            #6. merge gate: strict win over current best, above noise floor
            best = current_best()
            threshold = (best["decode_tok_s"] if best else 0.0) \
                        + ACCEPT_MULTIPLIER * noise_floor()
            if rec.decode_tok_s > threshold:
                rec.accepted = True
            else:
                rec.reject_reason = (
                    f"tok/s {rec.decode_tok_s:.1f} <= threshold {threshold:.1f}"
                )
        else:
            rec.reject_reason = f"parity failure (max delta {delta:.4f}, {detail})"

        #7. single atomic ledger write — the only externally visible output
        append(rec)
        print(json.dumps({"experiment_id": exp_id, "accepted": rec.accepted,
                          "reason": rec.reject_reason}))
    finally:
        #8. teardown: kill engine, remove worktree, free the GPU slot
        if engine_proc:
            engine_proc.terminate()
            engine_proc.wait(timeout=30)
        sh(f"git worktree remove --force {wt}", cwd=REPO_ROOT, check=False)


if __name__ == "__main__":
    main()