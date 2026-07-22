#!/usr/bin/env python3
"""
Render the deliverable trajectory chart + ITERATION_LOG.md — FROM DATA ONLY.

Inputs (the only two sources; if the chart needs a datum not present, add the
field to the ledger schema — never hand-edit output):
  experiments/ledger.jsonl        rows per docs/LEDGER_SCHEMA.md
  configs/canonical.lock.json     `report` block: labels, footnote, headline

Outputs:
  docs/trajectory_<workload>.png  (one per workload with accepted rows)
  ITERATION_LOG.md                (per-iteration table, case-study style)

Usage: python scripts/plot_trajectory.py [--ledger experiments/ledger.jsonl]
"""
import argparse, json, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(ledger_path, lock_path):
    rows = [json.loads(l) for l in pathlib.Path(ledger_path).read_text().splitlines() if l.strip()]
    lock = json.loads(pathlib.Path(lock_path).read_text())
    key = lock["cache_key"]
    #1. validity guard: only rows measured under the frozen contract plot
    rows = [r for r in rows if r.get("cache_key") == key]
    return rows, lock


def plot_workload(rows, lock, workload):
    accepted = sorted([r for r in rows if r.get("workload") == workload
                       and r.get("accepted") and r.get("engine", "pyengine") != "vllm"],
                      key=lambda r: r["iteration"])
    baselines = {r["baseline_id"]: r for r in rows
                 if r.get("workload") == workload and r.get("engine") == "vllm"}
    if not accepted:
        print(f"[plot] {workload}: no accepted engine rows yet"); return

    fig, ax = plt.subplots(figsize=(11, 6))
    xs = [r["iteration"] for r in accepted]
    ys = [r["tok_per_s"] for r in accepted]
    ax.plot(xs, ys, "-o", color="#b8860b", lw=2, ms=4, label=f"{workload} — pyengine")
    #2. milestone dots: annotate every k-th + the last, like the case study
    for r in accepted[:: max(len(accepted) // 8, 1)] + [accepted[-1]]:
        ax.annotate(r.get("label", ""), (r["iteration"], r["tok_per_s"]),
                    textcoords="offset points", xytext=(0, 9), fontsize=7, ha="center")
    #3. dashed baseline lines from vllm iteration-0 rows (MTP off/on)
    for spec in lock["report"]["baseline_lines"]:
        b = baselines.get(spec["id"])
        if b:
            ax.axhline(b["tok_per_s"], ls="--", color="#b03030", lw=1)
            ax.text(xs[0], b["tok_per_s"], f' {spec["label"]}: {b["tok_per_s"]:,.0f} tok/s',
                    fontsize=8, color="#b03030", va="bottom")
    ax.set_xlabel("Optimization Iteration"); ax.set_ylabel("Total tok/s")
    ax.set_title(f"tm-opt Optimization Trajectory — Inkling on 4xB300 ({workload})")
    ax.figure.text(0.5, 0.005, lock["report"]["footnote"], fontsize=7,
                   ha="center", color="#666")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    out = f"docs/trajectory_{workload}.png"
    fig.tight_layout(); fig.savefig(out, dpi=160)
    print(f"[plot] wrote {out} ({len(accepted)} points)")


def write_iteration_log(rows, lock):
    lines = ["# ITERATION_LOG.md — generated from experiments/ledger.jsonl. Do not hand-edit.",
             "", "| iter | workload | label | mechanism | tok/s | vs vLLM | commit | log |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted([r for r in rows if r.get("engine", "pyengine") != "vllm"],
                    key=lambda r: (r["iteration"], r.get("workload", ""))):
        mark = "" if r.get("accepted") else " (rejected)"
        lines.append(f'| {r["iteration"]}{mark} | {r.get("workload","")} | '
                     f'{r.get("label","")} | {r.get("mechanism","")} | '
                     f'{r.get("tok_per_s",0):,.0f} | {r.get("pct_vs_baseline",0):.1f}% | '
                     f'{r.get("commit","")[:7]} | {r.get("log_path","")} |')
    pathlib.Path("ITERATION_LOG.md").write_text("\n".join(lines) + "\n")
    print(f"[plot] wrote ITERATION_LOG.md ({len(lines)-4} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="experiments/ledger.jsonl")
    ap.add_argument("--lock", default="configs/canonical.lock.json")
    a = ap.parse_args()
    rows, lock = load(a.ledger, a.lock)
    for wl in ("decode_heavy", "prefill_heavy"):
        plot_workload(rows, lock, wl)
    write_iteration_log(rows, lock)


if __name__ == "__main__":
    main()
