"""
@brief  Export experiments/ledger.jsonl + canonical.lock.json into ONE
        web-ready JSON (docs/web/trajectory.json) for the external
        scrollytelling page (Infinity case-study style: stepped tok/s line
        drawn progressively on scroll, dashed vLLM baseline, per-iteration
        annotation callouts from label/mechanism, rejected attempts shown
        as muted markers).

        The page repo NEVER reads the ledger directly — this exporter is
        the single contract between repos. Regenerate + commit on every
        review; the page fetches the raw GitHub URL.

        Output schema (versioned):
        {
          "schema": 1, "generated_at": ..., "cache_key": ...,
          "workloads": {
            "<workload>": {
              "baseline": {"tok_per_s", "noise_floor_pct", "merge_bar_pct",
                            "engine": "vllm", "protocol": "canonical"},
              "series": [ {iteration, tok_per_s, pct_vs_baseline, label,
                           mechanism, accepted, protocol, commit,
                           measured_at, hypothesis?, predicted_delta?,
                           log_path}, ... ]   # iteration-ascending
            }
          }
        }
"""
import json
import pathlib
import time

LEDGER = pathlib.Path("experiments/ledger.jsonl")
LOCK = pathlib.Path("configs/canonical.lock.json")
OUT = pathlib.Path("docs/web/trajectory.json")

MERGE_BAR_MIN_PCT = 0.3  # D12: max(2 x noise, 0.3%)


def main():
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    cache_keys = {r.get("cache_key") for r in rows}

    out = {"schema": 1,
           "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "cache_key": sorted(k for k in cache_keys if k),
           "workloads": {}}

    workloads = sorted({r.get("workload") for r in rows if r.get("workload")})
    for w in workloads:
        wrows = [r for r in rows if r.get("workload") == w]
        #1. baseline = LAST canonical vllm row (same rule as benchmark.py's
        #   pct scan — keep the two consistent forever)
        base = None
        for r in wrows:
            if r.get("engine") == "vllm":
                base = r
        wl = {"baseline": None, "series": []}
        if base:
            noise = float(base.get("noise_floor_pct") or 0.0)
            wl["baseline"] = {
                "tok_per_s": base["tok_per_s"],
                "noise_floor_pct": noise,
                "merge_bar_pct": round(max(2 * noise, MERGE_BAR_MIN_PCT), 2),
                "engine": "vllm",
                "protocol": base.get("protocol", "canonical"),
            }
        #2. engine series: every non-vllm row, accepted AND rejected —
        #   rejected attempts are part of the story (muted markers on the page)
        for r in wrows:
            if r.get("engine") == "vllm":
                continue
            wl["series"].append({k: r.get(k) for k in (
                "iteration", "tok_per_s", "pct_vs_baseline", "label",
                "mechanism", "accepted", "protocol", "commit",
                "measured_at", "hypothesis", "predicted_delta", "log_path")})
        wl["series"].sort(key=lambda r: (r.get("iteration") or 0,
                                         r.get("measured_at") or ""))
        out["workloads"][w] = wl

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    n = sum(len(w["series"]) for w in out["workloads"].values())
    print(f"[export] wrote {OUT} — {len(workloads)} workloads, "
          f"{n} engine rows, baselines "
          f"{[w['baseline']['tok_per_s'] if w['baseline'] else None for w in out['workloads'].values()]}")


if __name__ == "__main__":
    main()