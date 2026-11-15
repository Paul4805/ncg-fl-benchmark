"""Hourly status + next-batch planner.

    python -m scripts.status                 # status report
    python -m scripts.status --next-batch 100  # how many jobs fit ~100 worker-minutes
    python -m scripts.status --start-queue     # record the queue start time

Estimates use the measured wall-clock times of completed runs when available and
fall back to conservative per-(method, dataset) defaults.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core import config, report  # noqa: E402
from scripts.run_all import build_tasks  # noqa: E402

DEFAULT_MIN = {
    ("BloodMNIST", "FedAvg"): 3.0, ("BloodMNIST", "FedFV"): 4.0,
    ("BloodMNIST", "FedAPEN"): 7.0, ("BloodMNIST", "PHP-FL"): 8.0,
    ("FashionMNIST", "FedAvg"): 6.5, ("FashionMNIST", "FedFV"): 9.0,
    ("FashionMNIST", "FedAPEN"): 17.0, ("FashionMNIST", "PHP-FL"): 18.0,
}
STATE = config.RESULTS_DIR / "queue_state.json"


def measured_minutes():
    """Median wall-clock minutes per (dataset, method) from completed runs."""
    out = {}
    buckets = {}
    for p in config.RUNS_DIR.glob("*.json"):
        try:
            d = config.load_json(p)
        except Exception:
            continue
        key = (d["dataset"], d["method"])
        buckets.setdefault(key, []).append(float(d.get("wall_clock_seconds", 0.0)) / 60.0)
    for key, vals in buckets.items():
        if vals:
            out[key] = float(np.median(vals))
    return out


def estimate(method, dataset, measured):
    if (dataset, method) in measured:
        return measured[(dataset, method)]
    return DEFAULT_MIN.get((dataset, method), 10.0)


def queue_state():
    if STATE.exists():
        return config.load_json(STATE)
    return {}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--next-batch", type=float, default=None, metavar="WORKER_MINUTES")
    ap.add_argument("--start-queue", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)

    if args.start_queue:
        config.dump_json({"started_at": time.time(),
                          "started_human": time.strftime("%Y-%m-%d %H:%M:%S")}, STATE)
        print(f"[status] queue start recorded: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return

    tasks = build_tasks(resume=True)
    measured = measured_minutes()
    remaining_fl = [t for t in tasks if t.kind == "fl"]
    remaining_cent = [t for t in tasks if t.kind == "centralized"]

    manifest = report.build_manifest()
    t = manifest["totals"]

    state = queue_state()
    elapsed_h = (time.time() - state["started_at"]) / 3600.0 if "started_at" in state else 0.0

    print(f"[status] {time.strftime('%Y-%m-%d %H:%M:%S')}  "
          f"queue elapsed: {elapsed_h:.2f} h / {config.WALL_CLOCK_BUDGET_HOURS:.0f} h budget")
    print(f"[status] centralized {t['centralized_complete']}/{t['centralized']} | "
          f"FL {t['fl_complete']}/{t['fl']} | remaining tasks queued: {len(tasks)}")

    # per (setting, dataset) progress matrix
    done = {}
    for p in config.RUNS_DIR.glob("*.json"):
        try:
            d = config.load_json(p)
        except Exception:
            continue
        done[(d["setting"], d["dataset"])] = done.get((d["setting"], d["dataset"]), 0) + 1
    print("\n[status] FL completion by cell (out of 12 = 4 methods x 3 seeds)")
    for setting in ["B1", "B2", "B3"]:
        row = "  ".join(f"{ds}={done.get((setting, ds), 0):2d}/12" for ds in config.DATASET_NAMES)
        print(f"  {setting}: {row}")

    if remaining_fl:
        total_min = 0.0
        for tk in remaining_fl:
            parts = tk.task_id.split("__")
            method, dataset = parts[1], parts[2]
            total_min += estimate(method, dataset, measured)
        wall_h = total_min / max(1, args.workers) / 60.0
        print(f"\n[status] estimated remaining compute: {total_min:.0f} worker-min "
              f"= {wall_h:.1f} h wall with {args.workers} workers")
        if "started_at" in state:
            print(f"[status] projected completion: "
                  f"{time.strftime('%H:%M', time.localtime(time.time() + total_min / max(1, args.workers) * 60))}")

    ncg_path = config.RESULTS_DIR / "summary_ncg.csv"
    if ncg_path.exists():
        import pandas as pd
        ncg = pd.read_csv(ncg_path)
        ncg = ncg[ncg["method"] != "Centralized"]
        if not ncg.empty:
            print("\n[status] partial NCG (mean over completed seeds)")
            for dataset in config.DATASET_NAMES:
                sub = ncg[ncg["dataset"] == dataset]
                if sub.empty:
                    continue
                print(f"  {dataset}")
                for method in config.FL_METHODS:
                    cells = []
                    for setting in ["B1", "B2", "B3"]:
                        row = sub[(sub["method"] == method) & (sub["setting"] == setting)]
                        cells.append(f"{setting}={row['ncg_mean'].iloc[0]:+.3f}"
                                     if len(row) else f"{setting}=  n/a")
                    print(f"    {method:10s} " + "  ".join(cells))

    if args.next_batch:
        budget = float(args.next_batch)
        chosen, used = [], 0.0
        for tk in remaining_fl:
            parts = tk.task_id.split("__")
            m = estimate(parts[1], parts[2], measured)
            if chosen and used + m > budget:
                break
            chosen.append(tk)
            used += m
        print(f"\n[status] next batch: {len(chosen)} jobs, ~{used:.0f} worker-min "
              f"(~{used / max(1, args.workers):.0f} min wall)")
        for tk in chosen:
            print(f"  - {tk.task_id}")
        print(f"NEXT_BATCH_SIZE={len(chosen)}")


if __name__ == "__main__":
    main()
