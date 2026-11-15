"""Hourly status for the heterogeneous queue (read-only)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from benchmark_ncg.core import config, hetero_config as hc  # noqa: E402


def queue_start() -> float | None:
    times = []
    for d in (config.LOGS_DIR / "run_all_hetero", hc.HET_CENTRAL_DIR, hc.HET_RUNS_DIR):
        if d.exists():
            for p in d.glob("*"):
                if "__SMOKE__" in p.name:
                    continue
                times.append(p.stat().st_mtime)
    return min(times) if times else None


def main():
    t0 = queue_start()
    elapsed = (time.time() - t0) / 3600 if t0 else 0.0
    print(f"[hetero status] {time.strftime('%Y-%m-%d %H:%M:%S')}  "
          f"queue elapsed {elapsed:.2f} h / 12 h hard cap (internal stop 10 h)")

    refs = [hc.ref_path(m, d, a, s) for m in hc.HET_METHODS for d in config.DATASET_NAMES
            for s in config.SEEDS for a in hc.arch_ids(m)]
    refs_done = sum(p.exists() for p in refs)
    means = [hc.mean_path(m, d, s) for m in hc.HET_METHODS for d in config.DATASET_NAMES
             for s in config.SEEDS]
    print(f"[hetero status] per-arch refs {refs_done}/{len(refs)} | "
          f"mean refs {sum(p.exists() for p in means)}/{len(means)}")

    runs = []
    for setting in config.SETTINGS:
        for method in hc.HET_METHODS:
            for dataset in config.DATASET_NAMES:
                for seed in config.SEEDS:
                    if setting not in hc.HET_SETTINGS[method]:
                        continue
                    p = hc.run_path(setting, method, dataset, seed)
                    if p.exists():
                        try:
                            runs.append(config.load_json(p))
                        except Exception:
                            pass
    expected = hc.num_expected_runs()
    print(f"[hetero status] FL runs {len(runs)}/{expected}")
    print("  cell completion (out of 3 seeds)")
    for method in hc.HET_METHODS:
        for setting in hc.HET_SETTINGS[method]:
            row = "  ".join(
                f"{d}={sum(1 for r in runs if r['method'] == method and r['setting'] == setting and r['dataset'] == d)}/3"
                for d in config.DATASET_NAMES)
            print(f"    {method:8s} {setting}: {row}")

    if runs:
        print("\n[hetero status] partial NCG_hetero (mean over completed seeds)")
        for dataset in config.DATASET_NAMES:
            for method in hc.HET_METHODS:
                cells = []
                for setting in hc.HET_SETTINGS[method]:
                    g = [r["ncg_hetero"] for r in runs
                         if r["dataset"] == dataset and r["method"] == method
                         and r["setting"] == setting]
                    cells.append(f"{setting}={np.mean(g):+.3f}" if g else f"{setting}=n/a")
                print(f"    {dataset:13s} {method:8s} " + "  ".join(cells))

    # ETA from completed wall clocks
    if runs:
        done_min = sum(float(r.get("wall_clock_seconds", 0)) for r in runs) / 60.0
        per_run = done_min / len(runs)
        remaining = expected - len(runs)
        est_fl = per_run * remaining / 60.0
        print(f"\n[hetero status] mean FL wall {per_run:.1f} min/run; "
              f"estimated remaining FL {est_fl:.1f} GPU-h ({remaining} runs)")
    ref_min = 0.0
    if refs_done:
        ref_times = [p.stat().st_mtime for p in refs if p.exists()]
        print(f"[hetero status] ref files so far: {len(ref_times)}")

    print(f"[hetero status] hard stop at {12 - elapsed:.1f} h remaining; "
          f"internal stop at {10 - elapsed:.1f} h remaining")


if __name__ == "__main__":
    main()
