"""Reference-sensitivity table: fixed (homogeneous) FL recall vectors, two centralized refs.

Reads the 36 valid homogeneous FedAPEN/PHP-FL runs and recomputes NCG against
  (a) the single CNN-2 centralized reference (main table), and
  (b) the method's per-architecture mean reference (144 valid hetero refs).
No training; pure recomputation.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from benchmark_ncg.core import config, metrics  # noqa: E402
from benchmark_ncg.core import hetero_config as hc  # noqa: E402


def ncg_from(fl_gap, cent_gap):
    return (fl_gap - cent_gap) / (1.0 - cent_gap)


def main():
    rows = []
    for method in hc.HET_METHODS:
        for setting in ["B1", "B2", "B3"]:
            for dataset in config.DATASET_NAMES:
                for seed in config.SEEDS:
                    run = config.load_json(config.run_path(setting, method, dataset, seed))
                    fl_rec = np.asarray(run["per_class_recall"], dtype=float)
                    fl_gap = metrics.class_gap(fl_rec)
                    homo = config.load_json(config.central_path(dataset, seed))
                    het = config.load_json(hc.mean_path(method, dataset, seed))
                    hg = float(homo["class_gap"])
                    tg = float(het["cent_class_gap_hetero"])
                    a = ncg_from(fl_gap, hg)
                    b = ncg_from(fl_gap, tg)
                    rows.append({
                        "method": method, "setting": setting, "dataset": dataset,
                        "seed": int(seed), "fl_class_gap": round(fl_gap, 6),
                        "cent_class_gap_homogeneous": round(hg, 6),
                        "ncg_homogeneous_ref": round(a, 6),
                        "cent_class_gap_per_architecture": round(tg, 6),
                        "ncg_per_architecture_ref": round(b, 6),
                        "delta_ncg_ref_change": round(b - a, 6),
                        "note": ("fixed homogeneous FL model; only the centralized reference "
                                 "construction changes"),
                    })
    df = pd.DataFrame(rows).sort_values(["method", "setting", "dataset", "seed"])
    out = config.RESULTS_DIR / "summary_reference_sensitivity.csv"
    df.to_csv(out, index=False)
    print(f"[sensitivity] wrote {out} ({len(df)} rows)")
    print(df.groupby(["method", "setting", "dataset"])[
        ["ncg_homogeneous_ref", "ncg_per_architecture_ref", "delta_ncg_ref_change"]
    ].mean().round(4).to_string())


if __name__ == "__main__":
    main()
