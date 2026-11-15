"""Compute NCG for every stored run, rebuild raw_results.csv and the manifest.

    python -m benchmark_ncg.scripts.compute_ncg

NCG = (ClassGap_FL - ClassGap_Centralized) / (1 - ClassGap_Centralized)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from benchmark_ncg.core import config, report  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args = ap.parse_args(argv)
    df = report.write_raw()
    print(f"[compute_ncg] raw_results.csv rows={len(df)} (FL + reference rows)")
    if df.empty:
        return

    fl = df[df["method"] != "Centralized"]
    missing = int(fl["ncg"].isna().sum())
    if missing:
        print(f"[compute_ncg] WARNING {missing} FL rows lack a centralized reference")

    for dataset in config.DATASET_NAMES:
        print(f"\n{dataset}")
        print(f"  {'setting':8s} {'method':10s} {'NCG mean':>9s} {'std':>7s} "
              f"{'FL gap':>8s} {'cent gap':>9s} {'n':>2s}")
        for setting in ["B1", "B2", "B3"]:
            for method in config.FL_METHODS:
                g = fl[(fl["setting"] == setting) & (fl["method"] == method) &
                       (fl["dataset"] == dataset)]
                if g.empty:
                    continue
                ncg = g["ncg"].astype(float)
                print(f"  {setting:8s} {method:10s} {ncg.mean():>9.4f} "
                      f"{ncg.std(ddof=1) if ncg.notna().sum() > 1 else 0.0:>7.4f} "
                      f"{g['class_gap'].astype(float).mean():>8.4f} "
                      f"{g['cent_class_gap'].astype(float).mean():>9.4f} "
                      f"{g['seed'].nunique():>2d}")

    manifest = report.write_manifest()
    t = manifest["totals"]
    print(f"\n[compute_ncg] manifest: centralized {t['centralized_complete']}/{t['centralized']}, "
          f"FL {t['fl_complete']}/{t['fl']}, grand {t['grand']}")


if __name__ == "__main__":
    main()
