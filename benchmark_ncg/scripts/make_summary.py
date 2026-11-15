"""Build summary_ncg.csv and summary_class_gap.csv (mean / std over the 3 seeds)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from benchmark_ncg.core import config, report  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-raw", action="store_true")
    args = ap.parse_args(argv)
    raw = config.RESULTS_DIR / "raw_results.csv"
    if args.from_raw and raw.exists():
        df = pd.read_csv(raw)
        ncg_df = report.summary_ncg(df)
        gap_df = report.summary_class_gap(df)
        ncg_df.to_csv(config.RESULTS_DIR / "summary_ncg.csv", index=False)
        gap_df.to_csv(config.RESULTS_DIR / "summary_class_gap.csv", index=False)
    else:
        df, ncg_df, gap_df = report.write_summaries()

    pd.set_option("display.width", 180)
    print(f"[make_summary] {config.RESULTS_DIR / 'summary_ncg.csv'}")
    print(ncg_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n[make_summary] {config.RESULTS_DIR / 'summary_class_gap.csv'}")
    print(gap_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
