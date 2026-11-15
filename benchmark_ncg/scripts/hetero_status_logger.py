"""Append an hourly hetero status snapshot to logs/hetero_status.log (background)."""
from __future__ import annotations

import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_ncg.core import config, hetero_config as hc  # noqa: E402
from benchmark_ncg.scripts import status_hetero  # noqa: E402

LOG = config.LOGS_DIR / "hetero_status.log"
MAX_HOURS = 14


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    for i in range(MAX_HOURS):
        buf = io.StringIO()
        with redirect_stdout(buf):
            status_hetero.main()
        text = buf.getvalue().strip()
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"\n===== snapshot {i + 1} @ {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            f.write(text + "\n")
        print(f"[status-logger] snapshot {i + 1} written", flush=True)
        if sum(hc.run_path(s, m, d, sd).exists()
               for m in hc.HET_METHODS for s in hc.HET_SETTINGS[m]
               for d in config.DATASET_NAMES for sd in config.SEEDS) >= hc.num_expected_runs():
            print("[status-logger] all FL runs complete, stopping", flush=True)
            break
        time.sleep(3600)


if __name__ == "__main__":
    main()
