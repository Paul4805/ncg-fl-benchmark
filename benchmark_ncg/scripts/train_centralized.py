"""Train the 6 centralized CNN-2 references (homogeneous mode only).

    python -m benchmark_ncg.scripts.train_centralized --all
    python -m benchmark_ncg.scripts.train_centralized --dataset BloodMNIST --seed 0

Output: results/centralized/cent__<dataset>__cnn2__s<seed>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_ncg.centralized.train import train_centralized  # noqa: E402
from benchmark_ncg.core import config  # noqa: E402
from benchmark_ncg.core.train_utils import Logger  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(config.DATASET_NAMES))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    datasets = [args.dataset] if args.dataset else list(config.DATASET_NAMES)
    seeds = [args.seed] if args.seed is not None else list(config.SEEDS)
    if args.seed is not None and args.seed not in config.SEEDS:
        ap.error(f"seed must be one of {list(config.SEEDS)}")

    for dataset in datasets:
        for seed in seeds:
            path = config.central_path(dataset, seed)
            if path.exists() and not args.force:
                print(f"[centralized] reuse {path.name}")
                continue
            logger = Logger(config.LOGS_DIR / "centralized" / f"{config.central_key(dataset, seed)}.log")
            result = train_centralized(dataset, seed, logger=logger)
            config.dump_json(result, path)
            logger(f"[centralized] wrote {path.name}")
            logger.close()


if __name__ == "__main__":
    main()
