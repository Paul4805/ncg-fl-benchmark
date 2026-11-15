"""Train the per-architecture centralized references for one (method, dataset, seed).

    python -m benchmark_ncg.scripts.train_centralized_hetero --method FedAPEN \
        --dataset BloodMNIST --seed 0

Trains one centralized model per distinct client architecture of the method
(K=20 official CNN2_BN widths for FedAPEN, K=4 approved substitutes for PHP-FL)
on the full aggregated training set with the SAME protocol as the main table
(max_epochs=250, ES patience 20, official val for BloodMNIST / 10% train for
FashionMNIST), then writes the mean reference:

  results/centralized_hetero/<Method>__<Dataset>__<arch_id>__s<seed>__R250.json
  results/centralized_hetero/<Method>__<Dataset>__mean__s<seed>__R250.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from benchmark_ncg.centralized.train import train_centralized  # noqa: E402
from benchmark_ncg.core import config, metrics  # noqa: E402
from benchmark_ncg.core import hetero_config as hc  # noqa: E402
from benchmark_ncg.core.train_utils import Logger  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=list(hc.HET_METHODS))
    ap.add_argument("--dataset", choices=list(config.DATASET_NAMES))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    methods = list(hc.HET_METHODS) if args.all else [args.method]
    datasets = list(config.DATASET_NAMES) if args.all else [args.dataset]
    seeds = list(config.SEEDS) if args.all else [args.seed]
    if None in methods or None in datasets or None in seeds:
        ap.error("provide --method/--dataset/--seed or --all")

    ok, diffs = hc.verify_main_table()
    if not ok:
        raise SystemExit(f"main-table checksum changed: {diffs}")

    for method in methods:
        for dataset in datasets:
            for seed in seeds:
                t0 = time.time()
                logger = Logger(config.LOGS_DIR / "hetero_centralized" /
                                f"{method}__{dataset}__s{seed}.log")
                logger(f"=== hetero refs {method} {dataset} s{seed} "
                       f"K={len(hc.arch_ids(method))} max_epochs={config.CENTRAL_TRAINING['max_epochs']}")
                for aid in hc.arch_ids(method):
                    path = hc.ref_path(method, dataset, aid, seed)
                    if path.exists() and not args.force:
                        logger(f"reuse {path.name}")
                        continue
                    spec = hc.arch_spec(method, aid)
                    label = spec["name"] + (f"(d={spec['params']['channel_diff']})"
                                            if spec.get("params") else "")
                    res = train_centralized(
                        dataset, seed, logger=logger, model_spec=spec, model_label=label,
                        run_id_override=f"{method}__{dataset}__{aid}__s{seed}__R250",
                    )
                    res.update({"method": method, "arch_id": aid,
                                "arch_spec": spec, "mode": "per_architecture"})
                    config.dump_json(res, path)

                refs = [config.load_json(hc.ref_path(method, dataset, a, seed))
                        for a in hc.arch_ids(method)]
                vecs = [np.asarray(r["per_class_recall"], dtype=float) for r in refs]
                cent = metrics.average_recall_vectors(vecs)
                mean = {
                    "run_id": f"{method}__{dataset}__mean__s{seed}__R250",
                    "kind": "centralized_hetero_mean",
                    "method": method,
                    "dataset": dataset,
                    "seed": int(seed),
                    "mode": "per_architecture",
                    "arch_ids": hc.arch_ids(method),
                    "arch_backbones": [r["backbone_label"] for r in refs],
                    "per_arch_gaps": {r["backbone_label"]: r["class_gap"] for r in refs},
                    "per_arch_recall": [r["per_class_recall"] for r in refs],
                    "cent_recall_hetero": [float(v) for v in cent],
                    "cent_class_gap_hetero": metrics.class_gap(cent),
                    "minority_class_hetero": metrics.minority_class(cent),
                    "num_architectures": len(refs),
                    "wall_clock_seconds": float(sum(r.get("wall_clock_seconds", 0.0) for r in refs)),
                    "status": "complete",
                    "notes": ("element-wise mean of per-architecture centralized recall vectors "
                              "(class gap computed AFTER averaging, never averaged gaps)"),
                }
                config.dump_json(mean, hc.mean_path(method, dataset, seed))
                logger(f"=== mean ref gap={mean['cent_class_gap_hetero']:.4f} "
                       f"({time.time() - t0:.0f}s)")
                logger.close()


if __name__ == "__main__":
    main()
