"""Generate the per-seed FashionMNIST subsample and all 18 client partitions.

    python -m scripts.generate_partitions [--force]

Every partition is verified (each training sample assigned exactly once, no
overlap, no empty clients) before it is accepted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config, datasets, partition as partition_mod  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    print("[partitions] FashionMNIST per-seed subsamples")
    for seed in config.SEEDS:
        path = config.subsample_path(seed)
        if path.exists() and not args.force:
            meta = config.load_json(path)
            status = "reused"
        else:
            meta = datasets.build_subsample(seed)
            config.dump_json(meta, path)
            status = "created"
        print(f"  {status:7s} {path.name}  train={meta['train_size']} "
              f"test={meta['test_size']} train/class={meta['train_class_counts'][0]} "
              f"test/class={meta['test_class_counts'][0]}")

    req = config.required_partitions()
    print(f"[partitions] {len(req)} unique partitions required")
    index = []
    for (dataset, protocol, alpha, n_clients, seed), path_str in sorted(req.items(), key=lambda kv: str(kv[1])):
        path = Path(path_str)
        if path.exists() and not args.force:
            meta = config.load_json(path)
            status = "reused"
        else:
            meta = partition_mod.build_partition(dataset, protocol, alpha, n_clients, seed)
            path = partition_mod.save_partition(meta)
            status = "created"
        problems = partition_mod.verify_partition(meta)
        if problems:
            raise SystemExit(f"INVALID partition {path}: {problems}")
        stats = partition_mod.partition_stats(meta)
        index.append({"file": str(path), "dataset": dataset, "protocol": protocol,
                      "alpha": alpha, "client_count": n_clients, "seed": seed,
                      "status": status, **stats,
                      "subsample_file": meta.get("subsample_file"),
                      "total_class_counts": meta["total_class_counts"]})
        print(f"  {status:7s} {path.name}  sizes[{stats['client_sizes_min']},"
              f"{stats['client_sizes_max']}] classes/client "
              f"{stats['classes_per_client_mean']:.2f} (min {stats['classes_per_client_min']}) "
              f"repairs={meta['repairs']}")

    config.dump_json({"partitions": index, "subsamples": [
        str(config.subsample_path(s)) for s in config.SEEDS
    ]}, config.PARTITION_DIR / "index.json")
    print(f"[partitions] wrote {config.PARTITION_DIR / 'index.json'}")


if __name__ == "__main__":
    main()
