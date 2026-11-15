"""Deterministic client partitioning (IID and Dirichlet label skew).

A partition is fully determined by (dataset, protocol, alpha, n_clients, seed)
and saved as JSON with the exact client index lists.  For FashionMNIST the
indices are positions inside that seed's 30k subsample.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config


def iid_partition(num_samples: int, num_clients: int, seed: int):
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(num_samples)
    return [np.sort(p) for p in np.array_split(perm, num_clients)]


def dirichlet_partition(y, num_clients: int, alpha: float, num_classes: int, seed: int):
    rng = np.random.default_rng(int(seed))
    y = np.asarray(y).ravel()
    parts = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        props = rng.dirichlet(np.repeat(float(alpha), num_clients))
        bounds = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
        for cid, chunk in enumerate(np.split(idx_c, bounds)):
            parts[cid].extend(chunk.tolist())

    repairs = 0
    for cid in range(num_clients):
        while len(parts[cid]) == 0:
            donor = max(range(num_clients), key=lambda j: (len(parts[j]), -j))
            if len(parts[donor]) <= 1:
                break
            parts[cid].append(parts[donor].pop())
            repairs += 1
    out = [np.sort(np.array(p, dtype=np.int64)) for p in parts]
    return out, repairs


def build_partition(dataset: str, protocol: str, alpha, client_count: int, seed: int) -> dict:
    from . import datasets

    bundle = datasets.load_bundle(dataset, seed)
    y = bundle.train_y.numpy()
    n = len(y)
    if protocol == "iid":
        parts, repairs = iid_partition(n, client_count, seed), 0
    elif protocol == "dirichlet":
        parts, repairs = dirichlet_partition(
            y, client_count, float(alpha), bundle.num_classes, seed
        )
    else:
        raise ValueError(protocol)

    return {
        "dataset": dataset,
        "protocol": protocol,
        "alpha": None if protocol == "iid" else float(alpha),
        "client_count": client_count,
        "seed": int(seed),
        "num_samples": n,
        "num_classes": bundle.num_classes,
        "repairs": repairs,
        "subsample_file": bundle.subsample_file,
        "clients": [p.tolist() for p in parts],
        "client_sizes": [int(len(p)) for p in parts],
        "client_class_counts": [
            np.bincount(y[p], minlength=bundle.num_classes).tolist() for p in parts
        ],
        "total_class_counts": np.bincount(y, minlength=bundle.num_classes).tolist(),
    }


def save_partition(meta: dict) -> Path:
    path = config.partition_path(meta["dataset"], meta["protocol"], meta["alpha"],
                                 meta["client_count"], meta["seed"])
    config.dump_json(meta, path)
    return path


def load_partition(path, dataset=None, protocol=None, alpha=None, client_count=None, seed=None):
    meta = config.load_json(Path(path))
    for k, v in {"dataset": dataset, "protocol": protocol, "alpha": alpha,
                 "client_count": client_count, "seed": seed}.items():
        if v is not None and meta[k] != v:
            raise ValueError(f"partition {path} mismatch on {k}: file={meta[k]!r} expected={v!r}")
    return meta


def verify_partition(meta: dict) -> list[str]:
    problems = []
    parts = [np.asarray(p, dtype=np.int64) for p in meta["clients"]]
    all_idx = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
    n = int(meta["num_samples"])
    if len(all_idx) != n:
        problems.append(f"assigned {len(all_idx)} of {n} samples")
    if len(np.unique(all_idx)) != len(all_idx):
        problems.append("overlapping samples")
    if set(all_idx.tolist()) != set(range(n)):
        problems.append("assigned indices != range(num_samples)")
    for cid, p in enumerate(parts):
        if len(p) == 0:
            problems.append(f"client {cid} empty")
        if len(np.unique(p)) != len(p):
            problems.append(f"client {cid} has duplicates")
    return problems


def partition_stats(meta: dict) -> dict:
    sizes = np.asarray(meta["client_sizes"])
    present = (np.asarray(meta["client_class_counts"]) > 0).sum(axis=1)
    return {
        "client_sizes_min": int(sizes.min()),
        "client_sizes_max": int(sizes.max()),
        "client_sizes_mean": float(sizes.mean()),
        "classes_per_client_min": int(present.min()),
        "classes_per_client_max": int(present.max()),
        "classes_per_client_mean": float(present.mean()),
    }
