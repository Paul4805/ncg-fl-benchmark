"""Dataset loading with the locked splits.

BloodMNIST (Option 1'):
  official MedMNIST v2 split -> train 11,959 | val 1,712 | test 3,421.
  * FL: train only (no validation set; fixed R rounds, no early stopping).
  * Centralized: trained on all 11,959 train samples; early stopping on the
    official val split (1,712); scoring on the official test split (3,421).
  * The test split is never used for training, early stopping or selection.
  Counts are asserted on load; a mismatch aborts the run instead of proceeding.

FashionMNIST (R3):
  per-seed stratified subsample -> 30,000 train (3,000/class, drawn from the
  official 60k train pool) and 5,000 test (500/class, drawn from the official
  10k test pool).  Indices are drawn once per seed with default_rng(seed) and
  persisted to partitions/fashionmnist_subsample_seed{seed}.json.
  * FL: train only.
  * Centralized: stratified 10% of the 30k train subsample for early stopping.

No data augmentation is used anywhere (FL and centralized see identical inputs).
"""
from __future__ import annotations

import os
import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import config


# ---------------------------------------------------------------------------
# Raw loading + normalisation cache
# ---------------------------------------------------------------------------
def _ensure_bloodmnist_npz() -> Path:
    target = config.RAW_DIR / "bloodmnist.npz"
    if target.exists():
        return target
    home_cache = Path(os.path.expanduser("~")) / ".medmnist" / "bloodmnist.npz"
    if home_cache.exists():
        shutil.copyfile(home_cache, target)
        return target
    from medmnist import BloodMNIST  # downloads into RAW_DIR

    BloodMNIST(split="train", size=28, download=True, root=str(config.RAW_DIR))
    if not target.exists():
        raise RuntimeError("BloodMNIST download did not produce bloodmnist.npz")
    return target


def _normalize(imgs: np.ndarray, mean, std) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(imgs)).float().div_(255.0)
    if x.dim() == 3:                                  # (N, H, W) grayscale
        x = x.unsqueeze(1)
    elif x.dim() == 4 and x.shape[-1] in (1, 3):      # (N, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()
    m = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
    return (x - m) / s


def _load_bloodmnist():
    npz = _ensure_bloodmnist_npz()
    d = np.load(npz)
    out = {}
    for split in ("train", "val", "test"):
        out[f"{split}_x_raw"] = d[f"{split}_images"]
        out[f"{split}_y"] = np.asarray(d[f"{split}_labels"]).reshape(-1)
    spec = config.DATASETS["BloodMNIST"]
    expected = spec["official_split"]
    actual = {s: len(out[f"{s}_y"]) for s in ("train", "val", "test")}
    if actual != {"train": int(expected["train"]), "val": int(expected["val"]), "test": int(expected["test"])}:
        raise RuntimeError(
            f"BloodMNIST split mismatch: actual={actual} expected={expected}. "
            "Stopping instead of proceeding with an unexpected split (Option 1')."
        )
    for split in ("train", "val", "test"):
        out[f"{split}_x"] = _normalize(out.pop(f"{split}_x_raw"), spec["mean"], spec["std"])
        out[f"{split}_y"] = torch.from_numpy(out[f"{split}_y"]).long()
    return out


def _load_fashionmnist():
    import torchvision

    ds_train = torchvision.datasets.FashionMNIST(
        root=str(config.RAW_DIR), train=True, download=True
    )
    ds_test = torchvision.datasets.FashionMNIST(
        root=str(config.RAW_DIR), train=False, download=True
    )
    spec = config.DATASETS["FashionMNIST"]
    x_tr = _normalize(ds_train.data.numpy(), spec["mean"], spec["std"])
    x_te = _normalize(ds_test.data.numpy(), spec["mean"], spec["std"])
    y_tr = torch.from_numpy(ds_train.targets.numpy()).long()
    y_te = torch.from_numpy(ds_test.targets.numpy()).long()
    expected = spec["official_split"]
    if len(y_tr) != int(expected["train"]) or len(y_te) != int(expected["test"]):
        raise RuntimeError(
            f"FashionMNIST split mismatch: train={len(y_tr)} test={len(y_te)} "
            f"expected={expected}"
        )
    return {"train_x": x_tr, "train_y": y_tr, "test_x": x_te, "test_y": y_te}


def load_full(dataset: str):
    """Normalised full tensors, cached on disk."""
    cache = config.CACHE_DIR / f"{dataset}.pt"
    if cache.exists():
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        blob["_cache"] = str(cache)
        return blob
    blob = _load_bloodmnist() if dataset == "BloodMNIST" else _load_fashionmnist()
    torch.save({k: v for k, v in blob.items()}, cache)
    blob["_cache"] = str(cache)
    return blob


# ---------------------------------------------------------------------------
# FashionMNIST per-seed subsample (R3)
# ---------------------------------------------------------------------------
def build_subsample(seed: int) -> dict:
    full = load_full("FashionMNIST")
    y_tr = full["train_y"].numpy()
    y_te = full["test_y"].numpy()
    spec = config.DATASETS["FashionMNIST"]["subsample"]
    rng = np.random.default_rng(int(seed))
    train_idx, test_idx = [], []
    for c in range(config.DATASETS["FashionMNIST"]["num_classes"]):
        tr_c = np.where(y_tr == c)[0]
        rng.shuffle(tr_c)
        train_idx.append(tr_c[: int(spec["train_per_class"])])
        te_c = np.where(y_te == c)[0]
        rng.shuffle(te_c)
        test_idx.append(te_c[: int(spec["test_per_class"])])
    train_idx = np.sort(np.concatenate(train_idx)).astype(np.int64)
    test_idx = np.sort(np.concatenate(test_idx)).astype(np.int64)
    meta = {
        "dataset": "FashionMNIST",
        "seed": int(seed),
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "train_class_counts": np.bincount(y_tr[train_idx], minlength=10).tolist(),
        "test_class_counts": np.bincount(y_te[test_idx], minlength=10).tolist(),
        "source": "official 60k train pool (3,000/class) and official 10k test pool (500/class)",
        "rng": "numpy.random.default_rng(seed)",
    }
    if meta["train_size"] != int(spec["train"]) or meta["test_size"] != int(spec["test"]):
        raise RuntimeError(f"subsample size mismatch: {meta['train_size']}/{meta['test_size']}")
    if set(meta["train_class_counts"]) != {int(spec["train_per_class"])}:
        raise RuntimeError(f"subsample train class counts: {meta['train_class_counts']}")
    if set(meta["test_class_counts"]) != {int(spec["test_per_class"])}:
        raise RuntimeError(f"subsample test class counts: {meta['test_class_counts']}")
    return meta


def get_subsample(seed: int) -> dict:
    path = config.subsample_path(int(seed))
    if path.exists():
        meta = config.load_json(path)
        if meta.get("seed") != int(seed):
            raise RuntimeError(f"subsample file {path} has wrong seed")
        return meta
    meta = build_subsample(int(seed))
    config.dump_json(meta, path)
    return meta


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------
@dataclass
class Bundle:
    dataset: str
    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    val_x: torch.Tensor | None = None
    val_y: torch.Tensor | None = None
    num_classes: int = 10
    in_channels: int = 1
    subsample_file: str | None = None
    subsample_meta: dict = field(default_factory=dict)
    split_meta: dict = field(default_factory=dict)


def load_bundle(dataset: str, seed: int) -> Bundle:
    spec = config.DATASETS[dataset]
    full = load_full(dataset)
    if dataset == "BloodMNIST":
        bundle = Bundle(
            dataset=dataset,
            train_x=full["train_x"], train_y=full["train_y"],
            test_x=full["test_x"], test_y=full["test_y"],
            val_x=full["val_x"], val_y=full["val_y"],
            num_classes=int(spec["num_classes"]), in_channels=int(spec["in_channels"]),
            split_meta={
                "train": int(len(full["train_y"])),
                "val": int(len(full["val_y"])),
                "test": int(len(full["test_y"])),
                "centralized_val_source": "official_val",
            },
        )
        return bundle

    meta = get_subsample(int(seed))
    tr = torch.as_tensor(meta["train_indices"], dtype=torch.long)
    te = torch.as_tensor(meta["test_indices"], dtype=torch.long)
    bundle = Bundle(
        dataset=dataset,
        train_x=full["train_x"][tr], train_y=full["train_y"][tr],
        test_x=full["test_x"][te], test_y=full["test_y"][te],
        val_x=None, val_y=None,
        num_classes=int(spec["num_classes"]), in_channels=int(spec["in_channels"]),
        subsample_file=str(config.subsample_path(int(seed))),
        subsample_meta={k: meta[k] for k in ("seed", "train_size", "test_size",
                                             "train_class_counts", "test_class_counts")},
        split_meta={
            "train": int(meta["train_size"]),
            "test": int(meta["test_size"]),
            "centralized_val_source": "stratified_train_10pct",
        },
    )
    return bundle


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def stratified_split(y, fraction: float, seed: int):
    """Stratified (train_idx, val_idx) over the given label vector."""
    y = np.asarray(y).ravel()
    rng = np.random.default_rng(int(seed))
    tr_parts, va_parts = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * fraction))
        va_parts.append(idx[:n_val])
        tr_parts.append(idx[n_val:])
    return np.sort(np.concatenate(tr_parts)), np.sort(np.concatenate(va_parts))


def class_counts(indices, y) -> list[int]:
    y = np.asarray(y)
    return np.bincount(y[np.asarray(indices, dtype=np.int64)], minlength=int(y.max()) + 1).tolist()
