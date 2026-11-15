"""Shared FL plumbing."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..core import config, datasets, metrics, models, train_utils


class RunContext:
    def __init__(self, cfg: dict, logger=print, device=None):
        self.cfg = cfg
        self.logger = logger
        self.device = device or train_utils.get_device()
        train_utils.set_seed(cfg["seed"])
        self.bundle = datasets.load_bundle(cfg["dataset"], cfg["seed"])
        self.train_x = self.bundle.train_x
        self.train_y = self.bundle.train_y
        self.test_x = self.bundle.test_x
        self.test_y = self.bundle.test_y
        self.num_classes = self.bundle.num_classes
        self.in_channels = self.bundle.in_channels
        self.partition = train_utils.load_run_partition(cfg)
        self.client_indices = [np.asarray(c, dtype=np.int64) for c in self.partition["clients"]]
        self.num_clients = len(self.client_indices)
        if self.num_clients != cfg["client_count"]:
            raise ValueError(f"partition has {self.num_clients} clients, config says {cfg['client_count']}")

    # -- models -------------------------------------------------------------
    def build_model(self, spec: dict | None = None) -> torch.nn.Module:
        spec = spec or {"name": config.BACKBONE, "params": {}}
        return models.build_model(spec["name"], self.in_channels, self.num_classes,
                                  **(spec.get("params") or {})).to(self.device)

    def build_client_model(self, cid: int) -> torch.nn.Module:
        specs = self.cfg.get("client_model_specs")
        return self.build_model(specs[cid] if specs else None)

    def build_global_model(self) -> torch.nn.Module:
        return self.build_model(self.cfg.get("global_model_spec"))

    # -- optimisation -------------------------------------------------------
    def lr_for_round(self, r: int) -> float:
        base = float(self.cfg["training"]["lr"])
        if self.cfg["training"]["lr_schedule"] == "cosine":
            R = int(self.cfg["training"]["rounds"])
            return base * 0.5 * (1.0 + math.cos(math.pi * r / max(1, R)))
        return base

    def loader(self, indices, shuffle=True, seed=0, drop_last=False):
        subset = Subset(torch.utils.data.TensorDataset(self.train_x, self.train_y),
                        [int(i) for i in indices])
        gen = torch.Generator().manual_seed(int(seed)) if shuffle else None
        return DataLoader(subset, batch_size=int(self.cfg["training"]["batch_size"]),
                          shuffle=shuffle, num_workers=0, drop_last=drop_last, generator=gen)

    def select_clients(self, r: int):
        return list(range(self.num_clients))   # full participation in B1-B3

    # -- evaluation ---------------------------------------------------------
    def eval_recall(self, model) -> np.ndarray:
        return train_utils.eval_recall(model, self.test_x, self.test_y, self.num_classes, self.device)

    def eval_accuracy(self, model) -> float:
        return train_utils.accuracy(model, self.test_x, self.test_y, self.device)

    def eval_deployed(self, private, shared, weights):
        """Per-client recall + accuracy of  w*private + (1-w)*shared  on the global test set."""
        y_true = self.test_y.numpy()
        recalls, accs = [], []
        for cid in range(self.num_clients):
            w = float(weights[cid])
            private[cid].eval()
            shared[cid].eval()
            preds = []
            with torch.no_grad():
                for start in range(0, self.test_x.shape[0], 512):
                    xb = self.test_x[start:start + 512].to(self.device)
                    logits = w * private[cid](xb) + (1.0 - w) * shared[cid](xb)
                    preds.append(logits.argmax(dim=1).cpu())
            preds = torch.cat(preds).numpy()
            recalls.append(metrics.per_class_recall(y_true, preds, self.num_classes))
            accs.append(float((preds == y_true).mean()))
        return recalls, accs


def make_result(ctx: RunContext, fl_recall, t0: float, notes: str,
                per_client_recall=None, per_client_accuracy=None,
                extra: dict | None = None) -> dict:
    cfg = ctx.cfg
    rec = np.asarray(fl_recall, dtype=np.float64)
    payload = {
        "run_id": cfg["run_id"],
        "kind": "fl",
        "setting": cfg["setting"],
        "method": cfg["method"],
        "dataset": cfg["dataset"],
        "seed": cfg["seed"],
        "alpha": cfg["alpha"],
        "client_count": cfg["client_count"],
        "backbone": cfg["backbone"],
        "partition_file": cfg["partition_file"],
        "fashionmnist_subsample_file": cfg["fashionmnist_subsample_file"],
        "per_class_recall": [float(v) for v in rec],
        "fl_class_gap": metrics.class_gap(rec),
        "class_gap": metrics.class_gap(rec),
        "minority_class": metrics.minority_class(rec),
        "per_client_recall": (
            [[float(v) for v in r] for r in per_client_recall] if per_client_recall is not None else None
        ),
        "per_client_accuracy": (
            [float(v) for v in per_client_accuracy] if per_client_accuracy is not None else None
        ),
        "test_accuracy": (
            float(np.mean(per_client_accuracy)) if per_client_accuracy is not None else None
        ),
        "rounds": int(cfg["training"]["rounds"]),
        "local_epochs": int(cfg["training"]["local_epochs"]),
        "wall_clock_seconds": float(time.time() - t0),
        "status": "complete",
        "centralized_ref_key": cfg["centralized_ref_key"],
        "notes": notes,
        "config": cfg,
    }
    if cfg.get("run_role"):
        payload["run_role"] = cfg["run_role"]
    if extra:
        payload.update(extra)
    return payload


def record_eval_round(curve: dict | None, r: int, eval_rounds, acc: float):
    if curve is not None and eval_rounds and (r + 1) in set(int(x) for x in eval_rounds):
        curve[str(r + 1)] = float(acc)


# ---------------------------------------------------------------------------
# Small shared split policies (FedAPEN adaptability / PHP-FL validation)
# ---------------------------------------------------------------------------
def _stratified_selection(indices, labels, size, rng):
    """Pick `size` indices, balanced across classes; leaves >=1 sample per class."""
    idx = np.asarray(indices)
    lab = np.asarray(labels)
    classes = [c for c in np.unique(lab) if int((lab == c).sum()) >= 2]
    if not classes or size <= 0:
        chosen = rng.choice(idx, size=min(size, max(1, len(idx) - 1)), replace=False)
        return np.sort(chosen)

    capacity = {c: int((lab == c).sum()) - 1 for c in classes}
    alloc = {c: 0 for c in classes}
    remaining = int(size)
    # one per class first, largest classes first on ties
    for c in sorted(classes, key=lambda c: (-capacity[c], c)):
        if remaining == 0:
            break
        alloc[c] = 1
        remaining -= 1
    # distribute the rest proportionally to remaining capacity
    while remaining > 0:
        open_c = [c for c in classes if alloc[c] < capacity[c]]
        if not open_c:
            break
        total_cap = sum(capacity[c] for c in open_c)
        progressed = False
        for c in open_c:
            if remaining == 0:
                break
            share = max(1, int(round(remaining * capacity[c] / max(1, total_cap))))
            add = min(share, capacity[c] - alloc[c], remaining)
            if add > 0:
                alloc[c] += add
                remaining -= add
                progressed = True
        if not progressed:
            break
    if remaining > 0:
        # any spare capacity anywhere
        for c in classes:
            add = min(capacity[c] - alloc[c], remaining)
            if add > 0:
                alloc[c] += add
                remaining -= add
            if remaining == 0:
                break

    picks = []
    for c in classes:
        if alloc[c] <= 0:
            continue
        c_idx = idx[lab == c]
        rng.shuffle(c_idx)
        picks.append(c_idx[: alloc[c]])
    chosen = np.concatenate(picks) if picks else np.array([], dtype=idx.dtype)
    if len(chosen) < size:
        pool = np.setdiff1d(idx, chosen, assume_unique=False)
        extra = rng.choice(pool, size=min(size - len(chosen), max(0, len(pool) - 1)), replace=False)
        chosen = np.concatenate([chosen, extra])
    return np.sort(chosen[:size])


def auxiliary_split(indices, labels, seed: int, ratio: float, min_size: int,
                    max_size: int, divisor: int):
    """Train / auxiliary split with the correction-2 floor rule.

    nominal = floor(ratio * n); if nominal < min_size the auxiliary size is
    forced to min(max_size, max(min_size, n // divisor)) so it is never empty or
    degenerate.  Clients with fewer than 4 samples get no auxiliary set (the
    caller falls back to a fixed ensemble weight of 0.5).
    """
    idx = np.asarray(indices, dtype=np.int64)
    n = int(len(idx))
    info = {"n": n, "nominal": int(np.floor(ratio * n)), "forced": False,
            "fallback": False, "aux_size": 0}
    if n < 4:
        info["fallback"] = True
        return idx, np.array([], dtype=np.int64), info
    nominal = int(np.floor(ratio * n))
    if nominal < int(min_size):
        aux_size = min(int(max_size), max(int(min_size), n // int(divisor)))
        info["forced"] = True
    else:
        aux_size = nominal
    aux_size = int(max(0, min(aux_size, n - 1)))
    if aux_size < int(min_size):
        info["fallback"] = True
        return idx, np.array([], dtype=np.int64), info

    rng = np.random.default_rng(int(seed))
    aux = _stratified_selection(idx, np.asarray(labels)[idx], aux_size, rng)
    train = np.setdiff1d(idx, aux, assume_unique=False)
    info["aux_size"] = int(len(aux))
    return train, aux, info
