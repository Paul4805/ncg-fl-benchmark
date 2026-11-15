"""Shared training / evaluation helpers."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    return torch.device("cuda") if (prefer_cuda and torch.cuda.is_available()) else torch.device("cpu")


def make_optimizer(model: nn.Module, lr: float, momentum: float, weight_decay: float):
    return torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=momentum, weight_decay=weight_decay,
    )


def cosine_scheduler(optimizer, t_max: int):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, t_max))


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, x.shape[0], batch_size):
        logits = model(x[start:start + batch_size].to(device))
        out.append(logits.argmax(dim=1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def accuracy(model: nn.Module, x, y, device, batch_size: int = 512) -> float:
    pred = predict(model, x, device, batch_size)
    return float((pred == np.asarray(y)).mean())


@torch.no_grad()
def eval_recall(model: nn.Module, x, y, num_classes: int, device) -> np.ndarray:
    from .metrics import per_class_recall

    return per_class_recall(np.asarray(y), predict(model, x, device), num_classes)


@torch.no_grad()
def eval_loss(model: nn.Module, x, y, device, batch_size: int = 512) -> float:
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    total = 0.0
    for start in range(0, x.shape[0], batch_size):
        xb = x[start:start + batch_size].to(device)
        yb = y[start:start + batch_size].to(device)
        total += float(crit(model(xb), yb).item())
    return total / max(1, x.shape[0])


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total, count = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * yb.shape[0]
        count += yb.shape[0]
    return total / max(1, count)


class Logger:
    def __init__(self, path: Path, echo: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a", encoding="utf-8")
        self.echo = echo
        self.t0 = time.time()

    def __call__(self, msg: str):
        line = f"[{time.time() - self.t0:8.1f}s][{time.strftime('%H:%M:%S')}] {msg}"
        self.fh.write(line + "\n")
        self.fh.flush()
        if self.echo:
            print(line, flush=True)

    def close(self):
        try:
            self.fh.close()
        except Exception:
            pass


def write_result(path: Path, payload: dict) -> None:
    config.dump_json(payload, path)


def load_run_partition(cfg: dict) -> dict:
    from . import partition as partition_mod

    meta = partition_mod.load_partition(
        cfg["partition_file"],
        dataset=cfg["dataset"],
        protocol=cfg["protocol"],
        alpha=cfg["alpha"],
        client_count=cfg["client_count"],
        seed=cfg["seed"],
    )
    problems = partition_mod.verify_partition(meta)
    if problems:
        raise ValueError(f"invalid partition {cfg['partition_file']}: {problems}")
    return meta


def write_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
