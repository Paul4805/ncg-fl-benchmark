"""Homogeneous CNN-2 centralized reference (the only centralized mode).

Protocol (locked):
  * full aggregated training set of the dataset
  * early stopping on a validation set that is NOT the test set:
      - BloodMNIST: the official val split (1,712), per Option 1'
      - FashionMNIST: stratified 10% of the 30k train subsample
  * SGD, momentum 0.9, weight decay 1e-4, batch 64, lr 0.01, cosine over max_epochs=250
  * early stopping patience 20, restore the best validation checkpoint
  * final per-class recall on the official test split (Blood 3,421 / Fashion 5,000)
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from core import config, datasets, metrics, models, train_utils


def train_centralized(dataset: str, seed: int, device=None, logger=print, protocol: dict | None = None) -> dict:
    t0 = time.time()
    device = device or train_utils.get_device()
    train_utils.set_seed(seed)
    protocol = dict(config.CENTRAL_TRAINING if protocol is None else protocol)
    spec = config.DATASETS[dataset]
    bundle = datasets.load_bundle(dataset, seed)

    y = bundle.train_y.numpy()
    if spec["centralized_val"] == "official_val":
        tr_idx = np.arange(len(y))
        val_ds = torch.utils.data.TensorDataset(bundle.val_x, bundle.val_y)
        val_size = len(bundle.val_y)
        val_source = "official_val"
        es_seed = seed
    else:
        tr_idx, va_idx = datasets.stratified_split(y, protocol["val_fraction"], seed)
        base = torch.utils.data.TensorDataset(bundle.train_x, bundle.train_y)
        val_ds = Subset(base, [int(i) for i in va_idx])
        val_size = len(va_idx)
        val_source = "stratified_train_10pct"
        es_seed = seed

    train_base = torch.utils.data.TensorDataset(bundle.train_x, bundle.train_y)
    train_subset = Subset(train_base, [int(i) for i in tr_idx])
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    model = models.build_model(config.BACKBONE, bundle.in_channels, bundle.num_classes).to(device)
    optimizer = train_utils.make_optimizer(model, protocol["lr"], protocol["momentum"],
                                          protocol["weight_decay"])
    scheduler = train_utils.cosine_scheduler(optimizer, protocol["max_epochs"])
    criterion = nn.CrossEntropyLoss()

    logger(
        f"[centralized {dataset} s{seed}] train={len(tr_idx)} val={val_size} "
        f"({val_source}) test={len(bundle.test_y)} max_epochs={protocol['max_epochs']} "
        f"batch={protocol['batch_size']} lr={protocol['lr']} patience={protocol['early_stopping_patience']}"
    )

    best_acc, best_epoch, best_state, patience = -1.0, -1, None, 0
    epochs_run = 0
    for epoch in range(protocol["max_epochs"]):
        epochs_run = epoch + 1
        gen = torch.Generator().manual_seed(int(es_seed) * 100000 + epoch)
        train_loader = DataLoader(
            train_subset, batch_size=protocol["batch_size"], shuffle=True,
            num_workers=0, drop_last=False, generator=gen,
        )
        loss = train_utils.train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
        val_acc = correct / max(1, total)

        if val_acc > best_acc + 1e-6:
            best_acc, best_epoch = val_acc, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            patience = 0
        else:
            patience += 1

        if epoch == 0 or (epoch + 1) % 10 == 0:
            logger(
                f"[centralized {dataset} s{seed}] epoch {epoch + 1}/{protocol['max_epochs']} "
                f"loss={loss:.4f} val_acc={val_acc:.4f} best={best_acc:.4f}@{best_epoch + 1} "
                f"lr={optimizer.param_groups[0]['lr']:.5f}"
            )
        if patience >= protocol["early_stopping_patience"]:
            logger(f"[centralized {dataset} s{seed}] early stop at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    recall = train_utils.eval_recall(model, bundle.test_x, bundle.test_y,
                                     bundle.num_classes, device)
    test_acc = train_utils.accuracy(model, bundle.test_x, bundle.test_y, device)

    result = {
        "run_id": config.central_key(dataset, seed),
        "kind": "centralized",
        "dataset": dataset,
        "seed": int(seed),
        "backbone": config.BACKBONE,
        "per_class_recall": [float(v) for v in recall],
        "class_gap": metrics.class_gap(recall),
        "minority_class": metrics.minority_class(recall),
        "test_accuracy": float(test_acc),
        "best_val_accuracy": float(best_acc),
        "best_epoch": int(best_epoch),
        "epochs_run": int(epochs_run),
        "max_epochs": int(protocol["max_epochs"]),
        "early_stopping_patience": int(protocol["early_stopping_patience"]),
        "lr": float(protocol["lr"]),
        "batch_size": int(protocol["batch_size"]),
        "momentum": float(protocol["momentum"]),
        "weight_decay": float(protocol["weight_decay"]),
        "val_source": val_source,
        "train_size": int(len(tr_idx)),
        "val_size": int(val_size),
        "test_size": int(len(bundle.test_y)),
        "num_classes": int(bundle.num_classes),
        "wall_clock_seconds": float(time.time() - t0),
        "status": "complete",
        "n_parameters": models.count_parameters(model),
        "subsample_file": bundle.subsample_file,
        "notes": f"centralized CNN-2 on the full aggregated train set; ES on {val_source}",
    }
    logger(
        f"[centralized {dataset} s{seed}] done: gap={result['class_gap']:.4f} "
        f"test_acc={test_acc:.4f} best_epoch={best_epoch + 1} ({result['wall_clock_seconds']:.0f}s)"
    )
    return result


def save_result(result: dict):
    config.dump_json(result, config.central_path(result["dataset"], result["seed"]))
