"""FedAvg baseline (standard Federated Averaging).

Each round every client starts from the broadcast global model, runs E local
epochs with SGD, and the server forms the sample-count weighted average of the
returned weights (standard FedAvg definition; the method-specific FedFV /
FedAPEN / PHP-FL implementations use their own official aggregation rules).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from core import models, train_utils
from .common import RunContext, make_result, record_eval_round


def run(cfg: dict, logger=print, device=None, eval_rounds=None) -> dict:
    t0 = time.time()
    ctx = RunContext(cfg, logger=logger, device=device)
    tr = cfg["training"]
    R, E = int(tr["rounds"]), int(tr["local_epochs"])

    global_model = ctx.build_model()
    local_model = ctx.build_model()
    criterion = nn.CrossEntropyLoss()
    curve = {} if eval_rounds else None

    for r in range(R):
        lr = ctx.lr_for_round(r)
        selected = ctx.select_clients(r)
        states, weights, losses = [], [], []
        for cid in selected:
            idx = ctx.client_indices[cid]
            local_model.load_state_dict(global_model.state_dict())
            opt = train_utils.make_optimizer(local_model, lr, tr["momentum"], tr["weight_decay"])
            loader = ctx.loader(idx, shuffle=True, seed=cfg["seed"] * 100000 + r * 1000 + cid)
            losses.append(train_utils.train_epoch(local_model, loader, opt, criterion, ctx.device))
            states.append(models.clone_state(local_model))
            weights.append(float(len(idx)))
        global_model.load_state_dict(models.average_states(states, weights))

        record_eval_round(curve, r, eval_rounds, ctx.eval_accuracy(global_model))
        if (r + 1) % 10 == 0 or r == 0:
            logger(f"[FedAvg {cfg['run_id']}] round {r + 1}/{R} lr={lr:.5f} "
                   f"clients={len(selected)} loss={np.mean(losses):.4f}")

    recall = ctx.eval_recall(global_model)
    acc = ctx.eval_accuracy(global_model)
    logger(f"[FedAvg {cfg['run_id']}] final gap={recall.max() - recall.min():.4f} acc={acc:.4f}")
    return make_result(
        ctx, recall, t0,
        notes=(f"sample-weighted FedAvg; {R} rounds x {E} local epochs; full participation; "
               f"CNN-2; fixed schedule, no early stopping"),
        per_client_accuracy=None,
        extra={"test_accuracy": acc, "eval_curve": curve},
    )
