"""FedFV -- Federated Learning with Fair Averaging (Wang et al., IJCAI 2021).

Faithful re-implementation of the official easyFL aggregation rule
(WwZzz/easyFL, flgo/algorithm/fedfv.py, alpha=0.1, tau=1):

  * deltas g_i = w_global - w_local_i, plus each client's training loss of the
    broadcast global model *before* local training;
  * clients sorted by that loss; the ceil((n-1)*(1-alpha)) highest-loss clients
    keep their original direction;
  * internal conflicts: project away any delta with a negative inner product;
  * average the projected deltas, then project out external conflicts with
    deltas received in the previous tau rounds;
  * rescale to the norm of the plain mean delta and subtract from the global model.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn

from core import models, train_utils
from .common import RunContext, make_result, record_eval_round


def fedfv_aggregate(grads, losses, history, last_round, current_round, alpha, tau, num_clients):
    order = sorted(range(len(grads)), key=lambda i: losses[i])
    keep_original = order[math.ceil((len(order) - 1) * (1 - alpha)):] if alpha > 0 else []

    order_grads = [g.clone() for g in grads]
    for i in range(len(order_grads)):
        if i in keep_original:
            continue
        for j in order:
            if j == i:
                continue
            dot = float(torch.dot(grads[j], order_grads[i]).item())
            if dot < 0:
                denom = float(torch.dot(grads[j], grads[j]).item())
                if denom > 0:
                    order_grads[i] = order_grads[i] - grads[j] * (dot / denom)

    gt = torch.stack(order_grads, dim=0).mean(dim=0)
    if current_round >= tau:
        for k in range(tau - 1, -1, -1):
            gcs = [history[c] for c in range(num_clients)
                   if history[c] is not None and last_round[c] == current_round - k
                   and float(torch.dot(gt, history[c]).item()) < 0]
            if gcs:
                g_con = torch.stack(gcs, dim=0).sum(dim=0)
                dot = float(torch.dot(gt, g_con).item())
                if dot < 0:
                    denom = float(torch.dot(g_con, g_con).item())
                    if denom > 0:
                        gt = gt - g_con * (dot / denom)

    gnorm = float(torch.stack(grads, dim=0).mean(dim=0).norm().item())
    gt_norm = float(gt.norm().item())
    if gt_norm > 0:
        gt = gt / gt_norm * gnorm
    return gt


def run(cfg: dict, logger=print, device=None, eval_rounds=None) -> dict:
    t0 = time.time()
    ctx = RunContext(cfg, logger=logger, device=device)
    tr = cfg["training"]
    R, E = int(tr["rounds"]), int(tr["local_epochs"])
    alpha = float(cfg["hparams"]["fedfv_alpha"])
    tau = int(cfg["hparams"]["fedfv_tau"])
    n_clients = ctx.num_clients

    global_model = ctx.build_model()
    local_model = ctx.build_model()
    criterion = nn.CrossEntropyLoss()
    base_state = models.clone_state(global_model)
    global_flat = models.flatten_state(base_state)
    history = [None] * n_clients
    last_round = [-1] * n_clients
    curve = {} if eval_rounds else None

    for r in range(R):
        lr = ctx.lr_for_round(r)
        selected = ctx.select_clients(r)
        deltas, losses = [], []
        for cid in selected:
            idx = ctx.client_indices[cid]
            local_model.load_state_dict(base_state)
            loss_before = train_utils.eval_loss(local_model, ctx.train_x[idx], ctx.train_y[idx], ctx.device)
            opt = train_utils.make_optimizer(local_model, lr, tr["momentum"], tr["weight_decay"])
            loader = ctx.loader(idx, shuffle=True, seed=cfg["seed"] * 100000 + r * 1000 + cid)
            train_utils.train_epoch(local_model, loader, opt, criterion, ctx.device)
            delta = global_flat - models.flatten_state(local_model.state_dict())
            deltas.append(delta)
            losses.append(loss_before)
            history[cid] = delta
            last_round[cid] = r

        gt = fedfv_aggregate(deltas, losses, history, last_round, r, alpha, tau, n_clients)
        global_flat = global_flat - gt
        base_state = models.unflatten_into_state(base_state, global_flat)
        global_model.load_state_dict(base_state)

        record_eval_round(curve, r, eval_rounds, ctx.eval_accuracy(global_model))
        if (r + 1) % 10 == 0 or r == 0:
            logger(f"[FedFV {cfg['run_id']}] round {r + 1}/{R} lr={lr:.5f} clients={len(selected)} "
                   f"mean_loss_before={np.mean(losses):.4f} ||gt||={float(gt.norm()):.4f}")

    recall = ctx.eval_recall(global_model)
    acc = ctx.eval_accuracy(global_model)
    logger(f"[FedFV {cfg['run_id']}] final gap={recall.max() - recall.min():.4f} acc={acc:.4f}")
    return make_result(
        ctx, recall, t0,
        notes=(f"easyFL FedFV aggregation (alpha={alpha}, tau={tau}, uniform mean of projected "
               f"deltas, loss-sorted); {R} rounds x {E} local epochs; full participation; CNN-2"),
        extra={"test_accuracy": acc, "fedfv_alpha": alpha, "fedfv_tau": tau, "eval_curve": curve},
    )
