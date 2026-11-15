"""FedAPEN -- Personalized Cross-silo FL with Adaptability to Statistical
Heterogeneity (Qin et al., KDD 2023), official code zhenqincn/FedAPEN.

Faithful minimal re-implementation of ``learned_adaptive_training``:
  * each client owns a private model and a copy of the shared model;
  * per local step, the private model minimises
        CE(private) + KL(private || shared) + CE(w*private + (1-w)*shared)
    and the shared model symmetrically (all cross terms detached);
  * the server uniformly averages the shared models and broadcasts the average;
  * each client learns a scalar ensemble weight on its own 5% adaptability set
    (10 SGD steps on the scalar, clipped to [0,1]);
  * the deployed personalised model is  w*private + (1-w)*shared.

Correction-2 guard (locked): if floor(0.05*n) < 2 the adaptability set is forced
to min(5, max(2, n//4)) samples (stratified where possible) so it can never be
empty or degenerate; clients with n < 4 skip weight learning and keep w = 0.5.
Per-client sizes and fallbacks are recorded in the run JSON.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from core import metrics, models, train_utils
from .common import RunContext, auxiliary_split, make_result, record_eval_round


def _learn_weight(ctx, private, shared, w, opt_w, loader, criterion, steps):
    private.eval()
    shared.eval()
    for _ in range(steps):
        for xb, yb in loader:
            opt_w.zero_grad(set_to_none=True)
            xb, yb = xb.to(ctx.device), yb.to(ctx.device)
            with torch.no_grad():
                out_p, out_s = private(xb), shared(xb)
            loss = criterion(w * out_p + (1.0 - w) * out_s, yb)
            loss.backward()
            opt_w.step()
            with torch.no_grad():
                w.clamp_(0.0, 1.0)
    return float(w.detach().item())


def run(cfg: dict, logger=print, device=None, eval_rounds=None) -> dict:
    t0 = time.time()
    ctx = RunContext(cfg, logger=logger, device=device)
    tr, hp = cfg["training"], cfg["hparams"]
    R, E = int(tr["rounds"]), int(tr["local_epochs"])
    n_clients = ctx.num_clients
    clip_norm = float(hp.get("grad_clip_norm", 0.0))
    criterion = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")
    log_softmax, softmax = nn.LogSoftmax(dim=1), nn.Softmax(dim=1)

    shared_global = ctx.build_global_model()
    private = {cid: ctx.build_client_model(cid) for cid in range(n_clients)}
    shared = {cid: ctx.build_global_model() for cid in range(n_clients)}
    for cid in range(n_clients):
        shared[cid].load_state_dict(shared_global.state_dict())

    weights = {cid: torch.tensor([0.5], requires_grad=True, device=ctx.device) for cid in range(n_clients)}
    opt_weights = {cid: torch.optim.SGD([weights[cid]], lr=float(hp["private_weight_lr"]))
                   for cid in range(n_clients)}
    staged = {cid: [] for cid in range(n_clients)}

    labels = ctx.train_y.numpy()
    splits, adapt_sizes, fallback_clients, forced_clients = {}, [], [], []
    for cid in range(n_clients):
        tr_idx, aux_idx, info = auxiliary_split(
            ctx.client_indices[cid], labels, seed=cfg["seed"] * 31 + cid,
            ratio=float(hp["adapt_ratio"]), min_size=int(hp["min_adapt"]),
            max_size=int(hp["max_adapt"]), divisor=int(hp["adapt_floor_divisor"]),
        )
        splits[cid] = (tr_idx, aux_idx)
        adapt_sizes.append(int(info["aux_size"]))
        if info["fallback"]:
            fallback_clients.append(cid)
        if info["forced"]:
            forced_clients.append(cid)

    logger(f"[FedAPEN {cfg['run_id']}] adapt sizes={adapt_sizes} "
           f"forced_rule={forced_clients} fallback={fallback_clients}")

    curve = {} if eval_rounds else None
    for r in range(R):
        lr = ctx.lr_for_round(r)
        selected = ctx.select_clients(r)
        losses = []
        for cid in selected:
            tr_idx, _ = splits[cid]
            w_private = staged[cid][-1] if staged[cid] else 0.5
            opt_p = train_utils.make_optimizer(private[cid], lr, tr["momentum"], tr["weight_decay"])
            opt_s = train_utils.make_optimizer(shared[cid], lr, tr["momentum"], tr["weight_decay"])
            private[cid].train()
            shared[cid].train()
            loader = ctx.loader(tr_idx, shuffle=True, seed=cfg["seed"] * 100000 + r * 1000 + cid)
            for _ in range(E):
                for xb, yb in loader:
                    xb, yb = xb.to(ctx.device), yb.to(ctx.device)
                    opt_p.zero_grad(set_to_none=True)
                    opt_s.zero_grad(set_to_none=True)
                    out_p = private[cid](xb)
                    out_s = shared[cid](xb)
                    ens_p = w_private * out_p + (1.0 - w_private) * out_s.detach()
                    ens_s = w_private * out_p.detach() + (1.0 - w_private) * out_s
                    loss_p = (criterion(out_p, yb)
                              + kl(log_softmax(out_p), softmax(out_s.detach()))
                              + criterion(ens_p, yb))
                    loss_s = (criterion(out_s, yb)
                              + kl(log_softmax(out_s), softmax(out_p.detach()))
                              + criterion(ens_s, yb))
                    loss_p.backward()
                    loss_s.backward()
                    if not (torch.isfinite(loss_p).all() and torch.isfinite(loss_s).all()):
                        raise FloatingPointError(
                            f"{cfg['run_id']}: non-finite loss at round {r} client {cid}"
                        )
                    if clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(private[cid].parameters(), clip_norm)
                        torch.nn.utils.clip_grad_norm_(shared[cid].parameters(), clip_norm)
                    opt_p.step()
                    opt_s.step()
                    losses.append(float(loss_p.item()))

        avg = models.average_states([shared[cid].state_dict() for cid in selected])
        shared_global.load_state_dict(avg)
        for cid in range(n_clients):
            shared[cid].load_state_dict(avg)

        for cid in range(n_clients):
            _, aux_idx = splits[cid]
            if len(aux_idx) < 2:
                staged[cid].append(0.5)          # uniform fallback, no weight learning
                continue
            vds = Subset(torch.utils.data.TensorDataset(ctx.train_x, ctx.train_y),
                         [int(i) for i in aux_idx])
            vloader = DataLoader(
                vds, batch_size=min(int(tr["batch_size"]), len(aux_idx)), shuffle=True,
                num_workers=0, drop_last=False,
                generator=torch.Generator().manual_seed(cfg["seed"] * 7 + cid * 131 + r),
            )
            staged[cid].append(_learn_weight(ctx, private[cid], shared[cid], weights[cid],
                                             opt_weights[cid], vloader, criterion,
                                             int(hp["weight_steps"])))

        if eval_rounds and (r + 1) in set(int(x) for x in eval_rounds):
            _, accs = ctx.eval_deployed(private, shared, [staged[c][-1] for c in range(n_clients)])
            record_eval_round(curve, r, eval_rounds, float(np.mean(accs)))

        if (r + 1) % 10 == 0 or r == 0:
            logger(f"[FedAPEN {cfg['run_id']}] round {r + 1}/{R} lr={lr:.5f} "
                   f"loss={np.mean(losses):.4f} "
                   f"w_private_mean={np.mean([staged[c][-1] for c in range(n_clients)]):.3f}")

    recalled = [staged[c][-1] for c in range(n_clients)]
    recalls, accs = ctx.eval_deployed(private, shared, recalled)
    fl_recall = metrics.average_recall_vectors(recalls)
    logger(f"[FedAPEN {cfg['run_id']}] final gap={metrics.class_gap(fl_recall):.4f} "
           f"acc={np.mean(accs):.4f}")

    return make_result(
        ctx, fl_recall, t0,
        notes=(f"FedAPEN personalised ensembles w*private+(1-w)*shared; per-client per-class "
               f"recalls averaged across {n_clients} clients BEFORE the gap; {R} rounds x {E} "
               f"local epochs; CNN-2 everywhere; grad-norm clip {clip_norm}; adaptation floor rule "
               f"applied to {len(forced_clients)} clients, uniform fallback for "
               f"{len(fallback_clients)}"),
        per_client_recall=recalls,
        per_client_accuracy=accs,
        extra={
            "fedapen_adapt_size_per_client": adapt_sizes,
            "fedapen_uniform_fallback_clients": fallback_clients,
            "fedapen_forced_rule_clients": forced_clients,
            "learned_weights": recalled,
            "eval_curve": curve,
        },
    )
