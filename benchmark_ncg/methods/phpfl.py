"""PHP-FL -- NeurIPS 2025, official code Siyuan01/PHP-FL-main.

Faithful minimal re-implementation of flcore/clients/clientphp.py +
flcore/servers/serverphp.py, instantiated on the benchmark's homogeneous CNN-2:

  * the server keeps a global model; every client owns a private model and a
    personal copy of the global model;
  * each client learns a scalar inference weight on its own validation split
    (10 SGD steps on the scalar), then trains both its private model and its
    personal global copy with
        CE + symmetric KL + MMD(features) + CE(ensemble);
  * after local training the client computes a top-p parameter mask with
    p = sigmoid(sig_k, (join_times+1)/(round+1)) * tau;
  * the server uniformly averages the uploaded personal global copies (1/K);
  * each uploaded client overwrites the top-p fraction of its personal global
    copy with the server parameters (masked update);
  * the deployed model is  w*private + (1-w)*personal_global, evaluated on the
    global test set per client; per-class recalls are averaged across clients.

Robustness guard (same floor policy as the FedAPEN correction, recorded in the
run JSON): if floor(0.1*n) < 2 the validation split is forced to
min(5, max(2, n//4)) samples; clients with n < 4 skip weight learning and keep
w = 0.5, so a tiny Dirichlet shard cannot crash the run.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..core import metrics, models, train_utils
from .common import RunContext, auxiliary_split, make_result, record_eval_round


class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_mul, self.kernel_num = kernel_mul, kernel_num

    def forward(self, source, target):
        s = source / (source.norm(dim=1, keepdim=True).detach()) + 1e-10
        t = target / (target.norm(dim=1, keepdim=True).detach()) + 1e-10
        return (self._k(s, s).mean() + self._k(t, t).mean() - 2 * self._k(s, t).mean())

    def _k(self, x1, x2):
        l2 = torch.cdist(x1, x2, p=2) ** 2
        bandwidth = torch.median(l2.detach()) / (2 * self.kernel_mul ** (self.kernel_num // 2))
        bandwidth = torch.clamp(bandwidth, min=1e-5)
        return torch.exp(-(1.0 / (2 * bandwidth)) * l2)


def _top_p_mask(model: nn.Module, p: float):
    named = list(model.named_parameters())
    if p <= 0:
        return {name: torch.zeros_like(param, dtype=torch.bool) for name, param in named}
    all_params = torch.cat([param.detach().reshape(-1) for _, param in named])
    threshold = torch.quantile(all_params.abs(), 1.0 - min(1.0, float(p)))
    return {name: (param.detach().abs() >= threshold) for name, param in named}


def _sigmoid(k, x):
    return float(1.0 / (1.0 + np.exp(k * (x - 0.5))))


def _learn_weight(ctx, private, personal, w, opt_w, loader, criterion, steps):
    private.eval()
    personal.eval()
    for _ in range(steps):
        for xb, yb in loader:
            opt_w.zero_grad(set_to_none=True)
            xb, yb = xb.to(ctx.device), yb.to(ctx.device)
            with torch.no_grad():
                out_p, out_g = private(xb), personal(xb)
            loss = criterion(w * out_p + (1.0 - w) * out_g, yb)
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
    sig_k, tau = float(hp["sig_k"]), float(hp["tau"])
    clip_norm = float(hp.get("grad_clip_norm", 0.0))

    criterion = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")
    log_softmax, softmax = nn.LogSoftmax(dim=1), nn.Softmax(dim=1)
    mmd = MMDLoss()

    global_model = ctx.build_model()
    server_state = models.clone_state(global_model)
    private = {cid: ctx.build_model() for cid in range(n_clients)}
    personal = {cid: ctx.build_model() for cid in range(n_clients)}
    for cid in range(n_clients):
        personal[cid].load_state_dict(server_state)

    weights = {cid: torch.tensor([0.5], requires_grad=True, device=ctx.device) for cid in range(n_clients)}
    opt_weights = {cid: torch.optim.SGD([weights[cid]], lr=float(hp["private_weight_lr"]))
                   for cid in range(n_clients)}
    staged = {cid: [] for cid in range(n_clients)}
    join_times = {cid: 0 for cid in range(n_clients)}
    masks = {cid: None for cid in range(n_clients)}
    update_ratio = {cid: 1.0 for cid in range(n_clients)}

    labels = ctx.train_y.numpy()
    splits, valid_sizes, fallback_clients, forced_clients = {}, [], [], []
    for cid in range(n_clients):
        tr_idx, val_idx, info = auxiliary_split(
            ctx.client_indices[cid], labels, seed=cfg["seed"] * 37 + cid,
            ratio=float(hp["valid_ratio"]), min_size=int(hp["min_valid"]),
            max_size=int(hp["max_valid"]), divisor=int(hp["valid_floor_divisor"]),
        )
        splits[cid] = (tr_idx, val_idx)
        valid_sizes.append(int(info["aux_size"]))
        if info["fallback"]:
            fallback_clients.append(cid)
        if info["forced"]:
            forced_clients.append(cid)

    logger(f"[PHP-FL {cfg['run_id']}] valid sizes={valid_sizes} "
           f"forced_rule={forced_clients} fallback={fallback_clients}")

    curve = {} if eval_rounds else None
    for r in range(R):
        lr = ctx.lr_for_round(r)
        selected = ctx.select_clients(r)

        for cid in selected:
            _, val_idx = splits[cid]
            join_times[cid] += 1
            if len(val_idx) < 2:
                staged[cid].append(0.5)
                continue
            vds = Subset(torch.utils.data.TensorDataset(ctx.train_x, ctx.train_y),
                         [int(i) for i in val_idx])
            vloader = DataLoader(
                vds, batch_size=min(int(tr["batch_size"]), len(val_idx)), shuffle=True,
                num_workers=0, drop_last=False,
                generator=torch.Generator().manual_seed(cfg["seed"] * 7 + cid * 131 + r),
            )
            staged[cid].append(_learn_weight(ctx, private[cid], personal[cid], weights[cid],
                                             opt_weights[cid], vloader, criterion,
                                             int(hp["weight_steps"])))

        for cid in selected:
            tr_idx, _ = splits[cid]
            w_private = staged[cid][-1] if staged[cid] else 0.5
            private[cid].train()
            personal[cid].train()
            opt_p = train_utils.make_optimizer(private[cid], lr, tr["momentum"], tr["weight_decay"])
            opt_g = train_utils.make_optimizer(personal[cid], lr, tr["momentum"], tr["weight_decay"])
            loader = ctx.loader(tr_idx, shuffle=True, seed=cfg["seed"] * 100000 + r * 1000 + cid,
                                drop_last=len(tr_idx) >= int(tr["batch_size"]))
            for _ in range(E):
                for xb, yb in loader:
                    xb, yb = xb.to(ctx.device), yb.to(ctx.device)
                    opt_p.zero_grad(set_to_none=True)
                    opt_g.zero_grad(set_to_none=True)
                    rep_p = private[cid].base(xb)
                    out_p = private[cid].head(rep_p)
                    rep_g = personal[cid].base(xb)
                    out_g = personal[cid].head(rep_g)
                    ens_p = w_private * out_p + (1 - w_private) * out_g.detach()
                    ens_g = w_private * out_p.detach() + (1 - w_private) * out_g
                    loss_p = (mmd(rep_p, rep_g.detach()) + criterion(out_p, yb)
                              + kl(log_softmax(out_p), softmax(out_g.detach()))
                              + criterion(ens_p, yb))
                    loss_g = (mmd(rep_p.detach(), rep_g) + criterion(out_g, yb)
                              + kl(log_softmax(out_g), softmax(out_p.detach()))
                              + criterion(ens_g, yb))
                    loss_p.backward(retain_graph=True)
                    loss_g.backward()
                    if clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(private[cid].parameters(), clip_norm)
                        torch.nn.utils.clip_grad_norm_(personal[cid].parameters(), clip_norm)
                    opt_p.step()
                    opt_g.step()

            update_ratio[cid] = _sigmoid(sig_k, (join_times[cid] + 1) * 1.0 / (r + 1))
            masks[cid] = _top_p_mask(personal[cid], update_ratio[cid] * tau)

        server_state = models.average_states([personal[cid].state_dict() for cid in selected])
        global_model.load_state_dict(server_state)

        for cid in selected:
            local_state = personal[cid].state_dict()
            new_state = {}
            for name, tensor in local_state.items():
                mask = masks[cid].get(name)
                if mask is not None and name in server_state:
                    new_state[name] = torch.where(mask.to(tensor.device),
                                                  server_state[name].to(tensor.device), tensor)
                else:
                    new_state[name] = tensor
            personal[cid].load_state_dict(new_state)

        if eval_rounds and (r + 1) in set(int(x) for x in eval_rounds):
            _, accs = ctx.eval_deployed(private, personal, [staged[c][-1] for c in range(n_clients)])
            record_eval_round(curve, r, eval_rounds, float(np.mean(accs)))

        if (r + 1) % 10 == 0 or r == 0:
            logger(f"[PHP-FL {cfg['run_id']}] round {r + 1}/{R} lr={lr:.5f} "
                   f"joined={len(selected)} "
                   f"w_private_mean={np.mean([staged[c][-1] for c in selected]):.3f} "
                   f"update_ratio_mean={np.mean([update_ratio[c] for c in selected]):.4f}")

    deployed_w = [staged[c][-1] if staged[c] else 0.5 for c in range(n_clients)]
    recalls, accs = ctx.eval_deployed(private, personal, deployed_w)
    fl_recall = metrics.average_recall_vectors(recalls)
    logger(f"[PHP-FL {cfg['run_id']}] final gap={metrics.class_gap(fl_recall):.4f} "
           f"acc={np.mean(accs):.4f}")

    return make_result(
        ctx, fl_recall, t0,
        notes=(f"PHP-FL deployed client ensembles w*private+(1-w)*personal_global; per-client "
               f"per-class recalls averaged across {n_clients} clients BEFORE the gap; {R} rounds "
               f"x {E} local epochs; CNN-2 everywhere; sig_k={sig_k}, tau={tau}; "
               f"grad-norm clip {clip_norm}; validation floor rule applied to "
               f"{len(forced_clients)} clients, uniform fallback for {len(fallback_clients)}"),
        per_client_recall=recalls,
        per_client_accuracy=accs,
        extra={
            "phpfl_valid_size_per_client": valid_sizes,
            "phpfl_uniform_fallback_clients": fallback_clients,
            "phpfl_forced_rule_clients": forced_clients,
            "learned_weights": deployed_w,
            "eval_curve": curve,
        },
    )
