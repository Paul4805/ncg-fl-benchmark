"""The only backbone used by this benchmark: CNN-2 for 28x28 inputs.

    base: conv3x3(in -> 32)-BN-ReLU-MaxPool -> conv3x3(32 -> 64)-BN-ReLU-MaxPool -> flatten
    head: linear(64*7*7 -> C)

`base`/`head` are exposed because PHP-FL trains a private model and a personal
copy of the global model jointly and needs feature representations for its MMD
term.  No other architecture can be constructed: `build_model` rejects anything
that is not named "cnn2".
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class CNN2(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(1),
        )
        self.feature_dim = 64 * 7 * 7
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        return self.head(self.base(x))


def build_model(name: str, in_channels: int, num_classes: int) -> nn.Module:
    if name != "cnn2":
        raise ValueError(f"only 'cnn2' is allowed, got {name!r}")
    return CNN2(in_channels=in_channels, num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ---------------------------------------------------------------------------
# State-dict helpers
# ---------------------------------------------------------------------------
def clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def state_cpu(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def average_states(states: List[Dict[str, torch.Tensor]], weights: List[float] | None = None):
    n = len(states)
    if weights is None:
        weights = [1.0 / n] * n
    total = float(sum(weights))
    out = {}
    for k in states[0]:
        v0 = states[0][k]
        if not torch.is_floating_point(v0):
            out[k] = v0.clone()
            continue
        acc = torch.zeros_like(v0, dtype=torch.float32)
        for st, w in zip(states, weights):
            acc += st[k].to(torch.float32) * (w / total)
        out[k] = acc.to(v0.dtype)
    return out


def flatten_state(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.detach().reshape(-1).to(torch.float32).cpu() for v in state.values()])


def unflatten_into_state(state: Dict[str, torch.Tensor], flat: torch.Tensor):
    out, offset = {}, 0
    for k, v in state.items():
        numel = v.numel()
        out[k] = flat[offset:offset + numel].reshape(v.shape).to(v.dtype).clone()
        offset += numel
    if offset != flat.numel():
        raise ValueError("flat buffer size mismatch")
    return out
