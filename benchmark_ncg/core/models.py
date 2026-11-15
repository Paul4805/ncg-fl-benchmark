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


class CNN2BN(nn.Module):
    """Official FedAPEN cnn2_bn (models/cnn.py:81-109), 28x28-adapted flatten size.

    ``channel_diff`` reproduces the official ``--he 1`` heterogeneity:
    channel_diff = (5 - client_id) * 2 - 1, widths = [64+d, 128+2d, 256+4d].
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10, channel_diff: int = 0):
        super().__init__()
        ch = [64 + channel_diff, 128 + 2 * channel_diff, 256 + 4 * channel_diff]
        if min(ch) <= 0:
            raise ValueError(f"invalid channel_diff {channel_diff}: widths {ch}")
        self.channel_num = ch
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, ch[0], 3), nn.BatchNorm2d(ch[0]), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(ch[0], ch[1], 3), nn.BatchNorm2d(ch[1]), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(ch[1], ch[2], 3), nn.BatchNorm2d(ch[2]), nn.ReLU())
        with torch.no_grad():
            n_feat = self._features(torch.zeros(1, in_channels, 28, 28)).shape[1]
        self.fc = nn.Linear(n_feat, num_classes)

    def _features(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = self.conv3(x)
        return x.flatten(1)

    def forward(self, x):
        return self.fc(self._features(x))


def build_model(name: str, in_channels: int, num_classes: int, **params) -> nn.Module:
    if name == "cnn2":
        return CNN2(in_channels=in_channels, num_classes=num_classes)
    if name == "cnn2_bn":
        return CNN2BN(in_channels=in_channels, num_classes=num_classes,
                      channel_diff=int(params.get("channel_diff", 0)))
    from .hetero_models import CANDIDATES, SUBSTITUTES, build_substitute
    if name in SUBSTITUTES or name in CANDIDATES:
        return build_substitute(name, in_channels, num_classes)
    raise ValueError(f"unknown backbone {name!r}")


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
