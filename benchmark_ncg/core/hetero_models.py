"""Small, 28x28-native substitutes for the PHP-FL HtM4 family.

The official HtM4 list (googlenet / densenet121 / efficientnet-b1 / resnet18)
ranges 5.6-11.2M parameters and is infeasible on this GPU (see DEVIATIONS.md).
These four networks preserve four distinct architecture families and the
round-robin assignment, while staying <= 0.5M parameters and <= ~3x CNN-2 cost.

All models expose ``base`` (flat 512-dim features, matching PHP-FL's
BaseHeadSplit AdaptiveAvgPool1d(512) projection) and ``head`` (Linear(512, C)),
so PHP-FL can train its private model and its personal global model jointly and
compute the MMD term on 512-dim representations.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_DIM = 512


def _cbn(in_c, out_c, k=3, s=1, p=None):
    p = k // 2 if p is None else p
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class _Inception(nn.Module):
    def __init__(self, in_c, c1, c3r, c3, c5r, c5, cpr, out_c):
        super().__init__()
        self.b1 = nn.Conv2d(in_c, c1, 1, bias=False)
        self.b2 = nn.Sequential(nn.Conv2d(in_c, c3r, 1, bias=False), nn.ReLU(inplace=True),
                                nn.Conv2d(c3r, c3, 3, padding=1, bias=False))
        self.b3 = nn.Sequential(nn.Conv2d(in_c, c5r, 1, bias=False), nn.ReLU(inplace=True),
                                nn.Conv2d(c5r, c5, 5, padding=2, bias=False))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                                nn.Conv2d(in_c, cpr, 1, bias=False))
        self.proj = nn.Sequential(
            nn.Conv2d(c1 + c3 + c5 + cpr, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1))


class GoogLeNetSmall(nn.Module):
    def __init__(self, in_c, C, stem1=48, stem2=96, inc=96, proj=96, n_inc=3):
        super().__init__()
        self.stem = nn.Sequential(_cbn(in_c, stem1), _cbn(stem1, stem2, s=2))
        incs = []
        ch = stem2
        for _ in range(n_inc):
            incs.append(_Inception(ch, inc // 3, inc // 4, inc // 3,
                                   inc // 12, inc // 6, inc // 6, inc))
            ch = inc
        self.base = nn.Sequential(
            self.stem, *incs,
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
            nn.Linear(ch, FEATURE_DIM), nn.BatchNorm1d(FEATURE_DIM), nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(FEATURE_DIM, C)

    def forward(self, x):
        return self.head(self.base(x))


class GoogLeNetTiny(GoogLeNetSmall):
    def __init__(self, in_c, C):
        super().__init__(in_c, C, stem1=16, stem2=32, inc=32, proj=32)


class GoogLeNetMicro(GoogLeNetSmall):
    def __init__(self, in_c, C):
        super().__init__(in_c, C, stem1=16, stem2=24, inc=24, proj=24, n_inc=2)


class GoogLeNetMicro2(GoogLeNetSmall):
    def __init__(self, in_c, C):
        super().__init__(in_c, C, stem1=12, stem2=20, inc=20, proj=20, n_inc=1)


class _DenseLayer(nn.Module):
    def __init__(self, in_c, growth, bn_size=4):
        super().__init__()
        mid = bn_size * growth
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_c), nn.ReLU(inplace=True), nn.Conv2d(in_c, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True), nn.Conv2d(mid, growth, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return torch.cat([x, self.net(x)], dim=1)


class _Transition(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_c), nn.ReLU(inplace=True), nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class DenseNetSmall(nn.Module):
    def __init__(self, in_c, C, growth=16, layers=(3, 3, 3), stem_ch=32):
        super().__init__()
        stem = _cbn(in_c, stem_ch)
        ch = stem_ch
        blocks = []
        for i, n in enumerate(layers):
            for _ in range(n):
                blocks.append(_DenseLayer(ch, growth))
                ch += growth
            if i < len(layers) - 1:
                blocks.append(_Transition(ch, ch // 2))
                ch = ch // 2
        self.base = nn.Sequential(
            stem, *blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
            nn.Linear(ch, FEATURE_DIM), nn.BatchNorm1d(FEATURE_DIM), nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(FEATURE_DIM, C)

    def forward(self, x):
        return self.head(self.base(x))


class DenseNetTiny(DenseNetSmall):
    def __init__(self, in_c, C):
        super().__init__(in_c, C, growth=8, layers=(2, 2), stem_ch=16)


class DenseNetMicro(DenseNetSmall):
    def __init__(self, in_c, C):
        super().__init__(in_c, C, growth=8, layers=(2,), stem_ch=16)


class DenseNetMicro2(DenseNetSmall):
    def __init__(self, in_c, C):
        super().__init__(in_c, C, growth=8, layers=(1,), stem_ch=12)


class _MBConv(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, expand=4):
        super().__init__()
        mid = in_c * expand
        self.use_res = (s == 1 and in_c == out_c)
        self.block = nn.Sequential(
            nn.Conv2d(in_c, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.SiLU(inplace=True),
            nn.Conv2d(mid, mid, k, s, k // 2, groups=mid, bias=False),
            nn.BatchNorm2d(mid), nn.SiLU(inplace=True),
            nn.Conv2d(mid, out_c, 1, bias=False), nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


class EfficientNetSmall(nn.Module):
    def __init__(self, in_c, C):
        super().__init__()
        self.base = nn.Sequential(
            _cbn(in_c, 16, s=2),
            _MBConv(16, 24, 3, 1),
            _MBConv(24, 40, 5, 2),
            _MBConv(40, 64, 3, 2),
            _cbn(64, 128, k=1, p=0),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
            nn.Linear(128, FEATURE_DIM), nn.BatchNorm1d(FEATURE_DIM), nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(FEATURE_DIM, C)

    def forward(self, x):
        return self.head(self.base(x))


class EfficientNetMicro(nn.Module):
    def __init__(self, in_c, C):
        super().__init__()
        self.base = nn.Sequential(
            _cbn(in_c, 16, s=2),
            _MBConv(16, 24, 3, 1),
            _MBConv(24, 32, 5, 2),
            _cbn(32, 64, k=1, p=0),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
            nn.Linear(64, FEATURE_DIM), nn.BatchNorm1d(FEATURE_DIM), nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(FEATURE_DIM, C)

    def forward(self, x):
        return self.head(self.base(x))


class EfficientNetMicro2(nn.Module):
    def __init__(self, in_c, C):
        super().__init__()
        self.base = nn.Sequential(
            _cbn(in_c, 12, s=2),
            _MBConv(12, 20, 3, 1),
            _cbn(20, 32, k=1, p=0),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
            nn.Linear(32, FEATURE_DIM), nn.BatchNorm1d(FEATURE_DIM), nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(FEATURE_DIM, C)

    def forward(self, x):
        return self.head(self.base(x))


class _BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.down = None
        if stride != 1 or in_c != out_c:
            self.down = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                                      nn.BatchNorm2d(out_c))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        r = self.down(x) if self.down is not None else x
        return F.relu(y + r, inplace=True)


class ResNet8Small(nn.Module):
    def __init__(self, in_c, C):
        super().__init__()
        self.base = nn.Sequential(
            _cbn(in_c, 16),
            _BasicBlock(16, 16, 1),
            _BasicBlock(16, 32, 2),
            _BasicBlock(32, 64, 2),
            _BasicBlock(64, 128, 2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
            nn.Linear(128, FEATURE_DIM), nn.BatchNorm1d(FEATURE_DIM), nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(FEATURE_DIM, C)

    def forward(self, x):
        return self.head(self.base(x))


class ResNet6Micro(nn.Module):
    def __init__(self, in_c, C):
        super().__init__()
        self.base = nn.Sequential(
            _cbn(in_c, 16),
            _BasicBlock(16, 16, 1),
            _BasicBlock(16, 32, 2),
            _BasicBlock(32, 48, 2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
            nn.Linear(48, FEATURE_DIM), nn.BatchNorm1d(FEATURE_DIM), nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(FEATURE_DIM, C)

    def forward(self, x):
        return self.head(self.base(x))


class ResNet4Micro(nn.Module):
    def __init__(self, in_c, C):
        super().__init__()
        self.base = nn.Sequential(
            _cbn(in_c, 12),
            _BasicBlock(12, 12, 1),
            _BasicBlock(12, 24, 2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
            nn.Linear(24, FEATURE_DIM), nn.BatchNorm1d(FEATURE_DIM), nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(FEATURE_DIM, C)

    def forward(self, x):
        return self.head(self.base(x))


SUBSTITUTES = {
    "googlenet_small": GoogLeNetSmall,
    "densenet_small": DenseNetSmall,
    "efficientnet_b0_small": EfficientNetSmall,
    "resnet8_small": ResNet8Small,
}

# Not active: smaller candidates measured for the 3x budget rule (see
# logs/hetero_substitute_benchmark.txt and the approval request).
CANDIDATES = {
    "googlenet_tiny": GoogLeNetTiny,
    "densenet_tiny": DenseNetTiny,
    "googlenet_micro": GoogLeNetMicro,
    "densenet_micro": DenseNetMicro,
    "efficientnet_micro": EfficientNetMicro,
    "resnet6_micro": ResNet6Micro,
    "googlenet_micro2": GoogLeNetMicro2,
    "densenet_micro2": DenseNetMicro2,
    "efficientnet_micro2": EfficientNetMicro2,
    "resnet4_micro": ResNet4Micro,
}

PHPFL_SUBSTITUTE_IDS = [
    {"id": 0, "name": "googlenet_micro2", "family": "GoogLeNet-style inception"},
    {"id": 1, "name": "densenet_micro2", "family": "DenseNet-style dense blocks"},
    {"id": 2, "name": "efficientnet_micro2", "family": "EfficientNet-style MBConv"},
    {"id": 3, "name": "resnet4_micro", "family": "ResNet-style residual"},
]

PHPFL_GLOBAL_ARCH = "googlenet_micro2"


def build_substitute(name: str, in_channels: int, num_classes: int) -> nn.Module:
    cls = SUBSTITUTES.get(name) or CANDIDATES.get(name)
    if cls is None:
        raise KeyError(name)
    return cls(in_channels, num_classes)
