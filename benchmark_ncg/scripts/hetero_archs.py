"""Audit the OFFICIAL heterogeneous architecture lists + write the mandated artifacts.

Writes:
  benchmark_ncg/logs/heterogeneous_archs.txt
  benchmark_ncg/partitions/hetero_arch_map.json
  benchmark_ncg/results/main_table_checksums.json   (baseline; main table untouched)
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmark_ncg"
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torchvision


class CNN2BN(nn.Module):
    """Official FedAPEN cnn2_bn (models/cnn.py:81-109), 28x28-adapted flatten size."""

    def __init__(self, in_channels=3, num_classes=10, channel_diff=0):
        super().__init__()
        ch = [64 + channel_diff, 128 + 2 * channel_diff, 256 + 4 * channel_diff]
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, ch[0], 3), nn.BatchNorm2d(ch[0]), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(ch[0], ch[1], 3), nn.BatchNorm2d(ch[1]), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(ch[1], ch[2], 3), nn.BatchNorm2d(ch[2]), nn.ReLU())
        with torch.no_grad():
            n = self._f(torch.zeros(1, in_channels, 28, 28)).shape[1]
        self.fc = nn.Linear(n, num_classes)

    def _f(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = self.conv3(x)
        return x.flatten(1)

    def forward(self, x):
        return self.fc(self._f(x))

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C_BLOOD, C_FASHION = 8, 10

# ---------------------------------------------------------------------------
# FedAPEN: models/cnn.py:87-89  channel_dif = (5 - client_id) * 2 - 1
#          node.py:65-67         used when --he 1 for the adaptive algorithms
# ---------------------------------------------------------------------------
def fedapen_arch(cid):
    d = (5 - cid) * 2 - 1
    return {
        "id": cid,
        "class_name": "CNN2_BN",
        "channel_diff": d,
        "channel_widths": [64 + d, 128 + 2 * d, 256 + 4 * d],
        "source": "FedAPEN/models/cnn.py:87-89 + FedAPEN/node.py:65-67 (--he 1)",
    }


def fedapen_params(cid, in_ch, ncls):
    torch.manual_seed(0)
    m = CNN2BN(in_channels=in_ch, num_classes=ncls, channel_diff=(5 - cid) * 2 - 1)
    return int(sum(p.numel() for p in m.parameters()))


# ---------------------------------------------------------------------------
# PHP-FL: system/main.py:44-51  HtM4 list (round-robin by client id)
# ---------------------------------------------------------------------------
def phpfl_backbone(name, in_ch, ncls):
    name = name.replace("-", "_")
    if name == "googlenet":
        net = torchvision.models.googlenet(weights=None, num_classes=ncls,
                                           aux_logits=False, init_weights=False)
        if in_ch != 3:
            net.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        net.fc = nn.Identity()
    elif name == "densenet121":
        net = torchvision.models.densenet121(weights=None, num_classes=ncls)
        # 28x28 adaptation (CIFAR-style stem); also used in the homogeneous run
        net.features.conv0 = nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False)
        net.features.pool0 = nn.Identity()
        net.classifier = nn.Identity()
    elif name == "efficientnet_b1":
        net = torchvision.models.efficientnet_b1(weights=None, num_classes=ncls)
        if in_ch != 3:
            net.features[0][0] = nn.Conv2d(in_ch, 32, 3, 2, 1, bias=False)
        net.classifier = nn.Identity()
    elif name == "resnet18":
        net = torchvision.models.resnet18(weights=None, num_classes=ncls)
        if in_ch != 3:
            net.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        net.fc = nn.Identity()
    else:
        raise KeyError(name)
    return net, nn.Linear(512, ncls)


def phpfl_params(name, in_ch, ncls):
    net, head = phpfl_backbone(name, in_ch, ncls)
    return int(sum(p.numel() for p in net.parameters()) + sum(p.numel() for p in head.parameters()))


PHPFL_LIST = [
    {"id": 0, "class_name": "googlenet", "source": "PHP-FL/system/main.py:46"},
    {"id": 1, "class_name": "densenet121", "source": "PHP-FL/system/main.py:47"},
    {"id": 2, "class_name": "efficientnet-b1", "source": "PHP-FL/system/main.py:48 (efficientnet_pytorch)"},
    {"id": 3, "class_name": "resnet18", "source": "PHP-FL/system/main.py:49"},
]


class PHPFLWrap(nn.Module):
    """BaseHeadSplit equivalent: flat features -> AdaptiveAvgPool1d(512) -> linear head."""

    def __init__(self, net, head):
        super().__init__()
        self.net = net
        self.head = head

    def forward(self, x):
        f = self.net(x)
        f = torch.nn.functional.adaptive_avg_pool1d(f, 512)
        return self.head(f)


def timed_fwd_bwd(net, head, in_ch, iters=3, bs=64, ncls=C_BLOOD):
    net = net.to(DEV).train()
    head = head.to(DEV)
    x = torch.randn(bs, in_ch, 28, 28, device=DEV)
    y = torch.randint(0, ncls, (bs,), device=DEV)
    crit = nn.CrossEntropyLoss()
    params = list(net.parameters()) + list(head.parameters())
    opt = torch.optim.SGD(params, lr=0.01)
    for _ in range(1):
        opt.zero_grad()
        crit(head(net(x)), y).backward()
        opt.step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        opt.zero_grad()
        crit(head(net(x)), y).backward()
        opt.step()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0


def main():
    lines = []
    lines.append("HETEROGENEOUS ARCHITECTURE AUDIT (official repos, read-only)")
    lines.append(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    lines.append("")
    lines.append("FedAPEN  source: --he 1 -> node.py:65-67 -> models/cnn.py:87-89")
    lines.append("  channel_dif = (5 - client_id) * 2 - 1 ; widths = [64+d, 128+2d, 256+4d]")
    lines.append("  one distinct architecture per client (20 clients -> 20 archs)")
    fa_map = {}
    fa_archs = []
    for cid in range(20):
        a = fedapen_arch(cid)
        p_blood = fedapen_params(cid, 3, C_BLOOD)
        p_fashion = fedapen_params(cid, 1, C_FASHION)
        a.update({"params": p_blood,
                  "params_by_dataset": {"BloodMNIST": p_blood, "FashionMNIST": p_fashion}})
        fa_archs.append(a)
        fa_map[str(cid)] = cid
        lines.append(f"    arch {cid:2d} {a['class_name']:8s} d={a['channel_diff']:+3d} "
                     f"widths={a['channel_widths']} params(blood)={p_blood:,} "
                     f"params(fashion)={p_fashion:,}  [{a['source']}]")

    lines.append("")
    lines.append("PHP-FL   source: system/main.py:44-51 'HtM4' (args.models[cid % 4])")
    php_map = {str(cid): cid % len(PHPFL_LIST) for cid in range(20)}
    php_archs = []
    for a in PHPFL_LIST:
        p_blood = phpfl_params(a["class_name"], 3, C_BLOOD)
        p_fashion = phpfl_params(a["class_name"], 1, C_FASHION)
        rec = dict(a)
        rec.update({"params": p_blood,
                    "params_by_dataset": {"BloodMNIST": p_blood, "FashionMNIST": p_fashion}})
        php_archs.append(rec)
        lines.append(f"    arch {a['id']} {a['class_name']:16s} params(blood)={p_blood:,} "
                     f"params(fashion)={p_fashion:,}  [{a['source']}]")

    # measured cost per batch (batch 64, 28x28, fwd+bwd)
    lines.append("")
    lines.append("Measured cost (batch 64, 28x28, forward+backward, ms) on this GPU:")
    ref_ms = timed_fwd_bwd(CNN2BN(in_channels=3, num_classes=C_BLOOD), nn.Identity(), 3, 5)
    lines.append(f"    CNN2_BN(d=0, 3ch)                     {ref_ms:8.2f} ms  (homogeneous reference)")
    for a in php_archs:
        net, head = phpfl_backbone(a["class_name"], 3, C_BLOOD)
        ms = timed_fwd_bwd(PHPFLWrap(net, head), nn.Identity(), 3, 3)
        a["fwd_bwd_ms_batch64"] = round(ms, 2)
        lines.append(f"    {a['class_name']:22s} {ms:8.2f} ms  ({ms / ref_ms:5.1f}x CNN2_BN)")
    for cid in (0, 19):
        d = (5 - cid) * 2 - 1
        ms = timed_fwd_bwd(CNN2BN(in_channels=3, num_classes=C_BLOOD, channel_diff=d),
                           nn.Identity(), 3, 3)
        lines.append(f"    CNN2_BN(d={d:+d}, cid={cid})              {ms:8.2f} ms  ({ms / ref_ms:5.1f}x)")

    over = [a for a in php_archs if a["params"] > 5_000_000]
    lines.append("")
    lines.append("5M-parameter rule check (PHP-FL official HtM4):")
    for a in php_archs:
        lines.append(f"    {a['class_name']:16s} {a['params']:>12,}  "
                     f"{'OK' if a['params'] <= 5_000_000 else 'EXCEEDS 5M -> STOP REQUIRED'}")
    lines.append(f"    -> {len(over)} of {len(php_archs)} official PHP-FL models exceed 5M params")
    lines.append("")
    lines.append("FedAPEN max params: %s (all <= 5M)" % f"{max(a['params'] for a in fa_archs):,}")

    log_dir = BENCH / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "heterogeneous_archs.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    arch_map = {
        "FedAPEN": {"method": "FedAPEN", "architectures": fa_archs, "client_to_arch": fa_map},
        "PHP-FL": {"method": "PHP-FL", "architectures": php_archs, "client_to_arch": php_map},
        "note": ("Official lists. FedAPEN: --he 1 per-client width rule (20 clients -> 20 distinct "
                 "CNN2_BN variants). PHP-FL: HtM4 round-robin (client cid -> cid % 4).")
    }
    (BENCH / "partitions" / "hetero_arch_map.json").write_text(
        json.dumps(arch_map, indent=2) + "\n", encoding="utf-8")

    # checksums of the main table (baseline; nothing is modified)
    files = sorted((BENCH / "results" / "runs").glob("*.json")) + \
            sorted((BENCH / "results" / "centralized").glob("*.json")) + \
            sorted((BENCH / "results" / "plots").glob("*.png")) + \
            [BENCH / "results" / n for n in ("raw_results.csv", "summary_ncg.csv",
                                             "summary_class_gap.csv", "manifest.json",
                                             "manifest.csv")]
    hashes = {}
    for f in files:
        if f.exists():
            hashes[str(f.relative_to(BENCH))] = hashlib.sha256(f.read_bytes()).hexdigest()
    combined = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    (BENCH / "results" / "main_table_checksums.json").write_text(
        json.dumps({"combined_sha256": combined, "n_files": len(hashes), "files": hashes},
                   indent=2) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nwrote {log_dir / 'heterogeneous_archs.txt'}")
    print(f"wrote {BENCH / 'partitions' / 'hetero_arch_map.json'}")
    print(f"wrote {BENCH / 'results' / 'main_table_checksums.json'} "
          f"({len(hashes)} files, combined={combined[:16]}...)")


if __name__ == "__main__":
    main()
