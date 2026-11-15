"""MANDATORY pre-launch benchmark for the heterogeneous supplementary run.

Measures CNN-2, the FedAPEN heterogeneous client models and the approved PHP-FL
micro2 substitutes with an INTERLEAVED schedule (3 rounds x 10 iterations, all
models in each round) so GPU clock/thermal drift affects every model equally.
Writes logs/hetero_substitute_benchmark.txt, logs/heterogeneous_archs.txt and
partitions/hetero_arch_map.json, and asserts:
  * every PHP-FL substitute <= 0.5M parameters
  * average round time <= 3x the CNN-2 average round time for BOTH methods
Exit code 2 on failure (no training may start).
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from benchmark_ncg.core import config, hetero_config as hc, models  # noqa: E402
from benchmark_ncg.core.hetero_models import PHPFL_SUBSTITUTE_IDS  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 64
ROUNDS = 3
ITERS = 10
EPOCHS = 5


def build(spec, in_ch, C):
    return models.build_model(spec["name"], in_ch, C, **(spec.get("params") or {})).to(DEV)


def _one(model, in_ch, C, iters):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(BATCH, in_ch, 28, 28, device=DEV)
    y = torch.randint(0, C, (BATCH,), device=DEV)
    for _ in range(2):
        opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        opt.zero_grad()
        crit(model(x), y).backward()
        opt.step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return sum(times) / len(times)


def measure_interleaved(models_dict, in_ch, C):
    for m in models_dict.values():
        _one(m, in_ch, C, 2)
    per = {k: [] for k in models_dict}
    for _ in range(ROUNDS):
        for k, m in models_dict.items():
            per[k].append(_one(m, in_ch, C, ITERS))
    return {k: float(statistics.median(v)) for k, v in per.items()}


def round_seconds(pair_ms, sizes, aux_ratio):
    vals = []
    for n in sizes:
        nb = max(1, int(np.ceil(n / BATCH)))
        ab = max(1, int(np.ceil(aux_ratio * n / BATCH)))
        pair = pair_ms / 1000.0
        vals.append(EPOCHS * nb * pair + 10 * ab * pair)
    return float(np.mean(vals)), float(np.max(vals))


def main():
    ok, diffs = hc.verify_main_table()
    if not ok:
        raise SystemExit(f"ABORT: main-table checksum changed: {diffs}")
    print("[bench] main-table checksum: unchanged")

    lines = ["HETERO PRE-LAUNCH BENCHMARK (batch 64, 28x28, fwd+bwd; "
             f"{ROUNDS} interleaved rounds x {ITERS} iters, median of round means)",
             f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", "",
             "method | arch_id | class_name | params | ms_batch_mean | x_cnn2 | "
             "est_round_seconds_avg | est_round_seconds_max"]

    est = {}
    for dataset, C in (("BloodMNIST", 8), ("FashionMNIST", 10)):
        in_ch = 3 if dataset == "BloodMNIST" else 1
        models_dict = {"cnn2": build({"name": "cnn2", "params": {}}, in_ch, C),
                       "shared": build(hc.fedapen_shared_spec(), in_ch, C),
                       "global": build(hc.phpfl_global_spec(), in_ch, C)}
        for cid in range(20):
            models_dict[f"fa{cid}"] = build(hc.fedapen_spec(cid), in_ch, C)
        for rec in PHPFL_SUBSTITUTE_IDS:
            models_dict[f"sub{rec['id']}"] = build({"name": rec["name"], "params": {}}, in_ch, C)

        ms = measure_interleaved(models_dict, in_ch, C)
        part = config.load_json(config.partition_path(dataset, "dirichlet", 0.1, 20, 0))
        sizes = part["client_sizes"]

        ref_avg, ref_max = round_seconds(2 * ms["cnn2"], sizes, 0.05)
        fa_pairs = [ms[f"fa{cid}"] + ms["shared"] for cid in range(20)]
        fa_avg = float(np.mean([round_seconds(p, [s], 0.05)[0] for p, s in zip(fa_pairs, sizes)]))
        fa_max = float(np.max([round_seconds(p, [s], 0.05)[1] for p, s in zip(fa_pairs, sizes)]))
        php_pairs = [ms[f"sub{cid % 4}"] + ms["global"] for cid in range(20)]
        php_avg = float(np.mean([round_seconds(p, [s], 0.10)[0] for p, s in zip(php_pairs, sizes)]))
        php_max = float(np.max([round_seconds(p, [s], 0.10)[1] for p, s in zip(php_pairs, sizes)]))

        est[dataset] = {"cnn2_avg": ref_avg, "cnn2_max": ref_max,
                        "fa_avg": fa_avg, "fa_max": fa_max, "fa_ratio": fa_avg / ref_avg,
                        "php_avg": php_avg, "php_max": php_max, "php_ratio": php_avg / ref_avg}

        # per-model log rows from the interleaved medians
        for label, key in [(f"CNN2 ({dataset})", "cnn2")]:
            m = models_dict[key]
            lines.append(f"CNN2 |  | cnn2 | {sum(p.numel() for p in m.parameters()):,} | "
                         f"{ms[key]:.2f} | 1.00x | nan | nan")
        lines.append(f"FedAPEN | 0 | cnn2_bn(d=+9) | "
                     f"{sum(p.numel() for p in models_dict['fa0'].parameters()):,} | "
                     f"{ms['fa0']:.2f} | {ms['fa0'] / ms['cnn2']:.2f}x | {fa_avg:.1f} | {fa_max:.1f}")
        lines.append(f"FedAPEN | 19 | cnn2_bn(d=-29) | "
                     f"{sum(p.numel() for p in models_dict['fa19'].parameters()):,} | "
                     f"{ms['fa19']:.2f} | {ms['fa19'] / ms['cnn2']:.2f}x | {fa_avg:.1f} | {fa_max:.1f}")
        for rec in PHPFL_SUBSTITUTE_IDS:
            k = f"sub{rec['id']}"
            lines.append(f"PHP-FL | {rec['id']} | {rec['name']} | "
                         f"{sum(p.numel() for p in models_dict[k].parameters()):,} | "
                         f"{ms[k]:.2f} | {ms[k] / ms['cnn2']:.2f}x | {php_avg:.1f} | {php_max:.1f}")

    lines += ["", "Round-time estimates from the B3 partition sizes (5 local epochs + weight learning):"]
    for dataset in ("BloodMNIST", "FashionMNIST"):
        e = est[dataset]
        lines += [f"  {dataset}:",
                  f"    CNN2 reference : avg {e['cnn2_avg']:.1f}s  max {e['cnn2_max']:.1f}s",
                  f"    FedAPEN hetero : avg {e['fa_avg']:.1f}s  max {e['fa_max']:.1f}s  "
                  f"ratio {e['fa_ratio']:.2f}x",
                  f"    PHP-FL hetero  : avg {e['php_avg']:.1f}s  max {e['php_max']:.1f}s  "
                  f"ratio {e['php_ratio']:.2f}x"]

    sub_max = max(int(sum(p.numel() for p in models.build_model(
        rec["name"], 3, 8).parameters())) for rec in PHPFL_SUBSTITUTE_IDS)
    checks = {
        "all_phpfl_substitutes_le_0.5M": sub_max <= 500_000,
        "fedapen_avg_round_le_3x_cnn2": max(est["BloodMNIST"]["fa_ratio"],
                                            est["FashionMNIST"]["fa_ratio"]) <= 3.0,
        "phpfl_avg_round_le_3x_cnn2": max(est["BloodMNIST"]["php_ratio"],
                                          est["FashionMNIST"]["php_ratio"]) <= 3.0,
    }
    lines += ["", "Assertions:"]
    for k, v in checks.items():
        lines.append(f"  {k}: {'PASS' if v else 'FAIL'}")
    lines.append(f"  max PHP-FL substitute params: {sub_max:,}")

    (config.LOGS_DIR / "hetero_substitute_benchmark.txt").write_text("\n".join(lines) + "\n",
                                                                     encoding="utf-8")

    # architecture list
    arch_lines = ["HETEROGENEOUS ARCHITECTURES (approved B2 micro2)",
                  "", "FedAPEN -- official heterogeneous setting (--he 1):",
                  "  source: FedAPEN/main.py:50, node.py:65-67, models/cnn.py:87-89",
                  "  channel_dif = (5 - client_id) * 2 - 1; widths = [64+d, 128+2d, 256+4d]"]
    for a in hc.method_arch_list("FedAPEN"):
        arch_lines.append(f"    arch {a['id']:2d} {a['class_name']:8s} d={a['channel_diff']:+3d} "
                          f"params={a['params']:,}")
    arch_lines += ["", "PHP-FL -- approved micro2 substitutes for official HtM4:",
                   "  official: googlenet 5.60M, densenet121 6.95M, efficientnet-b1 6.52M, "
                   "resnet18 11.18M (all > 5M; infeasible in 12h)",
                   "  source: " + hc.SUBSTITUTE_SOURCE]
    for a in hc.method_arch_list("PHP-FL"):
        arch_lines.append(f"    arch {a['id']} {a['class_name']:20s} params={a['params']:,} "
                          f"({a['family']})")
    arch_lines += ["", "Client assignment: FedAPEN cid -> cid; PHP-FL cid -> cid % 4; "
                       "PHP-FL global = googlenet_micro2"]
    (config.LOGS_DIR / "heterogeneous_archs.txt").write_text("\n".join(arch_lines) + "\n",
                                                             encoding="utf-8")
    hc.write_arch_map()

    print("\n".join(lines))
    print(f"\n[bench] wrote {config.LOGS_DIR / 'hetero_substitute_benchmark.txt'}")
    print(f"[bench] wrote {config.LOGS_DIR / 'heterogeneous_archs.txt'}")
    print(f"[bench] wrote {config.PARTITION_DIR / 'hetero_arch_map.json'}")
    if not all(checks.values()):
        print("[bench] FAILED ASSERTIONS -> STOP, no training")
        raise SystemExit(2)
    print(f"[bench] PASS: PHP-FL ratio {max(est['BloodMNIST']['php_ratio'], est['FashionMNIST']['php_ratio']):.2f}x, "
          f"FedAPEN ratio {max(est['BloodMNIST']['fa_ratio'], est['FashionMNIST']['fa_ratio']):.2f}x")


if __name__ == "__main__":
    main()
