"""Rebuild each hetero run's client models from its SAVED config and count parameters.

No model tensors were persisted by the queue (run JSONs store metrics + config only),
so this audit reconstructs the exact architecture each client was instantiated with:
    RunContext.build_client_model(cid) -> models.build_model(spec["name"], in_ch, C, **spec["params"])
using cfg["client_model_specs"] from the run JSON.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_ncg.core import config, hetero_config as hc, models  # noqa: E402

CNN2_PARAMS = {("BloodMNIST"): 44_584, ("FashionMNIST"): 50_282}


def count(spec, in_ch, C):
    m = models.build_model(spec["name"], in_ch, C, **(spec.get("params") or {}))
    return int(sum(p.numel() for p in m.parameters()))


def main():
    bad, all_runs = [], []
    for method in hc.HET_METHODS:
        for setting in hc.HET_SETTINGS[method]:
            for dataset in config.DATASET_NAMES:
                for seed in config.SEEDS:
                    path = hc.run_path(setting, method, dataset, seed)
                    if not path.exists():
                        continue
                    d = config.load_json(path)
                    cfg = d["config"]
                    in_ch = config.DATASETS[dataset]["in_channels"]
                    C = config.DATASETS[dataset]["num_classes"]
                    specs = cfg["client_model_specs"]
                    counts = [count(s, in_ch, C) for s in specs]
                    cnt = Counter(counts)
                    distinct = sorted(cnt)
                    print(f"{d['run_id']}")
                    print(f"  per-client params: {counts}")
                    print(f"  distinct={len(distinct)} {dict(cnt)}")
                    is_cnn2_all = all(c == CNN2_PARAMS[dataset] for c in counts)
                    identical = len(distinct) == 1
                    if is_cnn2_all or identical:
                        bad.append((d["run_id"], "CNN2-for-all" if is_cnn2_all else "identical"))
                    all_runs.append((d["run_id"], method, dataset, counts, len(distinct), is_cnn2_all))

    print("\n=== summary ===")
    for method in hc.HET_METHODS:
        for dataset in config.DATASET_NAMES:
            rows = [r for r in all_runs if r[1] == method and r[2] == dataset]
            if not rows:
                continue
            exp = 20 if method == "FedAPEN" else 4
            print(f"{method:8s} {dataset:13s} runs={len(rows)} distinct_per_run="
                  f"{sorted({r[4] for r in rows})} expected~{exp} "
                  f"cnn2_for_all={any(r[5] for r in rows)}")
            if method == "PHP-FL":
                r = rows[0][3]
                print(f"           value->client count: {dict(Counter(r))}")
            else:
                print(f"           value range: {min(rows[0][3]):,} .. {max(rows[0][3]):,}")
    print(f"\nFLAGGED (factory bug candidates): {bad if bad else 'none'}")


if __name__ == "__main__":
    main()
