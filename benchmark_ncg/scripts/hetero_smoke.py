"""Dataset-agnostic smoke test for the heterogeneous runners.

    python -m benchmark_ncg.scripts.hetero_smoke --dataset FashionMNIST --method FedAPEN
    python -m benchmark_ncg.scripts.hetero_smoke --dataset FashionMNIST --method PHP-FL
    python -m benchmark_ncg.scripts.hetero_smoke                      # BloodMNIST, both methods

R=2, E=1, B3 partition. Asserts:
  * client parameter counts match hetero_arch_map's params_by_dataset[dataset]
    (FedAPEN 20 distinct; PHP-FL 4 distinct x 5 clients)
  * no NaN/Inf loss (runners raise FloatingPointError if one occurs)
  * per-class recall over all C classes with zero NaNs
  * main-table checksums unchanged
Smoke outputs use a __SMOKE__ run_id and are excluded from every CSV.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from benchmark_ncg.core import config  # noqa: E402
from benchmark_ncg.core import hetero_config as hc  # noqa: E402
from benchmark_ncg.methods.registry import get_runner  # noqa: E402


def expected_counts(method: str, dataset: str, cfg: dict) -> list[int]:
    return [int(a["params_by_dataset"][dataset]) for a in cfg["method_arch_list"]]


def check(method: str, dataset: str, params: list[int], cfg: dict) -> None:
    exp = expected_counts(method, dataset, cfg)
    cnn2_ref = 44584 if dataset == "BloodMNIST" else 50282
    assert not all(p == cnn2_ref for p in params), "BUG: all clients built as CNN2"
    if method == "FedAPEN":
        assert params == exp, (params, exp)
        assert len(set(params)) == 20, len(set(params))
    else:
        assert set(params) == set(exp), (sorted(set(params)), sorted(set(exp)))
        assert all(params.count(v) == 5 for v in exp), params


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="BloodMNIST", choices=list(config.DATASET_NAMES))
    ap.add_argument("--method", default=None, choices=list(hc.HET_METHODS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--local-epochs", type=int, default=1)
    args = ap.parse_args(argv)

    ok, diffs = hc.verify_main_table()
    if not ok:
        raise SystemExit(f"main-table checksum changed before smoke: {diffs}")
    print("[smoke] main-table checksums unchanged at start")

    dataset = args.dataset
    num_classes = config.DATASETS[dataset]["num_classes"]
    methods = [args.method] if args.method else list(hc.HET_METHODS)

    for method in methods:
        cfg = copy.deepcopy(hc.resolve_run_hetero("B3", method, dataset, args.seed))
        cfg["training"]["rounds"] = args.rounds
        cfg["training"]["local_epochs"] = args.local_epochs
        cfg["run_id"] = (f"B3__{method}__{dataset}__s{args.seed}"
                         f"__SMOKE__HET__R{args.rounds}")
        out = hc.HET_RUNS_DIR / f"{cfg['run_id']}.json"
        t0 = time.time()
        print(f"\n[smoke] {method} {dataset} R={args.rounds} E={args.local_epochs}", flush=True)
        payload = get_runner(method)(cfg, logger=lambda m: print(m, flush=True))

        params = payload.get("client_params_actual")
        if params is None:
            raise SystemExit(f"[smoke] {method}: client_params_actual missing")
        check(method, dataset, params, cfg)

        payload.update({
            "smoke": True, "hetero": True,
            "centralized_ref_mode": "per_architecture",
            "method_arch_list": [
                {"id": i, "class_name": spec["name"], "params": int(p)}
                for i, (spec, p) in enumerate(zip(cfg["client_model_specs"], params))
            ],
            "wall_clock_seconds": time.time() - t0,
        })
        config.dump_json(payload, out)

        rec = np.asarray(payload["per_class_recall"], dtype=float)
        assert rec.shape == (num_classes,), (rec.shape, num_classes)
        assert not np.isnan(rec).any(), rec
        assert all(np.isfinite(w) for w in payload.get("learned_weights", [0.5]))
        pcr = payload.get("per_client_recall") or []
        assert all(np.isfinite(np.asarray(v, dtype=float)).all() for v in pcr)
        assert payload["status"] == "complete"
        print(f"[SMOKE {dataset.upper()}] {method}: PASS "
              f"(params {len(set(params))} distinct [{min(params)}, {max(params)}], "
              f"losses finite, {num_classes}-class recall no NaN, {time.time() - t0:.0f}s)",
              flush=True)

    ok, diffs = hc.verify_main_table()
    if not ok:
        raise SystemExit(f"main-table checksum changed after smoke: {diffs}")
    print("[CHECKSUM] homogeneous main table: unchanged")


if __name__ == "__main__":
    main()
