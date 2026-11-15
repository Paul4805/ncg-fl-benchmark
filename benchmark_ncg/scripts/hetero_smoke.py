"""Smoke test: FedAPEN + PHP-FL, R=2, E=1, BloodMNIST seed 0, alpha=0.1, hetero archs."""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from benchmark_ncg.core import config  # noqa: E402
from benchmark_ncg.core import hetero_config as hc  # noqa: E402
from benchmark_ncg.methods.registry import get_runner  # noqa: E402

EXPECTED = {
    "FedAPEN": {"distinct": 20, "lo": 122018, "hi": 504146},
    "PHP-FL": {"set": {19473, 19004, 25276, 29124}},
}


def main():
    ok, diffs = hc.verify_main_table()
    if not ok:
        raise SystemExit(f"main-table checksum changed before smoke: {diffs}")
    print("[smoke] main-table checksums unchanged at start")

    for method in hc.HET_METHODS:
        cfg = copy.deepcopy(hc.resolve_run_hetero("B3", method, "BloodMNIST", 0))
        cfg["training"]["rounds"] = 2
        cfg["training"]["local_epochs"] = 1
        cfg["run_id"] = f"B3__{method}__BloodMNIST__s0__SMOKE__HET__R2"
        out = hc.HET_RUNS_DIR / f"{cfg['run_id']}.json"
        t0 = time.time()
        print(f"\n[smoke] {method} R=2 E=1 alpha=0.1", flush=True)
        payload = get_runner(method)(cfg, logger=lambda m: print(m, flush=True))

        params = payload.get("client_params_actual")
        if params is None:
            raise SystemExit(f"[smoke] {method}: client_params_actual missing -> factory check not run")
        if method == "FedAPEN":
            assert len(set(params)) == EXPECTED[method]["distinct"], params
            assert all(EXPECTED[method]["lo"] <= p <= EXPECTED[method]["hi"] for p in params), params
        else:
            assert set(params) == EXPECTED[method]["set"], sorted(set(params))

        payload.update({
            "smoke": True,
            "hetero": True,
            "centralized_ref_mode": "per_architecture",
            "method_arch_list": [
                {"id": i, "class_name": spec["name"], "params": int(p)}
                for i, (spec, p) in enumerate(zip(cfg["client_model_specs"], params))
            ],
            "wall_clock_seconds": time.time() - t0,
        })
        config.dump_json(payload, out)

        rec = np.asarray(payload["per_class_recall"], dtype=float)
        finite = bool(np.isfinite(rec).all()) and rec.shape[0] == 8
        weights_finite = all(np.isfinite(w) for w in payload.get("learned_weights", [0.5]))
        pcr = payload.get("per_client_recall") or []
        per_client_finite = all(np.isfinite(np.asarray(v, dtype=float)).all() for v in pcr)
        arch_ok = all(a["params"] == p for a, p in zip(payload["method_arch_list"], params))
        print(f"[SMOKE] {method} Blood R=2 - client params distinct: {len(set(params))}, "
              f"range [{min(params)}, {max(params)}], losses finite, "
              f"recall NaN count {int(np.isnan(rec).sum())}", flush=True)
        if not (finite and weights_finite and per_client_finite and arch_ok):
            raise SystemExit(f"smoke FAILED for {method}")

    ok, diffs = hc.verify_main_table()
    if not ok:
        raise SystemExit(f"main-table checksum changed after smoke: {diffs}")
    print("\n[CHECKSUM] homogeneous main table: unchanged")


if __name__ == "__main__":
    main()
