"""Convergence gate before the 72-run queue (correction 3).

Runs on BloodMNIST seed 0 (diagnostic only; never written to results/runs and
never included in raw_results.csv):

  * FedAvg  R=50  (alpha=0.5 partition), test accuracy at rounds 10/20/30/40/50
  * FedAvg  R=100 (alpha=0.5 partition), test accuracy at rounds 50/100
  * PHP-FL  R=50  (alpha=0.5 partition), test accuracy at rounds 10/20/30/40/50
  * PHP-FL  R=50  (alpha=0.1 partition, the worst skew), same rounds

Stop rules:
  PHP-FL(50) - PHP-FL(40) > 1.0 pp  ->  stop (undertrained)
  FedAvg(50) < 0.90 * FedAvg(100)   ->  stop

Exit code 0 = both rules pass; 2 = stop and let the user decide (never raise R
automatically).
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config  # noqa: E402
from core.train_utils import Logger  # noqa: E402
from methods.registry import get_runner  # noqa: E402

EVAL_ROUNDS = [10, 20, 30, 40, 50]


def run_diag(method: str, setting: str, rounds: int, eval_rounds, logger) -> dict:
    cfg = copy.deepcopy(config.resolve_run(setting, method, "BloodMNIST", 0))
    cfg["training"]["rounds"] = int(rounds)
    cfg["run_id"] = f"DIAG__{method}__{setting}__s0__R{rounds}"
    logger(f"=== diagnostic {method} {setting} R={rounds} eval_rounds={eval_rounds}")
    payload = get_runner(method)(cfg, logger=logger, eval_rounds=eval_rounds)
    return {
        "method": method,
        "setting": setting,
        "rounds": int(rounds),
        "eval_curve": payload.get("eval_curve"),
        "final_test_accuracy": payload.get("test_accuracy"),
        "fl_class_gap": payload.get("fl_class_gap"),
        "wall_clock_seconds": payload.get("wall_clock_seconds"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(config.DIAG_DIR / "convergence_bloodmnist_s0.json"))
    args = ap.parse_args(argv)

    t0 = time.time()
    logger = Logger(config.LOGS_DIR / "diagnostics" / "convergence_bloodmnist_s0.log")
    logger(f"convergence diagnostic start (budget {config.WALL_CLOCK_BUDGET_HOURS}h)")

    runs = {}
    runs["fedavg_r50"] = run_diag("FedAvg", "B2", 50, EVAL_ROUNDS, logger)
    runs["fedavg_r100"] = run_diag("FedAvg", "B2", 100, [50, 100], logger)
    runs["phpfl_r50_b2"] = run_diag("PHP-FL", "B2", 50, EVAL_ROUNDS, logger)
    runs["phpfl_r50_b3"] = run_diag("PHP-FL", "B3", 50, EVAL_ROUNDS, logger)

    fed50 = runs["fedavg_r50"]["eval_curve"].get("50")
    fed100 = runs["fedavg_r100"]["eval_curve"].get("100")
    fed_ratio = (fed50 / fed100) if fed50 and fed100 else float("nan")

    php = {}
    for key in ("phpfl_r50_b2", "phpfl_r50_b3"):
        curve = runs[key]["eval_curve"] or {}
        deltas = {}
        for a, b in [(20, 30), (30, 40), (40, 50)]:
            if str(a) in curve and str(b) in curve:
                deltas[f"{a}->{b}"] = 100.0 * (curve[str(b)] - curve[str(a)])
        php[key] = {"curve": curve, "delta_pp": deltas,
                    "delta_40_50_pp": deltas.get("40->50")}

    failures = []
    if not (fed_ratio >= 0.90):
        failures.append(f"FedAvg R=50 reaches only {100 * fed_ratio:.1f}% of its R=100 accuracy "
                        f"({fed50:.4f} vs {fed100:.4f})")
    for key in ("phpfl_r50_b2", "phpfl_r50_b3"):
        d = php[key]["delta_40_50_pp"]
        if d is None:
            failures.append(f"{key}: missing rounds 40/50 in curve")
        elif d > 1.0:
            failures.append(f"{key}: PHP-FL still improving by {d:.2f} pp between rounds 40 and 50")

    verdict = {"passed": not failures, "failures": failures,
               "fedavg_ratio_50_over_100": fed_ratio,
               "phpfl": php, "runs": runs,
               "elapsed_seconds": time.time() - t0,
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    print("\n=== convergence diagnostic (BloodMNIST seed 0) ===")
    for key in ("fedavg_r50", "fedavg_r100"):
        print(f"  {key:14s} curve={runs[key]['eval_curve']}")
    for key in ("phpfl_r50_b2", "phpfl_r50_b3"):
        print(f"  {key:14s} curve={php[key]['curve']}  deltas(pp)={php[key]['delta_pp']}")
    print(f"  FedAvg R50/R100 accuracy ratio = {fed_ratio:.4f} (threshold 0.90)")
    print("  VERDICT:", "PASS" if not failures else "STOP")
    for f in failures:
        print("   -", f)

    config.dump_json(verdict, Path(args.out))
    logger(f"verdict={'PASS' if not failures else 'STOP'} failures={failures}")
    logger.close()
    raise SystemExit(0 if not failures else 2)


if __name__ == "__main__":
    main()
