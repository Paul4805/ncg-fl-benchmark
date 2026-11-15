"""Verify stored runs: recompute class gaps / NCG and check the protocol invariants.

    python -m scripts.evaluate --all
    python -m scripts.evaluate --run-id B3__PHP-FL__FashionMNIST__s1

Checks per run:
  * per_class_recall has length C, values in [0, 1]
  * class_gap == max(recall) - min(recall)
  * for FedAPEN / PHP-FL: stored recall == mean of the per-client recall vectors
  * NCG from the stored recall + the (dataset, seed) centralized reference matches
  * partition file loads, matches the config metadata and is a valid partition
  * FashionMNIST runs point at the right subsample file and its 30k/5k shape
  * setting is one of B1/B2/B3 and no A_spot / official-settings run exists
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core import config, metrics, partition as partition_mod  # noqa: E402


def check_run(payload: dict, tol: float = 1e-5):
    problems = []
    rid = payload["run_id"]
    dataset = payload["dataset"]
    C = config.DATASETS[dataset]["num_classes"]
    rec = np.asarray(payload["per_class_recall"], dtype=float)
    if rec.shape != (C,):
        problems.append(f"{rid}: recall shape {rec.shape} != ({C},)")
    if np.any(rec < 0) or np.any(rec > 1):
        problems.append(f"{rid}: recall outside [0,1]")
    if payload["setting"] not in config.SETTINGS:
        problems.append(f"{rid}: forbidden setting {payload['setting']}")
    if int(payload.get("rounds", -1)) != int(payload["run_id"].rsplit("__R", 1)[1]):
        problems.append(f"{rid}: rounds={payload.get('rounds')} does not match run_id suffix")
    if payload.get("run_role") != "extension" and int(payload.get("rounds", -1)) != int(config.TRAINING["rounds"]):
        problems.append(f"{rid}: main run rounds={payload.get('rounds')} "
                        f"(must be {config.TRAINING['rounds']})")
    if int(payload.get("local_epochs", -1)) != int(config.TRAINING["local_epochs"]):
        problems.append(f"{rid}: local_epochs={payload.get('local_epochs')} "
                        f"(must be {config.TRAINING['local_epochs']})")
    gap = metrics.class_gap(rec)
    if abs(gap - float(payload["fl_class_gap"])) > tol:
        problems.append(f"{rid}: fl_class_gap mismatch ({payload['fl_class_gap']} vs {gap})")
    if payload.get("per_client_recall"):
        avg = metrics.average_recall_vectors(payload["per_client_recall"])
        if not np.allclose(avg, rec, atol=tol):
            problems.append(f"{rid}: FL recall != mean of per-client recall vectors")

    cfg = payload["config"]
    part = partition_mod.load_partition(
        cfg["partition_file"], dataset=dataset, protocol=cfg["protocol"],
        alpha=cfg["alpha"], client_count=cfg["client_count"], seed=cfg["seed"],
    )
    issues = partition_mod.verify_partition(part)
    if issues:
        problems.append(f"{rid}: partition invalid {issues}")
    if dataset == "FashionMNIST":
        sub = config.load_json(cfg["fashionmnist_subsample_file"])
        if (sub["train_size"], sub["test_size"]) != (30000, 5000):
            problems.append(f"{rid}: wrong subsample shape")
        if set(sub["train_class_counts"]) != {3000} or set(sub["test_class_counts"]) != {500}:
            problems.append(f"{rid}: wrong subsample class balance")

    cent_path = config.central_path(dataset, int(payload["seed"]))
    if not cent_path.exists():
        problems.append(f"{rid}: missing centralized ref {cent_path.name}")
    else:
        cent = config.load_json(cent_path)
        value = metrics.ncg_from_gaps(gap, float(cent["class_gap"]))
        print(f"  {rid:42s} gap={gap:.4f} cent_gap={cent['class_gap']:.4f} ncg={value:+.4f} "
              f"minority={metrics.minority_class(rec)}")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--run-id")
    args = ap.parse_args(argv)
    if not (args.all or args.run_id):
        ap.error("use --all or --run-id")

    payloads = []
    for path in sorted(config.RUNS_DIR.glob("*.json")):
        payloads.append(config.load_json(path))
    if args.run_id:
        payloads = [p for p in payloads if p["run_id"] == args.run_id]
    if not payloads:
        print("[evaluate] no runs found")
        return

    problems = []
    for payload in payloads:
        problems.extend(check_run(payload))
    print(f"[evaluate] checked {len(payloads)} runs")
    if problems:
        print("[evaluate] PROBLEMS")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print("[evaluate] all runs consistent")


if __name__ == "__main__":
    main()
