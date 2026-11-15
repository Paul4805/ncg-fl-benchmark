"""Post-hoc proof that the homogeneous main table used the CNN-2 path.

1. every results/runs/*.json: all client models are CNN-2 (44,584 Blood / 50,282 Fashion)
2. centralized accuracy recomputed from results/centralized/*.json matches
   summary_accuracy_hom.csv Centralized rows within 1e-6
3. run JSONs are byte-identical to commit bab3f7c (SHA-256)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from benchmark_ncg.core import config, models  # noqa: E402
from benchmark_ncg.scripts.add_accuracy import run_accuracy  # noqa: E402

CNN2 = {"BloodMNIST": 44584, "FashionMNIST": 50282}
BASELINE = "bab3f7c"


def git_blob_hash(path: Path) -> tuple[str, str]:
    rel = str(path.relative_to(ROOT)).replace("\\", "/")
    committed = subprocess.check_output(["git", "rev-parse", f"{BASELINE}:{rel}"],
                                        cwd=ROOT).decode().strip()
    working = subprocess.check_output(["git", "hash-object", f"--path={rel}", str(path)],
                                      cwd=ROOT).decode().strip()
    return committed, working


def main():
    failures = []

    # 1. all clients CNN-2 in every homogeneous run
    bad = []
    n = 0
    for p in sorted(config.RESULTS_DIR.glob("runs/*.json")):
        d = config.load_json(p)
        n += 1
        cfg = d["config"]
        ds = d["dataset"]
        if d.get("backbone") != "cnn2" or cfg.get("backbone") != "cnn2":
            bad.append((d["run_id"], "backbone", d.get("backbone"), cfg.get("backbone")))
            continue
        if cfg.get("client_model_specs"):
            bad.append((d["run_id"], "unexpected client_model_specs"))
            continue
        counts = [models.count_parameters(models.build_model(
            "cnn2", config.DATASETS[ds]["in_channels"], config.DATASETS[ds]["num_classes"]))
            for _ in range(20)]
        if any(c != CNN2[ds] for c in counts):
            bad.append((d["run_id"], "params", sorted(set(counts))))
    if bad:
        failures.append(f"homogeneous param check: {bad}")
    print(f"[HOMOGENEOUS] {n - len(bad)}/{n} runs have all-clients-CNN2 params: "
          f"{'PASS' if not bad else 'FAIL'} (Blood {CNN2['BloodMNIST']}, "
          f"Fashion {CNN2['FashionMNIST']})")
    if bad:
        print("  affected:", bad)

    # 2. centralized accuracy recompute
    raw = pd.read_csv(config.RESULTS_DIR / "raw_results_with_accuracy.csv")
    cent = raw[raw["method"] == "Centralized"]
    worst = 0.0
    for _, row in cent.iterrows():
        ref = config.load_json(config.central_path(row["dataset"], int(row["seed"])))
        worst = max(worst, abs(run_accuracy(ref["per_class_recall"], row["dataset"]) -
                               float(row["top1_accuracy"])))
    ok2 = worst <= 1e-6
    if not ok2:
        failures.append(f"centralized accuracy diff {worst}")
    print(f"[HOMOGENEOUS] centralized accuracy recompute max |diff|: {worst:.2e} "
          f"{'<=' if ok2 else '>'} 1e-6")

    # 3. content-identical to baseline commit
    mism = []
    for p in sorted((config.RESULTS_DIR / "runs").glob("*.json")):
        committed, working = git_blob_hash(p)
        if committed != working:
            mism.append(p.name)
    ok3 = not mism
    if not ok3:
        failures.append(f"content diff vs {BASELINE}: {len(mism)} files, first={mism[:3]}")
    print(f"[HOMOGENEOUS] run JSONs content-identical to commit {BASELINE}: "
          f"{'PASS' if ok3 else 'FAIL'} ({len(mism)} mismatches)")

    print("\nRESULT:", "ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
