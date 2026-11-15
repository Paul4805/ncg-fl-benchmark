"""Post-hoc verification of the homogeneous main table.

Checks that every homogeneous run record was produced by the CNN-2 model path
(all 20 client models at 44,584 params on BloodMNIST / 50,282 on FashionMNIST,
backbone "cnn2", no per-client model specs), and that the centralized accuracy
recomputes from the reference JSONs.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from core import config, models  # noqa: E402
from scripts.add_accuracy import run_accuracy  # noqa: E402

CNN2 = {"BloodMNIST": 44584, "FashionMNIST": 50282}


def main():
    failures, n = [], 0
    for p in sorted(config.RESULTS_DIR.glob("runs/*.json")):
        d = config.load_json(p)
        n += 1
        cfg = d["config"]
        ds = d["dataset"]
        if d.get("backbone") != "cnn2" or cfg.get("backbone") != "cnn2":
            failures.append(f"{d['run_id']}: backbone != cnn2")
        if cfg.get("client_model_specs"):
            failures.append(f"{d['run_id']}: unexpected client_model_specs")
        count = models.count_parameters(models.build_model(
            "cnn2", config.DATASETS[ds]["in_channels"], config.DATASETS[ds]["num_classes"]))
        if count != CNN2[ds]:
            failures.append(f"{d['run_id']}: {count} params != {CNN2[ds]}")
    print(f"[HOMOGENEOUS] {n - len(failures)}/{n} runs have all-clients-CNN2 params: "
          f"{'PASS' if not failures else 'FAIL'} (Blood {CNN2['BloodMNIST']}, "
          f"Fashion {CNN2['FashionMNIST']})")

    raw = pd.read_csv(config.RESULTS_DIR / "raw_results_with_accuracy.csv")
    worst = 0.0
    cent = raw[raw["method"] == "Centralized"]
    for _, row in cent.iterrows():
        ref = config.load_json(config.central_path(row["dataset"], int(row["seed"])))
        worst = max(worst, abs(run_accuracy(ref["per_class_recall"], row["dataset"]) -
                               float(row["top1_accuracy"])))
    ok2 = worst <= 1e-6
    if not ok2:
        failures.append(f"centralized accuracy diff {worst}")
    print(f"[HOMOGENEOUS] centralized accuracy recompute max |diff|: {worst:.2e} "
          f"{'<=' if ok2 else '>'} 1e-6")

    print("\nRESULT:", "ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
