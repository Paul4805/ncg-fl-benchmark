"""Accuracy / balanced accuracy for the homogeneous main table (derived CSVs only).

accuracy          = sum_c(recall_c * n_c) / sum_c(n_c)
balanced_accuracy = mean_c(recall_c)

Support vectors (deterministic global-test class support):
  BloodMNIST   [244, 624, 311, 579, 243, 284, 666, 470]  (N=3421)
  FashionMNIST [500]*10                                  (N=5000)

Writes results/raw_results_with_accuracy.csv (78 rows) and
results/summary_accuracy_hom.csv (24 FL rows + 2 Centralized rows), with the
joint `acc_ncg_regime` column (thresholds: acc < 0.60 low, NCG > 0.20 high).
Run JSONs are never modified. No F1 / precision / AUROC is computed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from benchmark_ncg.core import config  # noqa: E402

SUPPORT = {
    "BloodMNIST": np.array([244, 624, 311, 579, 243, 284, 666, 470], dtype=float),
    "FashionMNIST": np.array([500] * 10, dtype=float),
}


def run_accuracy(recall, dataset: str) -> float:
    r = np.asarray(recall, dtype=float)
    n = SUPPORT[dataset]
    assert r.shape == n.shape, (r.shape, n.shape)
    return float((r * n).sum() / n.sum())


def run_balanced(recall) -> float:
    return float(np.asarray(recall, dtype=float).mean())


def acc_ncg_regime(acc, ncg) -> str:
    """Thresholds: acc < 0.60 = low; NCG > 0.20 = high."""
    if acc != acc:
        return ""
    acc_low = float(acc) < 0.60
    ncg_high = (ncg == ncg) and float(ncg) > 0.20
    if not acc_low and not ncg_high:
        return "high_acc_low_ncg"
    if not acc_low and ncg_high:
        return "high_acc_high_ncg"
    if acc_low and not ncg_high:
        return "low_acc_low_ncg"
    return "low_acc_high_ncg"


def main():
    raw = pd.read_csv(config.RESULTS_DIR / "raw_results.csv")
    accs, bals = [], []
    for _, row in raw.iterrows():
        rec = json.loads(row["per_class_recall_json"])
        accs.append(round(run_accuracy(rec, row["dataset"]), 6))
        bals.append(round(run_balanced(rec), 6))
    raw = raw.copy()
    raw["top1_accuracy"] = accs
    raw["balanced_accuracy"] = bals
    raw.to_csv(config.RESULTS_DIR / "raw_results_with_accuracy.csv", index=False)

    rows = []
    fl = raw[raw["method"] != "Centralized"]
    for (setting, method, dataset), g in fl.groupby(["setting", "method", "dataset"]):
        a = g["top1_accuracy"].astype(float)
        b = g["balanced_accuracy"].astype(float)
        rows.append({
            "setting": setting, "method": method, "dataset": dataset,
            "acc_mean": a.mean(), "acc_std": a.std(ddof=1) if len(a) > 1 else 0.0,
            "bal_acc_mean": b.mean(), "bal_acc_std": b.std(ddof=1) if len(b) > 1 else 0.0,
            "num_seeds": int(g["seed"].nunique()),
        })
    cent = raw[raw["method"] == "Centralized"]
    for dataset, g in cent.groupby("dataset"):
        rows.append({
            "setting": "reference", "method": "Centralized", "dataset": dataset,
            "acc_mean": g["top1_accuracy"].astype(float).mean(), "acc_std": np.nan,
            "bal_acc_mean": g["balanced_accuracy"].astype(float).mean(), "bal_acc_std": np.nan,
            "num_seeds": int(g["seed"].nunique()),
        })
    hom = pd.DataFrame(rows).sort_values(["dataset", "method", "setting"])
    try:
        ncg = pd.read_csv(config.RESULTS_DIR / "summary_ncg.csv")[
            ["setting", "method", "dataset", "ncg_mean"]]
        hom = hom.merge(ncg, on=["setting", "method", "dataset"], how="left")
    except Exception:
        hom["ncg_mean"] = np.nan
    hom.loc[hom["method"] == "Centralized", "ncg_mean"] = 0.0
    hom["acc_ncg_regime"] = [acc_ncg_regime(a, n) for a, n in
                             zip(hom["acc_mean"], hom["ncg_mean"])]
    hom.to_csv(config.RESULTS_DIR / "summary_accuracy_hom.csv", index=False)

    # ---- consistency checks -------------------------------------------------
    for col in ("top1_accuracy", "balanced_accuracy"):
        vals = raw[col].astype(float)
        assert ((vals >= 0) & (vals <= 1)).all(), f"{col} out of [0,1]"
    worst = 0.0
    for _, row in cent.iterrows():
        ref = config.load_json(config.central_path(row["dataset"], int(row["seed"])))
        worst = max(worst, abs(run_accuracy(ref["per_class_recall"], row["dataset"]) -
                               float(row["top1_accuracy"])))
    assert worst <= 1e-6, f"centralized accuracy mismatch {worst}"

    print(f"[accuracy] raw_results_with_accuracy.csv: {len(raw)} rows")
    print(f"[accuracy] summary_accuracy_hom.csv: {len(hom)} rows")
    print(f"[accuracy] centralized recompute max |diff| = {worst:.2e} (<= 1e-6) PASS")
    signs = []
    for (method, dataset), g in fl.groupby(["method", "dataset"]):
        diff = (g["balanced_accuracy"].astype(float) - g["top1_accuracy"].astype(float)).mean()
        signs.append((method, dataset, float(diff)))
    print("[accuracy] sign of (bal_acc - acc), homogeneous runs (mean over settings/seeds):")
    for method, dataset, diff in signs:
        label = ("minority overperforms" if diff > 1e-9 else
                 "minority underperforms" if diff < -1e-9 else "balanced (no difference)")
        print(f"    {method:8s} {dataset:13s} {diff:+.4f} ({label})")
    print(hom.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
