"""Accuracy computation for BOTH blocks (derived metrics only; run JSONs untouched).

accuracy          = sum_c(recall_c * n_c) / N
balanced_accuracy = mean_c(recall_c)

Support (deterministic global-test class support):
  BloodMNIST   [244, 624, 311, 579, 243, 284, 666, 470] (N=3421)
  FashionMNIST [500]*10                                 (N=5000)

Part A (homogeneous main table):
  results/raw_results_with_accuracy.csv
  results/summary_accuracy_hom.csv
Part B (heterogeneous supplement):
  results/summary_accuracy_hetero_per_arch.csv
  results/summary_accuracy_hetero.csv
Part D: range assertions, bal-vs-acc sign, centralized cross-check.
No F1 / precision / AUROC is computed (precision is not recoverable).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from benchmark_ncg.core import config  # noqa: E402
from benchmark_ncg.core import hetero_config as hc  # noqa: E402

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
    """Joint interpretation. Thresholds: acc < 0.60 = low; NCG > 0.20 = high."""
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


# ---------------------------------------------------------------------------
# Part A -- homogeneous
# ---------------------------------------------------------------------------
def part_a():
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
    print(f"[A] raw_results_with_accuracy.csv: {len(raw)} rows")
    print(f"[A] summary_accuracy_hom.csv: {len(hom)} rows")
    return raw, hom


# ---------------------------------------------------------------------------
# Part B -- heterogeneous
# ---------------------------------------------------------------------------
def part_b():
    runs = []
    for p in sorted(hc.HET_RUNS_DIR.glob("*.json")):
        if "__SMOKE__" in p.name:
            continue
        d = config.load_json(p)
        pcr = np.asarray(d.get("per_client_recall") or [], dtype=float)
        assert pcr.ndim == 2 and pcr.shape[0] == 20, (p.name, pcr.shape)
        per_acc = [run_accuracy(v, d["dataset"]) for v in pcr]
        per_bal = [run_balanced(v) for v in pcr]
        runs.append({
            "setting": d["setting"], "method": d["method"], "dataset": d["dataset"],
            "seed": int(d["seed"]),
            "acc": float(np.mean(per_acc)), "bal_acc": float(np.mean(per_bal)),
            "ncg_hetero": float(d["ncg_hetero"]),
            "per_client_acc": per_acc,
        })
    runs_df = pd.DataFrame(runs)

    # per-architecture and mean refs
    per_arch = []
    mean_rows = []
    for p in sorted(hc.HET_CENTRAL_DIR.glob("*.json")):
        d = config.load_json(p)
        if d.get("kind") == "centralized_hetero_mean" or "__mean__" in p.name:
            rec = d["cent_recall_hetero"]
            mean_rows.append({
                "method": d["method"], "dataset": d["dataset"], "seed": int(d["seed"]),
                "acc": run_accuracy(rec, d["dataset"]), "bal_acc": run_balanced(rec),
            })
            per_arch.append({
                "method": d["method"], "dataset": d["dataset"], "arch_id": "mean",
                "arch_class_name": "per_architecture_mean", "params": np.nan,
                "seed": int(d["seed"]),
                "acc": run_accuracy(rec, d["dataset"]), "bal_acc": run_balanced(rec),
            })
        else:
            rec = d["per_class_recall"]
            per_arch.append({
                "method": d["method"], "dataset": d["dataset"],
                "arch_id": int(d["arch_id"]),
                "arch_class_name": d.get("backbone_label", d.get("backbone", "")),
                "params": int(d.get("n_parameters", 0)),
                "seed": int(d["seed"]),
                "acc": run_accuracy(rec, d["dataset"]), "bal_acc": run_balanced(rec),
            })
    pa = pd.DataFrame(per_arch).sort_values(["method", "dataset", "seed", "arch_id"],
                                            key=lambda s: s.map(str) if s.name == "arch_id" else s)
    pa.to_csv(config.RESULTS_DIR / "summary_accuracy_hetero_per_arch.csv", index=False)
    mean_df = pd.DataFrame(mean_rows)

    rows = []
    for (setting, method, dataset), g in runs_df.groupby(["setting", "method", "dataset"]):
        m = mean_df[(mean_df["method"] == method) & (mean_df["dataset"] == dataset)]
        rows.append({
            "setting": setting, "method": method, "dataset": dataset,
            "acc_mean": g["acc"].mean(),
            "acc_std": g["acc"].std(ddof=1) if len(g) > 1 else 0.0,
            "bal_acc_mean": g["bal_acc"].mean(),
            "bal_acc_std": g["bal_acc"].std(ddof=1) if len(g) > 1 else 0.0,
            "cent_acc_mean": m["acc"].mean(),
            "cent_acc_std": m["acc"].std(ddof=1) if len(m) > 1 else 0.0,
            "cent_bal_acc_mean": m["bal_acc"].mean(),
            "cent_bal_acc_std": m["bal_acc"].std(ddof=1) if len(m) > 1 else 0.0,
            "num_seeds": int(g["seed"].nunique()),
        })
    het = pd.DataFrame(rows).sort_values(["dataset", "method", "setting"])
    het["acc_gap"] = het["cent_acc_mean"] - het["acc_mean"]
    het["bal_acc_gap"] = het["cent_bal_acc_mean"] - het["bal_acc_mean"]
    ncg_mean = runs_df.groupby(["setting", "method", "dataset"])["ncg_hetero"].mean() \
        if "ncg_hetero" in runs_df else None
    if ncg_mean is not None:
        het = het.merge(ncg_mean.rename("ncg_hetero_mean").reset_index(),
                        on=["setting", "method", "dataset"], how="left")
    else:
        het["ncg_hetero_mean"] = np.nan
    het["acc_ncg_regime"] = [acc_ncg_regime(a, n) for a, n in
                             zip(het["acc_mean"], het["ncg_hetero_mean"])]
    het = het[["setting", "method", "dataset", "acc_mean", "acc_std",
               "bal_acc_mean", "bal_acc_std", "cent_acc_mean", "cent_acc_std",
               "cent_bal_acc_mean", "cent_bal_acc_std", "acc_gap", "bal_acc_gap",
               "ncg_hetero_mean", "acc_ncg_regime", "num_seeds"]]
    het.to_csv(config.RESULTS_DIR / "summary_accuracy_hetero.csv", index=False)
    print(f"[B] summary_accuracy_hetero_per_arch.csv: {len(pa)} rows")
    print(f"[B] summary_accuracy_hetero.csv: {len(het)} rows")
    return runs_df, het, pa


# ---------------------------------------------------------------------------
# Part D -- consistency checks
# ---------------------------------------------------------------------------
def part_d(raw, het):
    for df, name in [(raw, "raw_results_with_accuracy")]:
        for col in ["top1_accuracy", "balanced_accuracy"]:
            vals = df[col].astype(float)
            assert ((vals >= 0) & (vals <= 1)).all(), f"{name}.{col} out of [0,1]"
    for col in ["acc_mean", "bal_acc_mean", "cent_acc_mean", "cent_bal_acc_mean"]:
        vals = het[col].astype(float)
        assert ((vals >= 0) & (vals <= 1)).all(), f"hetero {col} out of [0,1]"

    # D3: centralized accuracy from Part A vs recomputed from the ref JSONs
    cent_rows = raw[raw["method"] == "Centralized"]
    worst = 0.0
    for _, row in cent_rows.iterrows():
        ref = config.load_json(config.central_path(row["dataset"], int(row["seed"])))
        recomputed = run_accuracy(ref["per_class_recall"], row["dataset"])
        worst = max(worst, abs(recomputed - float(row["top1_accuracy"])))
    assert worst <= 1e-6, f"D3 centralized accuracy mismatch {worst}"
    print(f"[D] ranges OK; D3 centralized max|diff| = {worst:.2e} (<= 1e-6) PASS")

    signs = []
    for (method, dataset), g in raw[raw["method"] != "Centralized"].groupby(["method", "dataset"]):
        diff = (g["balanced_accuracy"].astype(float) - g["top1_accuracy"].astype(float)).mean()
        signs.append((method, dataset, float(diff)))
    print("[D] sign of (bal_acc - acc), homogeneous runs (mean over settings/seeds):")
    for method, dataset, diff in signs:
        label = ("minority overperforms" if diff > 1e-9 else
                 "minority underperforms" if diff < -1e-9 else "balanced (no difference)")
        print(f"    {method:8s} {dataset:13s} {diff:+.4f} ({label})")
    return signs


def main():
    raw, hom = part_a()
    runs_df, het, pa = part_b()
    part_d(raw, het)
    print("\n[A] homogeneous accuracy")
    print(hom.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n[B] heterogeneous accuracy")
    print(het.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
