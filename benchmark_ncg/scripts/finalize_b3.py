"""Finalize the B3 heterogeneous block: tagged summaries + accuracy + bar chart + note.

Runs after the Phase-3 queue writes results/runs_hetero/*B3*.json. Derives:
  results/summary_ncg_hetero_B3.csv          (NCG + accuracy columns)
  results/summary_class_gap_hetero_B3.csv
  results/plots_hetero/b3_homogeneous_vs_hetero.png
  results/HETERO_NOTE_B3.md                  (B3 results, verdicts, limitation, accuracy)
Does not modify any run JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from benchmark_ncg.core import config, hetero_config as hc  # noqa: E402
from benchmark_ncg.scripts.add_accuracy import accuracy, balanced_accuracy  # noqa: E402


def load_b3():
    rows = []
    for p in sorted(hc.HET_RUNS_DIR.glob("*B3*.json")):
        if "__SMOKE__" in p.name:
            continue
        d = config.load_json(p)
        rec = np.asarray(d["per_class_recall"], dtype=float)
        rows.append({
            "setting": "B3", "method": d["method"], "dataset": d["dataset"],
            "seed": int(d["seed"]), "ncg_hetero": float(d["ncg_hetero"]),
            "fl_class_gap": float(d["fl_class_gap"]),
            "cent_class_gap_hetero": float(d["cent_class_gap_hetero"]),
            "acc": accuracy(rec, d["dataset"]),
            "bal_acc": balanced_accuracy(rec),
        })
    return pd.DataFrame(rows)


def main():
    df = load_b3()
    if df.empty:
        raise SystemExit("no B3 hetero runs found")

    rows = []
    for (method, dataset), g in df.groupby(["method", "dataset"]):
        rows.append({
            "setting": "B3", "method": method, "dataset": dataset,
            "ncg_hetero_mean": g["ncg_hetero"].mean(),
            "ncg_hetero_std": g["ncg_hetero"].std(ddof=1) if len(g) > 1 else 0.0,
            "class_gap_mean": g["fl_class_gap"].mean(),
            "class_gap_std": g["fl_class_gap"].std(ddof=1) if len(g) > 1 else 0.0,
            "num_seeds": int(g["seed"].nunique()),
            "acc_mean": g["acc"].mean(),
            "acc_std": g["acc"].std(ddof=1) if len(g) > 1 else 0.0,
            "bal_acc_mean": g["bal_acc"].mean(),
            "bal_acc_std": g["bal_acc"].std(ddof=1) if len(g) > 1 else 0.0,
        })
    summ = pd.DataFrame(rows).sort_values(["dataset", "method"])
    summ.to_csv(config.RESULTS_DIR / "summary_ncg_hetero_B3.csv", index=False)

    gaps = []
    for (method, dataset), g in df.groupby(["method", "dataset"]):
        gaps.append({
            "setting": "B3", "method": method, "dataset": dataset,
            "cent_class_gap_hetero_mean": g["cent_class_gap_hetero"].mean(),
            "fl_class_gap_mean": g["fl_class_gap"].mean(),
            "fl_class_gap_std": g["fl_class_gap"].std(ddof=1) if len(g) > 1 else 0.0,
            "num_seeds": int(g["seed"].nunique()),
            "acc_mean": g["acc"].mean(), "bal_acc_mean": g["bal_acc"].mean(),
        })
    pd.DataFrame(gaps).sort_values(["dataset", "method"]).to_csv(
        config.RESULTS_DIR / "summary_class_gap_hetero_B3.csv", index=False)

    # ---- bar chart: B3 homogeneous vs B3 hetero ---------------------------
    hom = pd.read_csv(config.RESULTS_DIR / "summary_ncg.csv")
    methods = [m for m in hc.HET_METHODS if m in set(summ["method"])]
    datasets = list(config.DATASET_NAMES)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    width = 0.36
    xs = np.arange(len(datasets))
    for i, method in enumerate(methods):
        hv, he, hs = [], [], []
        for ds in datasets:
            h = hom[(hom.dataset == ds) & (hom.method == method) & (hom.setting == "B3")]
            t = summ[(summ.dataset == ds) & (summ.method == method)]
            hv.append(float(t["ncg_hetero_mean"].iloc[0]))
            he.append(float(h["ncg_mean"].iloc[0]))
            hs.append(float(t["ncg_hetero_std"].iloc[0]))
        off = (i - (len(methods) - 1) / 2) * (2 * width + 0.06)
        ax.bar(xs + off - width / 2, he, width, label=f"{method} homogeneous", alpha=0.55)
        ax.bar(xs + off + width / 2, hv, width, yerr=hs, capsize=3, label=f"{method} hetero")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("NCG at B3 (alpha=0.1)")
    ax.set_title("B3: homogeneous vs heterogeneous NCG (per-architecture central refs)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(hc.HET_PLOTS_DIR / "b3_homogeneous_vs_hetero.png", dpi=180)
    plt.close(fig)

    # ---- B3 note ---------------------------------------------------------
    base = config.BENCH_ROOT / "HETERO_NOTE.md"
    text = base.read_text(encoding="utf-8") if base.exists() else "# HETERO NOTE (B3)\n"
    acc_lines = ["", "## Accuracy (B3 hetero, mean +/- std over seeds)", "",
                 "| dataset | method | top-1 acc | balanced acc |", "|---|---|---|---|"]
    for _, r in summ.iterrows():
        acc_lines.append(f"| {r['dataset']} | {r['method']} | {r['acc_mean']:.4f} +/- "
                         f"{r['acc_std']:.4f} | {r['bal_acc_mean']:.4f} +/- "
                         f"{r['bal_acc_std']:.4f} |")
    acc_lines += ["", "## Limitation (budget)", "",
                  "B1 and B2 hetero not evaluated (budget). PHP-FL is B3-only; FedAPEN is B3-only "
                  "in this hetero block. The homogeneous R=50 main table remains the primary "
                  "result; this block is a B3 reference-sensitivity check under real heterogeneous "
                  "client architectures.", ""]
    out = config.RESULTS_DIR / "HETERO_NOTE_B3.md"
    out.write_text(text.rstrip() + "\n" + "\n".join(acc_lines), encoding="utf-8")
    if base.exists():
        base.unlink()
    print(f"[b3] wrote summary_ncg_hetero_B3.csv ({len(summ)} rows), class-gap summary, "
          f"bar chart and {out}")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
