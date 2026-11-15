"""Plots:
  1. results/plots/ncg_vs_setting.png      NCG vs B1/B2/B3 per method and dataset (mean +/- std, 3 seeds)
  2. results/plots/recall_heatmap_B3_<dataset>.png   per-class recall per method under B3 (severe skew)
  3. results/plots/class_gap_vs_setting.png          diagnostic: raw class gaps
"""
from __future__ import annotations

import argparse
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

from benchmark_ncg.core import config, report  # noqa: E402

STYLE = {
    "FedAvg": dict(color="#1f77b4", marker="o"),
    "FedAPEN": dict(color="#ff7f0e", marker="s"),
    "FedFV": dict(color="#2ca02c", marker="^"),
    "PHP-FL": dict(color="#d62728", marker="D"),
    "Centralized": dict(color="#444444", marker="x", linestyle="--"),
}
SETTINGS = ["B1", "B2", "B3"]


def load_tables():
    raw_path = config.RESULTS_DIR / "raw_results.csv"
    try:
        df = pd.read_csv(raw_path) if raw_path.exists() else report.write_raw()
    except Exception:
        df = report.write_raw()
    if df.empty:
        return df, report.summary_ncg(df), report.summary_class_gap(df)
    ncg_path = config.RESULTS_DIR / "summary_ncg.csv"
    gap_path = config.RESULTS_DIR / "summary_class_gap.csv"
    try:
        ncg_df, gap_df = pd.read_csv(ncg_path), pd.read_csv(gap_path)
    except Exception:
        ncg_df, gap_df = report.summary_ncg(df), report.summary_class_gap(df)
        ncg_df.to_csv(ncg_path, index=False)
        gap_df.to_csv(gap_path, index=False)
    return df, ncg_df, gap_df


def plot_ncg_vs_setting(ncg_df, out: Path):
    datasets = [d for d in config.DATASET_NAMES if d in set(ncg_df["dataset"])]
    if not datasets:
        print("[plot] no results yet")
        return None
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.4 * len(datasets), 4.6), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        sub = ncg_df[(ncg_df["dataset"] == dataset) & (ncg_df["setting"].isin(SETTINGS))]
        for method in config.FL_METHODS:
            m = sub[sub["method"] == method]
            ys, es = [], []
            for s in SETTINGS:
                row = m[m["setting"] == s]
                ys.append(float(row["ncg_mean"].iloc[0]) if len(row) else np.nan)
                es.append(float(row["ncg_std"].iloc[0]) if len(row) else 0.0)
            if all(np.isnan(ys)):
                continue
            ax.errorbar(SETTINGS, ys, yerr=es, label=method, capsize=4,
                        linewidth=1.8, markersize=6, **STYLE[method])
        ax.axhline(0.0, color=STYLE["Centralized"]["color"], linestyle="--",
                   linewidth=1.2, alpha=0.8, label="Centralized (NCG = 0)")
        ax.set_title(f"{dataset} (20 clients, CNN-2, R=50)")
        ax.set_xlabel("setting: IID  ->  alpha=0.5  ->  alpha=0.1")
        ax.set_ylabel("NCG  (0 = centralized class gap)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Normalized Class Gap relative to centralized training (mean +/- std over 3 seeds)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_b3_heatmaps(df: pd.DataFrame):
    outs = []
    for dataset in config.DATASET_NAMES:
        sub = df[(df["setting"] == "B3") & (df["dataset"] == dataset)]
        if sub.empty:
            print(f"[plot] no B3 results for {dataset}")
            continue
        methods = [m for m in config.FL_METHODS if m in set(sub["method"])]
        C = config.DATASETS[dataset]["num_classes"]
        mat = np.full((len(methods) + 1, C), np.nan)
        for i, method in enumerate(methods):
            rows = sub[sub["method"] == method]
            if rows.empty:
                continue
            vecs = [np.asarray(json.loads(v), dtype=float) for v in rows["per_class_recall_json"]]
            mat[i] = np.mean(np.vstack(vecs), axis=0)   # average recall VECTORS across seeds
        # centralized reference (one model per seed, averaged over seeds)
        cent_vecs = [np.asarray(config.load_json(config.central_path(dataset, s))["per_class_recall"])
                     for s in config.SEEDS if config.central_path(dataset, s).exists()]
        if cent_vecs:
            methods = methods + ["Centralized"]
            mat[len(methods) - 1] = np.mean(np.vstack(cent_vecs), axis=0)
        else:
            mat = mat[: len(methods)]
        fig, ax = plt.subplots(figsize=(1.0 + 0.62 * C, 0.7 * len(methods) + 1.6))
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(C))
        ax.set_xticklabels([str(c) for c in range(C)])
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.set_xlabel("class")
        ax.set_title(f"{dataset}: per-class recall, B3 (alpha=0.1), mean over 3 seeds")
        for i in range(len(methods)):
            for j in range(C):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if mat[i, j] < 0.55 else "black")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        fig.tight_layout()
        out = config.PLOTS_DIR / f"recall_heatmap_B3_{dataset}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        outs.append(out)
    return outs


def plot_class_gaps(gap_df, out: Path):
    fig, axes = plt.subplots(1, len(config.DATASET_NAMES), figsize=(12.6, 4.3), squeeze=False)
    for ax, dataset in zip(axes[0], config.DATASET_NAMES):
        sub = gap_df[(gap_df["dataset"] == dataset) & (gap_df["setting"].isin(SETTINGS))]
        if sub.empty:
            continue
        for method in config.FL_METHODS:
            m = sub[sub["method"] == method].set_index("setting").reindex(SETTINGS)
            if m["fl_class_gap_mean"].isna().all():
                continue
            ax.plot(SETTINGS, m["fl_class_gap_mean"], label=method, markersize=6,
                    linewidth=1.8, **STYLE[method])
        c = sub[sub["method"] != "Centralized"]["cent_class_gap_mean"]
        if len(c) and not c.isna().all():
            ax.axhline(float(c.mean()), color=STYLE["Centralized"]["color"], linestyle="--",
                       linewidth=1.2, alpha=0.8, label="Centralized ref (mean)")
        ax.set_title(f"{dataset}: class gap")
        ax.set_xlabel("setting")
        ax.set_ylabel("max_c recall - min_c recall")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    df, ncg_df, gap_df = load_tables()
    made = []
    p = plot_ncg_vs_setting(ncg_df, config.PLOTS_DIR / "ncg_vs_setting.png")
    if p:
        made.append(p)
    made.extend(plot_b3_heatmaps(df))
    p = plot_class_gaps(gap_df, config.PLOTS_DIR / "class_gap_vs_setting.png")
    if p:
        made.append(p)
    for p in made:
        print(f"[plot] wrote {p}")


if __name__ == "__main__":
    main()
