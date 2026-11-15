"""Finalize the heterogeneous block: tagged summaries + accuracy merge + bar chart + note.

Runs after the B3 (and optionally B2-FedAPEN) queue writes results/runs_hetero/*.json.
Derives (data-driven: whichever settings are present):
  results/summary_ncg_hetero_B3.csv          (or _B2B3.csv when B2 rows exist)
  results/summary_class_gap_hetero_B3.csv    (or _B2B3.csv)
  results/plots_hetero/b3_homogeneous_vs_hetero.png
  results/HETERO_NOTE_B3.md                  (or HETERO_NOTE_FINAL.md)
Accuracy columns are merged from results/summary_accuracy_hetero.csv (per-client
mean for FL, per-architecture mean ref for centralized). Run JSONs are untouched.
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


def load_runs():
    rows = []
    for p in sorted(hc.HET_RUNS_DIR.glob("*.json")):
        if "__SMOKE__" in p.name:
            continue
        d = config.load_json(p)
        rows.append({
            "setting": d["setting"], "method": d["method"], "dataset": d["dataset"],
            "seed": int(d["seed"]), "ncg_hetero": float(d["ncg_hetero"]),
            "fl_class_gap": float(d["fl_class_gap"]),
            "cent_class_gap_hetero": float(d["cent_class_gap_hetero"]),
            "test_accuracy": float(d["test_accuracy"]),
            "per_class_recall": d["per_class_recall"],
        })
    return pd.DataFrame(rows)


def main():
    df = load_runs()
    if df.empty:
        raise SystemExit("no hetero runs found")
    settings = sorted(df["setting"].unique())
    tag = "B3" if settings == ["B3"] else "B2B3"

    rows = []
    for (setting, method, dataset), g in df.groupby(["setting", "method", "dataset"]):
        rows.append({
            "setting": setting, "method": method, "dataset": dataset,
            "ncg_hetero_mean": g["ncg_hetero"].mean(),
            "ncg_hetero_std": g["ncg_hetero"].std(ddof=1) if len(g) > 1 else 0.0,
            "class_gap_mean": g["fl_class_gap"].mean(),
            "class_gap_std": g["fl_class_gap"].std(ddof=1) if len(g) > 1 else 0.0,
            "num_seeds": int(g["seed"].nunique()),
        })
    summ = pd.DataFrame(rows).sort_values(["dataset", "method", "setting"])

    acc_path = config.RESULTS_DIR / "summary_accuracy_hetero.csv"
    if acc_path.exists():
        acc = pd.read_csv(acc_path)
        keep = [c for c in ["setting", "method", "dataset", "acc_mean", "acc_std",
                            "bal_acc_mean", "bal_acc_std", "cent_acc_mean",
                            "cent_acc_std", "cent_bal_acc_mean", "cent_bal_acc_std",
                            "acc_gap", "bal_acc_gap", "ncg_hetero_mean",
                            "acc_ncg_regime"] if c in acc.columns]
        summ = summ.merge(acc[keep], on=["setting", "method", "dataset"], how="left",
                          suffixes=("", "_acc"))
        if "ncg_hetero_mean_acc" in summ.columns:
            summ = summ.drop(columns=["ncg_hetero_mean_acc"])
    summ.to_csv(config.RESULTS_DIR / f"summary_ncg_hetero_{tag}.csv", index=False)

    gaps = []
    for (setting, method, dataset), g in df.groupby(["setting", "method", "dataset"]):
        gaps.append({
            "setting": setting, "method": method, "dataset": dataset,
            "cent_class_gap_hetero_mean": g["cent_class_gap_hetero"].mean(),
            "fl_class_gap_mean": g["fl_class_gap"].mean(),
            "fl_class_gap_std": g["fl_class_gap"].std(ddof=1) if len(g) > 1 else 0.0,
            "num_seeds": int(g["seed"].nunique()),
        })
    pd.DataFrame(gaps).sort_values(["dataset", "method", "setting"]).to_csv(
        config.RESULTS_DIR / f"summary_class_gap_hetero_{tag}.csv", index=False)

    # ---- B3 bar chart: homogeneous vs heterogeneous ----------------------
    b3 = summ[summ["setting"] == "B3"]
    if not b3.empty:
        hom = pd.read_csv(config.RESULTS_DIR / "summary_ncg.csv")
        methods = [m for m in hc.HET_METHODS if m in set(b3["method"])]
        datasets = list(config.DATASET_NAMES)
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        width = 0.36
        xs = np.arange(len(datasets))
        for i, method in enumerate(methods):
            hv, he, hs = [], [], []
            for ds in datasets:
                h = hom[(hom.dataset == ds) & (hom.method == method) & (hom.setting == "B3")]
                t = b3[(b3.dataset == ds) & (b3.method == method)]
                hv.append(float(t["ncg_hetero_mean"].iloc[0]) if len(t) else np.nan)
                he.append(float(h["ncg_mean"].iloc[0]) if len(h) else np.nan)
                hs.append(float(t["ncg_hetero_std"].iloc[0]) if len(t) else 0.0)
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

    # ---- note ------------------------------------------------------------
    base = config.BENCH_ROOT / "HETERO_NOTE.md"
    text = base.read_text(encoding="utf-8") if base.exists() else "# HETERO NOTE\n"
    acc_lines = ["", f"## Accuracy ({tag} hetero, mean +/- std over seeds)", ""]
    if "acc_mean" in summ.columns:
        acc_lines += ["| setting | dataset | method | top-1 acc | balanced acc |",
                      "|---|---|---|---|---|"]
        for _, r in summ.iterrows():
            acc_lines.append(f"| {r['setting']} | {r['dataset']} | {r['method']} | "
                             f"{r['acc_mean']:.4f} +/- {r['acc_std']:.4f} | "
                             f"{r['bal_acc_mean']:.4f} +/- {r['bal_acc_std']:.4f} |")

    # ---- per-seed PHP-FL Fashion B3 -------------------------------------
    sub = df[(df["method"] == "PHP-FL") & (df["dataset"] == "FashionMNIST") &
             (df["setting"] == "B3")]
    if not sub.empty:
        acc_lines += ["", "## Per-seed PHP-FL Fashion B3 (hetero)", "",
                      "| seed | NCG_hetero | top-1 acc | FL class gap | minority class | "
                      "minority recall | classes with recall < 0.05 |",
                      "|---|---|---|---|---|---|---|"]
        neg_rows = []
        for _, r in sub.sort_values("seed").iterrows():
            rec = np.asarray(r["per_class_recall"], dtype=float)
            low = [(int(i), round(float(v), 4)) for i, v in enumerate(rec) if v < 0.05]
            if r["ncg_hetero"] < 0:
                neg_rows.append((int(r["seed"]), r, rec))
            acc_lines.append(
                f"| {int(r['seed'])} | {r['ncg_hetero']:+.4f} | {r['test_accuracy']:.4f} | "
                f"{r['fl_class_gap']:.4f} | {int(rec.argmin())} | {rec.min():.4f} | "
                f"{low if low else 'none'} |")
        sd = float(sub["ncg_hetero"].std(ddof=1)) if len(sub) > 1 else 0.0
        if neg_rows:
            acc_lines += ["", "### Negative-NCG seeds", ""]
            for seed, r, rec in neg_rows:
                acc_lines += [
                    f"Seed {seed} of PHP-FL Fashion B3 achieved NCG = {r['ncg_hetero']:+.4f}, "
                    f"i.e. FL class gap ({r['fl_class_gap']:.4f}) was smaller than the centralized "
                    f"mean-reference gap ({r['cent_class_gap_hetero']:.4f}). Per-class recalls: "
                    f"{[round(float(v), 3) for v in rec]} (minority class {int(rec.argmin())}, "
                    f"recall {rec.min():.4f}). The negative value coincides with uniformly low "
                    f"accuracy ({r['test_accuracy']:.4f}) and reflects the uniform-failure regime "
                    f"already described, not a fairness improvement: a model that predicts "
                    f"near-uniformly across classes will have a small per-class recall spread "
                    f"irrespective of accuracy. Report alongside the mean "
                    f"({sub['ncg_hetero'].mean():+.3f}) to avoid the impression that PHP-FL is "
                    f"'fair' on Fashion.", ""]
        else:
            acc_lines += [
                "", f"PHP-FL Fashion B3 hetero NCG ranged from {sub['ncg_hetero'].min():+.3f} to "
                f"{sub['ncg_hetero'].max():+.3f} across seeds (mean {sub['ncg_hetero'].mean():.3f}, "
                f"std {sd:.3f}), with the width attributable to per-class recall variation across "
                f"Dirichlet partitions rather than to accuracy variance (accuracy std was only "
                f"{sub['test_accuracy'].std(ddof=1):.3f}).", ""]
    acc_lines += ["", "## Limitation (budget)", "",
                  "Heterogeneous results cover "
                  + ", ".join(f"{m} {sorted(df[df['method'] == m]['setting'].unique())}"
                              for m in sorted(df["method"].unique()))
                  + ". B1 hetero was not evaluated in this block. The homogeneous R=50 "
                    "main table remains the primary result; this block is a reference-"
                    "sensitivity check under real heterogeneous client architectures.", ""]
    out = config.RESULTS_DIR / (f"HETERO_NOTE_FINAL.md" if tag == "B2B3" else "HETERO_NOTE_B3.md")
    out.write_text(text.rstrip() + "\n" + "\n".join(acc_lines), encoding="utf-8")
    for old in [config.RESULTS_DIR / "HETERO_NOTE_B3.md", config.RESULTS_DIR / "HETERO_NOTE_FINAL.md"]:
        if old != out and old.exists():
            old.unlink()
    if base.exists():
        base.unlink()
    print(f"[finalize] settings={settings} tag={tag} rows={len(summ)} -> {out.name}")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
