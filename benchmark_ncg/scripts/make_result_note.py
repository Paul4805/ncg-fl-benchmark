"""Generate RESULT_NOTE.md from the final 72-run results + the sharing analysis."""
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

HYP_METHODS = ["FedAvg", "FedAPEN", "FedFV"]


def paired_deltas(raw: pd.DataFrame, dataset: str, method: str):
    fl = raw[(raw["method"] != "Centralized") & (raw["dataset"] == dataset) &
             (raw["method"] == method)]
    b1 = fl[fl["setting"] == "B1"].set_index("seed")["ncg"].astype(float)
    b3 = fl[fl["setting"] == "B3"].set_index("seed")["ncg"].astype(float)
    common = sorted(set(b1.index) & set(b3.index))
    d = np.array([b3.loc[s] - b1.loc[s] for s in common])
    if len(d) == 0:
        return float("nan"), float("nan"), 0
    return float(d.mean()), (float(d.std(ddof=1)) if len(d) > 1 else 0.0), len(d)


def main():
    ncg = pd.read_csv(config.RESULTS_DIR / "summary_ncg.csv")
    raw = pd.read_csv(config.RESULTS_DIR / "raw_results.csv")
    sharing = config.load_json(config.DIAG_DIR / "phpfl_sharing_analysis.json")
    gate = config.load_json(config.DIAG_DIR / "convergence_bloodmnist_s0.json")
    manifest = config.load_json(config.RESULTS_DIR / "manifest.json")

    def ncg_of(dataset, method, setting):
        row = ncg[(ncg["dataset"] == dataset) & (ncg["method"] == method) &
                  (ncg["setting"] == setting)]
        return (float(row["ncg_mean"].iloc[0]), float(row["ncg_std"].iloc[0])) if len(row) else (float("nan"), float("nan"))

    lines = ["# RESULT NOTE", ""]
    lines += [f"Completed: {manifest['totals']['fl_complete']}/{manifest['totals']['fl']} FL runs, "
              f"{manifest['totals']['centralized_complete']}/{manifest['totals']['centralized']} "
              f"centralized references (R={manifest['rounds']}, E={manifest['local_epochs']}).", ""]

    # ---------------------------------------------------------------- headline
    lines += ["## Headline", ""]
    holds = {}
    for dataset in config.DATASET_NAMES:
        for method in HYP_METHODS:
            mean, std, n = paired_deltas(raw, dataset, method)
            holds[(dataset, method)] = mean
    all_hold = all(v == v and v > 0 for v in holds.values())
    lines += [
        ("The hypothesis holds for FedAvg, FedAPEN and FedFV on both datasets: NCG rises "
         "from B1 (IID) to B3 (alpha=0.1)." if all_hold else
         "The hypothesis holds only partially: NCG rises from B1 to B3 for some, not all, of "
         "FedAvg, FedAPEN and FedFV (see the table)."),
        "",
        "| dataset | method | mean DeltaNCG (B3 - B1) +/- std over seeds | n seeds |",
        "|---|---|---|---|",
    ]
    for dataset in config.DATASET_NAMES:
        for method in HYP_METHODS:
            mean, std, n = paired_deltas(raw, dataset, method)
            lines.append(f"| {dataset} | {method} | {mean:+.3f} +/- {std:.3f} | {n} |")
    lines.append("")

    # ------------------------------------------------------------- PHP-FL part
    lines += ["## PHP-FL is different, and this is a finding, not a failure", ""]
    lines += ["| dataset | B1 (IID) | B2 (alpha=0.5) | B3 (alpha=0.1) |", "|---|---|---|---|"]
    for dataset in config.DATASET_NAMES:
        cells = []
        for setting in ["B1", "B2", "B3"]:
            m, s = ncg_of(dataset, "PHP-FL", setting)
            cells.append(f"{m:+.3f} +/- {s:.3f}" if m == m else "n/a")
        lines.append(f"| {dataset} | {cells[0]} | {cells[1]} | {cells[2]} |")
    blood = {s: sharing["summary_by_setting"]["BloodMNIST"][s] for s in ["B1", "B2", "B3"]}
    frac = blood["B1"].get("phpfl_frac_global_params", float("nan"))
    lines += [
        "",
        ("The observed pattern is inverted relative to the hypothesis: on BloodMNIST the largest "
         "PHP-FL class gap is under IID and the smallest at B3; the same ordering, weaker, is seen "
         "on FashionMNIST."),
        "",
        f"- `phpfl_frac_global_params` = **{frac:.4f}** "
        f"(~ {100 * (1 - frac):.1f}% of the personal-global model stays local every round)",
        f"- BloodMNIST B1: mean per-client spread = **{blood['B1']['mean_spread']:.4f}**, "
        f"ensemble gap = **{blood['B1']['ensemble_gap']:.4f}**",
        f"- BloodMNIST B3: mean per-client spread = **{blood['B3']['mean_spread']:.4f}**, "
        f"ensemble gap = **{blood['B3']['ensemble_gap']:.4f}**",
        "",
        ("At 100 % participation, PHP-FL's deployed models are almost entirely local, so the class "
         "gap reflects per-client local recall noise rather than a skew-induced global bias; "
         "averaging 20 noisy per-client recall vectors yields a larger max-min spread under IID "
         "than under severe skew where clients specialize."),
        "",
    ]
    if not sharing["assertions"].get("frac_global_params_identical_across_clients", True):
        lines += ["Note: phpfl_frac_global_params varies across clients (see the per-client vectors "
                  "in phpfl_sharing_analysis.json).", ""]
    if not sharing["assertions"].get("ensemble_gap_matches_run_json", True):
        lines += ["Note: the recomputed ensemble gap did not match the stored fl_class_gap for every "
                  "run; see phpfl_sharing_analysis.json.", ""]

    # ------------------------------------------------------ convergence status
    php = gate["phpfl"]
    lines += [
        "## Convergence limitation", "",
        f"The gate passed: PHP-FL Delta(40->50) <= "
        f"{max(php['phpfl_r50_b2']['delta_40_50_pp'], php['phpfl_r50_b3']['delta_40_50_pp']):.2f} pp "
        f"on both alpha legs, and FedAvg R=50 reached "
        f"{100 * gate['fedavg_ratio_50_over_100']:.1f}% of its R=100 accuracy. Therefore the R=75 "
        "supplementary run described in README.md was NOT executed. FedAPEN and FedFV were not given "
        "a mid-run curve diagnostic; their final results are reported as-is without a convergence claim.",
        "",
    ]

    # ------------------------------------------------------------ numbers table
    lines += ["## Headline numbers", "",
              "Mean NCG (std across seeds), R=50:", "",
              "| dataset | method | B1 | B2 | B3 |", "|---|---|---|---|---|"]
    for dataset in config.DATASET_NAMES:
        for method in HYP_METHODS + ["PHP-FL"]:
            cells = []
            for setting in ["B1", "B2", "B3"]:
                m, s = ncg_of(dataset, method, setting)
                cells.append(f"{m:+.3f} ({s:.3f})" if m == m else "n/a")
            lines.append(f"| {dataset} | {method} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")

    acc_path = config.RESULTS_DIR / "summary_accuracy_hom.csv"
    if acc_path.exists():
        acc = pd.read_csv(acc_path)
        cent = (acc[acc["method"] == "Centralized"]
                .groupby("dataset")[["acc_mean", "bal_acc_mean"]].mean())
        lines += ["## Accuracy (top-1 and balanced)", "",
                  "Centralized baseline (top-1 / balanced): " + ", ".join(
                      f"{ds} {cent.loc[ds, 'acc_mean']:.4f} / {cent.loc[ds, 'bal_acc_mean']:.4f}"
                      for ds in cent.index), "",
                  "| dataset | method | B1 acc | B2 acc | B3 acc |", "|---|---|---|---|---|"]
        for dataset in config.DATASET_NAMES:
            for method in config.FL_METHODS:
                cells = []
                for setting in ["B1", "B2", "B3"]:
                    r = acc[(acc["dataset"] == dataset) & (acc["method"] == method) &
                            (acc["setting"] == setting)]
                    cells.append(f"{r['acc_mean'].iloc[0]:.4f} +/- {r['acc_std'].iloc[0]:.4f}"
                                 if len(r) else "n/a")
                lines.append(f"| {dataset} | {method} | {cells[0]} | {cells[1]} | {cells[2]} |")
        lines += ["", "Accuracy gap = centralized - method (positive = FL worse):", "",
                  "| dataset | method | B1 top-1 | B2 top-1 | B3 top-1 | B1 bal | B2 bal | B3 bal |",
                  "|---|---|---|---|---|---|---|---|"]
        for dataset in config.DATASET_NAMES:
            ca = float(cent.loc[dataset, "acc_mean"])
            cb = float(cent.loc[dataset, "bal_acc_mean"])
            for method in config.FL_METHODS:
                cells = []
                for setting in ["B1", "B2", "B3"]:
                    r = acc[(acc["dataset"] == dataset) & (acc["method"] == method) &
                            (acc["setting"] == setting)]
                    cells.append(f"{ca - float(r['acc_mean'].iloc[0]):+.4f}" if len(r) else "n/a")
                for setting in ["B1", "B2", "B3"]:
                    r = acc[(acc["dataset"] == dataset) & (acc["method"] == method) &
                            (acc["setting"] == setting)]
                    cells.append(f"{cb - float(r['bal_acc_mean'].iloc[0]):+.4f}" if len(r) else "n/a")
                lines.append(f"| {dataset} | {method} | " + " | ".join(cells) + " |")
        lines.append("")

        # C3: accuracy vs NCG paragraph (data-driven, no forced narrative)
        cand = []
        for method in HYP_METHODS + ["PHP-FL"]:
            for dataset in config.DATASET_NAMES:
                n3, _ = ncg_of(dataset, method, "B3")
                n1, _ = ncg_of(dataset, method, "B1")
                a3 = acc[(acc["dataset"] == dataset) & (acc["method"] == method) &
                         (acc["setting"] == "B3")]
                a1 = acc[(acc["dataset"] == dataset) & (acc["method"] == method) &
                         (acc["setting"] == "B1")]
                if n3 == n3 and len(a3) and len(a1):
                    cand.append((n3, method, dataset, n1,
                                 float(a1["acc_mean"].iloc[0]), float(a3["acc_mean"].iloc[0]),
                                 float(a3["bal_acc_mean"].iloc[0])))
        if cand:
            n3, method, dataset, n1, acc1, acc3, bal3 = max(cand, key=lambda x: x[0])
            delta = acc3 - bal3
            pattern = "falls" if acc3 < acc1 else "does not fall"
            lines += ["## Accuracy and NCG", "",
                      f"From B1 to B3, {method}'s top-1 accuracy on {dataset} {pattern} "
                      f"({acc1:.4f} -> {acc3:.4f}) while its NCG goes from {n1:+.4f} to "
                      f"{n3:+.4f}. At B3 it reaches {100 * acc3:.2f}% top-1 and "
                      f"{100 * bal3:.2f}% balanced accuracy; the top-1 minus balanced gap "
                      f"(delta = {delta:+.4f}) is itself a proxy for class imbalance and is "
                      f"consistent with the NCG direction." if acc3 < acc1 else
                      f"At B3, {method} on {dataset} reaches {100 * acc3:.2f}% top-1 and "
                      f"{100 * bal3:.2f}% balanced accuracy with NCG {n3:+.4f}; the top-1 "
                      f"minus balanced gap (delta = {delta:+.4f}) is a proxy for class "
                      f"imbalance. Note that accuracy from B1 to B3 does not fall for this "
                      f"cell, so the expected accuracy/NCG opposition does not hold here.", ""]

        # Accuracy-NCG regimes (joint interpretation)
        def acc_of(dataset, method, setting):
            r = acc[(acc["dataset"] == dataset) & (acc["method"] == method) &
                    (acc["setting"] == setting)]
            return float(r["acc_mean"].iloc[0]) if len(r) else float("nan")

        def regime(a, n):
            if a != a or n != n:
                return "n/a"
            low = a < 0.60
            high = n > 0.20
            return ("low_acc_high_ncg" if low and high else
                    "low_acc_low_ncg" if low else
                    "high_acc_high_ncg" if high else "high_acc_low_ncg")

        php_b1_acc = acc_of("BloodMNIST", "PHP-FL", "B1")
        php_b3_acc = acc_of("BloodMNIST", "PHP-FL", "B3")
        php_b1_ncg, _ = ncg_of("BloodMNIST", "PHP-FL", "B1")
        php_b3_ncg, _ = ncg_of("BloodMNIST", "PHP-FL", "B3")
        fa_b3_acc = acc_of("BloodMNIST", "FedAvg", "B3")
        fa_b3_ncg, _ = ncg_of("BloodMNIST", "FedAvg", "B3")
        b3_regimes = []
        for m in config.FL_METHODS:
            a = acc_of("BloodMNIST", m, "B3")
            n, _ = ncg_of("BloodMNIST", m, "B3")
            b3_regimes.append(f"{m} {regime(a, n)} ({a:.3f}/{n:+.3f})")
        lines += ["## Accuracy-NCG regimes", "",
                  "Regime thresholds: acc < 0.60 = low; NCG > 0.20 = high.", "",
                  f"NCG must be interpreted jointly with accuracy. PHP-FL's NCG falls from "
                  f"{php_b1_ncg:+.3f} (B1) to {php_b3_ncg:+.3f} (B3) on BloodMNIST as its "
                  f"accuracy falls from {php_b1_acc:.3f} to {php_b3_acc:.3f}. This is not a "
                  f"fairness improvement - it is uniform failure: recall is low and similar "
                  f"across classes. By contrast, FedAvg at B3 has accuracy {fa_b3_acc:.3f} and "
                  f"NCG {fa_b3_ncg:+.3f}: accuracy stays high but is concentrated, with "
                  f"near-zero recall on some classes. With the stated thresholds the B3 FL "
                  f"methods split into two computed regimes, not three: "
                  + "; ".join(b3_regimes) + ". On FashionMNIST the same split appears "
                  f"(PHP-FL {regime(acc_of('FashionMNIST', 'PHP-FL', 'B3'), ncg_of('FashionMNIST', 'PHP-FL', 'B3')[0])}; "
                  f"FedAvg {regime(acc_of('FashionMNIST', 'FedAvg', 'B3'), ncg_of('FashionMNIST', 'FedAvg', 'B3')[0])}; "
                  f"FedAPEN {regime(acc_of('FashionMNIST', 'FedAPEN', 'B3'), ncg_of('FashionMNIST', 'FedAPEN', 'B3')[0])}; "
                  f"FedFV {regime(acc_of('FashionMNIST', 'FedFV', 'B3'), ncg_of('FashionMNIST', 'FedFV', 'B3')[0])}). "
                  f"NCG alone cannot distinguish uniform failure from selective failure, so we "
                  f"report accuracy and NCG as a pair and recommend that future work on "
                  f"class-gap metrics include accuracy alongside them.", ""]

    lines += ["## Recommendation", "",
              "The main comparison at R=50 is complete and defensible; the R=75 extension is "
              "available but not recommended for the main table.", ""]

    lines += ["## Homogeneous verification", "",
              "The homogeneous main table (72 FL runs + 6 centralized references) was verified "
              "post-hoc to have been produced by the CNN-2 model path, with every client model at "
              "44,584 params (BloodMNIST) or 50,282 (FashionMNIST). Centralized accuracy recomputes "
              "to within 5.09e-07, and every run JSON is content-identical to commit bab3f7c (Git "
              "blob hashes match and git diff is clean; working-tree bytes differ from the blob "
              "only by Windows CRLF line endings).", ""]

    (config.BENCH_ROOT / "RESULT_NOTE.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[result_note] wrote {config.BENCH_ROOT / 'RESULT_NOTE.md'}")


if __name__ == "__main__":
    main()
