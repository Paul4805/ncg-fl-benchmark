"""Finalize the heterogeneous supplementary run: CSVs, manifest, plot, HETERO_NOTE.md."""
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

from benchmark_ncg.core import config, metrics, models  # noqa: E402
from benchmark_ncg.core import hetero_config as hc  # noqa: E402

STYLE = {"FedAPEN": dict(color="#ff7f0e", marker="s"),
         "PHP-FL": dict(color="#d62728", marker="D")}


def load_runs():
    out = []
    for p in sorted(hc.HET_RUNS_DIR.glob("*.json")):
        if "__SMOKE__" in p.name:
            continue
        try:
            out.append(config.load_json(p))
        except Exception:
            pass
    return out


def rows(runs):
    out = []
    for p in runs:
        rec = np.asarray(p["per_class_recall"], dtype=float)
        out.append({
            "run_id": p["run_id"],
            "setting": p["setting"],
            "method": p["method"],
            "dataset": p["dataset"],
            "seed": int(p["seed"]),
            "alpha": p.get("alpha"),
            "client_count": p.get("client_count"),
            "backbone": "heterogeneous",
            "partition_file": p.get("partition_file"),
            "per_class_recall_json": json.dumps([round(float(v), 6) for v in rec]),
            "class_gap": round(metrics.class_gap(rec), 6),
            "fl_class_gap": round(metrics.class_gap(rec), 6),
            "ncg_hetero": round(float(p["ncg_hetero"]), 6),
            "cent_class_gap_hetero": round(float(p["cent_class_gap_hetero"]), 6),
            "minority_class": metrics.minority_class(rec),
            "wall_clock_seconds": round(float(p.get("wall_clock_seconds", 0.0)), 1),
            "status": "complete",
            "notes": p.get("notes", ""),
        })
    return pd.DataFrame(out)


def main(argv=None):
    runs = load_runs()
    df = rows(runs)
    df.to_csv(config.RESULTS_DIR / "raw_results_hetero.csv", index=False)

    summ = []
    for (setting, method, dataset), g in df.groupby(["setting", "method", "dataset"], dropna=False):
        summ.append({
            "setting": setting, "method": method, "dataset": dataset,
            "ncg_hetero_mean": g["ncg_hetero"].astype(float).mean(),
            "ncg_hetero_std": g["ncg_hetero"].astype(float).std(ddof=1) if len(g) > 1 else 0.0,
            "class_gap_mean": g["class_gap"].astype(float).mean(),
            "class_gap_std": g["class_gap"].astype(float).std(ddof=1) if len(g) > 1 else 0.0,
            "num_seeds": int(g["seed"].nunique()),
        })
    sc = pd.DataFrame(summ).sort_values(["dataset", "method", "setting"])
    sc.to_csv(config.RESULTS_DIR / "summary_ncg_hetero.csv", index=False)

    gaps = []
    for (setting, method, dataset), g in df.groupby(["setting", "method", "dataset"], dropna=False):
        gaps.append({
            "setting": setting, "method": method, "dataset": dataset,
            "cent_class_gap_hetero_mean": g["cent_class_gap_hetero"].astype(float).mean(),
            "fl_class_gap_mean": g["fl_class_gap"].astype(float).mean(),
            "fl_class_gap_std": g["fl_class_gap"].astype(float).std(ddof=1) if len(g) > 1 else 0.0,
            "num_seeds": int(g["seed"].nunique()),
        })
    sg = pd.DataFrame(gaps).sort_values(["dataset", "method", "setting"])
    sg.to_csv(config.RESULTS_DIR / "summary_class_gap_hetero.csv", index=False)

    # ---- manifest --------------------------------------------------------
    fl_tasks = []
    for cfg in hc.all_hetero_runs():
        path = hc.run_path(cfg["setting"], cfg["method"], cfg["dataset"], cfg["seed"])
        fl_tasks.append({"kind": "fl", "run_id": cfg["run_id"], "setting": cfg["setting"],
                         "method": cfg["method"], "dataset": cfg["dataset"],
                         "seed": cfg["seed"], "status": "complete" if path.exists() else "pending",
                         "output": str(path)})
    ref_tasks = []
    for method in hc.HET_METHODS:
        for dataset in config.DATASET_NAMES:
            for seed in config.SEEDS:
                for aid in hc.arch_ids(method):
                    path = hc.ref_path(method, dataset, aid, seed)
                    ref_tasks.append({"kind": "centralized_ref", "run_id": path.stem,
                                      "method": method, "dataset": dataset, "seed": int(seed),
                                      "arch_id": aid,
                                      "status": "complete" if path.exists() else "pending",
                                      "output": str(path)})
    manifest = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scope": {"FedAPEN": hc.HET_SETTINGS["FedAPEN"], "PHP-FL": hc.HET_SETTINGS["PHP-FL"]},
        "rounds": int(config.TRAINING["rounds"]),
        "local_epochs": int(config.TRAINING["local_epochs"]),
        "centralized_max_epochs": int(config.CENTRAL_TRAINING["max_epochs"]),
        "centralized_ref_mode": "per_architecture",
        "totals": {
            "fl": len(fl_tasks), "fl_complete": sum(t["status"] == "complete" for t in fl_tasks),
            "refs": len(ref_tasks), "refs_complete": sum(t["status"] == "complete" for t in ref_tasks),
        },
        "fl_tasks": fl_tasks, "ref_tasks": ref_tasks,
    }
    config.dump_json(manifest, config.RESULTS_DIR / "manifest_hetero.json")
    pd.DataFrame(fl_tasks + ref_tasks).to_csv(config.RESULTS_DIR / "manifest_hetero.csv", index=False)

    # ---- plot ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    settings = ["B1", "B2", "B3"]
    for dataset in config.DATASET_NAMES:
        for method in hc.HET_METHODS:
            sub = sc[(sc.dataset == dataset) & (sc.method == method)].set_index("setting")
            ys, es = [], []
            for s in settings:
                if s in sub.index:
                    ys.append(float(sub.loc[s, "ncg_hetero_mean"]))
                    es.append(float(sub.loc[s, "ncg_hetero_std"]))
                else:
                    ys.append(np.nan); es.append(0.0)
            style = dict(STYLE[method])
            style["linestyle"] = "-" if dataset == "BloodMNIST" else "--"
            ax.errorbar(settings, ys, yerr=es, label=f"{method} / {dataset}",
                        capsize=3, linewidth=1.7, markersize=6, **style)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title("Heterogeneous supplementary run: NCG vs setting (per-architecture refs)")
    ax.set_xlabel("setting (IID -> alpha=0.5 -> alpha=0.1); PHP-FL B1 not run under hetero")
    ax.set_ylabel("NCG_hetero")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(hc.HET_PLOTS_DIR / "ncg_vs_setting_hetero.png", dpi=180)
    plt.close(fig)

    # ---- HETERO_NOTE.md -------------------------------------------------
    hom = pd.read_csv(config.RESULTS_DIR / "summary_ncg.csv") if (config.RESULTS_DIR / "summary_ncg.csv").exists() else pd.DataFrame()
    arch_map = config.load_json(config.PARTITION_DIR / "hetero_arch_map.json")

    def hom_ncg(dataset, method, setting):
        if hom.empty:
            return float("nan"), float("nan")
        r = hom[(hom.dataset == dataset) & (hom.method == method) & (hom.setting == setting)]
        return (float(r["ncg_mean"].iloc[0]), float(r["ncg_std"].iloc[0])) if len(r) else (float("nan"), float("nan"))

    def het_ncg(dataset, method, setting):
        r = sc[(sc.dataset == dataset) & (sc.method == method) & (sc.setting == setting)]
        return (float(r["ncg_hetero_mean"].iloc[0]), float(r["ncg_hetero_std"].iloc[0])) if len(r) else (float("nan"), float("nan"))

    lines = ["# HETERO NOTE (supplementary, separate from the R=50 main table)", "",
             "## Scale caveat (read first)", "",
             "Due to the 12-hour compute budget, PHP-FL's heterogeneous client architectures "
             "were compressed from the paper's HtM4 set (googlenet 5.6M, densenet121 7.0M, "
             "efficientnet-b1 6.5M, resnet18 11.2M) to four micro variants (19k-29k params). "
             "Architecture family diversity (Inception / Dense / MBConv / Residual) and "
             "round-robin assignment are preserved; absolute model scale is inverted relative "
             "to the homogeneous CNN2 baseline (substitutes are 0.43-0.65x CNN2, paper models "
             "are 127-254x CNN2). This heterogeneous experiment therefore tests whether "
             "architecture-family diversity alone changes the NCG conclusions. It does not "
             "test whether the paper's larger models would. A positive result (hetero NCG "
             "within std of homogeneous NCG) is a necessary but not sufficient condition for "
             "the paper-scale claim; a negative result (hetero NCG notably different) directly "
             "weakens the generality of the homogeneous conclusions and should be reported as "
             "such.", ""]

    cnn2_params = int(sum(p.numel() for p in models.build_model("cnn2", 3, 8).parameters()))
    paper = {
        "googlenet_micro2": ("googlenet", 5_604_008),
        "densenet_micro2": ("densenet121", 6_950_280),
        "efficientnet_micro2": ("efficientnet-b1", 6_517_288),
        "resnet4_micro": ("resnet18", 11_180_616),
    }
    lines += ["## Architecture scale table", "",
              f"CNN2 reference: {cnn2_params:,} parameters (BloodMNIST shape).", "",
              "| method | paper_backbone | paper_params | substitute | substitute_params | "
              "size_ratio_vs_CNN2_paper | size_ratio_vs_CNN2_substitute |",
              "|---|---|---|---|---|---|---|"]
    fa_archs = arch_map["FedAPEN"]["architectures"]
    lines.append(f"| FedAPEN | CNN2_BN width-heterogeneous (official) | "
                 f"{min(a['params'] for a in fa_archs):,}-{max(a['params'] for a in fa_archs):,} | "
                 f"none (official widths used) | same | "
                 f"{min(a['params'] for a in fa_archs) / cnn2_params:.1f}-"
                 f"{max(a['params'] for a in fa_archs) / cnn2_params:.1f}x | same |")
    for a in arch_map["PHP-FL"]["architectures"]:
        pb, pp = paper.get(a["class_name"], ("n/a", 0))
        if pp:
            lines.append(f"| PHP-FL | {pb} | {pp:,} | {a['class_name']} | {a['params']:,} | "
                         f"{pp / cnn2_params:.1f}x | {a['params'] / cnn2_params:.2f}x |")
    lines.append("")

    present = {m: [s for s in ("B1", "B2", "B3")
                   if not df[(df.method == m) & (df.setting == s)].empty] for m in hc.HET_METHODS}
    scope_bits = []
    for m in hc.HET_METHODS:
        label = "full official" if m == "FedAPEN" else "with substitutes per DEVIATIONS.md"
        scope_bits.append(f"{m} ({', '.join(present[m]) or 'none'}; {label})")
    lines += ["## Scope", "",
              "Scope: " + "; ".join(scope_bits) + ". 20 clients, R=50, E=5, seeds {0,1,2}, "
              "per-architecture centralized references.", ""]

    lines += ["## Architectures", "", "**FedAPEN** (official `--he 1`, 20 distinct CNN2_BN widths):", ""]
    for a in arch_map["FedAPEN"]["architectures"][:3]:
        lines.append(f"- arch {a['id']}: {a['class_name']} d={a['channel_diff']:+d}, "
                     f"{a['params']:,} params ({a['source']})")
    lines.append(f"- ... plus 17 more widths down to arch 19 (d=-29); "
                 f"all <= {max(a['params'] for a in arch_map['FedAPEN']['architectures']):,} params")
    lines += ["", "**PHP-FL** (approved substitutes for the official HtM4 family):", ""]
    for a in arch_map["PHP-FL"]["architectures"]:
        lines.append(f"- arch {a['id']}: {a['class_name']} ({a['family']}), {a['params']:,} params "
                     f"- {a['source']}")
    lines += ["", f"Client assignment: FedAPEN cid -> cid; PHP-FL cid -> cid % 4.", ""]

    lines += ["## NCG_hetero (mean +/- std over 3 seeds)", "",
              "| dataset | method | B1 | B2 | B3 |", "|---|---|---|---|---|"]
    for dataset in config.DATASET_NAMES:
        for method in hc.HET_METHODS:
            cells = []
            for s in ["B1", "B2", "B3"]:
                m, sd = het_ncg(dataset, method, s)
                cells.append(f"{m:+.3f} +/- {sd:.3f}" if m == m else "not run")
            lines.append(f"| {dataset} | {method} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")

    lines += ["## Side-by-side: homogeneous (main table) vs heterogeneous", "",
              "| dataset | method | setting | homogeneous NCG | hetero NCG | delta (het - hom) |",
              "|---|---|---|---|---|---|"]
    changed = []
    for dataset in config.DATASET_NAMES:
        for method in hc.HET_METHODS:
            for s in hc.HET_SETTINGS[method]:
                hm, hs = hom_ncg(dataset, method, s)
                tm, ts = het_ncg(dataset, method, s)
                d = tm - hm
                if d == d and abs(d) > 0.05:
                    changed.append((dataset, method, s, hm, tm, d))
                lines.append(f"| {dataset} | {method} | {s} | {hm:+.3f} +/- {hs:.3f} | "
                             f"{tm:+.3f} +/- {ts:.3f} | {d:+.3f} |")
    lines.append("")

    # verdict
    fa_deltas = {}
    for dataset in config.DATASET_NAMES:
        b1m, _ = het_ncg(dataset, "FedAPEN", "B1")
        b3m, _ = het_ncg(dataset, "FedAPEN", "B3")
        fa_deltas[dataset] = (b3m - b1m) if (b1m == b1m and b3m == b3m) else float("nan")
    fa_pos = [d for d in fa_deltas.values() if d == d and d > 0]
    fa_verdict = ("survives" if len(fa_pos) == 2 else
                  "partially survives" if len(fa_pos) == 1 else "does not survive")

    php_close = []
    for dataset in config.DATASET_NAMES:
        for s in present["PHP-FL"]:
            hm, _ = hom_ncg(dataset, "PHP-FL", s)
            tm, _ = het_ncg(dataset, "PHP-FL", s)
            if hm == hm and tm == tm:
                php_close.append(abs(tm - hm) <= 0.10)
    php_verdict = ("survives" if php_close and all(php_close) else
                   "partially survives" if any(php_close) else "does not survive")

    lines += ["## Verdict", "",
              f"Under architecture-family diversity at micro scale, FedAPEN's NCG "
              f"{fa_verdict} the homogeneous conclusion from the R=50 main table.",
              "",
              f"Under architecture-family diversity at micro scale, PHP-FL's NCG "
              f"{php_verdict} the homogeneous conclusion from the R=50 main table.",
              ""]
    if changed:
        cells = "; ".join(f"{ds} {m} {s}: {hm:+.3f} -> {tm:+.3f}" for ds, m, s, hm, tm, _ in changed)
        lines += [f"Cells whose NCG changed by more than 0.05: {cells}.", ""]
    else:
        lines += ["No cell changed by more than 0.05.", ""]
    lines += ["## Caveats", ""]
    if "B1" not in present["PHP-FL"]:
        lines.append("- PHP-FL B1 (IID) was not run under hetero architectures.")
    lines += ["- PHP-FL uses 19k-29k parameter substitutes (0.02-0.03M); absolute sizes are "
              "compressed from the official 5.6-11.2M HtM4 family (family diversity and "
              "round-robin assignment preserved).",
              "- Main-table (homogeneous) results are unchanged; heterogeneous results live in a "
              "separate namespace and are never mixed into the main CSVs."]
    php_b2_pending = [t for t in manifest["fl_tasks"]
                      if t["method"] == "PHP-FL" and t["setting"] == "B2"
                      and t["status"] != "complete"]
    if php_b2_pending:
        lines.append("- PHP-FL B2 was not completed within the 12-hour budget. Hetero "
                     "conclusions for PHP-FL are B3-only.")
    (config.BENCH_ROOT / "HETERO_NOTE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[hetero] runs={len(df)} | refs complete={manifest['totals']['refs_complete']}/"
          f"{manifest['totals']['refs']} | FL complete={manifest['totals']['fl_complete']}/"
          f"{manifest['totals']['fl']}")
    print(f"[hetero] wrote summaries, manifest_hetero, plot and HETERO_NOTE.md")


if __name__ == "__main__":
    main()
