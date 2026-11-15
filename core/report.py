"""Result collection: raw_results.csv, summaries, manifest.

NCG is always computed from per-class recall VECTORS.  The centralized side is
the single CNN-2 reference of the same (dataset, seed); the FL side is the final
global model (FedAvg/FedFV) or the mean of the deployed client-model recall
vectors (FedAPEN/PHP-FL).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, metrics

RAW_COLUMNS = [
    "run_id", "setting", "method", "dataset", "seed", "alpha", "client_count", "backbone",
    "partition_file", "fashionmnist_subsample_file", "per_class_recall_json", "class_gap",
    "ncg", "cent_class_gap", "fl_class_gap", "minority_class", "wall_clock_seconds",
    "status", "notes",
]
SETTING_ORDER = ["B1", "B2", "B3"]
METHOD_ORDER = ["FedAvg", "FedAPEN", "FedFV", "PHP-FL", "Centralized"]


def load_runs() -> list[dict]:
    out = []
    for path in sorted(config.RUNS_DIR.glob("*.json")):
        try:
            out.append(config.load_json(path))
        except Exception as exc:
            print(f"[report] unreadable {path.name}: {exc}")
    return out


def load_central(dataset: str, seed: int):
    path = config.central_path(dataset, seed)
    return config.load_json(path) if path.exists() else None


def _fl_row(payload: dict) -> dict:
    cent = load_central(payload["dataset"], int(payload["seed"]))
    rec = np.asarray(payload["per_class_recall"], dtype=float)
    fl_gap = metrics.class_gap(rec)
    cent_gap = float(cent["class_gap"]) if cent else float("nan")
    value = metrics.ncg_from_gaps(fl_gap, cent_gap) if cent else float("nan")
    notes = payload.get("notes", "")
    if cent is None:
        notes = (notes + " | MISSING centralized reference").strip(" |")
    return {
        "run_id": payload["run_id"],
        "setting": payload["setting"],
        "method": payload["method"],
        "dataset": payload["dataset"],
        "seed": int(payload["seed"]),
        "alpha": payload.get("alpha"),
        "client_count": payload.get("client_count"),
        "backbone": payload.get("backbone", config.BACKBONE),
        "partition_file": payload.get("partition_file"),
        "fashionmnist_subsample_file": payload.get("fashionmnist_subsample_file"),
        "per_class_recall_json": json.dumps([round(float(v), 6) for v in rec]),
        "class_gap": round(fl_gap, 6),
        "ncg": None if not np.isfinite(value) else round(value, 6),
        "cent_class_gap": None if cent is None else round(cent_gap, 6),
        "fl_class_gap": round(fl_gap, 6),
        "minority_class": metrics.minority_class(rec),
        "wall_clock_seconds": round(float(payload.get("wall_clock_seconds", 0.0)), 1),
        "status": payload.get("status", "complete"),
        "notes": notes,
    }


def _central_rows(runs: list[dict]) -> list[dict]:
    """One row per centralized model (6 total), shared across settings."""
    used = sorted({(p["dataset"], int(p["seed"])) for p in runs})
    rows = []
    for dataset, seed in used:
        cent = load_central(dataset, seed)
        if cent is None:
            continue
        rec = np.asarray(cent["per_class_recall"], dtype=float)
        rows.append({
            "run_id": config.central_key(dataset, seed),
            "setting": "reference",
            "method": "Centralized",
            "dataset": dataset,
            "seed": seed,
            "alpha": None,
            "client_count": None,
            "backbone": config.BACKBONE,
            "partition_file": "",
            "fashionmnist_subsample_file": cent.get("subsample_file"),
            "per_class_recall_json": json.dumps([round(float(v), 6) for v in rec]),
            "class_gap": round(float(cent["class_gap"]), 6),
            "ncg": 0.0,
            "cent_class_gap": round(float(cent["class_gap"]), 6),
            "fl_class_gap": None,
            "minority_class": int(cent["minority_class"]),
            "wall_clock_seconds": round(float(cent.get("wall_clock_seconds", 0.0)), 1),
            "status": "reference",
            "notes": (f"centralized CNN-2 reference (one per dataset/seed, shared by B1/B2/B3 and "
                      f"all methods); ES on {cent.get('val_source')}; "
                      f"test acc={cent.get('test_accuracy', float('nan')):.4f}"),
        })
    return rows


def build_raw() -> pd.DataFrame:
    runs = [p for p in load_runs() if p.get("run_role") != "extension"]
    rows = [_fl_row(p) for p in runs]
    rows.extend(_central_rows(runs))
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    if not df.empty:
        df["_s"] = pd.Categorical(df["setting"], SETTING_ORDER, ordered=True)
        df["_m"] = pd.Categorical(df["method"], METHOD_ORDER, ordered=True)
        df = df.sort_values(["_s", "_m", "dataset", "seed"]).drop(columns=["_s", "_m"])
    return df.reset_index(drop=True)


def build_raw_extension() -> pd.DataFrame:
    """Extension runs (run_role == 'extension'), kept out of the main table."""
    runs = [p for p in load_runs() if p.get("run_role") == "extension"]
    rows = [_fl_row(p) for p in runs]
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    if not df.empty:
        df["_s"] = pd.Categorical(df["setting"], SETTING_ORDER, ordered=True)
        df["_m"] = pd.Categorical(df["method"], METHOD_ORDER, ordered=True)
        df = df.sort_values(["_s", "_m", "dataset", "seed"]).drop(columns=["_s", "_m"])
    return df.reset_index(drop=True)


def write_raw() -> pd.DataFrame:
    df = build_raw()
    df.to_csv(config.RESULTS_DIR / "raw_results.csv", index=False)
    ext = build_raw_extension()
    if not ext.empty:
        ext.to_csv(config.RESULTS_DIR / "raw_results_extension.csv", index=False)
    return df


def _order(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["_s"] = pd.Categorical(df["setting"], SETTING_ORDER, ordered=True)
    df["_m"] = pd.Categorical(df["method"], METHOD_ORDER, ordered=True)
    return df.sort_values(["_s", "_m", "dataset"]).drop(columns=["_s", "_m"]).reset_index(drop=True)


NCG_COLUMNS = ["setting", "method", "dataset", "ncg_mean", "ncg_std",
               "class_gap_mean", "class_gap_std", "num_seeds"]
GAP_COLUMNS = ["setting", "method", "dataset", "cent_class_gap_mean",
               "fl_class_gap_mean", "fl_class_gap_std", "num_seeds"]


def summary_ncg(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["method"] != "Centralized"] if not df.empty else df
    if df.empty:
        return pd.DataFrame(columns=NCG_COLUMNS)
    rows = []
    for (setting, method, dataset), g in df.groupby(["setting", "method", "dataset"], dropna=False):
        ncg = g["ncg"].astype(float)
        gap = g["class_gap"].astype(float)
        rows.append({
            "setting": setting, "method": method, "dataset": dataset,
            "ncg_mean": ncg.mean(),
            "ncg_std": ncg.std(ddof=1) if ncg.notna().sum() > 1 else 0.0,
            "class_gap_mean": gap.mean(),
            "class_gap_std": gap.std(ddof=1) if gap.notna().sum() > 1 else 0.0,
            "num_seeds": int(g["seed"].nunique()),
        })
    return _order(pd.DataFrame(rows))


def summary_class_gap(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["method"] != "Centralized"] if not df.empty else df
    if df.empty:
        return pd.DataFrame(columns=GAP_COLUMNS)
    rows = []
    for (setting, method, dataset), g in df.groupby(["setting", "method", "dataset"], dropna=False):
        fl = g["fl_class_gap"].astype(float)
        rows.append({
            "setting": setting, "method": method, "dataset": dataset,
            "cent_class_gap_mean": g["cent_class_gap"].astype(float).mean(),
            "fl_class_gap_mean": fl.mean(),
            "fl_class_gap_std": fl.std(ddof=1) if fl.notna().sum() > 1 else 0.0,
            "num_seeds": int(g["seed"].nunique()),
        })
    return _order(pd.DataFrame(rows))


def write_summaries():
    raw = config.RESULTS_DIR / "raw_results.csv"
    df = pd.read_csv(raw) if raw.exists() else write_raw()
    ncg_df = summary_ncg(df)
    gap_df = summary_class_gap(df)
    ncg_df.to_csv(config.RESULTS_DIR / "summary_ncg.csv", index=False)
    gap_df.to_csv(config.RESULTS_DIR / "summary_class_gap.csv", index=False)
    ext = build_raw_extension()
    if not ext.empty:
        summary_ncg(ext).to_csv(config.RESULTS_DIR / "summary_ncg_extension.csv", index=False)
        summary_class_gap(ext).to_csv(config.RESULTS_DIR / "summary_class_gap_extension.csv",
                                      index=False)
    return df, ncg_df, gap_df


def build_manifest() -> dict:
    runs = load_runs()
    main_done = {p["run_id"] for p in runs if p.get("run_role") != "extension"}
    ext_runs = [p for p in runs if p.get("run_role") == "extension"]

    fl_tasks = []
    for cfg in config.all_run_configs():
        fl_tasks.append({
            "run_id": cfg["run_id"], "kind": "fl", "tier": "single",
            "setting": cfg["setting"], "method": cfg["method"], "dataset": cfg["dataset"],
            "seed": cfg["seed"],
            "output": str(config.run_path(cfg["setting"], cfg["method"], cfg["dataset"], cfg["seed"])),
            "status": "complete" if cfg["run_id"] in main_done else "pending",
        })

    cent_tasks = []
    for dataset in config.DATASET_NAMES:
        for seed in config.SEEDS:
            path = config.central_path(dataset, seed)
            cent_tasks.append({
                "run_id": config.central_key(dataset, seed), "kind": "centralized", "tier": "single",
                "setting": "all", "method": "Centralized", "dataset": dataset, "seed": int(seed),
                "output": str(path),
                "status": "complete" if path.exists() else "pending",
            })

    extension_tasks = []
    for p in ext_runs:
        extension_tasks.append({
            "run_id": p["run_id"], "kind": "fl", "tier": "extension", "run_role": "extension",
            "setting": p["setting"], "method": p["method"], "dataset": p["dataset"],
            "seed": int(p["seed"]), "output": str(config.RUNS_DIR / f"{p['run_id']}.json"),
            "status": "complete",
        })

    return {
        "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": list(config.SEEDS),
        "settings": list(config.SETTINGS),
        "methods": list(config.FL_METHODS) + ["Centralized"],
        "datasets": list(config.DATASET_NAMES),
        "rounds": int(config.TRAINING["rounds"]),
        "local_epochs": int(config.TRAINING["local_epochs"]),
        "centralized_max_epochs": int(config.CENTRAL_TRAINING["max_epochs"]),
        "run_id_convention": {
            "centralized": "cent__<Dataset>__cnn2__s<seed>__R<centralized.max_epochs>",
            "fl": "<Setting>__<Method>__<Dataset>__s<seed>__R<rounds>",
        },
        "totals": {
            "centralized": len(cent_tasks),
            "fl": len(fl_tasks),
            "grand": len(cent_tasks) + len(fl_tasks),
            "centralized_complete": sum(t["status"] == "complete" for t in cent_tasks),
            "fl_complete": sum(t["status"] == "complete" for t in fl_tasks),
            "extension": len(extension_tasks),
        },
        "centralized_tasks": cent_tasks,
        "fl_tasks": fl_tasks,
        "extension_tasks": extension_tasks,
    }


def write_manifest():
    manifest = build_manifest()
    config.dump_json(manifest, config.RESULTS_DIR / "manifest.json")
    rows = [{"kind": t["kind"], "run_id": t["run_id"], "setting": t["setting"],
             "method": t["method"], "dataset": t["dataset"], "seed": t["seed"],
             "tier": t.get("tier", "single"), "run_role": t.get("run_role", ""),
             "status": t["status"], "output": t["output"]}
            for t in manifest["centralized_tasks"] + manifest["fl_tasks"]
            + manifest["extension_tasks"]]
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "manifest.csv", index=False)
    return manifest
