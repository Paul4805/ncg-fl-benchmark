"""Read-only PHP-FL sharing analysis (no training, no inference on new data).

Reads the 18 saved PHP-FL run JSONs and reports, per (setting, dataset, seed):

  a) phpfl_frac_global_params
     Fraction of the personal-global model's parameters that are overwritten by
     the server average at the final round: p = sigmoid(sig_k,(join_times+1)/R) * tau.
     This is the only parameter-level quantity that defines the deployed model's
     global/private alignment in PHP-FL (the deployed output itself is the
     ensemble w*private + (1-w)*personal_global, whose w is also recorded).
     Model tensors were not persisted by the queue, so the fraction is derived
     from the saved per-client join_times and the exact mask schedule that was
     used during training; it is identical across clients when all clients
     participate every round.

  b) phpfl_per_client_recall_spread  (from the saved per-client recall vectors)
  c) phpfl_ensemble_vs_mean_gap      (gap of the averaged vector vs mean client gap)
  d) run metadata + the fl_class_gap already stored in the run JSON

Output: results/diagnostics/phpfl_sharing_analysis.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core import config, metrics  # noqa: E402


def _sigmoid(k, x):
    return 1.0 / (1.0 + np.exp(k * (x - 0.5)))


def analyse_run(payload: dict) -> dict:
    cfg = payload["config"]
    hp = cfg["hparams"]
    rounds = int(payload["rounds"])
    vectors = np.asarray(payload["per_client_recall"], dtype=float)
    n_clients, C = vectors.shape
    spreads = vectors.max(axis=1) - vectors.min(axis=1)
    avg = metrics.average_recall_vectors(vectors)
    ensemble_gap = metrics.class_gap(avg)
    stored_gap = float(payload["fl_class_gap"])

    join_times = payload.get("join_times") or [rounds] * n_clients
    sig_k, tau = float(hp["sig_k"]), float(hp["tau"])
    frac_per_client = [
        float(_sigmoid(sig_k, (int(jt) + 1) * 1.0 / rounds) * tau) for jt in join_times
    ]
    frac = float(np.mean(frac_per_client))
    identical = bool(np.ptp(frac_per_client) <= 1e-12)

    return {
        "setting": payload["setting"],
        "alpha": payload.get("alpha"),
        "dataset": payload["dataset"],
        "seed": int(payload["seed"]),
        "run_id": payload["run_id"],
        "centralized_ref_key": payload["centralized_ref_key"],
        "fl_class_gap": stored_gap,
        "phpfl_frac_global_params": frac,
        "phpfl_frac_global_params_per_client": frac_per_client,
        "phpfl_frac_global_params_definition": (
            "final-round fraction of personal-global parameters overwritten by the server "
            "average: sigmoid(sig_k,(join_times+1)/rounds)*tau, computed from saved join_times"
        ),
        "phpfl_frac_private_params": 1.0 - frac,
        "deployed_ensemble_weight_private_mean": float(np.mean(payload.get("learned_weights", [0.5]))),
        "phpfl_per_client_recall_spread": {
            "mean_spread": float(spreads.mean()),
            "std_spread": float(spreads.std(ddof=1)) if n_clients > 1 else 0.0,
            "per_client_spread": [float(v) for v in spreads],
        },
        "phpfl_ensemble_vs_mean_gap": {
            "ensemble_gap": float(ensemble_gap),
            "mean_of_client_gaps": float(spreads.mean()),
            "difference_ensemble_minus_mean": float(ensemble_gap - spreads.mean()),
            "abs_diff_vs_run_json": float(abs(ensemble_gap - stored_gap)),
        },
        "num_clients": int(n_clients),
        "num_classes": int(C),
        "frac_identical_across_clients": identical,
        "ensemble_gap_matches_run_json": bool(abs(ensemble_gap - stored_gap) <= 1e-6),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(config.DIAG_DIR / "phpfl_sharing_analysis.json"))
    args = ap.parse_args(argv)

    per_run, missing = [], []
    for dataset in config.DATASET_NAMES:
        for setting in ["B1", "B2", "B3"]:
            for seed in config.SEEDS:
                path = config.run_path(setting, "PHP-FL", dataset, seed)
                if not path.exists():
                    missing.append(str(path))
                    continue
                per_run.append(analyse_run(config.load_json(path)))

    if missing:
        print(f"[sharing] WARNING missing {len(missing)} PHP-FL runs: {missing}")

    summary = {}
    for dataset in config.DATASET_NAMES:
        summary[dataset] = {}
        for setting in ["B1", "B2", "B3"]:
            recs = [r for r in per_run if r["dataset"] == dataset and r["setting"] == setting]
            if not recs:
                summary[dataset][setting] = {"n": 0}
                continue
            summary[dataset][setting] = {
                "n": len(recs),
                "phpfl_frac_global_params": float(np.mean([r["phpfl_frac_global_params"] for r in recs])),
                "mean_spread": float(np.mean([r["phpfl_per_client_recall_spread"]["mean_spread"] for r in recs])),
                "ensemble_gap": float(np.mean([r["phpfl_ensemble_vs_mean_gap"]["ensemble_gap"] for r in recs])),
                "mean_of_client_gaps": float(np.mean([r["phpfl_ensemble_vs_mean_gap"]["mean_of_client_gaps"] for r in recs])),
                "ensemble_gap_std": float(np.std([r["phpfl_ensemble_vs_mean_gap"]["ensemble_gap"] for r in recs], ddof=1)) if len(recs) > 1 else 0.0,
            }

    out = {
        "per_run": per_run,
        "summary_by_setting": summary,
        "assertions": {
            "frac_global_params_identical_across_clients": bool(all(r["frac_identical_across_clients"] for r in per_run)),
            "ensemble_gap_matches_run_json": bool(all(r["ensemble_gap_matches_run_json"] for r in per_run)),
        },
        "notes": (
            "Read-only analysis of saved PHP-FL run records. Per-client recall vectors were "
            "computed on the global test set during the original runs (labels=range(C), "
            "zero_division=0); no model was retrained or re-evaluated here. Model tensors were "
            "not persisted, so phpfl_frac_global_params is derived from the exact training-time "
            "mask schedule and the saved per-client join_times."
        ),
    }
    config.dump_json(out, Path(args.out))

    print(f"[sharing] {len(per_run)} PHP-FL records -> {args.out}")
    for dataset in config.DATASET_NAMES:
        for setting in ["B1", "B2", "B3"]:
            s = summary[dataset][setting]
            if s.get("n"):
                print(f"  {dataset:13s} {setting}: frac_global={s['phpfl_frac_global_params']:.4f} "
                      f"mean_spread={s['mean_spread']:.4f} ensemble_gap={s['ensemble_gap']:.4f} "
                      f"mean_client_gaps={s['mean_of_client_gaps']:.4f}")
    print(f"  assertions: {out['assertions']}")


if __name__ == "__main__":
    main()
