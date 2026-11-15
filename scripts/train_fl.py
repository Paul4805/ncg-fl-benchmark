"""Run one FL job (B1/B2/B3 x method x dataset x seed).

    python -m scripts.train_fl --setting B2 --method FedAvg --dataset BloodMNIST --seed 0
    python -m scripts.train_fl --run-id B3__PHP-FL__FashionMNIST__s2__R50

Writes results/runs/<run_id>.json on completion (skipped if it already exists,
unless --force).  --eval-rounds and --run-role are used by diagnostics and by
the (not-run-in-this-session) extension path.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from core import config  # noqa: E402
from core.train_utils import Logger  # noqa: E402
from methods.registry import get_runner  # noqa: E402

MAIN_ROUNDS = int(config.TRAINING["rounds"])


def parse_run_id(run_id: str):
    """<Setting>__<Method>__<Dataset>__s<seed>__R<rounds>"""
    parts = run_id.split("__")
    if (len(parts) != 5 or not parts[3].startswith("s") or not parts[4].startswith("R")):
        raise ValueError(
            f"bad run id {run_id!r}; expected <Setting>__<Method>__<Dataset>__s<seed>__R<rounds>"
        )
    return parts[0], parts[1], parts[2], int(parts[3][1:]), int(parts[4][1:])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", choices=list(config.SETTINGS))
    ap.add_argument("--method", choices=list(config.FL_METHODS))
    ap.add_argument("--dataset", choices=list(config.DATASET_NAMES))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--run-id")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--run-role", default=None, choices=[None, "extension"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--eval-rounds", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)

    if args.run_id:
        setting, method, dataset, seed, rounds = parse_run_id(args.run_id)
        if args.rounds is not None and args.rounds != rounds:
            ap.error(f"--rounds {args.rounds} conflicts with run id R{rounds}")
        args.rounds = rounds
        run_role = args.run_role or ("extension" if rounds != MAIN_ROUNDS else None)
    else:
        if None in (args.setting, args.method, args.dataset, args.seed):
            ap.error("provide --setting/--method/--dataset/--seed or --run-id")
        setting, method, dataset, seed = args.setting, args.method, args.dataset, args.seed
        rounds = args.rounds
        run_role = args.run_role
        if rounds is not None and rounds != MAIN_ROUNDS and run_role is None:
            ap.error(f"--rounds {rounds} requires --run-role extension")

    cfg = config.resolve_run(setting, method, dataset, seed, rounds=rounds, run_role=run_role)
    out_dir = Path(args.out_dir) if args.out_dir else config.RUNS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg['run_id']}.json"
    if out_path.exists() and not args.force:
        print(f"[train_fl] {out_path.name} exists, skipping")
        return

    logger = Logger(config.LOGS_DIR / "fl" / f"{cfg['run_id']}.log")
    logger(f"=== {cfg['run_id']} | {cfg['setting_label']} | {cfg['dataset']} | seed={seed} | "
           f"clients={cfg['client_count']} alpha={cfg['alpha']} R={cfg['training']['rounds']} "
           f"E={cfg['training']['local_epochs']} lr={cfg['training']['lr']} | CNN-2 | "
           f"centralized_ref={cfg['centralized_ref_key']} | role={cfg.get('run_role', 'main')}")
    logger(f"partition: {cfg['partition_file']}")
    logger(f"subsample: {cfg['fashionmnist_subsample_file']}")
    eval_rounds = [int(x) for x in args.eval_rounds.split(",")] if args.eval_rounds else None

    t0 = time.time()
    payload = get_runner(method)(cfg, logger=logger, device=None, eval_rounds=eval_rounds)
    payload["env"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config.dump_json(payload, out_path)
    logger(f"=== wrote {out_path} in {time.time() - t0:.1f}s")
    logger.close()


if __name__ == "__main__":
    main()
