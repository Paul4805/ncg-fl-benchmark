"""Run one heterogeneous FL job (FedAPEN B1-B3, PHP-FL B2-B3).

    python -m benchmark_ncg.scripts.train_fl_hetero --setting B3 --method FedAPEN \
        --dataset BloodMNIST --seed 0

Writes results/runs_hetero/<Setting>__<Method>__<Dataset>__s<seed>__HET__R50.json
with the heterogeneous fields required by the supplementary protocol.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from benchmark_ncg.core import config, metrics  # noqa: E402
from benchmark_ncg.core import hetero_config as hc  # noqa: E402
from benchmark_ncg.core.train_utils import Logger  # noqa: E402
from benchmark_ncg.methods.registry import get_runner  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", choices=list(config.SETTINGS))
    ap.add_argument("--method", choices=list(hc.HET_METHODS))
    ap.add_argument("--dataset", choices=list(config.DATASET_NAMES))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)

    if None in (args.setting, args.method, args.dataset, args.seed):
        ap.error("provide --setting/--method/--dataset/--seed")

    cfg = hc.resolve_run_hetero(args.setting, args.method, args.dataset, args.seed)
    out_path = hc.run_path(args.setting, args.method, args.dataset, args.seed)
    if out_path.exists() and not args.force:
        print(f"[train_fl_hetero] {out_path.name} exists, skipping")
        return

    mean_path = hc.mean_path(args.method, args.dataset, args.seed)
    if not mean_path.exists():
        raise SystemExit(f"missing per-architecture mean reference {mean_path}")

    ok, diffs = hc.verify_main_table()
    if not ok:
        raise SystemExit(f"main-table checksum changed before run: {diffs}")

    logger = Logger(config.LOGS_DIR / "hetero_fl" / f"{cfg['run_id']}.log")
    logger(f"=== {cfg['run_id']} | {args.method} | {cfg['dataset']} | seed={args.seed} | "
           f"R={cfg['training']['rounds']} E={cfg['training']['local_epochs']} | hetero=True | "
           f"ref={mean_path.name}")
    logger(f"scope: {cfg['hetero_scope']}; architectures: "
           f"{[a['class_name'] for a in cfg['method_arch_list']]}")

    t0 = time.time()
    payload = get_runner(args.method)(cfg, logger=logger,
                                      device=torch.device("cpu") if args.cpu else None)

    mean = config.load_json(mean_path)
    cent_recall = mean["cent_recall_hetero"]
    cent_gap = float(mean["cent_class_gap_hetero"])
    fl_gap = float(payload["fl_class_gap"])
    ncg = float((fl_gap - cent_gap) / (1.0 - cent_gap)) if abs(1.0 - cent_gap) > 1e-12 else float("nan")

    payload.update({
        "hetero": True,
        "method_arch_list": cfg["method_arch_list"],
        "client_to_arch": cfg["client_to_arch"],
        "centralized_ref_mode": "per_architecture",
        "centralized_ref_key": mean_path.stem,
        "cent_recall_hetero": cent_recall,
        "cent_class_gap_hetero": cent_gap,
        "ncg_hetero": ncg,
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    })
    config.dump_json(payload, out_path)
    logger(f"=== wrote {out_path.name} gap={fl_gap:.4f} cent_gap={cent_gap:.4f} "
           f"ncg_hetero={ncg:+.4f} in {time.time() - t0:.0f}s")
    logger.close()

    ok, diffs = hc.verify_main_table()
    if not ok:
        raise SystemExit(f"main-table checksum changed after run: {diffs}")


if __name__ == "__main__":
    main()
