"""Single-tier driver with resume, 2-3 parallel workers and a hard wall-clock stop.

    python -m scripts.run_all --workers 2 --resume
    python -m scripts.run_all --stage centralized
    python -m scripts.run_all --stage report

Stages: partitions -> centralized (6 CNN-2 refs) -> fl (72 B1-B3 runs) -> report.
A job is skipped when its output JSON already exists (resume).  If the wall clock
reaches --max-hours (default 15, from config), no further jobs are launched,
remaining jobs stay `pending` in the manifest, and summaries/plots are finalized
from whatever is complete.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config, report  # noqa: E402

PY = sys.executable
LOG_DIR = config.LOGS_DIR / "run_all"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SETTING_IDX = {"B1": 0, "B2": 1, "B3": 2}
METHOD_IDX = {m: i for i, m in enumerate(config.FL_METHODS)}
DATASET_IDX = {d: i for i, d in enumerate(config.DATASET_NAMES)}


@dataclass
class Task:
    task_id: str
    kind: str
    cmd: list[str]
    output: Path
    priority: tuple
    deps: list[str] = field(default_factory=list)

    @property
    def log_path(self) -> Path:
        return LOG_DIR / (self.task_id.replace("__", "_").replace("/", "_") + ".out")


def build_tasks(resume: bool = True, settings=None, methods=None, datasets=None, seeds=None,
                rounds: int | None = None, run_role: str | None = None):
    settings = settings or list(config.SETTINGS)
    methods = methods or list(config.FL_METHODS)
    datasets = datasets or list(config.DATASET_NAMES)
    seeds = seeds or list(config.SEEDS)
    tasks = []

    for dataset in datasets:
        for seed in seeds:
            path = config.central_path(dataset, seed)
            if resume and path.exists():
                continue
            tasks.append(Task(
                task_id=config.central_key(dataset, seed), kind="centralized",
                cmd=[PY, "-m", "scripts.train_centralized",
                     "--dataset", dataset, "--seed", str(seed)],
                output=path, priority=(0, DATASET_IDX[dataset], int(seed), 0, 0),
            ))

    for setting in settings:
        for dataset in datasets:
            for method in methods:
                for seed in seeds:
                    cfg = config.resolve_run(setting, method, dataset, seed,
                                             rounds=rounds, run_role=run_role)
                    path = config.run_path(setting, method, dataset, seed, rounds=rounds)
                    if resume and path.exists():
                        continue
                    cmd = [PY, "-m", "scripts.train_fl",
                           "--setting", setting, "--method", method,
                           "--dataset", dataset, "--seed", str(seed)]
                    if rounds is not None:
                        cmd += ["--rounds", str(rounds)]
                    if run_role is not None:
                        cmd += ["--run-role", run_role]
                    tasks.append(Task(
                        task_id=cfg["run_id"], kind="fl",
                        cmd=cmd,
                        output=path,
                        priority=((2 if run_role == "extension" else 1),
                                  SETTING_IDX[setting], DATASET_IDX[dataset],
                                  METHOD_IDX[method], int(seed)),
                        deps=[cfg["centralized_ref_key"]],
                    ))
    tasks.sort(key=lambda t: t.priority)
    return tasks


def run_task(task: Task, deadline: float):
    if task.output.exists():
        return task.task_id, True, 0.0, "exists"
    if time.time() > deadline:
        return task.task_id, None, 0.0, "wall-clock budget exhausted before launch"
    t0 = time.time()
    with open(task.log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n=== {' '.join(task.cmd)}\n")
        fh.flush()
        proc = subprocess.run(task.cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
    ok = proc.returncode == 0 and task.output.exists()
    tail = ""
    if not ok:
        try:
            lines = task.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-20:])
        except Exception:
            pass
    return task.task_id, ok, time.time() - t0, tail


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["all", "partitions", "centralized", "fl", "report"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--force", action="store_true", help="ignore existing outputs")
    ap.add_argument("--settings", nargs="+", default=list(config.SETTINGS))
    ap.add_argument("--methods", nargs="+", default=list(config.FL_METHODS))
    ap.add_argument("--datasets", nargs="+", default=list(config.DATASET_NAMES))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(config.SEEDS))
    ap.add_argument("--rounds", type=int, default=None,
                    help="extension only: e.g. 75 (main queue is R=50)")
    ap.add_argument("--run-role", default=None, choices=[None, "extension"],
                    help="tag written into every run JSON; required for --rounds != 50")
    ap.add_argument("--max-hours", type=float, default=config.WALL_CLOCK_BUDGET_HOURS)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    t_start = time.time()
    deadline = t_start + args.max_hours * 3600.0

    if args.rounds is not None and args.rounds != int(config.TRAINING["rounds"]):
        if args.run_role != "extension":
            ap.error(f"--rounds {args.rounds} requires --run-role extension")
    if args.run_role == "extension" and args.rounds in (None, int(config.TRAINING["rounds"])):
        ap.error("--run-role extension requires --rounds different from the main queue")

    if args.stage in ("all", "partitions"):
        from scripts.generate_partitions import main as gen
        gen(["--force"] if args.force else [])
        if args.stage == "partitions":
            return

    if args.stage in ("all", "report"):
        pass  # report is done at the end of every stage that trains

    tasks = []
    if args.stage in ("all", "centralized", "fl"):
        tasks = build_tasks(resume=not args.force, settings=args.settings, methods=args.methods,
                            datasets=args.datasets, seeds=args.seeds,
                            rounds=args.rounds, run_role=args.run_role)
        if args.stage == "centralized":
            tasks = [t for t in tasks if t.kind == "centralized"]
        if args.stage == "fl":
            tasks = [t for t in tasks if t.kind == "fl"]
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    print(f"[run_all] {len(tasks)} jobs queued "
          f"({sum(t.kind == 'centralized' for t in tasks)} centralized, "
          f"{sum(t.kind == 'fl' for t in tasks)} FL), workers={args.workers}, "
          f"max_hours={args.max_hours}")
    for t in tasks:
        print(f"  {t.priority} {t.kind:11s} {t.task_id}  deps={t.deps}")
    if args.dry_run:
        return

    done, failed, skipped = set(), set(), set()
    running = {}
    pending = list(tasks)

    def dep_ready(dep: str, pending_ids: set) -> bool:
        if dep in done:
            return True
        if dep in pending_ids:
            return False
        return (config.CENTRAL_DIR / f"{dep}.json").exists()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        while pending or running:
            for t in list(pending):
                if len(running) >= args.workers:
                    break
                if any(d in failed for d in t.deps):
                    pending.remove(t)
                    skipped.add(t.task_id)
                    print(f"[run_all] SKIP {t.task_id} (dependency failed)")
                    continue
                if time.time() > deadline:
                    skipped.update(x.task_id for x in pending)
                    print(f"[run_all] HARD STOP at {args.max_hours}h: "
                          f"{len(pending)} jobs left pending")
                    pending = []
                    break
                if all(dep_ready(d, {p.task_id for p in pending}) for d in t.deps):
                    running[pool.submit(run_task, t, deadline)] = t
                    pending.remove(t)
            if not running:
                break
            finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in finished:
                t = running.pop(fut)
                task_id, ok, dt, info = fut.result()
                if ok is True:
                    done.add(task_id)
                    print(f"[run_all] DONE {task_id} ({dt:.0f}s)")
                elif ok is None:
                    skipped.add(task_id)
                    print(f"[run_all] PENDING {task_id}: {info}")
                else:
                    failed.add(task_id)
                    print(f"[run_all] FAIL {task_id}\n{info}")

    elapsed_h = (time.time() - t_start) / 3600.0
    print(f"\n[run_all] {elapsed_h:.2f}h elapsed: {len(done)} done, "
          f"{len(failed)} failed, {len(skipped)} skipped/pending")
    if failed:
        print("[run_all] failed:", sorted(failed))

    manifest = report.write_manifest()
    print(f"[run_all] manifest: centralized {manifest['totals']['centralized_complete']}/"
          f"{manifest['totals']['centralized']}, FL {manifest['totals']['fl_complete']}/"
          f"{manifest['totals']['fl']}")

    if args.stage != "centralized" or manifest["totals"]["fl_complete"] > 0:
        from scripts.compute_ncg import main as ncg_main
        from scripts.make_summary import main as sum_main
        from scripts.plot_ncg import main as plot_main
        ncg_main([])
        sum_main(["--from-raw"])
        plot_main([])


if __name__ == "__main__":
    main()
