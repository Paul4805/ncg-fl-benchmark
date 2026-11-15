"""Driver for the heterogeneous supplementary run (separate namespace).

Order: per-architecture centralized refs -> FedAPEN B3/B2/B1 -> PHP-FL B3 ->
PHP-FL B2 (only if projected to finish before the internal stop). Hard cap 12 h,
internal stop 10 h, --workers 2, --resume.

    python -m benchmark_ncg.scripts.run_all_hetero --workers 2 --resume
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_ncg.core import config, hetero_config as hc  # noqa: E402

PY = sys.executable
LOG_DIR = config.LOGS_DIR / "run_all_hetero"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SETTING_IDX = {s: i for i, s in enumerate(["B1", "B2", "B3"])}
DS_IDX = {d: i for i, d in enumerate(config.DATASET_NAMES)}
METHOD_IDX = {m: i for i, m in enumerate(hc.HET_METHODS)}


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


def build_tasks(resume=True, settings=None):
    tasks = []
    for dataset in config.DATASET_NAMES:
        for seed in config.SEEDS:
            for method in hc.HET_METHODS:
                out = hc.mean_path(method, dataset, seed)
                if not (resume and out.exists()):
                    tasks.append(Task(
                        task_id=f"ref__{method}__{dataset}__s{seed}", kind="ref",
                        cmd=[PY, "-m", "benchmark_ncg.scripts.train_centralized_hetero",
                             "--method", method, "--dataset", dataset, "--seed", str(seed)],
                        output=out,
                        priority=(0, DS_IDX[dataset], int(seed), METHOD_IDX[method]),
                    ))
    for method in hc.HET_METHODS:
        for setting in hc.HET_SETTINGS[method]:
            if settings is not None and setting not in settings:
                continue
            if method == "FedAPEN":
                phase = 10
            else:
                phase = {"B3": 20, "B2": 30, "B1": 40}.get(setting, 40)
            for dataset in config.DATASET_NAMES:
                for seed in config.SEEDS:
                    out = hc.run_path(setting, method, dataset, seed)
                    if resume and out.exists():
                        continue
                    tasks.append(Task(
                        task_id=hc.run_id(setting, method, dataset, seed), kind="fl",
                        cmd=[PY, "-m", "benchmark_ncg.scripts.train_fl_hetero",
                             "--setting", setting, "--method", method,
                             "--dataset", dataset, "--seed", str(seed)],
                        output=out,
                        priority=(phase, SETTING_IDX[setting], DS_IDX[dataset], int(seed)),
                        deps=[f"ref__{method}__{dataset}__s{seed}"],
                    ))
    tasks.sort(key=lambda t: t.priority)
    return tasks


def run_task(task: Task, deadline: float):
    if task.output.exists():
        return task.task_id, True, 0.0, "exists"
    if time.time() > deadline:
        return task.task_id, None, 0.0, "wall-clock cap"
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


def ref_ready(dep: str) -> bool:
    """dep = ref__<Method>__<Dataset>__s<seed>"""
    _, method, dataset, s = dep.split("__")
    return hc.mean_path(method, dataset, int(s[1:])).exists()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-hours", type=float, default=12.0)
    ap.add_argument("--internal-stop", type=float, default=10.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--settings", nargs="+", default=None, choices=list(config.SETTINGS),
                    help="restrict to these settings (e.g. --settings B3)")
    args = ap.parse_args(argv)

    ok, diffs = hc.verify_main_table()
    if not ok:
        raise SystemExit(f"ABORT: main-table checksum mismatch: {diffs}")
    print("[hetero] main-table checksum unchanged at start")

    t_start = time.time()
    deadline = t_start + args.max_hours * 3600
    internal = t_start + args.internal_stop * 3600
    tasks = build_tasks(resume=not args.force, settings=args.settings)
    print(f"[hetero] {len(tasks)} tasks queued "
          f"({sum(t.kind == 'fl' for t in tasks)} FL, {sum(t.kind == 'ref' for t in tasks)} ref-groups) "
          f"workers={args.workers} hard={args.max_hours}h internal={args.internal_stop}h")
    for t in tasks[:8]:
        print(f"  {t.priority} {t.kind:3s} {t.task_id}")
    if len(tasks) > 8:
        print(f"  ... {len(tasks) - 8} more")
    if args.dry_run:
        return

    done, failed, skipped, cut = set(), set(), set(), set()
    running, pending = {}, list(tasks)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        while pending or running:
            now = time.time()
            for t in list(pending):
                if len(running) >= args.workers:
                    break
                if any(d in failed for d in t.deps):
                    pending.remove(t); skipped.add(t.task_id); continue
                if now > deadline:
                    skipped.update(x.task_id for x in pending)
                    print(f"[hetero] HARD CAP {args.max_hours}h: {len(pending)} tasks left")
                    pending = []
                    break
                if now > internal and t.kind == "fl" and t.task_id.startswith("B2__") \
                        and "__PHP-FL__" in t.task_id:
                    pending.remove(t)
                    cut.add(t.task_id)
                    print(f"[hetero] INTERNAL STOP: cutting PHP-FL B2 task {t.task_id}")
                    continue
                if all(d in done or ref_ready(d) for d in t.deps):
                    running[pool.submit(run_task, t, deadline)] = t
                    pending.remove(t)
            if not running:
                if pending:
                    print(f"[hetero] BLOCKED {len(pending)} tasks (unmet deps)")
                break
            finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in finished:
                t = running.pop(fut)
                task_id, ok, dt, info = fut.result()
                if ok is True:
                    done.add(task_id)
                    print(f"[hetero] DONE {task_id} ({dt:.0f}s)")
                elif ok is None:
                    skipped.add(task_id)
                    print(f"[hetero] PENDING {task_id}: {info}")
                else:
                    failed.add(task_id)
                    print(f"[hetero] FAIL {task_id}\n{info}")

    print(f"\n[hetero] {(time.time() - t_start) / 3600:.2f}h elapsed: "
          f"{len(done)} done, {len(failed)} failed, {len(skipped)} pending, {len(cut)} cut")
    if failed:
        print("[hetero] failed:", sorted(failed))

    ok, diffs = hc.verify_main_table()
    print(f"[hetero] main-table checksum at end: {'unchanged' if ok else 'CHANGED ' + str(diffs)}")

    from benchmark_ncg.scripts.compute_hetero import main as compute_main
    compute_main([])


if __name__ == "__main__":
    main()
