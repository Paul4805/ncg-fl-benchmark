"""Central configuration for the NCG benchmark.

Single source of truth: configs/benchmark.json + configs/methods.json.
Locked reductions (must never be rolled back):
  R1  only B1/B2/B3 homogeneous CNN-2 runs + 6 centralized CNN-2 references
  R2  R = 50 FL rounds, centralized max_epochs = R * E = 250
  R3  FashionMNIST 30k/5k per-seed stratified subsample
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = BENCH_ROOT / "configs"
DATA_DIR = BENCH_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
PARTITION_DIR = BENCH_ROOT / "partitions"
RESULTS_DIR = BENCH_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
CENTRAL_DIR = RESULTS_DIR / "centralized"
DIAG_DIR = RESULTS_DIR / "diagnostics"
SMOKE_DIR = RESULTS_DIR / "smoke"
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = BENCH_ROOT / "logs"
METHODS_DIR = BENCH_ROOT / "methods"

for _p in (CONFIG_DIR, DATA_DIR, RAW_DIR, CACHE_DIR, PARTITION_DIR, RESULTS_DIR,
           RUNS_DIR, CENTRAL_DIR, DIAG_DIR, SMOKE_DIR, PLOTS_DIR, LOGS_DIR):
    _p.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


_BENCH = load_json(CONFIG_DIR / "benchmark.json")
_METHODS = load_json(CONFIG_DIR / "methods.json")

SEEDS = tuple(int(s) for s in _BENCH["seeds"])
FL_METHODS = tuple(_BENCH["fl_methods"])
CENTRALIZED = "Centralized"
SETTINGS = _BENCH["settings"]
DATASETS = _BENCH["datasets"]
TRAINING = _BENCH["training"]
CENTRAL_TRAINING = _BENCH["centralized"]
NUM_CLIENTS = int(_BENCH["num_clients"])
BACKBONE = _BENCH["backbone"]
METHOD_CFG = _METHODS
WALL_CLOCK_BUDGET_HOURS = float(_BENCH["wall_clock_budget_hours"])
DATASET_NAMES = tuple(DATASETS)


def assert_locked() -> None:
    """Hard guards against silently rolling back R1/R2/R3."""
    if TRAINING["rounds"] != 50:
        raise RuntimeError(f"R2 violated: rounds={TRAINING['rounds']} (must be 50)")
    if TRAINING["local_epochs"] != 5:
        raise RuntimeError("R2 violated: local_epochs must be 5")
    if CENTRAL_TRAINING["max_epochs"] != TRAINING["rounds"] * TRAINING["local_epochs"]:
        raise RuntimeError("centralized max_epochs must equal R*E = 250")
    if set(SETTINGS) != {"B1", "B2", "B3"}:
        raise RuntimeError(f"only B1/B2/B3 are allowed, got {sorted(SETTINGS)}")
    if BACKBONE != "cnn2":
        raise RuntimeError("only the CNN-2 backbone is allowed")
    if NUM_CLIENTS != 20:
        raise RuntimeError("20 clients required")
    if tuple(FL_METHODS) != ("FedAvg", "FedAPEN", "FedFV", "PHP-FL"):
        raise RuntimeError("unexpected method set")


assert_locked()


# ---------------------------------------------------------------------------
# Naming / paths
# ---------------------------------------------------------------------------
def _alpha_tag(protocol: str, alpha) -> str:
    if protocol == "iid":
        return "iid"
    return f"{protocol}_a{('%g' % alpha).replace('.', 'p')}"


def partition_path(dataset: str, protocol: str, alpha, client_count: int, seed: int) -> Path:
    return PARTITION_DIR / dataset / f"{dataset}__{_alpha_tag(protocol, alpha)}__n{client_count}__s{seed}.json"


def subsample_path(seed: int) -> Path:
    return PARTITION_DIR / f"fashionmnist_subsample_seed{seed}.json"


def run_id(setting: str, method: str, dataset: str, seed: int, rounds: int | None = None) -> str:
    r = int(TRAINING["rounds"] if rounds is None else rounds)
    return f"{setting}__{method}__{dataset}__s{int(seed)}__R{r}"


def run_path(setting: str, method: str, dataset: str, seed: int, rounds: int | None = None) -> Path:
    return RUNS_DIR / f"{run_id(setting, method, dataset, seed, rounds)}.json"


def central_key(dataset: str, seed: int) -> str:
    return f"cent__{dataset}__cnn2__s{int(seed)}__R{int(CENTRAL_TRAINING['max_epochs'])}"


def central_path(dataset: str, seed: int) -> Path:
    return CENTRAL_DIR / f"{central_key(dataset, seed)}.json"


# ---------------------------------------------------------------------------
# Run resolution
# ---------------------------------------------------------------------------
def resolve_run(setting: str, method: str, dataset: str, seed: int,
                rounds: int | None = None, run_role: str | None = None) -> dict:
    if setting not in SETTINGS:
        raise ValueError(
            f"invalid setting {setting!r}: only {sorted(SETTINGS)} exist "
            "(no Block A / A_spot / official-settings runs)"
        )
    if method not in FL_METHODS:
        raise ValueError(f"invalid method {method!r}: only {list(FL_METHODS)} exist")
    if dataset not in DATASETS:
        raise ValueError(f"invalid dataset {dataset!r}")
    if int(seed) not in SEEDS:
        raise ValueError(f"invalid seed {seed!r}: only {list(SEEDS)} exist")

    rounds = int(TRAINING["rounds"] if rounds is None else rounds)
    if run_role == "extension":
        if rounds == int(TRAINING["rounds"]):
            raise ValueError("extension runs must use a different round budget than the main queue")
    elif rounds != int(TRAINING["rounds"]):
        raise ValueError(
            f"rounds={rounds} is only allowed with run_role='extension' "
            f"(main queue is R={TRAINING['rounds']})"
        )

    s = SETTINGS[setting]
    ds = copy.deepcopy(DATASETS[dataset])
    n = NUM_CLIENTS
    training = copy.deepcopy(TRAINING)
    training["rounds"] = rounds
    cfg = {
        "run_id": run_id(setting, method, dataset, int(seed), rounds),
        "setting": setting,
        "setting_label": s["label"],
        "method": method,
        "dataset": dataset,
        "seed": int(seed),
        "alpha": s["alpha"],
        "protocol": s["protocol"],
        "client_count": n,
        "backbone": BACKBONE,
        "num_classes": ds["num_classes"],
        "in_channels": ds["in_channels"],
        "partition_file": str(partition_path(dataset, s["protocol"], s["alpha"], n, int(seed))),
        "fashionmnist_subsample_file": (
            str(subsample_path(int(seed))) if dataset == "FashionMNIST" else None
        ),
        "training": training,
        "centralized_protocol": copy.deepcopy(CENTRAL_TRAINING),
        "hparams": copy.deepcopy(METHOD_CFG[method]["hparams"]),
        "method_source": METHOD_CFG[method]["source"],
        "centralized_ref_key": central_key(dataset, int(seed)),
        "centralized_ref_file": str(central_path(dataset, int(seed))),
        "data_spec": ds,
        "wall_clock_budget_hours": WALL_CLOCK_BUDGET_HOURS,
    }
    if run_role:
        cfg["run_role"] = run_role
    return cfg


def all_run_configs(settings=None, methods=None, datasets=None, seeds=None):
    for setting in (settings or list(SETTINGS)):
        for method in (methods or list(FL_METHODS)):
            for dataset in (datasets or DATASET_NAMES):
                for seed in (SEEDS if seeds is None else seeds):
                    yield resolve_run(setting, method, dataset, seed)


def required_partitions():
    """Unique (dataset, protocol, alpha, n_clients, seed) partitions for all 72 runs."""
    seen = {}
    for cfg in all_run_configs():
        key = (cfg["dataset"], cfg["protocol"], cfg["alpha"], cfg["client_count"], cfg["seed"])
        seen[key] = cfg["partition_file"]
    return seen
