"""Configuration for the heterogeneous supplementary run (separate namespace)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from . import config, models
from .hetero_models import PHPFL_GLOBAL_ARCH as _PHPFL_GLOBAL_ARCH

HET_SUFFIX = "HET__R50"
HET_RUNS_DIR = config.RESULTS_DIR / "runs_hetero"
HET_CENTRAL_DIR = config.RESULTS_DIR / "centralized_hetero"
HET_PLOTS_DIR = config.RESULTS_DIR / "plots_hetero"
for _p in (HET_RUNS_DIR, HET_CENTRAL_DIR, HET_PLOTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

HET_METHODS = ("FedAPEN", "PHP-FL")
FEDAPEN_ARCH_IDS = list(range(20))          # official --he 1: one width per client
PHPFL_ARCH_IDS = [0, 1, 2, 3]               # approved substitutes, cid % 4
PHPFL_GLOBAL_ARCH = _PHPFL_GLOBAL_ARCH

# PHP-FL: B1 (IID) was added at the user's request after the initial B2/B3 scope.
HET_SETTINGS = {"FedAPEN": ["B1", "B2", "B3"], "PHP-FL": ["B1", "B2", "B3"]}

SUBSTITUTE_SOURCE = (
    "substituted from official HtM4 (5.6-11.2M) due to 5M cap and 12h budget; "
    "approved micro2 set (candidate B2); see DEVIATIONS.md"
)


def run_id(setting: str, method: str, dataset: str, seed: int) -> str:
    return f"{setting}__{method}__{dataset}__s{int(seed)}__{HET_SUFFIX}"


def run_path(setting: str, method: str, dataset: str, seed: int) -> Path:
    return HET_RUNS_DIR / f"{run_id(setting, method, dataset, seed)}.json"


def ref_path(method: str, dataset: str, arch_id: int, seed: int) -> Path:
    return HET_CENTRAL_DIR / f"{method}__{dataset}__{int(arch_id)}__s{int(seed)}__R250.json"


def mean_path(method: str, dataset: str, seed: int) -> Path:
    return HET_CENTRAL_DIR / f"{method}__{dataset}__mean__s{int(seed)}__R250.json"


def fedapen_spec(cid: int) -> dict:
    return {"name": "cnn2_bn", "params": {"channel_diff": (5 - cid) * 2 - 1}}


def fedapen_shared_spec() -> dict:
    return {"name": "cnn2_bn", "params": {}}


def phpfl_spec(arch_id: int) -> dict:
    from .hetero_models import PHPFL_SUBSTITUTE_IDS

    return {"name": PHPFL_SUBSTITUTE_IDS[arch_id]["name"]}


def phpfl_global_spec() -> dict:
    return {"name": PHPFL_GLOBAL_ARCH}


def arch_spec(method: str, arch_id: int) -> dict:
    return fedapen_spec(arch_id) if method == "FedAPEN" else phpfl_spec(arch_id)


def client_model_specs(method: str, n_clients: int = 20) -> list[dict]:
    if method == "FedAPEN":
        return [fedapen_spec(cid) for cid in range(n_clients)]
    return [phpfl_spec(cid % len(PHPFL_ARCH_IDS)) for cid in range(n_clients)]


def client_to_arch(method: str, n_clients: int = 20) -> dict:
    if method == "FedAPEN":
        return {str(cid): cid for cid in range(n_clients)}
    return {str(cid): cid % len(PHPFL_ARCH_IDS) for cid in range(n_clients)}


def arch_ids(method: str) -> list[int]:
    return FEDAPEN_ARCH_IDS if method == "FedAPEN" else PHPFL_ARCH_IDS


def method_arch_list(method: str) -> list[dict]:
    """Official/spec entries with parameter counts for both dataset shapes."""
    from .hetero_models import PHPFL_SUBSTITUTE_IDS

    def count(name, in_ch, ncls, **params):
        return int(sum(x.numel() for x in models.build_model(name, in_ch, ncls, **params).parameters()))

    out = []
    if method == "FedAPEN":
        for cid in FEDAPEN_ARCH_IDS:
            d = (5 - cid) * 2 - 1
            pb = count("cnn2_bn", 3, 8, channel_diff=d)
            pf = count("cnn2_bn", 1, 10, channel_diff=d)
            out.append({"id": cid, "class_name": "CNN2_BN", "params": pb,
                        "params_by_dataset": {"BloodMNIST": pb, "FashionMNIST": pf},
                        "source": "FedAPEN/models/cnn.py:87-89 + node.py:65-67 (--he 1)",
                        "channel_diff": d})
    else:
        for rec in PHPFL_SUBSTITUTE_IDS:
            pb = count(rec["name"], 3, 8)
            pf = count(rec["name"], 1, 10)
            out.append({"id": rec["id"], "class_name": rec["name"], "params": pb,
                        "params_by_dataset": {"BloodMNIST": pb, "FashionMNIST": pf},
                        "source": SUBSTITUTE_SOURCE, "family": rec["family"]})
    return out


def write_arch_map() -> Path:
    payload = {
        "FedAPEN": {"method": "FedAPEN",
                    "architectures": method_arch_list("FedAPEN"),
                    "client_to_arch": client_to_arch("FedAPEN")},
        "PHP-FL": {"method": "PHP-FL",
                   "architectures": method_arch_list("PHP-FL"),
                   "client_to_arch": client_to_arch("PHP-FL")},
        "note": ("FedAPEN: official --he 1 per-client CNN2_BN widths (20 clients -> 20 archs). "
                 "PHP-FL: approved 4-model substitutes for the official HtM4 family, "
                 "round-robin cid % 4; " + SUBSTITUTE_SOURCE),
    }
    path = config.PARTITION_DIR / "hetero_arch_map.json"
    config.dump_json(payload, path)
    return path


def num_expected_refs() -> int:
    return sum(len(arch_ids(m)) * len(config.DATASET_NAMES) * len(config.SEEDS)
               for m in HET_METHODS)


def num_expected_runs() -> int:
    return sum(len(HET_SETTINGS[m]) * len(config.DATASET_NAMES) * len(config.SEEDS)
               for m in HET_METHODS)


def resolve_run_hetero(setting: str, method: str, dataset: str, seed: int) -> dict:
    if method not in HET_METHODS:
        raise ValueError(f"hetero runs only support {HET_METHODS}, got {method!r}")
    if setting not in config.SETTINGS:
        raise ValueError(f"invalid setting {setting!r}")
    if setting not in HET_SETTINGS[method]:
        raise ValueError(
            f"{method} hetero scope is {HET_SETTINGS[method]}; {setting} is out of scope"
        )
    cfg = copy.deepcopy(config.resolve_run(setting, method, dataset, seed))
    cfg["hetero"] = True
    cfg["output_suffix"] = HET_SUFFIX
    cfg["run_id"] = run_id(setting, method, dataset, seed)
    cfg["backbone"] = "heterogeneous"
    cfg["client_model_specs"] = client_model_specs(method)
    cfg["global_model_spec"] = (fedapen_shared_spec() if method == "FedAPEN"
                                else phpfl_global_spec())
    cfg["centralized_ref_mode"] = "per_architecture"
    cfg["arch_ids"] = arch_ids(method)
    cfg["method_arch_list"] = method_arch_list(method)
    cfg["client_to_arch"] = client_to_arch(method)
    cfg["centralized_ref_keys"] = [mean_path(method, dataset, seed).stem]
    cfg["centralized_ref_files"] = [str(mean_path(method, dataset, seed))]
    cfg["hetero_scope"] = "FedAPEN (B1,B2,B3); PHP-FL (B2,B3) with substitutes"
    return cfg


def all_hetero_runs():
    for method in HET_METHODS:
        for setting in HET_SETTINGS[method]:
            for dataset in config.DATASET_NAMES:
                for seed in config.SEEDS:
                    yield resolve_run_hetero(setting, method, dataset, seed)


# ---------------------------------------------------------------------------
# Main-table integrity (checksums)
# ---------------------------------------------------------------------------
MAIN_SUMMARY_FILES = ["raw_results.csv", "summary_ncg.csv", "summary_class_gap.csv",
                      "manifest.json", "manifest.csv"]


def main_table_files() -> list[Path]:
    b = config.BENCH_ROOT
    files = sorted((b / "results" / "runs").glob("*.json"))
    files += sorted((b / "results" / "centralized").glob("*.json"))
    files += sorted((b / "results" / "plots").glob("*.png"))
    files += [b / "results" / n for n in MAIN_SUMMARY_FILES]
    return files


def recompute_main_hashes() -> dict:
    b = config.BENCH_ROOT
    return {str(f.relative_to(b)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in main_table_files() if f.exists()}


def verify_main_table() -> tuple[bool, list[str]]:
    baseline_path = config.RESULTS_DIR / "main_table_checksums.json"
    if not baseline_path.exists():
        return False, ["missing results/main_table_checksums.json baseline"]
    baseline = config.load_json(baseline_path)
    current = recompute_main_hashes()
    expected = baseline["files"]
    diffs = []
    for k, v in expected.items():
        if k not in current:
            diffs.append(f"missing: {k}")
        elif current[k] != v:
            diffs.append(f"changed: {k}")
    for k in current:
        if k not in expected:
            diffs.append(f"added: {k}")
    return not diffs, diffs
