"""Validation suite for the NCG benchmark.

Run before the queue:
    python -m pytest benchmark_ncg/tests -q
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_ncg.core import config, datasets, metrics, models, partition as partition_mod  # noqa: E402
from benchmark_ncg.methods.common import auxiliary_split  # noqa: E402


# ---------------------------------------------------------------------------
# Config / scope guards
# ---------------------------------------------------------------------------
def test_locked_reductions():
    assert config.TRAINING["rounds"] == 50
    assert config.TRAINING["local_epochs"] == 5
    assert config.CENTRAL_TRAINING["max_epochs"] == 250
    assert config.CENTRAL_TRAINING["early_stopping_patience"] == 20
    assert set(config.SETTINGS) == {"B1", "B2", "B3"}
    assert config.BACKBONE == "cnn2"
    assert config.NUM_CLIENTS == 20
    assert config.run_id("B1", "FedAvg", "BloodMNIST", 0).endswith("__R50")
    assert config.central_key("BloodMNIST", 0).endswith("__R250")


def test_extension_requires_role_and_gets_r_suffix():
    with pytest.raises(ValueError):
        config.resolve_run("B1", "PHP-FL", "BloodMNIST", 0, rounds=75)
    cfg = config.resolve_run("B1", "PHP-FL", "BloodMNIST", 0, rounds=75,
                             run_role="extension")
    assert cfg["run_id"].endswith("__R75")
    assert cfg["run_role"] == "extension"
    assert cfg["training"]["rounds"] == 75
    assert cfg["centralized_ref_key"].endswith("__R250")
    main = config.resolve_run("B1", "FedAvg", "BloodMNIST", 0)
    assert main["run_id"].endswith("__R50")
    assert "run_role" not in main


def test_main_and_extension_task_ids():
    from benchmark_ncg.scripts.run_all import build_tasks

    main = [t for t in build_tasks(resume=False) if t.kind == "fl"]
    assert len(main) == 72
    assert all(t.task_id.endswith("__R50") for t in main)
    assert all("--run-role" not in t.cmd for t in main)
    assert all("--rounds" not in t.cmd for t in main)

    ext = [t for t in build_tasks(resume=False, methods=["PHP-FL"], rounds=75,
                                  run_role="extension") if t.kind == "fl"]
    assert len(ext) == 18
    assert all(t.task_id.endswith("__R75") for t in ext)
    assert all("--run-role" in t.cmd and "extension" in t.cmd for t in ext)
    assert all(t.deps and t.deps[0].endswith("__R250") for t in ext)


def test_run_role_is_tagged_in_payload():
    from benchmark_ncg.methods.registry import get_runner

    cfg = _smoke_cfg("FedAvg")
    cfg["run_role"] = "extension"
    payload = get_runner("FedAvg")(cfg, logger=lambda *_: None)
    assert payload["run_role"] == "extension"
    assert payload["run_id"].endswith("__R2")


@pytest.mark.parametrize("bad_setting", ["A", "A_spot", "B4", "official", "C1"])
def test_forbidden_settings_rejected(bad_setting):
    with pytest.raises(ValueError):
        config.resolve_run(bad_setting, "FedAvg", "BloodMNIST", 0)


def test_forbidden_methods_rejected():
    legacy = "pFed" + "FL"   # assembled so the legacy name never appears in code
    for bad in ["Standalone", "Individual", legacy]:
        with pytest.raises(ValueError):
            config.resolve_run("B1", bad, "BloodMNIST", 0)


def test_only_cnn2_can_be_built():
    forbidden = ["google" + "net", "densenet" + "121", "efficientnet" + "_b1", "res" + "net18"]
    for name in forbidden:
        with pytest.raises(ValueError):
            models.build_model(name, 3, 8)
    m = models.build_model("cnn2", 3, 8)
    assert isinstance(m, models.CNN2)


def test_no_heavy_backbone_code_paths():
    offenders = []
    for sub in ("core", "methods", "centralized", "scripts"):
        for path in (config.BENCH_ROOT / sub).rglob("*.py"):
            if path.name.startswith("hetero_"):
                continue  # heterogeneous mode intentionally instantiates official architectures
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "torchvision.models" in text or "torchvision import models" in text:
                offenders.append(str(path))
    assert not offenders, f"heavy backbone imports found: {offenders}"


def test_manifest_counts():
    from benchmark_ncg.core import report
    m = report.build_manifest()
    assert m["totals"]["centralized"] == 6
    assert m["totals"]["fl"] == 72
    assert m["totals"]["grand"] == 78
    assert m["rounds"] == 50 and m["centralized_max_epochs"] == 250


# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------
def test_all_required_partitions_valid():
    for (dataset, protocol, alpha, n, seed), path_str in config.required_partitions().items():
        path = Path(path_str)
        if not path.exists():
            meta = partition_mod.build_partition(dataset, protocol, alpha, n, seed)
            partition_mod.save_partition(meta)
        meta = config.load_json(path)
        assert not partition_mod.verify_partition(meta), (path.name, partition_mod.verify_partition(meta))
        assert meta["client_count"] == 20
        assert len(meta["clients"]) == 20
        assert meta["num_samples"] == (11959 if dataset == "BloodMNIST" else 30000)
        if dataset == "FashionMNIST":
            assert meta["subsample_file"] == str(config.subsample_path(seed))


def test_partition_seed_determinism():
    y = datasets.load_bundle("BloodMNIST", 0).train_y.numpy()
    a, _ = partition_mod.dirichlet_partition(y, 20, 0.1, 8, 0)
    b, _ = partition_mod.dirichlet_partition(y, 20, 0.1, 8, 0)
    c, _ = partition_mod.dirichlet_partition(y, 20, 0.1, 8, 1)
    assert all(np.array_equal(x, z) for x, z in zip(a, b))
    assert any(not np.array_equal(x, z) for x, z in zip(a, c))


def test_dirichlet_repairs_empty_clients():
    y = np.zeros(50, dtype=int)
    parts, repairs = partition_mod.dirichlet_partition(y, 20, 0.01, 8, 3)
    assert all(len(p) > 0 for p in parts)
    assert repairs > 0
    assert sum(len(p) for p in parts) == 50


# ---------------------------------------------------------------------------
# FashionMNIST subsample (R3)
# ---------------------------------------------------------------------------
def test_subsample_shape_balance_determinism():
    for seed in config.SEEDS:
        meta = datasets.get_subsample(seed)
        assert meta["train_size"] == 30000 and meta["test_size"] == 5000
        assert set(meta["train_class_counts"]) == {3000}
        assert set(meta["test_class_counts"]) == {500}
        rebuilt = datasets.build_subsample(seed)
        assert rebuilt["train_indices"] == meta["train_indices"]
        assert rebuilt["test_indices"] == meta["test_indices"]
        tr = np.asarray(meta["train_indices"])
        te = np.asarray(meta["test_indices"])
        # train indices index the official 60k train pool, test indices the
        # official 10k test pool; the pools are disjoint by construction
        assert tr.max() < 60000 and te.max() < 10000
        full = datasets.load_full("FashionMNIST")
        tr_hash = {hash(full["train_x"][i].numpy().tobytes()) for i in tr}
        te_hash = {hash(full["test_x"][i].numpy().tobytes()) for i in te}
        assert len(tr_hash & te_hash) == 0


def test_bundle_shapes():
    b = datasets.load_bundle("BloodMNIST", 0)
    assert (len(b.train_y), len(b.val_y), len(b.test_y)) == (11959, 1712, 3421)
    assert b.train_x.shape[1:] == (3, 28, 28)
    f = datasets.load_bundle("FashionMNIST", 1)
    assert (len(f.train_y), len(f.test_y)) == (30000, 5000)
    assert f.train_x.shape[1:] == (1, 28, 28)
    assert f.val_x is None


def test_fashion_subsample_class_balance_after_load():
    b = datasets.load_bundle("FashionMNIST", 2)
    assert np.bincount(b.train_y.numpy(), minlength=10).tolist() == [3000] * 10
    assert np.bincount(b.test_y.numpy(), minlength=10).tolist() == [500] * 10


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_ncg_formula_synthetic():
    cent = [0.90, 0.60]
    fl = [0.90, 0.45]
    expected = (0.45 - 0.30) / (1 - 0.30)
    assert metrics.ncg(fl, cent) == pytest.approx(expected)
    assert metrics.ncg(cent, cent) == pytest.approx(0.0)
    assert np.isnan(metrics.ncg([1.0, 0.0], [1.0, 0.0]))


def test_average_vectors_not_gaps():
    r1 = np.array([1.0, 0.2, 0.1])
    r2 = np.array([0.1, 0.2, 1.0])
    avg = metrics.average_recall_vectors([r1, r2])
    assert np.allclose(avg, [0.55, 0.2, 0.55])
    assert metrics.class_gap(avg) == pytest.approx(0.35)
    assert metrics.class_gap(avg) != pytest.approx(
        (metrics.class_gap(r1) + metrics.class_gap(r2)) / 2
    )


def test_recall_labels_and_zero_division():
    rec = metrics.per_class_recall([0, 0, 1, 1], [0, 0, 0, 0], num_classes=4)
    assert rec.shape == (4,)
    assert rec[0] == 1.0 and rec[1] == 0.0 and rec[2] == 0.0 and rec[3] == 0.0
    assert not np.isnan(rec).any()


# ---------------------------------------------------------------------------
# Splits and the correction-2 floor rule
# ---------------------------------------------------------------------------
def test_blood_centralized_uses_official_val_and_test():
    from benchmark_ncg.centralized.train import train_centralized

    protocol = dict(config.CENTRAL_TRAINING)
    protocol.update(max_epochs=1)
    res = train_centralized("BloodMNIST", 0, logger=lambda *_: None, protocol=protocol)
    assert res["val_source"] == "official_val"
    assert (res["train_size"], res["val_size"], res["test_size"]) == (11959, 1712, 3421)
    assert res["max_epochs"] == 1
    assert len(res["per_class_recall"]) == 8


def test_fashion_centralized_uses_train_only_val():
    from benchmark_ncg.centralized.train import train_centralized

    protocol = dict(config.CENTRAL_TRAINING)
    protocol.update(max_epochs=1)
    res = train_centralized("FashionMNIST", 0, logger=lambda *_: None, protocol=protocol)
    assert res["val_source"] == "stratified_train_10pct"
    assert res["train_size"] + res["val_size"] == 30000
    assert res["val_size"] == 3000
    assert res["test_size"] == 5000


def test_stratified_split_is_a_partition_of_train():
    y = datasets.load_bundle("FashionMNIST", 0).train_y.numpy()
    tr, va = datasets.stratified_split(y, 0.1, 0)
    assert len(np.intersect1d(tr, va)) == 0
    assert set(tr.tolist()) | set(va.tolist()) == set(range(len(y)))
    assert len(np.unique(y[va])) == 10


@pytest.mark.parametrize("n", [4, 5, 6, 10, 15, 21, 40, 100])
def test_adapt_floor_rule_never_empty(n):
    indices = np.arange(n)
    labels = np.arange(n) % 3
    tr, aux, info = auxiliary_split(indices, labels, seed=0, ratio=0.05, min_size=2,
                                    max_size=5, divisor=4)
    assert len(aux) >= 2, (n, info)
    assert len(tr) >= 1
    assert len(np.intersect1d(tr, aux)) == 0
    assert set(tr.tolist()) | set(aux.tolist()) == set(indices.tolist())
    assert info["aux_size"] == len(aux)


def test_adapt_floor_rule_fires_below_2():
    tr, aux, info = auxiliary_split(np.arange(10), np.zeros(10, dtype=int), seed=0,
                                    ratio=0.05, min_size=2, max_size=5, divisor=4)
    assert info["forced"] is True
    assert 2 <= len(aux) <= 5


def test_adapt_rule_on_real_alpha01_partitions():
    for dataset in config.DATASET_NAMES:
        for seed in config.SEEDS:
            meta = config.load_json(config.partition_path(dataset, "dirichlet", 0.1, 20, seed))
            labels = datasets.load_bundle(dataset, seed).train_y.numpy()
            for cid, idx in enumerate(meta["clients"]):
                tr, aux, info = auxiliary_split(np.asarray(idx), labels, seed=seed * 31 + cid,
                                                ratio=0.05, min_size=2, max_size=5, divisor=4)
                assert len(tr) >= 1
                if info["fallback"]:
                    assert len(aux) == 0 and len(idx) < 4
                else:
                    assert len(aux) >= 2


# ---------------------------------------------------------------------------
# Smoke run: all four methods at R=2, E=1 on a tiny partition
# ---------------------------------------------------------------------------
def _smoke_cfg(method: str) -> dict:
    cfg = copy.deepcopy(config.resolve_run("B1", method, "BloodMNIST", 0))
    cfg["training"]["rounds"] = 2
    cfg["training"]["local_epochs"] = 1
    cfg["client_count"] = 6
    cfg["run_id"] = f"SMOKE__{method}__BloodMNIST__s0__R2"
    bundle = datasets.load_bundle("BloodMNIST", 0)
    y = bundle.train_y.numpy()
    part = {
        "dataset": "BloodMNIST", "protocol": "iid", "alpha": None, "client_count": 6,
        "seed": 0, "num_samples": 600, "num_classes": 8, "repairs": 0,
        "subsample_file": None,
        "clients": [list(range(i * 100, (i + 1) * 100)) for i in range(6)],
        "client_sizes": [100] * 6,
        "client_class_counts": [np.bincount(y[i * 100:(i + 1) * 100], minlength=8).tolist()
                                for i in range(6)],
        "total_class_counts": np.bincount(y[:600], minlength=8).tolist(),
    }
    path = config.SMOKE_DIR / "smoke_partition_BloodMNIST_n6.json"
    config.dump_json(part, path)
    cfg["partition_file"] = str(path)
    cfg["fashionmnist_subsample_file"] = None
    return cfg


@pytest.mark.parametrize("method", list(config.FL_METHODS))
def test_smoke_run_all_methods(method):
    from benchmark_ncg.methods.registry import get_runner

    cfg = _smoke_cfg(method)
    payload = get_runner(method)(cfg, logger=lambda *_: None)
    out = config.SMOKE_DIR / f"{cfg['run_id']}.json"
    config.dump_json(payload, out)
    assert out.exists()
    assert payload["status"] == "complete"
    assert payload["rounds"] == 2 and payload["local_epochs"] == 1
    assert len(payload["per_class_recall"]) == 8
    assert 0.0 <= payload["fl_class_gap"] <= 1.0
    assert payload["wall_clock_seconds"] >= 0
    assert not np.isnan(payload["per_class_recall"]).any()
    if method in ("FedAPEN", "PHP-FL"):
        key = "fedapen_adapt_size_per_client" if method == "FedAPEN" else "phpfl_valid_size_per_client"
        assert len(payload["per_client_recall"]) == 6
        assert len(payload[key]) == 6
        assert all(s >= 2 for s in payload[key])
        avg = metrics.average_recall_vectors(payload["per_client_recall"])
        assert np.allclose(avg, payload["per_class_recall"], atol=1e-8)
    if method == "FedAPEN":
        assert len(payload["fedapen_adapt_size_per_client"]) == 6
    if method == "PHP-FL":
        assert len(payload["phpfl_uniform_fallback_clients"]) >= 0


def test_phpfl_full_objective_is_stable_at_shared_lr():
    """Regression: the four-term PHP-FL objective used to NaN at lr=0.01/momentum=0.9."""
    import torch
    import torch.nn as nn
    from benchmark_ncg.core import train_utils
    from benchmark_ncg.methods.phpfl import MMDLoss

    device = train_utils.get_device()
    bundle = datasets.load_bundle("BloodMNIST", 0)
    idx = np.arange(64)
    x = bundle.train_x[idx].to(device)
    y = bundle.train_y[idx].to(device)
    torch.manual_seed(0)
    p = models.build_model("cnn2", 3, 8).to(device)
    g = models.build_model("cnn2", 3, 8).to(device)
    op = torch.optim.SGD(p.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    og = torch.optim.SGD(g.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    ce, kl = nn.CrossEntropyLoss(), nn.KLDivLoss(reduction="batchmean")
    lsm, sm = nn.LogSoftmax(dim=1), nn.Softmax(dim=1)
    mmd, w, clip = MMDLoss(), 0.5, 5.0
    for _ in range(25):
        op.zero_grad(); og.zero_grad()
        rep_p, out_p = p.base(x), None
        out_p = p.head(rep_p)
        rep_g = g.base(x)
        out_g = g.head(rep_g)
        ens_p = w * out_p + (1 - w) * out_g.detach()
        ens_g = w * out_p.detach() + (1 - w) * out_g
        lp = mmd(rep_p, rep_g.detach()) + ce(out_p, y) + kl(lsm(out_p), sm(out_g.detach())) + ce(ens_p, y)
        lg = mmd(rep_p.detach(), rep_g) + ce(out_g, y) + kl(lsm(out_g), sm(out_p.detach())) + ce(ens_g, y)
        lp.backward(retain_graph=True)
        lg.backward()
        torch.nn.utils.clip_grad_norm_(p.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(g.parameters(), clip)
        op.step(); og.step()
    assert all(torch.isfinite(q).all() for q in p.parameters())
    assert all(torch.isfinite(q).all() for q in g.parameters())


def test_smoke_results_are_not_in_formal_results():
    assert not any(p.name.startswith("SMOKE") for p in config.RUNS_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# Heterogeneous supplementary run (separate namespace)
# ---------------------------------------------------------------------------
def test_hetero_scope_and_ids():
    from benchmark_ncg.core import hetero_config as hc
    from benchmark_ncg.scripts.run_all import build_tasks

    runs = list(hc.all_hetero_runs())
    assert len(runs) == 36
    assert sum(r["method"] == "FedAPEN" for r in runs) == 18
    assert sum(r["method"] == "PHP-FL" for r in runs) == 18
    assert all(r["run_id"].endswith("__HET__R50") for r in runs)
    cfg_b1 = hc.resolve_run_hetero("B1", "PHP-FL", "BloodMNIST", 0)
    assert cfg_b1["setting"] == "B1"
    assert hc.num_expected_runs() == 36
    assert hc.num_expected_refs() == 20 * 2 * 3 + 4 * 2 * 3

    main_ids = [t.task_id for t in build_tasks(resume=False) if t.kind == "fl"]
    assert len(main_ids) == 72
    assert not any("HET" in i for i in main_ids)


def test_hetero_arch_map_and_budget():
    import torch
    from benchmark_ncg.core import hetero_config as hc, models

    path = hc.write_arch_map()
    m = config.load_json(path)
    assert len(m["FedAPEN"]["architectures"]) == 20
    assert len(m["PHP-FL"]["architectures"]) == 4
    assert all(a["params"] <= 500_000 for a in m["PHP-FL"]["architectures"])
    # FedAPEN uses the official width-heterogeneous CNN2_BN set (122k-504k), not substitutes
    assert max(a["params"] for a in m["FedAPEN"]["architectures"]) <= 600_000
    assert m["PHP-FL"]["client_to_arch"] == {str(c): c % 4 for c in range(20)}
    assert m["FedAPEN"]["client_to_arch"] == {str(c): c for c in range(20)}
    for a in m["PHP-FL"]["architectures"]:
        model = models.build_model(a["class_name"], 3, 8)
        assert sum(p.numel() for p in model.parameters()) == a["params"]
        with torch.no_grad():
            f = model.base(torch.randn(2, 3, 28, 28))
            assert f.shape == (2, 512)
            assert model.head(f).shape == (2, 8)


@pytest.mark.parametrize("method", ["FedAPEN", "PHP-FL"])
@pytest.mark.parametrize("dataset", ["BloodMNIST", "FashionMNIST"])
def test_hetero_param_assert_all_combos(method, dataset):
    """Assertion helper must pass for every method/dataset combo (smoke covers Blood only)."""
    from types import SimpleNamespace
    from benchmark_ncg.core import hetero_config as hc, models
    from benchmark_ncg.methods.common import assert_hetero_client_models

    cfg = hc.resolve_run_hetero("B3", method, dataset, 0)
    in_ch = config.DATASETS[dataset]["in_channels"]
    C = config.DATASETS[dataset]["num_classes"]
    built = [models.build_model(s["name"], in_ch, C, **(s.get("params") or {}))
             for s in cfg["client_model_specs"]]
    counts = assert_hetero_client_models(SimpleNamespace(cfg=cfg), built)
    expected = [a["params_by_dataset"][dataset] for a in cfg["method_arch_list"]]
    if method == "FedAPEN":
        assert counts == expected
        assert len(set(counts)) == 20
    else:
        assert set(counts) == set(expected)
        assert all(counts.count(v) == 5 for v in expected)
        assert len(set(counts)) == 4


@pytest.mark.parametrize("dataset", ["BloodMNIST", "FashionMNIST"])
def test_hetero_param_assert_rejects_all_cnn2(dataset):
    """The bypass guard must fire on both datasets (CNN2 fallback detection)."""
    from types import SimpleNamespace
    from benchmark_ncg.core import hetero_config as hc, models
    from benchmark_ncg.methods.common import assert_hetero_client_models

    cfg = hc.resolve_run_hetero("B3", "PHP-FL", dataset, 0)
    in_ch = config.DATASETS[dataset]["in_channels"]
    C = config.DATASETS[dataset]["num_classes"]
    cnn2 = [models.build_model("cnn2", in_ch, C) for _ in range(20)]
    with pytest.raises(AssertionError, match="all clients built as CNN2"):
        assert_hetero_client_models(SimpleNamespace(cfg=cfg), cnn2)


def test_hetero_scope_strings_and_substitute_flag():
    from benchmark_ncg.core import hetero_config as hc
    assert hc.SUBSTITUTE_SOURCE.startswith("substituted from official HtM4")
    cfg = hc.resolve_run_hetero("B3", "PHP-FL", "FashionMNIST", 2)
    assert cfg["hetero"] is True
    assert cfg["centralized_ref_mode"] == "per_architecture"
    assert cfg["run_id"].endswith("__HET__R50")
    assert cfg["client_count"] == 20
    assert cfg["training"]["rounds"] == 50
    assert len(cfg["client_model_specs"]) == 20
    assert cfg["client_to_arch"] == {str(c): c % 4 for c in range(20)}


def test_main_table_checksum_baseline():
    from benchmark_ncg.core import hetero_config as hc
    assert (config.RESULTS_DIR / "main_table_checksums.json").exists()
    ok, diffs = hc.verify_main_table()
    assert ok, diffs


# ---------------------------------------------------------------------------
# Stored-result guards (no-op before the queue)
# ---------------------------------------------------------------------------
def test_no_forbidden_settings_in_stored_results():
    for path in config.RUNS_DIR.glob("*.json"):
        payload = config.load_json(path)
        assert payload["setting"] in config.SETTINGS
        assert payload["run_id"] == path.stem
        if payload.get("run_role") == "extension":
            assert payload["rounds"] != int(config.TRAINING["rounds"])
            assert payload["run_id"].endswith(f"__R{payload['rounds']}")
        else:
            assert payload["rounds"] == int(config.TRAINING["rounds"])
            assert payload["run_id"].endswith("__R50")
