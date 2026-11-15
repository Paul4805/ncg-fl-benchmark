# Normalized Class Gap (NCG): Federated Learning vs Centralized Training

Benchmark measuring whether federated learning leaves a larger **class gap** than
centralized training, and whether that gap grows as label skew worsens, on
**BloodMNIST** and a subsampled **FashionMNIST**.

## Hypothesis

> NCG should be **near 0 under IID**, **positive under non-IID**, and **increase
> as alpha decreases** for FedAvg, FedAPEN, FedFV and PHP-FL, relative to
> centralized learning.

A negative NCG means the FL method has a *smaller* class gap than centralized
training; such results are reported as-is.

## Methods

| method | implementation | personalization |
|---|---|---|
| FedAvg | standard sample-weighted Federated Averaging | none (global model) |
| FedAPEN | official `learned_adaptive_training` (KDD 2023, [zhenqincn/FedAPEN](https://github.com/zhenqincn/FedAPEN)) | `w * private + (1-w) * shared` per client |
| FedFV | official easyFL aggregation rule (IJCAI 2021, [WwZzz/easyFL](https://github.com/WwZzz/easyFL)) | none (global model) |
| PHP-FL | official `clientPHP` / `serverPHP` (NeurIPS 2025, [Siyuan01/PHP-FL-main](https://github.com/Siyuan01/PHP-FL-main)) | `w * private + (1-w) * personal_global` per client |
| Centralized | single CNN-2 on the fully aggregated training set | n/a |

PHP-FL is the NeurIPS 2025 paper *"A Fair Federated Learning Method for Handling
Client Participation Probability Inconsistencies in Heterogeneous
Environments"* ([poster](https://neurips.cc/virtual/2025/loc/san-diego/poster/115969)).

## Setup

| item | value |
|---|---|
| settings | B1 IID, B2 Dirichlet alpha=0.5, B3 Dirichlet alpha=0.1 |
| clients | 20, 100 % participation |
| rounds / local epochs | R = 50, E = 5 |
| batch / optimizer | 64, SGD momentum 0.9, weight decay 1e-4, lr 0.01 cosine |
| backbone | CNN-2 (2 conv layers, 28x28 input) for every model |
| seeds | 0, 1, 2 |
| BloodMNIST | official split: 11,959 train / 1,712 val / 3,421 test |
| FashionMNIST | per-seed subsample: 30,000 train / 5,000 test (3,000 / 500 per class) |
| centralized | CNN-2, max 250 epochs, early stopping patience 20, best-val checkpoint |

The centralized reference is **one CNN-2 model per (dataset, seed)** and is shared
by all four methods and all three settings. No per-architecture or
backbone-averaged reference is used.

## Metric

```
recall_c   = TP_c / (TP_c + FN_c)          # labels=range(C), zero_division=0
ClassGap   = max_c(recall_c) - min_c(recall_c)
NCG        = (ClassGap_FL - ClassGap_Centralized) / (1 - ClassGap_Centralized)
```

Per-class recall **vectors** are always averaged first (across deployed client
models for FedAPEN/PHP-FL, across client architectures where applicable), then the
class gap is computed. Gaps are never averaged.

## Results (mean NCG over 3 seeds, R=50, CNN-2)

| dataset | method | B1 (IID) | B2 (alpha=0.5) | B3 (alpha=0.1) |
|---|---|---|---|---|
| BloodMNIST | FedAvg | +0.098 | +0.086 | +0.624 |
| BloodMNIST | FedAPEN | +0.039 | +0.037 | +0.315 |
| BloodMNIST | FedFV | +0.100 | +0.082 | +0.375 |
| BloodMNIST | PHP-FL | +0.338 | +0.379 | +0.102 |
| FashionMNIST | FedAvg | +0.057 | +0.073 | +0.090 |
| FashionMNIST | FedAPEN | +0.031 | +0.085 | +0.378 |
| FashionMNIST | FedFV | +0.058 | +0.070 | +0.237 |
| FashionMNIST | PHP-FL | +0.131 | +0.169 | +0.156 |

Top-1 and balanced accuracy are reported alongside NCG in
`results/summary_accuracy_hom.csv`; NCG must be read jointly with accuracy
(a low NCG from a uniformly weak model is not a fairness improvement). See
`benchmark_ncg/RESULT_NOTE.md` for the full discussion.

## Repository structure

```
.
├── README.md                      <- this file
├── .gitignore
└── benchmark_ncg/
    ├── README.md                  <- detailed protocol and reproduction commands
    ├── DEVIATIONS.md              <- every deviation from the original spec/papers
    ├── RESULT_NOTE.md             <- headline numbers, accuracy/NCG discussion
    ├── configs/                   <- benchmark.json, methods.json (single source of truth)
    ├── core/                      <- config, datasets, partitioning, metrics, models, report
    ├── methods/                   <- fedavg, fedfv, fedapen, phpfl + shared plumbing
    ├── centralized/               <- CNN-2 centralized training with early stopping
    ├── scripts/                   <- partitions, training, evaluation, NCG, summaries, plots
    ├── tests/                     <- 42 validation tests
    ├── partitions/                <- 18 client partitions + 3 FashionMNIST subsample files
    └── results/
        ├── runs/                  <- 72 FL run records (config + recalls + metrics)
        ├── centralized/           <- 6 centralized references
        ├── plots/                 <- NCG-vs-setting and B3 recall heatmaps
        ├── raw_results.csv        <- 78 rows (72 FL + 6 centralized)
        ├── summary_ncg.csv        <- 24 rows (3 settings x 4 methods x 2 datasets)
        ├── summary_class_gap.csv
        ├── summary_accuracy_hom.csv
        └── manifest.json / .csv   <- 78 jobs, complete/pending
```

Datasets, model tensors and per-epoch logs are generated locally and are not
committed (`benchmark_ncg/data/`, `benchmark_ncg/logs/`).

## Quick start

```bash
# partitions + per-seed FashionMNIST subsamples
python -m benchmark_ncg.scripts.generate_partitions

# validation suite (42 tests)
python -m pytest benchmark_ncg/tests -q

# six centralized CNN-2 references (max 250 epochs each)
python -m benchmark_ncg.scripts.train_centralized --all

# convergence gate (FedAvg R=50 vs R=100, PHP-FL R=50 on both alpha legs)
python -m benchmark_ncg.scripts.convergence_check

# 72 B1-B3 FL runs, 2 workers, resumable
python -m benchmark_ncg.scripts.run_all --stage fl --workers 2 --resume

# derived tables, plots and notes
python -m benchmark_ncg.scripts.compute_ncg
python -m benchmark_ncg.scripts.make_summary
python -m benchmark_ncg.scripts.plot_ncg
python -m benchmark_ncg.scripts.add_accuracy
python -m benchmark_ncg.scripts.make_result_note
```

## Documentation

- [`benchmark_ncg/README.md`](benchmark_ncg/README.md) - full protocol, commands, artifacts
- [`benchmark_ncg/DEVIATIONS.md`](benchmark_ncg/DEVIATIONS.md) - R=50, FashionMNIST subsample, method-specific adaptations
- [`benchmark_ncg/RESULT_NOTE.md`](benchmark_ncg/RESULT_NOTE.md) - headline numbers, accuracy/NCG regimes, convergence note
