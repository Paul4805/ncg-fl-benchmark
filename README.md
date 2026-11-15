# NCG-FL Benchmark

## What this repository is

A reproducible benchmark that measures whether federated learning (FL) methods
leave a larger per-class performance gap than centralized training on the same
data. We introduce the Normalized Class Gap (NCG) as the metric and report it for
four FL methods (FedAvg, FedAPEN, FedFV, PHP-FL) across two datasets (BloodMNIST,
FashionMNIST) under three data distributions (IID, Dirichlet alpha=0.5, Dirichlet
alpha=0.1). Every run starts from a committed client partition, a fixed seed and a
fully specified configuration, and every number in `results/` can be recomputed
from the run JSONs in the same directory.

## Why class gap needs its own metric

**The gap that accuracy hides.** Mean accuracy conflates two failure modes:
models that are uniformly mediocre, and models that are excellent on some classes
but fail catastrophically on others. On naturally imbalanced or Dirichlet-skewed
data the second failure mode is common and fairness-relevant, but it is invisible
in top-1 accuracy. Two methods can report the same mean accuracy while producing
very different per-class recall distributions.

**Why FL is exposed to this.** In federated learning each client optimizes on a
local shard that may contain only a subset of the classes. Aggregation or
personalization mechanisms must reconcile these biased local views, and the
reconciliation does not always preserve class balance. The result is that a
global FL model can learn some classes well and effectively abandon others, even
when its mean accuracy is competitive with centralized training.

**The measurement gap.** Existing FL fairness benchmarks report mean accuracy,
worst-client accuracy, or variance across clients. All three are client-level
metrics and can be high while the model fails on specific classes globally. What
is missing is a class-level comparison against a centralized reference trained on
the fully aggregated data - the gold standard for what a model "should" learn
from that data.

## The NCG metric

### Definition

For any model evaluated on the global test set:

```
recall_c  = TP_c / (TP_c + FN_c)        # per class c, zero_division=0
ClassGap  = max_c(recall_c) - min_c(recall_c)

NCG = (ClassGap_FL - ClassGap_Centralized) / (1 - ClassGap_Centralized)
```

Per-class recall vectors are always averaged **before** a class gap is computed
(across deployed client models for personalized methods); gaps themselves are
never averaged.

### Interpretation

| NCG value | Meaning |
|---|---|
| `= 0` | FL matches the centralized class gap |
| `> 0` | FL leaves a larger class gap (worse fairness) |
| `< 0` | FL leaves a smaller class gap (report it, do not hide it) |

### Why normalize

Raw class gap is dataset-dependent. A class gap of 0.30 is alarming on a dataset
where centralized training achieves 0.15, but unremarkable on a dataset where
centralized training also achieves 0.30. Subtracting the centralized class gap
and dividing by `1 - centralized class gap` puts every dataset and method on the
same scale: NCG = 0 always means "FL is exactly as class-fair as centralized
training on this data", and NCG = 1 always means "FL has the maximum possible
class gap", independent of the dataset's intrinsic imbalance.

### Why a centralized reference per dataset

Centralized training on the fully aggregated data is the best model the data can
produce without federation. Any class gap that centralized training leaves is
intrinsic to the data and should not be counted against the FL method. NCG
measures the gap attributable to federation, not the gap attributable to the
dataset.

### A worked example

Suppose centralized training reaches a class gap of 0.20 on a dataset and an FL
method reaches 0.45. The raw difference (0.25) says nothing about how much room
there was to begin with. Normalizing gives

```
NCG = (0.45 - 0.20) / (1 - 0.20) = 0.3125
```

so the FL method fills roughly 31% of the remaining class gap; if the FL model had
instead matched the centralized gap of 0.20, NCG would be exactly 0. The same
computation on a dataset whose centralized gap is already 0.60 would map an FL gap
of 0.75 to NCG = 0.375, not 0.15 - which is why the raw difference is not
comparable across datasets and the normalized value is.

## What is benchmarked

| Method | Family | Official code |
|---|---|---|
| FedAvg | Aggregation baseline | - (standard) |
| FedAPEN | Personalized, shared/private with mutual learning | https://github.com/zhenqincn/FedAPEN |
| FedFV | Fair aggregation with gradient conflict correction | https://github.com/WwZzz/easyFL |
| PHP-FL | Fair FL for client participation inconsistencies (NeurIPS 2025) | https://github.com/Siyuan01/PHP-FL-main |

FedAvg is the sample-count weighted average of client weights, with no
personalization. FedAPEN and PHP-FL deploy a personalized ensemble per client -
`w * private + (1 - w) * shared` for FedAPEN and `w * private + (1 - w) *
personal_global` for PHP-FL, with `w` learned on a small local adaptation split -
and their per-class recall vectors are averaged over clients before the gap is
taken. PHP-FL is the canonical name used here; "pFedFL" in some earlier internal
drafts refers to the same method.

## Datasets

**BloodMNIST.** 8 classes, 28x28 RGB, official MedMNIST v2 split (11,959 train /
1,712 val / 3,421 test). The natural class imbalance is preserved exactly as
published. The validation split is used only for centralized early stopping, and
the test split is never partitioned, never trained on, and never used for model
selection.

**FashionMNIST (subsampled).** 10 classes, 28x28 grayscale. A stratified subsample
of 30,000 train / 5,000 test images is drawn once per seed from the official
60,000 / 10,000 pools (3,000 and 500 per class), and the indices are committed
under `partitions/`. Normalization uses mean 0.2860 and std 0.3530. The subsample
bounds compute while keeping the class distribution exactly balanced, which
isolates the effect of Dirichlet partitioning from the effect of intrinsic class
imbalance.

## Experimental setup

All methods share one protocol; only the aggregation/personalization rule differs.

- 20 clients, 100% participation in every round.
- R = 50 communication rounds, E = 5 local epochs.
- Batch size 64, SGD with momentum 0.9 and weight decay 1e-4.
- Learning rate 0.01 with cosine decay over rounds.
- Backbone: 2-convolution CNN for 28x28 input (44,584 params on BloodMNIST,
  50,282 on FashionMNIST), identical for every client and every method.
- Seeds 0, 1, 2; every cell is reported as mean +/- std over the three seeds.
- Partitions are generated once per (dataset, setting, seed) and committed:
  - IID - shuffled, equally sized shards;
  - Dirichlet alpha=0.5 - moderate label skew;
  - Dirichlet alpha=0.1 - severe label skew.
- Standalone and local-only baselines are not evaluated.

The three settings appear as `B1`, `B2`, `B3` in every results file:

| Code | Label | Partition |
|---|---|---|
| `B1` | IID | Shuffled, equally sized client shards |
| `B2` | Moderate label skew | Dirichlet alpha = 0.5 |
| `B3` | Severe label skew | Dirichlet alpha = 0.1 |

## The centralized reference

For each (dataset, seed), one centralized model is trained on the fully
aggregated training set with the same backbone, optimizer, and batch size as the
FL runs. Centralized training uses a 250-epoch cap (= R x E) with a held-out
validation split for early stopping - the official MedMNIST validation split for
BloodMNIST, and a stratified 10% of the training set for FashionMNIST - and the
best-validation checkpoint is restored. The test set is never used for model
selection.

The 6 centralized models (2 datasets x 3 seeds) are trained once and reused as
the NCG denominator for all 24 FL cells that share their (dataset, seed), so every
method is compared against the same reference on the same data.

## How to reproduce

Run from the repository root; datasets are downloaded and cached under `data/`
on first use.

```bash
# 1. Partitions and validation
python scripts/generate_partitions.py
pytest tests/ -v

# 2. Centralized references (6 models)
python scripts/train_centralized.py

# 3. FL runs (72 runs: 3 settings x 4 methods x 2 datasets x 3 seeds)
python scripts/run_all.py --workers 2 --resume

# 4. Summaries and plots
python scripts/compute_ncg.py
python scripts/make_summary.py
python scripts/add_accuracy.py
python scripts/plot_ncg.py
```

`run_all.py --resume` skips any job whose output JSON already exists and stops
launching new jobs at the configured wall-clock budget, so the pipeline can be
interrupted and restarted. Individual runs can also be launched directly, for
example:

```bash
python scripts/train_fl.py --setting B3 --method FedAPEN --dataset BloodMNIST --seed 0
python scripts/train_centralized.py --dataset BloodMNIST --seed 0
```

## Outputs

| File | Contents |
|---|---|
| `results/summary_ncg.csv` | NCG per (setting, method, dataset), mean +/- std over 3 seeds |
| `results/summary_class_gap.csv` | Class gap per cell (FL and centralized) |
| `results/summary_accuracy_hom.csv` | Top-1 and balanced accuracy per cell |
| `results/raw_results.csv` | Per-run per-class recall vectors, configs, gaps, timings |
| `results/runs/` | 72 FL run JSONs (config + recalls + metrics) |
| `results/centralized/` | 6 centralized reference JSONs |
| `results/plots/ncg_vs_setting.png` | Headline figure: NCG vs skew, per method and dataset |
| `results/plots/recall_heatmap_B3_*.png` | Per-class recall at alpha=0.1 |
| `RESULT_NOTE.md` | Headline numbers, interpretation, limitations |
| `DEVIATIONS.md` | Documented deviations from the official settings (R=50, subsample, etc.) |

Datasets, model tensors and per-epoch logs are generated locally and are not
committed (`data/`, `logs/`).

### Column reference

`results/summary_ncg.csv` has 24 rows - one per (setting, method, dataset):

| Column | Meaning |
|---|---|
| `setting` | `B1`, `B2` or `B3` |
| `method` | `FedAvg`, `FedAPEN`, `FedFV` or `PHP-FL` |
| `dataset` | `BloodMNIST` or `FashionMNIST` |
| `ncg_mean`, `ncg_std` | NCG across the three seeds (sample std, ddof=1) |
| `class_gap_mean`, `class_gap_std` | The FL class gap those NCG values came from |
| `num_seeds` | 3 |

`results/raw_results.csv` has 78 rows: the 72 FL runs plus the 6 centralized
references. Its `per_class_recall_json` column stores the full recall vector that
every gap and NCG value is derived from, so any reported number can be recomputed
without re-training.

## Scope and limitations

This benchmark evaluates homogeneous federated learning: every client uses the
same CNN-2 backbone, the same optimizer, and the same data pipeline, so the only
difference between cells is the FL method and the label distribution. The R=50
round budget and the FashionMNIST subsample were chosen to fit a fixed compute
budget and are documented in `DEVIATIONS.md`; neither changes the direction of the
reported NCG values. Standalone and local-only baselines are not evaluated, so the
benchmark answers "how much class gap does federation add relative to centralized
training", not "which method is best overall". NCG is a gap measure: a low NCG
obtained by a uniformly weak model is not a fairness improvement, which is why
`results/summary_accuracy_hom.csv` should always be read alongside the NCG table.
Nothing outside the homogeneous protocol described above is used to support the
results in this repository.
