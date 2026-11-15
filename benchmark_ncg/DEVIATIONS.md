# DEVIATIONS

Every deviation from the full specification, the paper configurations, or the
original prompt is listed here with its reason. Nothing is silent.

## Mandatory reductions

### R1 — Block A dropped entirely
The study contains **only** B1/B2/B3 homogeneous CNN-2 runs plus the 6
centralized CNN-2 references. There are no official-settings runs, no A_spot,
no per-architecture centralized references, no backbone-averaged references and
no model-heterogeneous settings (no C1/C2). No ImageNet-style backbone
(GoogLeNet / DenseNet / EfficientNet / ResNet family) is instantiated anywhere:
`core.models.build_model` accepts only `"cnn2"`, and
`tests/test_benchmark.py::test_only_cnn2_can_be_built` /
`::test_no_heavy_backbone_code_paths` enforce this.

### R2 — R = 50, centralized max_epochs = 250
All 72 FL runs use `rounds = 50`, `local_epochs = 5` (250 local epochs per
client in total). All 6 centralized models use `max_epochs = 250` with cosine
decay over 250 epochs, early stopping patience 20. `R=100` and `max_epochs=500`
appear nowhere; `core.config.assert_locked()` raises at import if these values
are changed. The only exception is a **diagnostic** FedAvg R=100 run used by the
convergence gate (correction 3); it is stored under `results/diagnostics/`,
never in `results/runs/`, and is excluded from every CSV and plot.

### R3 — FashionMNIST subsample
Per seed, the 30k train set is drawn from the **official 60k train pool**
(3,000 per class) and the 5k test set from the **official 10k test pool**
(500 per class), using `numpy.random.default_rng(seed)`. Rationale: the
requested 30k/5k shape is exactly half of the official 60k/10k split, so
halving each pool preserves the official train/test separation by construction
(zero train/test contamination; verified in
`test_subsample_shape_balance_determinism`, which also checks pixel-level
disjointness of the selected pools).
Indices are persisted once per seed to
`partitions/fashionmnist_subsample_seed{0,1,2}.json` and reused by every
setting and method. Client partition indices are positions inside that seed's
30k subsample. Because the subsample is per seed, the FashionMNIST test set
differs slightly between seeds; FL and centralized runs with the same seed are
always scored on the identical 5k set.

## Dataset split (locked before implementation)

### BloodMNIST — Option 1′
Verified from `bloodmnist.npz`: **train 11,959 / val 1,712 / test 3,421**
(total 17,092). The val/test sizes in the review note were transposed; the
assertions in `core/datasets._load_bloodmnist` use the real counts and abort on
any mismatch.

* FL runs train on the 11,959 train samples only and have **no validation set**
  (fixed R rounds, no early stopping).
* Centralized models train on all 11,959 train samples and use the **official
  val split (1,712)** for early stopping; the official **test split (3,421)** is
  used only for scoring.
* Deviation from the specification's "stratified 10 % validation split carved
  from TRAINING data": for BloodMNIST the validation set is the official val
  split instead of a train carve-out. This is the accepted Option 1′ and keeps
  the official test split untouched and comparable to published BloodMNIST
  numbers. The val split is never used by FL or for scoring.
* Consequently the BloodMNIST centralized model fits on 100 % of the train
  split, while the FashionMNIST centralized model fits on 90 % (10 % carved for
  early stopping, since FashionMNIST has no official val split).

### FashionMNIST
No official val split exists, so centralized early stopping uses a stratified
10 % split of the 30k train subsample (3,000 samples); scoring uses the 5k test
subsample. FL runs use the 30k train subsample only.

## Method implementations and method-level deviations

### Gradient-norm clipping for the multi-term personalised objectives (stability fix)
FedAPEN and PHP-FL both minimise a sum of CE + symmetric KL + ensemble CE (+ MMD
for PHP-FL) at the benchmark's shared `lr = 0.01` with `momentum = 0.9`. Measured
on a real client (BloodMNIST, α=0.5): CE-only local training is stable, the
three-term FedAPEN loss is stable but spiky, and the full four-term PHP-FL loss
**diverges to NaN at local step 19**. The first convergence diagnostic reproduced
this as a total collapse (deployed accuracy pinned at 0.0713 = the class-0 base
rate, `w_private_mean = nan`, gap = 1.0 for every round).

R4 fixes the shared optimizer, so instead of lowering the learning rate or
dropping momentum per method, both personalised methods apply a per-model
**gradient-norm clip of 5.0** before each SGD step (private and shared/personal
models separately). Learning rate, momentum, weight decay and batch size remain
exactly the mandated values. Verified stable: FedAPEN 0.46 → 0.80 accuracy in 5
rounds, PHP-FL 0.41 → 0.59, no non-finite parameters or weights. The clip value
is recorded in `configs/methods.json` (`grad_clip_norm`) and in each run's notes.

### All methods
* Backbone is the benchmark **CNN-2** everywhere (2 conv layers: 32/64 channels,
  3×3 kernels, BN, 2×2 max-pool, linear head), for private, shared and global
  models alike. The original prompt mandated this; the papers' own backbones are
  not used.
* Shared hyper-parameters everywhere: SGD, momentum 0.9, weight decay 1e-4,
  lr 0.01 with cosine decay over the 50 rounds, batch 64, 20 clients, 100 %
  participation, E = 5 local epochs, no local early stopping.
* No data augmentation anywhere; FL and centralized see identical inputs.
* Determinism: Python/NumPy/PyTorch/CUDA are seeded per run; cuDNN is set to
  deterministic mode; partition generation uses
  `numpy.random.default_rng(seed)`; minibatch order uses per-(round, client)
  seeded generators.

### FedAvg
No official repo exists, so the standard definition is used: sample-count
weighted averaging of client weights. FedFV/FedAPEN/PHP-FL use their official
(uniform) aggregation rules, so aggregation weighting differs across methods by
design.

### FedFV (easyFL)
The aggregation rule is re-implemented exactly from
`flgo/algorithm/fedfv.py`: deltas `w_global − w_local`, loss-before-training
sorting, `ceil((n−1)·(1−alpha))` highest-loss clients unprojected, iterative
negative-inner-product projection, external-conflict projection over `tau`
previous rounds, rescaling to the plain mean-delta norm, then
`w_global ← w_global − g_t`. Official `alpha = 0.1`, `tau = 1`. Local training
uses the B-block shared hyper-parameters (lr 0.01, momentum 0.9, wd 1e-4)
rather than easyFL's own defaults.

### FedAPEN (KDD 2023)
`learned_adaptive_training` is re-implemented at CNN-2 scale: private + shared
mutual learning (CE + symmetric KL + CE of the ensemble), uniform shared-model
averaging, and per-client scalar ensemble-weight learning on a 5 % adaptability
split (10 SGD steps, lr 1e-3, clipped to [0,1]). Deviations:
* Official `cnn2_bn` (and its width-heterogeneous variant) is replaced by CNN-2,
  as required by R1/B1-B3.
* Official exponential LR decay (0.99/round) is replaced by the shared cosine
  schedule.
* **Correction 2 (adaptability floor rule).** If `floor(0.05·n) < 2`, the
  adaptability set is forced to `min(5, max(2, n // 4))` samples (stratified by
  class where a class has ≥ 2 samples, else random), clamped to `n − 1` so at
  least one training sample remains. Clients with `n < 4` skip weight learning
  and use the uniform weight 0.5. Every run records
  `fedapen_adapt_size_per_client`, `fedapen_forced_rule_clients` and
  `fedapen_uniform_fallback_clients`.
* This rule fires on severe Dirichlet shards (e.g. 10-sample clients in
  BloodMNIST α=0.1), where the nominal 5 % split would be 0 samples.

### PHP-FL (NeurIPS 2025)
`clientPHP` + `serverPHP` are re-implemented at CNN-2 scale: private model plus a
personal copy of the global model, per-client scalar inference weight learned on
a validation split (10 steps, lr 1e-3), local objective
CE + symmetric KL + MMD(features) + CE(ensemble), top-p mask
`p = sigmoid(sig_k, (join_times+1)/(round+1)) · tau`, uniform averaging of the
uploaded personal global copies, masked overwrite of each personal copy.
Official constants: `sig_k = 5`, `tau = 0.2`, `valid_ratio = 0.1`,
`feature_dim` MMD on the CNN-2 base representation.
Deviations:
* Official HtM4 model heterogeneity (four different architectures, linear
  participation probabilities) is replaced by homogeneous CNN-2 and 100 %
  participation, as required by R1/B1-B3.
* Official lr 1e-3 and constant schedule are replaced by the shared lr 0.01 +
  cosine schedule.
* **Validation floor guard** (same policy as correction 2, applied to PHP-FL's
  10 % validation split as a robustness measure): if `floor(0.1·n) < 2`, the
  validation set is forced to `min(5, max(2, n//4))` samples; clients with
  `n < 4` skip weight learning and keep w = 0.5. Recorded as
  `phpfl_valid_size_per_client`, `phpfl_forced_rule_clients`,
  `phpfl_uniform_fallback_clients`.
* Local training uses `drop_last=True` only when a shard has at least one full
  batch (official code always drops the last batch); tiny α=0.1 shards keep
  their partial batch so no client is left with zero batches.

## Evaluation protocol notes
* Per-class recall: `sklearn.metrics.recall_score(labels=range(C),
  average=None, zero_division=0)`; the vector always has C entries and never
  contains NaN.
* Class gap `= max(recall) − min(recall)` per model/profile.
* FL side for FedAPEN/PHP-FL: each client's deployed ensemble is evaluated on the
  **global** test set; the per-class recall vectors are averaged across the 20
  clients **before** the class gap is computed, matching the centralized side's
  single model. FedAvg/FedFV use the final global model.
* No client-local test set is evaluated anywhere.
* The centralized reference is the single CNN-2 model of the same
  (dataset, seed), shared by all four methods and all three settings.

## Convergence gate (correction 3)
Before the 72-run queue, BloodMNIST seed 0 is run at:
FedAvg R=50 (α=0.5), FedAvg R=100 (α=0.5, reference), PHP-FL R=50 (α=0.5), and
PHP-FL R=50 (α=0.1, extra conservative check). Global test accuracy is recorded
at rounds 10/20/30/40/50 (50/100 for the R=100 reference).
Stop rules: `PHP-FL(50) − PHP-FL(40) > 1.0 pp` or
`FedAvg(50) < 0.90 · FedAvg(100)`. If either trips, the queue is not launched
and the decision (R=75 vs `convergence_status: undertrained` caveat) is left to
the user. R is never raised silently.

## Reproducibility and resume
* `results/runs/<run_id>.json` and `results/centralized/<key>.json` are written
  atomically on completion; `run_all.py --resume` skips existing outputs.
* `results/manifest.json` / `manifest.csv` list all 78 jobs with
  `status ∈ {complete, pending}`, tier `single`, and their output path.
* The 15-hour budget is enforced by `run_all.py --max-hours 15`: no new jobs are
  launched after the deadline and the manifest reports what remains.

## Heterogeneous supplementary run (approved scope: 30 FL runs)

Scope: FedAPEN full official (B1, B2, B3); PHP-FL hetero on B2 and B3 only.
20 clients, 100% participation, R=50, E=5, batch 64, SGD momentum 0.9, wd 1e-4,
lr 0.01 cosine, seeds {0,1,2}, BloodMNIST + FashionMNIST (30k/5k subsample).
Per-architecture centralized references (recall vectors averaged element-wise
before the class gap; gaps are never averaged).

FedAPEN hetero architectures: official `--he 1` rule (`main.py:50`, `node.py:65-67`,
`models/cnn.py:87-89`): channel_dif = (5 - client_id) * 2 - 1, widths
[64+d, 128+2d, 256+4d], one distinct CNN2_BN per client (20 clients -> 20
architectures, 122k-504k params). The 28x28 input requires the flatten size
channel_num[2]*3*3 instead of the official 32x32 `*4*4` (recorded here).

PHP-FL hetero substitute: official HtM4 uses googlenet/densenet121/
efficientnet-b1/resnet18 (5.6-11.2M). Substituted with googlenet_small/
densenet_small/efficientnet_b0_small/resnet8_small (0.05-0.4M) to fit 12h
budget. Architecture-name diversity (4 distinct families) and round-robin
assignment are preserved. Size range compressed from 5.6-11.2M to 0.2-0.4M
(2x ratio preserved). PHP-FL runs on B2 and B3 only; B1 (IID) is not run under
hetero architectures because the homogeneous B1 inversion finding is already
established and B1 is the least informative hetero cell.

Main-table results unchanged: the heterogeneous results live in
results/runs_hetero/, results/centralized_hetero/, summary_*_hetero.csv and
plots_hetero/, and a SHA-256 checksum assertion verifies the main table before
and after every hetero job.

Heterogeneous substitutes are smaller than CNN2 (19k-29k vs 44k params). The
heterogeneous test probes architecture-family diversity at small scale, not
architecture-scale diversity. See HETERO_NOTE.md 'Scale caveat'.

Hetero scope amendment (user request, 06:45): PHP-FL B1 (IID) was added to the
heterogeneous supplementary run, so PHP-FL now covers B1, B2 and B3. The earlier
statement that "PHP-FL runs on B2 and B3 only; B1 is not run under hetero
architectures" no longer applies. Total hetero FL runs: 36 (FedAPEN 18 +
PHP-FL 18). Per-architecture references are unchanged (the same (method,
dataset, seed) refs serve all settings). Main-table results remain untouched.
Correction: the final approved PHP-FL substitutes are the micro2 set (19,473 /
19,004 / 25,276 / 29,124 params; 0.02-0.03M), which supersedes the 0.05-0.4M
range quoted in the earlier substitute paragraph. The benchmark gate measured
2.11x (Blood) / 2.05x (Fashion) CNN2 average round time.
