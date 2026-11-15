# NCG benchmark: federated learning vs centralized class gaps

Measures the **Normalized Class Gap (NCG)** of four federated-learning methods
relative to centralized training on BloodMNIST and a subsampled FashionMNIST,
as label skew increases.

## Hypothesis

> NCG should be **near 0 under IID** (B1), **positive under non-IID**
> (B2, alpha = 0.5) and **increase as alpha decreases** (B3, alpha = 0.1) for **FedAvg,
> FedAPEN, FedFV and PHP-FL**, relative to centralized learning. That is, FL
> leaves a positive and increasing class gap as label skew worsens.

A negative NCG means the FL method has a *smaller* class gap than centralized
training; such results are reported as-is, not hidden.

## Method identity

The method referred to in earlier drafts as "pFedFL" is **PHP-FL**:

> *A Fair Federated Learning Method for Handling Client Participation
> Probability Inconsistencies in Heterogeneous Environments*, NeurIPS 2025.
> Poster: https://neurips.cc/virtual/2025/loc/san-diego/poster/115969
> Official code: https://github.com/Siyuan01/PHP-FL-main

No other method named "pFedFL" exists in this repository, and none is
implemented. The only method names used in code, configs, results and logs are
**FedAvg, FedAPEN, FedFV, PHP-FL, Centralized**.

## Setup

| item | value |
|---|---|
| settings | B1 IID, B2 Dirichlet alpha=0.5, B3 Dirichlet alpha=0.1 |
| clients | 20, 100 % participation |
| rounds / local epochs | R = 50, E = 5 (250 local epochs per client) |
| batch / optimizer | 64, SGD momentum 0.9, weight decay 1e-4, lr 0.01 cosine |
| backbone | CNN-2 (2 conv layers, 28x28 input) for every model |
| seeds | 0, 1, 2 (partitions, initialization, batch order) |
| BloodMNIST | official split: 11,959 train / 1,712 val / 3,421 test |
| FashionMNIST | per-seed subsample: 30,000 train / 5,000 test (3,000 / 500 per class) |
| centralized | CNN-2, max 250 epochs, ES patience 20, best-val checkpoint |
| centralized val | BloodMNIST: official val (1,712); FashionMNIST: stratified 10 % of train |

The centralized reference is **one CNN-2 model per (dataset, seed)**, shared by
all four FL methods and all three settings. There are no per-architecture or
backbone-averaged references; FL methods are compared against the same
homogeneous reference.

## Definitions

For a model evaluated on the fixed global test set:

```
recall_c   = TP_c / (TP_c + FN_c)          # labels=range(C), zero_division=0
ClassGap   = max_c(recall_c) - min_c(recall_c)
NCG        = (ClassGap_FL - ClassGap_Centralized) / (1 - ClassGap_Centralized)
```

* NCG = 0 -> FL matches the centralized class gap.
* NCG > 0 -> FL has a larger class gap than centralized training.
* NCG < 0 -> FL has a smaller class gap (reported, not hidden).

**Averaging rule:** per-class recall vectors are always averaged first, then the
class gap is computed. Class gaps are never averaged. For FedAPEN and PHP-FL the
deployed/personalised model of each client is evaluated on the global test set,
and the 20 per-class recall vectors are averaged before the gap. FedAvg/FedFV
use the final global model. The centralized side uses the single CNN-2 model.

## Reproduce

```bash
# 1. partitions + per-seed FashionMNIST subsamples (also verifies them)
python -m benchmark_ncg.scripts.generate_partitions

# 2. validation suite (38 tests: partitions, subsample, NCG formula, splits,
#    adaptation floor rule, forbidden settings/backbones, smoke run)
python -m pytest benchmark_ncg/tests -q

# 3. six centralized CNN-2 references
python -m benchmark_ncg.scripts.train_centralized --all

# 4. convergence gate (FedAvg R=50/R=100 + PHP-FL R=50 on BloodMNIST s0)
python -m benchmark_ncg.scripts.convergence_check

# 5. all 72 B1-B3 FL runs, 2 workers, resumable, 15 h hard stop
python -m benchmark_ncg.scripts.run_all --stage fl --workers 2 --resume

# 6. results tables and plots
python -m benchmark_ncg.scripts.compute_ncg
python -m benchmark_ncg.scripts.make_summary
python -m benchmark_ncg.scripts.plot_ncg
python -m benchmark_ncg.scripts.evaluate --all
```

`run_all.py --resume` skips any job whose JSON already exists. Jobs are
independent subprocesses; FL jobs wait for their centralized reference.

## Methods

| method | implementation | personalization |
|---|---|---|
| FedAvg | standard sample-weighted FedAvg | none (global model) |
| FedFV | official easyFL FedFV aggregation (`alpha=0.1`, `tau=1`) | none (global model) |
| FedAPEN | official `learned_adaptive_training` (KDD 2023) | `w-private + (1-w)-shared` per client |
| PHP-FL | official `clientPHP`/`serverPHP` (NeurIPS 2025) | `w-private + (1-w)-personal_global` per client |
| Centralized | single CNN-2 on the full aggregated train set | n/a |

FedAPEN and PHP-FL are run at CNN-2 scale on homogeneous data, as required by
the benchmark's B1-B3 reduction; their own model-heterogeneous/official
settings are out of scope. Deviations (including the adaptability/validation
floor guards needed for tiny alpha=0.1 shards) are documented in
[`DEVIATIONS.md`](DEVIATIONS.md).

## Artifacts

```
benchmark_ncg/
  configs/            benchmark.json, methods.json (single source of truth)
  data/               raw downloads + normalized tensor cache
  partitions/         18 client partitions + 3 FashionMNIST subsample files + index.json
  core/               config, datasets, partitioning, metrics, models, training utils, report
  methods/            fedavg, fedfv, fedapen, phpfl, shared plumbing
  centralized/        CNN-2 centralized training with early stopping
  scripts/            generate_partitions, train_centralized, train_fl, evaluate,
                      compute_ncg, run_all, make_summary, plot_ncg, convergence_check
  tests/              38 validation tests
  results/
    raw_results.csv            one row per FL run (+ Centralized reference rows)
    summary_ncg.csv            mean / std over 3 seeds
    summary_class_gap.csv      centralized vs FL class gaps
    manifest.json / .csv       all 78 jobs with status complete|pending
    runs/<run_id>.json         72 run records (config, recalls, NCG inputs)
    centralized/<key>.json     6 centralized references
    diagnostics/               convergence-gate curves (not part of results)
    plots/
      ncg_vs_setting.png
      recall_heatmap_B3_BloodMNIST.png
      recall_heatmap_B3_FashionMNIST.png
      class_gap_vs_setting.png
  DEVIATIONS.md
  README.md
  RESULT_NOTE.md
```

## Extension path (not run in this session)

If the convergence gate trips, the supplementary experiment is to retrain ONLY
the affected method(s) at a higher R (candidate: R=75) and compare their NCG
against the R=50 main table. This must NOT be mixed into the main comparison
table, because doing so would give the retrained method more compute than the
others and confound the NCG. The supplementary results belong in a separate
section of the paper titled "Converged-method check", with the following
protocol:

- Re-run only the affected method(s), all 3 settings (B1/B2/B3), both datasets,
  all 3 seeds, at the higher R.
- Reuse the existing 6 centralized CNN-2 references unchanged (run_ids end in
  `__R250`; the extension runs reuse them by key).
- Write outputs to `results/runs/<run_id>.json`; the `__R` suffix in the run_id
  guarantees no collision with the `__R50` main-table results, e.g.
  `B3__PHP-FL__FashionMNIST__s2__R75.json`.
- Tag every extension run with `"run_role": "extension"` in the JSON so
  `summary_ncg.csv` can be filtered. `report.write_raw` keeps extension rows out
  of the main table and writes them to `results/raw_results_extension.csv`
  (`summary_ncg_extension.csv`, `summary_class_gap_extension.csv` alongside).
- Report the extension NCG side by side with the R=50 NCG for the same method
  and setting, and state explicitly that only the affected method(s) received
  the higher budget.
- If the R=75 run also fails the same Delta(last-10-rounds) > 1.0 pp rule, report
  that fact and do not extend further; the method is genuinely slow to converge
  on this data and the lower-bound framing stands.

The extension command, to be run by the user in a later session, is:

```bash
python scripts/run_all.py \
    --methods PHP-FL \
    --rounds 75 \
    --run-role extension \
    --resume --workers 2 --max-hours 3
```

(from the repository root: `python -m benchmark_ncg.scripts.run_all ...` with the
same flags). `--methods`, `--rounds` and `--run-role` are implemented now:
`--rounds` must differ from 50 and `--run-role extension` is required, the
resulting run_ids end in `__R<rounds>`, and each payload carries
`"run_role": "extension"`. The main queue uses none of these flags and produces
`__R50` run_ids with no role tag.

Status at session close: gate passed, extension not executed. The
`--rounds 75 --run-role extension` path remains available for future sessions
but is not required for the main comparison.

## Notes on comparability

* Every run uses the same global test set within a (dataset, seed): FL and
  centralized are always scored on identical samples.
* FedAPEN/PHP-FL are personalised methods; their FL-side recall is the mean of
  the deployed client models' recall vectors, computed **before** the class gap.
  This is the protocol the study asks for and is applied identically to every
  personalised method.
* The centralized reference is trained once and reused; it is not re-tuned per
  method or per setting.
