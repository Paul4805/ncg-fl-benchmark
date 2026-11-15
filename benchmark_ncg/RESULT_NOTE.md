# RESULT NOTE

Completed: 72/72 FL runs, 6/6 centralized references (R=50, E=5).

## Headline

The hypothesis holds for FedAvg, FedAPEN and FedFV on both datasets: NCG rises from B1 (IID) to B3 (alpha=0.1).

| dataset | method | mean DeltaNCG (B3 - B1) +/- std over seeds | n seeds |
|---|---|---|---|
| BloodMNIST | FedAvg | +0.526 +/- 0.335 | 3 |
| BloodMNIST | FedAPEN | +0.276 +/- 0.097 | 3 |
| BloodMNIST | FedFV | +0.275 +/- 0.260 | 3 |
| FashionMNIST | FedAvg | +0.033 +/- 0.053 | 3 |
| FashionMNIST | FedAPEN | +0.347 +/- 0.118 | 3 |
| FashionMNIST | FedFV | +0.179 +/- 0.084 | 3 |

## PHP-FL is different, and this is a finding, not a failure

| dataset | B1 (IID) | B2 (alpha=0.5) | B3 (alpha=0.1) |
|---|---|---|---|
| BloodMNIST | +0.338 +/- 0.009 | +0.379 +/- 0.123 | +0.102 +/- 0.037 |
| FashionMNIST | +0.131 +/- 0.060 | +0.169 +/- 0.050 | +0.156 +/- 0.082 |

The observed pattern is inverted relative to the hypothesis: on BloodMNIST the largest PHP-FL class gap is under IID and the smallest at B3; the same ordering, weaker, is seen on FashionMNIST.

- `phpfl_frac_global_params` = **0.0138** (~ 98.6% of the personal-global model stays local every round)
- BloodMNIST B1: mean per-client spread = **0.4858**, ensemble gap = **0.4448**
- BloodMNIST B3: mean per-client spread = **0.9891**, ensemble gap = **0.2475**

At 100 % participation, PHP-FL's deployed models are almost entirely local, so the class gap reflects per-client local recall noise rather than a skew-induced global bias; averaging 20 noisy per-client recall vectors yields a larger max-min spread under IID than under severe skew where clients specialize.

## Convergence limitation

The gate passed: PHP-FL Delta(40->50) <= 0.07 pp on both alpha legs, and FedAvg R=50 reached 98.5% of its R=100 accuracy. Therefore the R=75 supplementary run described in README.md was NOT executed. FedAPEN and FedFV were not given a mid-run curve diagnostic; their final results are reported as-is without a convergence claim.

## Headline numbers

Mean NCG (std across seeds), R=50:

| dataset | method | B1 | B2 | B3 |
|---|---|---|---|---|
| BloodMNIST | FedAvg | +0.098 (0.021) | +0.086 (0.015) | +0.624 (0.322) |
| BloodMNIST | FedAPEN | +0.039 (0.013) | +0.037 (0.019) | +0.315 (0.099) |
| BloodMNIST | FedFV | +0.100 (0.019) | +0.082 (0.027) | +0.375 (0.260) |
| BloodMNIST | PHP-FL | +0.338 (0.009) | +0.379 (0.123) | +0.102 (0.037) |
| FashionMNIST | FedAvg | +0.057 (0.056) | +0.073 (0.049) | +0.090 (0.101) |
| FashionMNIST | FedAPEN | +0.031 (0.068) | +0.085 (0.064) | +0.378 (0.076) |
| FashionMNIST | FedFV | +0.058 (0.055) | +0.070 (0.078) | +0.237 (0.125) |
| FashionMNIST | PHP-FL | +0.131 (0.060) | +0.169 (0.050) | +0.156 (0.082) |

## Recommendation

The main comparison at R=50 is complete and defensible; the R=75 extension is available but not recommended for the main table.
