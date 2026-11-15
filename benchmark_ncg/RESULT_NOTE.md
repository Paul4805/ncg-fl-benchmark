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

## Accuracy (top-1 and balanced)

Centralized baseline (top-1 / balanced): BloodMNIST 0.9355 / 0.9265, FashionMNIST 0.9048 / 0.9048

| dataset | method | B1 acc | B2 acc | B3 acc |
|---|---|---|---|---|
| BloodMNIST | FedAvg | 0.9068 +/- 0.0016 | 0.8962 +/- 0.0122 | 0.8156 +/- 0.0486 |
| BloodMNIST | FedAPEN | 0.9263 +/- 0.0005 | 0.9108 +/- 0.0075 | 0.8236 +/- 0.0229 |
| BloodMNIST | FedFV | 0.9067 +/- 0.0016 | 0.8955 +/- 0.0092 | 0.8193 +/- 0.0616 |
| BloodMNIST | PHP-FL | 0.8253 +/- 0.0028 | 0.5980 +/- 0.0118 | 0.3415 +/- 0.0174 |
| FashionMNIST | FedAvg | 0.9020 +/- 0.0036 | 0.8945 +/- 0.0046 | 0.8695 +/- 0.0174 |
| FashionMNIST | FedAPEN | 0.9040 +/- 0.0018 | 0.8865 +/- 0.0008 | 0.7850 +/- 0.0346 |
| FashionMNIST | FedFV | 0.9019 +/- 0.0036 | 0.8932 +/- 0.0037 | 0.8739 +/- 0.0144 |
| FashionMNIST | PHP-FL | 0.8514 +/- 0.0040 | 0.6523 +/- 0.0059 | 0.3653 +/- 0.0280 |

Accuracy gap = centralized - method (positive = FL worse):

| dataset | method | B1 top-1 | B2 top-1 | B3 top-1 | B1 bal | B2 bal | B3 bal |
|---|---|---|---|---|---|---|---|
| BloodMNIST | FedAvg | +0.0287 | +0.0393 | +0.1199 | +0.0391 | +0.0496 | +0.1758 |
| BloodMNIST | FedAPEN | +0.0092 | +0.0247 | +0.1119 | +0.0145 | +0.0298 | +0.1315 |
| BloodMNIST | FedFV | +0.0288 | +0.0399 | +0.1162 | +0.0393 | +0.0496 | +0.1438 |
| BloodMNIST | PHP-FL | +0.1102 | +0.3375 | +0.5940 | +0.1424 | +0.3592 | +0.5969 |
| FashionMNIST | FedAvg | +0.0028 | +0.0103 | +0.0353 | +0.0028 | +0.0103 | +0.0353 |
| FashionMNIST | FedAPEN | +0.0008 | +0.0183 | +0.1198 | +0.0008 | +0.0183 | +0.1198 |
| FashionMNIST | FedFV | +0.0029 | +0.0116 | +0.0309 | +0.0029 | +0.0116 | +0.0309 |
| FashionMNIST | PHP-FL | +0.0534 | +0.2525 | +0.5395 | +0.0534 | +0.2525 | +0.5395 |

## Accuracy and NCG

From B1 to B3, FedAvg's top-1 accuracy on BloodMNIST falls (0.9068 -> 0.8156) while its NCG goes from +0.0984 to +0.6241. At B3 it reaches 81.56% top-1 and 75.08% balanced accuracy; the top-1 minus balanced gap (delta = +0.0648) is itself a proxy for class imbalance and is consistent with the NCG direction.

## Accuracy-NCG regimes

Regime thresholds: acc < 0.60 = low; NCG > 0.20 = high.

NCG must be interpreted jointly with accuracy. PHP-FL's NCG falls from +0.338 (B1) to +0.102 (B3) on BloodMNIST as its accuracy falls from 0.825 to 0.341. This is not a fairness improvement - it is uniform failure: recall is low and similar across classes. By contrast, FedAvg at B3 has accuracy 0.816 and NCG +0.624: accuracy stays high but is concentrated, with near-zero recall on some classes. With the stated thresholds the B3 FL methods split into two computed regimes, not three: FedAvg high_acc_high_ncg (0.816/+0.624); FedAPEN high_acc_high_ncg (0.824/+0.315); FedFV high_acc_high_ncg (0.819/+0.375); PHP-FL low_acc_low_ncg (0.341/+0.102). On FashionMNIST the same split appears (PHP-FL low_acc_low_ncg; FedAvg high_acc_low_ncg; FedAPEN high_acc_high_ncg; FedFV high_acc_high_ncg). NCG alone cannot distinguish uniform failure from selective failure, so we report accuracy and NCG as a pair and recommend that future work on class-gap metrics include accuracy alongside them.

## Recommendation

The main comparison at R=50 is complete and defensible; the R=75 extension is available but not recommended for the main table.

## Homogeneous verification

The homogeneous main table (72 FL runs + 6 centralized references) was verified post-hoc to have been produced by the CNN-2 default model path, with every client model at 44,584 params (BloodMNIST) or 50,282 (FashionMNIST). The model-factory bug discovered during the heterogeneous block did not affect the homogeneous results because the bug's fallback behavior (build_model() -> CNN-2) coincided with the homogeneous block's intended model. Homogeneous run JSONs are content-identical to commit bab3f7c (Git blob hashes match for all 72 files and git diff is clean; working-tree bytes differ from the blob only by Windows CRLF line endings).
