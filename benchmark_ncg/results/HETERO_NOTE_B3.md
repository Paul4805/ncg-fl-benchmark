# HETERO NOTE (supplementary, separate from the R=50 main table)

## Scale caveat (read first)

Due to the 12-hour compute budget, PHP-FL's heterogeneous client architectures were compressed from the paper's HtM4 set (googlenet 5.6M, densenet121 7.0M, efficientnet-b1 6.5M, resnet18 11.2M) to four micro variants (19k-29k params). Architecture family diversity (Inception / Dense / MBConv / Residual) and round-robin assignment are preserved; absolute model scale is inverted relative to the homogeneous CNN2 baseline (substitutes are 0.43-0.65x CNN2, paper models are 127-254x CNN2). This heterogeneous experiment therefore tests whether architecture-family diversity alone changes the NCG conclusions. It does not test whether the paper's larger models would. A positive result (hetero NCG within std of homogeneous NCG) is a necessary but not sufficient condition for the paper-scale claim; a negative result (hetero NCG notably different) directly weakens the generality of the homogeneous conclusions and should be reported as such.

## Architecture scale table

CNN2 reference: 44,584 parameters (BloodMNIST shape).

| method | paper_backbone | paper_params | substitute | substitute_params | size_ratio_vs_CNN2_paper | size_ratio_vs_CNN2_substitute |
|---|---|---|---|---|---|---|
| FedAPEN | CNN2_BN width-heterogeneous (official) | 122,018-504,146 | none (official widths used) | same | 2.7-11.3x | same |
| PHP-FL | googlenet | 5,604,008 | googlenet_micro2 | 19,473 | 125.7x | 0.44x |
| PHP-FL | densenet121 | 6,950,280 | densenet_micro2 | 19,004 | 155.9x | 0.43x |
| PHP-FL | efficientnet-b1 | 6,517,288 | efficientnet_micro2 | 25,276 | 146.2x | 0.57x |
| PHP-FL | resnet18 | 11,180,616 | resnet4_micro | 29,124 | 250.8x | 0.65x |

## Scope

Scope: FedAPEN (B3; full official); PHP-FL (B3; with substitutes per DEVIATIONS.md). 20 clients, R=50, E=5, seeds {0,1,2}, per-architecture centralized references.

## Architectures

**FedAPEN** (official `--he 1`, 20 distinct CNN2_BN widths):

- arch 0: CNN2_BN d=+9, 504,146 params (FedAPEN/models/cnn.py:87-89 + node.py:65-67 (--he 1))
- arch 1: CNN2_BN d=+7, 477,554 params (FedAPEN/models/cnn.py:87-89 + node.py:65-67 (--he 1))
- arch 2: CNN2_BN d=+5, 451,682 params (FedAPEN/models/cnn.py:87-89 + node.py:65-67 (--he 1))
- ... plus 17 more widths down to arch 19 (d=-29); all <= 504,146 params

**PHP-FL** (approved substitutes for the official HtM4 family):

- arch 0: googlenet_micro2 (GoogLeNet-style inception), 19,473 params - substituted from official HtM4 (5.6-11.2M) due to 5M cap and 12h budget; approved micro2 set (candidate B2); see DEVIATIONS.md
- arch 1: densenet_micro2 (DenseNet-style dense blocks), 19,004 params - substituted from official HtM4 (5.6-11.2M) due to 5M cap and 12h budget; approved micro2 set (candidate B2); see DEVIATIONS.md
- arch 2: efficientnet_micro2 (EfficientNet-style MBConv), 25,276 params - substituted from official HtM4 (5.6-11.2M) due to 5M cap and 12h budget; approved micro2 set (candidate B2); see DEVIATIONS.md
- arch 3: resnet4_micro (ResNet-style residual), 29,124 params - substituted from official HtM4 (5.6-11.2M) due to 5M cap and 12h budget; approved micro2 set (candidate B2); see DEVIATIONS.md

Client assignment: FedAPEN cid -> cid; PHP-FL cid -> cid % 4.

## NCG_hetero (mean +/- std over 3 seeds)

| dataset | method | B1 | B2 | B3 |
|---|---|---|---|---|
| BloodMNIST | FedAPEN | not run | not run | +0.349 +/- 0.049 |
| BloodMNIST | PHP-FL | not run | not run | +0.125 +/- 0.029 |
| FashionMNIST | FedAPEN | not run | not run | +0.361 +/- 0.069 |
| FashionMNIST | PHP-FL | not run | not run | +0.107 +/- 0.140 |

## Side-by-side: homogeneous (main table) vs heterogeneous

| dataset | method | setting | homogeneous NCG | hetero NCG | delta (het - hom) |
|---|---|---|---|---|---|
| BloodMNIST | FedAPEN | B1 | +0.039 +/- 0.013 | +nan +/- nan | +nan |
| BloodMNIST | FedAPEN | B2 | +0.037 +/- 0.019 | +nan +/- nan | +nan |
| BloodMNIST | FedAPEN | B3 | +0.315 +/- 0.099 | +0.349 +/- 0.049 | +0.034 |
| BloodMNIST | PHP-FL | B1 | +0.338 +/- 0.009 | +nan +/- nan | +nan |
| BloodMNIST | PHP-FL | B2 | +0.379 +/- 0.123 | +nan +/- nan | +nan |
| BloodMNIST | PHP-FL | B3 | +0.102 +/- 0.037 | +0.125 +/- 0.029 | +0.023 |
| FashionMNIST | FedAPEN | B1 | +0.031 +/- 0.068 | +nan +/- nan | +nan |
| FashionMNIST | FedAPEN | B2 | +0.085 +/- 0.064 | +nan +/- nan | +nan |
| FashionMNIST | FedAPEN | B3 | +0.378 +/- 0.076 | +0.361 +/- 0.069 | -0.017 |
| FashionMNIST | PHP-FL | B1 | +0.131 +/- 0.060 | +nan +/- nan | +nan |
| FashionMNIST | PHP-FL | B2 | +0.169 +/- 0.050 | +nan +/- nan | +nan |
| FashionMNIST | PHP-FL | B3 | +0.156 +/- 0.082 | +0.107 +/- 0.140 | -0.050 |

## Verdict

Under architecture-family diversity at micro scale, FedAPEN's NCG does not survive the homogeneous conclusion from the R=50 main table.

Under architecture-family diversity at micro scale, PHP-FL's NCG survives the homogeneous conclusion from the R=50 main table.

No cell changed by more than 0.05.

## Caveats

- PHP-FL B1 (IID) was not run under hetero architectures.
- PHP-FL uses 19k-29k parameter substitutes (0.02-0.03M); absolute sizes are compressed from the official 5.6-11.2M HtM4 family (family diversity and round-robin assignment preserved).
- Main-table (homogeneous) results are unchanged; heterogeneous results live in a separate namespace and are never mixed into the main CSVs.
- PHP-FL B2 was not completed within the 12-hour budget. Hetero conclusions for PHP-FL are B3-only.

## Accuracy (B3 hetero, mean +/- std over seeds)

| setting | dataset | method | top-1 acc | balanced acc |
|---|---|---|---|---|
| B3 | BloodMNIST | FedAPEN | 0.8216 +/- 0.0211 | 0.7925 +/- 0.0168 |
| B3 | BloodMNIST | PHP-FL | 0.3596 +/- 0.0219 | 0.3466 +/- 0.0219 |
| B3 | FashionMNIST | FedAPEN | 0.7651 +/- 0.0385 | 0.7651 +/- 0.0385 |
| B3 | FashionMNIST | PHP-FL | 0.3360 +/- 0.0197 | 0.3360 +/- 0.0197 |

## Per-seed PHP-FL Fashion B3 (hetero)

| seed | NCG_hetero | top-1 acc | FL class gap | minority class | minority recall | classes with recall < 0.05 |
|---|---|---|---|---|---|---|
| 0 | +0.2627 | 0.3342 | 0.4636 | 6 | 0.1462 | none |
| 1 | +0.0657 | 0.3565 | 0.3240 | 6 | 0.2216 | none |
| 2 | -0.0085 | 0.3173 | 0.3500 | 3 | 0.1734 | none |

### Negative-NCG seeds

Seed 2 of PHP-FL Fashion B3 achieved NCG = -0.0085, i.e. FL class gap (0.3500) was smaller than the centralized mean-reference gap (0.3555). Per-class recalls: [0.322, 0.296, 0.328, 0.173, 0.318, 0.239, 0.198, 0.523, 0.379, 0.397] (minority class 3, recall 0.1734). The negative value coincides with uniformly low accuracy (0.3173) and reflects the uniform-failure regime already described, not a fairness improvement: a model that predicts near-uniformly across classes will have a small per-class recall spread irrespective of accuracy. Report alongside the mean (+0.107) to avoid the impression that PHP-FL is 'fair' on Fashion.


## Limitation (budget)

Heterogeneous results cover FedAPEN ['B3'], PHP-FL ['B3']. B1 hetero was not evaluated in this block. The homogeneous R=50 main table remains the primary result; this block is a reference-sensitivity check under real heterogeneous client architectures.
