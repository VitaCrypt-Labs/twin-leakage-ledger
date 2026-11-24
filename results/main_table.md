<!--
SPDX-FileCopyrightText: 2026 Paul Burg
SPDX-License-Identifier: CC0-1.0
-->

# Results (synthetic data, no real persons)

- Main mode verdict: **bound held** (0 of 240 cells refuted).
- Control mode verdict: **violation caught** (14 of 240 cells refuted).
- Rule, cells, clip and seeds: `PREREGISTRATION.md`. Per-cell numbers: `results/verdict.json`.

## Main mode: rules and renderer, ledger counts the rules

At 24 months. Mean R and mean D are per-persona means (R must stay at or below 1, D at or below 0); the bracket is the distribution-free lower bound at 95% before Holm. Trivial: the recipient already knows the attribute.

| Recipient | Attribute | 95th pct L | Median L_A | 95th pct L_A | A0 mean R | A1 mean R | A0 mean D | A1 mean D | Refuted cells (all checkpoints) |
|---|---|---|---|---|---|---|---|---|---|
| Doctor | diabetes (known) | 145.68 | 1.000 | 1.00 | 1.000 [0.957] | 0.998 [0.955] | 0.000 [-0.055] | 0.000 [-0.055] | 0 |
| Doctor | depression | 145.68 | 1.179 | 17.33 | 0.915 [0.857] | 0.916 [0.856] | -0.073 [-0.128] | -0.082 [-0.136] | 0 |
| Doctor | pregnancy | 145.68 | 1.001 | 1.66 | 0.966 [0.914] | 0.958 [0.904] | -0.007 [-0.062] | -0.010 [-0.065] | 0 |
| Pharmacy bot | diabetes | 41.58 | 1.153 | 15.18 | 0.914 [0.854] | 0.905 [0.845] | -0.084 [-0.139] | -0.095 [-0.150] | 0 |
| Pharmacy bot | depression | 41.58 | 1.161 | 14.44 | 0.916 [0.857] | 0.909 [0.850] | -0.073 [-0.128] | -0.086 [-0.141] | 0 |
| Pharmacy bot | pregnancy | 41.58 | 1.010 | 1.25 | 0.964 [0.916] | 0.967 [0.910] | -0.010 [-0.065] | -0.010 [-0.065] | 0 |
| Insurer | diabetes | 33.40 | 1.143 | 15.79 | 0.913 [0.854] | 0.911 [0.852] | -0.080 [-0.134] | -0.081 [-0.136] | 0 |
| Insurer | depression | 33.40 | 1.025 | 1.56 | 0.930 [0.878] | 0.929 [0.877] | -0.042 [-0.096] | -0.042 [-0.096] | 0 |
| Insurer | pregnancy | 33.40 | 1.005 | 1.11 | 1.000 [0.957] | 1.000 [0.956] | 0.000 [-0.055] | 0.000 [-0.055] | 0 |
| Family member | diabetes | 20.90 | 1.023 | 1.68 | 0.929 [0.877] | 0.927 [0.874] | -0.034 [-0.089] | -0.034 [-0.089] | 0 |
| Family member | depression | 20.90 | 1.230 | 16.09 | 0.879 [0.818] | 0.878 [0.818] | -0.085 [-0.140] | -0.084 [-0.138] | 0 |
| Family member | pregnancy | 20.90 | 1.005 | 1.11 | 1.000 [0.957] | 0.999 [0.955] | 0.000 [-0.055] | -0.000 [-0.055] | 0 |

## Control mode: free drafts with planted phrases, ledger counts only extractor output

At 24 months. Mean R and mean D are per-persona means (R must stay at or below 1, D at or below 0); the bracket is the distribution-free lower bound at 95% before Holm. Trivial: the recipient already knows the attribute.

| Recipient | Attribute | 95th pct L | Median L_A | 95th pct L_A | A0 mean R | A1 mean R | A0 mean D | A1 mean D | Refuted cells (all checkpoints) |
|---|---|---|---|---|---|---|---|---|---|
| Doctor | diabetes (known) | 145.68 | 1.000 | 1.00 | 1.000 [0.957] | 0.999 [0.955] | 0.000 [-0.055] | 0.000 [-0.055] | 0 |
| Doctor | depression | 145.68 | 1.179 | 17.33 | 1.252 [1.132] | 1.244 [1.124] | 0.007 [-0.048] | 0.006 [-0.049] | 2 |
| Doctor | pregnancy | 145.68 | 1.001 | 1.66 | 1.024 [0.939] | 1.022 [0.939] | 0.003 [-0.052] | 0.002 [-0.053] | 0 |
| Pharmacy bot | diabetes | 41.58 | 1.153 | 15.18 | 1.016 [0.936] | 0.996 [0.920] | -0.027 [-0.081] | -0.036 [-0.091] | 0 |
| Pharmacy bot | depression | 41.58 | 1.161 | 14.44 | 1.043 [0.963] | 1.034 [0.954] | -0.020 [-0.075] | -0.024 [-0.079] | 0 |
| Pharmacy bot | pregnancy | 41.58 | 1.010 | 1.25 | 1.088 [0.983] | 1.089 [0.983] | -0.001 [-0.056] | -0.003 [-0.058] | 0 |
| Insurer | diabetes | 33.40 | 1.143 | 15.79 | 1.120 [1.019] | 1.104 [1.007] | -0.014 [-0.068] | -0.015 [-0.070] | 0 |
| Insurer | depression | 33.40 | 1.025 | 1.56 | 2.134 [1.921] | 2.116 [1.905] | 0.059 [0.004] | 0.056 [0.002] | 6 |
| Insurer | pregnancy | 33.40 | 1.005 | 1.11 | 1.000 [0.957] | 1.000 [0.956] | 0.000 [-0.055] | 0.000 [-0.055] | 0 |
| Family member | diabetes | 20.90 | 1.023 | 1.68 | 2.091 [1.880] | 2.064 [1.856] | 0.068 [0.013] | 0.065 [0.010] | 6 |
| Family member | depression | 20.90 | 1.230 | 16.09 | 1.179 [1.062] | 1.171 [1.054] | -0.007 [-0.061] | -0.008 [-0.062] | 0 |
| Family member | pregnancy | 20.90 | 1.005 | 1.11 | 1.000 [0.957] | 0.999 [0.955] | 0.000 [-0.055] | -0.000 [-0.055] | 0 |

## Refuted cells by statistic

| Mode | Attacker | Statistic | Refuted | Cells |
|---|---|---|---|---|
| main | A0 | R | 0 | 60 |
| main | A0 | D | 0 | 60 |
| main | A1 | R | 0 | 60 |
| main | A1 | D | 0 | 60 |
| control | A0 | R | 7 | 60 |
| control | A0 | D | 0 | 60 |
| control | A1 | R | 7 | 60 |
| control | A1 | D | 0 | 60 |

## Sensitivity

Minimal detectable excess (width of the lower bound at the strictest Holm step): R median 0.115, D median 0.092. Per-cell values are in `results/verdict.json`.

## Gaps at 24 months (main mode)

| Recipient | Best attacker mean R | Shortfall 1 / mean R | Mean accuracy | Mean accuracy ceiling | Median L / L_A | Median log L | Median log odometer |
|---|---|---|---|---|---|---|---|
| Doctor | 0.941 | 1.06 | 0.946 | 0.987 | 1.40 | 0.440 | 8.142 |
| Pharmacy bot | 0.932 | 1.07 | 0.926 | 0.982 | 1.43 | 0.370 | 4.799 |
| Insurer | 0.948 | 1.06 | 0.915 | 0.955 | 1.20 | 0.237 | 1.125 |
| Family member | 0.936 | 1.07 | 0.923 | 0.962 | 1.12 | 0.256 | 6.668 |

## Recipient knows more (A0 only, descriptive)

- Ledger configured with the published `more` prior: 0 of 120 cells refuted.
- Ledger configured with `base` while A0 holds the `more` knowledge: 8 of 120 cells refuted (no verdict).
