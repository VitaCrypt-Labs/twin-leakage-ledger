<!--
SPDX-FileCopyrightText: 2026 Paul Burg
SPDX-License-Identifier: CC0-1.0
-->

# Cost in usefulness (synthetic data, base prior)

## 1. Price of the rules

Needed true facts stated in messages that are sent, as a share of all needed true facts at those messages (rules vs simulated free agent with recall 0.95). The schedule is the same in both.

| Recipient | Needed facts | Rules | Free agent | Rules / free agent |
|---|---|---|---|---|
| Doctor | 2836 | 0.755 | 0.957 | 0.789 |
| Pharmacy bot | 1445 | 1.000 | 0.936 | 1.068 |
| Insurer | 870 | 0.343 | 0.956 | 0.358 |
| Family member | 705 | 0.084 | 0.942 | 0.089 |

## 2. Price of a limit

A limit caps the per-attribute bound for the three sensitive attributes. The ledger coarsens or drops a message before sending when the worst case over its possible contents would pass the limit. Shares are over messages the rules would send. Lost facts are relative to the rules without a limit.

| Recipient | Limit | Coarsened | Not sent | Needed facts lost | Max bound reached |
|---|---|---|---|---|---|
| Doctor | 2x | 0.853 | 0.022 | 0.638 | 1.000 |
| Doctor | 5x | 0.703 | 0.022 | 0.390 | 4.381 |
| Doctor | 10x | 0.420 | 0.014 | 0.176 | 9.870 |
| Pharmacy bot | 2x | 0.000 | 0.987 | 1.000 | 1.000 |
| Pharmacy bot | 5x | 0.148 | 0.838 | 1.000 | 4.882 |
| Pharmacy bot | 10x | 0.987 | 0.000 | 1.000 | 9.006 |
| Insurer | 2x | 1.000 | 0.000 | 1.000 | 1.000 |
| Insurer | 5x | 0.936 | 0.000 | 0.819 | 4.767 |
| Insurer | 10x | 0.688 | 0.000 | 0.544 | 9.494 |
| Family member | 2x | 0.058 | 0.793 | 1.000 | 1.859 |
| Family member | 5x | 0.270 | 0.379 | 1.000 | 4.995 |
| Family member | 10x | 0.389 | 0.139 | 1.000 | 9.985 |

## 3. Price of the coarseness of the bound

At 24 months, base prior. Bound L_A against what the best attacker achieved, and per-attribute against general bound.

| Recipient | Best attacker mean R | Bound shortfall (1 / mean R) | Mean accuracy | Mean accuracy ceiling | Median L / L_A |
|---|---|---|---|---|---|
| Doctor | 0.941 | 1.06 | 0.946 | 0.987 | 1.40 |
| Pharmacy bot | 0.932 | 1.07 | 0.926 | 0.982 | 1.43 |
| Insurer | 0.948 | 1.06 | 0.915 | 0.955 | 1.20 |
| Family member | 0.936 | 1.07 | 0.923 | 0.962 | 1.12 |
