<!--
SPDX-FileCopyrightText: 2026 Paul Burg
SPDX-License-Identifier: CC-BY-4.0
-->

# Preregistration v1 (tag `prereg-v1`)

Copyright 2026 Paul Burg. SPDX-License-Identifier: CC-BY-4.0

This file is committed and tagged `prereg-v1` before any result is produced or committed. The order is visible in the git history: the tag precedes the first commit that contains anything under `results/`. Everything here is research on synthetic data. It is not a product guarantee and not a clinical statement.

## 1. Question

A ledger kept on the person's side computes, for each recipient, a pointwise maximal leakage bound from the declared release rules. Does that bound hold against attackers that read the same messages? Does a control mode, where the ledger counts only what a rule-based extractor finds in free drafts with planted extra phrases, break the bound and get caught by the same test?

## 2. Objects

- Persona: a 24-month history from a closed enumerable set (`data/population.json`, fewer than 65 000 histories). Personas are drawn from the prior of that table.
- Recipients: doctor, pharmacy bot, insurer, family member, with the prior knowledge published in `data/recipients.json`. The main run uses the `base` variant.
- Modes: `main` (messages written by a deterministic renderer from claims chosen by declared rules) and `control` (free drafts with planted phrases; the ledger counts only extractor output).
- Attackers in this version: A0 (exact Bayesian, knows the population and the generating rules) and A1 (multinomial logistic regression trained on the raw text of 20 000 separate synthetic personas, given the recipient's prior knowledge as extra features). Attackers based on open LLMs are not part of v1 and will have their own preregistration.
- Sensitive attributes: diabetes, pregnancy, depression (each as the full onset variable, including "none").
- Checkpoints: after months 1, 3, 6, 12 and 24, cumulative.

## 3. Bound and the two consequences

For recipient R with prior knowledge k and all messages y so far: `L = max_x P(y|x) / P_R(y)` and `L_A = max_a P(a|y) / P_R(a)` for each sensitive attribute A, both computed under the prior conditioned on k. Then `L_A <= L`, and for any attacker that outputs a distribution q over the values of A as a function of y:

1. Guess: `P(attacker names the true value | y) <= L_A * max_a P_R(a)`.
2. Stated probability: `E[ q(a_true) / (P_R(a_true) * L_A) | y ] <= 1`.

## 4. Per-persona statistics

For persona i, cell (attacker, attribute, recipient, checkpoint):

- `R_i = min(B, q_i(a_true) / (P_R(a_true) * L_A,i))` with clip `B = 10`. Clipping can only lower the mean, so refutation stays valid.
- `D_i = [attacker's top value is correct] - min(1, L_A,i * max_a P_R(a))`. Ties go to the lowest value index.

Claims per cell: `mean(R_i) <= 1` (statistic R) and `mean(D_i) <= 0` (statistic D). The unit of analysis is the persona. Number of test personas: `n = 2000` per mode, seed `20261207`. A1 is trained on 20 000 other personas, seed `20261208`.

## 5. Decision rule

- One-sided tests at level `alpha = 0.05`.
- Statistic R: empirical Bernstein (Maurer and Pontil 2009) with range `[0, B]`. Statistic D: Hoeffding with range `[-1, 1]`. The p-value of a cell is the smallest delta at which the lower confidence bound exceeds the threshold (1 for R, 0 for D).
- Holm step-down correction over the family of all cells of the mode: attackers (2) x attributes (3) x recipients (4) x checkpoints (5) x statistics (2) = 240 cells per mode. The architecture draft counted 480 because it included three LLM attackers that are not in v1; those cells will form their own family.
- A cell is refuted if its Holm-adjusted p-value is at most 0.05.
- Plain bootstrap lower bounds are printed next to each cell for reference only and never decide.
- Minimal detectable excess per cell: the width of the lower confidence bound at the strictest Holm step (`alpha / 240`), computed from the observed variance.
- A cell where the recipient already knows the attribute is kept in the family and marked trivial.

## 6. Verdicts

- Main mode: `bound held` if no cell is refuted; otherwise `bound refuted`, reported as is with the refuting cells.
- Control mode: `violation caught` if at least one cell is refuted; otherwise the experiment is declared `uninformative` and `make reproduce` exits with an error.
- The sample of cells and sizes are not changed after seeing results. A different analysis may be added later but never replaces these.

## 7. Descriptive outputs (no verdict)

- Gaps: how far the best attacker is below the per-attribute bound, and how far the per-attribute bound is below the general bound `L`.
- Odometer comparison: the sum of per-message log losses next to the exact `log L`.
- Recipient knows more: (a) ledger configured with the published `more` variant against A0 (own family of 120 cells, same rule); (b) ledger configured with `base` while A0 holds the `more` knowledge, reported without a verdict.
- Usefulness: three numbers per recipient (price of the rules, price of a limit at 2x, 5x, 10x, price of the bound's coarseness) on a sample of 300 personas.

## 8. Assumptions of the bound (published)

1. The table of what the recipient knows in advance is correct.
2. Recipients do not pool what they receive.
3. A fact about the person is determined by the history from the closed set, or is tied to the person only through it.
4. The code of the release rules is the code the ledger computes with (checked by tests).

## 9. Deviations

Any change to this rule after the tag is published as a new tag (`prereg-v2`) with a stated reason, and v1 results stay in the history.
