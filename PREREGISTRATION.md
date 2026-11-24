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

---

# Preregistration v2: LLM attackers (tag `prereg-v2`)

This section is committed and tagged `prereg-v2` before any LLM answer exists. Section 1 to 9 above (v1) stay as they are; v1 results are not touched. Nothing here changes the rule of v1: same per-persona statistics `R_i` and `D_i`, same clip `B = 10`, same tests, same Holm correction, same `alpha = 0.05`.

## v2.1 Attackers

Three open-weight models of three families, weights under Apache-2.0 or MIT, pinned by revision and file hash in `data/models.json` and recorded with licence and release date in `docs/MODELS.md`: Qwen3-4B-Instruct-2507 (Apache-2.0), Mistral-7B-Instruct-v0.3 (Apache-2.0), Phi-4-mini-instruct (MIT). Each model gets, per person, recipient and checkpoint: the recipient's published prior knowledge (`base` variant), the recipient's base rates for the three sensitive attributes, and the messages that recipient received up to the checkpoint. It gets nothing else about the person. It returns a probability distribution over the values of each sensitive attribute.

- Three runs per model: the three prompt wordings `p1`, `p2`, `p3` (`WORDINGS` in `tll/attackers/a_llm.py`), seeds 101, 102, 103. Sampling: temperature 0.3, top-p 0.95, at most 160 new tokens.
- Parsing: first JSON object in the answer; values are normalised; names such as `none`, `month 5` or `5` are accepted. An attribute that cannot be read falls back to the base rates and is counted; counts per model and attribute are reported.
- The per-person answer of a model is the mean of its three runs. That mean is the `q` of the cell.

## v2.2 Personas, cells, sizes

- Personas: the first `n` of 200 personas drawn from the population with seed `20261210`, main mode only. The control mode is not run with LLM attackers.
- Full set (primary): `n = 200`, four recipients, checkpoints 1, 3, 6, 12, 24; 36 000 prompts; cells = 3 models x 3 attributes x 4 recipients x 5 checkpoints x 2 statistics = 360.
- Reduced set (only if the full set is not finished; reported as reduced): `n = 60`, recipients doctor and family member, checkpoints 3, 12, 24; 3 240 prompts; 108 cells.
- One Holm family per set that was run, over all its cells. If both exist they are separate families and the full set is the primary result.

## v2.3 Verdict

- LLM family `bound held` if no cell is refuted, otherwise `bound refuted` with the refuting cells listed, reported as is.
- A model whose cache is incomplete is left out and named; with no complete model there is no verdict.
- Minimal detectable excess per cell is printed. With 200 personas only coarse understatement is detectable.

## v2.4 Deviations

A change to this section after the tag is published as `prereg-v3` with a reason.
