<!--
SPDX-FileCopyrightText: 2026 Paul Burg
SPDX-License-Identifier: CC-BY-4.0
-->

# Known issues and fragile places

Copyright 2026 Paul Burg.

- Cells of the decisive test where the recipient already knows the attribute are trivial (bound 1) but stay in the Holm family, as preregistered.
- Early checkpoints (month 1) carry almost no information; their cells sit at the threshold and test nothing.
- A1 numbers can differ in the last digits between machines (BLAS); `results/expected.json` uses tolerances, not equality.
- The limit (cap) acts on the three sensitive attributes only, not on the general bound L.
- Pregnancy for a person who is male in the table has probability zero and is not part of the closed set.
- Sensitivity: with 2 000 personas the test catches coarse understatement only; the minimal detectable excess per cell is printed.
- Not included in this release: the full LLM set (200 personas, four recipients, five checkpoints), LLM-written drafts, an LLM-based renderer, collusion between recipients, the adapter to external synthetic cohorts, Docker and CI. Three LLM attackers on the reduced set are included, with results.

- Mistral-7B-Instruct-v0.3 often answers with blocks that are not usable distributions: all zeros (mostly on wording p2), a single key such as `{"none": 1.0}`, percent signs or expressions. These are abstentions (base rates used), reported per model and attribute and marking the model a weak attacker above 20%. Measured on the v3 run: 38.4% diabetes, 9.3% depression, 28.2% pregnancy under the strict parser of v4.
- The stop-on-close criterion, the balanced two-GPU load and the out-of-memory retry could not be run on a GPU by the author of the code; they were exercised by the Kaggle run at tag `prereg-v4`, which completed all three models. There is no automated test of the GPU path.
- LLM results are from the reduced set only: n = 60 personas per cell, doctor and family member, checkpoints 3, 12, 24. Minimal detectable excess is about 3.4 (R) and 0.51 (D), so the LLM verdict "bound held" excludes only very large understatement. The full set (200 personas, four recipients, five checkpoints) has not been run; the control mode with LLM attackers is not defined in the preregistration and was not run.
- Abstention of the LLM attackers (base rates used when a block is not a usable distribution): Qwen 0.0 / 1.1 / 0.0%, Phi 3.8 / 15.2 / 9.9%, Mistral 39.7 / 10.0 / 26.4% for diabetes / depression / pregnancy. Mistral is marked a weak attacker; nothing is dropped.
- The LLM family is computed from a cache of sampled answers (temperature 0.3); regenerating the cache gives different answers. `results/expected.json` compares the numbers computed from the committed cache (exact), not a fresh model run.
