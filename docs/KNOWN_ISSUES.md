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
- LLM attackers, LLM drafts, collusion, the adapter to external synthetic cohorts, Docker and CI are not part of this version.

- Mistral-7B-Instruct-v0.3 often answers with blocks that are not usable distributions: all zeros (mostly on wording p2), a single key such as `{"none": 1.0}`, percent signs or expressions. These are abstentions (base rates used), reported per model and attribute and marking the model a weak attacker above 20%. Measured on the v3 run: 38.4% diabetes, 9.3% depression, 28.2% pregnancy under the strict parser of v4.
- The stop-on-close criterion, the balanced two-GPU load and the out-of-memory retry were not run on a GPU by the author of the code; they are checked by the first speed-check run on Kaggle.
