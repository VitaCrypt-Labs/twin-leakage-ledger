<!--
SPDX-FileCopyrightText: 2026 Paul Burg
SPDX-License-Identifier: CC-BY-4.0
-->

# Decisions

Copyright 2026 Paul Burg.

- **Exact ledger, not an estimate.** The set of histories is closed (46 875), so the likelihood vector `P(y|x)` is kept in full and updated by multiplication. Alternative: estimate leakage from samples; rejected because the decisive test needs exact prior and posterior.
- **No extractor in the main mode.** What leaves is chosen by declared rules and written by a template renderer that sees only the allowed claims. The extractor is used only in the control mode and in the usefulness count.
- **Persona state is the whole 24-month history.** Fixed `x`, so the likelihood vector is updated by multiplication; the absence of an expected scheduled message is an observation; a limit decision uses the worst case over possible contents and only the already released history.
- **Statistic clip.** `R_i` is clipped at 10 (preregistered). Without a clip the range of `R_i` is the inverse of the rarest prior value and the empirical Bernstein bound is useless. Clipping lowers the mean, so a refutation stays valid; the cost is lost sensitivity for rare values.
- **Family size 240 per mode** (2 attackers), not the 480 of the architecture draft (which counted three LLM attackers not in v1).
- **Control mode shares the sending decisions with the main mode** (same random numbers), so the two modes differ only in what the text contains. The control ledger equals the main ledger because the extractor recovers the rule claims from the drafts; this is checked, and a full control pass runs if it ever differs.
- **A1 gets the recipient's prior knowledge as extra features.** Without it the attacker would be weaker than the recipient it stands for.
- **Invented tables.** `data/population.json` is an assumption, not epidemiology. Passing on it says nothing about real deployments.

Trade-offs accepted: a small closed set instead of realistic records; template text instead of language-model text; no collusion between recipients.
