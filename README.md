<!--
SPDX-FileCopyrightText: 2026 Paul Burg
SPDX-License-Identifier: CC-BY-4.0
-->

# Twin Leakage Ledger

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24502998.svg)](https://doi.org/10.5281/zenodo.24502998)

Copyright 2026 Paul Burg. Text: CC BY 4.0. Code: Apache-2.0. Data: CC0-1.0.

A research module on **synthetic data**. A personal health agent sends small messages to a doctor, a pharmacy bot, an insurer and a family member. One message says little; hundreds together can reveal a diagnosis the person never shared. The ledger keeps, per recipient, one number: by how much, at most, that recipient's ability to guess any fact about the person has grown over what the recipient already knew. The number is checked by attackers that read the same messages.

This is research code. It is not a product, makes no guarantee about any shipping product, and says nothing clinical. No real person, no real health record and no personal data are used anywhere. In the current VitaCrypt MVP the server can decrypt user data; nothing in this repository changes that.

The repository became public on GitHub on 24.11.2026, about an hour after the first local commits. The full history from the first commit is public.

## Run it

```bash
make reproduce
```

One command, no GPU, no network after the install, fixed seeds. It prints the verdict for both modes and writes `results/main_table.md`, `results/figure1.png`, `results/verdict.json`, `results/utility.md`, then compares the numbers with `results/expected.json` (tolerances, same verdicts). `make test` runs the tests, `make lint` runs `reuse lint`. Python 3.12 and [uv](https://docs.astral.sh/uv/); versions are locked in `uv.lock`, see `docs/DEPENDENCIES.md`.

## Results at a glance

Numbers from `make reproduce` (2 000 test personas per mode, fixed seeds, `results/`). All on invented synthetic data.

- **Main mode: bound held.** 0 of 240 cells refuted (A0 and A1, three sensitive attributes, four recipients, five checkpoints, statistics R and D). A0's history-level ratio equals the ledger's L to 1e-15.
- **Control mode: violation caught.** 14 of 240 cells refuted, all on statistic R: insurer/depression, family member/diabetes and doctor/depression, where the planted phrases carry information that the extractor does not see. Mean R of the best attacker reaches about 2.1 against a bound of 1.
- **Gap, best attacker vs per-attribute bound.** At 24 months the best attacker's mean R is 0.93 to 0.95 per recipient, that is the bound is 1.06 to 1.07 times looser than the best attacker on average. The average hides a heavy tail: the 95th percentile of L_A is 14 to 17 for the attributes a recipient receives claims about.
- **Gap, per-attribute vs general bound.** Median L / L_A is 1.12 to 1.43 depending on the recipient.
- **Usefulness.** (1) Rules deliver 0.76 / 1.00 / 0.34 / 0.08 of the needed true facts for doctor / pharmacy / insurer / family against about 0.95 for the simulated free agent. (2) A limit of 5x coarsens or drops 70 to 100 percent of messages for the doctor, pharmacy and insurer, because the worst case over possible contents passes it; a limit of 10x costs the doctor 18 percent of needed facts and the pharmacy all of them. (3) See the gap lines above. Details: `results/utility.md`.
- **Recipient knows more.** With the ledger configured for the published `more` variant: 0 of 120 cells refuted. With the ledger left on `base` while the attacker holds the `more` knowledge: 8 of 120 refuted, the stated edge of the method.
- Sensitivity: minimal detectable excess is about 0.12 for R and 0.09 for D per cell at 2 000 personas; thin understatement is not detectable.

## Results of the LLM attackers (reduced set, preregistration v2 to v4)

Three open-weight models (Qwen3-4B-Instruct-2507, Mistral-7B-Instruct-v0.3, Phi-4-mini-instruct) read the main-mode messages of 60 personas; recipients doctor and family member; checkpoints 3, 12 and 24 months; three prompt wordings per model (3 240 answers, `cache/llm/`, sums in `cache/llm/SHA256SUMS`). They form their own Holm family of 108 cells (`results/llm_table.md`, `results/llm_verdict.json`).

- **Verdict: bound held**, 0 of 108 cells refuted.
- **This is a weak test.** With n = 60 personas per cell the minimal detectable excess is about 3.4 for R and 0.51 for D, and every lower bound is negative: only a very large understatement of the bound could have been caught. The full set (200 personas, four recipients, five checkpoints, 36 000 prompts) has not been run. The control mode was not run with LLM attackers; the preregistration does not define it.
- **Abstention** (complete answer, unusable block, base rates used): Qwen 0.0% diabetes, 1.1% depression, 0.0% pregnancy; Phi 3.8%, 15.2%, 9.9%; Mistral 39.7%, 10.0%, 26.4%. Mistral is marked a **weak attacker** (above 20%). Technical loss (answers cut off) is 0% for all three.
- **How close to the bound:** at 24 months the mean R of the three LLMs over the five non-trivial doctor and family cells is 0.82 to 0.85, against 0.94 for A0 and A1 (n = 2 000) on the same cells. The LLMs are further from the per-attribute bound than the exact and trained attackers, and A0 is the closest.

## What is simulated

- **Persona.** A 24-month history from a closed, enumerable set of 46 875 histories (age group, sex, pregnancy start, and start months of diabetes, depression, antidiabetic medication, SSRI-class medication, raised HbA1c). The probability tables in `data/population.json` are invented assumptions, not epidemiology.
- **Recipients.** Doctor, pharmacy bot, insurer, family member. What each knows in advance is in `data/recipients.json` (a `base` and a `more` variant) and is published before any attack runs.
- **Main mode.** Declared release rules (`tll/policy.py`, plain code) decide which claims go to which recipient and on what schedule. A deterministic template renderer (`tll/renderer.py`) writes the message from those claims only; it never sees the rest of the record or any draft. The probability of every possible message is therefore known exactly, which is why the ledger can be exact. Whether a message is sent at all, and when, is part of what is counted. The absence of an expected message counts like a message.
- **Control mode.** A simulated free agent writes drafts with extra phrases planted in them (`tll/agent_sim.py`). The draft goes out as written. The ledger counts only what a rule-based extractor finds (`tll/extractor.py`). Here the bound must break and the decisive test must catch it; if it does not, the experiment is declared uninformative and `make reproduce` fails.
- **Limit.** The person can set a limit (2x, 5x, 10x). Before a message goes out the ledger picks the finest version (full, coarse, none) whose worst case over every possible content stays within the limit. The decision uses only the released history and the rules, never the current content.
- **Attackers.** A0, exact Bayesian (knows the population and the rules). A1, a logistic regression trained on the raw text of 20 000 separate synthetic personas. Three open-weight LLM attackers (Qwen, Mistral, Phi) are included on the reduced set, with results below; the full LLM set is not run.

## The bound

For recipient R with prior knowledge k and all messages y so far:

- `L = max_x P(y|x) / P_R(y)`, pointwise maximal leakage (Saeidian et al., arXiv 2205.04935). The ledger stores the vector `P(y|x)` over all histories and updates it by multiplication after every slot, so the number is exact.
- `L_A = max_a P(a|y) / P_R(a)` for each attribute A the person named sensitive (diabetes, pregnancy, depression). `L_A <= L`.

Two consequences hold for any attacker that outputs a distribution q over the values of A as a function of y. Proofs in two lines each:

1. **Guessing.** `P(attacker names the true value | y) = [P(a_hat|y) / P_R(a_hat)] * P_R(a_hat) <= L_A * max_a P_R(a)`.
2. **Stated probability.** `E[ q(a) / (P_R(a) * L_A) | y ] = sum_a [P(a|y) / P_R(a)] * q(a) / L_A <= sum_a q(a) = 1`, because every ratio `P(a|y) / P_R(a)` is at most `L_A`. This holds even for an overconfident attacker.

The bound is a statement about the average over personas, not about a single prediction: one overconfident guess can exceed it with no leak at all. So the test (`PREREGISTRATION.md`) compares per-persona statistics `R_i` and `D_i` averaged over 2 000 personas with distribution-free bounds (empirical Bernstein for R, Hoeffding for D), corrected for many cells with Holm's method.

### Four published assumptions

The bound is valid in the main mode if all four hold:

1. The table of what the recipient knows in advance is correct.
2. Recipients do not pool what they receive (collusion is future work).
3. A fact about the person is determined by the history from the closed set, or is tied to the person only through it.
4. The code of the release rules is the code the ledger computes with (checked by `tests/test_policy.py`).

The `more` variant and the mismatched run in `results/` show what happens when assumption 1 is stretched.

## Layout

| Path | What it is |
|---|---|
| `data/` | population tables, claim vocabulary, recipients (CC0-1.0) |
| `tll/population.py`, `vocabulary.py` | the closed set of histories, the exact prior, the claims |
| `tll/policy.py`, `renderer.py` | release rules and the template renderer |
| `tll/ledger.py` | the ledger and the limit decision |
| `tll/agent_sim.py`, `extractor.py` | control mode: free drafts with planted phrases, claim extractor |
| `tll/attackers/` | A0 exact Bayesian, A1 trained classifier, `a_llm.py` open-weight LLM attackers |
| `tll/llm_cache.py`, `llm_family.py`, `data/models.json`, `cache/llm/`, `notebooks/` | LLM answer cache, the LLM family of the test, pinned models, the committed answers, the Kaggle notebook |
| `tll/test_claim.py` | decisive test: intervals, Holm, minimal detectable excess |
| `tll/utility.py` | cost in usefulness: price of the rules, price of a limit, price of the bound's coarseness |
| `tll/reproduce.py`, `Makefile`, `results/expected.json` | one command and the check against expected numbers |
| `PREREGISTRATION.md` | the refutation rule, committed and tagged `prereg-v1` before any result; v2 to v4 for the LLM attackers |

## Limits of this release

- A small closed set of histories, not realistic records. It buys exact numbers; scaling is future work.
- The rules and the planted phrases are our own assumptions. Passing on this population says nothing about a real deployment.
- The renderer is a template. Not included: a renderer based on an open language model that sees only the allowed claims, LLM-written drafts, collusion between recipients, the adapter to external synthetic cohorts, Docker, CI and an on-device prototype. The LLM attackers ran on the reduced set only (60 personas, doctor and family member, three checkpoints); the full set of 200 personas, four recipients and five checkpoints has not been run, and the LLM test at n = 60 is weak (see above).

## Use of AI tools

Parts of the code and text were written with Claude Code (Anthropic) under the author's direction; commits carry a co-author line. The author is responsible for the content.

## LLM attackers (GPU step, once)

Three open-weight models (Qwen, Mistral, Phi; Apache-2.0 or MIT; pins in `docs/MODELS.md`) run as attackers with `make reproduce-llm`, once, on a free GPU: see `docs/KAGGLE.md`. The answers of the reduced set are committed in `cache/llm/` (CC0-1.0, sums in `SHA256SUMS`). `make reproduce` stays CPU-only and offline; with the cache present, as in this repository, it adds the LLM attackers as their own Holm family (`PREREGISTRATION.md`, v2 to v4); without the cache it runs the CPU-only part alone.

## How to cite

Cite the software with the concept DOI, which always resolves to the latest version: [10.5281/zenodo.24502998](https://doi.org/10.5281/zenodo.24502998). Burg, P. (2026). Twin Leakage Ledger [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.24502998. Machine-readable: `CITATION.cff`.

## License

Code Apache-2.0 (`LICENSE`), data CC0-1.0, texts CC BY 4.0; every file carries an SPDX identifier (`REUSE.toml` covers files that cannot hold comments).
