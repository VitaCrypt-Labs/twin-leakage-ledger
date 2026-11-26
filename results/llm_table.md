<!--
SPDX-FileCopyrightText: 2026 Paul Burg
SPDX-License-Identifier: CC0-1.0
-->

# LLM attackers (own Holm family, see PREREGISTRATION.md v2)

- Set: reduced (60 personas, recipients ['doctor', 'family'], checkpoints [3, 12, 24]).
- Verdict: **bound held (weak attackers, abstention above 20%: mistral)** (0 of 108 cells refuted).
- Incomplete models left out: none.
- Degraded models left out (technical loss above the limit): none.
- Technical loss: the answer was cut off and the base rates were used. Limit 5% per model and attribute; above it the model is refused.
- Abstention: the answer was complete but the attribute block was not a usable distribution; the base rates were used. Valid for the test, but it weakens the attacker. A model with abstention above 20% on any attribute is marked weak; no model is dropped for it.

| Model | Answers | Loss diabetes | Loss depression | Loss pregnancy | Abstain diabetes | Abstain depression | Abstain pregnancy | Mark |
|---|---|---|---|---|---|---|---|---|
| mistral | 1080 | 0.0% | 0.0% | 0.0% | 39.7% | 10.0% | 26.4% | weak attacker |
| phi | 1080 | 0.0% | 0.0% | 0.0% | 3.8% | 15.2% | 9.9% |  |
| qwen | 1080 | 0.0% | 0.0% | 0.0% | 0.0% | 1.1% | 0.0% |  |

Sensitivity with n = 60 personas per cell: minimal detectable excess (width of the lower bound at the strictest Holm step), median R 3.404 (range 3.311 to 3.842), median D 0.506 (range 0.506 to 0.506). Only large understatement is detectable.

At checkpoint 24. Mean R and D with the distribution-free 95% lower bound in brackets.

| Model | Recipient | Attribute | Mean R | Mean D | Refuted cells |
|---|---|---|---|---|---|
| mistral | doctor | diabetes | 0.961 [-0.540] | -0.017 [-0.333] | 0 |
| mistral | doctor | depression | 0.750 [-0.823] | -0.144 [-0.460] | 0 |
| mistral | doctor | pregnancy | 0.906 [-0.631] | -0.026 [-0.342] | 0 |
| mistral | family | diabetes | 0.913 [-0.622] | -0.070 [-0.386] | 0 |
| mistral | family | depression | 0.707 [-0.870] | -0.127 [-0.443] | 0 |
| mistral | family | pregnancy | 0.947 [-0.567] | -0.033 [-0.349] | 0 |
| phi | doctor | diabetes | 0.981 [-0.513] | -0.017 [-0.333] | 0 |
| phi | doctor | depression | 0.686 [-0.889] | -0.161 [-0.477] | 0 |
| phi | doctor | pregnancy | 0.878 [-0.666] | -0.043 [-0.359] | 0 |
| phi | family | diabetes | 0.936 [-0.569] | -0.070 [-0.386] | 0 |
| phi | family | depression | 0.675 [-0.893] | -0.127 [-0.443] | 0 |
| phi | family | pregnancy | 0.945 [-0.573] | -0.033 [-0.349] | 0 |
| qwen | doctor | diabetes | 1.000 [-0.459] | 0.000 [-0.316] | 0 |
| qwen | doctor | depression | 0.749 [-0.818] | -0.128 [-0.444] | 0 |
| qwen | doctor | pregnancy | 0.916 [-0.620] | -0.026 [-0.342] | 0 |
| qwen | family | diabetes | 0.902 [-0.671] | -0.070 [-0.386] | 0 |
| qwen | family | depression | 0.715 [-0.866] | -0.127 [-0.443] | 0 |
| qwen | family | pregnancy | 0.955 [-0.556] | -0.033 [-0.349] | 0 |
