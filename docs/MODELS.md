<!--
SPDX-FileCopyrightText: 2026 Paul Burg
SPDX-License-Identifier: CC-BY-4.0
-->

# Models of the LLM attackers

Copyright 2026 Paul Burg. Machine-readable copy: `data/models.json` (used by the code to download and to verify hashes).

Licences were read on the model cards and in the model repositories on 24.11.2026. Rule: weights under Apache-2.0 or MIT only; a model whose licence carries field-of-use or non-commercial terms is excluded.

| Family | Model id | Revision (commit) | Licence | Release (repository created) |
|---|---|---|---|---|
| Qwen | `Qwen/Qwen3-4B-Instruct-2507` | `cdbee75f17c01a7cc42f958dc650907174af0554` | Apache-2.0 | 2025-08-05 |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | Apache-2.0 | 2024-05-22 |
| Phi | `microsoft/Phi-4-mini-instruct` | `cfbefacb99257ffa30c83adab238a50856ac3083` | MIT | 2025-02-19 |

Where the licence was read: Qwen3-4B-Instruct-2507 (model card header `license: apache-2.0` and an Apache-2.0 `LICENSE` file in the repository), Phi-4-mini-instruct (model card header `license: mit` and an MIT `LICENSE` file), Mistral-7B-Instruct-v0.3 (model card header `license: apache-2.0`; the repository has no separate licence file).

## File hashes (SHA-256)

The code downloads the pinned revision and compares these hashes after the download; a mismatch stops the run. Files not listed (configuration and small tokenizer files) are fixed by the commit hash above.

| Model | File | SHA-256 |
|---|---|---|
| qwen | `model-00001-of-00003.safetensors` | `75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6` |
| qwen | `model-00002-of-00003.safetensors` | `0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1` |
| qwen | `model-00003-of-00003.safetensors` | `7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d` |
| qwen | `tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| mistral | `model-00001-of-00003.safetensors` | `ce6fb6f6f4d0183f4813cbf4ece24109da629a08d4210da46f77e1d8b0bd5c19` |
| mistral | `model-00002-of-00003.safetensors` | `8c0e72f148366b6a3709e002a98706a33d31aec8515090c856c95b2044f92ae0` |
| mistral | `model-00003-of-00003.safetensors` | `905dd405363e43d95779c1c1155a2dbfd36155914ae95dbd934e12e490cfb4ca` |
| mistral | `tokenizer.model` | `37f00374dea48658ee8f5d0f21895b9bc55cb0103939607c8185bfd1c6ca1f89` |
| phi | `model-00001-of-00002.safetensors` | `bc703090b63eda16f639fa4de7ac54635c23105ab1da2f6ec4d3403151d38ee6` |
| phi | `model-00002-of-00002.safetensors` | `7ff79b9d2d31076bac2663393451f6530f4fc8ca49b09002116c92c373dba983` |
| phi | `tokenizer.json` | `382cc235b56c725945e149cc25f191da667c836655efd0857b004320e90e91ea` |

## Not used, and why

- Llama and Gemma families: licence terms with use restrictions.
- `mistralai/Ministral-8B-Instruct-2410`: licence listed as `other` (research licence).
- Other Qwen sizes with a non-Apache licence (for example some 3B releases): checked per model; only repositories marked Apache-2.0 are candidates.

## Run assumptions

Weights are loaded in float16 on one or two GPUs (a free Kaggle T4 x2 session). The assumed time per prompt in `data/models.json` is a guess for planning, not a measurement; `make reproduce-llm LLM_ARGS="--set reduced --limit 32"` measures it.

## Answer length (measured with each model's own tokenizer)

Tokens of a complete answer for the largest value sets (diabetes and depression 5 values each, pregnancy 4), by number format. The first real run used a budget of 160 tokens, which is below the pretty-printed answer even with 3-digit numbers for Qwen and Mistral.

| Model | Pretty, 3 digits | Pretty, 10 digits | Pretty, 16 digits | One line, 10 digits |
|---|---|---|---|---|
| qwen | 186 | 284 | 368 | 256 |
| mistral | 207 | 305 | 389 | 259 |
| phi | 152 | 194 | 222 | 166 |

The budget is `max_new_tokens = 800` (more than twice the longest value, 389); generation stops when the JSON object closes.

## Memory plan on two 15 GiB GPUs (estimate, not measured)

Longest prompt of the full set (doctor, month 24): Qwen 1 640, Mistral 1 800, Phi 1 550 tokens. Weights are spread evenly over both cards; the answer budget is 800 tokens. Key-value cache per token: Qwen 0.14 MiB, Mistral 0.125 MiB, Phi 0.125 MiB.

| Model | Weights per card | Token budget | Largest batch at the longest prompt | Cache of that batch | Expected peak per card |
|---|---|---|---|---|---|
| qwen | about 3.8 GiB | 14 000 | 5 | about 1.7 GiB | about 5.5 GiB |
| mistral | about 6.8 GiB | 10 000 | 3 | about 1.0 GiB | about 8.5 GiB |
| phi | about 3.8 GiB | 14 000 | 5 | about 1.4 GiB | about 5.5 GiB |

The v3 run died with Mistral weights (13.5 GiB) on one card and batch 8. Peak includes weights, cache, prefill activations and about 0.5 GiB of overhead. Check the first minutes of the run: the loader prints the free memory after loading each model.
