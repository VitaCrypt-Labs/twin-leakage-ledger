<!--
SPDX-FileCopyrightText: 2026 Paul Burg
SPDX-License-Identifier: CC-BY-4.0
-->

# Running the LLM attackers once on a free Kaggle GPU

Copyright 2026 Paul Burg.

The notebook `notebooks/kaggle_llm_attackers.ipynb` clones this public repository at the tag `prereg-v4`, installs the pinned extra packages (`requirements-llm.txt`), downloads the pinned weights listed in `docs/MODELS.md` and checks their SHA-256, runs `make reproduce-llm` and leaves the answers in `cache/llm/`. No secrets, no paid resources.

## What to switch on in Kaggle

1. A Kaggle account with a verified phone number (Kaggle requires it for GPU and for Internet access).
2. Create a notebook, then File, Import Notebook, and upload `notebooks/kaggle_llm_attackers.ipynb`.
3. Right sidebar, *Session options*:
   - **Accelerator: GPU T4 x2**
   - **Internet: On**
   - Persistence: no persistence needed.
4. Run All. The first cells print the GPU, the code commit and the plan with the estimated time.

## What the first real run taught

- Interactive sessions are stopped after about 40 idle minutes and lose their files. An unattended run must be a background version: **Save Version, Save & Run All (Commit), with the GPU accelerator and Internet on**. The queue for a background GPU version can take many hours; the output (`cache_llm.zip`) appears under the version's Output after it finishes.
- The reduced set for three models takes about 2.5 to 3.5 hours on two T4 (measured: about 2.1 to 2.7 s per answer, 49 minutes for Qwen).
- The run at `prereg-v3` showed two more things (see `PREREGISTRATION.md`, v4): Mistral-7B ran out of GPU memory with its weights on one card, so weights are now spread over both cards, batches are limited by a token budget and retried in halves on an out-of-memory error, and one failing model no longer stops the next; and Mistral often returns blocks that are not distributions, which is reported as abstention and marks the model a weak attacker above 20% (it is not dropped).
- The speed-check cell now runs the 32 longest prompts of every model and prints technical-loss and abstention rates; an out-of-memory problem shows up there, not after hours.
- The first run was invalid because answers were cut off by the token limit (see `PREREGISTRATION.md`, v3). The code now prints the share of unreadable answers per model and attribute and stops with an error above 5%. Look at the speed-check cell (32 answers): it must print a rate below 5% for all three attributes before you start the long run.

## Sizes and limits

- `SET = "reduced"` (default in the notebook): 60 personas, doctor and family member, three checkpoints, 3 240 prompts for three models.
- `SET = "full"`: 200 personas, four recipients, five checkpoints, 36 000 prompts for three models.
- A free session is limited (about 12 hours) and the weekly GPU quota is limited (about 30 hours). The estimate printed by the plan cell is a guess (`assumed_seconds_per_prompt` in `data/models.json`); run the speed-check cell first and multiply. For the full set run one model per session: set `MODELS = "qwen"`, then `"mistral"`, then `"phi"`.
- The run is resumable: answers are appended to `cache/llm/<model>.jsonl` and already finished prompts are skipped. After an interrupted session, add the previous output as notebook input, copy its `cache/llm/*.jsonl` into the clone and run the last code cell again.

## After the run

1. Download `cache_llm.zip` from the notebook Output.
2. Unzip it into the repository so that the files are `cache/llm/qwen.jsonl`, `cache/llm/mistral.jsonl`, `cache/llm/phi.jsonl`.
3. `make reproduce` (CPU only, offline) now adds the LLM attackers as their own Holm family and writes `results/llm_verdict.json` and `results/llm_table.md`. Without the cache it runs the CPU-only part alone.
4. The cache is data under CC0-1.0.

## Not verified

The code was tested on a server without GPU with a stub model and recorded answers (parsing, cache, resume, family assembly); the GPU path ran on Kaggle at tag `prereg-v4` (three models, reduced set). Possible friction on other hosts: the torch version of the Kaggle image against the pinned `transformers`, memory of Mistral-7B in float16 across two T4 cards, and the real speed.
