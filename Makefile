# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
# One command: make reproduce. Needs uv and network only for the first install.
export OMP_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS ?= 1
export TLL_WORKERS ?= 4
export PYTHONHASHSEED = 0

.PHONY: install reproduce reproduce-llm test lint

install:
	uv sync --frozen

reproduce: install
	uv run --frozen --no-sync python -m tll.reproduce $(ARGS)

# GPU step, run once on a free Kaggle GPU (docs/KAGGLE.md). Writes cache/llm/. make reproduce then adds the family.
LLM_ARGS ?= --set full
# On a host that already has torch and the pinned requirements-llm.txt (Kaggle): make reproduce-llm SYSTEM_PYTHON=1
reproduce-llm:
ifeq ($(SYSTEM_PYTHON),1)
	python -m tll.attackers.a_llm $(LLM_ARGS)
else
	uv sync --frozen --extra llm
	uv run --frozen --no-sync python -m tll.attackers.a_llm $(LLM_ARGS)
endif

test: install
	uv run --frozen --no-sync pytest -q

lint: install
	uv run --frozen --no-sync reuse lint
