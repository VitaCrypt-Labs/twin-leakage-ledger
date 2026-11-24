# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
# One command: make reproduce. Needs uv and network only for the first install.
export OMP_NUM_THREADS ?= 4
export OPENBLAS_NUM_THREADS ?= 4
export PYTHONHASHSEED = 0

.PHONY: install reproduce test lint

install:
	uv sync --frozen

reproduce: install
	uv run --frozen --no-sync python -m tll.reproduce

test: install
	uv run --frozen --no-sync pytest -q

lint: install
	uv run --frozen --no-sync reuse lint
