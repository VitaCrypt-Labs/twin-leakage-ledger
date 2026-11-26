<!--
SPDX-FileCopyrightText: 2026 Paul Burg
SPDX-License-Identifier: CC-BY-4.0
-->

# Dependencies

Copyright 2026 Paul Burg. Locked in `uv.lock` (hashes included), `exclude-newer = "2026-10-20"` in `pyproject.toml`.
Rule: every pinned release is more than one month older than the lock date (24.11.2026), so the upload date
is before 24.10.2026. `pip-audit` was run on the full lock (runtime and dev) before installing: no known vulnerabilities.
Python 3.12 is installed by uv (CPython build from the python-build-standalone project); it is not a PyPI package.
The optional `llm` extra (GPU step only, `make reproduce-llm`) adds transformers, accelerate and safetensors and their dependencies; `torch` is locked too but on a GPU host the host image's torch is used, see `docs/KAGGLE.md`.
Paid licences: none. Network at run time: none after install.

| Package | Version | Released (PyPI upload) | Role | Older than one month |
|---|---|---|---|---|
| accelerate | 1.15.0 | 2026-09-09 | llm extra | yes |
| annotated-doc | 0.0.5 | 2026-07-28 | transitive | yes |
| anyio | 4.15.1 | 2026-09-05 | transitive | yes |
| attrs | 26.1.0 | 2026-03-19 | transitive | yes |
| boolean-py | 5.0 | 2025-04-03 | transitive | yes |
| cachecontrol | 0.14.4 | 2025-11-14 | transitive | yes |
| certifi | 2026.10.18 | 2026-10-18 | transitive | yes |
| charset-normalizer | 3.5.2 | 2026-09-30 | transitive | yes |
| click | 8.5.0 | 2026-08-26 | transitive | yes |
| cloudpickle | 3.1.2 | 2025-11-03 | transitive | yes |
| colorama | 0.4.6 | 2022-10-25 | transitive | yes |
| contourpy | 1.4.0 | 2026-09-11 | transitive | yes |
| cuda-bindings | 13.4.4 | 2026-10-14 | transitive | yes |
| cuda-pathfinder | 1.8.3 | 2026-10-02 | transitive | yes |
| cuda-toolkit | 13.0.3.0 | 2026-04-14 | transitive | yes |
| cycler | 0.12.1 | 2023-10-07 | transitive | yes |
| cyclonedx-python-lib | 11.12.0 | 2026-08-13 | transitive | yes |
| defusedxml | 0.7.1 | 2021-03-08 | transitive | yes |
| filelock | 4.2.4 | 2026-10-19 | transitive | yes |
| fonttools | 4.67.0 | 2026-10-14 | transitive | yes |
| fsspec | 2026.9.0 | 2026-09-18 | transitive | yes |
| h11 | 0.16.0 | 2025-04-24 | transitive | yes |
| hf-xet | 1.7.0 | 2026-10-06 | transitive | yes |
| httpcore2 | 2.15.1 | 2026-10-19 | transitive | yes |
| httpx2 | 2.15.1 | 2026-10-19 | transitive | yes |
| httpx2-jsfetch | 1.0 | 2026-08-07 | transitive | yes |
| huggingface-hub | 2.2.0 | 2026-10-08 | transitive | yes |
| idna | 3.20 | 2026-09-17 | transitive | yes |
| iniconfig | 2.3.1 | 2026-10-06 | transitive | yes |
| jinja2 | 3.1.6 | 2025-03-05 | transitive | yes |
| joblib | 1.6.0 | 2026-08-31 | transitive | yes |
| kiwisolver | 1.5.1 | 2026-08-28 | transitive | yes |
| license-expression | 30.4.4 | 2025-07-22 | transitive | yes |
| markdown-it-py | 4.2.0 | 2026-05-07 | transitive | yes |
| markupsafe | 3.0.4 | 2026-10-02 | transitive | yes |
| matplotlib | 3.11.2 | 2026-09-11 | runtime | yes |
| mdurl | 0.1.2 | 2022-08-14 | transitive | yes |
| mpmath | 1.3.0 | 2023-03-07 | transitive | yes |
| msgpack | 1.2.3 | 2026-09-29 | transitive | yes |
| narwhals | 2.27.1 | 2026-10-10 | transitive | yes |
| networkx | 3.7 | 2026-09-21 | transitive | yes |
| numpy | 2.5.4 | 2026-10-10 | runtime | yes |
| nvidia-cublas | 13.1.1.3 | 2026-04-08 | transitive | yes |
| nvidia-cuda-cupti | 13.0.85 | 2025-09-04 | transitive | yes |
| nvidia-cuda-nvrtc | 13.0.88 | 2025-09-04 | transitive | yes |
| nvidia-cuda-runtime | 13.0.96 | 2025-10-09 | transitive | yes |
| nvidia-cudnn-cu13 | 9.24.0.43 | 2026-07-02 | transitive | yes |
| nvidia-cufft | 12.0.0.61 | 2025-09-04 | transitive | yes |
| nvidia-cufile | 1.15.1.6 | 2025-09-04 | transitive | yes |
| nvidia-curand | 10.4.0.35 | 2025-08-04 | transitive | yes |
| nvidia-cusolver | 12.0.4.66 | 2025-09-04 | transitive | yes |
| nvidia-cusparse | 12.6.3.3 | 2025-09-04 | transitive | yes |
| nvidia-cusparselt-cu13 | 0.8.1 | 2025-09-05 | transitive | yes |
| nvidia-nccl-cu13 | 2.30.7 | 2026-06-09 | transitive | yes |
| nvidia-nvjitlink | 13.4.92 | 2026-09-16 | transitive | yes |
| nvidia-nvshmem-cu13 | 3.4.5 | 2025-09-06 | transitive | yes |
| nvidia-nvtx | 13.0.85 | 2025-09-04 | transitive | yes |
| packageurl-python | 0.17.6 | 2025-11-24 | transitive | yes |
| packaging | 26.3 | 2026-08-04 | transitive | yes |
| pillow | 13.0.0 | 2026-10-16 | transitive | yes |
| pip | 26.2.1 | 2026-08-04 | transitive | yes |
| pip-api | 0.0.35 | 2026-09-08 | transitive | yes |
| pip-audit | 2.10.1 | 2026-06-10 | dev | yes |
| pip-requirements-parser | 32.0.1 | 2022-12-21 | transitive | yes |
| platformdirs | 4.13.5 | 2026-10-19 | transitive | yes |
| pluggy | 1.7.0 | 2026-10-15 | transitive | yes |
| psutil | 7.2.2 | 2026-01-28 | transitive | yes |
| py-serializable | 2.1.0 | 2025-07-21 | transitive | yes |
| pygments | 2.21.0 | 2026-08-17 | transitive | yes |
| pyparsing | 3.3.3 | 2026-09-20 | transitive | yes |
| pytest | 9.1.1 | 2026-06-19 | dev | yes |
| python-dateutil | 2.9.0.post0 | 2024-03-01 | transitive | yes |
| python-debian | 1.1.1 | 2026-06-07 | transitive | yes |
| python-magic | 0.4.27 | 2022-06-07 | transitive | yes |
| pyyaml | 6.0.3 | 2025-09-25 | transitive | yes |
| regex | 2026.9.29 | 2026-09-29 | transitive | yes |
| requests | 2.34.2 | 2026-05-14 | transitive | yes |
| reuse | 6.2.0 | 2025-10-27 | dev | yes |
| rich | 15.0.0 | 2026-04-12 | transitive | yes |
| safetensors | 0.8.0 | 2026-06-09 | llm extra | yes |
| scikit-learn | 1.9.1 | 2026-09-10 | runtime | yes |
| scipy | 1.18.1 | 2026-08-21 | runtime | yes |
| setuptools | 84.2.0 | 2026-10-19 | transitive | yes |
| shellingham | 1.5.4 | 2023-10-24 | transitive | yes |
| six | 1.17.0 | 2024-12-04 | transitive | yes |
| sortedcontainers | 2.4.0 | 2021-05-16 | transitive | yes |
| sympy | 1.14.0 | 2025-04-27 | transitive | yes |
| threadpoolctl | 3.7.0 | 2026-09-15 | transitive | yes |
| tokenizers | 0.23.3 | 2026-10-09 | transitive | yes |
| tomli | 2.5.0 | 2026-10-07 | transitive | yes |
| tomli-w | 1.2.0 | 2025-01-15 | transitive | yes |
| tomlkit | 0.15.2 | 2026-10-16 | transitive | yes |
| torch | 2.14.1 | 2026-09-30 | transitive | yes |
| tqdm | 4.70.1 | 2026-09-11 | transitive | yes |
| transformers | 5.20.2 | 2026-10-19 | llm extra | yes |
| triton | 3.8.0 | 2026-08-28 | transitive | yes |
| truststore | 0.10.5 | 2026-10-12 | transitive | yes |
| typer | 0.27.3 | 2026-10-06 | transitive | yes |
| typing-extensions | 4.16.0 | 2026-07-02 | transitive | yes |
| urllib3 | 2.8.0 | 2026-09-15 | transitive | yes |
