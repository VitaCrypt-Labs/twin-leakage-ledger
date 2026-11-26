# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""Development helper (needs network, not used by make reproduce): writes docs/DEPENDENCIES.md from uv.lock
with the PyPI upload date of every locked wheel/sdist set. Usage: python scripts/dependency_table.py"""
import json
import re
import sys
import tomllib
import urllib.request
from datetime import date
from pathlib import Path

root = Path(__file__).resolve().parent.parent
lock = tomllib.loads((root / "uv.lock").read_text())
pyproject = tomllib.loads((root / "pyproject.toml").read_text())
direct = {re.split(r"[=<>]", d)[0] for d in pyproject["project"]["dependencies"]}
dev = {re.split(r"[=<>]", d)[0] for d in pyproject["dependency-groups"]["dev"]}
llm = {re.split(r"[=<>]", d)[0] for d in pyproject["project"]["optional-dependencies"]["llm"]}
cutoff = date(2026, 10, 24)  # one month before the lock date, 24.11.2026
rows, bad = [], 0
for pkg in sorted(lock["package"], key=lambda p: p["name"]):
    name, ver = pkg["name"], pkg["version"]
    if name == pyproject["project"]["name"]:
        continue
    d = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{ver}/json", timeout=30))
    day = min(f["upload_time_iso_8601"] for f in d["urls"])[:10]
    old = date.fromisoformat(day) < cutoff
    bad += not old
    role = "runtime" if name in direct else "dev" if name in dev else "llm extra" if name in llm else "transitive"
    rows.append(f"| {name} | {ver} | {day} | {role} | {'yes' if old else 'NO'} |")
text = f"""<!--
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
""" + "\n".join(rows) + "\n"
(root / "docs" / "DEPENDENCIES.md").write_text(text)
print(f"{len(rows)} packages, {bad} too new")
sys.exit(1 if bad else 0)
