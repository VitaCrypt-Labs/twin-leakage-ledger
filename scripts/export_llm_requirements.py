# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""Development helper: writes requirements-llm.txt, the pinned packages that the optional `llm` extra adds on top
of the base install, without torch (the host image provides torch and the GPU libraries).
Run from the repo root after `uv lock`: python scripts/export_llm_requirements.py"""
import re
import subprocess
import tomllib
from pathlib import Path

root = Path(__file__).resolve().parent.parent
lock = tomllib.loads((root / "uv.lock").read_text())
pk = {p["name"]: p for p in lock["package"]}
me = pk["twin-leakage-ledger"]


def closure(names):
    seen, todo = set(), list(names)
    while todo:
        n = todo.pop()
        if n in seen or n not in pk:
            continue
        seen.add(n)
        todo += [d["name"] for d in pk[n].get("dependencies", [])]
    return seen


base = closure(d["name"] for d in me.get("dependencies", []))
llm = closure(d["name"] for d in me["optional-dependencies"]["llm"])
torch = closure(["torch"])
keep = llm - base - torch
out = subprocess.run(["uv", "export", "--frozen", "--extra", "llm", "--no-hashes", "--no-emit-project"],
                     cwd=root, capture_output=True, text=True, check=True).stdout
lines = []
for line in out.splitlines():
    m = re.match(r"^([A-Za-z0-9_.\-]+)==", line)
    if m and m.group(1).lower().replace("_", "-") in keep:
        lines.append(line.split(" \\")[0])
head = ["# SPDX-FileCopyrightText: 2026 Paul Burg", "# SPDX-License-Identifier: Apache-2.0",
        "# Pinned extras for the LLM attackers on a host that already has torch (see docs/KAGGLE.md).",
        "# Install with: pip install --no-deps -r requirements-llm.txt"]
(root / "requirements-llm.txt").write_text("\n".join(head + sorted(lines)) + "\n")
print(len(lines), "packages")
