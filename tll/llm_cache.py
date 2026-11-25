# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""JSONL cache of LLM attacker answers. One file per model: cache/llm/<model_key>.jsonl.
First line is a header (model id, revision, persona set); every other line is one answer."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "cache" / "llm"
FORMAT_VERSION = 3
FALLBACK_THRESHOLD = 0.05  # preregistered in v3: above this per model and attribute the cache is not used
LICENSE_TAG = ["Copyright 2026 Paul Burg", "SPDX-License-" + "Identifier: CC0-1.0"]


def key_of(rec: dict) -> tuple:
    return (rec["prompt_id"], rec["persona"], rec["recipient"], rec["checkpoint"])


def read_header(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        line = f.readline()
    return json.loads(line) if line.strip() else None


def open_for_append(path: Path, header: dict):
    """Open a model file for appending; write the header when new, check it when the file exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    old = read_header(path)
    if old is None:
        f = open(path, "w", encoding="utf-8")
        f.write(json.dumps({"header": True, **header}, sort_keys=True) + "\n")
        f.flush()
        return f
    for k in ("model_key", "model_id", "revision", "set", "format_version"):
        if old.get(k) != header.get(k):
            raise ValueError(f"{path.name}: header differs on '{k}' ({old.get(k)} vs {header.get(k)}); use another cache dir")
    return open(path, "a", encoding="utf-8")


def append(f, rec: dict) -> None:
    f.write(json.dumps(rec, sort_keys=True) + "\n")
    f.flush()


def load_model(path: Path) -> tuple[dict, dict]:
    """Returns (header, {key: record}). The last record wins if a key repeats."""
    header, recs = None, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("header"):
                header = d
            else:
                recs[key_of(d)] = d
    if header is None:
        raise ValueError(f"{path}: no header")
    return header, recs


def list_model_files(cache_dir: Path) -> list[Path]:
    return sorted(Path(cache_dir).glob("*.jsonl")) if Path(cache_dir).is_dir() else []


def fallback_rates(path: Path) -> dict[str, float]:
    """Share of answers per attribute that fell back to the base rates (from the stored records)."""
    _, recs = load_model(path)
    n = max(len(recs), 1)
    counts: dict[str, int] = {}
    for r in recs.values():
        for a in r.get("fallback", []):
            counts[a] = counts.get(a, 0) + 1
    return {a: counts.get(a, 0) / n for a in ("diabetes", "depression", "pregnancy")}
