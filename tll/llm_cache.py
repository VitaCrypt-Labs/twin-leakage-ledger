# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""JSONL cache of LLM attacker answers. One file per model: cache/llm/<model_key>.jsonl.
First line is a header (model id, revision, persona set); every other line is one answer."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "cache" / "llm"
FORMAT_VERSION = 4
ACCEPTED_FORMATS = (3, 4)  # 3: written by the v3 code, readable for local checks only (see PREREGISTRATION.md v4)
FALLBACK_THRESHOLD = 0.05  # technical loss (answer cut off): above this per model and attribute the cache is refused
WEAK_ABSTENTION = 0.20  # abstention above this on any attribute marks the model a weak attacker; nothing is dropped
ATTRS = ("diabetes", "depression", "pregnancy")
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


def record_kinds(rec: dict) -> tuple[list[str], list[str]]:
    """(lost, abstained) attributes of one record. Records of the v3 code have only `fallback` and `closed`."""
    if "loss" in rec:
        return rec["loss"], rec.get("abstain", [])
    fb = rec.get("fallback", [])
    return ([], fb) if rec.get("closed", False) else (fb, [])


def rates(path: Path) -> dict:
    """Per attribute: share of answers lost to a cut-off (technical loss) and share of abstentions."""
    _, recs = load_model(path)
    n = max(len(recs), 1)
    loss = {a: 0 for a in ATTRS}
    abst = {a: 0 for a in ATTRS}
    for r in recs.values():
        lo, ab = record_kinds(r)
        for a in lo:
            loss[a] += 1
        for a in ab:
            abst[a] += 1
    return {"n": len(recs), "loss": {a: v / n for a, v in loss.items()}, "abstain": {a: v / n for a, v in abst.items()}}


def fallback_rates(path: Path) -> dict[str, float]:
    """Technical-loss rate per attribute (the quantity the 5% limit applies to)."""
    return rates(path)["loss"]
