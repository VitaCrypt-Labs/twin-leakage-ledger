# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""The committed LLM answers must match the recorded SHA-256 sums."""
import hashlib
from pathlib import Path

import pytest

DIR = Path(__file__).resolve().parent.parent / "cache" / "llm"


@pytest.mark.skipif(not (DIR / "SHA256SUMS").exists(), reason="no committed LLM cache")
def test_cache_files_match_recorded_sums():
    lines = [x.split() for x in (DIR / "SHA256SUMS").read_text().splitlines() if x.strip()]
    assert lines
    for digest, name in lines:
        assert hashlib.sha256((DIR / name).read_bytes()).hexdigest() == digest, name
    assert {n for _, n in lines} == {p.name for p in DIR.glob("*.jsonl")}
