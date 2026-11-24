# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""Deterministic template renderer. Its only inputs are the kind, the month and the claims the rules allowed.
It has no access to the persona record, to a draft, or to any randomness."""
from __future__ import annotations

from collections.abc import Mapping

from .vocabulary import Vocabulary


def render(vocab: Vocabulary, kind: str, month: int, claims: Mapping[str, bool]) -> str:
    lines = [f"Month {month} | {vocab.kinds[kind]['title']}"]
    for cid, value in claims.items():
        lines.append(f"Month {month} | {vocab.label(cid)}: {'yes' if value else 'no'}")
    return "\n".join(lines)
