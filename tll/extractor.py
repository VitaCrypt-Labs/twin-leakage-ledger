# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""Rule-based extractor of vocabulary claims from a text. Exact-sentence rules only.
Used in the control mode (what the ledger counts) and in the usefulness count.
It does not see anything outside the vocabulary, which is the point of the control mode."""
from __future__ import annotations

import re

from .vocabulary import Vocabulary


class Extractor:
    def __init__(self, vocab: Vocabulary):
        self.rules = []
        for cid, c in vocab.claims.items():
            lab = re.escape(c["label"])
            yes = re.compile(rf"(?:{lab}: yes$|{re.escape(c['agent_yes'])})", re.M)
            no = re.compile(rf"(?:{lab}: no$|{re.escape(c['agent_no'])})", re.M)
            self.rules.append((cid, yes, no))

    def extract(self, text: str) -> dict[str, bool]:
        found = {}
        for cid, yes, no in self.rules:
            if yes.search(text):
                found[cid] = True
            elif no.search(text):
                found[cid] = False
        return found
