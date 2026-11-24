# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""Closed list of claims. Each claim is a yes/no fact about the history at a given month."""
from __future__ import annotations

import numpy as np

from .population import Population, load_json


class Vocabulary:
    def __init__(self, spec: dict | None = None, duration: int = 9):
        spec = spec or load_json("vocabulary.json")
        self.kinds = spec["kinds"]
        self.claims = {c["id"]: c for c in spec["claims"]}
        self.duration = duration

    def ids(self) -> list[str]:
        return list(self.claims)

    def label(self, cid: str) -> str:
        return self.claims[cid]["label"]

    def vector(self, pop: Population, cid: str, month: int) -> np.ndarray:
        """Boolean value of the claim at `month` for every history."""
        c = self.claims[cid]
        s = pop.start_month[c["var"]]
        if c["type"] == "onset":
            return s <= month
        return (s <= month) & (month <= s + self.duration - 1)

    def scalar(self, record: dict, cid: str, month: int) -> bool:
        """Same claim from one plain record (independent code path, used by tests and the scalar sampler)."""
        c = self.claims[cid]
        s = record[c["var"]]
        if s is None:
            return False
        if c["type"] == "onset":
            return s <= month
        return s <= month <= s + self.duration - 1
