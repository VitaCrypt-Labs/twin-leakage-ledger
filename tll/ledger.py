# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""Ledger for one recipient and one person.

State: the likelihood vector ell[x] = P(y | x) over every history x consistent with what the recipient
knows in advance (the vector is rescaled, only ratios matter). After each message ell is multiplied
by the likelihood of what was sent, including the case "nothing sent". The bounds are exact:

  L   = max_x P(y|x) / P_R(y)                 (pointwise maximal leakage)
  L_A = max_a P(a|y) / P_R(a)                 (per attribute, never above L)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Bounds:
    L: float
    LA: dict


class Ledger:
    def __init__(self, prior: np.ndarray, attrs: dict):
        """prior: P_R(x) over the histories of this ledger (sums to 1).
        attrs: name -> (integer value index per history, number of values)."""
        self.p = np.asarray(prior, dtype=float)
        self.attrs = attrs
        self.ell = np.ones(len(self.p))
        self.log_odometer = 0.0  # sum of per-message log losses, for comparison only
        self.prior_marginal = {a: np.bincount(idx, weights=self.p, minlength=k) for a, (idx, k) in attrs.items()}

    def update(self, lik: np.ndarray) -> None:
        total = float((self.p * lik).sum())
        if total <= 0.0:
            raise ValueError("observed outcome has zero probability under the prior")
        self.log_odometer += float(np.log(lik.max() / total))
        self.ell = self.ell * lik
        self.ell /= self.ell.max()

    def posterior(self, name: str) -> np.ndarray:
        idx, k = self.attrs[name]
        w = np.bincount(idx, weights=self.p * self.ell, minlength=k)
        return w / w.sum()

    @staticmethod
    def _ratio(post: np.ndarray, prior: np.ndarray) -> float:
        m = prior > 0
        return float((post[m] / prior[m]).max())

    def bounds(self) -> Bounds:
        L = float(self.ell.max() / (self.p * self.ell).sum())
        return Bounds(L, {a: self._ratio(self.posterior(a), self.prior_marginal[a]) for a in self.attrs})

    def worst_case_LA(self, liks: list[np.ndarray], names: list[str]) -> float:
        """Largest max_A L_A over every possible outcome of the coming slot (nothing sent included)."""
        worst = 1.0
        base = self.p * self.ell
        for lik in liks:
            w = base * lik
            tot = w.sum()
            if tot <= 0:
                continue
            for a in names:
                idx, k = self.attrs[a]
                post = np.bincount(idx, weights=w, minlength=k) / tot
                worst = max(worst, self._ratio(post, self.prior_marginal[a]))
        return worst


def decide_level(ledger: Ledger, liks_by_level: dict, cap: float, names: list[str]) -> int:
    """Pick the finest level whose worst case over all possible contents stays within the cap.
    Uses only the ledger state (already released history) and the rules, never the current content.
    liks_by_level: level -> list of likelihood vectors, one per possible outcome."""
    for level in (0, 1):
        if ledger.worst_case_LA(liks_by_level[level], names) <= cap + 1e-12:
            return level
    return 2


def run_ledger(world, recipient: str, variant: str, h: np.ndarray, outcomes: np.ndarray, attrs: list[str],
               checkpoints, mode: str = "main", max_month: int | None = None) -> dict:
    """Run the ledger of `recipient` for every person. mode "control": the ledger sees only what the
    extractor finds in the draft text. Returns per checkpoint the bounds and the prior facts the test needs."""
    pop = world.pop
    known = world.variants[recipient][variant]
    n = len(h)
    cks = sorted(checkpoints)
    out = {c: {"L": np.zeros(n), "LA": {a: np.zeros(n) for a in attrs},
               "PR_true": {a: np.zeros(n) for a in attrs}, "pmax": {a: np.zeros(n) for a in attrs},
               "log_odo": np.zeros(n)} for c in cks}
    templates: dict = {}
    for i in range(n):
        key = pop.known_key(known, h[i])
        t = templates.get(key)
        if t is None:
            sub = pop.subset(known, key)
            p = pop.prior[sub] / pop.prior[sub].sum()
            idx = {a: (pop.values[a][sub].astype(np.int64), pop.n_values[a]) for a in attrs}
            t = (sub, p, idx)
            templates[key] = t
        sub, p, idx = t
        led = Ledger(p, idx)
        ci = 0

        def snap(c):
            b = led.bounds()
            out[c]["L"][i] = b.L
            out[c]["log_odo"][i] = led.log_odometer
            for a in attrs:
                out[c]["LA"][a][i] = b.LA[a]
                pm = led.prior_marginal[a]
                out[c]["PR_true"][a][i] = pm[int(pop.values[a][h[i]])]
                out[c]["pmax"][a][i] = pm.max()

        for s in world.slots_of[recipient]:
            if max_month is not None and s.month > max_month:
                break
            while ci < len(cks) and s.month > cks[ci]:
                snap(cks[ci])
                ci += 1
            o = int(outcomes[i, s.idx])
            if mode == "control":
                o = world.parse(s, world.text(s, o, "control"), "main")
            led.update(world.lik(s, 0, o, sub))
        while ci < len(cks):
            snap(cks[ci])
            ci += 1
    return out
