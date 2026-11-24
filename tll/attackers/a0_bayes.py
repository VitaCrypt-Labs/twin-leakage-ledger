# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""A0: exact Bayesian attacker. Knows the population table, the rules (or the agent model in the
control mode) and the recipient's prior knowledge. Reads raw message text, parses it, and keeps its
own posterior. Written separately from the ledger so that the two can be compared."""
from __future__ import annotations

import numpy as np

from ..sim import CHECKPOINTS, World


def run(world: World, recipient: str, variant: str, h: np.ndarray, outcomes: np.ndarray, mode: str,
        attrs: list[str], checkpoints=CHECKPOINTS, max_month: int | None = None):
    """outcomes[i, s.idx] identifies the message person i's recipient got in slot s; the attacker reads its text.
    Returns {checkpoint: {"post": {attr: (n, K)}, "ratio_hist": (n,)}} with the exact posteriors."""
    pop = world.pop
    known = world.variants[recipient][variant]
    n = len(h)
    slots = world.slots_of[recipient]
    K = {a: pop.n_values[a] for a in attrs}
    out = {c: {"post": {a: np.zeros((n, K[a])) for a in attrs}, "ratio_hist": np.zeros(n)} for c in checkpoints}
    for i in range(n):
        key = pop.known_key(known, h[i])
        sub = pop.subset(known, key)
        prior = pop.prior[sub]
        prior = prior / prior.sum()
        post_w = prior.copy()  # unnormalised posterior over histories
        idx = {a: pop.values[a][sub].astype(np.int64) for a in attrs}
        cks = sorted(checkpoints)
        ci = 0
        for s in slots:
            if max_month is not None and s.month > max_month:
                break
            while ci < len(cks) and s.month > cks[ci]:  # checkpoint passed before this slot
                _snapshot(out, cks[ci], i, post_w, prior, idx, attrs, K)
                ci += 1
            o = world.parse(s, world.text(s, int(outcomes[i, s.idx]), mode), mode)
            lik = world.lik(s, 0, o, sub) if mode == "main" else world.lik_control(s, o, sub)
            post_w = post_w * lik
            post_w /= post_w.sum()
        while ci < len(cks):
            _snapshot(out, cks[ci], i, post_w, prior, idx, attrs, K)
            ci += 1
    return out


def _snapshot(out, ck, i, post_w, prior, idx, attrs, K):
    for a in attrs:
        out[ck]["post"][a][i] = np.bincount(idx[a], weights=post_w, minlength=K[a])
    out[ck]["ratio_hist"][i] = float((post_w / prior).max())
