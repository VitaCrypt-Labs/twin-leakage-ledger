# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""A1: multinomial logistic regression on the raw text one recipient received. Features are the
sentences of each message tagged with the month it was sent, plus a one-hot of what the recipient
already knows. Trained on separate synthetic personas; knows nothing about the rules."""
from __future__ import annotations

import re
import warnings

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .. import policy
from ..sim import CHECKPOINTS, World

_SPLIT = re.compile(r"(?<=[.!?])\s+|\n")


class FeatureSpace:
    """Fixed feature dictionary built from every possible text of one recipient (no data dependence)."""

    def __init__(self, world: World, recipient: str, variant: str, mode: str):
        self.world, self.recipient, self.mode = world, recipient, mode
        self.known = world.variants[recipient][variant]
        self.col: dict = {}
        self.month: list[int] = []
        self.cols_of: dict = {}
        for s in world.slots_of[recipient]:
            n_out = policy.n_outcomes(s.kind, 0) if mode == "main" else 1 + world.n_codes(s) * 8
            for o in range(1, n_out):
                cols = []
                for sent in _SPLIT.split(world.text(s, o, mode)):
                    if not sent:
                        continue
                    key = f"{s.month}|{sent}"
                    if key not in self.col:
                        self.col[key] = len(self.month)
                        self.month.append(s.month)
                    cols.append(self.col[key])
                self.cols_of[(s.idx, o)] = cols
        pop = world.pop
        self.known_off = {}
        for a in self.known:
            self.known_off[a] = len(self.month)
            self.month += [0] * pop.n_values[a]
        self.month = np.array(self.month)

    @property
    def n_features(self) -> int:
        return len(self.month)

    def matrix(self, h: np.ndarray, outcomes: np.ndarray):
        rows, cols = [], []
        for s in self.world.slots_of[self.recipient]:
            col = outcomes[:, s.idx]
            for o in np.unique(col):
                if o == 0:
                    continue
                r = np.flatnonzero(col == o)
                for c in self.cols_of[(s.idx, int(o))]:
                    rows.append(r)
                    cols.append(np.full(len(r), c))
        for a, off in self.known_off.items():
            rows.append(np.arange(len(h)))
            cols.append(off + self.world.pop.values[a][h].astype(np.int64))
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        data = np.ones(len(rows))
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(h), self.n_features))


def fit_predict(world: World, recipient: str, variant: str, mode: str, h_tr, out_tr, h_te, out_te,
                attrs: list[str], checkpoints=CHECKPOINTS, C: float = 1.0):
    """Returns {checkpoint: {attr: q (n_test, K)}}."""
    fs = FeatureSpace(world, recipient, variant, mode)
    Xtr, Xte = fs.matrix(h_tr, out_tr), fs.matrix(h_te, out_te)
    res = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for ck in checkpoints:
            keep = np.flatnonzero(fs.month <= ck)
            A, B = Xtr[:, keep], Xte[:, keep]
            res[ck] = {}
            for a in attrs:
                y = world.pop.values[a][h_tr]
                clf = LogisticRegression(C=C, max_iter=300)
                clf.fit(A, y)
                q = np.zeros((len(h_te), world.pop.n_values[a]))
                q[:, clf.classes_] = clf.predict_proba(B)
                res[ck][a] = q
    return res
