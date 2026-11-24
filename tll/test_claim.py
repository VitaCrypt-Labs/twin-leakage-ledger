# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""Decisive test. Per cell: mean of the per-persona statistic, distribution-free lower bound
(empirical Bernstein for R, Hoeffding for D), p-value, Holm adjustment, minimal detectable excess.

Claims under test, per cell:  mean(R_i) <= 1  and  mean(D_i) <= 0.
A cell is refuted when its Holm-adjusted p-value is at most alpha."""
from __future__ import annotations

import math

import numpy as np

ALPHA = 0.05
R_CLIP = 10.0
D_RANGE = (-1.0, 1.0)


def person_stats(q: np.ndarray, true_val: np.ndarray, LA: np.ndarray, PR_true: np.ndarray, pmax: np.ndarray):
    """R_i (clipped), D_i, accuracy ceiling and correctness from a distribution q over the values of one attribute."""
    n = len(true_val)
    q_true = q[np.arange(n), true_val]
    R = np.minimum(R_CLIP, q_true / (PR_true * LA))
    guess = np.argmax(q, axis=1)  # ties go to the lowest value index
    correct = (guess == true_val).astype(float)
    ceil = np.minimum(1.0, LA * pmax)
    return R, correct - ceil, ceil, correct


def eb_width(var: float, n: int, b: float, delta: float) -> float:
    u = math.log(2.0 / delta)
    return math.sqrt(2.0 * var * u / n) + 7.0 * b * u / (3.0 * (n - 1))


def eb_pvalue(x: np.ndarray, b: float, thr: float) -> float:
    """Smallest delta at which the empirical Bernstein lower bound of the mean exceeds thr."""
    n = len(x)
    gap = float(x.mean()) - thr
    if gap <= 0:
        return 1.0
    var = float(x.var(ddof=1))
    a = 7.0 * b / (3.0 * (n - 1))
    c = math.sqrt(2.0 * var / n)
    s = (-c + math.sqrt(c * c + 4.0 * a * gap)) / (2.0 * a)
    return min(1.0, 2.0 * math.exp(-s * s))


def hoeffding_width(n: int, rng_width: float, delta: float) -> float:
    return rng_width * math.sqrt(math.log(1.0 / delta) / (2.0 * n))


def hoeffding_pvalue(x: np.ndarray, rng_width: float, thr: float) -> float:
    n = len(x)
    gap = float(x.mean()) - thr
    if gap <= 0:
        return 1.0
    return min(1.0, math.exp(-2.0 * n * gap * gap / (rng_width ** 2)))


def bootstrap_lcb(x: np.ndarray, seed: int, reps: int = 400, level: float = ALPHA) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(reps, len(x)))
    return float(np.quantile(x[idx].mean(axis=1), level))


def holm(pvals: list[float], alpha: float = ALPHA):
    """Holm step-down. Returns (adjusted p-values, reject flags)."""
    m = len(pvals)
    order = np.argsort(pvals, kind="stable")
    adj = np.zeros(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj.tolist(), (adj <= alpha).tolist()


def make_cell(meta: dict, stat: str, x: np.ndarray, m_family: int, seed: int) -> dict:
    n = len(x)
    if stat == "R":
        thr, p = 1.0, eb_pvalue(x, R_CLIP, 1.0)
        var = float(x.var(ddof=1))
        lcb95 = float(x.mean()) - eb_width(var, n, R_CLIP, ALPHA)
        mde = eb_width(var, n, R_CLIP, ALPHA / m_family)
    else:
        w = D_RANGE[1] - D_RANGE[0]
        thr, p = 0.0, hoeffding_pvalue(x, w, 0.0)
        lcb95 = float(x.mean()) - hoeffding_width(n, w, ALPHA)
        mde = hoeffding_width(n, w, ALPHA / m_family)
    return {**meta, "statistic": stat, "n": n, "threshold": thr, "mean": float(x.mean()),
            "lcb_95": lcb95, "bootstrap_lcb_95": bootstrap_lcb(x, seed), "p": p, "min_detectable_excess": mde}


def evaluate_family(cells: list[dict], alpha: float = ALPHA) -> list[dict]:
    adj, rej = holm([c["p"] for c in cells], alpha)
    for c, a, r in zip(cells, adj, rej):
        c["p_holm"] = a
        c["refuted"] = bool(r)
    return cells


def verdict_main(cells: list[dict]) -> str:
    return "bound refuted" if any(c["refuted"] for c in cells) else "bound held"


def verdict_control(cells: list[dict]) -> str:
    return "violation caught" if any(c["refuted"] for c in cells) else "uninformative"
