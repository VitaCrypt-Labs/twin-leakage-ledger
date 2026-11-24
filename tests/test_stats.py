# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np

from tll import test_claim as tc


def test_holm_known_example():
    adj, rej = tc.holm([0.01, 0.04, 0.03, 0.005], alpha=0.05)
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.06 (monotone)
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])
    assert rej == [True, False, False, True]


def test_pvalues_invert_the_bounds():
    rng = np.random.default_rng(0)
    x = np.minimum(10, rng.exponential(1.3, 800))
    p = tc.eb_pvalue(x, 10.0, 1.0)
    lcb = x.mean() - tc.eb_width(x.var(ddof=1), len(x), 10.0, p)
    assert abs(lcb - 1.0) < 1e-9
    d = rng.choice([-1.0, 0.0, 1.0], 800, p=[0.3, 0.3, 0.4])
    p = tc.hoeffding_pvalue(d, 2.0, 0.0)
    assert abs(d.mean() - tc.hoeffding_width(len(d), 2.0, p)) < 1e-9
    assert tc.eb_pvalue(np.zeros(10), 10.0, 1.0) == 1.0


def test_false_refutation_rate_below_alpha_at_the_threshold():
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(400):
        x = np.where(rng.random(500) < 0.1, 10.0, 0.0)  # mean exactly 1
        hits += tc.eb_pvalue(x, 10.0, 1.0) <= 0.05
    assert hits / 400 <= 0.05


def test_detects_a_real_excess():
    rng = np.random.default_rng(2)
    x = np.minimum(10, rng.exponential(2.0, 2000))  # mean about 2, threshold 1
    assert tc.eb_pvalue(x, 10.0, 1.0) < 1e-6


def test_person_stats_uses_lowest_index_on_ties():
    q = np.array([[0.5, 0.5], [0.2, 0.8]])
    R, D, ceil, correct = tc.person_stats(q, np.array([0, 1]), np.array([1.0, 2.0]), np.array([0.5, 0.5]), np.array([0.5, 0.5]))
    assert correct.tolist() == [1.0, 1.0]
    assert math.isclose(R[1], 0.8 / (0.5 * 2.0))
    assert math.isclose(D[1], 1.0 - 1.0)
