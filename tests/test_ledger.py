# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from tll.attackers import a0_bayes
from tll.ledger import Ledger, run_ledger


def test_bound_equals_hand_computed_example():
    # three histories, prior 0.5 / 0.3 / 0.2; attribute A is 0 for the first two, 1 for the third
    led = Ledger(np.array([0.5, 0.3, 0.2]), {"A": (np.array([0, 0, 1]), 2)})
    led.update(np.array([0.8, 0.4, 0.1]))
    led.update(np.array([0.5, 0.5, 1.0]))
    # P(y|x) = [0.40, 0.20, 0.10]; P_R(y) = 0.2 + 0.06 + 0.02 = 0.28
    b = led.bounds()
    assert abs(b.L - 0.40 / 0.28) < 1e-12  # 10/7
    # P(A=0|y) = 0.26/0.28, prior 0.8, so 65/56; P(A=1|y) = 0.02/0.28, prior 0.2, ratio 0.357
    assert abs(b.LA["A"] - 65 / 56) < 1e-12
    assert b.LA["A"] <= b.L
    # odometer: step 1 is 0.8/0.54, step 2 is 1.0/0.6; the exact bound is below their product
    assert abs(led.log_odometer - np.log((0.8 / 0.54) * (1.0 / 0.6))) < 1e-12
    assert b.L < np.exp(led.log_odometer)


def test_worst_case_covers_every_outcome():
    led = Ledger(np.array([0.5, 0.3, 0.2]), {"A": (np.array([0, 0, 1]), 2)})
    liks = [np.array([0.9, 0.9, 0.1]), np.array([0.1, 0.1, 0.9])]  # two outcomes of one slot
    worst = led.worst_case_LA(liks, ["A"])
    # outcome 2: P(A=1|y) = 0.18/(0.05+0.03+0.18) = 0.692, prior 0.2 -> ratio 3.46
    assert abs(worst - (0.18 / 0.26) / 0.2) < 1e-9


def _prior_marginals(world, recipient, h, attrs):
    """Prior marginal of each attribute under the recipient's knowledge, per person, built independently."""
    pop = world.pop
    known = world.variants[recipient]["base"]
    cache, out = {}, {a: np.zeros((len(h), pop.n_values[a])) for a in attrs}
    for i, hi in enumerate(h):
        key = pop.known_key(known, hi)
        if key not in cache:
            sub = pop.subset(known, key)
            w = pop.prior[sub] / pop.prior[sub].sum()
            cache[key] = {a: np.bincount(pop.values[a][sub].astype(np.int64), weights=w, minlength=pop.n_values[a])
                          for a in attrs}
        for a in attrs:
            out[a][i] = cache[key][a]
    return out


def test_a0_never_above_attribute_bound_and_equals_L_on_10000_histories(world):
    n, months = 10_000, 6
    h, U, main, ctrl = world.draw_personas(n, 777)
    attrs = world.sensitive
    cks = (1, 3, 6)
    for r in world.recipients:
        led = run_ledger(world, r, "base", h, main, attrs, cks, max_month=months)
        a0 = a0_bayes.run(world, r, "base", h, main, "main", attrs, cks, max_month=months)
        pm = _prior_marginals(world, r, h, attrs)
        for ck in cks:
            # most revealed fact = the whole history: A0's ratio equals L
            np.testing.assert_allclose(a0[ck]["ratio_hist"], led[ck]["L"], rtol=1e-9)
            for a in attrs:
                post = a0[ck]["post"][a]
                ratio = np.where(pm[a] > 0, post / np.where(pm[a] > 0, pm[a], 1), 0).max(axis=1)
                assert np.all(ratio <= led[ck]["LA"][a] * (1 + 1e-9))
                np.testing.assert_allclose(ratio, led[ck]["LA"][a], rtol=1e-9)
                assert np.all(led[ck]["LA"][a] <= led[ck]["L"] * (1 + 1e-9))


def test_control_ledger_counts_only_extractor_output(world):
    """The ledger of the control mode equals the main ledger because the extractor finds the rule claims,
    while A0 reading the same drafts learns more (planted phrases)."""
    h, U, main, ctrl = world.draw_personas(300, 5)
    attrs = world.sensitive
    cks = (3, 6)
    led_c = run_ledger(world, "family", "base", h, ctrl, attrs, cks, mode="control", max_month=6)
    led_m = run_ledger(world, "family", "base", h, main, attrs, cks, max_month=6)
    np.testing.assert_allclose(led_c[6]["L"], led_m[6]["L"], rtol=1e-12)
    a0c = a0_bayes.run(world, "family", "base", h, ctrl, "control", attrs, cks, max_month=6)
    assert np.any(a0c[6]["ratio_hist"] > led_c[6]["L"] * 1.5)
