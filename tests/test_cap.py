# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
import inspect

import numpy as np

from tll import ledger, utility


def test_decision_function_takes_no_content():
    assert list(inspect.signature(ledger.decide_level).parameters) == ["ledger", "liks_by_level", "cap", "names"]


def test_cap_decision_unchanged_when_current_content_is_swapped(world):
    """Two people with the same released history (nothing sent in months 1 to 4) whose current content
    differs: one starts the antidiabetic at month 5, the other never. The decision at month 5 is the same."""
    pop = world.pop
    v = pop.values
    cand_a = np.flatnonzero((v["antidiabetic"] == 2) & (v["ssri"] == 0))  # starts at month 5
    cand_b = np.flatnonzero((v["antidiabetic"] == 0) & (v["ssri"] == 0))  # never
    ha, hb = int(cand_a[0]), int(cand_b[0])
    U = np.full(world.S, 0.5)  # 0.5 > 0.05: nothing sent while no medication, sent once medication starts
    for cap in (1.2, 2.0, 5.0):
        la, oa, _ = utility.capped_run(world, "pharmacy", ha, U, cap)
        lb, ob, _ = utility.capped_run(world, "pharmacy", hb, U, cap)
        months = [s.month for s in world.slots_of["pharmacy"]]
        upto = months.index(5)  # first slot of month 5
        assert oa[:upto] == ob[:upto] == [0] * upto  # same released history
        assert la[: upto + 1] == lb[: upto + 1]  # decisions up to and including month 5 agree
    # the content at month 5 does differ between the two people
    s5 = [s for s in world.slots_of["pharmacy"] if s.month == 5][0]
    assert world.sched(s5, 0)[1][ha] != world.sched(s5, 0)[1][hb]


def test_limit_is_never_exceeded(world):
    rng = np.random.default_rng(9)
    h = world.pop.sample(40, rng)
    U = rng.random((40, world.S))
    for r in world.recipients:
        for cap in (2.0, 5.0):
            for i in range(40):
                _, _, led = utility.capped_run(world, r, int(h[i]), U[i], cap)
                assert max(led.bounds().LA[a] for a in world.sensitive) <= cap + 1e-9
