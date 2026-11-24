# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
import inspect

import numpy as np

from tll import agent_sim, policy, renderer


def test_population_is_small_closed_set(world):
    assert world.pop.N <= 65_000
    assert abs(world.pop.prior.sum() - 1) < 1e-12


def test_renderer_sees_only_kind_month_claims():
    assert list(inspect.signature(renderer.render).parameters) == ["vocab", "kind", "month", "claims"]


def test_renderer_output_unchanged_when_rest_of_record_is_swapped(world):
    rng = np.random.default_rng(1)
    pop, vocab = world.pop, world.vocab
    checked = 0
    for _ in range(400):
        i = int(pop.sample(1, rng)[0])
        rec = pop.record(i)
        s = world.slots[int(rng.integers(world.S))]
        allowed = policy.level_claims(s.kind, 0)
        used_vars = {vocab.claims[c]["var"] for c in allowed}
        other = pop.record(int(pop.sample(1, rng)[0]))
        swapped = {k: (rec[k] if k in used_vars else other[k]) for k in rec}
        assert policy.message_for_record(vocab, rec, s.kind, s.month) == \
            policy.message_for_record(vocab, swapped, s.kind, s.month)
        if swapped != rec:
            checked += 1
    assert checked > 300


def test_vector_and_scalar_rules_agree(world):
    rng = np.random.default_rng(2)
    pop, vocab = world.pop, world.vocab
    for _ in range(150):
        i = int(pop.sample(1, rng)[0])
        rec = pop.record(i)
        s = world.slots[int(rng.integers(world.S))]
        p, code = world.sched(s, 0)
        assert abs(p[i] - policy.p_send_scalar(vocab, rec, s.kind, s.month)) < 1e-12
        assert int(code[i]) == policy.code_scalar(vocab, rec, s.kind, s.month, 0)
        _, code1 = world.sched(s, 1)
        assert int(code1[i]) == policy.code_scalar(vocab, rec, s.kind, s.month, 1)
        for cid in vocab.ids():
            assert bool(vocab.vector(pop, cid, s.month)[i]) == vocab.scalar(rec, cid, s.month)


def test_ledger_probabilities_match_simulated_frequencies(world):
    """Exact likelihood of every outcome vs frequencies from the scalar simulation of the rules."""
    rng = np.random.default_rng(3)
    pop, vocab = world.pop, world.vocab
    draws = 20_000
    hist = [int(x) for x in pop.sample(6, np.random.default_rng(11))]
    # add one history that is certain to carry medication, diabetes and pregnancy
    sel = np.flatnonzero((pop.values["antidiabetic"] == 1) & (pop.values["diabetes"] == 1) & (pop.values["pregnancy"] == 1))
    hist.append(int(sel[0]))
    for i in hist:
        rec = pop.record(i)
        for s in [world.slots[k] for k in rng.integers(0, world.S, 6)] + [world.slots[-1], world.slots[-5]]:
            for level in (0, 1):
                k = policy.n_outcomes(s.kind, level)
                freq = np.bincount([policy.sample_slot_scalar(vocab, rec, s.kind, s.month, level, rng)
                                    for _ in range(draws)], minlength=k) / draws
                exact = np.array([world.lik(s, level, o)[i] for o in range(k)])
                assert abs(exact.sum() - 1) < 1e-12
                tol = 5 * np.sqrt(exact * (1 - exact) / draws) + 1e-3
                assert np.all(np.abs(freq - exact) <= tol), (s, level, freq, exact)


def test_control_likelihood_normalises_and_marginalises(world):
    rng = np.random.default_rng(4)
    for _ in range(12):
        s = world.slots[int(rng.integers(world.S))]
        sub = np.sort(rng.choice(world.pop.N, 200, replace=False))
        n_out = 1 + world.n_codes(s) * 8
        liks = [world.lik_control(s, o, sub) for o in range(n_out)]
        np.testing.assert_allclose(sum(liks), 1.0, atol=1e-12)
        nc = world.n_codes(s)
        for c in range(nc):  # summing out the planted phrases gives the main-mode likelihood
            tot = sum(liks[1 + c + nc * m] for m in range(8))
            np.testing.assert_allclose(tot, world.lik(s, 0, 1 + c, sub), atol=1e-12)


def test_texts_parse_back_in_both_modes(world):
    for s in world.slots:
        for o in range(1, policy.n_outcomes(s.kind, 0)):
            assert world.parse(s, world.text(s, o, "main"), "main") == o
        for o in range(1, 1 + world.n_codes(s) * 8):
            assert world.parse(s, world.text(s, o, "control"), "control") == o
            # the extractor alone sees only the rule claims, not the planted phrases
            nc = world.n_codes(s)
            assert world.parse(s, world.text(s, o, "control"), "main") == 1 + (o - 1) % nc


def test_planted_phrases_are_invisible_to_extractor(world):
    text = agent_sim.draft(world.vocab, "family.check_in", 5, {"pregnant": False}, planted_mask=0b111)
    assert world.extractor.extract(text) == {"pregnant": False}
    assert agent_sim.planted_mask_in_text(text) == 0b111
