# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""World: population, rules, slot tables, exact likelihoods and sampling of outcomes.

Outcome codes per slot. Main mode: 0 = nothing sent, 1 + c = sent with claim code c.
Control mode: 1 + c + n_codes * m, where m is the bitmask of planted phrases in the draft."""
from __future__ import annotations

import numpy as np

from . import agent_sim, policy, renderer
from .extractor import Extractor
from .population import Population, load_json
from .vocabulary import Vocabulary

CHECKPOINTS = (1, 3, 6, 12, 24)


class World:
    def __init__(self):
        self.pop = Population()
        self.vocab = Vocabulary(duration=self.pop.duration)
        self.extractor = Extractor(self.vocab)
        rec = load_json("recipients.json")
        self.recipients = rec["order"]
        self.sensitive = rec["sensitive_attributes"]
        self.variants = {r: {v: tuple(d["known"]) for v, d in rec["recipients"][r]["variants"].items()}
                         for r in self.recipients}
        self.labels = {r: rec["recipients"][r]["label"] for r in self.recipients}
        self.slots = policy.build_slots(self.pop.horizon)
        self.S = len(self.slots)
        self.slots_of = {r: [s for s in self.slots if s.recipient == r] for r in self.recipients}
        self._sched: dict = {}
        self._planted: dict = {}
        self._text: dict = {}
        self._parse: dict = {}

    # ---- exact tables -------------------------------------------------
    def sched(self, s: policy.Slot, level: int):
        """(probability of sending, claim code) per history for a slot at a level."""
        key = (s.idx, level)
        v = self._sched.get(key)
        if v is None:
            if level == 2:
                v = (np.zeros(self.pop.N), np.zeros(self.pop.N, dtype=np.int32))
            else:
                p0 = self._sched.get((s.idx, 0)) if level == 1 else None
                p = p0[0] if p0 is not None else policy.p_send_vector(self.vocab, self.pop, s.kind, s.month)
                v = (p, policy.code_vector(self.vocab, self.pop, s.kind, s.month, level))
            self._sched[key] = v
        return v

    def n_codes(self, s: policy.Slot, level: int = 0) -> int:
        return 2 ** len(policy.level_claims(s.kind, level))

    def lik(self, s: policy.Slot, level: int, outcome: int, sub: np.ndarray | None = None) -> np.ndarray:
        """P(outcome | history) for the histories in `sub` (all if None)."""
        p, code = self.sched(s, level)
        if sub is not None:
            p, code = p[sub], code[sub]
        if outcome == 0:
            return 1.0 - p
        return p * (code == outcome - 1)

    def all_liks(self, s: policy.Slot, level: int, sub: np.ndarray) -> list[np.ndarray]:
        return [self.lik(s, level, o, sub) for o in range(policy.n_outcomes(s.kind, level))]

    def planted_p(self, j: int, month: int) -> np.ndarray:
        key = (j, month)
        v = self._planted.get(key)
        if v is None:
            v = agent_sim.planted_probability(self.vocab, self.pop, j, month)
            self._planted[key] = v
        return v

    def lik_control(self, s: policy.Slot, outcome: int, sub: np.ndarray | None = None) -> np.ndarray:
        """P(draft | history) for the control mode: schedule x claim code x planted phrases."""
        p, code = self.sched(s, 0)
        nc = self.n_codes(s)
        if outcome == 0:
            return 1.0 - (p if sub is None else p[sub])
        c, mask = (outcome - 1) % nc, (outcome - 1) // nc
        out = self.lik(s, 0, 1 + c, sub)
        for j in range(len(agent_sim.PLANTED)):
            pj = self.planted_p(j, s.month)
            pj = pj if sub is None else pj[sub]
            out = out * (pj if (mask >> j) & 1 else 1.0 - pj)
        return out

    # ---- texts and parsing ---------------------------------------------
    def text(self, s: policy.Slot, outcome: int, mode: str) -> str | None:
        """Text of the message for an outcome (None when nothing was sent)."""
        if outcome == 0:
            return None
        key = (s.idx, outcome, mode)
        t = self._text.get(key)
        if t is None:
            if mode == "main":
                claims = policy.claims_from_code(s.kind, 0, outcome - 1)
                t = renderer.render(self.vocab, s.kind, s.month, claims)
            else:
                nc = self.n_codes(s)
                claims = policy.claims_from_code(s.kind, 0, (outcome - 1) % nc)
                t = agent_sim.draft(self.vocab, s.kind, s.month, claims, (outcome - 1) // nc)
            self._text[key] = t
        return t

    def parse(self, s: policy.Slot, text: str | None, mode: str) -> int:
        """Read an outcome back from raw text with the extractor (and the planted-phrase list in control)."""
        if text is None:
            return 0
        key = (s.idx, text, mode)
        o = self._parse.get(key)
        if o is None:
            found = self.extractor.extract(text)
            names = policy.level_claims(s.kind, 0)
            if any(n not in found for n in names):
                raise ValueError(f"claim missing in text for slot {s}")
            code = sum(int(found[n]) << j for j, n in enumerate(names))
            o = 1 + code
            if mode == "control":
                o += self.n_codes(s) * agent_sim.planted_mask_in_text(text)
            self._parse[key] = o
        return o

    # ---- sampling -----------------------------------------------------
    def draw(self, h: np.ndarray, U: np.ndarray, V: np.ndarray | None = None):
        """Outcomes for personas h (history indices). U: (n, S) uniforms for sending,
        V: (n, S, 3) uniforms for planted phrases. Returns (main, control) outcome matrices."""
        n = len(h)
        main = np.zeros((n, self.S), dtype=np.int16)
        ctrl = np.zeros((n, self.S), dtype=np.int16) if V is not None else None
        for s in self.slots:
            p, code = self.sched(s, 0)
            sent = U[:, s.idx] < p[h]
            main[:, s.idx] = np.where(sent, 1 + code[h], 0)
            if V is not None:
                nc = self.n_codes(s)
                mask = np.zeros(n, dtype=np.int32)
                for j in range(len(agent_sim.PLANTED)):
                    mask |= (V[:, s.idx, j] < self.planted_p(j, s.month)[h]).astype(np.int32) << j
                ctrl[:, s.idx] = np.where(sent, 1 + code[h] + nc * mask, 0)
        return main, ctrl

    def draw_personas(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        h = self.pop.sample(n, rng)
        U = rng.random((n, self.S))
        V = rng.random((n, self.S, len(agent_sim.PLANTED)))
        main, ctrl = self.draw(h, U, V)
        return h, U, main, ctrl
