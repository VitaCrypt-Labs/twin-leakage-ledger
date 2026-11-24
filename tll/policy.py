# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""Declared release rules. Plain code, no LLM. For every kind of message: when it is sent
(a probability that depends on the history), which claims it carries, and which claims
it carries at the coarse level used when the person has set a limit.

Levels: 0 = full rule, 1 = coarse (fewer claims, same schedule), 2 = off (nothing is sent).
The probability of every possible outcome of a slot is known exactly for every history."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import renderer
from .population import Population
from .vocabulary import Vocabulary

KIND_ORDER = ["doctor.summary", "doctor.visit_request", "pharmacy.refill",
              "insurer.claims_export", "family.check_in"]

RULES = {
    "doctor.summary": dict(
        recipient="doctor", schedule={"type": "const", "p": 0.85},
        claims=["takes_antidiabetic", "takes_ssri", "hba1c_elevated"],
        coarse=["hba1c_elevated"],
        needs=["takes_antidiabetic", "takes_ssri", "hba1c_elevated", "diagnosis_depression", "pregnant"]),
    "doctor.visit_request": dict(
        recipient="doctor",
        schedule={"type": "any_claim", "claims": ["hba1c_elevated", "pregnant"], "p_yes": 0.5, "p_no": 0.04},
        claims=[], coarse=[], needs=[]),
    "pharmacy.refill": dict(
        recipient="pharmacy",
        schedule={"type": "any_claim", "claims": ["takes_antidiabetic", "takes_ssri"], "p_yes": 0.9, "p_no": 0.05},
        claims=["takes_antidiabetic", "takes_ssri"], coarse=[],
        needs=["takes_antidiabetic", "takes_ssri"]),
    "insurer.claims_export": dict(
        recipient="insurer", schedule={"type": "every", "months": 3},
        claims=["takes_antidiabetic", "pregnant"], coarse=[],
        needs=["takes_antidiabetic", "pregnant", "hba1c_elevated", "diagnosis_diabetes"]),
    "family.check_in": dict(
        recipient="family",
        schedule={"type": "any_claim", "claims": ["diagnosis_depression"], "p_yes": 0.9, "p_no": 0.55},
        claims=["pregnant"], coarse=[], needs=["pregnant", "diagnosis_depression"]),
}


@dataclass(frozen=True)
class Slot:
    idx: int
    month: int
    kind: str

    @property
    def recipient(self) -> str:
        return RULES[self.kind]["recipient"]


def build_slots(horizon: int) -> list[Slot]:
    slots = []
    for m in range(1, horizon + 1):
        for k in KIND_ORDER:
            slots.append(Slot(len(slots), m, k))
    return slots


def level_claims(kind: str, level: int) -> list[str]:
    return list(RULES[kind]["claims"]) if level == 0 else list(RULES[kind]["coarse"]) if level == 1 else []


def n_outcomes(kind: str, level: int) -> int:
    """Outcome 0 = nothing sent; outcome 1 + c = sent with claim values coded by bits c."""
    return 1 if level == 2 else 1 + 2 ** len(level_claims(kind, level))


def p_send_vector(vocab: Vocabulary, pop: Population, kind: str, month: int) -> np.ndarray:
    sch = RULES[kind]["schedule"]
    if sch["type"] == "const":
        return np.full(pop.N, float(sch["p"]))
    if sch["type"] == "every":
        return np.full(pop.N, 1.0 if month % sch["months"] == 0 else 0.0)
    any_c = np.zeros(pop.N, dtype=bool)
    for cid in sch["claims"]:
        any_c |= vocab.vector(pop, cid, month)
    return np.where(any_c, float(sch["p_yes"]), float(sch["p_no"]))


def p_send_scalar(vocab: Vocabulary, record: dict, kind: str, month: int) -> float:
    sch = RULES[kind]["schedule"]
    if sch["type"] == "const":
        return float(sch["p"])
    if sch["type"] == "every":
        return 1.0 if month % sch["months"] == 0 else 0.0
    hit = any(vocab.scalar(record, c, month) for c in sch["claims"])
    return float(sch["p_yes"] if hit else sch["p_no"])


def code_vector(vocab: Vocabulary, pop: Population, kind: str, month: int, level: int) -> np.ndarray:
    code = np.zeros(pop.N, dtype=np.int32)
    for j, cid in enumerate(level_claims(kind, level)):
        code |= vocab.vector(pop, cid, month).astype(np.int32) << j
    return code


def code_scalar(vocab: Vocabulary, record: dict, kind: str, month: int, level: int) -> int:
    code = 0
    for j, cid in enumerate(level_claims(kind, level)):
        code |= int(vocab.scalar(record, cid, month)) << j
    return code


def claims_from_code(kind: str, level: int, code: int) -> dict[str, bool]:
    return {cid: bool((code >> j) & 1) for j, cid in enumerate(level_claims(kind, level))}


def allowed_claims(vocab: Vocabulary, record: dict, kind: str, month: int, level: int) -> dict[str, bool]:
    """The only data the renderer is allowed to see: allowed claims read from the record."""
    return {cid: vocab.scalar(record, cid, month) for cid in level_claims(kind, level)}


def message_for_record(vocab: Vocabulary, record: dict, kind: str, month: int, level: int = 0) -> str:
    """Text of the message if it is sent. Built only from the allowed claims."""
    return renderer.render(vocab, kind, month, allowed_claims(vocab, record, kind, month, level))


def sample_slot_scalar(vocab: Vocabulary, record: dict, kind: str, month: int, level: int,
                       rng: np.random.Generator) -> int:
    """Simulate the rule once for one record (independent of the vectorised likelihood)."""
    if level == 2:
        return 0
    if rng.random() < p_send_scalar(vocab, record, kind, month):
        return 1 + code_scalar(vocab, record, kind, month, level)
    return 0
