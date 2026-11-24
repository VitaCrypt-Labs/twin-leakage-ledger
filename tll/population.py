# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""Closed set of 24-month histories and the exact prior over it."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NEVER = 10_000  # sentinel month for "does not happen"
ONSET_VARS = ("diabetes", "depression", "antidiabetic", "ssri", "hba1c")
ATTRS = ("age", "sex", "pregnancy") + ONSET_VARS


def load_json(name: str) -> dict:
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


class Population:
    """All histories (age, sex, pregnancy start, five onset variables) with prior probabilities."""

    def __init__(self, spec: dict | None = None):
        spec = spec or load_json("population.json")
        self.spec = spec
        self.horizon = int(spec["horizon_months"])
        og, pg = spec["onset_grid"], spec["pregnancy_grid"]
        n_age, n_sex, n_preg, n_on = len(spec["age_groups"]), len(spec["sexes"]), len(pg), len(og)
        axes = [np.arange(n_age), np.arange(n_sex), np.arange(n_preg)] + [np.arange(n_on)] * 5
        flat = [g.ravel() for g in np.meshgrid(*axes, indexing="ij")]
        age, sex, preg, dia, dep, m1, m2, hb = flat
        valid = (sex == 0) | (preg == 0)  # only the first sex can be pregnant

        p_age = np.array(spec["p_age"])
        p_sex = np.array(spec["p_sex"])
        pp_f = np.array(spec["p_pregnancy_given_age_female"])
        pp = np.where(sex == 0, pp_f[age, preg], (preg == 0).astype(float))
        pd_ = np.array(spec["p_diabetes_given_age"])[age, dia]
        dep_tab = np.array([[spec["p_depression_given_sex_diabetes"][s][k]
                             for k in ("no_diabetes", "diabetes")] for s in spec["sexes"]])
        pdep = dep_tab[sex, (dia > 0).astype(int), dep]
        pm1 = np.array(spec["p_antidiabetic_given_diabetes"])[dia, m1]
        pm2 = np.array(spec["p_ssri_given_depression"])[dep, m2]
        phb = np.array(spec["p_hba1c_given_diabetes"])[dia, hb]
        prior = p_age[age] * p_sex[sex] * pp * pd_ * pdep * pm1 * pm2 * phb

        cols = {"age": age, "sex": sex, "pregnancy": preg, "diabetes": dia, "depression": dep,
                "antidiabetic": m1, "ssri": m2, "hba1c": hb}
        self.values = {k: v[valid].astype(np.int8) for k, v in cols.items()}
        self.prior = prior[valid]
        self.N = len(self.prior)
        if abs(self.prior.sum() - 1.0) > 1e-9 or (self.prior <= 0).any():
            raise ValueError("prior must be positive and sum to 1")
        self.n_values = {"age": n_age, "sex": n_sex, "pregnancy": n_preg, **{v: n_on for v in ONSET_VARS}}
        self.grid = {"pregnancy": pg, **{v: og for v in ONSET_VARS}}
        self.duration = int(spec["pregnancy_duration_months"])
        # onset month per history (NEVER when absent)
        self.start_month = {}
        for var, grid in self.grid.items():
            lut = np.array([NEVER if m is None else m for m in grid], dtype=np.int32)
            self.start_month[var] = lut[self.values[var]]
        self._sub_cache: dict = {}
        self._cum = np.cumsum(self.prior)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        idx = np.searchsorted(self._cum, rng.random(n) * self._cum[-1], side="right")
        return np.minimum(idx, self.N - 1)

    def record(self, i: int) -> dict:
        """Plain dict for one history: indices for age and sex, start months (or None) for the rest."""
        rec = {"age": int(self.values["age"][i]), "sex": int(self.values["sex"][i])}
        for var, grid in self.grid.items():
            rec[var] = grid[int(self.values[var][i])]
        return rec

    def known_key(self, known: tuple, i: int) -> tuple:
        return tuple(int(self.values[a][i]) for a in known)

    def subset(self, known: tuple, key: tuple) -> np.ndarray:
        """Indices of histories consistent with the recipient's prior knowledge."""
        ck = (known, key)
        sub = self._sub_cache.get(ck)
        if sub is None:
            mask = np.ones(self.N, dtype=bool)
            for a, v in zip(known, key):
                mask &= self.values[a] == v
            sub = np.flatnonzero(mask)
            self._sub_cache[ck] = sub
        return sub
