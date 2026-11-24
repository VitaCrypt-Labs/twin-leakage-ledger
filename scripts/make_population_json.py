# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""One-off generator of data/population.json. The tables are invented assumptions,
not epidemiology. Run: python scripts/make_population_json.py > data/population.json"""
import json


def fix(row):
    row = [round(v, 4) for v in row]
    row[0] = round(1.0 - sum(row[1:]), 4)
    return row


def onset_row(p_none, n=4, weights=None):
    rest = 1.0 - p_none
    w = weights or [1.0] * n
    s = sum(w)
    return fix([p_none] + [rest * v / s for v in w])


def follower(p_none_no_cause, p_none_cause, pre, post_w):
    """Row per onset index of the cause variable (0 = none) for a follower variable."""
    rows = [onset_row(p_none_no_cause)]
    for d in range(1, 5):
        n_pre = d - 1
        post = list(post_w[: 5 - d])
        rest = 1.0 - p_none_cause - pre * n_pre
        s = sum(post)
        row = [p_none_cause] + [pre] * n_pre + [rest * v / s for v in post]
        rows.append(fix(row))
    return rows


spec = {
    "_license": ["Copyright 2026 Paul Burg", "SPDX-License-" "Identifier: CC0-1.0"],
    "_note": "Invented assumptions for a synthetic population. Not epidemiology.",
    "horizon_months": 24,
    "age_groups": ["young", "middle", "senior"],
    "sexes": ["F", "M"],
    "onset_grid": [None, 2, 5, 11, 18],
    "pregnancy_grid": [None, 3, 9, 15],
    "pregnancy_duration_months": 9,
    "p_age": [0.30, 0.45, 0.25],
    "p_sex": [0.5, 0.5],
    "p_pregnancy_given_age_female": [
        onset_row(0.90, 3), onset_row(0.95, 3), onset_row(0.995, 3)],
    "p_diabetes_given_age": [onset_row(0.94), onset_row(0.86), onset_row(0.74)],
    "p_depression_given_sex_diabetes": {
        "F": {"no_diabetes": onset_row(0.82), "diabetes": onset_row(0.66)},
        "M": {"no_diabetes": onset_row(0.90), "diabetes": onset_row(0.78)},
    },
    "p_antidiabetic_given_diabetes": follower(0.95, 0.10, 0.01, [0.6, 0.25, 0.1, 0.05]),
    "p_ssri_given_depression": follower(0.95, 0.15, 0.01, [0.6, 0.25, 0.1, 0.05]),
    "p_hba1c_given_diabetes": follower(0.93, 0.20, 0.02, [0.5, 0.3, 0.12, 0.08]),
}
print(json.dumps(spec, indent=1))
