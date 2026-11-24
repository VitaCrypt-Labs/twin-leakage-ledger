# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""One command: python -m tll.reproduce  (or make reproduce).
Runs both modes with attackers A0 and A1, the decisive test, usefulness, and writes results/."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from . import test_claim as tc
from . import utility
from .attackers import a0_bayes, a1_classifier
from .ledger import run_ledger
from .sim import CHECKPOINTS, World

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
N_TEST, N_TRAIN, N_UTIL = 2000, 20000, 300
SEED_TEST, SEED_TRAIN, SEED_UTIL = 20261207, 20261208, 20261209
ATTACKERS = ("A0", "A1")
TOL_MEAN = {"A0": 1e-6, "A1": 2e-3}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def roundf(o, nd=10):
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: roundf(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [roundf(v, nd) for v in o]
    return o


def build_cells(world, mode, q_by_att, bounds, h, recipients, variant="base", attackers=ATTACKERS):
    """q_by_att[attacker][recipient][ck][attr] -> (n, K). bounds[recipient][ck]: ledger output."""
    cells, arrays = [], {}
    for att in attackers:
        for r in recipients:
            known = world.variants[r][variant]
            for a in world.sensitive:
                true_val = world.pop.values[a][h].astype(np.int64)
                for ck in CHECKPOINTS:
                    b = bounds[r][ck]
                    R, D, ceil, correct = tc.person_stats(q_by_att[att][r][ck][a], true_val, b["LA"][a],
                                                          b["PR_true"][a], b["pmax"][a])
                    meta = {"mode": mode, "attacker": att, "recipient": r, "attribute": a, "checkpoint": ck,
                            "trivial": a in known}
                    arrays[(att, r, a, ck)] = (R, D, ceil, correct)
                    cells.append((meta, R, D))
    m = len(cells) * 2
    out = []
    for k, (meta, R, D) in enumerate(cells):
        out.append(tc.make_cell(meta, "R", R, m, 2 * k))
        out.append(tc.make_cell(meta, "D", D, m, 2 * k + 1))
    return tc.evaluate_family(out), arrays


def extractor_matches_rules(world, r, main_out, ctrl_out) -> bool:
    """True when, for every slot, the claims the extractor finds in a control draft equal the rule claims."""
    for s in world.slots_of[r]:
        for o in np.unique(ctrl_out[:, s.idx]):
            parsed = world.parse(s, world.text(s, int(o), "control"), "main")
            if not np.all(main_out[ctrl_out[:, s.idx] == o, s.idx] == parsed):
                return False
    return True


def a0_to_q(res):
    return {ck: res[ck]["post"] for ck in res}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    write_expected = "--write-expected" in argv
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)
    world = World()
    pop = world.pop
    attrs = world.sensitive
    log(f"population: {pop.N} histories, {world.S} slots per person")

    h, U, main_out, ctrl_out = world.draw_personas(N_TEST, SEED_TEST)
    htr, _, mtr, ctr = world.draw_personas(N_TRAIN, SEED_TRAIN)
    log(f"test personas {N_TEST}, training personas {N_TRAIN}")

    led, ledc, a0m, a0c, a1m, a1c = {}, {}, {}, {}, {}, {}
    for r in world.recipients:
        led[r] = run_ledger(world, r, "base", h, main_out, attrs, CHECKPOINTS)
        # control mode: the ledger counts only what the extractor finds in each draft
        same = extractor_matches_rules(world, r, main_out, ctrl_out)
        ledc[r] = led[r] if same else run_ledger(world, r, "base", h, ctrl_out, attrs, CHECKPOINTS, mode="control")
        a0m[r] = a0_bayes.run(world, r, "base", h, main_out, "main", attrs)
        a0c[r] = a0_bayes.run(world, r, "base", h, ctrl_out, "control", attrs)
        log(f"{r}: ledger and A0 done (control ledger equals main ledger: {same})")
        a1m[r] = a1_classifier.fit_predict(world, r, "base", "main", htr, mtr, h, main_out, attrs)
        a1c[r] = a1_classifier.fit_predict(world, r, "base", "control", htr, ctr, h, ctrl_out, attrs)
        log(f"{r}: A1 done")

    def q_of(a0, a1):
        return {"A0": {r: a0_to_q(a0[r]) for r in world.recipients}, "A1": a1}

    cells_main, arr_main = build_cells(world, "main", q_of(a0m, a1m), led, h, world.recipients)
    cells_ctrl, arr_ctrl = build_cells(world, "control", q_of(a0c, a1c), ledc, h, world.recipients)
    v_main, v_ctrl = tc.verdict_main(cells_main), tc.verdict_control(cells_ctrl)
    log(f"main: {v_main} ({sum(c['refuted'] for c in cells_main)} of {len(cells_main)} cells refuted)")
    log(f"control: {v_ctrl} ({sum(c['refuted'] for c in cells_ctrl)} of {len(cells_ctrl)} cells refuted)")

    # A0 reads the same text with its own code: its history-level ratio must equal the ledger's L
    drift = max(float(np.max(np.abs(a0m[r][ck]["ratio_hist"] / led[r][ck]["L"] - 1)))
                for r in world.recipients for ck in CHECKPOINTS)
    log(f"A0 history-level ratio vs ledger L, max relative difference: {drift:.2e}")
    if drift > 1e-8:
        print("ERROR: A0 and the ledger disagree")
        return 1
    # robustness: recipient knows more
    led_more, a0_more = {}, {}
    for r in world.recipients:
        led_more[r] = run_ledger(world, r, "more", h, main_out, attrs, CHECKPOINTS)
        a0_more[r] = a0_bayes.run(world, r, "more", h, main_out, "main", attrs)
    cells_more, _ = build_cells(world, "main_more_variant", {"A0": {r: a0_to_q(a0_more[r]) for r in world.recipients}},
                                led_more, h, world.recipients, variant="more", attackers=("A0",))
    cells_mis, _ = build_cells(world, "main_ledger_base_attacker_more", {"A0": {r: a0_to_q(a0_more[r]) for r in world.recipients}},
                               led, h, world.recipients, variant="base", attackers=("A0",))
    log(f"knows-more, matching ledger: {sum(c['refuted'] for c in cells_more)} of {len(cells_more)} cells refuted; "
        f"mismatched ledger: {sum(c['refuted'] for c in cells_mis)} of {len(cells_mis)}")

    # gaps and odometer
    gaps, odo = {}, {}
    for r in world.recipients:
        nontrivial = [a for a in attrs if a not in world.variants[r]["base"]]
        best_R, acc, ceil_m, ratios = [], [], [], []
        for a in nontrivial:
            R_best = max(float(arr_main[(att, r, a, 24)][0].mean()) for att in ATTACKERS)
            best_R.append(R_best)
            att_best = max(ATTACKERS, key=lambda t: float(arr_main[(t, r, a, 24)][3].mean()))
            acc.append(float(arr_main[(att_best, r, a, 24)][3].mean()))
            ceil_m.append(float(arr_main[(att_best, r, a, 24)][2].mean()))
            ratios.append(led[r][24]["L"] / led[r][24]["LA"][a])
        gaps[r] = {"best_mean_R": float(np.mean(best_R)), "shortfall": 1.0 / float(np.mean(best_R)),
                   "mean_accuracy": float(np.mean(acc)), "mean_ceiling": float(np.mean(ceil_m)),
                   "median_L_over_LA": float(np.median(np.concatenate(ratios)))}
        lo = led[r][24]
        odo[r] = {"median_log_L": float(np.median(np.log(lo["L"]))),
                  "median_log_odometer": float(np.median(lo["log_odo"]))}
    log("gaps computed")

    util = utility.run(world, N_UTIL, SEED_UTIL)
    (RESULTS / "utility.md").write_text(utility.to_markdown(util, world, gaps), encoding="utf-8")
    log("utility done")

    write_outputs(world, cells_main, cells_ctrl, cells_more, cells_mis, v_main, v_ctrl, gaps, odo, led, ledc,
                  arr_main, arr_ctrl, util)
    log(f"outputs written to {RESULTS}")

    print(f"VERDICT main mode: {v_main}")
    print(f"VERDICT control mode: {v_ctrl}")
    status = 0
    if v_ctrl != "violation caught":
        print("ERROR: control mode did not produce a violation; experiment is uninformative")
        status = 1
    verdict = json.loads((RESULTS / "verdict.json").read_text())
    if write_expected:
        write_expected_file(verdict)
        log("expected.json written")
    else:
        status = max(status, compare_expected(verdict))
    log(f"total time {time.time() - t0:.0f} s")
    return status


def cell_key(c):
    return f"{c['mode']}|{c['attacker']}|{c['recipient']}|{c['attribute']}|{c['checkpoint']}|{c['statistic']}"


def write_expected_file(verdict: dict) -> None:
    exp = {"_spdx": "SPDX-License-Identifier: CC0-1.0", "_copyright": "Copyright 2026 Paul Burg",
           "verdict_main": verdict["main"]["verdict"], "verdict_control": verdict["control"]["verdict"],
           "tolerance_mean": TOL_MEAN, "cells": {}}
    for fam in ("main", "control"):
        for c in verdict[fam]["cells"]:
            exp["cells"][cell_key(c)] = {"mean": c["mean"], "refuted": c["refuted"], "p_holm": c["p_holm"]}
    (RESULTS / "expected.json").write_text(json.dumps(roundf(exp, 8), indent=1, sort_keys=True) + "\n", encoding="utf-8")


def compare_expected(verdict: dict) -> int:
    path = RESULTS / "expected.json"
    if not path.exists():
        print("ERROR: results/expected.json is missing")
        return 1
    exp = json.loads(path.read_text())
    bad = []
    if exp["verdict_main"] != verdict["main"]["verdict"]:
        bad.append("main verdict differs")
    if exp["verdict_control"] != verdict["control"]["verdict"]:
        bad.append("control verdict differs")
    seen = set()
    for fam in ("main", "control"):
        for c in verdict[fam]["cells"]:
            k = cell_key(c)
            seen.add(k)
            e = exp["cells"].get(k)
            if e is None:
                bad.append(f"extra cell {k}")
                continue
            tol = TOL_MEAN[c["attacker"]]
            if abs(c["mean"] - e["mean"]) > tol:
                bad.append(f"{k}: mean {c['mean']:.6f} vs {e['mean']:.6f}")
            near = 0.01 <= e["p_holm"] <= 0.25
            if c["refuted"] != e["refuted"] and not near:
                bad.append(f"{k}: decision differs")
    for k in exp["cells"]:
        if k not in seen:
            bad.append(f"missing cell {k}")
    if bad:
        print(f"ERROR: {len(bad)} differences from results/expected.json, first: {bad[:5]}")
        return 1
    print("expected.json: all cells within tolerance, same verdicts")
    return 0


def write_outputs(world, cells_main, cells_ctrl, cells_more, cells_mis, v_main, v_ctrl, gaps, odo, led, ledc,
                  arr_main, arr_ctrl, util):
    from . import report
    verdict = {
        "_spdx": "SPDX-License-Identifier: CC0-1.0", "_copyright": "Copyright 2026 Paul Burg",
        "design": {"n_test_personas": N_TEST, "n_training_personas": N_TRAIN, "n_utility_personas": N_UTIL,
                   "seeds": {"test": SEED_TEST, "train": SEED_TRAIN, "utility": SEED_UTIL},
                   "alpha": tc.ALPHA, "r_clip": tc.R_CLIP, "checkpoints": list(CHECKPOINTS),
                   "family_size_per_mode": len(cells_main), "attackers": list(ATTACKERS)},
        "main": {"verdict": v_main, "cells": cells_main},
        "control": {"verdict": v_ctrl, "cells": cells_ctrl},
        "knows_more_matching_ledger": {"note": "descriptive; own Holm family; ledger uses the published 'more' prior",
                                       "refuted_cells": sum(c["refuted"] for c in cells_more), "n_cells": len(cells_more),
                                       "cells": cells_more},
        "knows_more_mismatched_ledger": {"note": "descriptive, no verdict; ledger uses 'base' prior, attacker holds 'more' knowledge",
                                         "refuted_cells": sum(c["refuted"] for c in cells_mis), "n_cells": len(cells_mis),
                                         "cells": cells_mis},
        "gaps_24_months": gaps,
        "odometer_24_months": odo,
    }
    (RESULTS / "verdict.json").write_text(json.dumps(roundf(verdict), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    report.write_tables(world, cells_main, cells_ctrl, cells_more, cells_mis, v_main, v_ctrl, gaps, odo, led, ledc,
                        RESULTS / "main_table.md")
    report.write_figure(world, cells_main, cells_ctrl, led, RESULTS / "figure1.png")


if __name__ == "__main__":
    raise SystemExit(main())
