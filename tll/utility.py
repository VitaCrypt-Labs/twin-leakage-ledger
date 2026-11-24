# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""Cost in usefulness. Every kind of message has a list of facts the recipient needs.

1. Price of the rules: needed true facts delivered by the rules vs by the simulated free agent.
2. Price of a limit: share of messages coarsened or not sent, and needed facts lost, for limits 2x, 5x, 10x.
3. Price of the coarseness of the bound: gap between the bound and the best attacker (computed in reproduce)."""
from __future__ import annotations

import numpy as np

from . import agent_sim, policy
from .ledger import Ledger, decide_level
from .sim import World

CAPS = (2.0, 5.0, 10.0)


def _needs_truth(world: World, h: np.ndarray, s: policy.Slot) -> dict[str, np.ndarray]:
    return {c: world.vocab.vector(world.pop, c, s.month)[h] for c in policy.RULES[s.kind]["needs"]}


def _delivered(kind: str, level: int, code: np.ndarray, truth: dict) -> np.ndarray:
    """Needed facts that are true and stated, per person, for message codes at a level."""
    need = set(policy.RULES[kind]["needs"])
    tot = np.zeros(len(code), dtype=np.int64)
    for j, cid in enumerate(policy.level_claims(kind, level)):
        if cid in need:
            tot += ((code >> j) & 1).astype(bool) & truth[cid]
    return tot


def capped_run(world: World, recipient: str, hist: int, U_row: np.ndarray, cap: float, variant: str = "base"):
    """One person, one recipient, with a limit. Decisions use only the ledger state (released history)
    and the rules. Returns per slot of the recipient: levels, outcomes, and the final ledger."""
    pop = world.pop
    known = world.variants[recipient][variant]
    sub = pop.subset(known, pop.known_key(known, hist))
    p0 = pop.prior[sub] / pop.prior[sub].sum()
    led = Ledger(p0, {x: (pop.values[x][sub].astype(np.int64), pop.n_values[x]) for x in world.sensitive})
    levels, outcomes = [], []
    for s in world.slots_of[recipient]:
        liks = {0: world.all_liks(s, 0, sub), 1: world.all_liks(s, 1, sub)}
        level = decide_level(led, liks, cap, world.sensitive)  # no access to the current content
        sent = U_row[s.idx] < world.sched(s, 0)[0][hist]
        outcome = 0 if (level == 2 or not sent) else 1 + int(world.sched(s, level)[1][hist])
        if level < 2:
            led.update(world.lik(s, level, outcome, sub))
        levels.append(level)
        outcomes.append(outcome)
    return levels, outcomes, led


def run_recipient(world: World, r: str, n: int, seed: int, caps=CAPS) -> dict:
    """Usefulness numbers for one recipient on n personas (same personas and uniforms for every recipient)."""
    rng = np.random.default_rng(seed)
    h = world.pop.sample(n, rng)
    U = rng.random((n, world.S))
    rng_agent = np.random.default_rng(seed + 1 + world.recipients.index(r))
    pop = world.pop
    acc = {"needed": 0, "main": 0, "free": 0, "sent": 0,
           **{f"cap{c}": {"delivered": 0, "coarse": 0, "off": 0, "max_LA": 1.0} for c in caps}}
    # uncapped numbers (vectorised over persons)
    for s in world.slots_of[r]:
        truth = _needs_truth(world, h, s)
        p, code = world.sched(s, 0)
        sent = U[:, s.idx] < p[h]
        acc["needed"] += int(sum(v[sent].sum() for v in truth.values()))
        acc["sent"] += int(sent.sum())
        acc["main"] += int(_delivered(s.kind, 0, code[h], truth)[sent].sum())
        for i in np.flatnonzero(sent):
            needed_vals = {c: bool(truth[c][i]) for c in truth}
            if not needed_vals:
                continue
            text = agent_sim.needs_draft(world.vocab, s.kind, s.month, needed_vals, rng_agent)
            found = world.extractor.extract(text)
            acc["free"] += sum(1 for c, v in found.items() if v and needed_vals.get(c, False))
    # capped runs, person by person (the ledger state depends on what was released)
    for c in caps:
        a = acc[f"cap{c}"]
        for i in range(n):
            levels, outcomes, led = capped_run(world, r, int(h[i]), U[i], c)
            for s, level, outcome in zip(world.slots_of[r], levels, outcomes):
                sent = U[i, s.idx] < world.sched(s, 0)[0][h[i]]
                if sent and level == 1:
                    a["coarse"] += 1
                if sent and level == 2:
                    a["off"] += 1
                if outcome and policy.RULES[s.kind]["needs"]:
                    truth = {cid: world.vocab.vector(pop, cid, s.month)[h[i:i + 1]]
                             for cid in policy.RULES[s.kind]["needs"]}
                    a["delivered"] += int(_delivered(s.kind, level, np.array([outcome - 1]), truth)[0])
            a["max_LA"] = max(a["max_LA"], max(led.bounds().LA[x] for x in world.sensitive))
    return acc


def to_markdown(res: dict, world: World, gaps: dict, caps=CAPS) -> str:
    L = ["<!--", "SPDX-FileCopyrightText: 2026 Paul Burg", "SPDX-License-Identifier: CC0-1.0", "-->", "",
         "# Cost in usefulness (synthetic data, base prior)", "",
         "## 1. Price of the rules", "",
         "Needed true facts stated in messages that are sent, as a share of all needed true facts at those messages (rules vs simulated free agent with recall 0.95). The schedule is the same in both.", "",
         "| Recipient | Needed facts | Rules | Free agent | Rules / free agent |", "|---|---|---|---|---|"]
    for r in world.recipients:
        a = res[r]
        need = max(a["needed"], 1)
        fm, ff = a["main"] / need, a["free"] / need
        L.append(f"| {world.labels[r]} | {a['needed']} | {fm:.3f} | {ff:.3f} | {fm / ff if ff else float('nan'):.3f} |")
    L += ["", "## 2. Price of a limit", "",
          "A limit caps the per-attribute bound for the three sensitive attributes. The ledger coarsens or drops a message "
          "before sending when the worst case over its possible contents would pass the limit. "
          "Shares are over messages the rules would send. Lost facts are relative to the rules without a limit.", "",
          "| Recipient | Limit | Coarsened | Not sent | Needed facts lost | Max bound reached |", "|---|---|---|---|---|---|"]
    for r in world.recipients:
        a = res[r]
        for c in caps:
            x = a[f"cap{c}"]
            sent = max(a["sent"], 1)
            lost = 1 - x["delivered"] / a["main"] if a["main"] else float("nan")
            L.append(f"| {world.labels[r]} | {c:g}x | {x['coarse'] / sent:.3f} | {x['off'] / sent:.3f} | "
                     f"{lost:.3f} | {x['max_LA']:.3f} |")
    L += ["", "## 3. Price of the coarseness of the bound", "",
          "At 24 months, base prior. Bound L_A against what the best attacker achieved, and per-attribute against general bound.", "",
          "| Recipient | Best attacker mean R | Bound shortfall (1 / mean R) | Mean accuracy | Mean accuracy ceiling | Median L / L_A |",
          "|---|---|---|---|---|---|"]
    for r in world.recipients:
        g = gaps[r]
        L.append(f"| {world.labels[r]} | {g['best_mean_R']:.3f} | {g['shortfall']:.2f} | "
                 f"{g['mean_accuracy']:.3f} | {g['mean_ceiling']:.3f} | {g['median_L_over_LA']:.2f} |")
    L.append("")
    return "\n".join(L)
