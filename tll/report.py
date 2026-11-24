# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""Tables and the figure written to results/."""
from __future__ import annotations

import numpy as np

from .sim import CHECKPOINTS

HEAD = ["<!--", "SPDX-FileCopyrightText: 2026 Paul Burg", "SPDX-License-Identifier: CC0-1.0", "-->", ""]


def _find(cells, **kw):
    for c in cells:
        if all(c[k] == v for k, v in kw.items()):
            return c
    raise KeyError(kw)


def _mode_table(world, cells, led, title):
    L = [f"## {title}", "",
         "At 24 months. Mean R and mean D are per-persona means (R must stay at or below 1, D at or below 0); "
         "the bracket is the distribution-free lower bound at 95% before Holm. Trivial: the recipient already knows the attribute.", "",
         "| Recipient | Attribute | 95th pct L | Median L_A | 95th pct L_A | A0 mean R | A1 mean R | A0 mean D | A1 mean D | Refuted cells (all checkpoints) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in world.recipients:
        for a in world.sensitive:
            row = [world.labels[r], a + (" (known)" if a in world.variants[r]["base"] else "")]
            row.append(f"{np.percentile(led[r][24]['L'], 95):.2f}")
            row.append(f"{np.median(led[r][24]['LA'][a]):.3f}")
            row.append(f"{np.percentile(led[r][24]['LA'][a], 95):.2f}")
            for stat in ("R", "D"):
                for att in ("A0", "A1"):
                    c = _find(cells, attacker=att, recipient=r, attribute=a, checkpoint=24, statistic=stat)
                    row.append(f"{c['mean']:.3f} [{c['lcb_95']:.3f}]")
            row.append(str(sum(c["refuted"] for c in cells if c["recipient"] == r and c["attribute"] == a)))
            L.append("| " + " | ".join(row) + " |")
    return L


def write_tables(world, cells_main, cells_ctrl, cells_more, cells_mis, v_main, v_ctrl, gaps, odo, led, ledc, path):
    L = HEAD + ["# Results (synthetic data, no real persons)", "",
                f"- Main mode verdict: **{v_main}** ({sum(c['refuted'] for c in cells_main)} of {len(cells_main)} cells refuted).",
                f"- Control mode verdict: **{v_ctrl}** ({sum(c['refuted'] for c in cells_ctrl)} of {len(cells_ctrl)} cells refuted).",
                "- Rule, cells, clip and seeds: `PREREGISTRATION.md`. Per-cell numbers: `results/verdict.json`.", ""]
    L += _mode_table(world, cells_main, led, "Main mode: rules and renderer, ledger counts the rules")
    L += [""] + _mode_table(world, cells_ctrl, ledc, "Control mode: free drafts with planted phrases, ledger counts only extractor output")
    L += ["", "## Refuted cells by statistic", "", "| Mode | Attacker | Statistic | Refuted | Cells |", "|---|---|---|---|---|"]
    for name, cells in (("main", cells_main), ("control", cells_ctrl)):
        for att in ("A0", "A1"):
            for stat in ("R", "D"):
                sel = [c for c in cells if c["attacker"] == att and c["statistic"] == stat]
                L.append(f"| {name} | {att} | {stat} | {sum(c['refuted'] for c in sel)} | {len(sel)} |")
    L += ["", "## Sensitivity", "",
          "Minimal detectable excess (width of the lower bound at the strictest Holm step): "
          f"R median {np.median([c['min_detectable_excess'] for c in cells_main if c['statistic'] == 'R']):.3f}, "
          f"D median {np.median([c['min_detectable_excess'] for c in cells_main if c['statistic'] == 'D']):.3f}. "
          "Per-cell values are in `results/verdict.json`.", "",
          "## Gaps at 24 months (main mode)", "",
          "| Recipient | Best attacker mean R | Shortfall 1 / mean R | Mean accuracy | Mean accuracy ceiling | Median L / L_A | Median log L | Median log odometer |",
          "|---|---|---|---|---|---|---|---|"]
    for r in world.recipients:
        g, o = gaps[r], odo[r]
        L.append(f"| {world.labels[r]} | {g['best_mean_R']:.3f} | {g['shortfall']:.2f} | {g['mean_accuracy']:.3f} | "
                 f"{g['mean_ceiling']:.3f} | {g['median_L_over_LA']:.2f} | {o['median_log_L']:.3f} | {o['median_log_odometer']:.3f} |")
    L += ["", "## Recipient knows more (A0 only, descriptive)", "",
          f"- Ledger configured with the published `more` prior: {sum(c['refuted'] for c in cells_more)} of {len(cells_more)} cells refuted.",
          f"- Ledger configured with `base` while A0 holds the `more` knowledge: {sum(c['refuted'] for c in cells_mis)} of {len(cells_mis)} cells refuted (no verdict).", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def write_figure(world, cells_main, cells_ctrl, led, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, grey = "#0072B2", "#D55E00", "#555555"
    combos = [(r, a) for r in world.recipients for a in world.sensitive if a not in world.variants[r]["base"]]
    xs = np.arange(len(combos))
    names = [f"{world.labels[r]}\n{a}" for r, a in combos]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    ax = axes[0]
    for off, key, col, mk, lab in ((-0.15, "L", grey, "D", "general bound L"), (0.15, "LA", blue, "o", "per-attribute bound L_A")):
        med, p95 = [], []
        for r, a in combos:
            v = led[r][24]["L"] if key == "L" else led[r][24]["LA"][a]
            med.append(np.median(v))
            p95.append(np.percentile(v, 95))
        ax.vlines(xs + off, med, p95, color=col, lw=1.5)
        ax.scatter(xs + off, med, marker=mk, facecolors="white", edgecolors=col, zorder=3)
        ax.scatter(xs + off, p95, marker=mk, color=col, zorder=3, label=lab + " (filled: 95th percentile, open: median)")
    ax.set_yscale("log")
    ax.set_xticks(xs, names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("bound after 24 months (x prior knowledge)")
    ax.set_title("Bounds the ledger reports")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    for off, cells, col, lab in ((-0.15, cells_main, blue, "main mode"), (0.15, cells_ctrl, orange, "control mode")):
        vals = [max(_find(cells, attacker=t, recipient=r, attribute=a, checkpoint=24, statistic="R")["mean"]
                    for t in ("A0", "A1")) for r, a in combos]
        ax.scatter(xs + off, vals, color=col, label=lab, zorder=3)
    ax.axhline(1.0, color=grey, ls="--", lw=1)
    ax.text(len(combos) - 0.5, 1.05, "bound: mean R = 1", ha="right", color=grey, fontsize=8)
    ax.set_xticks(xs, names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("best attacker, mean R (24 months)")
    ax.set_title("Attackers against the bound")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Twin Leakage Ledger, synthetic data", fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)
