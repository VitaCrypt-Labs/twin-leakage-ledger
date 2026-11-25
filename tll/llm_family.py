# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""LLM attackers as their own Holm family (preregistration v2). Reads cache/llm/*.jsonl, averages the
three prompt runs per persona, and tests with the same rule as the main family. Without a cache nothing
happens and `make reproduce` is exactly the CPU-only run of v0.1.0."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import llm_cache
from . import test_claim as tc
from .attackers import a_llm
from .ledger import run_ledger
from .sim import World


def load_family(world: World, cache_dir: Path):
    """Returns (spec, q, fallbacks, incomplete, degraded) or None when there is no cache.
    q[model][recipient][checkpoint][attribute] -> (n, K) averaged over the prompt runs."""
    files = llm_cache.list_model_files(cache_dir)
    if not files:
        return None
    headers, recs = {}, {}
    for p in files:
        h, r = llm_cache.load_model(p)
        headers[h["model_key"]], recs[h["model_key"]] = h, r
    sets = {h["set"] for h in headers.values()}
    if len(sets) != 1:
        raise ValueError(f"cache files hold different sets: {sorted(sets)}")
    spec = a_llm.resolve_set(world, sets.pop())
    hist, _ = a_llm.personas(world, spec["n"])
    expected = {(pid, i, r, ck) for pid in a_llm.PROMPT_SEEDS for i in range(spec["n"])
                for r in spec["recipients"] for ck in spec["checkpoints"]}
    q, fallbacks, incomplete, degraded = {}, {}, {}, {}
    for m, rs in recs.items():
        rates = llm_cache.fallback_rates(next(p for p in files if p.stem == m))
        high = {a: r for a, r in rates.items() if r > llm_cache.FALLBACK_THRESHOLD}
        if high:
            degraded[m] = high
            continue
        if headers[m].get("format_version") != llm_cache.FORMAT_VERSION:
            degraded[m] = {"format_version": headers[m].get("format_version")}
            continue
        missing = expected - set(rs)
        if missing:
            incomplete[m] = len(missing)
            continue
        qm = {r: {ck: {a: np.zeros((spec["n"], world.pop.n_values[a])) for a in world.sensitive}
                  for ck in spec["checkpoints"]} for r in spec["recipients"]}
        fb = {a: 0 for a in world.sensitive}
        for (pid, i, r, ck), rec in rs.items():
            if (pid, i, r, ck) not in expected:
                continue
            if rec["history"] != int(hist[i]):
                raise ValueError(f"{m}: persona {i} has another history than the planned persona set")
            for a in world.sensitive:
                qm[r][ck][a][i] += np.array(rec["parsed"][a]) / len(a_llm.PROMPT_SEEDS)
                fb[a] += a in rec["fallback"]
        q[m], fallbacks[m] = qm, {"answers": len(expected), **fb}
    return spec, q, fallbacks, incomplete, degraded


def evaluate(world: World, cache_dir: Path):
    from .reproduce import build_cells
    loaded = load_family(world, cache_dir)
    if loaded is None:
        return None
    spec, q, fallbacks, incomplete, degraded = loaded
    out = {"status": "ok", "set": spec, "incomplete_models": incomplete, "degraded_models": degraded,
           "fallbacks": fallbacks}
    if degraded and not q:
        out["status"] = "refused"
        return out
    if not q:
        out["status"] = "incomplete"
        return out
    hist, main = a_llm.personas(world, spec["n"])
    attrs = world.sensitive
    led = {r: run_ledger(world, r, "base", hist, main, attrs, spec["checkpoints"]) for r in spec["recipients"]}
    cells, _ = build_cells(world, "main_llm", q, led, hist, spec["recipients"], attackers=tuple(q),
                           checkpoints=tuple(spec["checkpoints"]))
    out["family_size"] = len(cells)
    out["refuted_cells"] = sum(c["refuted"] for c in cells)
    out["verdict"] = tc.verdict_main(cells)
    out["cells"] = cells
    return out


def write_outputs(world: World, res: dict, results_dir: Path) -> None:
    from .reproduce import roundf
    results_dir.mkdir(exist_ok=True)
    body = {"_license": llm_cache.LICENSE_TAG, **res}
    (results_dir / "llm_verdict.json").write_text(json.dumps(roundf(body), indent=1, sort_keys=True) + "\n",
                                                  encoding="utf-8")
    L = ["<!--", "SPDX-FileCopyrightText: 2026 Paul Burg", "SPDX-License-Identifier: CC0-1.0", "-->", "",
         "# LLM attackers (own Holm family, see PREREGISTRATION.md v2)", ""]
    if res["status"] == "refused":
        L.append(f"Cache refused, no verdict: {refusal_message(res)}")
    elif res["status"] != "ok":
        L.append(f"Cache incomplete, no verdict. Missing answers per model: {res['incomplete_models']}.")
    else:
        spec = res["set"]
        L += [f"- Set: {spec['name']} ({spec['n']} personas, recipients {spec['recipients']}, checkpoints {spec['checkpoints']}).",
              f"- Verdict: **{res['verdict']}** ({res['refuted_cells']} of {res['family_size']} cells refuted).",
              f"- Incomplete models left out: {res['incomplete_models'] or 'none'}.",
              f"- Degraded models left out (unreadable-answer rate above the limit): {res['degraded_models'] or 'none'}.", "",
              "| Model | Answers | Unreadable diabetes | Unreadable depression | Unreadable pregnancy |", "|---|---|---|---|---|"]
        for m, f in res["fallbacks"].items():
            L.append(f"| {m} | {f['answers']} | {f['diabetes']} | {f['depression']} | {f['pregnancy']} |")
        last = spec["checkpoints"][-1]
        L += ["", f"At checkpoint {last}. Mean R and D with the distribution-free 95% lower bound in brackets.", "",
              "| Model | Recipient | Attribute | Mean R | Mean D | Refuted cells |", "|---|---|---|---|---|---|"]
        for m in res["fallbacks"]:
            if m in res["incomplete_models"]:
                continue
            for r in spec["recipients"]:
                for a in world.sensitive:
                    sel = [c for c in res["cells"] if c["attacker"] == m and c["recipient"] == r and c["attribute"] == a]
                    cr = next(c for c in sel if c["checkpoint"] == last and c["statistic"] == "R")
                    cd = next(c for c in sel if c["checkpoint"] == last and c["statistic"] == "D")
                    L.append(f"| {m} | {r} | {a} | {cr['mean']:.3f} [{cr['lcb_95']:.3f}] | "
                             f"{cd['mean']:.3f} [{cd['lcb_95']:.3f}] | {sum(c['refuted'] for c in sel)} |")
    L.append("")
    (results_dir / "llm_table.md").write_text("\n".join(L), encoding="utf-8")


def refusal_message(res: dict) -> str:
    parts = []
    for m, d in res["degraded_models"].items():
        if "format_version" in d:
            parts.append(f"{m}: cache written by an older format ({d['format_version']}), not valid")
        else:
            parts.append(f"{m}: unreadable-answer rate " + ", ".join(f"{a} {r * 100:.1f}%" for a, r in d.items())
                         + f" is above the limit of {llm_cache.FALLBACK_THRESHOLD * 100:.0f}%")
    return "; ".join(parts) + ". The LLM family is not built from a degraded cache."


def run_if_cache(world: World, results_dir: Path, cache_dir: Path = llm_cache.DEFAULT_DIR) -> None:
    res = evaluate(world, cache_dir)
    if res is None:
        print("LLM cache: none, CPU-only run (as in v0.1.0)")
        return
    write_outputs(world, res, results_dir)
    if res["status"] == "refused":
        print("LLM cache refused: " + refusal_message(res))
    elif res["status"] == "ok":
        print(f"LLM family verdict: {res['verdict']} ({res['refuted_cells']} of {res['family_size']} cells refuted)")
    else:
        print(f"LLM cache incomplete, family not evaluated: {res['incomplete_models']}")
