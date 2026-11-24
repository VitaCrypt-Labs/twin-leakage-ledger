# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""LLM path without a model: parsing, prompts, cache round trip, family assembly (fake answers)."""
import json

import numpy as np
import pytest

from tll import llm_cache, llm_family
from tll.attackers import a_llm
from tll.population import load_json

LABELS = {"diabetes": ["none", "2", "5", "11", "18"], "depression": ["none", "2", "5", "11", "18"],
          "pregnancy": ["none", "3", "9", "15"]}


def test_parse_clean_json():
    raw = json.dumps({"diabetes": {"none": 0.5, "2": 0.1, "5": 0.1, "11": 0.1, "18": 0.2},
                      "depression": {"none": 1, "2": 0, "5": 0, "11": 0, "18": 0},
                      "pregnancy": {"none": 0.9, "3": 0.1, "9": 0, "15": 0}})
    out = a_llm.parse_answer(raw, LABELS)
    assert abs(sum(out["diabetes"]) - 1) < 1e-12 and out["depression"][0] == 1.0 and out["pregnancy"][1] == 0.1


def test_parse_robust_forms():
    raw = ('Sure! Here you go:\n```json\n{"Diabetes": {"None": "60%", "month 2": "10%", "5": 10, "11": 10, "18": 10},'
           ' "depression": [0.2, 0.2, 0.2, 0.2, 0.2], "pregnancy": {"none": 3, "3": 1}}\n``` hope it helps')
    out = a_llm.parse_answer(raw, LABELS)
    assert abs(out["diabetes"][0] - 0.6) < 1e-12 and abs(out["diabetes"][1] - 0.1) < 1e-12
    assert out["depression"] == [0.2] * 5
    assert abs(out["pregnancy"][0] - 0.75) < 1e-12 and out["pregnancy"][2] == 0.0  # missing keys count as zero


@pytest.mark.parametrize("raw", ["", "I cannot say.", "{not json", '{"diabetes": "high"}', '{"diabetes": {"none": -1, "2": 1, "5": 0, "11": 0, "18": 0}}',
                                 '{"diabetes": {"none": 0, "2": 0, "5": 0, "11": 0, "18": 0}}', '{"diabetes": [1, 2]}',
                                 '{"diabetes": {"none": "x"}}'])
def test_parse_unreadable_gives_none(raw):
    out = a_llm.parse_answer(raw, LABELS)
    assert out["diabetes"] is None


def test_unreadable_answer_falls_back_to_prior_and_is_counted(world):
    prior = a_llm.prior_marginals(world, "doctor", 5)
    parsed, fb = a_llm.finalize(world, '{"diabetes": [0.1, 0.9, 0, 0, 0]}', prior)
    assert parsed["diabetes"] == [0.1, 0.9, 0, 0, 0] or np.allclose(parsed["diabetes"], [0.1, 0.9, 0, 0, 0])
    assert fb == ["depression", "pregnancy"]
    assert parsed["depression"] == prior["depression"]


def test_prompt_uses_only_prior_knowledge_and_messages(world):
    """Two people who agree on what the doctor knows and on the messages received get the same prompt."""
    pop = world.pop
    h, main = a_llm.personas(world, 40)
    base = world.variants["doctor"]["base"]
    for i in range(40):
        for j in range(i + 1, 40):
            if pop.known_key(base, h[i]) == pop.known_key(base, h[j]):
                s = [s for s in world.slots_of["doctor"] if s.month <= 3]
                if all(main[i, x.idx] == main[j, x.idx] for x in s) and h[i] != h[j]:
                    for pid in a_llm.WORDINGS:
                        pi, _ = a_llm.build_prompt(world, pid, "doctor", int(h[i]), main[i], 3)
                        pj, _ = a_llm.build_prompt(world, pid, "doctor", int(h[j]), main[j], 3)
                        assert pi == pj
                    return
    pytest.skip("no suitable pair among the first 40 personas")


def test_prompt_contains_known_facts_and_no_future_messages(world):
    h, main = a_llm.personas(world, 3)
    text, prior = a_llm.build_prompt(world, "p1", "doctor", int(h[0]), main[0], 3)
    assert "diabetes diagnosis start" in text
    assert "Month 4 |" not in text
    assert all(abs(sum(p) - 1) < 1e-9 for p in prior.values())


def test_plan_sizes():
    class W:  # only recipients are needed
        recipients = ["a", "b", "c", "d"]
    full = a_llm.resolve_set(W, "full")
    red = a_llm.resolve_set(W, "reduced")
    assert a_llm.prompt_count(full, 3) == 200 * 4 * 5 * 3 * 3 == 36_000
    assert a_llm.prompt_count(red, 3) == 60 * 2 * 3 * 3 * 3 == 3_240


def test_registry_models_have_permissive_licences_and_pins():
    reg = load_json("models.json")["models"]
    assert len(reg) == 3 and len({m["family"] for m in reg.values()}) == 3
    for m in reg.values():
        assert m["license"] in ("Apache-2.0", "MIT")
        assert len(m["revision"]) == 40
        assert all(len(f["sha256"]) == 64 for f in m["files"])


def _fill(world, tmp_path, models=("qwen", "mistral")):
    a_llm.SETS["tiny"] = {"n": 6, "recipients": ("doctor",), "checkpoints": (3, 6)}
    reg = load_json("models.json")["models"]
    spec = a_llm.resolve_set(world, "tiny")
    for m in models:
        a_llm.run_model(world, spec, m, reg[m], a_llm.StubBackend(), tmp_path)
    return spec


def test_cache_round_trip_and_family_assembly(world, tmp_path):
    spec = _fill(world, tmp_path)
    try:
        header, recs = llm_cache.load_model(tmp_path / "qwen.jsonl")
        assert header["revision"] == load_json("models.json")["models"]["qwen"]["revision"]
        assert len(recs) == 6 * 1 * 2 * 3 == 36
        rec = next(iter(recs.values()))
        assert {"model_revision", "prompt_id", "seed", "persona", "recipient", "checkpoint", "raw", "parsed"} <= set(rec)
        # a second run resumes: nothing new to write
        reg = load_json("models.json")["models"]
        assert a_llm.run_model(world, spec, "qwen", reg["qwen"], a_llm.StubBackend(), tmp_path) == 0
        res = llm_family.evaluate(world, tmp_path)
        assert res["status"] == "ok" and res["family_size"] == 2 * 1 * 3 * 2 * 2
        n_fb = sum(1 for r in recs.values() if r["fallback"])
        assert sum(res["fallbacks"]["qwen"][a] for a in world.sensitive) >= n_fb
        # attackers that answer with the base rates cannot beat the bound
        assert res["verdict"] == "bound held"
        assert all(abs(c["n"] - 6) < 1e-9 for c in res["cells"])
        llm_family.write_outputs(world, res, tmp_path / "out")
        assert (tmp_path / "out" / "llm_table.md").exists() and (tmp_path / "out" / "llm_verdict.json").exists()
    finally:
        a_llm.SETS.pop("tiny", None)


def test_incomplete_cache_is_not_evaluated(world, tmp_path):
    _fill(world, tmp_path, models=("qwen",))
    try:
        p = tmp_path / "qwen.jsonl"
        lines = p.read_text().splitlines()
        p.write_text("\n".join(lines[:-3]) + "\n")
        res = llm_family.evaluate(world, tmp_path)
        assert res["status"] == "incomplete" and res["incomplete_models"] == {"qwen": 3}
    finally:
        a_llm.SETS.pop("tiny", None)


def test_no_cache_means_no_family(world, tmp_path, capsys):
    assert llm_family.evaluate(world, tmp_path / "missing") is None
    llm_family.run_if_cache(world, tmp_path / "res", tmp_path / "missing")
    assert "none" in capsys.readouterr().out
    assert not (tmp_path / "res").exists()


def test_header_mismatch_is_refused(world, tmp_path):
    _fill(world, tmp_path, models=("qwen",))
    try:
        with pytest.raises(ValueError):
            llm_cache.open_for_append(tmp_path / "qwen.jsonl", {"model_key": "qwen", "model_id": "x", "revision": "y", "set": "tiny"})
    finally:
        a_llm.SETS.pop("tiny", None)
