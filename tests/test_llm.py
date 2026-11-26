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
    assert out["pregnancy"] is None  # a block with missing keys is not guessed (v4)


@pytest.mark.parametrize("raw", ["", "I cannot say.", "{not json", '{"diabetes": "high"}', '{"diabetes": {"none": -1, "2": 1, "5": 0, "11": 0, "18": 0}}',
                                 '{"diabetes": {"none": 0, "2": 0, "5": 0, "11": 0, "18": 0}}', '{"diabetes": [1, 2]}',
                                 '{"diabetes": {"none": "x"}}'])
def test_parse_unreadable_gives_none(raw):
    out = a_llm.parse_answer(raw, LABELS)
    assert out["diabetes"] is None


def test_unreadable_answer_falls_back_to_prior_and_is_counted(world):
    prior = a_llm.prior_marginals(world, "doctor", 5)
    parsed, fb, ab, lost = a_llm.finalize(world, '{"diabetes": [0.1, 0.9, 0, 0, 0]}', prior)
    assert parsed["diabetes"] == [0.1, 0.9, 0, 0, 0] or np.allclose(parsed["diabetes"], [0.1, 0.9, 0, 0, 0])
    assert fb == ["depression", "pregnancy"] and ab == fb and lost == []  # closed object: abstention
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


# ---- v3: truncated answers, salvage, fallback gate (real Qwen answers from the invalid v2 run) ----
from pathlib import Path  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "qwen_v2_truncated.jsonl"


def _fixture_records():
    header, recs = llm_cache.load_model(FIXTURE)
    return header, list(recs.values())


def test_json_closed():
    assert not a_llm.json_closed('{"a": {"x": 1}')
    assert a_llm.json_closed('Sure: {"a": {"x": 1}} thanks')
    assert not a_llm.json_closed('{"a": "}"')
    assert a_llm.json_closed('{"a": "}"}')
    assert not a_llm.json_closed("no object here")


def test_real_truncated_answers_are_salvaged(world):
    _, recs = _fixture_records()
    cut = [r for r in recs if len(r["fallback"]) == 3]
    assert len(cut) == 20
    for r in cut:
        assert not a_llm.json_closed(r["raw"])  # cut inside the object
        out = a_llm.parse_answer(r["raw"], LABELS)
        assert out["diabetes"] is not None and out["depression"] is not None  # complete blocks are kept
        assert abs(sum(out["diabetes"]) - 1) < 1e-12 and abs(sum(out["depression"]) - 1) < 1e-12
        assert out["pregnancy"] is None  # cut inside the last block
        prior = a_llm.prior_marginals(world, "doctor", 5)
        parsed, fb, ab, lost = a_llm.finalize(world, r["raw"], prior)
        assert fb == ["pregnancy"] and parsed["pregnancy"] == prior["pregnancy"]
        assert lost == ["pregnancy"] and ab == []  # cut off: technical loss, not abstention


def test_complete_real_answers_parse_fully_and_partial_ones_keep_good_blocks():
    _, recs = _fixture_records()
    full = [r for r in recs if r["fallback"] == [] and a_llm.json_closed(r["raw"])]
    assert full and all(None not in a_llm.parse_answer(r["raw"], LABELS).values() for r in full)
    one = [r for r in recs if len(r["fallback"]) == 1]
    for r in one:  # a closed but broken block costs only that attribute
        out = a_llm.parse_answer(r["raw"], LABELS)
        assert sum(v is None for v in out.values()) <= 1


def test_partial_salvage_with_one_broken_block():
    raw = ('{"diabetes": {"none": 0.9, "2": 0.1, "5": 0, "11": 0, "18": 0}, "depression": {"none": 0.5, "2": x}, '
           '"pregnancy": {"none": 1.0, "3": 0.0, "9": 0.0, "15": 0.0}}')
    out = a_llm.parse_answer(raw, LABELS)
    assert out["diabetes"][0] == 0.9 and out["depression"] is None and out["pregnancy"][0] == 1.0


def test_fixture_cache_is_refused_with_fallback_message(world, tmp_path):
    (tmp_path / "qwen.jsonl").write_text(FIXTURE.read_text())
    res = llm_family.evaluate(world, tmp_path)
    assert res["status"] == "refused"
    msg = llm_family.refusal_message(res)
    assert "unreadable-answer rate" in msg and "above the limit of 5%" in msg
    assert llm_cache.fallback_rates(tmp_path / "qwen.jsonl")["diabetes"] > 0.5


def test_old_format_cache_is_refused_even_with_low_fallback(world, tmp_path):
    _fill(world, tmp_path, models=("qwen",))
    try:
        p = tmp_path / "qwen.jsonl"
        lines = p.read_text().splitlines()
        h = json.loads(lines[0])
        h.pop("format_version")
        p.write_text("\n".join([json.dumps(h)] + lines[1:]) + "\n")
        res = llm_family.evaluate(world, tmp_path)
        assert res["status"] == "refused" and "unsupported format" in llm_family.refusal_message(res)
    finally:
        a_llm.SETS.pop("tiny", None)


def test_run_fails_loudly_above_threshold(world, tmp_path, capsys):
    a_llm.SETS["tiny"] = {"n": 6, "recipients": ("doctor",), "checkpoints": (3, 6)}
    try:
        rc = a_llm.main(["--set", "tiny", "--models", "qwen", "--backend", "stub", "--stub-garbage", "3",
                         "--cache-dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert rc == 3 and "TOO HIGH" in out and "Technical loss" in out
        rc = a_llm.main(["--set", "tiny", "--models", "phi", "--backend", "stub", "--cache-dir", str(tmp_path)])
        assert rc == 0 and "0.0%" in capsys.readouterr().out
    finally:
        a_llm.SETS.pop("tiny", None)


def test_answer_budget_is_twice_the_longest_valid_answer():
    assert a_llm.MAX_NEW_TOKENS >= 2 * 389


def test_degraded_cache_of_another_set_does_not_crash_a_good_one(world, tmp_path, capsys):
    (tmp_path / "qwen.jsonl").write_text(FIXTURE.read_text())  # degraded, set "reduced"
    _fill(world, tmp_path, models=("mistral",))  # good, set "tiny"
    try:
        res = llm_family.evaluate(world, tmp_path)
        assert res["status"] == "ok" and "qwen" in res["degraded_models"] and "mistral" in res["fallbacks"]
        llm_family.run_if_cache(world, tmp_path / "res", tmp_path)
        assert "LLM family verdict" in capsys.readouterr().out
    finally:
        a_llm.SETS.pop("tiny", None)
