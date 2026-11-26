# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""Preregistration v4: abstention vs technical loss, memory plan, one model failing (fake backends)."""
import json
from pathlib import Path

import pytest

from tll import llm_cache, llm_family
from tll.attackers import a_llm
from tll.population import load_json

LABELS = {"diabetes": ["none", "2", "5", "11", "18"], "depression": ["none", "2", "5", "11", "18"],
          "pregnancy": ["none", "3", "9", "15"]}
FIX = Path(__file__).parent / "fixtures" / "mistral_v3_answers.jsonl"


def _rows():
    return list(llm_cache.load_model(FIX)[1].values())


def _kind(pred):
    rows = [r for r in _rows() if pred(r["raw"])]
    assert rows
    return rows


def test_all_zero_block_is_abstention(world):
    prior = a_llm.prior_marginals(world, "doctor", 5)
    import re
    zero = re.compile(r'"depression": \{(?:"[^"]+": 0(?:\.0+)?(?:, )?)+\}')
    for r in _kind(lambda t: bool(zero.search(t))):
        parsed, fb, ab, lost = a_llm.finalize(world, r["raw"], prior)
        assert "depression" in fb and set(fb) == set(ab) and lost == []
        assert parsed["depression"] == prior["depression"]
        assert parsed["pregnancy"][0] == 1.0  # the other blocks are kept


def test_single_key_block_is_abstention(world):
    prior = a_llm.prior_marginals(world, "doctor", 5)
    for r in _kind(lambda t: '"diabetes": {"2": 1.0}' in t):
        out = a_llm.parse_answer(r["raw"], LABELS)
        assert out["diabetes"] is None  # missing keys are not guessed
        _, fb, ab, lost = a_llm.finalize(world, r["raw"], prior)
        assert "diabetes" in ab and lost == []


def test_percent_signs_are_read():
    for r in _kind(lambda t: "%" in t and t.lstrip().startswith("{")):
        out = a_llm.parse_answer(r["raw"], LABELS)
        assert out["diabetes"] is not None and out["pregnancy"] is not None
        assert abs(sum(out["diabetes"]) - 1) < 1e-9


def test_expressions_are_not_evaluated():
    rows = _kind(lambda t: "/" in t and t.lstrip().startswith("{") and any(c.isdigit() for c in t))
    expr = [r for r in rows if any(x in r["raw"] for x in ("78.0/126", "1.7/12")) or "/" in r["raw"]]
    for r in expr:
        out = a_llm.parse_answer(r["raw"], LABELS)
        assert None in out.values()  # the block with an expression is an abstention
    raw = '{"diabetes": {"none": "__import__(\'os\').getcwd()", "2": 0, "5": 0, "11": 0, "18": 0}}'
    assert a_llm.parse_answer(raw, LABELS)["diabetes"] is None


def test_prose_before_the_object_is_skipped():
    for r in _kind(lambda t: not t.lstrip().startswith("{") and t.rstrip().endswith("}")):
        out = a_llm.parse_answer(r["raw"], LABELS)
        assert all(v is not None for v in out.values())


def test_clean_rows_read_fully():
    rows = [r for r in _rows() if r["closed"] and r["raw"].count('"18"') >= 2 and '"15"' in r["raw"] and "%" not in r["raw"]
            and "/" not in r["raw"] and r["raw"].lstrip().startswith("{") and "0.0, \"2\": 0.0" not in r["raw"]]
    assert rows
    for r in rows:
        if r["fallback"] == []:
            assert None not in a_llm.parse_answer(r["raw"], LABELS).values()


def test_record_kinds_of_older_records():
    assert llm_cache.record_kinds({"fallback": ["diabetes"], "closed": True}) == ([], ["diabetes"])
    assert llm_cache.record_kinds({"fallback": ["diabetes"], "closed": False}) == (["diabetes"], [])
    assert llm_cache.record_kinds({"fallback": ["diabetes"]}) == (["diabetes"], [])  # v2 records had no `closed`
    assert llm_cache.record_kinds({"loss": [], "abstain": ["pregnancy"], "fallback": ["pregnancy"]}) == ([], ["pregnancy"])


# ---- memory plan ------------------------------------------------------------------------------
def test_oom_retry_splits_in_halves_and_completes():
    calls = []

    class Oom(RuntimeError):
        pass

    def run(idx):
        calls.append(len(idx))
        if len(idx) > 2:
            raise Oom("CUDA out of memory")
        return {i: f"a{i}" for i in idx}

    retries = []
    out = a_llm.run_with_oom_retry(run, list(range(8)), lambda e: isinstance(e, Oom), retries.append)
    assert out == {i: f"a{i}" for i in range(8)}
    assert retries == [8, 4, 4] and max(c for c in calls if c <= 2) == 2


def test_oom_on_one_item_raises_and_other_errors_pass_through():
    class Oom(RuntimeError):
        pass

    def always(idx):
        raise Oom("out of memory")

    with pytest.raises(Oom):
        a_llm.run_with_oom_retry(always, [1], lambda e: True)
    with pytest.raises(ValueError):
        a_llm.run_with_oom_retry(lambda idx: (_ for _ in ()).throw(ValueError("x")), [1, 2], lambda e: False)


def test_batches_respect_cap_and_token_budget():
    lengths = [300] * 12 + [1800] * 4
    bs = a_llm.plan_batches(lengths, max_batch=8, token_budget=10_000, new_tokens=800)
    assert sorted(i for b in bs for i in b) == list(range(16))
    for b in bs:
        longest = max(lengths[i] for i in b)
        assert len(b) <= 8 and (len(b) == 1 or len(b) * (longest + 800) <= 10_000)
    assert max(len(b) for b in bs if max(lengths[i] for i in b) == 1800) == 3  # long prompts: small batches
    assert a_llm.plan_batches([5000], 8, 1000, 800) == [[0]]  # one item always goes through


def test_registry_has_batch_budgets():
    for m in load_json("models.json")["models"].values():
        assert 4000 <= m["batch_token_budget"] <= 20000


def test_one_model_failing_does_not_stop_the_next(world, tmp_path, monkeypatch, capsys):
    a_llm.SETS["tiny"] = {"n": 6, "recipients": ("doctor",), "checkpoints": (3, 6)}
    real = a_llm.run_model

    def flaky(w, spec, key, ms, backend, cache_dir, limit=None, chunk=64):
        if key == "qwen":
            raise RuntimeError("CUDA out of memory (fake)")
        return real(w, spec, key, ms, backend, cache_dir, limit, chunk)

    monkeypatch.setattr(a_llm, "run_model", flaky)
    try:
        rc = a_llm.main(["--set", "tiny", "--models", "qwen,phi", "--backend", "stub", "--cache-dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert rc == 4 and "[qwen] FAILED" in out and "Continuing with the next model" in out
        assert (tmp_path / "phi.jsonl").exists() and len(llm_cache.load_model(tmp_path / "phi.jsonl")[1]) == 36
    finally:
        a_llm.SETS.pop("tiny", None)


def test_check_run_uses_the_longest_prompts(world, tmp_path):
    a_llm.SETS["tiny"] = {"n": 6, "recipients": ("doctor",), "checkpoints": (3, 6)}
    try:
        reg = load_json("models.json")["models"]
        spec = a_llm.resolve_set(world, "tiny")
        a_llm.run_model(world, spec, "phi", reg["phi"], a_llm.StubBackend(), tmp_path, limit=6)
        recs = llm_cache.load_model(tmp_path / "phi.jsonl")[1]
        assert len(recs) == 6 and {k[3] for k in recs} == {6}  # the later checkpoint has the longer transcripts
    finally:
        a_llm.SETS.pop("tiny", None)


# ---- abstention is visible, weak attackers are marked, nothing is dropped ----------------------
def _fill(world, tmp_path, models):
    reg = load_json("models.json")["models"]
    spec = a_llm.resolve_set(world, "tiny")
    for m in models:
        a_llm.run_model(world, spec, m, reg[m], a_llm.StubBackend(), tmp_path)


def test_table_shows_abstention_and_marks_weak_attackers(world, tmp_path):
    a_llm.SETS["tiny"] = {"n": 6, "recipients": ("doctor",), "checkpoints": (3, 6)}
    try:
        _fill(world, tmp_path, ("qwen", "mistral"))
        p = tmp_path / "mistral.jsonl"
        lines = p.read_text().splitlines()
        out = [lines[0]]
        for i, line in enumerate(lines[1:]):
            r = json.loads(line)
            if i % 2 == 0:  # half of the answers abstain on depression (complete answer, unusable block)
                r["fallback"], r["abstain"], r["loss"], r["closed"] = ["depression"], ["depression"], [], True
            out.append(json.dumps(r, sort_keys=True))
        p.write_text("\n".join(out) + "\n")
        res = llm_family.evaluate(world, tmp_path)
        assert res["status"] == "ok" and res["weak_attackers"] == ["mistral"]
        assert "weak attackers" in res["verdict_text"] and "mistral" in res["verdict_text"]
        assert set(res["fallbacks"]) == {"qwen", "mistral"}  # nobody is dropped
        assert res["fallbacks"]["mistral"]["rates"]["abstain"]["depression"] == pytest.approx(0.5)
        llm_family.write_outputs(world, res, tmp_path / "out")
        table = (tmp_path / "out" / "llm_table.md").read_text()
        assert "weak attacker" in table and "Abstain depression" in table and "50.0%" in table
    finally:
        a_llm.SETS.pop("tiny", None)


def test_technical_loss_still_refuses_but_abstention_does_not(world, tmp_path):
    a_llm.SETS["tiny"] = {"n": 6, "recipients": ("doctor",), "checkpoints": (3, 6)}
    try:
        _fill(world, tmp_path, ("phi",))
        p = tmp_path / "phi.jsonl"
        lines = p.read_text().splitlines()
        rows = [json.loads(x) for x in lines[1:]]
        for r in rows:
            r["fallback"], r["abstain"], r["loss"], r["closed"] = ["pregnancy"], ["pregnancy"], [], True
        p.write_text("\n".join([lines[0]] + [json.dumps(r) for r in rows]) + "\n")
        assert llm_family.evaluate(world, tmp_path)["status"] == "ok"  # 100% abstention is visible, not refused
        for r in rows:
            r["abstain"], r["loss"], r["closed"] = [], ["pregnancy"], False
        p.write_text("\n".join([lines[0]] + [json.dumps(r) for r in rows]) + "\n")
        assert llm_family.evaluate(world, tmp_path)["status"] == "refused"
    finally:
        a_llm.SETS.pop("tiny", None)
