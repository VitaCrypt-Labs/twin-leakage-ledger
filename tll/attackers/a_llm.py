# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
"""LLM attackers (preregistration v2). Three open-weight models of three families read the messages a
recipient received up to a checkpoint and state a probability distribution over the values of each
sensitive attribute. Answers are cached in cache/llm/ (see llm_cache.py).

Prompt wordings p1, p2, p3 and their seeds are fixed here and are part of `prereg-v2`; do not edit them
after that tag. Run: python -m tll.attackers.a_llm --set reduced --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import llm_cache
from ..population import load_json
from ..sim import CHECKPOINTS, World

SEED_PERSONAS = 20261210
N_PERSONAS_DRAWN = 200
PROMPT_SEEDS = {"p1": 101, "p2": 102, "p3": 103}
MAX_NEW_TOKENS = 160
TEMPERATURE = 0.3
TOP_P = 0.95

SETS = {
    "full": {"n": 200, "recipients": None, "checkpoints": CHECKPOINTS},
    "reduced": {"n": 60, "recipients": ("doctor", "family"), "checkpoints": (3, 12, 24)},
}

ATTR_TEXT = {
    "diabetes": "diabetes diagnosis (month the diagnosis starts)",
    "depression": "depression diagnosis (month the diagnosis starts)",
    "pregnancy": "pregnancy (month it starts; it lasts nine months)",
}
KNOWN_TEXT = {
    "age": "age group", "sex": "sex", "pregnancy": "pregnancy start", "diabetes": "diabetes diagnosis start",
    "depression": "depression diagnosis start", "antidiabetic": "antidiabetic medication start",
    "ssri": "SSRI-class medication start", "hba1c": "raised HbA1c start",
}


# ---- plan ------------------------------------------------------------------------------------
def resolve_set(world: World, name: str) -> dict:
    s = SETS[name]
    return {"name": name, "n": s["n"], "recipients": list(s["recipients"] or world.recipients),
            "checkpoints": list(s["checkpoints"])}


def personas(world: World, n: int):
    """History indices and main-mode outcomes of the LLM personas (always drawn as 200, then a prefix)."""
    h, _U, main, _ctrl = world.draw_personas(N_PERSONAS_DRAWN, SEED_PERSONAS)
    return h[:n], main[:n]


def prompt_count(spec: dict, n_models: int) -> int:
    return spec["n"] * len(spec["recipients"]) * len(spec["checkpoints"]) * len(PROMPT_SEEDS) * n_models


# ---- prompt ----------------------------------------------------------------------------------
def value_labels(world: World, attr: str) -> list[str]:
    return ["none"] + [str(m) for m in world.pop.grid[attr][1:]]


def prior_marginals(world: World, recipient: str, hist: int) -> dict[str, list[float]]:
    pop = world.pop
    known = world.variants[recipient]["base"]
    sub = pop.subset(known, pop.known_key(known, hist))
    w = pop.prior[sub] / pop.prior[sub].sum()
    return {a: np.bincount(pop.values[a][sub].astype(np.int64), weights=w, minlength=pop.n_values[a]).tolist()
            for a in world.sensitive}


def known_text(world: World, recipient: str, hist: int) -> str:
    pop = world.pop
    rec = pop.record(hist)
    parts = []
    for a in world.variants[recipient]["base"]:
        if a == "age":
            v = pop.spec["age_groups"][rec["age"]]
        elif a == "sex":
            v = pop.spec["sexes"][rec["sex"]]
        else:
            v = "none in the 24 months" if rec[a] is None else f"month {rec[a]}"
        parts.append(f"{KNOWN_TEXT[a]}: {v}")
    return "; ".join(parts) if parts else "nothing"


def transcript(world: World, recipient: str, outcomes_row: np.ndarray, checkpoint: int) -> str:
    blocks = []
    for s in world.slots_of[recipient]:
        if s.month > checkpoint:
            continue
        t = world.text(s, int(outcomes_row[s.idx]), "main")
        if t is not None:
            blocks.append(t)
    return "\n\n".join(blocks) if blocks else "(no message was received)"


def _pct(p: list[float]) -> str:
    return ", ".join(f"{x * 100:.1f}%" for x in p)


def _spec_lines(world: World, prior: dict) -> str:
    lines = []
    for a in world.sensitive:
        labels = value_labels(world, a)
        pairs = ", ".join(f'"{k}" (base rate {x * 100:.1f}%)' for k, x in zip(labels, prior[a]))
        lines.append(f"- {a}: {ATTR_TEXT[a]}. Values: {pairs}")
    return "\n".join(lines)


def _schema(world: World) -> str:
    parts = []
    for a in world.sensitive:
        inner = ", ".join(f'"{k}": <probability>' for k in value_labels(world, a))
        parts.append(f'"{a}": {{{inner}}}')
    return "{" + ", ".join(parts) + "}"


WORDINGS = {
    "p1": (
        "You are a privacy auditor. A {label} received the messages below about one person, sent by that person's "
        "health agent. Before any message the {label} already knew: {known}.\n\n"
        "Messages received up to month {ck}, oldest first. A month without a message means that nothing was sent "
        "in that month; a missing expected message can be informative too.\n\n{transcript}\n\n"
        "Estimate, for each attribute, the probability of each value for this person.\n{spec}\n\n"
        "Probabilities of one attribute must sum to 1. Answer with one JSON object and nothing else, in this form:\n{schema}"
    ),
    "p2": (
        "Task: attribute inference. Observer: {label}. What the observer knew in advance: {known}.\n"
        "Observed messages (months 1 to {ck}):\n{transcript}\n\n"
        "Hidden attributes of the person, with the observer's base rates before any message:\n{spec}\n\n"
        "Using the observed messages, including their timing and absences, give the observer's updated probability "
        "for every value of every hidden attribute. Each attribute's probabilities sum to 1. "
        "Reply with a single JSON object, no other text. Format:\n{schema}"
    ),
    "p3": (
        "Below is everything a {label} has received from one person's health agent through month {ck}.\n\n"
        "{transcript}\n\n"
        "The {label} knew this beforehand: {known}.\n"
        "We want to know how well the messages reveal three hidden facts. For each fact, list the probability of "
        "each possible value. Base rates before reading any message:\n{spec}\n"
        "Update the base rates with what the messages show. Give probabilities that add up to 1 for each fact. "
        "Output only a JSON object like this:\n{schema}"
    ),
}


@dataclass
class Item:
    model: str
    prompt_id: str
    persona: int
    hist: int
    recipient: str
    checkpoint: int
    prompt: str
    prior: dict
    labels: dict


def build_prompt(world: World, prompt_id: str, recipient: str, hist: int, outcomes_row: np.ndarray, checkpoint: int):
    """Returns (prompt, prior marginals). Uses only the recipient's published prior knowledge and the
    messages the recipient received; nothing else about the person."""
    prior = prior_marginals(world, recipient, hist)
    text = WORDINGS[prompt_id].format(
        label=world.labels[recipient].lower(), known=known_text(world, recipient, hist), ck=checkpoint,
        transcript=transcript(world, recipient, outcomes_row, checkpoint), spec=_spec_lines(world, prior),
        schema=_schema(world))
    return text, prior


# ---- parsing ---------------------------------------------------------------------------------
_NONE_ALIASES = ("none", "no", "never", "absent")


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        x = float(v)
    elif isinstance(v, str):
        try:
            x = float(v.strip().rstrip("%").strip())
        except ValueError:
            return None
    else:
        return None
    return x if np.isfinite(x) and x >= 0 else None


def _find_object(raw: str):
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", raw):
        try:
            obj, _ = dec.raw_decode(raw[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _dist(val, labels: list[str]):
    if isinstance(val, dict):
        low = {str(k).strip().lower(): v for k, v in val.items()}
        out = []
        for lab in labels:
            cands = [lab, f"month {lab}", f"m{lab}"] + (list(_NONE_ALIASES) if lab == "none" else [])
            hit = next((low[c] for c in cands if c in low), 0.0)
            out.append(_num(hit))
        if any(x is None for x in out):
            return None
    elif isinstance(val, list) and len(val) == len(labels):
        out = [_num(x) for x in val]
        if any(x is None for x in out):
            return None
    else:
        return None
    tot = sum(out)
    return [x / tot for x in out] if tot > 0 else None


def parse_answer(raw: str, labels_by_attr: dict[str, list[str]]) -> dict[str, list[float] | None]:
    """Per attribute a normalised distribution, or None when it cannot be read."""
    obj = _find_object(raw or "")
    res = {a: None for a in labels_by_attr}
    if obj is None:
        return res
    low = {str(k).strip().lower(): v for k, v in obj.items()}
    for a, labels in labels_by_attr.items():
        if a in low:
            res[a] = _dist(low[a], labels)
    return res


def finalize(world: World, raw: str, prior: dict) -> tuple[dict, list[str]]:
    labels = {a: value_labels(world, a) for a in world.sensitive}
    parsed = parse_answer(raw, labels)
    fallback = [a for a in world.sensitive if parsed[a] is None]
    return {a: (parsed[a] if parsed[a] is not None else prior[a]) for a in world.sensitive}, fallback


# ---- backends --------------------------------------------------------------------------------
class StubBackend:
    """Deterministic fake model for tests and plumbing checks. Answers with the base rates, tilted
    a little by a hash of the prompt; every 11th answer is unreadable to exercise the fallback."""

    def generate(self, items: list[Item], seed: int) -> list[str]:
        out = []
        for it in items:
            h = int(hashlib.sha256(f"{seed}|{it.prompt}".encode()).hexdigest(), 16)
            if h % 11 == 0:
                out.append("I cannot answer that.")
                continue
            ans = {}
            for a, p in it.prior.items():
                ans[a] = {k: float(x) for k, x in zip(it.labels[a], p)}
            out.append("```json\n" + json.dumps(ans) + "\n```")
        return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


class HFBackend:
    """Hugging Face transformers on a GPU. Downloads the pinned revision, checks file hashes."""

    def __init__(self, spec: dict, batch_size: int):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.batch_size = batch_size
        path = Path(snapshot_download(spec["hf_id"], revision=spec["revision"], allow_patterns=spec["allow_patterns"]))
        for f in spec["files"]:
            got = sha256_file(path / f["path"])
            if got != f["sha256"]:
                raise RuntimeError(f"hash mismatch for {f['path']}: {got}")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map="auto").eval()
        self.torch_version = torch.__version__

    def generate(self, items: list[Item], seed: int) -> list[str]:
        torch = self.torch
        out = []
        for b in range(0, len(items), self.batch_size):
            chunk = items[b:b + self.batch_size]
            texts = [self.tok.apply_chat_template([{"role": "user", "content": it.prompt}],
                                                  add_generation_prompt=True, tokenize=False) for it in chunk]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            torch.manual_seed(seed + b // self.batch_size)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                                          temperature=TEMPERATURE, top_p=TOP_P, pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            out += self.tok.batch_decode(new, skip_special_tokens=True)
        return out


# ---- run -------------------------------------------------------------------------------------
def plan_items(world: World, spec: dict, model_key: str, done: set) -> list[Item]:
    h, main = personas(world, spec["n"])
    items = []
    for pid in PROMPT_SEEDS:
        for i in range(spec["n"]):
            for r in spec["recipients"]:
                for ck in spec["checkpoints"]:
                    if (pid, i, r, ck) in done:
                        continue
                    text, prior = build_prompt(world, pid, r, int(h[i]), main[i], ck)
                    items.append(Item(model_key, pid, i, int(h[i]), r, ck, text, prior,
                                      {a: value_labels(world, a) for a in world.sensitive}))
    return items


def run_model(world: World, spec: dict, model_key: str, model_spec: dict, backend, cache_dir: Path,
              limit: int | None = None, chunk: int = 64) -> int:
    path = Path(cache_dir) / f"{model_key}.jsonl"
    header = {"model_key": model_key, "model_id": model_spec["hf_id"], "revision": model_spec["revision"],
              "set": spec["name"], "n_personas": spec["n"], "recipients": spec["recipients"],
              "checkpoints": spec["checkpoints"], "persona_seed": SEED_PERSONAS,
              "prompt_seeds": PROMPT_SEEDS, "temperature": TEMPERATURE, "top_p": TOP_P,
              "max_new_tokens": MAX_NEW_TOKENS}
    f = llm_cache.open_for_append(path, header)
    done = set(llm_cache.load_model(path)[1])
    items = plan_items(world, spec, model_key, done)
    if limit is not None:
        items = items[:limit]
    items.sort(key=lambda it: (it.prompt_id, len(it.prompt)))
    t0, n_done = time.time(), 0
    for c in range(0, len(items), chunk):
        part = items[c:c + chunk]
        for pid in {it.prompt_id for it in part}:
            sub = [it for it in part if it.prompt_id == pid]
            raws = backend.generate(sub, PROMPT_SEEDS[pid])
            for it, raw in zip(sub, raws):
                parsed, fb = finalize(world, raw, it.prior)
                llm_cache.append(f, {"model_key": model_key, "model_revision": model_spec["revision"],
                                     "prompt_id": it.prompt_id, "seed": PROMPT_SEEDS[pid], "persona": it.persona,
                                     "history": it.hist, "recipient": it.recipient, "checkpoint": it.checkpoint,
                                     "raw": raw, "parsed": parsed, "fallback": fb})
        n_done += len(part)
        rate = (time.time() - t0) / n_done
        print(f"[{model_key}] {n_done}/{len(items)} answers, {rate:.2f} s each, "
              f"about {(len(items) - n_done) * rate / 60:.0f} min left", flush=True)
    f.close()
    return n_done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate LLM attacker answers into the cache.")
    ap.add_argument("--set", default="full", choices=list(SETS))
    ap.add_argument("--models", default="qwen,mistral,phi")
    ap.add_argument("--backend", default="hf", choices=["hf", "stub"])
    ap.add_argument("--cache-dir", default=str(llm_cache.DEFAULT_DIR))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="stop after this many answers per model (benchmark)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the estimate, then stop")
    a = ap.parse_args(argv)

    world = World()
    registry = load_json("models.json")["models"]
    keys = [k.strip() for k in a.models.split(",") if k.strip()]
    for k in keys:
        if k not in registry:
            print(f"unknown model {k}; known: {', '.join(registry)}")
            return 2
    spec = resolve_set(world, a.set)
    total = prompt_count(spec, len(keys))
    print(f"set '{spec['name']}': {spec['n']} personas x {len(spec['recipients'])} recipients x "
          f"{len(spec['checkpoints'])} checkpoints x {len(PROMPT_SEEDS)} prompts x {len(keys)} models = {total} prompts")
    est = 0.0
    for k in keys:
        n = prompt_count(spec, 1)
        s = registry[k]["assumed_seconds_per_prompt"] * n
        est += s
        print(f"  {k} ({registry[k]['hf_id']}): {n} prompts, about {s / 3600:.1f} h at an assumed "
              f"{registry[k]['assumed_seconds_per_prompt']} s per prompt (two T4 GPUs, batch {a.batch_size}); "
              f"measure with --limit 32 before the full run")
    print(f"estimated generation time: about {est / 3600:.1f} h plus downloads (assumption, not measured)")
    if a.dry_run:
        return 0
    for k in keys:
        backend = StubBackend() if a.backend == "stub" else HFBackend(registry[k], a.batch_size)
        n = run_model(world, spec, k, registry[k], backend, Path(a.cache_dir), a.limit)
        print(f"[{k}] wrote {n} answers to {a.cache_dir}")
        del backend
    return 0


if __name__ == "__main__":
    sys.exit(main())
