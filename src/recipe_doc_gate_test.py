"""Does the TRAINING RECIPE's prose of record still agree with the production mirror's recipe?

The sibling of `mode_flag_doc_gate_test.py`, and built to its pattern. Its subject is
`designs/endstate/design_learner_recipe.md` — the ALWAYS-CURRENT spec of record for the training
recipe — against `main.train.recipe_surface.production_recipe()`: the `recipe` block of
`designs/production_config.json` plus the mirror fields the recipe reads (K10(a)).

**Why it exists.** `--arch production` now APPLIES the recipe block, so the block is what a fresh
run trains with — and the doc is what a reader believes it trains with. Two copies of a number
with nothing holding them together drift; this gate is what holds them together.

**The extraction is DECLARED, never inferred.** `_CLAIMS` is an explicit table of (recipe key →
the regex that finds the doc's statement of its value). A row whose pattern stops matching FAILS
(the sentence was rewritten), and a key missing from the mirror FAILS (renamed) — a gate that went
quiet on a rename would be decorative.

**Four checks.** (1) every claim's value equals the mirror's; (2) every declared key exists and its
pattern matches; (3) EVERY recipe row (`recipe_surface.ALL_ROWS`) is covered by at least one claim,
so a new knob cannot land without its prose; (4) a planted change to the DOC alone, and to the
BLOCK alone, each fail the gate.

**Cost: milliseconds.** No cost marker. Opt out explicitly:

    GEN3AI_SKIP_RECIPE_DOC_GATE=1 pytest src/ -q
"""
from __future__ import annotations

import copy
import os
import pathlib
import re
from typing import Any, Dict, List, NamedTuple, Tuple

import pytest

from main.train.recipe_surface import (ALL_ROWS, FORK_OVERRIDES, FORK_ROWS, FRESH_ROWS, KL_CONSTANTS,
                                        KL_KEY, ROWS, SIZING_ROWS, _raw_mirror, recipe_blocks)
from utils.paths import repo_path

pytestmark = pytest.mark.static   # the `static` budget tier (conftest._STATIC_BUDGET_BASE_S)

_SKIP = pytest.mark.skipif(
    os.environ.get("GEN3AI_SKIP_RECIPE_DOC_GATE") == "1",
    reason="GEN3AI_SKIP_RECIPE_DOC_GATE=1",
)

_DOC = ("designs", "endstate", "design_learner_recipe.md")


class Claim(NamedTuple):
    key: str
    block: str          # "fresh" | "sizing" | "fork" | "kl" — which part of the `recipe` block
    pattern: str        # must capture the stated value in a group named `value`
    where: str


_HOME = {"fresh": "recipe.fresh", "sizing": "recipe.sizing", "fork": "recipe.fork",
         "kl": "recipe.fresh.kl_controller"}


def _t(key: str, block: str = "fresh") -> Claim:
    """A row of §3.22's table: `| `key` | value | `recipe.<block>` | source |`. Anchored on the KEY
    and the BLOCK column, never on the value — a pattern that hardcodes the value is vacuous."""
    return Claim(key, block, r"\|\s*`" + re.escape(key) + r"`\s*\|\s*(?P<value>[^|]+?)\s*\|\s*`"
                 + re.escape(_HOME[block]) + r"`\s*\|", f"§3.22 the recipe table, `{key}` in {_HOME[block]}")


# NOTE ON WRITING A ROW: anchor on the words around the value, not on the value itself.
_CLAIMS: Tuple[Claim, ...] = tuple(_t(r.dest) for r in FRESH_ROWS) + tuple(
    _t(r.dest, "sizing") for r in SIZING_ROWS) + tuple(
    _t(k, "kl") for k in KL_CONSTANTS) + (
    _t("n_epochs", "fork"), _t("fork_lr", "fork"), _t("fork_lr_freeze", "fork"),
    # ---- §1's one-page table: its live-value column states the recipe too ------------------
    Claim("n_envs", "sizing", r"\|\s*1\s*\|\s*`n_envs` \(N\)\s*\|\s*(?P<value>\d+)\s*\|", "§1 row 1"),
    Claim("n_steps", "sizing",
          r"\|\s*2\s*\|\s*rollout size N × `n_steps`\s*\|\s*\d+ × (?P<value>\d+) =", "§1 row 2"),
    Claim("batch_size", "fresh", r"\|\s*3\s*\|\s*micro-batch\s*\|\s*(?P<value>\d+)\s*\|", "§1 row 3"),
    Claim("grad_accum_steps", "fresh",
          r"\|\s*4\s*\|\s*accumulation K / effective batch\s*\|\s*(?P<value>\d+) →", "§1 row 4"),
    Claim("n_epochs", "fresh", r"\|\s*6\s*\|\s*`n_epochs`\s*\|\s*(?P<value>\d+) in the argv", "§1 row 6"),
    Claim("n_epochs", "fork", r"\*\*(?P<value>\d+) adopted\*\* for generalists", "§1 row 6"),
    Claim("clip_range", "fresh", r"\|\s*7\s*\|\s*`clip_range`\s*\|\s*(?P<value>[\d.]+),", "§1 row 7"),
    Claim("fork_lr", "fork", r"\*\*(?P<value>[\d.e-]+) with E5\*\*", "§1 row 9"),
    Claim("ent_coef", "fresh", r"\|\s*11\s*\|\s*`ent_coef`\s*\|\s*(?P<value>[\d.]+)\s*\|", "§1 row 11"),
    Claim("gamma", "fresh", r"\|\s*12\s*\|\s*γ \(discount\)\s*\|\s*(?P<value>[\d.]+),", "§1 row 12"),
    Claim("policy_gae_lambda", "fresh",
          r"\|\s*13\s*\|\s*policy GAE λ\s*\|\s*(?P<value>[\d.]+)\s*\|", "§1 row 13"),
    Claim("vf_coef", "fresh", r"`vf_coef` (?P<value>[\d.]+), no value clip", "§1 row 14"),
    Claim("weight_decay", "fresh", r"wd (?P<value>[\d.e-]+) on every param", "§1 row 18"),
    Claim("target_kl", "kl", r"\|\s*8\s*\|\s*KL → LR controller\s*\|\s*target (?P<value>[\d.]+),", "§1 row 8"),
    Claim("lr_factor", "kl", r"\|\s*8\s*\|\s*KL → LR controller\s*\|[^|]*×(?P<value>[\d.]+) per rollout",
          "§1 row 8"),
)


def doc_text() -> str:
    return pathlib.Path(repo_path(*_DOC)).read_text()


def parse_value(token: str) -> Any:
    t = token.strip().strip("`*\"' ").replace("−", "-")
    low = t.lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "on"):
        return True
    if low in ("false", "off"):
        return False
    for conv in (int, float):
        try:
            return conv(t)
        except ValueError:
            pass
    return t


def values_agree(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return bool(a == b)


class Violation(NamedTuple):
    key: str
    where: str
    doc_value: Any
    cfg_value: Any
    detail: str

    def line(self) -> str:
        return (f"  {self.key:24s} doc says {self.doc_value!r:12s} mirror says "
                f"{self.cfg_value!r:12s} — {self.where}" + (f"\n      {self.detail}" if self.detail else ""))


def check_claims(text: str, recipe: Dict[Tuple[str, str], Any],
                 claims: Tuple[Claim, ...] = _CLAIMS) -> List[Violation]:
    out: List[Violation] = []
    for c in claims:
        if (c.block, c.key) not in recipe:
            out.append(Violation(c.key, c.where, None, None,
                                 f"not a key of recipe.{c.block} at all — renamed?"))
            continue
        want = recipe[(c.block, c.key)]
        matches = list(re.finditer(c.pattern, text))
        if not matches:
            out.append(Violation(c.key, c.where, None, want,
                                 "the declared pattern matched NOTHING — the sentence was "
                                 "rewritten or removed; re-anchor the row"))
            continue
        for m in matches:
            got = parse_value(m.group("value"))
            if not values_agree(got, want):
                out.append(Violation(c.key, c.where, got, want, ""))
    return out


def _recipe(mirror: Any = None) -> Dict[Tuple[str, str], Any]:
    """`{(block, key): value}` through the READER (`recipe_blocks`), so a block the surface would
    refuse is refused here too."""
    fresh, fork = recipe_blocks(mirror)
    sizing = {r.dest for r in SIZING_ROWS}
    out: Dict[Tuple[str, str], Any] = {("sizing" if k in sizing else "fresh", k): v
                                       for k, v in fresh.items() if k != KL_KEY}
    out.update({("kl", k): v for k, v in fresh[KL_KEY].items()})
    out.update({("fork", k): v for k, v in fork.items()})
    return out


@_SKIP
def test_the_recipe_prose_agrees_with_the_production_mirror():
    violations = check_claims(doc_text(), _recipe())
    assert not violations, (
        "`designs/endstate/design_learner_recipe.md` disagrees with the production recipe "
        "(`designs/production_config.json`'s `recipe` block + the fields it reads). `--arch "
        "production` APPLIES the block, so a wrong number in either is a wrong run.\n\n"
        + "\n".join(v.line() for v in violations)
        + "\n\nChange the doc and the block TOGETHER, in one commit, with the source of the new "
          "value in §3.22's table and a Decision-record row.")


@_SKIP
def test_every_recipe_row_is_covered_by_a_claim():
    want = ({("fresh", r.dest) for r in FRESH_ROWS} | {("sizing", r.dest) for r in SIZING_ROWS}
            | {("kl", k) for k in KL_CONSTANTS}
            | {("fork", r.dest) for r in FORK_ROWS} | {("fork", d) for d in FORK_OVERRIDES})
    missing = sorted(want - {(c.block, c.key) for c in _CLAIMS})
    assert not missing, (
        "These recipe keys (`main.train.recipe_surface`) have no `_CLAIMS` row, so the gate says "
        "nothing about them:\n  " + "\n  ".join(f"recipe.{b}.{k}" for b, k in missing)
        + "\n\nAdd the knob to §3.22's table (with its source) and a claim here.")
    assert len(ALL_ROWS) == len(ROWS) + len(FORK_ROWS)


@_SKIP
def test_a_change_to_the_DOC_alone_FAILS_the_gate():
    """Reachability: each planted lie sits on a sentence a `_CLAIMS` row anchors on."""
    text = doc_text()
    recipe = _recipe()
    assert not check_claims(text, recipe), "the real document must be clean before planting"
    plants = (
        ("ent_coef", "| `ent_coef` | 0.05 | `recipe.fresh` |", "| `ent_coef` | 0.02 | `recipe.fresh` |"),
        ("n_envs", "| 1 | `n_envs` (N) | 256 |", "| 1 | `n_envs` (N) | 48 |"),
        ("clip_range_vf", "| `clip_range_vf` | none | `recipe.fresh` |",
         "| `clip_range_vf` | 0.5 | `recipe.fresh` |"),
        ("critic", "| `critic` | winprob | `recipe.fresh` |", "| `critic` | shaped | `recipe.fresh` |"),
        ("n_epochs", "| `n_epochs` | 10 | `recipe.fresh` |", "| `n_epochs` | 5 | `recipe.fresh` |"),
    )
    for key, original, lie in plants:
        assert original in text, f"the plant's anchor for {key!r} moved:\n  {original}"
        found = check_claims(text.replace(original, lie), recipe)
        assert any(v.key == key and v.doc_value is not None for v in found), (
            f"a planted lie on {key!r} did NOT fail the gate — its claim is decorative")


@_SKIP
def test_a_change_to_the_BLOCK_alone_FAILS_the_gate():
    """The other half: move a value in the mirror (in memory) and leave the doc — it must fail;
    rename a block key and the READER refuses (the rename tripwire)."""
    text = doc_text()
    raw = _raw_mirror()
    moved = copy.deepcopy(raw)
    moved["recipe"]["fresh"]["ent_coef"] = 0.02
    moved["recipe"]["fork"]["fork_lr"] = 2.8e-05
    moved["recipe"]["fresh"]["kl_controller"]["target_kl"] = 0.02
    found = {v.key for v in check_claims(text, _recipe(moved))}
    assert {"ent_coef", "fork_lr", "target_kl"} <= found, found
    epochs = copy.deepcopy(raw)
    epochs["recipe"]["fresh"]["n_epochs"] = 5          # E5's count leaking into the fresh block
    assert any(v.key == "n_epochs" for v in check_claims(text, _recipe(epochs)))
    renamed = copy.deepcopy(raw)
    renamed["recipe"]["fresh"]["entropy_coef"] = renamed["recipe"]["fresh"].pop("ent_coef")
    from main.train.recipe_surface import RecipeError
    with pytest.raises(RecipeError):
        _recipe(renamed)
