"""OUR rows are read off the REAL context: the compile-safe spelling is NUMERICALLY IDENTICAL (2026-11-11).

`damage_kinds.our_team_spread(ctx)` replaced ``ctx.pokemon_part[:, :TEAM_SIZE, SPREAD]`` at the five places the op
reads the whole team's spread. On a hypothesis context the two name the same memory's values (the substitution
touches opponent rows only), but the old one read OUR rows THROUGH the block's cat, and that kernel shape is what
Triton 3.4.0 fails to compile at ``XBLOCK = 16`` (`triton_xblock16_cat_read_test.py`: the 2026-11-11 red of the
CUDA forward + backward cell). The fix may change no number, so this file pins exactly that, on the production
extractor in eager mode:

* the READ: `our_rows` / `our_team_spread` on a hypothesis context equal the block read, bit for bit;
* the OP: with every site swapped back to the old spelling, the damage op's outputs (its block, the enriched
  opponent tokens, every tensor either stash holds) and the extractor's outputs are bit-identical;
* the SITES: no full-team spread read through ``ctx.pokemon_part`` is left in the model's source.

Each test fails on the revert it names.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Any, Dict

import pytest
import torch

from agents.model import damage_op, damage_op_blocks, damage_op_pairwise, damage_op_speed, eot_residual
from agents.model.damage_kinds import our_team_spread
from agents.model.extractor_ctx import our_rows
from agents.model.hypothesis_tokens import hypothesis_ctx
from agents.observation.constants import POKEMON_SPREAD_DIM, POKEMON_SPREAD_OFFSET, TEAM_SIZE
from utils.paths import src_path

pytestmark = pytest.mark.integration

# The five modules that read the whole team's spread: every one imports `our_team_spread` by name.
_SITES = (damage_op, damage_op_blocks, damage_op_pairwise, damage_op_speed, eot_residual)
_SPREAD = slice(POKEMON_SPREAD_OFFSET, POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM)


def _old_spelling(ctx: Any) -> torch.Tensor:
    """The read as it was written before 2026-11-11: through the context's own block."""
    return ctx.pokemon_part[:, :TEAM_SIZE, _SPREAD]


def _tensors(obj: Any, prefix: str, out: Dict[str, torch.Tensor], depth: int = 0) -> Dict[str, torch.Tensor]:
    """Every tensor reachable from a stash (dataclasses, tuples, lists, dicts), by path."""
    if isinstance(obj, torch.Tensor):
        out[prefix] = obj
    elif depth > 4 or obj is None:
        pass
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            _tensors(getattr(obj, f.name), f"{prefix}.{f.name}", out, depth + 1)
    elif isinstance(obj, (tuple, list)):
        for i, v in enumerate(obj):
            _tensors(v, f"{prefix}[{i}]", out, depth + 1)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _tensors(v, f"{prefix}[{k!r}]", out, depth + 1)
    return out


def _run(fe, obs) -> Dict[str, torch.Tensor]:
    """One eager forward; the extractor's outputs, `_spread_hp_damage`'s return and both stashes, by name."""
    seen: Dict[str, torch.Tensor] = {}
    inner = fe._spread_hp_damage

    def spy(*a, **k):
        opp_role, block = inner(*a, **k)
        seen["op.opp_role"] = opp_role
        assert block is not None, "the production extractor runs the damage op"
        seen["op.damage_block"] = block
        return opp_role, block

    fe._spread_hp_damage = spy
    try:
        with torch.no_grad():
            out = fe(obs)
    finally:
        del fe._spread_hp_damage
    for i, t in enumerate(out):
        seen[f"out[{i}]"] = t
    _tensors(fe.stash, "fe.stash", seen)
    _tensors(fe.damage_op.stash, "op.stash", seen)
    return {k: v.detach().clone() for k, v in seen.items()}


@pytest.fixture(scope="module")
def prod():
    from agents.model.extractor_compiles_test import _build_production_extractor
    from agents.model.hypothesis_set_test import _obs_from_golden
    fe, layout = _build_production_extractor()
    assert fe.hypothesis_builder is not None and fe.damage_op is not None
    obs = _obs_from_golden(16)      # real states: hidden opponent slots, so the hypothesis block substitutes rows
    return fe, layout, obs


def test_our_rows_on_a_hypothesis_context_are_the_block_read(prod):
    """Reverting `hypothesis_ctx`'s ``real=`` to anything but the context it was derived from fails here."""
    fe, layout, obs = prod
    with torch.no_grad():
        fe(obs)
    ctx = fe.unpack(obs)
    hs = fe.last_hypothesis
    hctx = hypothesis_ctx(ctx, hs, layout)
    assert bool(hs.slot_is_hypothesis.any()), "no hypothesis slot: the block's cat substitutes nothing here"
    assert not torch.equal(hctx.pokemon_part, ctx.pokemon_part), "the substitution changed no row"
    assert ctx.real is None and hctx.real is ctx
    assert hypothesis_ctx(hctx, hs, layout).real is ctx, "a context derived twice still names the REAL one"
    for c in (ctx, hctx):
        assert torch.equal(our_rows(c), c.pokemon_part[:, :TEAM_SIZE])
        assert torch.equal(our_team_spread(c), _old_spelling(c))
    assert float(our_team_spread(hctx).abs().sum()) > 0, "an all-zero spread block would make equality vacuous"


def test_the_damage_ops_outputs_are_bit_identical_under_the_old_spelling(prod, monkeypatch):
    """THE PROOF OF IDENTITY. Fails if `our_team_spread` ever returns anything but the block read."""
    fe, _layout, obs = prod
    new = _run(fe, obs)
    calls = {m.__name__: 0 for m in _SITES}
    for m in _SITES:
        assert m.our_team_spread is our_team_spread, f"{m.__name__} no longer reads `our_team_spread`"

        def old(ctx, _name=m.__name__):
            calls[_name] += ctx.real is not None      # counted on a HYPOTHESIS context only: where the cat is
            return _old_spelling(ctx)

        monkeypatch.setattr(m, "our_team_spread", old)
    old_out = _run(fe, obs)
    # the swap is REAL: the production forward reaches every one of the five reads on a hypothesis context
    assert all(n > 0 for n in calls.values()), calls
    assert set(new) == set(old_out) and len(new) > 20, sorted(set(new) ^ set(old_out))
    assert "op.damage_block" in new and new["op.damage_block"].numel() > 0
    differing = [k for k in new if not _same(new[k], old_out[k])]
    assert not differing, f"the two spellings differ in {len(differing)} of {len(new)} tensors: {differing[:8]}"


def test_the_comparison_can_see_a_changed_spread(prod, monkeypatch):
    """The control for the proof above: a spread that IS different changes the op's block, so `torch.equal` over
    these tensors is a live comparison (a stale `real` would be caught) and not an identity nothing can fail."""
    fe, _layout, obs = prod
    new = _run(fe, obs)
    for m in _SITES:
        monkeypatch.setattr(m, "our_team_spread", lambda ctx: _old_spelling(ctx).flip(1))   # OUR mons permuted
    moved = _run(fe, obs)
    assert not _same(new["op.damage_block"], moved["op.damage_block"])
    assert not all(_same(new[f"out[{i}]"], moved[f"out[{i}]"]) for i in range(2))


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-identical, with NaN equal to NaN at the same place (a stash may carry a masked NaN)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.is_floating_point():
        return bool(((a == b) | (a.isnan() & b.isnan())).all())
    return bool(torch.equal(a, b))


def test_no_full_team_spread_read_goes_through_the_block():
    """A new ``ctx.pokemon_part[:, :TEAM_SIZE, POKEMON_SPREAD_OFFSET...]`` read fails here: spell it
    `our_team_spread(ctx)`. (A gather of ONE mon's row, ``ctx.pokemon_part[ar, idx, ...]``, is a different kernel
    shape and is not this pin's subject.)"""
    pat = re.compile(r"pokemon_part\[\s*:\s*,\s*:\s*TEAM_SIZE\s*,\s*POKEMON_SPREAD_OFFSET")
    hits = []
    for path in sorted(src_path("agents", "model").glob("*.py")):
        if path.name.endswith("_test.py"):
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if pat.search(line):
                hits.append(f"{path.name}:{n}")
    assert not hits, f"a full-team spread read through the context's block: {hits}"
    assert pat.search("x = ctx.pokemon_part[:, :TEAM_SIZE, POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + 6]")
