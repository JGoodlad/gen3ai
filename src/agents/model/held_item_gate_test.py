"""gen3_op_held_item_v1 — the guard on the class of the 2026-10-26 GIGO audit's C-1.

After Knock Off / Thief / Covet / Trick (``-enditem``) or a berry eaten, a mon's item block keeps the GONE item's
id with ``consumed = 1`` (``encoder/slot.rs::item``). The damage operator compared the RAW id with an item num, so a
knocked-off Choice Band read as HELD: P(their active holds CB) = 1.0 (the Smogon prior 0.79 when unrevealed, 0 for any
other revealed item), and our own knocked-off Band still multiplied our outgoing damage by 1.5. The Quick Claw reads
had the same spelling (latent: Quick Claw is banned in gen3ou).

THE RULE. In every policy-forward module (`selection_sites.FORWARD_MODULES`) an item's IDENTITY reaches the physics
only through `extractor_ctx.held_item_ids` (0 once consumed) and the prior branch only through
`extractor_ctx.item_unrevealed`: no subscript of a raw ``<x>.item_ids`` and no table indexed by it. The embeddings
read the raw id unsubscripted (``item_embedding(ctx.item_ids)``), beside its ``consumed`` bit. The allowlist is
EMPTY: a new raw read FAILS here, naming the line.

The behavioural half runs the PRODUCTION extractor on a real fixture row: a consumed item never moves the op's
block whatever its id, and a knocked-off Choice Band on their active reads P(CB) = 0.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.model.selection_sites import FORWARD_MODULES

_MODEL_DIR = Path(__file__).resolve().parent


def _raw_reads(tree: ast.AST) -> list:
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        # ctx.item_ids[...]  — a raw id read by position
        if isinstance(node.value, ast.Attribute) and node.value.attr == "item_ids":
            hits.append(node)
        # TABLE[ctx.item_ids] — a table lookup on the raw id
        elif isinstance(node.slice, ast.Attribute) and node.slice.attr == "item_ids":
            hits.append(node)
    return hits


def _violations() -> list:
    bad = []
    for mod in FORWARD_MODULES:
        tree = ast.parse((_MODEL_DIR / f"{mod}.py").read_text())
        bad += [f"{mod}.py:{n.lineno} {ast.unparse(n)}" for n in _raw_reads(tree)]
    return bad


def test_no_forward_module_reads_an_item_by_raw_id():
    bad = _violations()
    assert not bad, (
        "an item id is read RAW in the forward — a consumed (Knocked Off / stolen / eaten) item keeps its id with "
        "consumed = 1, so a raw read treats a GONE item as held. Read `extractor_ctx.held_item_ids` (identity, table "
        f"lookups) / `item_unrevealed` (the prior branch): {bad}")


def test_the_gate_sees_the_defect_it_guards():
    """The gate is not vacuous: the shipped spellings of the defect are flagged."""
    src = ("opp_item = ctx.item_ids[ar, opp_act]\n"
           "our_cb = (ctx.item_ids[:, our] == self.cb_item_num).float()\n"
           "p = self.IS_LEFTOVERS[ctx.item_ids]\n"
           "e = embeddings.item_embedding(ctx.item_ids)\n")
    assert len(_raw_reads(ast.parse(src))) == 3


# ----------------------------------------------------------------------------------------------------------------
# behaviour: the production extractor on a real fixture row
# ----------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _prod():
    from agents import gen3_data
    from agents.model.delivery_graph import build_extractor
    from utils.rust_env.fixture_battles import play_rows

    torch.manual_seed(0)
    fe, _cfg, _layout = build_extractor()
    obs, _mask = play_rows(1, seed=7, n_teams=4)[0]
    items = gen3_data.items.raw()
    return fe, obs, {k: items[k]["num"] for k in ("choiceband", "leftovers", "lumberry", "chestoberry")}


def _active_slot(row: np.ndarray, side: int) -> int:
    from agents.observation.constants import (OFFSET_OPP_TEAM, OFFSET_OUR_TEAM, POKEMON_ACTIVE_OFFSET,
                                              POKEMON_FULL_DIM, POKEMON_HP_OFFSET)
    base = OFFSET_OUR_TEAM if side == 0 else OFFSET_OPP_TEAM
    for s in range(6):
        b = base + s * POKEMON_FULL_DIM
        if row[b + POKEMON_ACTIVE_OFFSET] > 0.5 and row[b + POKEMON_HP_OFFSET] > 0:
            return b
    raise AssertionError("fixture row has no live active")


def _row_with_item(row: np.ndarray, b: int, block) -> np.ndarray:
    from agents.observation.constants import POKEMON_ITEMS_OFFSET
    out = row.copy()
    out[b + POKEMON_ITEMS_OFFSET:b + POKEMON_ITEMS_OFFSET + 3] = block
    return out


def _op(fe, row: np.ndarray):
    from agents.model.extra_obs_keys import zero_extra_obs
    obs = {"observation": torch.from_numpy(row[None].astype(np.float32)), **zero_extra_obs(fe, 1)}
    with torch.no_grad():
        fe(obs)
    op = fe.damage_op
    parts = [x.detach().reshape(-1) for x in (op.last_raw_block, op.last_out_cells) if x is not None]
    return op.last_raw_tensors, torch.cat(parts)


def _pick_row(obs: np.ndarray) -> np.ndarray:
    """The first row whose two actives are both alive (a precondition, asserted)."""
    for r in obs:
        try:
            _active_slot(r, 0), _active_slot(r, 1)
        except AssertionError:
            continue
        return r
    raise AssertionError("no fixture row with two live actives")


@pytest.mark.parametrize("side", [0, 1], ids=["ours", "theirs"])
def test_a_consumed_item_never_moves_the_op_whatever_its_id(_prod, side):
    fe, obs, nums = _prod
    row = _pick_row(obs)
    b = _active_slot(row, side)
    # every consumed id must read as the same "no item": the block of a consumed Leftovers is the reference
    _, ref = _op(fe, _row_with_item(row, b, [nums["leftovers"], 1.0, 1.0]))
    for name in ("choiceband", "lumberry", "chestoberry"):
        _, got = _op(fe, _row_with_item(row, b, [nums[name], 1.0, 1.0]))
        assert torch.equal(got, ref), f"a consumed {name} on {('our', 'their')[side]} active moves the op block"
    # and the precondition that makes this test non-vacuous: a HELD Band does move it
    _, held = _op(fe, _row_with_item(row, b, [nums["choiceband"], 1.0, 0.0]))
    assert not torch.equal(held, ref), "a held Choice Band does not move the op block: the test cannot see the class"


def test_a_knocked_off_band_on_their_active_reads_p_cb_zero(_prod):
    fe, obs, nums = _prod
    row = _pick_row(obs)
    b = _active_slot(row, 1)
    t_gone, _ = _op(fe, _row_with_item(row, b, [nums["choiceband"], 1.0, 1.0]))
    t_held, _ = _op(fe, _row_with_item(row, b, [nums["choiceband"], 1.0, 0.0]))
    t_unrev, _ = _op(fe, _row_with_item(row, b, [0.0, 0.0, 0.0]))
    assert float(t_gone.p_cb[0, 0]) == 0.0
    assert float(t_held.p_cb[0, 0]) == 1.0
    assert 0.0 < float(t_unrev.p_cb[0, 0]) < 1.0      # the species' Smogon prior (the fixture's mon)
