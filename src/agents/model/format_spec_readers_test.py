"""The board-state clauses' model readers ask the FORMAT SPEC, never a constant (design_format_spec.md §5.2).

Sleep Clause Mod and Freeze Clause Mod are in force in gen3ou, so production is unchanged; a PLANTED spec
without the clause (``format_spec.without_rule``) must lift every gate that reads it — the op's outgoing and
incoming status landing, the shared incoming mask, and the move-resolution family's Yawn and switch-branch
sleep gates. Each assertion below fails on revert of its reader (a hard-coded clause ignores the planted spec).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from agents.gen3_data import format_spec
from agents.model import status_rules
from agents.model.damage_op_layout import _COND_FRZ_IDX, _COND_SLP_IDX
from agents.model.damage_op_test import _T2I, _status_land
from agents.model.move_resolution_tables import C_FRZ, C_SLP
from agents.model.move_resolution_test import _status_grid, cond, mf, ops, sf
from agents.model.op_status_rules_test import _incoming, op  # noqa: F401  (the module-scoped fixture)
from agents.observation.constants import POKEMON_CONDITION_OFFSET, POKEMON_SLEEP_BELIEF_OFFSET


@pytest.fixture
def no_sleep_clause(monkeypatch):
    monkeypatch.setattr(format_spec, "_ACTIVE", format_spec.without_rule(format_spec.GEN3OU, "sleepclausemod"))
    assert not format_spec.sleep_clause_mod() and format_spec.freeze_clause_mod()


@pytest.fixture
def no_freeze_clause(monkeypatch):
    monkeypatch.setattr(format_spec, "_ACTIVE", format_spec.without_rule(format_spec.GEN3OU, "freezeclausemod"))
    assert format_spec.sleep_clause_mod() and not format_spec.freeze_clause_mod()


def test_gen3ou_has_both_board_state_clauses():
    assert format_spec.sleep_clause_mod() and format_spec.freeze_clause_mod()
    for rid in ("sleepclausemod", "freezeclausemod"):
        assert format_spec.GEN3OU.rule(rid).story is format_spec.Story.BOARD_STATE


def _mask(slp_slot=None, frz_slot=None):
    z = torch.zeros(1, 6)
    slp, frz = z.clone(), z.clone()
    if slp_slot is not None:
        slp[0, slp_slot] = 1.0
    if frz_slot is not None:
        frz[0, frz_slot] = 1.0
    m = status_rules.incoming_status_mask(torch.zeros(1), slp, frz, torch.ones(1, 6), z, torch.zeros(1),
                                          torch.zeros(1, dtype=torch.long))
    return m[0, :, status_rules.COL_SLP], m[0, :, status_rules.COL_FRZ]


def test_incoming_mask_reads_the_spec(monkeypatch):
    assert _mask(slp_slot=3)[0].tolist() == [0.0] * 6                       # production: Sleep Clause blocks
    assert _mask(frz_slot=2)[1].tolist() == [0.0] * 6                       # production: Freeze Clause blocks
    monkeypatch.setattr(format_spec, "_ACTIVE", format_spec.without_rule(format_spec.GEN3OU, "sleepclausemod"))
    assert _mask(slp_slot=3)[0].tolist() == [1.0] * 6
    assert _mask(frz_slot=2)[1].tolist() == [0.0] * 6                       # the other clause is untouched
    monkeypatch.setattr(format_spec, "_ACTIVE", format_spec.without_rule(format_spec.GEN3OU, "freezeclausemod"))
    assert _mask(frz_slot=2)[1].tolist() == [1.0] * 6
    assert _mask(slp_slot=3)[0].tolist() == [0.0] * 6


def _our_sleeper(ctx):
    ctx.pokemon_part[:, 3, POKEMON_CONDITION_OFFSET + _COND_SLP_IDX] = 1.0
    ctx.pokemon_part[:, 3, POKEMON_SLEEP_BELIEF_OFFSET] = 0.0


def _our_frozen(ctx):
    ctx.pokemon_part[:, 4, POKEMON_CONDITION_OFFSET + _COND_FRZ_IDX] = 1.0


def test_op_incoming_sleep_reads_the_spec(op, no_sleep_clause):  # noqa: F811
    assert _incoming(op, "spore", setup=_our_sleeper)[0].item() == pytest.approx(1.0)


def test_op_incoming_sleep_production(op):  # noqa: F811
    assert _incoming(op, "spore", setup=_our_sleeper)[0].item() == 0.0


def test_op_incoming_freeze_reads_the_spec(op, no_freeze_clause):  # noqa: F811
    assert _incoming(op, "icebeam", setup=_our_frozen)[0].item() == pytest.approx(0.1, abs=1e-4)


def test_op_incoming_freeze_production(op):  # noqa: F811
    assert _incoming(op, "icebeam", setup=_our_frozen)[0].item() == 0.0


def _outgoing_sleep(op):  # noqa: F811
    y = ("hypnosis", "growl", "growl", "growl")
    return _status_land(op, our_moves=y, opp_t1=_T2I["NORMAL"], opp_t2=0, bench_sleep=True)[0][0].item()


def test_op_outgoing_sleep_reads_the_spec(op, no_sleep_clause):  # noqa: F811
    assert _outgoing_sleep(op) > 0.5


def test_op_outgoing_sleep_production(op):  # noqa: F811
    assert _outgoing_sleep(op) == 0.0


def test_move_resolution_yawn_and_switch_branch_read_the_spec(monkeypatch):
    sleeper = cond(3, C_SLP)
    asleep_active = cond(0, C_SLP)
    assert mf(ops(("yawn",), opp_cond=sleeper)) == 0.0                                   # production
    assert mf(ops(("hypnosis",), opp_cond=asleep_active), "p_lands_switch") == 0.0
    monkeypatch.setattr(format_spec, "_ACTIVE", format_spec.without_rule(format_spec.GEN3OU, "sleepclausemod"))
    assert mf(ops(("yawn",), opp_cond=sleeper)) == pytest.approx(1.0)
    assert mf(ops(("hypnosis",), opp_cond=asleep_active), "p_lands_switch") == pytest.approx(0.6)


def test_move_resolution_incoming_clauses_read_the_spec(monkeypatch):
    def p(move, col, our_cond, coord):
        o = ops(("tackle",), seats=(move, "growl", "growl"), pair_in=_status_grid(col), our_cond=our_cond)
        return sf(o, coord, 2)
    assert p("spore", 3, cond(4, C_SLP), "p_slp") == 0.0
    assert p("icebeam", 2, cond(5, C_FRZ), "p_frz") == 0.0
    monkeypatch.setattr(format_spec, "_ACTIVE", format_spec.without_rule(format_spec.GEN3OU, "sleepclausemod"))
    assert p("spore", 3, cond(4, C_SLP), "p_slp") > 0.0
    monkeypatch.setattr(format_spec, "_ACTIVE", format_spec.without_rule(format_spec.GEN3OU, "freezeclausemod"))
    assert p("icebeam", 2, cond(5, C_FRZ), "p_frz") > 0.0


#: Every model module that computes a board-state clause gate, and the spec reader each must call.
CLAUSE_READERS = {
    "status_rules.py": ("sleep_clause_mod", "freeze_clause_mod"),
    "damage_op_blocks.py": ("sleep_clause_mod",),
    "move_resolution.py": ("sleep_clause_mod",),
}


def test_every_clause_site_calls_the_spec_reader():
    """A clause gate must be one the spec can switch: each module's `format_spec.<reader>()` calls are present,
    one per clause tensor it builds (status_rules 2, damage_op_blocks 1, move_resolution 2)."""
    here = Path(__file__).parent
    want = {"status_rules.py": 2, "damage_op_blocks.py": 1, "move_resolution.py": 2}
    for fname, readers in CLAUSE_READERS.items():
        tree = ast.parse((here / fname).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == "format_spec"
                 and n.func.attr in readers]
        assert len(calls) == want[fname], (fname, len(calls))
