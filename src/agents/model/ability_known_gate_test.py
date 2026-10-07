"""gen3_op_ability_known_v1 — the STATIC guard on the class of the 2026-10-07 op GIGO.

An unrevealed opponent's ability slot carries its species' MOST LIKELY ability in ``id1`` with ``known = 0``
(`observation/abilities.py`). The damage operator read ``ability1_ids > 0`` as "revealed" and indexed its
ability tables by the raw id, so it asserted the top-1 prior ability as CERTAIN: Toxic "never landed" on an
unrevealed Snorlax (Smogon: Immunity 0.86 / Thick Fat 0.14), Levitate was a certain Ground immunity on every
unrevealed slot whose top-1 it is, and so on through the damage, status, secondary and pairwise kernels.

THE RULE. In every policy-forward module (`selection_sites.FORWARD_MODULES`), a subscript of
``<x>.ability1_ids`` may only take OUR side — the six-slot prefix or our active — whose abilities the team sheet
makes certain. An opponent's ability reaches the physics ONLY through `extractor_ctx.revealed_ability1_ids` /
`ability_known` (the op's `opp_ability_view` and `_known_or_prior`). The allowlist is EMPTY: a new opponent read
by raw id FAILS here, naming the line.
"""
from __future__ import annotations

import ast
from pathlib import Path

from agents.model.selection_sites import FORWARD_MODULES

_MODEL_DIR = Path(__file__).resolve().parent
#: The slice spellings that address OUR six (the first TEAM_SIZE slots) or our active.
_OUR_SIDE = frozenset({
    ":, :TEAM_SIZE", "ar, our_act", "ar, ctx.our_active_idx", ":, 0:TEAM_SIZE", ":, ours",
})


def _violations() -> list[str]:
    bad = []
    for mod in FORWARD_MODULES:
        path = _MODEL_DIR / f"{mod}.py"
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "ability1_ids"):
                sl = ast.unparse(node.slice).removeprefix("(").removesuffix(")")
                if sl not in _OUR_SIDE:
                    bad.append(f"{mod}.py:{node.lineno} {ast.unparse(node)}")
    return bad


def test_no_forward_module_reads_an_opponent_ability_by_raw_id():
    bad = _violations()
    assert not bad, (
        "an ability id is read by RAW id outside our side — an unrevealed opponent's id1 is its species' top-1 "
        "PRIOR, not a reveal. Read it through `extractor_ctx.revealed_ability1_ids` / `ability_known` (the op's "
        f"`opp_ability_view` + `_known_or_prior`): {bad}")


def test_the_gate_sees_the_defect_it_guards():
    """The gate is not vacuous: the shipped spelling of the defect is flagged."""
    tree = ast.parse("opp_ability = ctx.ability1_ids[ar, opp_act]\nx = ctx.ability1_ids[:, TEAM_SIZE:2 * TEAM_SIZE]")
    hits = [ast.unparse(n.slice).removeprefix("(").removesuffix(")") for n in ast.walk(tree)
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute) and n.value.attr == "ability1_ids"]
    assert hits and not any(h in _OUR_SIDE for h in hits)
