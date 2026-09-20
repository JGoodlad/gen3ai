"""EVENT-FOLD parity — ``ViewEventFolder``'s ply events vs ``Gen3Battle``'s, on the same bytes.

``gen3_view_successor_v1``. The full-observation half of
``one_sided_view_parity_fuzz_test.py`` already proves the two roads' vectors agree, which
implies the events agree. This gate exists anyway, and the reason is DIAGNOSIS: a tracker block
is a fold of a fold, so "the pair-history block differs at index 1732" and "the fold dropped the
``|-activate|`` line" are the same failure wearing very different clothes. Comparing the EVENTS
names it at the source, field by field, with a census keyed by ``<kind>.<field>``.

Two halves, the project's standard split:

* the **unit** half pins the light board's three parsers, which are the only genuinely new code
  in the module (everything else is ``Gen3Battle._build_event``, inherited unchanged);
* the **differential** half drives real gen3ou search plies. Collected, it runs the REPRODUCIBLE
  fixture battle; as a script it takes fresh random ones.

    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/battle/event_fold_parity_fuzz_test.py [n_battles] [--arms K]
    pytest -m sim src/agents/battle/event_fold_parity_fuzz_test.py
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections import Counter
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

from agents.battle.event_fold import (ViewEventFolder, _norm_ident, _split_condition)
from utils.bridge.search_session import SearchSession

pytestmark = [pytest.mark.sim, pytest.mark.integration]

#: The ``BattleEvent`` fields compared. ``raw`` is included on purpose — it is the line verbatim,
#: so a disagreement there means the two folds did not even look at the same set of lines, which
#: is a different (and worse) finding than a mis-built value.
_FIELDS = ("seq", "turn", "kind", "side", "actor_species", "target_species", "value", "raw")


# ---------------------------------------------------------------------------
# Unit: the light board's parsers
# ---------------------------------------------------------------------------

def test_the_condition_field_parses_to_poke_envs_own_hp_fraction():
    assert _split_condition("100/100") == (1.0, None)
    assert _split_condition("52/100 par") == (0.52, "PAR")
    # OUR side states true integers; the fraction is the same quantity either way.
    assert _split_condition("123/321")[0] == pytest.approx(123 / 321)
    # A faint clears the status, exactly as `Pokemon.faint` does.
    assert _split_condition("0 fnt") == (0.0, None)
    assert _split_condition("0") == (0.0, None)


def test_the_ident_normalisation_is_poke_envs_own():
    assert _norm_ident("p1a: Blissey") == "p1: Blissey"
    assert _norm_ident("p2b: Snorlax") == "p2: Snorlax"
    assert _norm_ident("p1: Blissey") == "p1: Blissey"          # already normalised


def test_the_species_comes_from_the_details_string_not_the_nickname():
    # poke-env reads the species off DETAILS; the ident's name half is the NICKNAME and is not
    # a species at all. Getting this backwards renames every nicknamed mon.
    assert ViewEventFolder._species_from_details("Blissey, F") == "blissey"
    assert ViewEventFolder._species_from_details("Deoxys-Speed, L100") == "deoxysspeed"
    assert ViewEventFolder._species_from_details("") is None


# ---------------------------------------------------------------------------
# Differential: real search plies
# ---------------------------------------------------------------------------

def run(n_battles: int = 2, arms: int = 5, turns: int = 3, impl: str = "rust",
        fixed_key: Optional[int] = None) -> Tuple[Counter, Dict[str, Any], int]:
    import agents.battle.one_sided_view_parity_fuzz_test as G

    cen: Counter = Counter()
    examples: Dict[str, Any] = {}
    plies = 0
    for b in range(n_battles):
        with tempfile.TemporaryDirectory() as td:
            record, summary, npz = G._record_one_battle(
                td, impl, None if fixed_key is None else fixed_key + b)
        actions = np.asarray(npz["actions"], dtype=int)
        invs = summary["invocations"]
        cand = [i for i, inv in enumerate(invs)
                if inv.get("phase") == "move_selection" and int(inv["turn"]) > 1]
        if not cand:
            continue
        side = record.side_of(record.trainee_username)
        other = "p2" if side == "p1" else "p1"
        with SearchSession(record, impl=impl) as ss:
            for anchor in [cand[int(len(cand) * f)] for f in (0.25, 0.5, 0.75)][:turns]:
                turn = int(invs[anchor]["turn"])
                try:
                    root = ss.open_root(turn)
                except Exception:                                  # noqa: BLE001
                    continue
                pfx = root.prefix_p1_chunks if side == "p1" else root.prefix_p2_chunks
                prefix_actions = [int(x) for x in actions[:anchor]]
                cmap = G._choice_map(record, side, prefix_actions, pfx, anchor)
                if not cmap:
                    continue
                opp_rec = root.recorded_choices.get(other) or "default"
                picks = sorted(cmap)[:arms]
                expand = [{"node_id": root.node_id, f"{side}_action": cmap[a],
                           f"{other}_action": opp_rec,
                           "seed": f"{turn},{k + 1},{anchor + 7},{k * 13 + 11}", "label": a}
                          for k, a in enumerate(picks)]
                for node in ss.expand_many(expand):
                    if node.ended or node.stuck:
                        continue
                    suffix = node.p1_chunks if side == "p1" else node.p2_chunks
                    road = G._ProtocolRoad(
                        record, side, prefix_actions + [int(node.label)] + [0] * 8)
                    road.feed(pfx)
                    battle = road.battle
                    if battle is None:
                        continue
                    # Seed BEFORE the protocol road advances, then feed the same bytes to both.
                    folder = ViewEventFolder.seed_from(battle)
                    cursor = battle.event_cursor
                    road.feed(suffix)
                    theirs = battle.events_since(cursor)
                    ours = folder.fold(suffix)
                    plies += 1
                    if len(theirs) != len(ours):
                        cen["[event COUNT]"] += 1
                        examples.setdefault("[event COUNT]", (
                            [e.kind.name for e in theirs], [e.kind.name for e in ours]))
                        continue
                    for t, o in zip(theirs, ours):
                        for f in _FIELDS:
                            x, y = getattr(t, f), getattr(o, f)
                            if x != y:
                                key = f"{t.kind.name}.{f}"
                                cen[key] += 1
                                examples.setdefault(key, (x, y, t.raw))
    return cen, examples, plies


def render(cen: Counter, examples: Dict[str, Any], plies: int) -> str:
    if not cen:
        return f"✅ no event divergence over {plies} plies"
    lines = [f"❌ {sum(cen.values())} divergences in {len(cen)} classes over {plies} plies"]
    for key, n in cen.most_common():
        lines.append(f"   {n:6d}  {key}")
        lines.append(f"           e.g. {examples[key]}")
    return "\n".join(lines)


def test_the_light_fold_reproduces_gen3battles_events_on_a_real_ply():
    """The collected gate — the REPRODUCIBLE fixture battle, so a failure is a regression."""
    cen, examples, plies = run(n_battles=1, arms=5, turns=3, fixed_key=0)
    print("\n" + render(cen, examples, plies))
    assert plies >= 8, f"only {plies} plies folded — the gate is vacuous"
    assert not cen, "\n" + render(cen, examples, plies)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("n_battles", nargs="?", type=int, default=2)
    ap.add_argument("--arms", type=int, default=5)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--impl", default="rust")
    ap.add_argument("--fixed-key", type=int, default=None)
    a = ap.parse_args()
    c, ex, n = run(a.n_battles, a.arms, a.turns, a.impl, a.fixed_key)
    print(render(c, ex, n))
    sys.exit(1 if c else 0)
