"""A depth-2 successor must be replayed with EVERY ply on its path — the chunk-gap defect
(``gen3_search_depth2_chunk_gap_v1``).

``SearchSession.expand_many`` returns the arm's OWN ply, and for a while
:class:`~utils.bridge.search_session.ExpandedNode` documented it as "the COMPLETE one-sided view,
root → this node". :mod:`main.search_dividend.search` believed the docstring and fed the bare
suffix to the materializer alongside an ``actions`` list spanning the whole path, so a depth-``d``
successor was replayed as ``prefix`` (ending at the root request) followed by ply ``d``, with plies
``1..d-1`` missing. At depth 1 the two readings coincide, which is why nothing caught it until the
first depth-2 run.

**What the hole does, and why the two reported symptoms are ONE bug.** poke-env keeps applying
protocol lines to the board it last saw, so:

* a switch inside the gap makes every later reference to that mon log
  ``"Message thinks p1: X is active, but it's not"`` — tens of thousands of them in the live run;
* an opponent REVEAL inside the gap means the ``|switch|`` carrying the species never arrived, so a
  later ``|move|p2a: <nick>|…`` reaches ``get_pokemon`` with no ``details`` and it constructs a
  Pokémon whose *species* is the NICKNAME — ``KeyError: to_id_str(nickname)``.

⚠️ **The reported ``KeyError: 'ptãra'`` made this look like an encoding defect and it is not.**
``'ptãra'`` is ``to_id_str("PtÃ©ra")``, and the ``Ã©`` is a real double-encode that lives in the
COMMITTED TEAM FILE — ``data/teams/others/mcmegan/*.txt`` hold the bytes ``c3 83 c2 a9`` where
``Ptéra`` was meant — not in the chunk transport, which round-trips it faithfully on both impls.
Any nickname raises here: measured on the same fixture, ``'airmure'`` and ``'tyranocif'`` fail
identically, and the only slot that SURVIVED the gap was the one whose nickname equals its species.
So this test asserts on a nickname, not on a non-ASCII one.

The tier is ``sim`` and not ``slow``: ONE battle, per the root ``CLAUDE.md`` cost rule.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import sys
from typing import List, Optional, Tuple

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from poke_env import AccountConfiguration  # noqa: E402
from poke_env.data.normalize import to_id_str  # noqa: E402

from agents.training.obs_materializer import materialize_decisions  # noqa: E402
from utils.bridge.local_battle_runner import run_local_battles  # noqa: E402
from utils.bridge.reconstruction import pop_record  # noqa: E402
from utils.bridge.search_session import SearchSession  # noqa: E402
from utils.paths import repo_path  # noqa: E402
from utils.team_loader.loader import TeamLoader  # noqa: E402

pytestmark = pytest.mark.sim

BATTLE_FORMAT = "gen3ou"
_SEED = [11, 22, 33, 44]
_TURN = 3
_IMPLS = ("node", "rust")
#: A committed pool team whose mons carry NICKNAMES. The defect needs a nickname that differs from
#: its species (a mon called "Jirachi" survives the gap), and this file has five of them.
_NICKNAMED_TEAM = repo_path("data", "teams", "others", "mcmegan", "7fda48d98ca8efdc.txt")


class _WarningSpy:
    """poke-env logs the active-mismatch on the per-battle logger, which the replay player builds
    itself — so the capture is on ``Logger.warning`` rather than on a handler we could attach."""

    def __init__(self) -> None:
        self.msgs: List[str] = []
        self._orig = logging.Logger.warning

    def __enter__(self) -> "_WarningSpy":
        spy = self

        def _warning(inner_self, msg, *args, **kwargs):
            try:
                spy.msgs.append(str(msg) % args if args else str(msg))
            except Exception:                                   # noqa: BLE001
                spy.msgs.append(str(msg))

        logging.Logger.warning = _warning                       # type: ignore[method-assign]
        return self

    def __exit__(self, *exc) -> None:
        logging.Logger.warning = self._orig                     # type: ignore[method-assign]

    @property
    def active_mismatches(self) -> List[str]:
        return [m for m in self.msgs if "is active, but it's not" in m]


def _recorded_battle():
    """ONE seeded bridge battle: our fixed pool team vs the NICKNAMED team.

    Recorded on node so the fixture is impl-neutral (a record produced by the impl under test
    could hide a symmetric defect), exactly as ``search_driver_turn1_integration_test`` does.
    """
    from poke_env.player import RandomPlayer

    random.seed(5)
    ours = TeamLoader().get_all_teams()[0]
    theirs = _NICKNAMED_TEAM.read_text(encoding="utf-8")
    p1 = RandomPlayer(battle_format=BATTLE_FORMAT, team=ours, start_listening=False,
                      account_configuration=AccountConfiguration("D2GapA", "password"))
    p2 = RandomPlayer(battle_format=BATTLE_FORMAT, team=theirs, start_listening=False,
                      account_configuration=AccountConfiguration("D2GapB", "password"))
    asyncio.run(run_local_battles(p1, p2, 1, battle_format=BATTLE_FORMAT,
                                  seed=_SEED, impl="node"))
    battle = next(iter(p1._battles.values()))
    # NON-VACUITY: every assertion below needs turn _TURN to be a real joint move decision with at
    # least two more turns after it, or the "gap" it probes would not exist.
    assert battle.finished and battle.turn > _TURN + 1, (
        f"degenerate fixture battle (finished={battle.finished}, turn={battle.turn})")
    record = pop_record(battle.battle_tag)
    assert record is not None, "the bridge emitted no __RECON__ — nothing to replay"
    return record


@pytest.fixture(scope="module")
def record():
    return _recorded_battle()


def _nicknamed_bench_slots(p2_request: dict) -> List[Tuple[int, str]]:
    """Every benched opponent slot whose NICKNAME differs from its species, ``(1-based, nick)``.

    Read off the request rather than hardcoded, and returned as a LIST rather than a pick: the
    fixture battle is seeded but two players share the global ``random``, so which mons are benched
    at turn ``_TURN`` is not something to assert on. The caller tries slots until one produces a
    usable two-ply pair, which makes the test depend only on the committed team carrying a
    nickname at all. A mon whose nickname EQUALS its species (``Jirachi`` here) is excluded because
    it survives the gap — it is the control, not a case.
    """
    out: List[Tuple[int, str]] = []
    for i, mon in enumerate(p2_request.get("side", {}).get("pokemon", []), start=1):
        if mon.get("active") or "fnt" in str(mon.get("condition", "")):
            continue
        nick = str(mon.get("ident", "")).split(": ", 1)[-1]
        species = str(mon.get("details", "")).split(", ")[0]
        if nick and to_id_str(nick) != to_id_str(species):
            out.append((i, nick))
    assert out, "fixture team carries no benched NICKNAMED mon — the probe is vacuous"
    return out


def _materialize(chunks, record) -> Tuple[Optional[str], int, int]:
    """``(error, n_decisions, n_active_mismatch_warnings)`` for OUR side's ``chunks``."""
    with _WarningSpy() as spy:
        try:
            trace = materialize_decisions(
                list(chunks), username=record.username("p1"),
                packed_team=record.packed_team("p1"), side="p1",
                # Long enough that the replay is never cut short by an exhausted action list —
                # the action VALUES only advance the tracker's history, never the protocol.
                actions=[0] * 60, battle_format=BATTLE_FORMAT)
        except Exception as e:                                  # noqa: BLE001
            return f"{type(e).__name__}: {e}", 0, len(spy.active_mismatches)
        return None, len(trace.decisions), len(spy.active_mismatches)


@pytest.mark.parametrize("impl", _IMPLS)
def test_a_depth2_successor_replays_clean_only_with_every_ply_on_its_path(record, impl):
    """THE REGRESSION, at the semantics. Pre-fix, ``prefix + <ply 2 only>`` is what
    ``_expand_ply`` handed the materializer; post-fix it hands ``prefix + ply1 + ply2``.

    Both compositions are built here from the SAME driver output, so the assertion is about the
    composition rule and nothing else — and it is impl-symmetric because the rule is Python-side
    and shared, which is exactly why both impls were affected.
    """
    with SearchSession(record, impl=impl) as ss:
        root = ss.open_root(_TURN)
        prefix = list(root.prefix_p1_chunks)
        tried: List[str] = []
        chosen = None
        for slot, nick in _nicknamed_bench_slots(root.requests["p2"]):
            # Ply 1: the opponent switches its nicknamed mon in. That |switch| — the ONLY line
            # that ever carries its species — is what the gap swallows.
            ply1 = ss.expand_many([{"node_id": root.node_id, "p1_action": "default",
                                    "p2_action": f"switch {slot}", "seed": "5,5,5,5",
                                    "label": 0}])[0]
            if not ply1.node_id:
                tried.append(f"{nick}: ply 1 ended the battle")
                continue
            joined1 = "\n".join(ply1.p1_chunks)
            if f"|switch|p2a: {nick}|" not in joined1:
                tried.append(f"{nick}: never revealed at ply 1")
                continue
            ply2 = ss.expand_many([{"node_id": ply1.node_id, "p1_action": "default",
                                    "p2_action": "default", "seed": "6,6,6,6", "label": 0}])[0]
            joined2 = "\n".join(ply2.p1_chunks)
            if f"p2a: {nick}" not in joined2:
                tried.append(f"{nick}: ply 2 never references it, so the gap breaks nothing")
                continue
            # The driver's ACTUAL contract: one ply, not the path. If this ever flips, the
            # Python-side accumulation becomes a DOUBLE-count — say so here rather than let the
            # search discover it.
            assert joined1 not in joined2, (
                f"{impl}: expand_many returned a cumulative view — stop accumulating in "
                f"`SearchEngine._expand_ply`")
            chosen = (nick, ply1, ply2)
            break
        assert chosen is not None, (
            f"{impl}: no benched nicknamed mon produced a two-ply reveal at turn {_TURN} — the "
            f"fixture no longer exercises the gap. Tried: {tried}")
        nick, ply1, ply2 = chosen

        broken_err, _broken_n, broken_warns = _materialize(prefix + list(ply2.p1_chunks), record)
        fixed_err, fixed_n, fixed_warns = _materialize(
            prefix + list(ply1.p1_chunks) + list(ply2.p1_chunks), record)

    # The fixed composition is the claim.
    assert fixed_err is None, f"{impl}: the cumulative composition failed to replay: {fixed_err}"
    assert fixed_warns == 0, (
        f"{impl}: the cumulative composition still warns {fixed_warns}x about the active mon")
    assert fixed_n >= 2, f"{impl}: only {fixed_n} decisions — the depth-2 successor never opened"

    # NON-VACUITY: the shipped composition really is broken on this fixture, so a revert of the
    # production fix cannot leave this file passing for the wrong reason.
    assert broken_err is not None or broken_warns > 0, (
        f"{impl}: the bare-suffix composition replayed CLEAN — this fixture no longer exercises "
        f"the gap, so the regression is not being tested")
    if broken_err is not None:
        assert broken_err == f"KeyError: {to_id_str(nick)!r}", (
            f"{impl}: expected the nickname-as-species KeyError, got {broken_err}")


def test_the_reported_mojibake_is_in_the_TEAM_FILE_not_the_transport():
    """``KeyError: 'ptãra'`` was filed as a chunk-transport double-encode. It is not one.

    The committed team file holds ``c3 83 c2 a9`` — the UTF-8 encoding of ``Ã©``, i.e. ``é``
    already mangled once before it was written — so the nickname is mojibake at REST and every
    layer above merely carries it faithfully. Pinned here so the next reader of that ``KeyError``
    does not go looking for an encoder again, and so a future re-sync of ``data/teams/`` that
    silently CHANGES these bytes is a visible event (team files are hashed into ``pin_sha`` and
    into ``gen3_team_archetypes.json``; rewriting one is not a cosmetic edit).

    Pure file I/O — no sim, no battle.
    """
    raw = _NICKNAMED_TEAM.read_bytes()
    assert b"Pt\xc3\x83\xc2\xa9ra" in raw, (
        "the mcmegan fixture team no longer carries the double-encoded nickname the depth-2 "
        "KeyError was reported against")
    nick = "PtÃ©ra"
    assert to_id_str(nick) == "ptãra", "the reported key is to_id_str of the mojibake"
    assert nick.encode("latin-1").decode("utf-8") == "Ptéra", (
        "the intended nickname is recoverable by undoing exactly one utf-8/latin-1 pass")
