"""SEARCH's successor rows — the Rust core's encoded row per arm vs the protocol road's row.

``gen3_core_encoder_v1`` / ``gen3_core_search_v1``. A search arm's leaf is a Rust-core
``BattleVersion`` and the driver hands back its ENCODED row + mask (``expand_many(rows=True)``):
no Python view, event fold, tracker or encoder stands between the version and the score. This gate
holds that row to the one poke-env builds by REPLAYING the same arm's one-sided protocol (the
``obs_materializer`` replay player — the oracle road the prober still uses), trackers included,
BYTE for byte, plus the mask, at every branch point of real gen3ou battles.

It is the successor of ``one_sided_view_parity_fuzz_test.py`` (deleted with the view road in the
Rust Core deletion pass, program §4 M2): that gate compared FOUR roads (the port's one-sided view,
the Python view successor, the core's view successor, the core row) and only the last is still a
road anything takes. The core's VIEW and legality are gated at every decision by slice V
(``rust_core_parity_views.py``'s core column); this gate is the SEARCH-shaped half — a successor
built from a ply the search expanded, not a decision of a recorded battle.

**D10 — an arm whose ply resolved an INTERMEDIATE decision.** A ply that KOs one of our mons opens a
replacement round inside the same arm. The core's leaf is the version AT that round (``mid``); the
protocol road is fed the arm's chunks up to and including the one that carried that round's
request (:func:`split_at_intermediate`, poke-env's own chunk-boundary rule), so both stand on the
same decision.

**TWO ENTRY POINTS** (the project's fuzz rule): the sweep (:func:`run`, as a script) records fresh
random battles; the collected tests take ``obs_roundtrip_fuzz_test.record_fixture_battle(key=…)``
so they are the same board every run. Run the sweep on at least TWO fresh seeds before calling it
green.

    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/battle/core_row_parity_fuzz_test.py [n_battles] [--arms K]
    pytest -m sim src/agents/battle/core_row_parity_fuzz_test.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

import agents.training.obs_materializer as OM
from agents.training.obs_roundtrip_fuzz_test import RecordingFuzzPlayer
from utils.bridge.local_battle_runner import run_local_battles
from utils.bridge.reconstruction import ReconstructionRecord
from utils.bridge.search_session import SearchSession
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

pytestmark = [pytest.mark.sim, pytest.mark.integration]

BATTLE_FORMAT = "gen3ou"
DEFAULT_ARMS = 6
DEFAULT_TURNS = 3


class Census:
    """Divergences keyed by class, one example each."""

    def __init__(self) -> None:
        self.rows: Counter = Counter()
        self.examples: Dict[str, str] = {}
        self.compared = 0
        self.d10 = 0
        self.deferred: Counter = Counter()

    def note(self, path: str, protocol: Any, core: Any, where: str) -> None:
        self.rows[path] += 1
        self.examples.setdefault(path, f"{where}: protocol={protocol!r} core={core!r}")

    def defer(self, reason: str) -> None:
        self.deferred[reason] += 1

    def render(self) -> str:
        head = (f"✅ no divergence over {self.compared} core rows ({self.d10} D10)" if not self.rows
                else f"❌ {sum(self.rows.values())} divergences in {len(self.rows)} classes over "
                     f"{self.compared} core rows")
        lines = [head]
        for path, n in self.rows.most_common():
            lines += [f"   {n:6d}  {path}", f"           e.g. {self.examples[path]}"]
        for reason, n in self.deferred.most_common():
            lines.append(f"   [deferred x{n}] {reason}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The protocol road (the oracle)
# ---------------------------------------------------------------------------

def _record_one_battle(out_dir: str, impl: str, fixed_key: Optional[int] = None,
                       source: str = "pool"):
    """A real gen3ou battle: the REPRODUCIBLE fixture when ``fixed_key`` is set, else a fresh one."""
    if fixed_key is not None:
        from agents.training.obs_roundtrip_fuzz_test import record_fixture_battle

        return record_fixture_battle(out_dir, key=fixed_key, tag="CR", impl=impl, source=source)
    ts = int(time.time() * 1000) % 100000
    if source == "pool":
        pool = TeamLoader().get_all_teams()
        team_a, team_b = Gen3Teambuilder(pool), Gen3Teambuilder(pool)
    else:  # the ladder corpus / the procedural generator (utils.team_sources)
        from utils import team_sources

        team_a = team_sources.teambuilder(source, rng_seed=ts)
        team_b = team_sources.teambuilder(source, rng_seed=ts + 1)
    trainee = RecordingFuzzPlayer(
        out_dir=out_dir, rng_seed=ts, battle_format=BATTLE_FORMAT, team=team_a,
        account_configuration=AccountConfiguration(f"CRt{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT, team=team_b,
        account_configuration=AccountConfiguration(f"CRo{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    asyncio.run(run_local_battles(trainee, opp, 1, impl=impl))
    prefix = trainee.trace_prefixes[0]
    record = ReconstructionRecord.load(f"{prefix}_reconstruction.json")
    with open(f"{prefix}_summary.json") as f:
        summary = json.load(f)
    with np.load(f"{prefix}_states.npz") as z:
        npz = {k: z[k] for k in z.files}
    return record, summary, npz


class _ProtocolRoad:
    """Feed one-sided protocol text to the ``obs_materializer`` replay player (trackers on) and
    keep its materialized rows — the same construction and cadence ``materialize_branches`` uses."""

    def __init__(self, record, side: str, actions: List[int]) -> None:
        self.tag = OM._next_tag(None, record.format_id)
        self.player, self.client = OM._build_replay_player(
            username=record.username(side), packed_team=record.packed_team(side), side=side,
            actions=actions, battle_format=record.format_id, mappings=None, stall_config=None,
            map_actions_at=None, stop_after_decision=None, encode_only_at=None)
        self._first = True

    def feed(self, chunks) -> None:
        OM._refuse_poke_loop("core row parity")
        asyncio.run_coroutine_threadsafe(
            OM._feed(self.client, self.player, list(chunks), self.tag, first=self._first),
            OM.POKE_LOOP).result()
        self._first = False


def _is_decision_request(raw: str) -> bool:
    """The replay player's rule: a non-empty, non-``wait`` payload with an ``active`` or a
    ``forceSwitch`` block becomes a decision row."""
    if not raw.strip():
        return False
    try:
        req = json.loads(raw)
    except ValueError:
        return False
    return bool(not req.get("wait") and (req.get("active") or req.get("forceSwitch")))


def _request_lines(chunk: str) -> List[str]:
    return [line[len("|request|"):] for line in str(chunk).split("\n")
            if line.startswith("|request|")]


def split_at_intermediate(chunks: Sequence[str]) -> Optional[List[str]]:
    """The arm's chunks up to and INCLUDING the one that closed its FIRST decision, or ``None``.
    The cut is at a CHUNK boundary because ``Player._handle_battle_message`` parses a whole message
    before it dispatches the request — so the protocol road's row has absorbed the whole chunk."""
    for i, chunk in enumerate(chunks):
        if any(_is_decision_request(raw) for raw in _request_lines(chunk)):
            return list(chunks[:i + 1])
    return None


def check_core_row(ccore: dict, road: _ProtocolRoad, dec_i: int, where: str, cen: Census) -> bool:
    """The core's ENCODED row + mask at ONE branch point against the protocol road's row
    ``dec_i``, BYTE for byte."""
    from agents.battle.core_obs import wrap_row

    mats = road.player._materialized
    if len(mats) <= dec_i:
        cen.defer("the protocol successor produced no decision row")
        return False
    d = mats[dec_i]
    if ccore.get("row") is None:
        cen.note("core_row.presence", "a decision", "row null", where)
        return False
    cen.compared += 1
    got = wrap_row(ccore["row"])
    ok = True
    if list(np.asarray(d.mask)) != list(ccore.get("mask") or []):
        cen.note("core_row.mask", list(np.asarray(d.mask)), ccore.get("mask"), where)
        ok = False
    if d.obs is None or np.asarray(d.obs, dtype=np.float32).tobytes() != got.tobytes():
        bad = [] if d.obs is None else [int(i) for i in np.flatnonzero(
            np.asarray(d.obs, dtype=np.float32).view(np.uint32) != got.view(np.uint32))]
        cen.note("core_row.obs", "protocol row", f"{len(bad)} cells differ (first {bad[:6]})", where)
        ok = False
    return ok


def _choice_map(record, side, prefix_actions, pfx, anchor) -> Dict[int, str]:
    """``{action index: sim choice string}`` for every LEGAL action at the branch decision, from
    the REAL action mapper — so the gate never branches on an action the mask forbids."""
    mt = OM.materialize_decisions(
        list(pfx), username=record.username(side), packed_team=record.packed_team(side),
        side=side, actions=prefix_actions, battle_format=record.format_id,
        battle_tag=record.battle_tag, map_actions_at=anchor, stop_after_decision=anchor)
    return mt.action_choices or {}


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run(n_battles: int = 2, arms: int = DEFAULT_ARMS, turns: int = DEFAULT_TURNS,
        impl: str = "rust", fixed_key: Optional[int] = None,
        source: str = "pool") -> Tuple[Census, int]:
    cen = Census()
    branch_points = 0
    for b in range(n_battles):
        with tempfile.TemporaryDirectory() as td:
            record, summary, npz = _record_one_battle(
                td, impl, None if fixed_key is None else fixed_key + b, source=source)
        actions = np.asarray(npz["actions"], dtype=int)
        invs = summary["invocations"]
        side = record.side_of(record.trainee_username)
        other = "p2" if side == "p1" else "p1"
        cand = [i for i, inv in enumerate(invs)
                if inv.get("phase") == "move_selection" and int(inv["turn"]) > 1]
        if not cand:
            continue
        picks = [cand[int(len(cand) * f)] for f in (0.25, 0.5, 0.75)][:turns]
        with SearchSession(record, impl="rust") as rs:
            for anchor in picks:
                turn = int(invs[anchor]["turn"])
                try:
                    root = rs.open_root(turn, core="text", side=side, trackers=True)
                except Exception as e:                                # noqa: BLE001
                    cen.defer(f"open_root({turn}) failed: {type(e).__name__}")
                    continue
                pfx = root.prefix_p1_chunks if side == "p1" else root.prefix_p2_chunks
                prefix_actions = [int(x) for x in actions[:anchor]]
                cmap = _choice_map(record, side, prefix_actions, pfx, anchor)
                if not cmap:
                    cen.defer("no legal arm at the root (wait / ended)")
                    continue
                opp_rec = root.recorded_choices.get(other) or "default"
                expand = [{"node_id": root.node_id, f"{side}_action": cmap[a],
                           f"{other}_action": opp_rec,
                           "seed": f"{turn},{k + 1},{anchor + 7},{k * 13 + 11}", "label": a}
                          for k, a in enumerate(sorted(cmap)[:arms])]
                for node in rs.expand_many(expand, side=side, rows=True):
                    if node.ended or node.stuck:
                        cen.defer("arm ended / stuck (no successor to compare)")
                        continue
                    ccore = node.core_p1 if side == "p1" else node.core_p2
                    where = f"{record.battle_tag}@t{turn}/arm{node.label}"
                    if ccore is None:
                        cen.note("core_row.payload", "a core payload", None, where)
                        continue
                    suffix = list(node.p1_chunks if side == "p1" else node.p2_chunks)
                    branch_points += 1
                    if ccore.get("mid"):
                        head = split_at_intermediate(suffix)
                        if head is None:
                            cen.note("core_row.mid", "an intermediate request chunk", None, where)
                            continue
                        suffix = head
                    # PAD the action list so a ply ending in a faint does not exhaust the replay's
                    # actions before the arm's decision row is appended.
                    road = _ProtocolRoad(record, side,
                                         prefix_actions + [int(node.label)] + [0] * 8)
                    road.feed(list(pfx) + suffix)
                    if check_core_row(ccore, road, anchor + 1,
                                      where + ("/D10" if ccore.get("mid") else ""), cen) \
                            and ccore.get("mid"):
                        cen.d10 += 1
    return cen, branch_points


def test_search_core_rows_equal_the_protocol_roads_rows():
    """One REPRODUCIBLE battle, three turns, five arms each."""
    cen, branch_points = run(n_battles=1, arms=5, turns=3, fixed_key=0)
    print("\n" + cen.render())
    assert branch_points >= 8, f"only {branch_points} branch points ran — vacuous"
    assert cen.compared >= 8, f"only {cen.compared} core rows compared — vacuous"
    assert not cen.rows, "\n" + cen.render()


def test_an_intermediate_decision_arm_is_the_version_at_that_round():
    """THE D10 GATE — a fixture battle whose arms include a ply that resolved a REPLACEMENT round
    inside itself (key 8 was chosen by a sweep of keys 0-19); ``d10 == 0`` means the battle changed
    and the gate is vacuous, so it is an assertion."""
    cen, branch_points = run(n_battles=1, arms=10, turns=3, fixed_key=8)
    print("\n" + cen.render())
    assert cen.d10 >= 1, (f"NO D10 arm compared ({branch_points} branch points) — the gate is "
                          f"vacuous about the intermediate-decision leaf")
    assert not cen.rows, "\n" + cen.render()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("n_battles", nargs="?", type=int, default=2)
    ap.add_argument("--arms", type=int, default=DEFAULT_ARMS)
    ap.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    ap.add_argument("--impl", default="rust", help="the bridge that PLAYS the recorded battle")
    ap.add_argument("--fixed-key", type=int, default=None,
                    help="use the REPRODUCIBLE fixture battle(s) from this key")
    from utils import team_sources

    team_sources.add_arguments(ap)
    a = ap.parse_args()
    census, bp = run(a.n_battles, a.arms, a.turns, a.impl, a.fixed_key, a.team_source)
    print(census.render())
    print(f"branch points: {bp}   core rows byte-equal: {census.compared - sum(census.rows.values())}"
          f" ({census.d10} D10)")
    sys.exit(1 if census.rows else 0)
