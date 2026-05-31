"""Event-log fuzz — the verification spine for the event-sourced battle (design §7).

Runs real ``gen3ou`` battles on the live Showdown server with both players backed by
:class:`Gen3Battle`, intercepts the **raw protocol** each player receives, and proves —
per battle, per turn — that the captured event log matches an *independent* re-derivation
from those raw lines. This is the canonical fuzz pattern (see
``src/agents/training/poke_env_gaps/``): the log is the oracle, the raw protocol is the
ground truth, and any disagreement raises with a detailed diff.

Two checks per battle:
  1. **Conservation (design §4.3):** every protocol line lands in exactly one policy
     bucket; ``Gen3Battle.assert_conservation()`` balances; zero unsupported/unknown.
  2. **Event-vs-protocol (design §7.3):** for every turn, independently re-derive who
     moved (and in what order), which moves were used, switches, drags, faints,
     crit/miss/fail, and effectiveness — straight from the archived ``|...|`` lines —
     and assert it equals what ``Gen3Battle`` / :class:`TurnView` reports.

Coverage is asserted across the corpus: the rare event kinds (crit, miss, fail, drag,
status, super-effective, immune/resisted) must each be exercised at least once, so a
green run actually means something.

Run directly (requires: npm run showdown):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/battle/event_log_fuzz_e2e_test.py [n_battles]
"""

from __future__ import annotations

import asyncio
import random
import sys
import time
import traceback
from collections import defaultdict
from typing import Dict, List, Optional, Set

from poke_env import AccountConfiguration
from poke_env.data.normalize import to_id_str
from poke_env.player import RandomPlayer
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.battle.battle_event import EVENT_VALUE_KEYS, OPP, OURS, EventKind
from agents.battle.gen3_battle import Gen3Battle
from agents.observation.gen3_effects import encode_volatiles, normalize_cant_reason
from agents.battle.turn_view import TurnView
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"


# --------------------------------------------------------------------------- #
# Independent re-derivation of one turn straight from raw protocol lines.       #
# Deliberately does NOT consult the event log — it is the cross-check.          #
# --------------------------------------------------------------------------- #
def _side_of(ident: str, player_role: str) -> Optional[str]:
    if len(ident) < 2 or ident[0] != "p" or ident[1] not in ("1", "2"):
        return None
    return OURS if ident[:2] == player_role else OPP


class _IndepTurn:
    __slots__ = ("move_order", "moves", "switched", "dragged", "fainted",
                 "crit", "missed", "failed", "eff")

    def __init__(self):
        self.move_order: List[str] = []          # first-occurrence side order
        self.moves: Dict[str, List[str]] = {OURS: [], OPP: []}  # raw move ids
        self.switched: Set[str] = set()
        self.dragged: Set[str] = set()
        self.fainted: Set[str] = set()
        self.crit: Dict[str, bool] = {}
        self.missed: Dict[str, bool] = {}
        self.failed: Dict[str, bool] = {}
        self.eff: Dict[str, float] = {}


def _rederive(raw_lines: List[List[str]], player_role: str) -> Dict[int, _IndepTurn]:
    turns: Dict[int, _IndepTurn] = defaultdict(_IndepTurn)
    cur = 0
    mover: Optional[str] = None
    _EFF = {"-supereffective": 2.0, "-resisted": 0.5, "-immune": 0.0}
    for msg in raw_lines:
        if len(msg) < 2:
            continue
        kw = msg[1]
        if kw == "turn":
            cur = int(msg[2])
            mover = None
            continue
        t = turns[cur]
        if kw == "move":
            side = _side_of(msg[2], player_role)
            if side is None:
                continue
            mover = side
            if side not in t.move_order:
                t.move_order.append(side)
            t.moves[side].append(to_id_str(msg[3]))
            # legacy variant: miss/notarget carried as a |move| suffix
            toks = set(msg[3:])
            if "[miss]" in toks:
                t.missed[side] = True
            if "[notarget]" in toks:
                t.failed[side] = True
        elif kw == "switch":
            side = _side_of(msg[2], player_role)
            if side:
                t.switched.add(side)
        elif kw == "drag":
            side = _side_of(msg[2], player_role)
            if side:
                t.switched.add(side)
                t.dragged.add(side)
        elif kw == "faint":
            side = _side_of(msg[2], player_role)
            if side:
                t.fainted.add(side)
        elif kw == "-crit" and mover:
            t.crit[mover] = True
        elif kw == "-miss" and mover:
            t.missed[mover] = True
        elif kw in ("-fail", "-notarget", "-nothing") and mover:
            t.failed[mover] = True
        elif kw in _EFF:
            m = mover
            if m is None:  # mirror Gen3Battle's fallback: opposite the named defender
                d = _side_of(msg[2], player_role)
                m = OPP if d == OURS else OURS if d == OPP else None
            if m is not None:
                t.eff[m] = _EFF[kw]
    return turns


# --------------------------------------------------------------------------- #
# Per-battle validation: log vs independent re-derivation.                      #
# --------------------------------------------------------------------------- #
def _event_moves(events, side: str) -> List[str]:
    return sorted(
        e.move_id
        for e in events
        if e.kind is EventKind.MOVE and e.side == side
    )


def _raw_window_for_turn(raw, turn: int) -> str:
    """The raw protocol lines belonging to game ``turn`` (between ``|turn|turn`` and the
    next ``|turn|``), for self-debugging a mismatch. Lines before the first ``|turn|``
    are turn 0."""
    out, cur = [], 0
    for msg in raw:
        if len(msg) < 2:
            continue
        if msg[1] == "turn":
            cur = int(msg[2])
            continue
        if cur == turn and msg[1] not in ("", "t:", ":"):
            out.append("|".join(str(x) for x in msg))
    return " ;; ".join(out)


def validate_battle(battle: Gen3Battle) -> List[str]:
    """Return a list of mismatch strings ([] == perfect agreement)."""
    problems: List[str] = []
    tag = battle.battle_tag

    # 1) conservation
    try:
        battle.assert_conservation()
    except AssertionError as e:
        problems.append(f"[{tag}] conservation: {e}")

    # 1b) payload schema — every event carries its kind's required keys (the
    #     structural no-silent-loss guard, checked here across ALL real kinds)
    for e in battle.events:
        missing = EVENT_VALUE_KEYS.get(e.kind, frozenset()) - set(e.value)
        if missing:
            problems.append(
                f"[{tag}] seq{e.seq} {e.kind.name} missing payload keys {missing}"
            )

    # 1c) live view — current-board snapshot is faithful to poke-env's state and
    #     carries no history. Builds without crashing and matches the tracker.
    try:
        lv = battle.live_view()
        for built, raw in (
            (lv.ours.active, battle.active_pokemon),
            (lv.opp.active, battle.opponent_active_pokemon),
        ):
            if raw is None:
                continue
            if built is None:
                problems.append(f"[{tag}] live_view missing active for {raw.species}")
                continue
            if built.species != raw.species:
                problems.append(
                    f"[{tag}] live_view active {built.species} != {raw.species}")
            if abs(built.hp_fraction - raw.current_hp_fraction) > 1e-9:
                problems.append(
                    f"[{tag}] live_view hp {built.hp_fraction} != "
                    f"{raw.current_hp_fraction} for {raw.species}")
            assert not hasattr(built, "last_move")  # boundary: no history fields
        # 1d) crash-don't-drop allowlists — every volatile on every mon in the live
        #     view must encode (no silently-dropped state), and every |cant| reason in
        #     the log must be a known gen3 cause. encode_*/normalize_* RAISE on an
        #     unclassified value, so a gap here fails the battle loudly.
        for side in (lv.ours, lv.opp):
            for mon in side.mons:
                encode_volatiles(mon.volatiles)  # raises UnknownVolatileError if gap
        for e in battle.events:
            if e.kind is EventKind.CANT and e.reason is not None:
                normalize_cant_reason(e.reason)  # raises if unknown gen3 cause
    except Exception as exc:  # pragma: no cover - defensive
        problems.append(f"[{tag}] live_view/allowlist raised: {exc!r}")

    raw = battle._fuzz_raw  # archived by the player below
    indep = _rederive(raw, battle._player_role)

    all_turns = set(indep) | {e.turn for e in battle.events}
    for t in sorted(all_turns):
        it = indep.get(t, _IndepTurn())
        v = TurnView.for_turn(battle, t)
        evs = battle.events_for_turn(t)

        def side_view(s):
            return v.ours if s == OURS else v.opp

        # move order (who acted first)
        if it.move_order != v.move_order:
            problems.append(
                f"[{tag}] t{t} move_order: raw={it.move_order} log={v.move_order}"
            )

        for s in (OURS, OPP):
            sv = side_view(s)
            # moves used (set) — delegation-aware logs keep both the delegator and
            # the called move; each is the protocol-named id straight off its line
            protocol_moves = sorted(it.moves[s])
            log_moves = _event_moves(evs, s)
            if protocol_moves != log_moves:
                problems.append(
                    f"[{tag}] t{t} {s} moves: raw={protocol_moves} log={log_moves}"
                )
            # executed move id — TurnView picks the LAST |move| of the turn as the
            # move that actually fired (delegation: Sleep Talk -> the called move).
            raw_executed = it.moves[s][-1] if it.moves[s] else None
            if raw_executed != sv.move_id:
                problems.append(
                    f"[{tag}] t{t} {s} executed move_id: raw={raw_executed} "
                    f"log={sv.move_id} | RAW={_raw_window_for_turn(raw, t)}"
                )
            # switched / dragged / fainted
            if (s in it.switched) != sv.switched:
                problems.append(
                    f"[{tag}] t{t} {s} switched: raw={s in it.switched} log={sv.switched}"
                )
            if (s in it.dragged) != sv.drag:
                problems.append(
                    f"[{tag}] t{t} {s} drag: raw={s in it.dragged} log={sv.drag}"
                )
            if (s in it.fainted) != sv.fainted:
                problems.append(
                    f"[{tag}] t{t} {s} fainted: raw={s in it.fainted} log={sv.fainted}"
                )
            # crit / miss / fail
            if it.crit.get(s, False) != sv.crit:
                problems.append(
                    f"[{tag}] t{t} {s} crit: raw={it.crit.get(s, False)} log={sv.crit}"
                )
            if it.missed.get(s, False) != sv.missed:
                problems.append(
                    f"[{tag}] t{t} {s} miss: raw={it.missed.get(s, False)} log={sv.missed}"
                )
            if it.failed.get(s, False) != sv.failed:
                problems.append(
                    f"[{tag}] t{t} {s} fail: raw={it.failed.get(s, False)} log={sv.failed}"
                )
            # effectiveness
            if it.eff.get(s) != sv.effectiveness:
                problems.append(
                    f"[{tag}] t{t} {s} eff: raw={it.eff.get(s)} log={sv.effectiveness}"
                )
    return problems


# --------------------------------------------------------------------------- #
# Fuzz player                                                                  #
# --------------------------------------------------------------------------- #
class EventLogFuzzPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._raw: Dict[str, List[List[str]]] = defaultdict(list)
        self.mismatches: List[str] = []
        self.battles_checked = 0
        self.kinds_seen: Set[EventKind] = set()
        self.coverage: Set[str] = set()
        self.errors: List[str] = []

    async def _handle_battle_message(self, split_messages):
        tag = split_messages[0][0].lstrip(">")
        for msg in split_messages[1:]:
            self._raw[tag].append(list(msg))  # copy — archive before parsing
        await super()._handle_battle_message(split_messages)

    def choose_move(self, battle):
        return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        try:
            # stash the archived raw lines on the battle for validate_battle()
            battle._fuzz_raw = self._raw.get(battle.battle_tag, [])
            self.battles_checked += 1
            for e in battle.events:
                self.kinds_seen.add(e.kind)
            self._note_coverage(battle)
            self.mismatches.extend(validate_battle(battle))
        except Exception:  # never let validation crash the event loop
            self.errors.append(traceback.format_exc())
        finally:
            self._raw.pop(battle.battle_tag, None)

    def _note_coverage(self, battle):
        for e in battle.events:
            if e.kind is EventKind.CRIT:
                self.coverage.add("crit")
            elif e.kind is EventKind.MISS:
                self.coverage.add("miss")
            elif e.kind is EventKind.FAIL:
                self.coverage.add("fail")
            elif e.kind is EventKind.DRAG:
                self.coverage.add("drag")
            elif e.kind is EventKind.STATUS:
                self.coverage.add("status")
            elif e.kind is EventKind.SUPEREFFECTIVE:
                self.coverage.add("supereffective")
            elif e.kind in (EventKind.IMMUNE, EventKind.RESISTED):
                self.coverage.add("immune/resisted")
            elif e.kind is EventKind.SWITCH:
                self.coverage.add("switch")
            elif e.kind is EventKind.FAINT:
                self.coverage.add("faint")
            elif e.kind in (EventKind.ITEM, EventKind.ENDITEM):
                self.coverage.add("item")


# --------------------------------------------------------------------------- #
# Runner                                                                        #
# --------------------------------------------------------------------------- #
_TEAM_POOL = None


def _random_team() -> str:
    global _TEAM_POOL
    if _TEAM_POOL is None:
        loader = TeamLoader()
        _TEAM_POOL = loader.get_sample_teams() or loader.get_all_teams()
        if not _TEAM_POOL:
            raise RuntimeError("no gen3ou teams found under data/teams")
    return random.choice(_TEAM_POOL)


REQUIRED_COVERAGE = {
    "switch", "faint", "crit", "miss", "supereffective", "immune/resisted", "status",
}


async def main(n_battles: int = 40) -> None:
    ts = int(time.time()) % 100000
    print(f"Event-Log Fuzz — {BATTLE_FORMAT} — {n_battles} battles", flush=True)

    # Proven recipe (mirrors the poke_env_gaps fuzz tests): our Gen3Battle-backed
    # validator challenges a RandomPlayer opponent. Passwords are set so the server
    # honours the usernames — guest logins race the challenge ("user not found").
    fuzz = EventLogFuzzPlayer(
        battle_format=BATTLE_FORMAT,
        team=Gen3Teambuilder(_random_team()),
        account_configuration=AccountConfiguration(f"ELz{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=5,
        battle_class=Gen3Battle,
    )
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=Gen3Teambuilder(_random_team()),
        account_configuration=AccountConfiguration(f"ELo{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=5,
    )
    await fuzz.battle_against(opp, n_battles=n_battles)

    checked = fuzz.battles_checked
    mismatches = fuzz.mismatches
    errors = fuzz.errors
    kinds = fuzz.kinds_seen
    coverage = fuzz.coverage

    print(f"\nBattles validated : {checked}")
    print(f"Distinct EventKinds seen: {sorted(k.name for k in kinds)}")
    print(f"Coverage flags    : {sorted(coverage)}")

    missing_cov = REQUIRED_COVERAGE - coverage
    print("=" * 70)
    ok = True
    if errors:
        ok = False
        print(f"VALIDATION ERRORS ({len(errors)}):")
        for e in errors[:5]:
            print(e)
    if mismatches:
        ok = False
        print(f"MISMATCHES ({len(mismatches)}) — first 25:")
        for m in mismatches[:25]:
            print(" ", m)
    if missing_cov:
        ok = False
        print(f"COVERAGE GAP: never exercised {sorted(missing_cov)} "
              f"(run more battles or engineer teams)")
    if checked == 0:
        ok = False
        print("NO BATTLES VALIDATED — is the server running and are teams loading?")

    if ok:
        print("PASS — event log matches raw protocol; conservation balanced; "
              "coverage satisfied.")
    else:
        print("FAIL")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    asyncio.run(main(n))
