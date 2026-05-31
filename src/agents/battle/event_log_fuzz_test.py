"""Event-log fuzz — the verification spine for the event-sourced battle (design §7).

Runs real ``gen3ou`` battles in-process via the local BattleStream bridge with both players backed by
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

Run directly (no server needed; runs in-process via the local BattleStream bridge):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/battle/event_log_fuzz_test.py [n_battles]
"""

from __future__ import annotations

import asyncio
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
from utils.bridge.local_battle_runner import run_local_battles

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

        # 1c-ii) Phase-1 widened fields — spread / PP / consumed-item / status-counter —
        #        faithful to poke-env per mon, and the own/opp spread gate respected.
        #        Within one side species are unique (gen3 OU), so match by species.
        for side_lv, raw_team, is_own in (
            (lv.ours, battle.team, True),
            (lv.opp, battle.opponent_team, False),
        ):
            raw_by_species = {m.species: m for m in raw_team.values()}
            for built in side_lv.mons:
                raw = raw_by_species.get(built.species)
                if raw is None:
                    problems.append(f"[{tag}] live_view mon {built.species} not in raw team")
                    continue
                if built.base_stats != raw.base_stats:
                    problems.append(
                        f"[{tag}] {built.species} base_stats {built.base_stats} != "
                        f"{raw.base_stats}")
                # spread gate: own side carries ivs/evs/nature, opp does not
                if built.spread_known != is_own:
                    problems.append(
                        f"[{tag}] {built.species} spread_known={built.spread_known} "
                        f"but is_own={is_own}")
                if not is_own and (built.ivs or built.evs or built.nature):
                    problems.append(
                        f"[{tag}] opp {built.species} leaked private spread "
                        f"ivs={built.ivs} evs={built.evs} nature={built.nature}")
                if is_own and raw.ivs is not None and built.ivs != tuple(raw.ivs):
                    problems.append(
                        f"[{tag}] {built.species} ivs {built.ivs} != {tuple(raw.ivs)}")
                if is_own and raw.evs is not None and built.evs != tuple(raw.evs):
                    problems.append(
                        f"[{tag}] {built.species} evs {built.evs} != {tuple(raw.evs)}")
                if is_own and built.nature != raw.nature:
                    problems.append(
                        f"[{tag}] {built.species} nature {built.nature} != {raw.nature}")
                # consumed item (id-form) and status counter mirror poke-env
                exp_consumed = to_id_str(raw.consumed_item) if raw.consumed_item else None
                if built.consumed_item != exp_consumed:
                    problems.append(
                        f"[{tag}] {built.species} consumed_item {built.consumed_item} "
                        f"!= {exp_consumed}")
                if built.status_counter != int(raw.status_counter or 0):
                    problems.append(
                        f"[{tag}] {built.species} status_counter "
                        f"{built.status_counter} != {raw.status_counter}")
                # enriched moves: PP per slot faithful to poke-env's Move objects
                for lm in built.moves:
                    rm = raw.moves.get(lm.id)
                    if rm is None:
                        problems.append(
                            f"[{tag}] {built.species} live move {lm.id} absent in raw")
                        continue
                    if (lm.current_pp, lm.max_pp) != (rm.current_pp, rm.max_pp):
                        problems.append(
                            f"[{tag}] {built.species} {lm.id} pp "
                            f"({lm.current_pp},{lm.max_pp}) != "
                            f"({rm.current_pp},{rm.max_pp})")

        # 1c-iii) LiveView meta passthroughs mirror the battle's own accessors
        if lv.battle_tag != battle.battle_tag:
            problems.append(f"[{tag}] live_view battle_tag {lv.battle_tag} != {tag}")
        if lv.finished != bool(battle.finished):
            problems.append(f"[{tag}] live_view finished {lv.finished} != {battle.finished}")
        if lv.won != battle.won or lv.lost != battle.lost:
            problems.append(
                f"[{tag}] live_view won/lost ({lv.won},{lv.lost}) != "
                f"({battle.won},{battle.lost})")
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
        self.decisions_checked = 0
        self.kinds_seen: Set[EventKind] = set()
        self.coverage: Set[str] = set()
        self.errors: List[str] = []
        # Coverage of the crash-don't-drop allowlists — which volatiles / cant
        # reasons were actually exercised across the run. Reported at the end so a
        # human can SEE that rare states (Future Sight, partial-trap, Encore, …) got
        # hit, and so a future allowlist gap shows up as a coverage miss, not luck.
        self.volatiles_seen: Set[str] = set()
        self.cant_reasons_seen: Set[str] = set()

    async def _handle_battle_message(self, split_messages):
        tag = split_messages[0][0].lstrip(">")
        for msg in split_messages[1:]:
            self._raw[tag].append(list(msg))  # copy — archive before parsing
        await super()._handle_battle_message(split_messages)

    def choose_move(self, battle):
        # Validate the strict-view legality surface against the live server request at
        # the exact moment we're asked to act — the one place the request is authoritative.
        try:
            self._validate_decision(battle)
        except Exception:  # never let validation crash the event loop
            self.errors.append(traceback.format_exc())
        # Per-decision crash-don't-drop check — mirrors the TRAINING obs path, which
        # encodes the live view on EVERY turn (see _check_live_state).
        self._check_live_state(battle)
        return self.choose_random_move(battle)

    def _validate_decision(self, battle):
        """Cross-check ``battle.strict_view().legal`` / ``.live`` against the raw,
        server-parsed request fields for THIS decision."""
        sv = battle.strict_view()
        legal = sv.legal
        tag = battle.battle_tag
        self.decisions_checked += 1

        # move-id echo: LegalActions extracts the request's active move slots verbatim,
        # EXCEPT the lone `struggle` entry — that is normalized OUT of move_slots and
        # surfaced ONLY as the `struggle` flag (the single-source contract that prevents
        # the "struggle double-enabling" bug; the flag itself is validated below). So the
        # echo is against the request move ids with struggle filtered out.
        req = battle.last_request or {}
        active = req.get("active") or [{}]
        req_move_ids = [m.get("id") for m in active[0].get("moves", []) if m.get("id") != "struggle"]
        if list(legal.move_ids) != req_move_ids:
            self.mismatches.append(
                f"[{tag}] t{battle.turn} legal.move_ids {list(legal.move_ids)} "
                f"!= request (struggle-excluded) {req_move_ids}")
        # Positive single-source guard: struggle must NEVER appear as a move slot.
        if any(m.id == "struggle" for m in legal.move_slots):
            self.mismatches.append(
                f"[{tag}] t{battle.turn} struggle leaked into legal.move_slots "
                f"{[m.id for m in legal.move_slots]} (it must be the flag only)")

        # switches: same species set as poke-env's available_switches (server-authoritative)
        avail = sorted(m.species for m in battle.available_switches)
        if sorted(legal.switch_species) != avail:
            self.mismatches.append(
                f"[{tag}] t{battle.turn} legal switches {sorted(legal.switch_species)} "
                f"!= available {avail}")

        # flags
        for name, got, exp in (
            ("force_switch", legal.force_switch, bool(battle.force_switch)),
            ("trapped", legal.trapped, bool(battle.trapped)),
            ("wait", legal.wait, bool(battle.wait)),
            ("maybe_trapped", legal.maybe_trapped, bool(battle.maybe_trapped)),
            ("struggle", legal.struggle,
             any(m.id == "struggle" for m in battle.available_moves)),
        ):
            if got != exp:
                self.mismatches.append(
                    f"[{tag}] t{battle.turn} legal.{name} {got} != {exp}")
            if exp and name in ("force_switch", "trapped", "maybe_trapped", "struggle"):
                self.coverage.add(name)
        if legal.switches:
            self.coverage.add("legal_switch")

        # live view builds and active-mon PP mirrors poke-env, mid-battle
        live = sv.live
        for built, raw in (
            (live.ours.active, battle.active_pokemon),
            (live.opp.active, battle.opponent_active_pokemon),
        ):
            if raw is None or built is None:
                continue
            for lm in built.moves:
                rm = raw.moves.get(lm.id)
                if rm is not None and (lm.current_pp, lm.max_pp) != (
                    rm.current_pp, rm.max_pp
                ):
                    self.mismatches.append(
                        f"[{tag}] t{battle.turn} {built.species} {lm.id} pp "
                        f"({lm.current_pp},{lm.max_pp}) != "
                        f"({rm.current_pp},{rm.max_pp})")

    def _check_live_state(self, battle) -> None:
        """Encode every live volatile at this decision (mirrors the TRAINING obs path,
        which encodes the live view EVERY turn), recording coverage and any
        crash-don't-drop gap into self.errors. validate_battle() only sees the FINAL
        state, so a volatile that appears and resolves MID-battle (Future Sight /
        Doom Desire pending for ~2 turns, Disable, Encore, partial-trap) would never
        reach the final-state check — this per-decision surface caught `doomdesire`/
        `immunity` that the battle-end check missed."""
        try:
            lv = battle.live_view()
            for side in (lv.ours, lv.opp):
                for mon in side.mons:
                    if mon.volatiles:
                        encode_volatiles(mon.volatiles)  # raises on an unclassified id
                        self.volatiles_seen.update(mon.volatiles)
        except Exception as exc:
            self.errors.append(
                f"[{battle.battle_tag}] per-decision live-state encode raised: {exc!r}"
            )

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
            elif e.kind is EventKind.CANT and e.reason is not None:
                # Record the normalised cant reason for coverage. validate_battle
                # already RAISES on an unknown reason, so a swallow here is safe —
                # we only want the coverage signal, not a second tripwire.
                try:
                    self.cant_reasons_seen.add(normalize_cant_reason(e.reason))
                except Exception:
                    pass
        # Volatile coverage is collected in id-form from the per-decision live view
        # (_check_live_state) — the exact form encode_volatiles validates — so it is
        # NOT re-collected here from VOLATILE_START (whose raw "move: Taunt" form
        # would just add noise to the report).

        # Phase-1 widened-field coverage: confirm the new LiveView fields were actually
        # exercised on real play (not just present in the schema).
        try:
            lv = battle.live_view()
            for side in (lv.ours, lv.opp):
                for mon in side.mons:
                    if mon.consumed_item:
                        self.coverage.add("consumed_item")
                    if mon.status_counter > 0:
                        self.coverage.add("status_counter")
                    if any(m.id.startswith("hiddenpower") for m in mon.moves):
                        self.coverage.add("hp_move")
                if side is lv.ours and side.active and side.active.spread_known:
                    # the own-side spread GATE fired on a real mon (always, by construction)
                    self.coverage.add("spread_known")
                    # ...and whether it actually carried IV/EV/nature DATA (gen3ou has no
                    # team preview, so this answers whether the spread block is populated)
                    if side.active.ivs or side.active.evs or side.active.nature:
                        self.coverage.add("spread_data")
        except Exception:  # pragma: no cover - coverage probe must never crash the loop
            pass


# --------------------------------------------------------------------------- #
# Runner                                                                        #
# --------------------------------------------------------------------------- #
_TEAM_POOL = None


def _team_pool() -> list:
    """The FULL gen3ou team pool (samples + others). Returning the whole list to
    Gen3Teambuilder makes ``yield_team()`` re-roll a random team EVERY battle, so a
    run of N battles exercises N different movesets — far broader volatile / cant /
    effect coverage than reusing two fixed teams. This breadth is what surfaces rare
    mid-battle volatiles (Future Sight / Doom Desire, partial-trap, Encore, …) that a
    small fixed team would never reveal."""
    global _TEAM_POOL
    if _TEAM_POOL is None:
        loader = TeamLoader()
        _TEAM_POOL = loader.get_all_teams()
        if not _TEAM_POOL:
            raise RuntimeError("no gen3ou teams found under data/teams")
    return _TEAM_POOL


REQUIRED_COVERAGE = {
    "switch", "faint", "crit", "miss", "supereffective", "immune/resisted", "status",
    # legality surface — both fire in essentially every real battle
    "force_switch", "legal_switch", "spread_known",
    # own-team spread DATA must actually reach LiveView every battle now that the poke-env
    # backfill_teambuilder_spread fix populates mon.ivs/evs/nature from the declared team
    # (gen3ou has no team preview). Before that fix this was never seen across 25k+ battles —
    # its presence here is the end-to-end proof of the fix.
    "spread_data",
}

# Exercised but rare/team-dependent — reported, not required (avoids spurious flakiness).
REPORTED_COVERAGE = {
    "consumed_item", "status_counter", "hp_move",
    "trapped", "maybe_trapped", "struggle",
}


async def main(n_battles: int = 40, max_seconds: "float | None" = None) -> None:
    ts = int(time.time()) % 100000
    mode = f"{max_seconds:.0f}s time budget" if max_seconds else f"{n_battles} battles"
    print(f"Event-Log Fuzz — {BATTLE_FORMAT} — {mode}", flush=True)

    # Proven recipe (mirrors the poke_env_gaps fuzz tests): our Gen3Battle-backed
    # validator challenges a RandomPlayer opponent. Passwords are set so the server
    # honours the usernames — guest logins race the challenge ("user not found").
    # Both players draw a RANDOM team per battle from the full pool (yield_team
    # re-rolls each game), so N battles cover N movesets — broad effect coverage.
    pool = _team_pool()
    fuzz = EventLogFuzzPlayer(
        battle_format=BATTLE_FORMAT,
        team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"ELz{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        max_concurrent_battles=5,
        battle_class=Gen3Battle,
    )
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"ELo{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        max_concurrent_battles=5,
    )

    if max_seconds:
        # Time-budget mode: keep playing chunks of battles (one persistent validator,
        # accumulating coverage/mismatches) until the wall-clock budget is spent. Lets us
        # run escalating soak tests (1m / 5m / 15m) to shake out rare edge cases. The run
        # stops at a battle BOUNDARY (full summary printed) rather than being SIGKILL'd
        # mid-battle by an external timeout. ABORT EARLY on the first error/mismatch so a
        # real gap surfaces fast, not after 15m.
        start = time.monotonic()
        chunk = 25
        while time.monotonic() - start < max_seconds:
            await run_local_battles(fuzz, opp, chunk)
            elapsed = time.monotonic() - start
            print(
                f"  …{fuzz.battles_checked} battles, {fuzz.decisions_checked} decisions "
                f"| {elapsed:.0f}/{max_seconds:.0f}s "
                f"| vol={len(fuzz.volatiles_seen)} cant={len(fuzz.cant_reasons_seen)} "
                f"| mismatch={len(fuzz.mismatches)} err={len(fuzz.errors)}",
                flush=True,
            )
            if fuzz.errors or fuzz.mismatches:
                print("  ABORTING — error/mismatch detected", flush=True)
                break
    else:
        await run_local_battles(fuzz, opp, n_battles)

    checked = fuzz.battles_checked
    mismatches = fuzz.mismatches
    errors = fuzz.errors
    kinds = fuzz.kinds_seen
    coverage = fuzz.coverage

    print(f"\nBattles validated : {checked}")
    print(f"Decisions validated: {fuzz.decisions_checked} (strict-view legality surface)")
    print(f"Distinct EventKinds seen: {sorted(k.name for k in kinds)}")
    print(f"Coverage flags    : {sorted(coverage)}")
    print(f"Phase-1 field coverage (required): "
          f"{sorted(c for c in coverage if c in REQUIRED_COVERAGE)}")
    print(f"Phase-1 field coverage (reported): "
          f"{sorted(REPORTED_COVERAGE & coverage)} "
          f"(not seen: {sorted(REPORTED_COVERAGE - coverage)})")
    print(f"Volatiles exercised ({len(fuzz.volatiles_seen)}): {sorted(fuzz.volatiles_seen)}")
    print(f"Cant reasons exercised ({len(fuzz.cant_reasons_seen)}): {sorted(fuzz.cant_reasons_seen)}")

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
    import argparse

    # Positional arg is a battle count ("80") OR a duration ("1m", "5m", "15m");
    # --seconds is the explicit wall-clock form and overrides a duration positional.
    p = argparse.ArgumentParser(description="Event-log / live-view e2e fuzz")
    p.add_argument("n", nargs="?", default="40",
                   help="battle count (e.g. 80) or a duration (e.g. 1m, 5m, 15m)")
    p.add_argument("--seconds", type=float, default=None,
                   help="run for this many wall-clock seconds (overrides a duration arg)")
    args = p.parse_args()
    max_seconds = args.seconds
    n_battles = 40
    if max_seconds is None and isinstance(args.n, str) and args.n.endswith("m"):
        max_seconds = float(args.n[:-1]) * 60.0
    elif max_seconds is None:
        n_battles = int(args.n)
    asyncio.run(main(n_battles=n_battles, max_seconds=max_seconds))
