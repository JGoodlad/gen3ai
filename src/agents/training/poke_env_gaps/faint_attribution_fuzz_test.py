"""Bridge fuzz test: a recorded faint event must name the mon that ACTUALLY fainted.

`BattleRecorder` writes one `<side>:<species>:fainted` event per faint into every trace. It used to
detect the faint by COUNT (`fainted_count` went up) and then label it with `prev_ctx.*_active` — the
mon that was active when the DECISION was made. That is the wrong mon whenever a switch resolved on
the same turn, and the trace then contradicts its own battle log two lines above:

    we switch cloyster → jolteon
    opp explosion → jolteon (now 0%)
    we cloyster fainted            ← the protocol says JOLTEON fainted

Two shapes produce it, and the second is why an HP-transition check is not enough:

  1. WE switch, the switch-IN eats the hit and dies. The pre-decision active (Cloyster) left the
     field safely and was named anyway.
  2. THE OPPONENT switches a mon in and it dies the same turn (Claydol → Dugtrio, our Ice Beam KOs
     Dugtrio). Dugtrio was never revealed before, so it has no previous HP to fall from — only a
     SET difference over the fainted species names catches it.

Measured on `ai_v9_17_tdaux_lam3_0818` before the fix: **25 of 466 turns** named a mon that had not
fainted. The fix reads the newly-fainted species as a set difference between the two snapshots'
`*_fainted_species` (which the snapshot already carried), and uses it for the HP-delta slot too —
both were wrong in the same way.

GROUND TRUTH IS THE PROTOCOL, not our own bookkeeping: the sim emits `|faint|pNa: Species` and this
test compares the recorder's events against those lines, per turn. Validating the recorder against
another of our own derived structures would only prove they agree with each other.

Per-turn invariants (a violation is collected and fails the run):
  1. Every `<side>:<species>:fainted` event names a species the protocol says fainted that turn.
  2. Every protocol faint is named by an event (nothing silently dropped).
  3. The SIDE is right — a faint is never attributed to the wrong player.

Coverage is asserted, not assumed: a run that never saw a switch-in die has not exercised the bug,
and says so instead of passing quietly.

Run directly (no server needed; runs in-process via the local BattleStream bridge):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/poke_env_gaps/faint_attribution_fuzz_test.py [n_battles]
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.battle.gen3_battle import Gen3Battle
from agents.training.battle_recorder import BattleRecorder
from agents.training.reward_manager import Gen3RewardManager
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

_FAINT_LINE = re.compile(r"^\|faint\|(p[12])[a-c]: (.+?)\s*$")
_EVENT_RE = re.compile(r"^(our|opp):(.+):fainted$")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


@dataclass
class _Stats:
    battles: int = 0
    turns_with_faints: int = 0
    faints_seen: int = 0
    switch_in_deaths: int = 0        # THE TRIGGER: a mon that entered this turn and died
    wrong_species: int = 0
    wrong_side: int = 0
    missed: int = 0
    examples: List[dict] = field(default_factory=list)

    def record(self, ex: dict) -> None:
        if len(self.examples) < 10:
            self.examples.append(ex)

    @property
    def failures(self) -> int:
        return self.wrong_species + self.wrong_side + self.missed


class _FaintPlayer(Player):
    """Plays randomly while driving a real `BattleRecorder`, and keeps the raw protocol per turn."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("battle_class", Gen3Battle)
        super().__init__(*args, **kwargs)
        self.stats = _Stats()
        self._recorders: Dict[str, BattleRecorder] = {}
        # battle_tag -> turn -> [(player_id, species), ...] straight off the wire
        self._proto: Dict[str, Dict[int, List[Tuple[str, str]]]] = {}
        self._switched_in: Dict[str, Dict[int, Set[str]]] = {}
        self._turn: Dict[str, int] = {}
        self._me: Dict[str, str] = {}          # battle_tag -> "p1" / "p2"

    def choose_move(self, battle):
        try:
            self._observe(battle)
        except Exception as e:  # noqa: BLE001
            print("\n🛑 [FAINT-ATTRIBUTION FUZZ CRITICAL FAILURE] 🛑")
            print(f"Battle {battle.battle_tag} turn {battle.turn}: {e}")
            traceback.print_exc()
            os._exit(1)
        return self.choose_random_move(battle)

    def _species_of(self, nickname: str) -> str:
        """The canonical species behind a protocol nickname (identity when it is already one)."""
        n = _norm(nickname)
        return getattr(self, "_nick_to_species", {}).get(n, n)

    def _observe(self, battle) -> None:
        rec = self._recorders.get(battle.battle_tag)
        if rec is None:
            rec = BattleRecorder(battle.battle_tag, lambda: Gen3RewardManager(), gamma=0.99)
            self._recorders[battle.battle_tag] = rec
        import numpy as np
        mask = np.ones(11, dtype=np.int8)
        probs = np.full(11, 1.0 / 11.0, dtype=np.float32)
        rec.record(battle, 0, probs, mask, state=None)

    def _battle_finished_callback(self, battle) -> None:
        super()._battle_finished_callback(battle)
        rec = self._recorders.pop(battle.battle_tag, None)
        if rec is None:
            return
        try:
            rec.finalize(battle)
        except Exception:  # noqa: BLE001 — finalize must never take the harness down
            traceback.print_exc()
        self._check(battle, rec)

    # -- validation ------------------------------------------------------------------------
    def _check(self, battle, rec) -> None:
        s = self.stats
        s.battles += 1
        # A protocol identifier carries the NICKNAME, and this pool contains teams whose nicknames
        # are LOCALIZED species names (`Triopikeur` = Dugtrio, `Airmure` = Skarmory) — comparing it
        # to the recorder's canonical species id reported 10 false failures before this map existed.
        # poke-env has already resolved each identifier to a real species, so ask it.
        nick_to_species = {}
        for team in (getattr(battle, "team", None) or {},
                     getattr(battle, "opponent_team", None) or {}):
            for ident, mon in team.items():
                nick = ident.split(":", 1)[1].strip() if ":" in ident else ident
                sp = getattr(mon, "species", None)
                if nick and sp:
                    nick_to_species[_norm(nick)] = _norm(sp)
        self._nick_to_species = nick_to_species
        proto = _protocol_faints(battle)
        switched = _protocol_switch_ins(battle)
        me = _our_player_id(battle)

        by_turn: Dict[int, List[str]] = {}
        for inv in rec._invocations:
            turn = int(inv.get("turn") or 0)
            for ev in ((inv.get("outcome") or {}).get("events") or []):
                if _EVENT_RE.match(str(ev)):
                    by_turn.setdefault(turn, []).append(str(ev))

        for turn in sorted(set(proto) | set(by_turn)):
            truth = proto.get(turn, [])            # [(player_id, species)]
            claims = by_turn.get(turn, [])
            if not truth and not claims:
                continue
            s.turns_with_faints += 1
            s.faints_seen += len(truth)
            for pid, sp in truth:
                if self._species_of(sp) in {self._species_of(x)
                                            for x in switched.get(turn, set())}:
                    s.switch_in_deaths += 1

            truth_keys = {("our" if pid == me else "opp", self._species_of(sp))
                          for pid, sp in truth}
            claim_keys = set()
            for ev in claims:
                m = _EVENT_RE.match(ev)
                claim_keys.add((m.group(1), _norm(m.group(2))))

            for key in claim_keys - truth_keys:
                # Wrong SPECIES vs wrong SIDE are different defects; separate them.
                if any(k[1] == key[1] for k in truth_keys):
                    s.wrong_side += 1
                else:
                    s.wrong_species += 1
                s.record({"battle": battle.battle_tag, "turn": turn, "claimed": key,
                          "protocol": sorted(truth_keys), "events": claims})
            for key in truth_keys - claim_keys:
                s.missed += 1
                s.record({"battle": battle.battle_tag, "turn": turn, "missed": key,
                          "claimed": sorted(claim_keys)})


def _our_player_id(battle) -> str:
    """"p1"/"p2" for OUR side, read from poke-env's own role assignment."""
    return str(getattr(battle, "player_role", None) or "p1")


def _protocol_faints(battle) -> Dict[int, List[Tuple[str, str]]]:
    """turn -> [(player_id, species)] from the battle's own protocol log."""
    out: Dict[int, List[Tuple[str, str]]] = {}
    turn = 0
    for line in _log_lines(battle):
        if line.startswith("|turn|"):
            try:
                turn = int(line.split("|")[2])
            except (IndexError, ValueError):
                pass
            continue
        m = _FAINT_LINE.match(line)
        if m:
            out.setdefault(turn, []).append((m.group(1), m.group(2)))
    return out


def _protocol_switch_ins(battle) -> Dict[int, Set[str]]:
    """turn -> {species that entered the field that turn} — the bug's trigger."""
    out: Dict[int, Set[str]] = {}
    turn = 0
    for line in _log_lines(battle):
        if line.startswith("|turn|"):
            try:
                turn = int(line.split("|")[2])
            except (IndexError, ValueError):
                pass
            continue
        if line.startswith("|switch|") or line.startswith("|drag|"):
            parts = line.split("|")
            if len(parts) > 2 and ":" in parts[2]:
                out.setdefault(turn, set()).add(parts[2].split(":", 1)[1].strip())
    return out


def _log_lines(battle) -> List[str]:
    """The raw protocol this battle saw. `Gen3Battle` keeps the message log the recorder's
    `write_battle_record` later renders into `*_replay.html`."""
    got = getattr(battle, "_replay_data", None)
    if not got:
        raise RuntimeError(
            "no protocol on the battle — `_replay_data` is what `save_replay` renders and what "
            "this test's ground truth reads; without it every comparison below is vacuous")
    lines = []
    for row in got:
        # poke-env stores each protocol message as its SPLIT parts (['', 'faint', 'p2a: Gengar']);
        # rejoin to the wire form the |-line parsers expect.
        lines.append(row if isinstance(row, str) else "|".join(str(x) for x in row))
    return lines


async def run(n_battles: int) -> None:
    print(f"Faint-attribution Fuzz — gen3ou — {n_battles} battles\n")
    teams = TeamLoader().get_all_teams()
    teambuilder = Gen3Teambuilder(teams)
    ts = int(time.time())

    player = _FaintPlayer(
        battle_format="gen3ou", team=teambuilder,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"FaintAttr{ts}", "x"),
        max_concurrent_battles=10)
    opponent = RandomPlayer(
        battle_format="gen3ou", team=teambuilder,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"FaintAttrOpp{ts}", "x"),
        max_concurrent_battles=10)

    await run_local_battles(player, opponent, n_battles)

    s = player.stats
    print("=" * 68)
    print(f"battles                    : {s.battles}")
    print(f"turns carrying a faint     : {s.turns_with_faints}")
    print(f"faints seen (protocol)     : {s.faints_seen}")
    print(f"switch-in deaths           : {s.switch_in_deaths}   (THE TRIGGER)")
    print(f"wrong species / side / miss: {s.wrong_species} / {s.wrong_side} / {s.missed}")
    for ex in s.examples:
        print("   example:", ex)
    print("=" * 68)

    if s.failures:
        print(f"\n❌ FAIL — {s.failures} mis-attributed faint(s)")
        os._exit(1)
    if s.switch_in_deaths == 0:
        print("\n⚠️  PASS but NO switch-in death occurred — the trigger for the original bug was "
              "never exercised, so this run is INCONCLUSIVE. Re-run with more battles.")
    else:
        print(f"\n✅ PASS — every faint named the right mon across {s.switch_in_deaths} switch-in "
              f"death(s), the exact case the old prev_ctx.*_active label got wrong.")


if __name__ == "__main__":
    asyncio.run(run(int(sys.argv[1]) if len(sys.argv) > 1 else 30))
