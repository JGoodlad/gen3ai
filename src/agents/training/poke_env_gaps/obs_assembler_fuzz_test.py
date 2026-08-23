"""Incremental obs ≡ full rebuild, BIT-FOR-BIT, at every decision (`gen3_obs_assembler_v1`).

The pre-enable gate for `designs/ai_v9/design_incremental_obs_encoder.md` Stage B. Real battles
in-process via the local BattleStream bridge (no server). At EVERY decision the player runs the
production three-step protocol (`EpisodeTracker.record` → `update_progress_clock` →
`encode(..., assembler=…)`), then encodes the SAME decision a second time through the FULL
rebuild — a fresh 2,501-dim vector, all 12 slots re-encoded, all 32 event rows rewritten, the
Wish and sleep-source maps re-folded from the whole event log — and asserts
``np.array_equal``. Not a tolerance: byte-identity is the contract that licenses the swap to
be internal with no flag.

**The oracle is the full rebuild run fresh, never a second read of cache state.** The two paths
share the per-block *writers*, so what is under test is the scheduler: which writers ran, and
whether skipping one served a stale byte.

**Trigger coverage is PRINTED, and a trap the corpus never exercised says so.** A random-play
corpus does not reliably contain a Castform, a Ditto or a 250-turn stall, and a clean PASS that
silently covered none of them is the failure shape the faint-attribution fuzz was fixed for. The
census below counts the §2.3 traps as they occur and marks each ``EXERCISED`` or ``NOT SEEN`` —
the second is a statement about the corpus, not a pass.

Run directly:
    python src/agents/training/poke_env_gaps/obs_assembler_fuzz_test.py [n_battles]
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.battle.battle_event import EventKind
from agents.battle.gen3_battle import Gen3Battle
from agents.battle.live_view import LegalActions
from agents.observation.assembler import describe_offset
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.training.episode_tracker import EpisodeTracker
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

# The §2.3 trap list, as observable protocol facts. Each is counted per OCCURRENCE (not per
# battle) so "EXERCISED" means the incremental path actually had to survive it.
TRIGGERS = (
    "forme_change",       # trap 2 — Castform / Deoxys: species, base stats and types all move
    "transform",          # trap 2 — Ditto: the whole moveset moves with them
    "partial_trap",       # the wrap family: volatiles + the trapping bits
    "baton_pass",         # trap 8 — a SWITCH that KEEPS boosts/volatiles
    "double_ko",          # the alive-filter resync class (Explosion / Destiny Bond)
    "pain_split",         # SETHP — an absolute HP correction, not a delta
    "knock_off",          # trap: ITEM_TR_REMOVED
    "trick",              # trap: ITEM_TR_SWAPPED — the item lands on the OTHER side
    "berry_consumed",     # trap: ITEM_TR_CONSUMED
    "forced_switch",      # trap 7 — a sub-turn decision (turn unchanged)
    "opp_reveal",         # a new opponent slot appears mid-battle (slot re-key)
    "hp_narrowing",       # the HiddenPowerTracker revision door
    "sleep",              # the incremental sleep-source fold
    "wish",               # the incremental Wish fold
    "choice_rejected",    # the out-of-band event (no parse pass)
    "window_saturated",   # ≥32 event rows — the ring actually rotates
    "long_game",          # ≥100 turns; the 250-turn stall is its own bucket
    "stall_250",          # trap: deque caps + clock saturation at the forfeit deadline
)


@dataclass
class _Stats:
    battles: int = 0
    decisions: int = 0
    warm_decisions: int = 0
    failures: list = field(default_factory=list)
    triggers: dict = field(default_factory=lambda: {k: 0 for k in TRIGGERS})


class _AssemblerFuzzPlayer(Player):
    def __init__(self, *args, stats: _Stats, **kwargs):
        kwargs.setdefault("battle_class", Gen3Battle)
        super().__init__(*args, **kwargs)
        self.stats = stats
        self.encoder = Gen3ObservationEncoder(load_mappings())
        self._trackers: dict = {}
        self._seen_opp: dict = {}
        self._seen_events: dict = {}

    # ------------------------------------------------------------------ trigger census
    def _count_triggers(self, battle, tag, live, legal) -> None:
        s = self.stats.triggers
        seen = self._seen_events.setdefault(tag, 0)
        events = battle.events_since(seen)
        self._seen_events[tag] = battle.event_cursor
        faints_this_window = 0
        for e in events:
            k = e.kind
            if k is EventKind.FORMECHANGE:
                s["forme_change"] += 1
            elif k is EventKind.TRANSFORM:
                s["transform"] += 1
            elif k is EventKind.FAINT:
                faints_this_window += 1
            elif k is EventKind.SETHP:
                s["pain_split"] += 1
            elif k is EventKind.CHOICE_REJECTED:
                s["choice_rejected"] += 1
            elif k is EventKind.ENDITEM:
                fc = (e.from_clause or "").lower()
                if "knock off" in fc or "knockoff" in fc:
                    s["knock_off"] += 1
                elif any(w in fc for w in ("trick", "thief", "covet")):
                    s["trick"] += 1
                else:
                    s["berry_consumed"] += 1
            elif k is EventKind.MOVE:
                mid = e.move_id or ""
                if mid == "batonpass":
                    s["baton_pass"] += 1
                elif mid == "wish":
                    s["wish"] += 1
                elif mid in ("wrap", "bind", "firespin", "clamp", "whirlpool", "sandtomb"):
                    s["partial_trap"] += 1
                elif mid.startswith("hiddenpower"):
                    s["hp_narrowing"] += 1
            elif k is EventKind.STATUS and e.status == "slp":
                s["sleep"] += 1
        if faints_this_window >= 2:
            s["double_ko"] += 1
        if legal is not None and legal.force_switch:
            s["forced_switch"] += 1
        n_opp = len(live.opp.mons)
        if n_opp > self._seen_opp.get(tag, 0):
            self._seen_opp[tag] = n_opp
            if n_opp > 1:
                s["opp_reveal"] += 1

    # ------------------------------------------------------------------ the decision hook
    def choose_move(self, battle):
        tag = battle.battle_tag
        tracker = self._trackers.get(tag)
        if tracker is None:
            tracker = self._trackers[tag] = EpisodeTracker()
            self.stats.battles += 1
        legal = LegalActions.from_battle(battle)
        mask = np.ones(11, dtype=np.int8)
        tracker.record(battle, mask, legal=legal)
        tracker.update_progress_clock(battle, legal)

        kw = dict(hp_tracker=tracker.hidden_power_tracker, legal=legal,
                  progress_clock=tracker.progress_clock, recency=tracker.recency,
                  pair_history=tracker.pair_history, event_window=tracker.event_window)
        asm = tracker.obs_assembler(self.encoder.dimension)
        was_warm = asm._ready and not asm._all_dirty
        got = self.encoder.encode(battle, assembler=asm, **kw)
        want = self.encoder.encode(battle, assembler=None, **kw)

        s = self.stats
        s.decisions += 1
        if was_warm:
            s.warm_decisions += 1
        live = battle.strict_view().live
        self._count_triggers(battle, tag, live, legal)
        if len(tracker.event_window.window()) >= 32:
            s.triggers["window_saturated"] += 1
        if live.turn >= 100:
            s.triggers["long_game"] += 1
        if live.turn >= 250:
            s.triggers["stall_250"] += 1

        if not np.array_equal(got, want):
            bad = np.flatnonzero(got != want)
            first = int(bad[0])
            s.failures.append(dict(
                tag=tag, turn=live.turn, warm=was_warm, n_diff=int(bad.size),
                first=first, where=describe_offset(first),
                got=float(got[first]), want=float(want[first]),
                indices=[(int(i), describe_offset(int(i))) for i in bad[:12]],
            ))

        if battle.finished:
            self._trackers.pop(tag, None)
        return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        self._trackers.pop(battle.battle_tag, None)
        return super()._battle_finished_callback(battle)


def _account(prefix: str) -> AccountConfiguration:
    return AccountConfiguration(f"{prefix}{int(time.time()) % 10000}", None)


def main(n_battles: int = 20) -> int:
    stats = _Stats()
    teams = TeamLoader().get_all_teams()
    common = dict(team=Gen3Teambuilder(teams), battle_format="gen3ou",
                  server_configuration=LocalhostServerConfiguration,
                  start_listening=False, max_concurrent_battles=1)
    p1 = _AssemblerFuzzPlayer(stats=stats, account_configuration=_account("ObsA"), **common)
    p2 = _AssemblerFuzzPlayer(stats=_Stats(), account_configuration=_account("ObsB"), **common)
    asyncio.run(run_local_battles(p1, p2, n_battles))

    print(f"\n[obs-assembler fuzz] {n_battles} battles, {stats.decisions} decisions "
          f"({stats.warm_decisions} on the WARM path), {len(stats.failures)} byte mismatches")
    print("  trigger coverage (a trap the corpus never hit is NOT a pass for that trap):")
    for name in TRIGGERS:
        n = stats.triggers[name]
        print(f"    {name:<18} {n:>6}   {'EXERCISED' if n else 'NOT SEEN'}")
    if stats.failures:
        for f in stats.failures[:5]:
            print("  FAIL", f)
        raise SystemExit(1)
    if stats.warm_decisions == 0:
        print("  ✗ ZERO warm decisions — the incremental path never ran; this is not a pass.")
        raise SystemExit(1)
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 20))
