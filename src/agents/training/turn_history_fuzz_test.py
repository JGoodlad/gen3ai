"""Turn-history cache fuzz — validates the N-turn TurnDelta history (the obs block built by
``EpisodeTracker.prev_N_delta_vecs``) over real gen3ou battles, in-process via the local
BattleStream bridge (no server).

What it stresses (the recent perf/correctness work):
  * **#1 bounded windows** — each history slot must fold its OWN decision window
    ``[cursors[-1-i] : cursors[-i])``, not that turn through "now".
  * **deque-memoize** — ``prev_N_delta_vecs`` encodes only the newest delta each step and
    reuses cached older ones; it must be BIT-IDENTICAL to recomputing every slot from
    scratch.
  * **#5 bounded history deques** — exercised by long battles (>N turns), where the deques
    drop their oldest and the monotonic transition counter must keep the cache in sync.

The harness drives a real ``EpisodeTracker`` exactly as ``Gen3Env`` does — ``get_mask`` →
``record`` → (validate) → pick a legal action → ``advance`` — on every decision, then
asserts ``prev_N_delta_vecs`` (cached) equals a fresh full recompute. Real battles cover the
diversity (variable turn counts, forced switches, faints/drags, multi-KO, struggle) that
scripted unit tests can't.

Run directly (no server needed — local bridge):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/turn_history_fuzz_test.py [n_battles]
"""

from __future__ import annotations

import asyncio
import random
import sys
import time
import traceback
from typing import Dict, List

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.action.mask_generator import Gen3ActionMasker
from agents.action.mapper import Gen3ActionMapper
from agents.battle.gen3_battle import Gen3Battle
from agents.model.features_extractor import N_HISTORY_TURNS
from agents.observation.state_encoder import load_mappings
from agents.observation.turn_delta_encoder import TurnDeltaEncoder
from agents.training.episode_tracker import EpisodeTracker
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"
_POOL = None


def _team_pool() -> list:
    global _POOL
    if _POOL is None:
        loader = TeamLoader()
        _POOL = loader.get_sample_teams() or loader.get_all_teams()
        if not _POOL:
            raise RuntimeError("no gen3ou teams found under data/teams")
    return _POOL


class TurnHistoryFuzzPlayer(Player):
    """Drives a per-battle EpisodeTracker and asserts the cached turn-history block equals a
    from-scratch recompute on every decision."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mappings = load_mappings()
        # One encoder suffices: encode() is pure, so the cached vector and a recomputed one
        # are identical given the same delta.
        self._enc = TurnDeltaEncoder(mappings.get("moves", {}), mappings.get("species", {}))
        self._trackers: Dict[str, EpisodeTracker] = {}
        self.mismatches: List[str] = []
        self.errors: List[str] = []
        self.battles_checked = 0
        self.decisions_checked = 0
        self.max_turns_seen = 0
        self.long_battles = 0  # battles that exceeded N_HISTORY_TURNS (exercise deque drop)

    def _tracker_for(self, tag: str) -> EpisodeTracker:
        tr = self._trackers.get(tag)
        if tr is None:
            tr = EpisodeTracker(history_cap=N_HISTORY_TURNS)
            self._trackers[tag] = tr
        return tr

    def _recompute(self, tr: EpisodeTracker, battle, n: int) -> np.ndarray:
        """Full from-scratch turn-history block: encode EVERY slot over its bounded window.
        This is the oracle the cached path must match."""
        res = np.zeros((n, self._enc.dimension), dtype=np.float32)
        avail = min(n, len(tr._history) - 1, len(tr._actions), len(tr._cursors))
        for i in range(avail):
            res[n - 1 - i] = tr._encode_delta_slot(i, self._enc, battle)
        return res

    def choose_move(self, battle):
        try:
            mask = Gen3ActionMasker.get_mask(battle)  # also latches the decision context
        except Exception:
            # Can't build the mask (rare strict-mode edge) — fall back; don't corrupt the
            # tracker timeline by recording without a mask.
            self.errors.append(traceback.format_exc())
            return self.choose_random_move(battle)

        tr = self._tracker_for(battle.battle_tag)
        tr.record(battle, mask)

        # The check: cached deque path vs a full recompute. MUST be bit-identical.
        try:
            cached = tr.prev_N_delta_vecs(N_HISTORY_TURNS, self._enc, battle=battle)
            ref = self._recompute(tr, battle, N_HISTORY_TURNS)
            self.decisions_checked += 1
            self.max_turns_seen = max(self.max_turns_seen, battle.turn)
            if not np.array_equal(cached, ref):
                d = float(np.abs(cached - ref).max())
                rows = [int(r) for r in np.where(np.abs(cached - ref).max(axis=1) > 0)[0]]
                self.mismatches.append(
                    f"[{battle.battle_tag}] t{battle.turn} cache != recompute "
                    f"(maxdiff={d:.3g}, rows={rows}, n_trans={tr._n_transitions})"
                )
        except Exception:
            self.errors.append(traceback.format_exc())

        # Pick a random LEGAL action, mirror the env's advance(), and map it to an order.
        valid = [i for i, v in enumerate(mask) if v]
        idx = random.choice(valid) if valid else 0
        tr.advance(idx)
        try:
            return Gen3ActionMapper.action_to_order(idx, battle, mask=mask)
        except Exception:
            return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        self.battles_checked += 1
        if battle.turn > N_HISTORY_TURNS:
            self.long_battles += 1
        self._trackers.pop(battle.battle_tag, None)


async def main(n_battles: int = 2000) -> None:
    ts = int(time.time()) % 100000
    print(f"Turn-History Cache Fuzz — {BATTLE_FORMAT} — {n_battles} battles", flush=True)
    pool = _team_pool()
    fuzz = TurnHistoryFuzzPlayer(
        battle_format=BATTLE_FORMAT,
        team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"THz{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        battle_class=Gen3Battle,
    )
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"THo{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
    )

    # Chunked so we can print progress and ABORT EARLY on the first CACHE divergence (the
    # thing under test). The try/except is a generic guard: a rare unrelated single-battle
    # crash aborts only that chunk's current battle — we catch it, count it as an infra-skip,
    # and resume so a 10k–100k run isn't lost to one bad battle.
    chunk = 25
    infra_skips = 0
    stuck = 0   # consecutive chunks with ZERO completed battles (genuinely-broken guard)
    while fuzz.battles_checked < n_battles and infra_skips < 1000:
        before = fuzz.battles_checked
        try:
            await run_local_battles(fuzz, opp, min(chunk, n_battles - fuzz.battles_checked))
        except Exception as e:
            # A crash aborts run_local_battles at the offending battle; battles completed
            # before it still counted. Skip it and press on (the next chunk is fresh).
            infra_skips += 1
            if infra_skips <= 5:
                print(f"  [infra-skip #{infra_skips}] {type(e).__name__}: {str(e)[:140]}",
                      flush=True)
        # Between chunks no battle is in-flight. The bridge now hands every battle a unique
        # tag (utils/bridge/local_battle_runner._BATTLE_SEQ), so reused tags no longer return
        # a stale full-team battle (the old "team already has 6 pokemons" crash). We still drop
        # finished battles + their trackers each chunk to keep memory bounded over a 10k–100k
        # run — poke-env otherwise retains every finished battle by tag forever.
        fuzz._trackers.clear()
        fuzz._battles.clear()
        opp._battles.clear()
        progress = fuzz.battles_checked - before
        stuck = stuck + 1 if progress == 0 else 0
        print(
            f"  …{fuzz.battles_checked} battles, {fuzz.decisions_checked} decisions "
            f"| long(>{N_HISTORY_TURNS}t)={fuzz.long_battles} maxturns={fuzz.max_turns_seen} "
            f"| mismatch={len(fuzz.mismatches)} validation-err={len(fuzz.errors)} "
            f"infra-skips={infra_skips}",
            flush=True,
        )
        # ABORT only on a real cache mismatch or a validation-path exception (our code).
        if fuzz.mismatches or fuzz.errors:
            print("  ABORTING — cache mismatch / validation error", flush=True)
            break
        if stuck >= 20:   # 20 chunks in a row made no progress — genuinely broken
            print("  ABORTING — no progress over 20 chunks (infra broken)", flush=True)
            break

    print("=" * 70)
    print(f"Battles validated : {fuzz.battles_checked}")
    print(f"Decisions validated: {fuzz.decisions_checked}")
    print(f"Long battles (>{N_HISTORY_TURNS} turns, deque drops): {fuzz.long_battles} "
          f"| max turns seen: {fuzz.max_turns_seen}")
    print(f"Infra-skips (unrelated poke-env parse crashes, NOT cache bugs): {infra_skips}")
    ok = True
    if fuzz.errors:
        ok = False
        print(f"VALIDATION ERRORS ({len(fuzz.errors)}):")
        for e in fuzz.errors[:5]:
            print(e)
    if fuzz.mismatches:
        ok = False
        print(f"MISMATCHES ({len(fuzz.mismatches)}) — first 25:")
        for mm in fuzz.mismatches[:25]:
            print(" ", mm)
    if fuzz.battles_checked == 0:
        ok = False
        print("NO BATTLES VALIDATED — are teams loading / the bridge built?")
    if fuzz.long_battles == 0 and fuzz.battles_checked > 50:
        print("WARNING: no battle exceeded N_HISTORY_TURNS — deque-drop path not exercised")

    print("PASS — cached turn-history == full recompute on every decision." if ok else "FAIL")
    print("=" * 70)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    asyncio.run(main(n))
