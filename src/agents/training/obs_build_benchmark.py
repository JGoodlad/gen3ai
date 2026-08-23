"""Per-decision observation-build benchmark — where does the obs encode spend its time?

This is a **benchmark, not a pass/fail test** (no ``test_*`` funcs, so ``pytest`` imports it
but collects nothing — same convention as ``*_fuzz_test.py``). It plays a real ``gen3ou``
battle **in-process via the local BattleStream bridge** (no server) until a representative
late-game decision is reached, then measures the cost of building one observation:

  * ``state_encoder.encode``             — the full 3299-dim obs vector
  * ``battle.live_view()``               — the current-board read-model

It prints a component wall-clock breakdown plus a ``cProfile`` ``tottime`` ranking so you can
see which functions dominate. Use it to catch obs-pipeline perf regressions and to confirm an
optimization actually moved the bottleneck.

It lives in ``training/`` (beside the obs fuzz tests, which validate the same
pipeline) rather than ``observation/`` on purpose: a directly-run script puts its own
directory on ``sys.path[0]``, and ``observation/types.py`` would then shadow the stdlib
``types`` module (circular import). ``training/`` has no stdlib-shadowing names.

**Reading the numbers:** absolute ms scale with machine load (a busy box inflates them); the
*ratios* between components and the ``cProfile`` call-count ranking are what stay meaningful
across runs. For a clean baseline, run it on an otherwise-idle machine.

Run directly (no server needed — local bridge):
    python src/agents/training/obs_build_benchmark.py                       # defaults
    python src/agents/training/obs_build_benchmark.py --turn 25 --reps 400 --top 22
    python src/agents/training/obs_build_benchmark.py --battles 400 --seed 0
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import io
import pstats
import random
import sys
import time
from typing import Optional


from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.action.mapper import Gen3ActionMapper
from agents.action.mask_generator import Gen3ActionMasker
from agents.battle.gen3_battle import Gen3Battle
from agents.observation.state_encoder import get_observation_encoder, load_mappings
from agents.training.episode_tracker import EpisodeTracker
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"


def _mean_ms(fn, reps: int) -> float:
    """Mean wall-clock per call over ``reps`` iterations, in milliseconds."""
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps * 1e3


def profile_obs_build(battle, tracker, obs_enc, *, reps: int, top: int) -> dict:
    """Time the per-decision obs build on a single (battle, tracker) decision and print a
    report. Returns the component timings (ms) as a dict so callers can assert/track them."""
    hpt = tracker.hidden_power_tracker
    n_opp_rev = sum(1 for m in battle.opponent_team.values() if m.moves)

    _kw = dict(hp_tracker=hpt, progress_clock=tracker.progress_clock,
               recency=tracker.recency, pair_history=tracker.pair_history,
               event_window=tracker.event_window)
    # gen3_frame_deletion_v1: the obs build IS `encode` now — the turn-history component and
    # its deque cache are deleted with the lag frames, so `full` and `enc_only` coincide. Both
    # are kept and reported so the percentages stay comparable to the archived baselines in
    # `src/agents/observation/CLAUDE.md`, and so a future appended block shows up as a gap.
    full = lambda: obs_enc.encode(battle, **_kw)
    enc_only = lambda: obs_enc.encode(battle, **_kw)
    live_view = lambda: battle.live_view()

    for _ in range(10):  # warm caches / JIT-y bits
        full()

    t_full = _mean_ms(full, reps)
    t_enc = _mean_ms(enc_only, reps)
    t_lv = _mean_ms(live_view, reps)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(reps):
        full()
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(top)

    pct = lambda x: f"({x / t_full * 100:4.0f}%)" if t_full else "( n/a)"
    print("\n" + "=" * 78)
    print(f"PER-DECISION OBS BUILD BENCHMARK  (obs dim {obs_enc.dimension}, turn {battle.turn}, "
          f"opp mons w/ revealed moves {n_opp_rev}/6)")
    print("=" * 78)
    print(f"  full per-decision obs build  : {t_full:7.3f} ms")
    print(f"    state_encoder.encode       : {t_enc:7.3f} ms  {pct(t_enc)}")
    print(f"    [live_view() alone         : {t_lv:7.3f} ms  {pct(t_lv)}]")
    print("\n  NOTE: absolute ms scale with machine load; the component ratios and the")
    print("        cProfile ranking below are the load-stable signal.")
    print(f"\n  Top {top} functions by tottime (cumtime shows the caller chain):")
    print(buf.getvalue())

    return {
        "obs_dim": obs_enc.dimension,
        "turn": battle.turn,
        "full_ms": t_full,
        "encode_ms": t_enc,
        "live_view_ms": t_lv,
    }


class _BenchmarkPlayer(Player):
    """Drives a per-battle ``EpisodeTracker`` exactly as ``Gen3Env`` does, and profiles the
    first decision that is deep enough to be representative (turn >= ``profile_at_turn`` with a
    full history window and a revealed opponent active)."""

    def __init__(self, *args, profile_at_turn: int, reps: int, top: int, **kwargs):
        super().__init__(*args, **kwargs)
        maps = load_mappings()
        self.obs_enc = get_observation_encoder(maps)
        self._trackers: dict = {}
        self._profile_at_turn = profile_at_turn
        self._reps = reps
        self._top = top
        self.result: Optional[dict] = None

    def choose_move(self, battle):
        mask = Gen3ActionMasker.get_mask(battle)
        tr = self._trackers.setdefault(
            battle.battle_tag, EpisodeTracker(history_cap=1))
        tr.record(battle, mask)
        # The env's full 3-step protocol, so the tracker-fed blocks (progress clock, recency,
        # H-A, the H-B event window) are LIVE in what gets measured — until 2026-08-16 this
        # call was missing and the benchmark silently timed those writes as skipped.
        tr.update_progress_clock(battle, None)

        if (self.result is None
                and battle.turn >= self._profile_at_turn
                and len(tr._history) > 1
                and battle.opponent_active_pokemon is not None):
            self.result = profile_obs_build(
                battle, tr, self.obs_enc, reps=self._reps, top=self._top)

        valid = [i for i, v in enumerate(mask) if v]
        idx = random.choice(valid) if valid else 0
        tr.advance(idx)
        try:
            return Gen3ActionMapper.action_to_order(idx, battle, mask=mask)
        except Exception:
            return self.choose_random_move(battle)


async def main(battles: int, profile_at_turn: int, reps: int, top: int, seed: int) -> int:
    random.seed(seed)
    ts = int(time.time()) % 100000
    pool = TeamLoader().get_sample_teams() or TeamLoader().get_all_teams()
    if not pool:
        print("no gen3ou teams found under data/teams", file=sys.stderr)
        return 1

    bench = _BenchmarkPlayer(
        profile_at_turn=profile_at_turn, reps=reps, top=top,
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"OBz{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        battle_class=Gen3Battle)
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"OBo{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False)

    print(f"Obs-build benchmark — {BATTLE_FORMAT} — searching for a turn>={profile_at_turn} "
          f"decision (≤{battles} battles, reps={reps}, top={top}, seed={seed})", flush=True)
    i = 0
    while bench.result is None and i < battles:
        # The bridge assigns each battle a process-unique tag
        # (local_battle_runner._BATTLE_SEQ), so repeated single-battle calls never collide
        # on a reused tag — no per-call cleanup needed (and we stop at the first deep
        # decision, so retained battles never accumulate enough to matter).
        await run_local_battles(bench, opp, 1)
        i += 1

    if bench.result is None:
        print(f"never reached a turn>={profile_at_turn} decision in {battles} battles "
              f"— raise --battles or lower --turn", file=sys.stderr)
        return 1
    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Per-decision observation-build benchmark (bridge).")
    p.add_argument("--battles", type=int, default=200,
                   help="max battles to play while searching for a deep decision (default 200)")
    p.add_argument("--turn", type=int, default=25, dest="profile_at_turn",
                   help="profile the first decision at turn >= this (default 25)")
    p.add_argument("--reps", type=int, default=400,
                   help="timing iterations per component (default 400)")
    p.add_argument("--top", type=int, default=22,
                   help="number of cProfile functions to show (default 22)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for action selection — partial reproducibility (default 0)")
    return p.parse_args(argv)


if __name__ == "__main__":
    # A benchmark on a busy box reports a confidently wrong number — say so up front.
    from utils.contention import warn_if_contended
    warn_if_contended("obs-build benchmark")
    a = _parse_args(sys.argv[1:])
    sys.exit(asyncio.run(
        main(a.battles, a.profile_at_turn, a.reps, a.top, a.seed)))
