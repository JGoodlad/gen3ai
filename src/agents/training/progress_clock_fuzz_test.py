"""Bridge fuzz test for the ProgressClock anti-stall predicate — the heal-war charge + the
winning-residual guard (the two behaviours added with the --bias-redesign no-progress charge).

Unit tests (`reward_progress_clock_test.TestProgressClock`) pin the PROGRESS / DENIED / NO_OP logic on
hand-built deltas. This drives the REAL pipeline end-to-end —
``Gen3Env → EpisodeTracker → event-sourced TurnDelta fold → ProgressClock`` — over real Showdown
battles via the in-process BattleStream bridge (no server), instrumenting the LIVE clock at every
decision window. It is the only way to catch a poke-env-shaped gap the mock unit tests can't see:
a status/volatile id the residual branch misreads, an ``opp_hp_delta`` the fold attributes wrong, a
real state that makes the new branch raise.

Per-window invariants (a violation raises immediately):
  1. NO CRASH — the clock + the new ``_is_progress`` residual branch + ``_denial_kind`` never raise
     on a real state (implicit: any raise aborts the episode).
  2. RANGES — ``last_penalty ∈ {0, -no_progress_penalty}``; ``n ∈ [0, PROGRESS_CLOCK_CAP]``;
     ``_heal_streak ≥ 0``.
  3. WINNING-RESIDUAL GUARD (the part-1 regression guard) — a window where an our-owned residual
     (Toxic / poison / burn status, or Leech Seed / Curse / Nightmare on the opp active) chipped the
     opp NET-down (``opp_hp_delta.sum() ≤ -PROGRESS_DMG_EPS``) is NEVER charged. A winning defensive
     stall must not be taxed — that is exactly the failure the fix had to avoid (charging a Suicune-
     Rest / Milotic-Recover endgame the Toxic clock is winning).

Scenario coverage (printed; soft — a run that never hits the path warns but does not fail):
  4. The no-progress penalty actually FIRES on real no-op windows (the mechanism is live).
  5. Residual-progress windows are actually observed (the guarded path was exercised).

Run directly (no server needed; in-process via the local BattleStream bridge):
    python src/agents/training/progress_clock_fuzz_test.py [n_battles]
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import sys
import traceback
from collections import Counter

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

from agents.observation.state_encoder import load_mappings
from agents.training.gen3_env import Gen3Env
from agents.training.progress_clock import PROGRESS_CLOCK_CAP, PROGRESS_DMG_EPS, HEAL_FREEZE_GRACE
from utils.bridge.bridge_session import attach_bridge_transport
from utils.team_loader.loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"
_OWNED_RESIDUAL_STATUS = ("tox", "psn", "brn")
_OWNED_RESIDUAL_VOLATILES = ("leechseed", "curse", "nightmare")


def _teams():
    loader = TeamLoader()
    return loader.get_sample_teams() or loader.get_all_teams()


def _build_bridge_env(teams, idx: int):
    """A real Gen3Env on the bridge transport + a RandomPlayer opponent — the training factory's
    bridge path, minimal. The env owns the EpisodeTracker whose ProgressClock we instrument."""
    env = Gen3Env(
        load_mappings(),
        battle_format=BATTLE_FORMAT,
        team=Gen3Teambuilder(teams),
        account_configuration1=AccountConfiguration(f"ClockFuzz{idx}", None),
        start_listening=False,
    )
    attach_bridge_transport(env, battle_format=BATTLE_FORMAT, persistent=True, recycle_every=10000)
    opponent = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=Gen3Teambuilder(teams),
        account_configuration=AccountConfiguration(f"ClockFuzzOpp{idx}", None),
        start_listening=False,
    )
    wrapped = SingleAgentWrapper(env, opponent)
    wrapped.action_space = env.action_space
    wrapped.observation_space = env.observation_space
    return wrapped, env


def _instrument_clock(env, windows: list):
    """Wrap the live ProgressClock.update to record (delta-derived + result) for every real window.
    The env reuses one clock across episodes (reset() only zeroes it), so wrapping once persists."""
    clock = env._tracker._progress_clock
    orig = clock.update

    def wrapped(delta, live, legal, **kw):
        # **kw passes `legal_prev` through untouched — the OPENING decision's legality, read only
        # under --progress-decision-tense. Swallowing it here would silently fuzz a different clock.
        orig(delta, live, legal, **kw)   # run the REAL logic (sets last_penalty / n / _heal_streak)
        opp_mon = getattr(getattr(live, "opp", None), "active", None) if live is not None else None
        status = getattr(opp_mon, "status", None) if opp_mon is not None else None
        vols = (getattr(opp_mon, "volatiles", None) or ()) if opp_mon is not None else ()
        owned_residual = (status in _OWNED_RESIDUAL_STATUS
                          or any(v in vols for v in _OWNED_RESIDUAL_VOLATILES))
        try:
            opp_net = float(delta.opp_hp_delta.sum())
        except (AttributeError, TypeError):
            opp_net = 0.0
        windows.append({
            "last_penalty": float(clock.last_penalty),
            "n": int(clock.n),
            "heal_streak": int(getattr(clock, "_heal_streak", 0)),
            "owned_residual": bool(owned_residual),
            "opp_net": opp_net,
            "forced_switch": bool(getattr(delta, "phase_is_forced_switch", False)),
        })

    clock.update = wrapped
    return clock


def _play_episode(wrapped, rng, max_steps: int = 600) -> int:
    obs, _ = wrapped.reset()
    for step in range(max_steps):
        mask = np.asarray(obs["action_mask"]).astype(bool)
        legal = np.flatnonzero(mask)
        action = int(rng.choice(legal)) if legal.size else 0
        obs, _r, terminated, truncated, _info = wrapped.step(action)
        if terminated or truncated:
            return step + 1
    raise AssertionError(f"episode did not finish within {max_steps} steps")


def _assert_window(w, charge_mag: float, battle_idx: int, win_idx: int):
    lp, n, hs = w["last_penalty"], w["n"], w["heal_streak"]
    # (2) ranges
    assert n == max(0, min(n, PROGRESS_CLOCK_CAP)) and 0 <= n <= PROGRESS_CLOCK_CAP, \
        f"battle {battle_idx} window {win_idx}: n={n} out of [0,{PROGRESS_CLOCK_CAP}]"
    assert hs >= 0, f"battle {battle_idx} window {win_idx}: _heal_streak={hs} < 0"
    assert abs(lp) < 1e-9 or abs(abs(lp) - charge_mag) < 1e-6, \
        f"battle {battle_idx} window {win_idx}: last_penalty={lp} not in {{0, -{charge_mag}}}"
    # (3) the winning-residual guard: opp dying to OUR residual must NOT be charged
    if w["owned_residual"] and w["opp_net"] <= -PROGRESS_DMG_EPS and not w["forced_switch"]:
        assert lp == 0.0, (
            f"WINNING-RESIDUAL GUARD VIOLATED — battle {battle_idx} window {win_idx}: our residual "
            f"chipped the opp net {w['opp_net']:+.3f} yet the clock charged last_penalty={lp}. A "
            f"winning defensive stall must never be taxed.")


def main(n_battles: int = 40) -> int:
    teams = _teams()
    rng = np.random.default_rng(0)
    wrapped, env = _build_bridge_env(teams, idx=1)
    windows: list = []
    _instrument_clock(env, windows)
    charge_mag = abs(env._tracker._progress_clock.no_progress_penalty)

    stats = Counter()
    t0 = __import__("time").time()
    try:
        win_cursor = 0
        for b in range(n_battles):
            steps = _play_episode(wrapped, rng)
            stats["battles"] += 1
            stats["steps"] += steps
            # validate the windows produced by THIS battle
            for wi in range(win_cursor, len(windows)):
                w = windows[wi]
                _assert_window(w, charge_mag, b, wi - win_cursor)
                stats["windows"] += 1
                if w["last_penalty"] < 0:
                    stats["charged"] += 1
                if w["owned_residual"] and w["opp_net"] <= -PROGRESS_DMG_EPS and not w["forced_switch"]:
                    stats["residual_progress"] += 1
                stats["max_heal_streak"] = max(stats["max_heal_streak"], w["heal_streak"])
            win_cursor = len(windows)
    except Exception:
        traceback.print_exc()
        wrapped.close()
        print("\n❌ FAIL — see traceback above")
        return 1
    finally:
        try:
            wrapped.close()
        except Exception:
            pass

    dt = __import__("time").time() - t0
    print(f"\n=== ProgressClock bridge fuzz — {stats['battles']} battles, {stats['windows']} windows, "
          f"{dt:.0f}s ===")
    print(f"  charged no-op windows (penalty fired) : {stats['charged']}")
    print(f"  winning-residual windows (guard held) : {stats['residual_progress']}")
    print(f"  max _heal_streak observed             : {stats['max_heal_streak']} "
          f"(HEAL_FREEZE_GRACE={HEAL_FREEZE_GRACE})")
    print(f"  avg steps/battle                      : {stats['steps'] / max(1, stats['battles']):.1f}")

    # Hard invariants already enforced per-window above. Soft coverage warnings:
    ok = True
    if stats["charged"] == 0:
        print("  ⚠️  coverage: the no-progress penalty NEVER fired — random play made progress every "
              "window; the charge path was not exercised (not a failure, but weak coverage).")
    if stats["residual_progress"] == 0:
        print("  ⚠️  coverage: NO winning-residual window occurred (our side never chipped the opp "
              "down with Toxic/Leech while they survived) — the guard held vacuously. Re-run with "
              "more battles to exercise it.")
    else:
        print(f"  ✅ winning-residual guard exercised on {stats['residual_progress']} real windows — "
              f"none charged.")
    print("\n✅ PASS — clock ran on every real window without crashing; ranges held; no winning-"
          "residual window was ever charged." if ok else "")
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    sys.exit(main(n))
