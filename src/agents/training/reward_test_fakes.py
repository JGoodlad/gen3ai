"""Shared fakes for the reward-manager unit tests — NOT a test module.

The Markovian / PBRS reward redesign (design_markovian_reward_and_features.md) is specified by a
family of per-term spec files (`reward_registry_test.py`, `reward_pbrs_*_test.py`,
`reward_bias_terms_test.py`, `reward_end_state_test.py`, `reward_progress_clock_test.py`). They all
need the same minimal LiveView / battle / TurnDelta stubs, so those live here once.

The name deliberately does NOT match pytest's `python_files` patterns (`*_test.py`), so this module
is imported, never collected.
"""
import numpy as np

from agents.training.reward_manager import Gen3RewardManager, RewardConfig
from agents.training.progress_clock import ProgressClock


def _mgr_additive_bias(**kw):
    """A manager in the ADDITIVE-BIAS regime — i.e. `--no-all-shaping-pbrs`.

    Since 2026-08-18 the DEFAULT config is `--all-shaping-pbrs` (the validated ai_v8 composition):
    the four end-state potentials FOLD and every BIAS term but `no_progress_tax` is ZEROED. A test
    that wants to see a BIAS term fire, or to observe the potentials' OFF branch, must therefore
    STATE the fallback regime rather than inherit it from a default that no longer means that.

    `src/main/reward_defaults_test.py` owns the default composition itself; this helper keeps the
    per-term mechanics tests readable about which regime they exercise.
    """
    kw.setdefault("config", RewardConfig(all_shaping_pbrs=False))
    return Gen3RewardManager(**kw)


# --------------------------------------------------------------------------- #
# Minimal LiveView / battle / delta stubs for Φ_mat + the fold.                 #
# --------------------------------------------------------------------------- #
class _Mon:
    """Rich enough for the full process_turn_reward fold (status/se/dead-matchup/belief helpers)."""
    def __init__(self, hp_fraction=1.0, species="mon", active=False):
        self.hp_fraction = hp_fraction
        self.fainted = hp_fraction <= 0.0
        self.species = species
        self.active = active
        self.status = None
        self.volatiles = {}
        self.types = ()
        self.move_ids = ()
        self.boosts = {}
        self.ability = None
        self.item = None
        self.consumed_item = None
        self.stats = {}
        self.current_hp = None
        self.max_hp = None


class _Side:
    def __init__(self, hps, team_size=6, spikes=0):
        self.mons = tuple(_Mon(h, active=(i == 0)) for i, h in enumerate(hps))
        self.active = self.mons[0] if (self.mons and not self.mons[0].fainted) else None
        self.team_size = team_size
        self.side_conditions = {"spikes": spikes} if spikes else {}


class _Live:
    def __init__(self, our_hps, opp_hps, opp_team_size=6, won=False, lost=False, finished=False):
        self.ours = _Side(our_hps)
        self.opp = _Side(opp_hps, team_size=opp_team_size)
        self.weather = None
        self.won, self.lost, self.finished = won, lost, finished


class _Battle:
    def __init__(self, live, turn=1):
        self._live = live
        self.turn = turn
        live.turn = turn   # mirror the real LiveView (its .turn comes from battle.turn) — the
                           # terminal block reads live.turn to detect the stall TIMEOUT (turn>=cap).
        self.won = live.won
        self.lost = live.lost
        self.finished = live.finished

    def live_view(self):
        return self._live


def _delta(**kw):
    """A minimal TurnDelta-like object exposing only the fields the fold/clock read."""
    from types import SimpleNamespace
    base = dict(
        our_hp_delta=np.zeros(6, dtype=np.float32), opp_hp_delta=np.zeros(6, dtype=np.float32),
        our_boost_delta=np.zeros(7, dtype=np.int8),
        we_fainted=False, opp_fainted=False, our_move_id=None, our_switch_to=None,
        opp_switch_to=None, opp_damaging_event=None, our_damaging_event=None,
        opp_target_hp_delta=None, our_move_outcome=None, our_failed_to_move=False,
        our_effectiveness=1.0,
        our_cant_reason=None, opp_status_applied=None, opp_resolved_move_id=None,
        phase_is_forced_switch=False, our_status_applied=None,
        our_status_cured=None, opp_status_cured=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class _Legal:
    def __init__(self, switches=()):
        self.switches = tuple(switches)


def _full_team_live(our_alive=6, opp_alive=6, our_hp=1.0, opp_hp=1.0, **kw):
    """A 6v6 LiveView with `our_alive`/`opp_alive` mons at the given HP, rest fainted."""
    our = [our_hp] * our_alive + [0.0] * (6 - our_alive)
    opp = [opp_hp] * opp_alive + [0.0] * (6 - opp_alive)
    return _Live(our, opp, **kw)


def _mgr_pbrs(**cfg):
    """A FULLY-PBRS manager (both end-state switches ON) + a real ProgressClock, so every new
    potential is live — incl. Φ_progress, which gates on stall_pbrs (the 'stall' switch)."""
    return Gen3RewardManager(config=RewardConfig(all_shaping_pbrs=True, stall_pbrs=True, **cfg),
                             progress_clock=ProgressClock())
