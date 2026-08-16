"""Deterministic golden-observation capture — the value-neutrality reference.

Plays a FIXED set of bridge battles (no server) and records the full per-decision
observation vector for the trainee side, assembled exactly as ``Gen3Env.embed_battle``
does (base ``encode`` + prev-mask + N-turn history).

**Determinism.** A live bridge battle is normally NOT reproducible across process runs:
both players share Python's global ``random`` and the async bridge interleaves their
``choose_move`` calls nondeterministically, so the RNG draw order — and the whole
trajectory — diverges (empirically the decision count swings by hundreds). This harness
removes every randomness source so the trajectory is a pure function of (teams, sim seed):

  * **policy** — a deterministic turn-based rotation over the legal actions (no RNG), used
    by BOTH players, so who-is-asked-first can't change what either picks;
  * **teams** — a teambuilder that *cycles* its pool in order instead of ``random.choice``;
  * **sim** — a fixed gen-5 PRNG ``seed`` per battle.

With those pinned the capture is byte-identical run to run, so the per-decision hashes make
a valid golden. Used to prove the ``gen3_data`` refactor (single data facade; type-chart/
natures sourced from ``data/`` instead of poke-env) changes **no observed value**: capture on
HEAD, assert byte-identical after every phase. See ``gen3_data_obs_parity_integration_test.py``.

Lives in ``training/`` (not ``observation/``) for the same reason as the benchmark: a script
run from ``observation/`` would shadow the stdlib ``types`` module (``observation/types.py``).
"""
from __future__ import annotations

import asyncio
import hashlib
import sys
from typing import List

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.action.mapper import Gen3ActionMapper
from agents.action.mask_generator import Gen3ActionMasker
from agents.battle.gen3_battle import Gen3Battle
from agents.battle.live_view import LegalActions
from agents.model.features_extractor import N_HISTORY_TURNS
from agents.observation.state_encoder import get_observation_encoder, load_mappings
from agents.observation.turn_delta_encoder import TurnDeltaEncoder
from agents.training.episode_tracker import EpisodeTracker
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"
# Fixed knobs — changing any of these invalidates the committed fixture.
BRIDGE_SEED = [1, 2, 3, 4]   # gen-5 sim PRNG seed (per battle)
N_BATTLES = 6
N_TEAMS = 6            # pool slice — kept small so per-team validation stays fast


def _det_action(battle, mask) -> int:
    """Deterministic policy: rotate through the legal actions by turn number. No RNG, and a
    pure function of this player's own battle state, so it is insensitive to async interleave."""
    valid = [i for i, v in enumerate(mask) if v]
    if not valid:
        return 0
    return valid[battle.turn % len(valid)]


class _CyclingTeambuilder(Gen3Teambuilder):
    """``yield_team`` cycles the validated pool in order instead of ``random.choice`` — so team
    selection draws no randomness and is identical every run."""

    def __init__(self, teams):
        super().__init__(teams)
        self._i = 0

    def yield_team(self):
        team = self.packed_teams[self._i % len(self.packed_teams)]
        self._i += 1
        return team


class _DetPlayer(Player):
    """Deterministic player. ``record`` is set on the trainee instance to also archive the
    per-decision obs vector (assembled exactly as the env does)."""

    def __init__(self, *args, record: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._record = record
        if record:
            maps = load_mappings()
            self.obs_enc = get_observation_encoder(maps)
            self.td_enc = TurnDeltaEncoder(maps.get("moves", {}), maps.get("species", {}))
            self._tracker: EpisodeTracker | None = None
            self._last_tag = None
            self.vectors: List[np.ndarray] = []

    def choose_move(self, battle):
        legal = LegalActions.from_battle(battle)
        mask = Gen3ActionMasker.get_mask(battle, legal=legal).astype(np.int8)

        if self._record:
            if battle.battle_tag != self._last_tag:   # one tracker per episode, as the env does
                self._tracker = EpisodeTracker(history_cap=N_HISTORY_TURNS)
                self._last_tag = battle.battle_tag
            tr = self._tracker
            if mask.sum() > 0:
                tr.record(battle, mask, legal=legal)
            # The env's 3-step protocol (record → update_progress_clock → encode), so the
            # tracker-fed blocks (progress clock, recency, H-A last-action/pair-history, the
            # H-B event window) are LIVE in the golden instead of asserting zeros — until
            # 2026-08-16 this call was missing and the golden silently ignored all of them.
            tr.update_progress_clock(battle, legal)
            obs = self.obs_enc.encode(battle, hp_tracker=tr.hidden_power_tracker, legal=legal,
                                      progress_clock=tr.progress_clock, recency=tr.recency,
                                      pair_history=tr.pair_history,
                                      event_window=tr.event_window)
            history = tr.prev_N_delta_vecs(N_HISTORY_TURNS, self.td_enc, battle=battle)
            full = np.concatenate([obs, tr.prev_mask, history.flatten()]).astype(np.float32)
            self.vectors.append(full)

        idx = _det_action(battle, mask)
        if self._record:
            self._tracker.advance(idx)
        try:
            return Gen3ActionMapper.action_to_order(idx, battle, mask=mask)
        except Exception:
            return self.choose_random_move(battle)


async def _capture() -> List[np.ndarray]:
    # The obs encoder memoizes move category in a process-global cache keyed by move id, off the
    # LIVE move.category. A prior test that encoded a mock move (real id, mock category) would
    # leave a wrong entry there and perturb a value here. Clear it so the capture always reflects
    # the real data — making this golden reliable regardless of suite ordering.
    from agents.observation import moves as _moves_enc
    _moves_enc._CATEGORY_VAL_CACHE.clear()

    pool = (TeamLoader().get_sample_teams() or TeamLoader().get_all_teams())[:N_TEAMS]
    if not pool:
        raise RuntimeError("no gen3ou teams under data/teams")
    p1 = _DetPlayer(
        record=True,
        battle_format=BATTLE_FORMAT, team=_CyclingTeambuilder(pool),
        account_configuration=AccountConfiguration("GoldCap", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        battle_class=Gen3Battle)
    # p2 cycles the pool offset by one so each battle is a distinct, fixed matchup.
    p2 = _DetPlayer(
        battle_format=BATTLE_FORMAT, team=_CyclingTeambuilder(pool[1:] + pool[:1]),
        account_configuration=AccountConfiguration("GoldOpp", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        battle_class=Gen3Battle)
    await run_local_battles(p1, p2, N_BATTLES, seed=BRIDGE_SEED)
    return p1.vectors


def capture_vectors() -> List[np.ndarray]:
    """Play the fixed battle set and return the per-decision obs vectors (in order)."""
    return asyncio.run(_capture())


def vector_hashes(vectors: List[np.ndarray]) -> List[str]:
    """Per-decision sha256 of the float32 bytes — the byte-exact value fingerprint."""
    return [hashlib.sha256(np.ascontiguousarray(v, dtype=np.float32).tobytes()).hexdigest()
            for v in vectors]


if __name__ == "__main__":
    vecs = capture_vectors()
    hashes = vector_hashes(vecs)
    dim = len(vecs[0]) if vecs else 0
    print(f"captured {len(vecs)} decision vectors (obs dim {dim})")
    print(f"digest of digests: {hashlib.sha256(''.join(hashes).encode()).hexdigest()}")
    if "--write" in sys.argv:
        import json
        fixture = {
            "_comment": "Golden per-decision observation hashes captured before the gen3_data "
                        "refactor. Regenerate ONLY when an intentional obs-value change "
                        "(ARCH_SIGNATURE bump) lands: "
                        "python src/agents/training/golden_obs_capture.py --write",
            "bridge_seed": BRIDGE_SEED, "n_battles": N_BATTLES, "n_teams": N_TEAMS,
            "obs_dim": int(dim), "n_decisions": len(vecs), "hashes": hashes,
        }
        path = "src/agents/training/golden_obs_fixture.json"
        with open(path, "w") as f:
            json.dump(fixture, f, indent=1)
            f.write("\n")
        print(f"wrote fixture -> {path}")
