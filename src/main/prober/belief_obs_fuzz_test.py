"""Fuzz test for the GPU-obs observability decoders — real bridge battles, no server.

Runs the FULL belief stack (species belief + move belief + spread belief + the differentiable damage op,
all ON) over the REAL live-battle state distribution and asserts every decision that the side-stashes the
prober reads are present + sane AND that the pure engine decoder `build_spread_belief` turns them into
populated, finite views:

  (1) `last_spread_belief` is `[1,6,5]`, finite, and clamped ≥ 1 (a stat value) — the DamageOperator's
      believed opp DERIVED stats the prober's spread panel surfaces;
  (2) `build_spread_belief` matches the live opp ACTIVE mon (placed at `last_opp_active_local`) to a
      synthetic team_details by SPECIES and produces finite believed-vs-true DERIVED-stat rows (the L100
      formula path on a real species).

Run directly (bridge-backed, no Showdown server):
    python3 src/main/prober/belief_obs_fuzz_test.py [n_battles]
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
import asyncio
import sys

import gymnasium as gym
import numpy as np
import torch
from poke_env.player import RandomPlayer
from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.battle.gen3_battle import Gen3Battle
from agents.inference.player import Gen3Player
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from main.prober.engine import build_spread_belief
from poke_env.data.normalize import to_id_str
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"

# The full belief/op stack — the spread belief feeds the op.
_TOGGLES = dict(
    attend_unrevealed_opponents=True, opp_belief_slots=True, move_belief_mode="both",
    move_prior_fusion=True, damage_op=True, move_latent=True, spread_belief=True,
)


class _BeliefObsFuzzPlayer(Gen3Player):
    """Random-policy player that runs the full belief/op extractor on every LIVE decision, then
    validates the observability stashes + engine decoders. Any violation raises immediately (the fuzz
    contract)."""

    def __init__(self, *, rng_seed, **kwargs):
        super().__init__(battle_class=Gen3Battle, **kwargs)
        self._rng = np.random.RandomState(rng_seed)
        mappings = load_mappings()
        layout = Gen3ObservationEncoder(mappings).get_layout()
        obs_space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
        self._fx = Gen3FeaturesExtractor(obs_space, layout=layout, mappings=mappings, **_TOGGLES)
        self._fx.eval()
        self.n_decisions = 0
        self.n_spread_truth_matched = 0

    def choose_move(self, battle):
        forfeit = self._handle_stall(battle, "BELIEF_OBS_FUZZ_STALL")
        if forfeit:
            return forfeit
        obs_dict = self.embed_battle(battle)
        mask = obs_dict["action_mask"]
        obs_t = torch.as_tensor(np.asarray(obs_dict["observation"], dtype=np.float32))[None, :]

        with torch.no_grad():
            pi, vf = self._fx({"observation": obs_t})
        assert torch.isfinite(pi).all() and torch.isfinite(vf).all(), \
            f"non-finite forward at turn {battle.turn}"
        self._check_spread(battle)

        self.n_decisions += 1
        if int(mask.sum()) == 0:
            return self.choose_default_move()
        idx = int(self._rng.choice(np.flatnonzero(mask)))
        self._get_tracker(battle).advance(idx)
        return self.action_to_order(idx, battle)

    def _check_spread(self, battle):
        """(1) + (2): `last_spread_belief` is finite [1,6,5] ≥ 1, and the engine spread builder matches the
        live opp ACTIVE mon (at `last_opp_active_local`) to a synthetic team_details by species + produces
        finite believed-vs-true rows (the L100 derived-stat path on a real species)."""
        sb = self._fx.last_spread_belief
        assert sb is not None and tuple(sb.shape) == (1, 6, 5), f"bad last_spread_belief shape turn {battle.turn}"
        assert torch.isfinite(sb).all(), f"non-finite spread belief turn {battle.turn}"
        assert (sb >= 1.0 - 1e-4).all(), f"spread belief not clamped >=1 turn {battle.turn}"
        active = int(self._fx.last_opp_active_local[0].item())
        opp = battle.opponent_active_pokemon
        if opp is None or not (0 <= active < 6):
            return
        sid = to_id_str(opp.species)
        opp_species = [""] * 6
        opp_species[active] = sid
        # the believed_mask the prober reads (True = hidden); the active opp is revealed → False there.
        bmask = self._fx.last_opp_believed_mask[0].detach().cpu().numpy().astype(bool)
        raw = {"spread": sb[0].detach().cpu().numpy(), "believed_mask": bmask,
               "opp_species": opp_species, "opp_active": active}
        # synthesize the privileged truth for just the active species (a max-Atk adamant set) so the
        # truth-join + L100 derived-stat math is exercised on a REAL species.
        details = [{"species": sid, "nature": "adamant",
                    "evs": {"hp": 4, "atk": 252, "def": 0, "spa": 0, "spd": 0, "spe": 252},
                    "ivs": {k: 31 for k in ("hp", "atk", "def", "spa", "spd", "spe")}}]
        sv = build_spread_belief(raw, details)
        if sv is None:                          # the active slot read as hidden (just-switched edge) — skip
            return
        slot = next((s for s in sv.slots if s.species == sid), None)
        assert slot is not None, f"spread view dropped the revealed active {sid} turn {battle.turn}"
        for r in slot.rows:
            assert np.isfinite(r.believed), f"non-finite believed {r.stat} turn {battle.turn}"
            if r.true is not None:
                assert r.true > 0 and np.isfinite(r.true), f"bad true {r.stat} turn {battle.turn}"
        if slot.matched:
            self.n_spread_truth_matched += 1


async def _run(n_battles: int, seed: int) -> int:
    pool = TeamLoader().get_all_teams()
    trainee = _BeliefObsFuzzPlayer(
        rng_seed=seed, battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"BOz{seed}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        max_concurrent_battles=1,
    )
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"BOo{seed}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        max_concurrent_battles=1,
    )
    print(f"belief-obs observability fuzz — {n_battles} real bridge battles (full belief stack)", flush=True)
    await run_local_battles(trainee, opp, n_battles, concurrency=1)
    print(f"  decisions checked: {trainee.n_decisions}; spread-truth matched: "
          f"{trainee.n_spread_truth_matched}", flush=True)
    assert trainee.n_decisions > 0, "no decisions exercised"
    assert trainee.n_spread_truth_matched > 0, \
        "the spread truth-join never matched a revealed active — feature not exercised"
    return trainee.n_decisions


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    total = asyncio.get_event_loop().run_until_complete(_run(n, seed=20260617))
    print(f"\nbelief-obs fuzz PASSED — {total} live decisions, all observability invariants held.", flush=True)


if __name__ == "__main__":
    main()
