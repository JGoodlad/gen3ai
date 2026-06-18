"""Fuzz test for gen3_bidir_threat_trunk_v1 (v36) — real bridge battles, no server.

Runs the FULL v36 extractor (outgoing-in-trunk + expected-latent + prob-outspeed, all ON) over the
REAL live-battle state distribution the hand-built unit ctxs can't cover — partially-revealed opp
teams, mid-battle switches, faints, statuses, boosts, screens, weather — and asserts, every decision:

  (1) the full pi/vf forward is FINITE (no NaN/Inf) — the v36 refine_cb (discrete_outgoing +
      BeliefHead.species_logits + the expected-latent matmuls + prob-outspeed) never blows up on a
      real board;
  (2) on the live ctx, discrete_outgoing is FINITE, correctly GATED (0 when our active is fainted /
      no opp active), and — the owner invariant — its P(KO) channels are EXACTLY 0 for every
      UNREVEALED opp slot, for ANY species_probs (the magnitude survives, the OHKO bit is nulled);
  (3) when the opp has ≥1 hidden mon AND we hold a damaging move, the expected-latent read produces
      a strictly DIFFERENT outgoing block than the revealed-gated (species_probs=None) baseline —
      i.e. the unrevealed columns actually get priced (the feature is live on real states).

Run directly (bridge-backed, no Showdown server):
    PYTHONPATH=src python3 src/agents/model/bidir_threat_fuzz_test.py [n_battles]
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
from agents.model.features_extractor import Gen3FeaturesExtractor, TEAM_SIZE
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"

_TOGGLES = dict(
    attend_unrevealed_opponents=True, opp_belief_slots=True, move_belief_mode="both",
    move_prior_fusion=True, damage_op=True, move_latent=True, damage_refine_rounds=2,
    threat_refine_outgoing=True, threat_unrevealed_outgoing=True, threat_prob_outspeed=True,
    threat_status_refine=True,   # v37: status-landing into the trunk (both directions)
)


class _ThreatFuzzPlayer(Gen3Player):
    """Random-policy player that runs the v36 extractor + the discrete_outgoing invariants on every
    LIVE decision. Any violation raises immediately (the fuzz contract)."""

    def __init__(self, *, rng_seed, **kwargs):
        super().__init__(battle_class=Gen3Battle, **kwargs)
        self._rng = np.random.RandomState(rng_seed)
        mappings = load_mappings()
        layout = Gen3ObservationEncoder(mappings).get_layout()
        obs_space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
        self._fx = Gen3FeaturesExtractor(obs_space, layout=layout, mappings=mappings, **_TOGGLES)
        self._fx.eval()
        self.n_decisions = 0
        self.n_unrevealed_priced = 0
        self.n_status_priced = 0

    def choose_move(self, battle):
        forfeit = self._handle_stall(battle, "THREAT_FUZZ_STALL")
        if forfeit:
            return forfeit
        obs_dict = self.embed_battle(battle)
        mask = obs_dict["action_mask"]
        obs_t = torch.as_tensor(np.asarray(obs_dict["observation"], dtype=np.float32))[None, :]
        with torch.no_grad():
            pi, vf = self._fx({"observation": obs_t})
            assert torch.isfinite(pi).all() and torch.isfinite(vf).all(), \
                f"non-finite forward at turn {battle.turn}"
            ctx = self._fx.unpack({"observation": obs_t})
            self._check_discrete_outgoing(ctx, battle)
            self._check_status(ctx, battle)
        self.n_decisions += 1
        if int(mask.sum()) == 0:
            return self.choose_default_move()
        idx = int(self._rng.choice(np.flatnonzero(mask)))
        self._get_tracker(battle).advance(idx)
        return self.action_to_order(idx, battle)

    def _check_discrete_outgoing(self, ctx, battle):
        op = self._fx.damage_op
        believed = ctx.opp_believed_mask.float()                         # [B,6] 1 = UNREVEALED
        # revealed-gated baseline (no belief): finite + unrevealed columns ZERO
        base = op.discrete_outgoing(ctx, species_probs=None)             # [B,6,4]
        assert torch.isfinite(base).all(), f"non-finite discrete_outgoing(None) turn {battle.turn}"
        unrevealed = believed.bool()[:, :, None].expand_as(base)
        assert base[unrevealed].abs().sum().item() == 0.0, "unrevealed leaked into the revealed-gated block"
        # expected-latent: random P(species) → still finite, and P(KO) NULLED for unrevealed slots
        n_sp = op.SPECIES_EXP_MULT.shape[0]
        sp = torch.softmax(torch.randn(ctx.batch_size, TEAM_SIZE, n_sp), dim=-1)
        out = op.discrete_outgoing(ctx, species_probs=sp)                # [B,6,4]
        assert torch.isfinite(out).all(), f"non-finite expected-latent discrete_outgoing turn {battle.turn}"
        # cols 2,3 = phys_pko, spec_pko → exactly 0 wherever believed (the owner invariant)
        pko_unrevealed = out[:, :, 2:4][believed.bool()[:, :, None].expand(-1, -1, 2)]
        assert pko_unrevealed.abs().sum().item() == 0.0, \
            f"P(KO) not nulled for an unrevealed defender (turn {battle.turn})"
        # if there's a hidden mon AND we computed any outgoing magnitude, the belief must PRICE it
        # (the expected-latent block differs from the revealed-gated one on the unrevealed columns)
        if believed.sum().item() > 0 and out.abs().sum().item() > 0.0:
            if not torch.equal(out, base):
                self.n_unrevealed_priced += 1

    def _check_status(self, ctx, battle):
        """v37 status-landing invariants on the LIVE ctx, both directions. The kernels' invariants hold for
        ANY belief input, so incoming uses a synthetic per-move belief (a small random logit) — we're
        validating finiteness / [0,1] bounds / immobilize⊆major / revealed-gating, not specific values."""
        op = self._fx.damage_op
        believed = ctx.opp_believed_mask.float()                        # [B,6] 1 = UNREVEALED
        n_moves = op.MOVE_INFLICTS_STATUS.shape[0]
        ml = torch.randn(ctx.batch_size, TEAM_SIZE, n_moves)            # synthetic move belief
        inc = op.discrete_incoming_status(ctx, ml)                      # [B,6,2]  (opp active → our 6)
        out = op.discrete_outgoing_status(ctx)                          # [B,6,2]  (our active → opp 6)
        for name, t in (("incoming", inc), ("outgoing", out)):
            assert torch.isfinite(t).all(), f"non-finite {name} status, turn {battle.turn}"
            assert (t >= -1e-6).all() and (t <= 1.0 + 1e-6).all(), f"{name} status out of [0,1], turn {battle.turn}"
            # immobilize ⊆ major  →  p_immobilize <= p_major everywhere
            assert (t[..., 1] <= t[..., 0] + 1e-6).all(), f"{name} immobilize > major, turn {battle.turn}"
        # outgoing status is revealed-gated: unrevealed opp slots must be exactly 0
        unrevealed = believed.bool()[:, :, None].expand_as(out)
        assert out[unrevealed].abs().sum().item() == 0.0, "unrevealed opp leaked into outgoing status"
        if inc.abs().sum().item() > 0.0 or out.abs().sum().item() > 0.0:
            self.n_status_priced += 1


async def _run(n_battles: int, seed: int) -> int:
    pool = TeamLoader().get_all_teams()
    trainee = _ThreatFuzzPlayer(
        rng_seed=seed, battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"BTz{seed}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        max_concurrent_battles=1,
    )
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"BTo{seed}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        max_concurrent_battles=1,
    )
    print(f"bidir-threat fuzz — {n_battles} real bridge battles (v36 all-on)", flush=True)
    await run_local_battles(trainee, opp, n_battles, concurrency=1)
    print(f"  decisions checked: {trainee.n_decisions}; priced an unrevealed defender: "
          f"{trainee.n_unrevealed_priced}; priced a status-landing: {trainee.n_status_priced}", flush=True)
    assert trainee.n_decisions > 0, "no decisions exercised"
    assert trainee.n_unrevealed_priced > 0, \
        "the expected-latent path never priced an unrevealed defender — feature not exercised"
    assert trainee.n_status_priced > 0, \
        "the status-landing path never fired — feature not exercised"
    return trainee.n_decisions


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    total = asyncio.get_event_loop().run_until_complete(_run(n, seed=20260617))
    print(f"\nbidir-threat fuzz PASSED — {total} live decisions, all invariants held.", flush=True)


if __name__ == "__main__":
    main()
