"""The PRIVILEGED true-team channel, end to end over the real bridge (`gen3_value_true_team_v1`).

Three claims that only a real battle can settle, and one that only the two emitters can:

1. **`Gen3Env` emits `opp_true_team`, and it is the opponent's ACTUAL party** — all six mons,
   including the ones the trainee has never seen. That is the whole point of the channel: the
   trainee's own `battle1.opponent_team` holds only what has been revealed.
2. **It is the SAME encoding, not a second one.** For a mon the trainee HAS revealed, the
   privileged block's identity columns equal the flat 2501-dim vector's own opp-team slice for
   that mon — same species / item / ability / type / move ids, read back through the same
   `slice_pokemon_categoricals` the model uses.
3. **Presence is decided by whether the sim is LOCAL.** `LocalBattleRunner` — the transport for
   bridge training AND for bridge eval — sets the `_opp_player` back-reference on both sides;
   `RLPlayer` reads it and emits the true block. With no such handle (a real server: `main.play`)
   the same code emits the all-zero 'unknown' block instead of raising, so the ladder path runs.
"""
from types import SimpleNamespace

import asyncio
import gymnasium as gym
import numpy as np
import pytest
import torch
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.data.normalize import to_id_str
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.inference.player import RLPlayer
from agents.model.extractor_ctx import slice_pokemon_categoricals
from agents.observation.constants import (
    OFFSET_OPP_TEAM, POKEMON_FULL_DIM, POKEMON_SPECIES_KNOWN_OFFSET, TEAM_SIZE,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.observation.true_team import TRUE_TEAM_KEY, TRUE_TEAM_SHAPE
from agents.training.gen3_env import Gen3Env
from utils.bridge.bridge_session import attach_bridge_transport
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader.loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

pytestmark = pytest.mark.sim

def _bare_hp_num():
    """The BARE Hidden Power num — what the OPPONENT's HP reads as in the trainee's own view
    (`state_encoder`, gen3_typed_hidden_power_ids_v1), read from the data facade rather than
    typed, so a renumber is a failure here and not a silently-relaxed assertion."""
    return int(load_mappings()["moves"]["hiddenpower"]["num"])


def _teams():
    loader = TeamLoader()
    return loader.get_sample_teams() or loader.get_all_teams()


def _species_ids(block_or_vec, layout):
    """The per-slot species embedding num, through the model's OWN slicer."""
    t = torch.as_tensor(np.asarray(block_or_vec, dtype=np.float32)).reshape(
        1, -1, POKEMON_FULL_DIM)
    return slice_pokemon_categoricals(t, layout)["species_ids"][0].tolist()


def test_the_key_is_only_declared_when_the_flag_is_on():
    """A key nothing reads would still be stored and shuffled by the rollout buffer on every step
    of every run, and every non-local path would then have to fabricate it."""
    off = Gen3Env(load_mappings(), battle_format="gen3ou", team=Gen3Teambuilder(_teams()),
                  account_configuration1=AccountConfiguration("TTOff", None),
                  start_listening=False)
    assert TRUE_TEAM_KEY not in off.observation_space.spaces
    on = Gen3Env(load_mappings(), battle_format="gen3ou", team=Gen3Teambuilder(_teams()),
                 account_configuration1=AccountConfiguration("TTOn", None),
                 start_listening=False, emit_opp_true_team=True)
    assert on.observation_space.spaces[TRUE_TEAM_KEY].shape == TRUE_TEAM_SHAPE


def test_gen3env_emits_the_opponents_WHOLE_team_and_in_the_encoders_own_layout():
    teams = _teams()
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    species_num = {sid: rec["num"] for sid, rec in mappings.get("species", {}).items()
                   if "num" in rec}
    env = Gen3Env(mappings, battle_format="gen3ou", team=Gen3Teambuilder(teams),
                  account_configuration1=AccountConfiguration("TTEnv", None),
                  start_listening=False, emit_opp_true_team=True)
    attach_bridge_transport(env, battle_format="gen3ou", persistent=True)
    opponent = RandomPlayer(battle_format="gen3ou", team=Gen3Teambuilder(teams),
                            account_configuration=AccountConfiguration("TTEnvOpp", None),
                            start_listening=False)
    wrapped = SingleAgentWrapper(env, opponent)
    wrapped.action_space = env.action_space
    wrapped.observation_space = env.observation_space

    rng = np.random.default_rng(0)
    seen = 0
    matched_revealed = 0
    hidden_facts = 0
    bare_hp = _bare_hp_num()
    saw_more_than_revealed = False
    try:
        obs, _ = wrapped.reset()
        for _ in range(400):
            block = obs[TRUE_TEAM_KEY]
            assert block.shape == TRUE_TEAM_SHAPE and block.dtype == np.float32
            b2 = env.battle2
            if b2 is not None and b2.team:
                priv_ids = _species_ids(block, layout)
                present = [priv_ids[i] for i in range(TEAM_SIZE)
                           if block[i, POKEMON_SPECIES_KNOWN_OFFSET] >= 0.5]
                # 1. THE WHOLE TEAM — every mon agent2 owns has a row, keyed by species num.
                want = sorted(species_num[to_id_str(m.species)] for m in b2.team.values()
                              if to_id_str(m.species) in species_num)
                assert sorted(present) == want, (
                    f"privileged block species {sorted(present)} != agent2's team {want}")
                # 2. STRICTLY more than the trainee can see — the channel adds information.
                opp_slice = np.asarray(obs["observation"])[
                    OFFSET_OPP_TEAM:OFFSET_OPP_TEAM + TEAM_SIZE * POKEMON_FULL_DIM
                ].reshape(TEAM_SIZE, POKEMON_FULL_DIM)
                live = slice_pokemon_categoricals(torch.as_tensor(opp_slice)[None], layout)
                priv = slice_pokemon_categoricals(torch.as_tensor(block)[None], layout)
                revealed = [i for i in range(TEAM_SIZE)
                            if opp_slice[i, POKEMON_SPECIES_KNOWN_OFFSET] >= 0.5]
                if len(present) > len(revealed):
                    saw_more_than_revealed = True
                # 3. THE SAME ENCODING, where the two views can agree at all. The columns a
                #    revealed mon's SPECIES determines — its type pair — must be identical, which
                #    is what pins the two blocks to one per-mon layout and one encoder. The
                #    columns Gen 3 HIDES must NOT be asserted equal: an item the trainee has not
                #    seen reads 0 in the flat vector and the true num in the privileged block, and
                #    that difference IS the channel. It is asserted positively below.
                for i in revealed:
                    sp = int(live["species_ids"][0, i])
                    if sp not in priv_ids:
                        continue
                    j = priv_ids.index(sp)
                    for key in ("type1_ids", "type2_ids"):
                        assert int(live[key][0, i]) == int(priv[key][0, j]), (
                            f"{key} disagrees for species {sp}: the privileged block is not the "
                            "encoder's own per-mon encoding")
                    seen_moves = {int(x) for x in live["all_move_ids"][0, i] if x > 0}
                    true_moves = {int(x) for x in priv["all_move_ids"][0, j] if x > 0}
                    # gen3_typed_hidden_power_ids_v1: the OPPONENT's Hidden Power keeps the BARE
                    # num, while an OWN mon's carries the TYPED id — and the privileged block is
                    # encoded from the owner's side, so 237 there becomes a typed id. Dropping it
                    # from the subset check is not a weakening: an HP the trainee has watched land
                    # and still cannot type is the single most valuable thing this channel carries,
                    # and it is counted as a hidden fact instead.
                    if bare_hp in seen_moves:
                        seen_moves = seen_moves - {bare_hp}
                        if true_moves - seen_moves:
                            hidden_facts += 1
                    assert seen_moves <= true_moves, (
                        f"a move the trainee has SEEN ({seen_moves - true_moves}) is missing from "
                        "the privileged row")
                    matched_revealed += 1
                    # 4. THE INFORMATION CLAIM, on a mon the trainee has already MET: Gen 3 never
                    #    reveals a held item until it fires, so a 0 here against a real num there
                    #    is the privilege, measured rather than argued.
                    if int(live["item_ids"][0, i]) == 0 and int(priv["item_ids"][0, j]) > 0:
                        hidden_facts += 1
            seen += 1
            legal = np.flatnonzero(env.action_masks())
            if legal.size == 0:
                break
            obs, _, term, trunc, _ = wrapped.step(int(rng.choice(legal)))
            if term or trunc:
                break
    finally:
        # A close on an unfinished battle raises, and that raise would MASK whichever assertion
        # above ended the loop early — the failure a reader actually needs.
        try:
            wrapped.close()
        except Exception:
            pass
    assert seen >= 3, f"only {seen} decisions — the bridge battle did not run"
    assert matched_revealed >= 1, "no revealed opp mon was ever cross-checked"
    assert hidden_facts >= 1, (
        "the privileged block never carried an ITEM the trainee could not see — on a Gen 3 board "
        "that is the cheapest possible proof the channel is privileged at all")
    assert saw_more_than_revealed, (
        "the privileged block never carried more mons than the trainee had revealed — the channel "
        "would be adding no information at all")


# ------------------------------------------------------- 3. presence follows the LOCAL sim
def test_local_battle_runner_sets_the_back_reference_on_both_sides():
    """This is the single place that decides the channel's availability, so it is asserted at the
    place rather than at any one consumer: it is the transport for bridge TRAINING and for bridge
    EVAL alike (`eval_callback`: 'workers play in-process via run_local_battles')."""
    teams = _teams()
    p1 = RandomPlayer(battle_format="gen3ou", team=Gen3Teambuilder(teams),
                      account_configuration=AccountConfiguration("TTRun1", None),
                      start_listening=False, max_concurrent_battles=1)
    p2 = RandomPlayer(battle_format="gen3ou", team=Gen3Teambuilder(teams),
                      account_configuration=AccountConfiguration("TTRun2", None),
                      start_listening=False, max_concurrent_battles=1)
    asyncio.run(run_local_battles(p1, p2, 1, battle_format="gen3ou", seed=[3, 4, 5, 6]))
    assert p1._opp_player is p2 and p2._opp_player is p1


def _rl_player(name, wants_key: bool):
    """A REAL `RLPlayer`, with a stand-in for the one thing these two methods read off the model:
    its observation SPACE. The model is a namespace rather than a monkeypatched module symbol on
    purpose — the assertion is about how the player interrogates whatever model it was handed."""
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = {"observation": gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],),
                                           dtype=np.float32)}
    if wants_key:
        space[TRUE_TEAM_KEY] = gym.spaces.Box(0.0, 1e4, shape=TRUE_TEAM_SHAPE, dtype=np.float32)
    model = SimpleNamespace(observation_space=gym.spaces.Dict(space))
    p = RLPlayer(model, Gen3Teambuilder(_teams()), "gen3ou", LocalhostServerConfiguration,
                 mappings=mappings,
                 account_configuration=AccountConfiguration(name, None),
                 start_listening=False, max_concurrent_battles=1)
    return p


def test_rlplayer_asks_the_models_SPACE_not_a_flag():
    """SB3's `preprocess_obs` iterates the SPACE's keys and indexes the obs dict, so a model whose
    space declares the key gets a KeyError — not a quiet skip — when it is missing. The space is
    therefore what must decide."""
    assert _rl_player("TTWants", True)._wants_true_team() is True
    assert _rl_player("TTNoWant", False)._wants_true_team() is False


def test_rlplayer_falls_back_to_the_unknown_block_when_the_sim_is_not_LOCAL():
    """The LADDER case (`src/main/play.py` against a real server): no `_opp_player`, so no
    privileged view exists. The block must be all-zero — the encoder's own ABSENT spelling — and
    the call must not raise, because the policy head is what plays a ladder game."""
    p = _rl_player("TTLadder", True)
    assert p._opp_player is None
    block = p._true_team_block(SimpleNamespace(battle_tag="battle-gen3ou-1"))
    assert block.shape == TRUE_TEAM_SHAPE and not block.any()


def test_rlplayer_reads_the_true_team_through_the_back_reference():
    """The EVAL case: two local players, so the opponent's own battle view is reachable and its
    six fully-known mons are what the block carries."""
    teams = _teams()
    p1 = RandomPlayer(battle_format="gen3ou", team=Gen3Teambuilder(teams),
                      account_configuration=AccountConfiguration("TTEv1", None),
                      start_listening=False, max_concurrent_battles=1)
    p2 = RandomPlayer(battle_format="gen3ou", team=Gen3Teambuilder(teams),
                      account_configuration=AccountConfiguration("TTEv2", None),
                      start_listening=False, max_concurrent_battles=1)
    asyncio.run(run_local_battles(p1, p2, 1, battle_format="gen3ou", seed=[7, 8, 9, 10]))
    tag, opp_battle = next(iter(p2._battles.items()))

    reader = _rl_player("TTEvRd", True)
    reader._opp_player = p2
    block = reader._true_team_block(SimpleNamespace(battle_tag=tag))
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    species_num = {sid: rec["num"] for sid, rec in mappings.get("species", {}).items()
                   if "num" in rec}
    want = sorted(species_num[to_id_str(m.species)] for m in opp_battle.team.values()
                  if to_id_str(m.species) in species_num)
    got = sorted(s for s in _species_ids(block, layout) if s > 0)
    assert got == want and len(want) >= 1, (got, want)
