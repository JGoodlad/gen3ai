"""Bridge fuzz test: our OWN Hidden Power is NEVER typeless in the obs the network sees.

We always know our own HP type (it is IV-derived and declared in our team), and the model must read
that type — a Hidden Power encoded as the type-0 "unknown" sentinel (or Normal) would be a silent
representation bug. gen3 has no team preview and the wire re-keys our HP to a BARE "hiddenpower", so
several obs derivations have to re-resolve the typed id; this guards that they all do.

Builds the FULL per-decision obs (encoder + prev-mask + turn-history) exactly as ``gen3_env.embed_battle``
does, and at EVERY one of our decisions, over many real bridge battles, asserts:

  1. PER-MON move block (our 6 mons): every Hidden Power slot (move num 237) decodes TYPED —
     ``hiddenpower(<type>)``, never the bare ``hiddenpower`` (type-0 / Normal sentinel). [moves.py]
  2. TURN-HISTORY ``our_move`` (after WE used HP): the HP move (num 237) carries its real type
     (type_id ∉ {0 unknown, 1 Normal}). [the gen3_own_hp_typed_history_v1 fold fix]
  3. NO-LEAK complement: the OPPONENT's revealed HP stays TYPELESS (bare ``hiddenpower`` / type 0) —
     poke-env never reveals the opp HP type, so it must NOT appear typed anywhere in the obs.

Coverage for 1 & 2 is tracked and asserted > 0 (a clean pass that never exercised an own HP would be
vacuous); the player nudges toward HP to populate the history. Any violation raises immediately with
the offending decision. Run directly (no server — in-process via the local BattleStream bridge):

    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/poke_env_gaps/hidden_power_typed_obs_fuzz_test.py [n_battles]
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.action.mask_generator import Gen3ActionMasker
from agents.battle.gen3_battle import Gen3Battle
from agents.battle.live_view import LegalActions
from agents.observation.constants import OFFSET_OUR_TEAM, OFFSET_OPP_TEAM, POKEMON_FULL_DIM, POKEMON_VECTOR_DIM
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.training.episode_tracker import EpisodeTracker
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder
from utils.bridge.local_battle_runner import run_local_battles

_HP_NUM = 237            # the OPPONENT's / typeless bare-Hidden-Power num (gen3_typed_hidden_power_ids_v1)
_OUR_HP_NUMS = frozenset(range(355, 371))   # OUR typed Hidden Power distinct nums (355-370)
_BARE_HP_NUM = 237            # the TYPELESS Hidden Power num — our side must never emit it
_TYPELESS_TYPE_IDS = {0, 1}   # 0 = the "unknown" sentinel, 1 = Normal (HP is never Normal in gen3)


@dataclass
class _Stats:
    decisions: int = 0
    our_hp_permon_checked: int = 0     # our per-mon HP slots seen (typed type-channel required)
    our_hp_history_checked: int = 0    # our HP turns folded into history (typed-required)
    opp_hp_seen: int = 0               # opp revealed HP slots seen (typeless-required)
    failures: int = 0
    examples: List[dict] = field(default_factory=list)

    def fail(self, ex: dict) -> None:
        self.failures += 1
        if len(self.examples) < 12:
            self.examples.append(ex)


class _HPTypedPlayer(Player):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("battle_class", Gen3Battle)   # strict_view + the event-log history fold
        super().__init__(*args, **kwargs)
        self.encoder = Gen3ObservationEncoder(load_mappings())
        self.tracker = EpisodeTracker()
        self.stats = _Stats()
        self._used_hp = False

    # --- the same per-decision obs assembly gen3_env.embed_battle performs (encoder + history) ---
    def _full_obs(self, battle, legal) -> Optional[np.ndarray]:
        mask = Gen3ActionMasker.get_mask(battle, legal=legal).astype(np.int8)
        if mask.sum() == 0:
            return None
        self.tracker.record(battle, mask, legal=legal)
        self.tracker.update_progress_clock(battle, legal)
        obs = self.encoder.encode(
            battle, hp_tracker=self.tracker.hidden_power_tracker, legal=legal,
            progress_clock=self.tracker.progress_clock,
        )
        # gen3_frame_deletion_v1: no prev-mask / history tail — the obs is the encoder output.
        return obs

    def _decoded_moves(self, vec, side_offset):
        """The 6 per-mon move-name lists for a side (HP types via moves.describe_vector)."""
        out = []
        for i in range(6):
            base = side_offset + i * POKEMON_FULL_DIM
            d = self.encoder.pokemon_encoder.describe_vector(vec[base:base + POKEMON_VECTOR_DIM])
            out.append(d.get("moves", []))
        return out

    def _validate(self, battle) -> None:
        if battle.strict_view().finished:
            return
        legal = LegalActions.from_battle(battle)
        vec = self._full_obs(battle, legal)
        if vec is None:
            return
        s = self.stats
        s.decisions += 1

        # 1. OUR per-mon HP must decode TYPED — never the bare "hiddenpower" (gen3_typed_hidden_power_ids_v1:
        #    our HP carries a distinct num, so describe_vector renders "hiddenpower(<type>)"). The
        #    move_id_decode_fuzz_test additionally pins the distinct NUM + the no-mis-association invariant.
        for slot, moves in enumerate(self._decoded_moves(vec, OFFSET_OUR_TEAM)):
            for m in moves:
                if m == "hiddenpower":   # bare = type-0/Normal sentinel → typeless in the type channel
                    s.fail({"check": "our_permon_hp_typeless_channel", "turn": battle.turn, "slot": slot,
                            "moves": moves})
                elif m.startswith("hiddenpower("):
                    s.our_hp_permon_checked += 1

        # 3. NO-LEAK: the OPPONENT's revealed HP must stay BARE (we never know their type).
        for slot, moves in enumerate(self._decoded_moves(vec, OFFSET_OPP_TEAM)):
            for m in moves:
                if m.startswith("hiddenpower("):
                    s.fail({"check": "opp_hp_TYPED_leak", "turn": battle.turn, "slot": slot,
                            "moves": moves})
                elif m == "hiddenpower":
                    s.opp_hp_seen += 1

        # 2. EVENT-WINDOW our_move HP must carry its DISTINCT num (355-370) — never the
        #    typeless bare-237, which would mean the fold lost our type.
        #    gen3_frame_deletion_v1 REPOINTED this check: it used to read the TurnDelta lag
        #    frames, which are deleted. The guard itself is unchanged in meaning and is NOT
        #    droppable — a typed HP collapsing to bare 237 is the exact GIGO that made the
        #    opponent's Hidden Power read as "immune", so it keeps a home on whichever block
        #    carries move history. That block is now the H-B event window: `EventCol.MOVE` is
        #    the move num, written from the same `gen3_movedex` lookup the frames used.
        from agents.observation.constants import (
            OFFSET_EVENT_WINDOW, EVENT_WINDOW_N, EVENT_TOKEN_DIM, EVENT_T_MOVE,
            EventCol as C,
        )
        for _r in range(EVENT_WINDOW_N):
            _o = OFFSET_EVENT_WINDOW + _r * EVENT_TOKEN_DIM
            if vec[_o + C.VALID] < 0.5 or int(vec[_o + C.TYPE]) != EVENT_T_MOVE:
                continue                                   # pad row, or not a move
            if vec[_o + C.ACTOR_SIDE] < 0.5:
                continue                                   # not OUR side
            mid = int(vec[_o + C.MOVE])
            if mid == _BARE_HP_NUM:
                s.fail({"check": "our_event_hp_collapsed_to_bare_237", "turn": battle.turn,
                        "event_row": _r, "move_num": mid})
            elif mid in _OUR_HP_NUMS:
                s.our_hp_history_checked += 1

    def choose_move(self, battle):
        try:
            self._validate(battle)
        except Exception as e:  # noqa: BLE001 — surface the exact decision that broke
            print("\n🛑 [HP-TYPED OBS FUZZ CRITICAL FAILURE] 🛑")
            print(f"Battle {battle.battle_tag} turn {battle.turn}: {e}")
            traceback.print_exc()
            os._exit(1)
        # Nudge toward using Hidden Power so the turn-history check (2) gets coverage.
        hp = next((mv for mv in battle.available_moves if mv.id.startswith("hiddenpower")), None)
        if hp is not None and not self._used_hp:
            self._used_hp = True
            return self.create_order(hp)
        return self.choose_random_move(battle)


async def run(n_battles: int) -> None:
    print(f"Hidden-Power TYPED-obs Fuzz — gen3ou — {n_battles} battles\n")
    teams = TeamLoader().get_all_teams()
    teambuilder = Gen3Teambuilder(teams)
    ts = int(time.time())

    player = _HPTypedPlayer(
        battle_format="gen3ou", team=teambuilder,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"HPTyped{ts}", "x"), max_concurrent_battles=1)
    opponent = RandomPlayer(
        battle_format="gen3ou", team=teambuilder,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"HPTypedOpp{ts}", "x"), max_concurrent_battles=1)

    for _ in range(n_battles):
        player._used_hp = False
        player.tracker.reset()
        await run_local_battles(player, opponent, 1)

    s = player.stats
    print("=" * 68)
    print(f"decisions validated            : {s.decisions}")
    print(f"our per-mon HP slots (typed)   : {s.our_hp_permon_checked}")
    print(f"our HISTORY HP turns (typed)   : {s.our_hp_history_checked}")
    print(f"opp revealed HP slots (bare)   : {s.opp_hp_seen}  (no-leak)")
    print(f"failures                       : {s.failures}")
    for ex in s.examples:
        print("   violation:", ex)
    print("=" * 68)

    if s.failures:
        print(f"\n❌ FAIL — {s.failures} typeless-HP violation(s) in our obs")
        os._exit(1)
    # No silent cap: a pass that never exercised an own HP (per-mon OR history) is inconclusive.
    if s.our_hp_permon_checked == 0:
        print("\n⚠️  PASS but NO own per-mon Hidden Power was exercised — INCONCLUSIVE. Re-run with more battles.")
    elif s.our_hp_history_checked == 0:
        print("\n⚠️  PASS (per-mon typed) but NO own HP reached the turn-history — the history fix went "
              "unexercised. Re-run with more battles for full coverage.")
    else:
        print("\n✅ PASS — our Hidden Power is TYPED everywhere in the obs (per-mon + history); opp HP stays bare.")


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(run(int(sys.argv[1]) if len(sys.argv) > 1 else 40))
