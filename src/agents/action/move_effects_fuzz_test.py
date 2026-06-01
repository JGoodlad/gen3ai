"""Move-effect fuzz — validates the gen3_move_effects_v1 action-aligned per-move effect
block over real battles run in-process via the local BattleStream bridge (no server).

The reactive block now carries, for each of the 4 REQUEST-order move slots, 8 effect features
(is_boost, is_heal, is_protect, is_phaze, is_hazard, inflicts_status, status_will_land,
pp_fraction) so the policy head can tell a setup move from a heal from a wasted status. This
test drives the production obs encode on every decision of many real battles and asserts, for
EVERY available move (in request order, the same order the action mask / mapper use), that the
encoded 8-feature slot equals an INDEPENDENT recompute from the `gen3_data.moves` facade +
`gen3_mechanics.status_move_lands` + the live PP. That catches a wrong block offset, a
slot-order / request-order desync, a per-feature index swap, the forced-Struggle path, and the
live Curse-by-non-Ghost resolution — exactly the wiring this feature introduces.

Independent hard checks (NOT routed through `status_move_lands`, so they can't be circular):
  * a status move into a TYPE-immune target (Toxic→Steel/Poison, Thunder Wave→Ground,
    Will-O-Wisp→Fire) must encode status_will_land = 0;
  * a Curse slot encodes is_boost = (user is NOT Ghost).

Required coverage (all must occur or it FAILS — proving the categories were actually exercised):
is_boost, is_heal, is_protect, is_phaze, is_hazard, status_will_land==1, status_will_land==0
(via type immunity), and at least one Curse slot. P1 leads Snorlax (Curse, non-Ghost) and runs a
team spanning every effect category; P2 fields Steel/Poison/Ground/Fire walls so the status
immunity path is hit. P1 cycles its bench early to bring Gengar (Ghost Curse) in.

Run directly (no server needed — local bridge):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/action/move_effects_fuzz_test.py [n_battles]
"""
from __future__ import annotations

import asyncio
import sys
import traceback

import numpy as np

from poke_env.player.player import Player
from poke_env.battle.pokemon_type import PokemonType
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from utils.bridge.local_battle_runner import run_local_battles
from agents.battle.gen3_battle import Gen3Battle
from agents.battle.live_view import LegalActions
from agents import gen3_data
from agents.gen3_mechanics import status_land_estimate, STATUS_MOVE_IMMUNITY
from agents.observation.state_encoder import load_mappings, get_observation_encoder
from agents.observation.reactive import ReactiveEncoder, _resolve_ability_distribution
from agents.observation.constants import OFFSET_REACTIVE, MOVE_EFFECT_FEATURES

FMT = "gen3ou"

_RL = ReactiveEncoder().get_layout()
_ME_OFF = OFFSET_REACTIVE + _RL["move_effects"]["offset"]   # absolute obs offset of the block
_PER = MOVE_EFFECT_FEATURES                                  # 8

# P1 (validator): one mon per effect category; Snorlax-Curse leads (non-Ghost Curse → boost),
# Gengar carries the Ghost Curse (→ NOT boost). Movesets span heal / protect / phaze / hazard /
# status / boost so every required category is reachable.
P1_TEAM = """
Snorlax
Ability: Thick Fat
EVs: 252 HP / 252 Atk / 4 Def
Adamant Nature
- Curse
- Body Slam
- Rest
- Earthquake

Gengar
Ability: Levitate
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Curse
- Will-O-Wisp
- Thunderbolt
- Ice Punch

Skarmory
Ability: Keen Eye
EVs: 252 HP / 252 Def / 4 SpD
Impish Nature
- Spikes
- Roar
- Toxic
- Rest

Blissey
Ability: Natural Cure
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Soft-Boiled
- Thunder Wave
- Seismic Toss
- Calm Mind

Suicune
Ability: Pressure
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Calm Mind
- Surf
- Rest
- Whirlwind

Swampert
Ability: Torrent
EVs: 252 HP / 252 Atk / 4 Def
Adamant Nature
- Protect
- Earthquake
- Surf
- Ice Beam
"""

# P2: Steel / Poison / Ground / Fire walls so P1's Toxic / Thunder Wave / Will-O-Wisp hit a
# type immunity (status_will_land == 0 coverage), plus ordinary mons for landable status.
P2_TEAM = """
Skarmory
Ability: Keen Eye
EVs: 252 HP / 252 Def / 4 SpD
Impish Nature
- Drill Peck
- Spikes
- Roar
- Rest

Forretress
Ability: Sturdy
EVs: 252 HP / 252 Def / 4 SpD
Relaxed Nature
- Rapid Spin
- Spikes
- Earthquake
- Explosion

Claydol
Ability: Levitate
EVs: 252 HP / 128 Def / 128 SpD
Impish Nature
- Earthquake
- Psychic
- Rapid Spin
- Explosion

Houndoom
Ability: Flash Fire
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Fire Blast
- Crunch
- Hidden Power Grass
- Will-O-Wisp

Tyranitar
Ability: Sand Stream
EVs: 252 HP / 252 Atk / 4 Spe
Adamant Nature
- Rock Slide
- Earthquake
- Crunch
- Pursuit

Snorlax
Ability: Immunity
EVs: 252 HP / 252 Atk / 4 Def
Adamant Nature
- Body Slam
- Earthquake
- Self-Destruct
- Rest
"""


def _expected_effects(move, active, opp, ability_priors):
    """Independent recompute of the 8-feature effect row for one available move — the same
    inputs the encoder uses, derived through the facade so a wiring bug (offset/order/index)
    surfaces as a mismatch. status_will_land (slot 6) is the prior-weighted probability,
    resolved through the same ability distribution the matchup cells use."""
    exp = np.zeros(_PER, dtype=np.float32)
    md = gen3_data.moves.get(move.id)
    user_is_ghost = active is not None and PokemonType.GHOST in (active.type_1, active.type_2)
    if md is not None:
        exp[0] = 1.0 if (md.is_boost or (move.id == "curse" and not user_is_ghost)) else 0.0
        exp[1] = 1.0 if md.is_heal else 0.0
        exp[2] = 1.0 if md.is_protect else 0.0
        exp[3] = 1.0 if md.is_phaze else 0.0
        exp[4] = 1.0 if md.is_hazard else 0.0
        if md.status_inflicted is not None:
            exp[5] = 1.0
            if opp is not None:
                dist = _resolve_ability_distribution(opp, ability_priors)
                prob, known = status_land_estimate(move.id, md.status_inflicted, opp, dist)
                exp[6] = prob
                exp[8] = 1.0 if known else 0.0
    max_pp = getattr(move, "max_pp", 0) or 0
    exp[7] = (move.current_pp / max_pp) if max_pp else 0.0
    return exp, md, user_is_ghost


def _type_immune(move_id, opp):
    """Direct type-chart immunity check (independent of status_move_lands) for the hard
    assertion: a status move into an immune-typed target must encode status_will_land == 0."""
    immune = STATUS_MOVE_IMMUNITY.get(move_id, frozenset())
    if not immune or opp is None:
        return False
    return bool(immune & {opp.type_1, opp.type_2} - {None})


class MoveEffectFuzzPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, battle_class=Gen3Battle, **kwargs)
        self._enc = get_observation_encoder(load_mappings())
        # The exact ability priors the encoder uses, so the independent recompute resolves
        # status_will_land through the identical distribution (priors → confirmation).
        self._priors = self._enc.reactive_encoder._ability_priors
        self._seen_active: dict[str, set] = {}
        self.violations: list[str] = []
        self.decisions = 0
        self.moves_checked = 0
        self.cov = {k: 0 for k in (
            "boost", "heal", "protect", "phaze", "hazard",
            "status_lands", "status_immune", "status_prior_fractional",
            "status_known_confirmed", "status_known_prior",
            "curse_nonghost", "curse_ghost", "struggle",
        )}

    def choose_move(self, battle):
        try:
            return self._choose(battle)
        except Exception as e:  # pragma: no cover - diagnostic only
            print(f"\n[FUZZ FATAL] {battle.battle_tag} t{battle.turn}: {e}", flush=True)
            traceback.print_exc(); sys.stdout.flush()
            import os; os._exit(1)

    def _choose(self, battle):
        legal = LegalActions.from_battle(battle)
        obs = self._enc.encode(battle, hp_tracker=None, legal=legal)
        self.decisions += 1

        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        avail = list(battle.available_moves)
        is_forced_struggle = len(avail) == 1 and avail[0].id == "struggle"

        if is_forced_struggle:
            self.cov["struggle"] += 1
            block = obs[_ME_OFF:_ME_OFF + 4 * _PER]
            if np.any(block != 0.0):
                self.violations.append(
                    f"[{battle.battle_tag}] t{battle.turn}: forced-struggle but move-effect block nonzero"
                )
        else:
            for i, move in enumerate(avail[:4]):
                exp, md, user_is_ghost = _expected_effects(move, active, opp, self._priors)
                got = np.asarray(obs[_ME_OFF + i * _PER: _ME_OFF + (i + 1) * _PER], dtype=np.float32)
                # bits (0..5, 8) exact; status_will_land (6) + pp (7) continuous → approximate.
                if (not np.array_equal(got[:6], exp[:6])
                        or abs(float(got[6]) - float(exp[6])) > 1e-4
                        or abs(float(got[7]) - float(exp[7])) > 1e-4
                        or float(got[8]) != float(exp[8])):
                    self.violations.append(
                        f"[{battle.battle_tag}] t{battle.turn} slot{i} {move.id}: "
                        f"got={got.round(3).tolist()} exp={exp.round(3).tolist()}"
                    )
                self.moves_checked += 1

                # --- coverage + independent hard checks ---
                if exp[0]: self.cov["boost"] += 1
                if exp[1]: self.cov["heal"] += 1
                if exp[2]: self.cov["protect"] += 1
                if exp[3]: self.cov["phaze"] += 1
                if exp[4]: self.cov["hazard"] += 1
                if exp[5] and exp[6] > 0: self.cov["status_lands"] += 1
                if exp[5]:
                    self.cov["status_known_confirmed" if got[8] else "status_known_prior"] += 1
                    # Hard invariant: a FRACTIONAL land-probability can only come from an
                    # unrevealed-ability prior → it must be flagged not-known.
                    if 0.0 < float(got[6]) < 1.0:
                        self.cov["status_prior_fractional"] += 1
                        if float(got[8]) != 0.0:
                            self.violations.append(
                                f"[{battle.battle_tag}] t{battle.turn} {move.id}: fractional "
                                f"status_will_land={got[6]:.3f} but status_known={got[8]} (must be 0)"
                            )
                if move.id == "curse":
                    self.cov["curse_ghost" if user_is_ghost else "curse_nonghost"] += 1
                    # Hard check (independent of the facade flag): Curse is boost iff non-Ghost.
                    if float(got[0]) != (0.0 if user_is_ghost else 1.0):
                        self.violations.append(
                            f"[{battle.battle_tag}] t{battle.turn} curse: is_boost={got[0]} "
                            f"but user_is_ghost={user_is_ghost}"
                        )
                # Hard check: a status move into a type-immune target must NOT 'land'.
                if md is not None and md.status_inflicted is not None and _type_immune(move.id, opp):
                    self.cov["status_immune"] += 1
                    if float(got[6]) != 0.0:
                        self.violations.append(
                            f"[{battle.battle_tag}] t{battle.turn} {move.id} into immune "
                            f"{opp.species}: status_will_land={got[6]} (must be 0)"
                        )

        # Policy: cycle bench early (reveal/activate every mon → Ghost-Curse Gengar comes in),
        # then attack. Random-free so it's deterministic given the sim seed.
        seen = self._seen_active.setdefault(battle.battle_tag, set())
        if active is not None:
            seen.add(active.species)
        if battle.available_switches and len(seen) < 4:
            return self.create_order(battle.available_switches[0])
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
        return self.choose_default_move()

    def _battle_finished_callback(self, battle):
        super()._battle_finished_callback(battle)
        self._seen_active.pop(battle.battle_tag, None)


class Mover(Player):
    """Opponent: keep mons in, just click a move (provides stable immune-typed targets)."""

    def choose_move(self, battle):
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
        return self.choose_default_move()


async def main(n_battles: int = 40) -> None:
    p1 = MoveEffectFuzzPlayer(
        account_configuration=AccountConfiguration("MoveEffP1", None),
        battle_format=FMT, team=ConstantTeambuilder(P1_TEAM),
        start_listening=False, server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
    )
    p2 = Mover(
        account_configuration=AccountConfiguration("MoveEffP2", None),
        battle_format=FMT, team=ConstantTeambuilder(P2_TEAM),
        start_listening=False, server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1, battle_class=Gen3Battle,
    )

    await run_local_battles(p1, p2, n_battles, battle_format=FMT)

    print("=" * 70)
    print("Move-effect fuzz — gen3_move_effects_v1")
    print("=" * 70)
    print(f"battles run        : {n_battles}")
    print(f"p1 decisions       : {p1.decisions}")
    print(f"move slots checked : {p1.moves_checked}")
    print(f"coverage           : {p1.cov}")
    print(f"violations         : {len(p1.violations)}")
    for v in p1.violations[:25]:
        print("  " + v)

    required = ["boost", "heal", "protect", "phaze", "hazard",
                "status_lands", "status_immune", "curse_nonghost",
                "status_known_confirmed", "status_known_prior", "status_prior_fractional"]
    holes = [k for k in required if p1.cov[k] == 0]
    if holes:
        print(f"\nCOVERAGE HOLES (required categories never exercised): {holes}")

    ok = not p1.violations and not holes and p1.moves_checked > 0
    print("\nPASS — every available move's effect row matches the facade-derived expectation, "
          "on every decision." if ok else "\nFAIL")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    asyncio.run(main(n))
