"""FOUL PLAY'S HAND-WRITTEN LEAF EVALUATOR, ported to read ONLY the searched side's observation row.

The hidden-information binding study's HAND EVALUATOR arm (`designs/endstate/design_hidden_info_binding.md` §6.5,
rule R-EVAL §12.6; backlog X3): is our critic the binding constraint (a fixed hand-written evaluator beats it
honestly), or our information (the hand evaluator collapses in fair worlds exactly as our critic does)? To make the
comparison evaluator-against-evaluator and never evaluator-against-information, this port reads exactly the row
our critic reads: our own mons exactly, the opponent's as our observation shows them.

Source and licence
------------------
A port of ``evaluate`` in poke-engine's ``src/gen3/evaluate.rs`` (https://github.com/pmariglia/poke-engine, the
search engine behind Foul Play), read at ``/home/goodlad/dev/poke-engine/src/gen3/evaluate.rs``. Its constants are
copied verbatim. poke-engine is distributed under the MIT License:

    MIT License

    Copyright (c) 2024 pmariglia

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

The original, term by term (side one = the searched side = the row's "ours"; score = side one − side two)
---------------------------------------------------------------------------------------------------------
For every ALIVE mon (hp > 0) of a side:

- ``evaluate_pokemon``: 100 × hp / maxhp; the status term (freeze −40, sleep −25, paralysis −25, Toxic −30,
  poison −10; with Guts or Marvel Scale poison and Toxic score +10 instead; burn is −25 × the count of the mon's
  PHYSICAL moves, halved when its Special Attack stat exceeds its Attack stat, and +50 with Guts or Marvel Scale);
  +10 when it holds an item; for each move with PP ≤ 10 a penalty of 3·pp − 30; then the mon's subtotal is
  FLOORED at 0 ("a low hp pokemon could get a negative score and incentivize the other side to keep it alive");
  then +30 for being alive.
- ``evaluate_hazards``: Spikes on the mon's own side, if it is grounded (not Flying-type, not Levitate):
  1 layer −12, 2 layers −16, 3 layers −25. Not floored.
- For the ACTIVE mon only: Leech Seed −30, Substitute +75, confusion −20, and each of the five stat stages through
  the step table (+6 → 3.3, +5 → 3.15, +4 → 3.0, +3 → 2.5, +2 → 2.0, +1 → 1.0, 0 → 0, and the negatives
  mirrored) × 30 (Attack, Special Attack, Speed) or × 15 (Defense, Special Defense). Accuracy and evasion are not
  scored.

Then per side: Reflect × 20 and Light Screen × 20, where poke-engine's ``reflect`` / ``light_screen`` side
conditions are TURN COUNTERS (``SIDE_CONDITION_DURATION`` = 5 when set, decremented at each end of turn,
``gen3/generate_instructions.rs``; Foul Play's ``protocol.py`` sets 5 on ``-sidestart`` and decrements per turn):
a fresh Reflect scores 100, not 20. The port reads the same counter from the observation's OBS-FACTS screens
block (turns left, held to the engine's remaining duration by ``obs_facts_truth_test.rs``).

Where each field is read (every offset through ``Gen3ObservationEncoder.get_layout()`` and the encoders' own
read-backs, resolved ONCE by :func:`resolve_layout`; nothing here is a literal index)
-------------------------------------------------------------------------------------------------------------
- the teams: ``parts.our_team`` / ``parts.opp_team`` (6 slots × the per-mon slot); a slot is SEEN when its species
  id is non-zero; the active mon by the slot's active flag (``constants.POKEMON_ACTIVE_OFFSET``, the last dim);
- HP: the slot's ``hp`` (ours: hp / maxhp exactly; the opponent's: the protocol's displayed percent / 100);
- status: the slot's ``condition`` one-hot through ``pokemon._STATUS_STR_IDX``;
- item: the slot's ``items`` ``[id, known, consumed]``;
- moves: the slot's 4 move slots (``known``, ``current_pp`` × ``constants.MAX_PP``, ``category``, ``id``, ``type``);
- ability: the slot's ``abilities.id1`` (the revealed ability when ``known``, else the Smogon prior's top one);
- types: ``types.type1`` / ``types.type2`` as ``TypeEncoder.TYPE_TO_IDX`` ids;
- stats (burn's special-attacker halving): ``species.base_stats`` × 255 and, when ``spread.spread_known``, the
  spread's IVs × 31, EVs × 252 and nature multipliers;
- boosts and volatiles: ``parts.context`` (ours, theirs), the stage decoded as ``ActiveContextEncoder`` reads it
  back, Leech Seed / Substitute / confusion at their ``gen3_effects.VOLATILE_SLOTS`` index;
- Spikes: the global block's ``hazards`` (layers / ``constants.MAX_SPIKES``), our / their cell found through
  ``GlobalEnvEncoder.describe_vector``;
- screens: the OBS-FACTS ``screens`` sub-block (turns left / ``constants.FACTS_SCREEN_TURNS``, ours then theirs),
  CROSS-CHECKED against the global block's screen presence bits (a disagreement raises :class:`HandEvalError`).

NAMED DEVIATIONS from the full-state original (where our observation cannot give a term exactly)
-----------------------------------------------------------------------------------------------
Foul Play runs ``evaluate`` on a FULL state it builds by predicting the opponent's unknown facts (a sampled set per
seen mon and a sampled mon per never-seen slot). Our port has only the row, so:

- **D-UNSEEN** — a never-seen opponent slot (Foul Play: a sampled full set) scores as a fresh mon: alive, full HP,
  no status, an item held (+10), no PP term, and GROUNDED (it takes its side's Spikes term): 140 points before
  hazards. The opponent's team size is taken as 6 (gen3 OU; the row does not carry ``|teamsize|``), so the number
  of never-seen slots is 6 − the seen slots.
- **D-ITEM** — an opponent item that is still UNKNOWN (``[0, 0, 0]``) is scored as HELD (+10): Foul Play's sampled
  sets nearly always carry an item in gen3 OU. A revealed item scores +10; a consumed or removed one (Knock Off,
  a berry, ``-enditem``: ``consumed`` = 1) scores 0, as Foul Play's ``knocked_off`` / empty item does. Our own mons
  are exact (``[0, 0, 0]`` on our side is an empty hand). A known-empty opponent hand cannot be told from an
  unknown one in the row (both encode ``[0, 0, 0]``) and reads as held.
- **D-PP** — only KNOWN move slots are scored. The opponent's unrevealed moves (Foul Play: its predicted moves at
  full PP, penalised only when their max PP ≤ 10, e.g. Explosion) score nothing; its revealed moves use the
  reading's tracked PP. An empty slot of OUR mon (a set with fewer than four moves; Foul Play pads with ``pp = 0``
  dummies that score −30 each) scores nothing.
- **D-ABILITY** — an unrevealed opponent ability is the Smogon prior's most likely one (``id1``), as Foul Play
  scores its sampled set's ability. Only Guts, Marvel Scale and Levitate matter here.
- **D-STATS** — burn's "Special Attack > Attack" uses the mon's level-100 stats from the row's spread when the
  spread is known (our mons; an opponent only under the diagnostic oracle reveal), else the BASE stats (equal
  investment, neutral nature); Foul Play compares its predicted set's stats.
- **D-HP-TYPE** — a Hidden Power of the opponent whose type is not yet known (the bare id, type channel 0) is not
  counted as a physical move; a revealed type counts by the gen3 type split. The row's category channel for the
  bare id reads the Normal dex row (physical), which is why it is not used for Hidden Power.
- **D-TERMINAL** — a row exists only at a decision; a finished battle has no row and no hand evaluation here (the
  playout driver records the outcome; Foul Play's MCTS scores a terminal 1 / 0 and a leaf
  ``sigmoid(0.0125 · (evaluate(leaf) − evaluate(root)))``).
- Level 100 is assumed for the stat computation (gen3 OU). A boost stage outside ±6 raises, as the original
  panics.

Interface: :func:`hand_eval` (the score), :func:`hand_eval_terms` (every term, for tests and audits),
:func:`resolve_layout` / :func:`default_layout` (the offsets, resolved from whatever ``agents.observation`` is
importable, so the measurement folder's pinned twin resolves the PIN's layout).
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple

import numpy as np

# ------------------------------------------------------------------------------------------------
# Foul Play's constants (poke-engine src/gen3/evaluate.rs), VERBATIM
# ------------------------------------------------------------------------------------------------
POKEMON_ALIVE = 30.0
POKEMON_HP = 100.0

POKEMON_ATTACK_BOOST = 30.0
POKEMON_DEFENSE_BOOST = 15.0
POKEMON_SPECIAL_ATTACK_BOOST = 30.0
POKEMON_SPECIAL_DEFENSE_BOOST = 15.0
POKEMON_SPEED_BOOST = 30.0

POKEMON_BOOST_MULTIPLIER_6 = 3.3
POKEMON_BOOST_MULTIPLIER_5 = 3.15
POKEMON_BOOST_MULTIPLIER_4 = 3.0
POKEMON_BOOST_MULTIPLIER_3 = 2.5
POKEMON_BOOST_MULTIPLIER_2 = 2.0
POKEMON_BOOST_MULTIPLIER_1 = 1.0
POKEMON_BOOST_MULTIPLIER_0 = 0.0
POKEMON_BOOST_MULTIPLIER_NEG_1 = -1.0
POKEMON_BOOST_MULTIPLIER_NEG_2 = -2.0
POKEMON_BOOST_MULTIPLIER_NEG_3 = -2.5
POKEMON_BOOST_MULTIPLIER_NEG_4 = -3.0
POKEMON_BOOST_MULTIPLIER_NEG_5 = -3.15
POKEMON_BOOST_MULTIPLIER_NEG_6 = -3.3

POKEMON_FROZEN = -40.0
POKEMON_ASLEEP = -25.0
POKEMON_PARALYZED = -25.0
POKEMON_TOXIC = -30.0
POKEMON_POISONED = -10.0
POKEMON_BURNED = -25.0

LEECH_SEED = -30.0
SUBSTITUTE = 75.0
CONFUSION = -20.0

REFLECT = 20.0
LIGHT_SCREEN = 20.0

SPIKES_ONE_LAYER = -12.0
SPIKES_TWO_LAYER = -16.0
SPIKES_THREE_LAYER = -25.0

#: ``get_boost_multiplier``'s match arms.
BOOST_MULTIPLIER: Dict[int, float] = {
    6: POKEMON_BOOST_MULTIPLIER_6, 5: POKEMON_BOOST_MULTIPLIER_5, 4: POKEMON_BOOST_MULTIPLIER_4,
    3: POKEMON_BOOST_MULTIPLIER_3, 2: POKEMON_BOOST_MULTIPLIER_2, 1: POKEMON_BOOST_MULTIPLIER_1,
    0: POKEMON_BOOST_MULTIPLIER_0, -1: POKEMON_BOOST_MULTIPLIER_NEG_1, -2: POKEMON_BOOST_MULTIPLIER_NEG_2,
    -3: POKEMON_BOOST_MULTIPLIER_NEG_3, -4: POKEMON_BOOST_MULTIPLIER_NEG_4, -5: POKEMON_BOOST_MULTIPLIER_NEG_5,
    -6: POKEMON_BOOST_MULTIPLIER_NEG_6,
}
#: the five scored stats, in ``evaluate``'s order, with their weights.
BOOST_WEIGHT: Dict[str, float] = {
    "atk": POKEMON_ATTACK_BOOST, "def": POKEMON_DEFENSE_BOOST, "spa": POKEMON_SPECIAL_ATTACK_BOOST,
    "spd": POKEMON_SPECIAL_DEFENSE_BOOST, "spe": POKEMON_SPEED_BOOST,
}
#: ``evaluate_hazards``'s Spikes arms (0 layers, or anything else, scores nothing).
SPIKES_LAYER: Dict[int, float] = {1: SPIKES_ONE_LAYER, 2: SPIKES_TWO_LAYER, 3: SPIKES_THREE_LAYER}
#: the non-burn, non-poison status arms.
STATUS_FLAT: Dict[str, float] = {"frz": POKEMON_FROZEN, "slp": POKEMON_ASLEEP, "par": POKEMON_PARALYZED}
#: ``evaluate_poison``'s base scores and its Guts / Marvel Scale arm.
POISON_BASE: Dict[str, float] = {"tox": POKEMON_TOXIC, "psn": POKEMON_POISONED}
POISON_WITH_GUTS = 10.0
#: the item term (``pokemon.item != Items::NONE``) and the PP penalty's threshold / slope / offset.
ITEM_HELD = 10.0
PP_PENALTY_MAX_PP = 10
PP_PENALTY_SLOPE = 3.0
PP_PENALTY_OFFSET = -30.0

# ------------------------------------------------------------------------------------------------
# facts of the game and of the encoding (not offsets)
# ------------------------------------------------------------------------------------------------
#: the move slot's ``category`` code for a PHYSICAL move (the Rust encoder's ``slot.rs::category``:
#: 0 status, 1 physical, 2 special by the gen-3 type split; ``designs/ARCHITECTURE.md`` §1.2).
CATEGORY_PHYSICAL = 1
#: the per-mon slot's scales (``slot.rs``): base stats / 255, IVs / 31, EVs / 252.
BASE_STAT_SCALE = 255.0
IV_SCALE = 31.0
EV_SCALE = 252.0
#: gen3 OU plays at level 100 with six mons a side (D-UNSEEN, D-STATS).
LEVEL = 100
TEAM_MONS = 6
SIDES = ("ours", "theirs")
_GUTS_LIKE = ("guts", "marvelscale")


#: every ``get_layout()`` key path :func:`resolve_layout` reads (the pinned twin walks them at each pin and prints
#: the check: a path missing at a pin is a refusal, never a silent default).
LAYOUT_PATHS: Tuple[str, ...] = (
    "total_dim", "parts.our_team.start", "parts.our_team.reshape", "parts.opp_team.start", "parts.opp_team.reshape",
    "parts.context.start", "parts.context.reshape", "parts.global.start", "parts.global.dim",
    "pokemon.species.offset", "pokemon.species.layout.species_id.offset", "pokemon.species.layout.base_stats.offset",
    "pokemon.items.offset", "pokemon.items.layout.id.offset", "pokemon.items.layout.known.offset",
    "pokemon.items.layout.consumed.offset", "pokemon.types.offset", "pokemon.types.layout.type1.offset",
    "pokemon.types.layout.type2.offset", "pokemon.abilities.offset", "pokemon.abilities.layout.id1.offset",
    "pokemon.condition.offset", "pokemon.moves.offset", "pokemon.moves.layout.slots",
    "pokemon.moves.layout.slot_layout.id.offset", "pokemon.moves.layout.slot_layout.type.offset",
    "pokemon.moves.layout.slot_layout.category.offset", "pokemon.moves.layout.slot_layout.known.offset",
    "pokemon.moves.layout.slot_layout.current_pp.offset", "pokemon.hp.offset", "pokemon.spread.offset",
    "pokemon.spread.layout.ivs.offset", "pokemon.spread.layout.ivs.stats", "pokemon.spread.layout.evs.offset",
    "pokemon.spread.layout.spread_known.offset", "pokemon.spread.layout.nature.offset",
    "pokemon.spread.layout.nature.stats", "global_layout.hazards.offset", "global_layout.hazards.dim",
    "global_layout.screens.offset", "global_layout.screens.dim", "obs_facts_offset", "obs_facts_dim",
    "obs_facts.screens.offset", "obs_facts.screens.dim",
)


class HandEvalError(ValueError):
    """The row contradicts itself or the layout (a GIGO guard: never scored silently)."""


@dataclass(frozen=True)
class HandEvalLayout:
    """Every offset the evaluator reads, resolved once from the observation package (:func:`resolve_layout`).
    Slot-relative offsets are inside one per-mon slot; context-relative ones inside one active-context row;
    everything else is absolute in the row."""

    obs_dim: int
    team_start: Tuple[int, int]            # ours, theirs
    team_size: int
    slot_dim: int
    species_id: int
    base_stats: int                        # 6: hp atk def spa spd spe
    item_id: int
    item_known: int
    item_consumed: int
    type1: int
    type2: int
    ability_id1: int
    condition: int
    status_index: Mapping[str, int]        # status id -> index inside the condition one-hot
    moves: Tuple[int, ...]                 # the 4 move slots' starts
    mv_id: int
    mv_type: int
    mv_category: int
    mv_known: int
    mv_current_pp: int
    hp: int
    spread_ivs: int
    spread_evs: int
    spread_known: int
    spread_nature: int
    iv_ev_stats: Tuple[str, ...]           # the IV / EV order
    nature_stats: Tuple[str, ...]          # the nature-multiplier order
    active_flag: int
    context_start: Tuple[int, int]         # ours, theirs
    boost_pos: Mapping[str, int]           # context-relative
    boost_neg: Mapping[str, int]
    boost_scale: float
    vol_index: Mapping[str, int]           # leechseed / substitute / confusion, context-relative
    spikes: Tuple[int, int]                # absolute: ours, theirs
    max_spikes: int
    screen_turns: Mapping[Tuple[str, str], int]    # (side, condition) -> absolute
    screen_turn_scale: int
    screen_present: Mapping[Tuple[str, str], int]  # (side, condition) -> absolute (global block)
    max_pp: int
    ability_num: Mapping[str, int]         # guts / marvelscale / levitate -> num
    flying_type: int
    physical_type_ids: FrozenSet[int]      # type ids whose damaging moves are physical (gen-3 split)
    hidden_power_num: int

    def __post_init__(self) -> None:
        """Refuse a layout whose offsets leave their block (a slot field outside the slot, ...)."""
        slot_fields = [self.species_id, self.base_stats + 5, self.item_id, self.item_known, self.item_consumed,
                       self.type1, self.type2, self.ability_id1, self.hp, self.spread_ivs + 5, self.spread_evs + 5,
                       self.spread_known, self.spread_nature + 4, self.active_flag,
                       self.condition + max(self.status_index.values())]
        slot_fields += [m + o for m in self.moves
                        for o in (self.mv_id, self.mv_type, self.mv_category, self.mv_known, self.mv_current_pp)]
        ctx_dim = self.context_start[1] - self.context_start[0]
        ctx_fields = [*self.boost_pos.values(), *self.boost_neg.values(), *self.vol_index.values()]
        absolute = [self.team_start[1] + self.team_size * self.slot_dim - 1, self.context_start[1] + ctx_dim - 1,
                    *self.spikes, *self.screen_turns.values(), *self.screen_present.values()]
        if (min(slot_fields) < 0 or max(slot_fields) >= self.slot_dim or min(ctx_fields) < 0
                or max(ctx_fields) >= ctx_dim or min(absolute) < 0 or max(absolute) >= self.obs_dim):
            raise HandEvalError("a layout offset lies outside its block")


def _probe(describe: Callable[[np.ndarray], Any], dim: int, read: Callable[[Any], Dict[Any, Any]]) -> Dict[Any, int]:
    """Which cell of a block each named fact lives in, found by the block's OWN read-back: describe a unit
    vector per cell and record the keys whose read changes from the zero vector's."""
    base = read(describe(np.zeros(dim, dtype=np.float32)))
    out: Dict[Any, int] = {}
    for k in range(dim):
        e = np.zeros(dim, dtype=np.float32)
        e[k] = 1.0
        got = read(describe(e))
        for key, val in got.items():
            if base.get(key) != val:
                if key in out:
                    raise HandEvalError(f"layout probe: {key!r} read from cells {out[key]} and {k}")
                out[key] = k
    return out


def resolve_layout() -> HandEvalLayout:
    """Resolve every offset from the importable ``agents.observation`` (at a pin: the PIN's)."""
    from agents import gen3_data
    from agents.observation import constants as K
    from agents.observation import gen3_effects, obs_facts
    from agents.observation.active_context import ActiveContextEncoder
    from agents.observation.global_env import GlobalEnvEncoder
    from agents.observation.moves import HIDDEN_POWER_MOVE_NUM
    from agents.observation.pokemon import _STATUS_STR_IDX
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    from agents.observation.types import TypeEncoder

    lay = Gen3ObservationEncoder(load_mappings()).get_layout()
    parts, pk = lay["parts"], lay["pokemon"]
    team_size, slot_dim = (int(x) for x in parts["our_team"]["reshape"])
    if tuple(parts["opp_team"]["reshape"]) != (team_size, slot_dim):
        raise HandEvalError("our and their team blocks differ in shape")
    n_ctx, ctx_dim = (int(x) for x in parts["context"]["reshape"])
    if n_ctx != 2:
        raise HandEvalError(f"expected 2 active-context rows, the layout has {n_ctx}")
    mv = pk["moves"]["layout"]
    sl = mv["slot_layout"]
    sp = pk["spread"]["layout"]

    # the active context: boosts read back by the encoder's own describe; volatiles by the slot vocabulary
    ace = ActiveContextEncoder()
    acl = ace.get_layout()
    boosts = _probe(ace.describe_vector, ctx_dim, lambda d: {(s, v[0]): 1 for s, v in d["boosts"].items()})
    boost_pos = {s: boosts[(s, "+")] for s in BOOST_WEIGHT if (s, "+") in boosts}
    boost_neg = {s: boosts[(s, "-")] for s in BOOST_WEIGHT if (s, "-") in boosts}
    if set(boost_pos) != set(BOOST_WEIGHT) or set(boost_neg) != set(BOOST_WEIGHT):
        raise HandEvalError(f"boost cells not all found: + {sorted(boost_pos)} - {sorted(boost_neg)}")
    probe_vec = np.zeros(ctx_dim, dtype=np.float32)
    probe_vec[boost_pos["atk"]] = 1.0
    boost_scale = float(int(ace.describe_vector(probe_vec)["boosts"]["atk"]))   # a full cell reads +scale
    vol_off = int(acl["volatiles"]["offset"])
    vol_index = {v: vol_off + gen3_effects.VOLATILE_SLOTS.index(v) for v in ("leechseed", "substitute", "confusion")}

    # the global block: Spikes and the screen presence bits, found by GlobalEnvEncoder's own describe
    g0 = int(parts["global"]["start"])
    gdim = int(parts["global"]["dim"])
    gl = lay["global_layout"]
    gcells = _probe(GlobalEnvEncoder().describe_vector, gdim,
                    lambda d: {k: d[k] for k in ("our_spikes", "opp_spikes", "our_reflect", "opp_reflect",
                                                 "our_light_screen", "opp_light_screen")})
    hz = (int(gl["hazards"]["offset"]), int(gl["hazards"]["offset"]) + int(gl["hazards"]["dim"]))
    sc = (int(gl["screens"]["offset"]), int(gl["screens"]["offset"]) + int(gl["screens"]["dim"]))
    for key, (lo, hi) in (("our_spikes", hz), ("opp_spikes", hz), ("our_reflect", sc), ("opp_reflect", sc),
                          ("our_light_screen", sc), ("opp_light_screen", sc)):
        if key not in gcells or not lo <= gcells[key] < hi:
            raise HandEvalError(f"global cell {key!r} not inside its layout block [{lo}, {hi})")
    spikes = (g0 + gcells["our_spikes"], g0 + gcells["opp_spikes"])
    screen_present = {(side, c): g0 + gcells[f"{pre}_{c}"]
                      for side, pre in (("ours", "our"), ("theirs", "opp")) for c in ("reflect", "light_screen")}

    # the OBS-FACTS screens: turns left, found by obs_facts.describe
    f0 = int(lay["obs_facts_offset"])
    fdim = int(lay["obs_facts_dim"])
    fs = lay["obs_facts"]["screens"]
    fcells = _probe(obs_facts.describe, fdim, lambda d: dict(d["screens"]))
    screen_turns: Dict[Tuple[str, str], int] = {}
    for side, pre in (("ours", "ours"), ("theirs", "opp")):
        for c in ("reflect", "light_screen"):
            k = fcells.get(f"{pre}_{c}")
            if k is None or not int(fs["offset"]) <= k < int(fs["offset"]) + int(fs["dim"]):
                raise HandEvalError(f"obs-facts screen cell {pre}_{c} not inside the screens sub-block")
            screen_turns[(side, c)] = f0 + k
    e = np.zeros(fdim, dtype=np.float32)
    e[screen_turns[("ours", "reflect")] - f0] = 1.0
    screen_turn_scale = int(obs_facts.describe(e)["screens"]["ours_reflect"])
    if screen_turn_scale != int(K.FACTS_SCREEN_TURNS):
        raise HandEvalError(f"screen turn scale {screen_turn_scale} != FACTS_SCREEN_TURNS {K.FACTS_SCREEN_TURNS}")

    abil = {a: int(gen3_data.abilities.ability_data(a).num) for a in (*_GUTS_LIKE, "levitate")}
    physical = set()
    for tname, tid in TypeEncoder.TYPE_TO_IDX.items():
        m = gen3_data.moves.get("hiddenpower" + tname.lower())
        if m is not None and m.category.name == "PHYSICAL":
            physical.add(int(tid))
    if not physical:
        raise HandEvalError("no physical Hidden Power type resolved from gen3_data.moves")

    return HandEvalLayout(
        obs_dim=int(lay["total_dim"]),
        team_start=(int(parts["our_team"]["start"]), int(parts["opp_team"]["start"])),
        team_size=team_size, slot_dim=slot_dim,
        species_id=int(pk["species"]["offset"]) + int(pk["species"]["layout"]["species_id"]["offset"]),
        base_stats=int(pk["species"]["offset"]) + int(pk["species"]["layout"]["base_stats"]["offset"]),
        item_id=int(pk["items"]["offset"]) + int(pk["items"]["layout"]["id"]["offset"]),
        item_known=int(pk["items"]["offset"]) + int(pk["items"]["layout"]["known"]["offset"]),
        item_consumed=int(pk["items"]["offset"]) + int(pk["items"]["layout"]["consumed"]["offset"]),
        type1=int(pk["types"]["offset"]) + int(pk["types"]["layout"]["type1"]["offset"]),
        type2=int(pk["types"]["offset"]) + int(pk["types"]["layout"]["type2"]["offset"]),
        ability_id1=int(pk["abilities"]["offset"]) + int(pk["abilities"]["layout"]["id1"]["offset"]),
        condition=int(pk["condition"]["offset"]),
        status_index=dict(_STATUS_STR_IDX),
        moves=tuple(int(pk["moves"]["offset"]) + int(s["offset"]) for s in mv["slots"]),
        mv_id=int(sl["id"]["offset"]), mv_type=int(sl["type"]["offset"]),
        mv_category=int(sl["category"]["offset"]), mv_known=int(sl["known"]["offset"]),
        mv_current_pp=int(sl["current_pp"]["offset"]),
        hp=int(pk["hp"]["offset"]),
        spread_ivs=int(pk["spread"]["offset"]) + int(sp["ivs"]["offset"]),
        spread_evs=int(pk["spread"]["offset"]) + int(sp["evs"]["offset"]),
        spread_known=int(pk["spread"]["offset"]) + int(sp["spread_known"]["offset"]),
        spread_nature=int(pk["spread"]["offset"]) + int(sp["nature"]["offset"]),
        iv_ev_stats=tuple(sp["ivs"]["stats"]), nature_stats=tuple(sp["nature"]["stats"]),
        active_flag=int(K.POKEMON_ACTIVE_OFFSET),
        context_start=(int(parts["context"]["start"]), int(parts["context"]["start"]) + ctx_dim),
        boost_pos=boost_pos, boost_neg=boost_neg, boost_scale=boost_scale, vol_index=vol_index,
        spikes=spikes, max_spikes=int(K.MAX_SPIKES),
        screen_turns=screen_turns, screen_turn_scale=screen_turn_scale, screen_present=screen_present,
        max_pp=int(K.MAX_PP), ability_num=abil, flying_type=int(TypeEncoder.TYPE_TO_IDX["FLYING"]),
        physical_type_ids=frozenset(physical), hidden_power_num=int(HIDDEN_POWER_MOVE_NUM),
    )


@functools.lru_cache(maxsize=1)
def default_layout() -> HandEvalLayout:
    return resolve_layout()


# ------------------------------------------------------------------------------------------------
# reading one mon
# ------------------------------------------------------------------------------------------------

def _stage(ctx: np.ndarray, L: HandEvalLayout, stat: str) -> int:
    raw = L.boost_scale * (float(ctx[L.boost_pos[stat]]) - float(ctx[L.boost_neg[stat]]))
    st = int(round(raw))
    if abs(raw - st) > 1e-3 or st not in BOOST_MULTIPLIER:
        raise HandEvalError(f"invalid boost stage for {stat}: {raw}")
    return st


def _status(slot: np.ndarray, L: HandEvalLayout) -> Optional[str]:
    hits = [name for name, i in L.status_index.items() if slot[L.condition + i] > 0.5]
    if len(hits) > 1:
        raise HandEvalError(f"two statuses at once: {hits}")
    return hits[0] if hits else None


def _stats(slot: np.ndarray, L: HandEvalLayout) -> Tuple[float, float, bool]:
    """(Attack, Special Attack, from_spread): level-100 stats when the spread is known (D-STATS), else base."""
    base = {s: round(float(slot[L.base_stats + j]) * BASE_STAT_SCALE)
            for j, s in enumerate(("hp", "atk", "def", "spa", "spd", "spe"))}
    if slot[L.spread_known] <= 0.5:
        return float(base["atk"]), float(base["spa"]), False
    out = []
    for s in ("atk", "spa"):
        j = L.iv_ev_stats.index(s)
        iv = round(float(slot[L.spread_ivs + j]) * IV_SCALE)
        ev = round(float(slot[L.spread_evs + j]) * EV_SCALE)
        nat = round(float(slot[L.spread_nature + L.nature_stats.index(s)]), 4)   # 0.9 / 1.0 / 1.1, float32-stored
        raw = math.floor((2 * base[s] + iv + ev // 4) * LEVEL / 100) + 5
        out.append(float(math.floor(raw * nat + 1e-6)))
    return out[0], out[1], True


def _moves(slot: np.ndarray, L: HandEvalLayout) -> List[Dict[str, Any]]:
    out = []
    for m0 in L.moves:
        if slot[m0 + L.mv_known] <= 0.5:
            continue
        num = int(round(float(slot[m0 + L.mv_id])))
        pp_raw = float(slot[m0 + L.mv_current_pp]) * L.max_pp
        pp = int(round(pp_raw))
        if abs(pp_raw - pp) > 1e-3:
            raise HandEvalError(f"move {num}: PP {pp_raw} is not an integer")
        if num == L.hidden_power_num:          # D-HP-TYPE: the bare id's category channel is Normal's
            physical = int(round(float(slot[m0 + L.mv_type]))) in L.physical_type_ids
        else:
            physical = int(round(float(slot[m0 + L.mv_category]))) == CATEGORY_PHYSICAL
        out.append({"num": num, "pp": pp, "physical": physical})
    return out


def _pokemon(slot: np.ndarray, L: HandEvalLayout, own: bool) -> Dict[str, Any]:
    """``evaluate_pokemon`` + the facts the side-level terms need, for one SEEN slot."""
    hp = float(slot[L.hp])
    ability = int(round(float(slot[L.ability_id1])))
    guts_like = ability in (L.ability_num["guts"], L.ability_num["marvelscale"])
    types = {int(round(float(slot[L.type1]))), int(round(float(slot[L.type2])))}
    grounded = L.flying_type not in types and ability != L.ability_num["levitate"]
    st = _status(slot, L)
    moves = _moves(slot, L)
    t: Dict[str, Any] = {"hp_fraction": hp, "alive": hp > 0.0, "status": st, "grounded": grounded,
                         "ability": ability}
    if not t["alive"]:
        return t
    status_term = 0.0
    if st == "brn":
        if guts_like:
            status_term = -2.0 * POKEMON_BURNED
        else:
            mult = float(sum(1 for m in moves if m["physical"]))
            atk, spa, from_spread = _stats(slot, L)
            t["stats"] = {"atk": atk, "spa": spa, "from_spread": from_spread}
            if spa > atk:
                mult /= 2.0
            status_term = mult * POKEMON_BURNED
    elif st in POISON_BASE:
        status_term = POISON_WITH_GUTS if guts_like else POISON_BASE[st]
    elif st in STATUS_FLAT:
        status_term = STATUS_FLAT[st]
    known = slot[L.item_known] > 0.5
    consumed = slot[L.item_consumed] > 0.5
    if consumed and not known:
        raise HandEvalError("item consumed but not known")
    item_state = "consumed" if consumed else ("held" if known else ("none" if own else "unknown"))
    item_term = ITEM_HELD if item_state in ("held", "unknown") else 0.0      # D-ITEM
    pp_term = sum(PP_PENALTY_SLOPE * m["pp"] + PP_PENALTY_OFFSET for m in moves if m["pp"] <= PP_PENALTY_MAX_PP)
    sub = POKEMON_HP * hp + status_term + item_term + pp_term
    t.update({"hp": POKEMON_HP * hp, "status_term": status_term, "item": item_state, "item_term": item_term,
              "pp_term": pp_term, "moves": moves, "subtotal": sub, "floored": sub < 0.0,
              "alive_term": POKEMON_ALIVE})
    t["score"] = max(sub, 0.0) + POKEMON_ALIVE
    return t


def _check_row(row: np.ndarray, L: HandEvalLayout) -> np.ndarray:
    r = np.asarray(row)
    if r.ndim != 1 or r.shape[0] != L.obs_dim:
        raise HandEvalError(f"a row of shape {r.shape}, the layout's is ({L.obs_dim},)")
    if not np.isfinite(r).all():
        raise HandEvalError("a non-finite cell in the row")
    return r


def _side(r: np.ndarray, L: HandEvalLayout, s: int) -> Dict[str, Any]:
    side = SIDES[s]
    own = s == 0
    layers_raw = float(r[L.spikes[s]]) * L.max_spikes
    layers = int(round(layers_raw))
    if abs(layers_raw - layers) > 1e-3 or not 0 <= layers <= L.max_spikes:
        raise HandEvalError(f"{side}: Spikes {layers_raw}")
    hazard = SPIKES_LAYER.get(layers, 0.0)
    mons: List[Dict[str, Any]] = []
    actives = []
    for i in range(L.team_size):
        b = L.team_start[s] + i * L.slot_dim
        slot = r[b:b + L.slot_dim]
        if float(slot[L.species_id]) == 0.0:
            if slot.any():
                raise HandEvalError(f"{side} slot {i}: no species but a populated slot")
            continue
        t = _pokemon(slot, L, own)
        t["slot"] = i
        t["active"] = bool(slot[L.active_flag] > 0.5)
        if t["active"]:
            actives.append(t)
        t["hazard"] = hazard if (t["alive"] and t["grounded"]) else 0.0
        mons.append(t)
    if len(actives) > 1:
        raise HandEvalError(f"{side}: {len(actives)} actives")
    n_unseen = 0 if own else TEAM_MONS - len(mons)                          # D-UNSEEN
    if n_unseen < 0:
        raise HandEvalError(f"{side}: {len(mons)} seen mons > {TEAM_MONS}")
    unseen_each = POKEMON_HP + ITEM_HELD + POKEMON_ALIVE + hazard
    active: Dict[str, Any] = {}
    ctx = r[L.context_start[s]:L.context_start[s] + (L.context_start[1] - L.context_start[0])]
    if actives and actives[0]["alive"]:
        active["leechseed"] = LEECH_SEED if ctx[L.vol_index["leechseed"]] > 0.5 else 0.0
        active["substitute"] = SUBSTITUTE if ctx[L.vol_index["substitute"]] > 0.5 else 0.0
        active["confusion"] = CONFUSION if ctx[L.vol_index["confusion"]] > 0.5 else 0.0
        stages = {st: _stage(ctx, L, st) for st in BOOST_WEIGHT}
        active["stages"] = stages
        active["boosts"] = {st: BOOST_MULTIPLIER[v] * BOOST_WEIGHT[st] for st, v in stages.items()}
    screens: Dict[str, Any] = {}
    for c, w in (("reflect", REFLECT), ("light_screen", LIGHT_SCREEN)):
        raw = float(r[L.screen_turns[(side, c)]]) * L.screen_turn_scale
        turns = int(round(raw))
        present = float(r[L.screen_present[(side, c)]]) > 0.5
        if abs(raw - turns) > 1e-3 or turns < 0 or (turns > 0) != present:
            raise HandEvalError(f"{side} {c}: turns left {raw} vs the presence bit {present}")
        screens[c] = {"turns": turns, "term": turns * w}
    total = sum(m["score"] + m["hazard"] for m in mons if m["alive"])
    total += n_unseen * unseen_each
    total += sum(v for k, v in active.items() if k in ("leechseed", "substitute", "confusion"))
    total += sum(active.get("boosts", {}).values())
    total += sum(x["term"] for x in screens.values())
    return {"mons": mons, "spikes": layers, "hazard_per_grounded": hazard, "unseen": n_unseen,
            "unseen_each": unseen_each if n_unseen else 0.0, "active": active, "screens": screens,
            "total": total}


def hand_eval_terms(row: np.ndarray, layout: Optional[HandEvalLayout] = None) -> Dict[str, Any]:
    """Every term of the evaluation: ``{"ours": side, "theirs": side, "total": ours − theirs}``; a side carries
    its ``mons`` (each seen slot's facts and terms), ``unseen`` (D-UNSEEN), ``active`` (volatiles, stages,
    boost terms), ``screens`` (turns left and term) and its ``total``."""
    L = layout or default_layout()
    r = _check_row(row, L)
    ours, theirs = _side(r, L, 0), _side(r, L, 1)
    return {"ours": ours, "theirs": theirs, "total": float(ours["total"] - theirs["total"])}


def hand_eval(row: np.ndarray, layout: Optional[HandEvalLayout] = None) -> float:
    """Foul Play's ``evaluate`` from the row's side (side one = ours): positive = good for the searched side."""
    return float(hand_eval_terms(row, layout)["total"])


def foul_play_leaf(value: float, root_value: float) -> float:
    """Foul Play's MCTS leaf scale (poke-engine ``mcts.rs``): ``sigmoid(0.0125 · (evaluate(leaf) − evaluate(root)))``
    ("tuned so that ~200 points is very close to 1.0"). A terminal scores 1 / 0 there."""
    return 1.0 / (1.0 + math.exp(-0.0125 * (value - root_value)))
