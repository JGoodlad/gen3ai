"""THE END-OF-TURN RESIDUAL per mon — built with the op (`gen3_static_recovery_v1`, config v150;
`designs/endstate/design_static_tokens.md` §13, `design_hand_computed_features.md` N4).

For EVERY mon on BOTH sides: the HP it gains or loses at the END OF THIS TURN **if it is the mon on its side's field at
that point** (the active if it stays in, a benched mon if it is the one that switches in), per component, as fractions
of its OWN max HP (signed: heal +, chip −), plus the net of the residual-order HP chain. Added to the mon's token as content (zero-init,
bias-free `IsolatedLinear`, built LAST: one-lever init), beside the static diagnostic's facts. Why: static loses most
on stall / semi-stall / Wish / Spikes teams (`measurements/static_diag_2026-10-09/`), the long residual race, and under
`static` the op's `g` ledger reaches a token only through the op content; the switch decision had no residual read.

THE COMPONENTS (`EOT_FACTS`), each VERIFIED in `deps/pokemon-showdown` at the pinned commit (gen 3 = the gen3 mod over
gen4 over gen5 … over the base; `measurements/static_recovery_2026-10-09/README.md` §2 carries every citation). Each
amount is a fraction ``1/d`` of the mon's max HP that Showdown hands ``this.damage`` / ``this.heal``, which make it the
INTEGER ``max(1, ⌊maxhp / d⌋)`` (`sim/battle.ts` ``spreadDamage`` ``clampIntRange(damage, 1)``; ``heal`` ``if (damage
&& damage <= 1) damage = 1; trunc``). gen3_residual_exact_v1: for OUR mons (exact max HP) the rule computes that
integer (`damage_kinds.residual_points`). THEIR max HP ``M`` is not observed: it is the species' Smogon max-HP BELIEF
(gen3_their_max_hp_belief_v1, `damage_kinds.their_max_hp`), and their amount is the EXPECTATION of the same integer
rule over it, ``E[max(1, ⌊M / d⌋) / M]`` of their max HP (`damage_kinds.their_residual_points`) — a Shedinja's (max HP
1, species-determined, on either side) its one HP, exactly. Below, ``⌊·/d⌋`` names that amount.

* ``leftovers`` +⌊·/16⌋ × P(holds Leftovers): the item is EXACT where it is known (our team; a revealed opponent item,
  0 once consumed / removed), the species' format-filtered Smogon prior where it is not (an X5 hidden slot: its
  hypothesis species). (`data/mods/gen4/items.ts` leftovers ``this.heal(baseMaxhp / 16)``, order 10 sub 4.)
* ``weather`` −⌊·/16⌋ under sand (Rock / Ground / Steel immune, and Sand Veil) or hail (Ice immune; gen 3 has no ability
  immunity to hail). Read on the mon's types as they will be on the field (the type columns: an active's current types,
  a benched mon's base ones). Zero on the weather's LAST turn (a timed weather with 1 turn left expires at this
  residual, before the chip: `sim/battle.ts` `fieldEvent`'s duration branch; the Rust engine's `residuals.rs`), and
  scaled by P(no Cloud Nine / Air Lock on the field). (`data/conditions.ts` sandstorm / hail ``onWeather``
  ``this.damage(baseMaxhp / 16)``; `data/mods/gen4/conditions.ts` field order 8.)
* ``rain_dish`` +⌊·/16⌋ in rain × P(Rain Dish) × the same live / unsuppressed weather. (`data/mods/gen3/abilities.ts`
  raindish ``this.heal(baseMaxhp / 16)``, order 10 sub 3.)
* ``status`` the next status tick: burn −⌊·/8⌋ (gen 3: `data/mods/gen6/conditions.ts` brn ``baseMaxhp / 8``) and
  poison −⌊·/8⌋ (`data/conditions.ts` psn); Toxic −⌊·/16⌋ × min(n + 1, 15) where n is the ticks taken (`data/conditions.ts`
  tox ``clampIntRange(baseMaxhp / 16, 1) * stage`` after ``if (stage < 15) stage++``; the observation's counter,
  min(n, 15)/15, decoded by `damage_op_pairwise.toxic_next_stage`; it resets on a switch, so a benched toxic mon
  switching in ticks at stage 1: tox ``onSwitchIn``); × (1 − P(Shed Skin)/3) (Shed Skin's 33 % cure resolves at
  sub-order 3, before the tick at 6; `data/mods/gen4/conditions.ts` order 10 sub 6).
* ``leech_drain`` for the ACTIVE mon carrying Leech Seed (a volatile: cleared on switch-out): −(the INTEGER drain)
  / its max HP, the drain ``min(hp, max(1, ⌊max HP / 8⌋))`` (`leech_seed_drained`, gen3_leech_seed_exact_v1:
  `data/moves.ts` leechseed ``this.damage(baseMaxhp / 8)`` → ``clampIntRange(·, 1)``, the HP stopping at 0), where
  ``hp`` is the HP LEFT after the residuals Showdown runs before it (the HP chain below).
* ``leech_gain`` for every mon of the side OPPOSITE a seeded active (the heal goes to whoever is in the seeder's slot):
  + that drain / this mon's max HP, × (1 − 2·P(the seeded mon has Liquid Ooze)) (gen 3: the seeder TAKES that amount
  instead: `data/mods/gen4/abilities.ts` liquidooze; no Big Root in gen 3). The drain is in HP points for both sides
  (ours exact; theirs on the scale of their max-HP belief's harmonic mean, so THEIR fractions are the expected ones
  and what OUR mon gains from their drain is the harmonic-scale amount, not E[⌊M / 8⌋] — at most 0.62 HP low
  (Diglett; over 0.3 HP for 25 of the 216 species with usage)). The one cross-mon quantity; it assumes the seeded
  active stays in.
* ``wish`` +⌊·/2⌋ for every mon of a side whose Wish heals at the next end-of-turn residual (gen 3 heals HALF THE
  RECIPIENT'S max HP: `data/mods/gen4/moves.ts` wish ``onEnd`` ``this.heal(target.baseMaxhp / 2)``, order 7; base gen
  5+ stores the wisher's). The observation's board flag is exactly that event at 0.5 (`trackers/history.rs::WishFold`:
  a successful Wish cast one residual ago; at a replacement decision AFTER this turn's residual it means the NEXT
  turn's, `gen3_wish_flag_truth_v1`).
* ``ingrain`` +⌊·/16⌋ for an active with Ingrain (`data/moves.ts` ingrain; gen 4 mod order 10 sub 1); ``curse``
  −⌊·/4⌋ for an active with Curse (the Ghost one; sub 8); ``nightmare`` −⌊·/4⌋ for an active with Nightmare (sub 7)
  × P(it does not wake this turn) (the observation's sleep-wake belief) × (1 − P(Shed Skin)/3) — Nightmare ends on
  waking.
* ``net`` the HP change of THE HP CHAIN: the components applied ONE AT A TIME in Showdown's gen-3 residual order
  (`sim/battle.ts` ``comparePriority``: order, then speed, then sub-order; gen 3 = the gen4 mod's orders) — Wish (7),
  the weather (field 8), then this mon's order-10 block: Ingrain (1), Rain Dish (3), Leftovers (4), Leech Seed's drain
  and the seeder slot's heal (5), the status tick (6), Nightmare (7), Curse (8) — each heal capped at max HP and void
  on a mon a chip has KO-ed, each chip stopping at 0 (`_residual_step`). A full-HP Leftovers holder that is poisoned
  nets −⌊·/8⌋ (the heal is wasted at full HP), where the clamp of the sum read −1/16. Probabilistic amounts (a prior
  item / ability, the wake belief, P(Cloud Nine)) enter the chain as their EXPECTED amounts: the chain of
  expectations, not the expectation of a chain. DECLARED LIMITS: the two actives' order-10 blocks run fastest first, so
  when BOTH actives are seeded the slower one's drain is capped after the faster one's heal — the chain places every
  heal at sub-order 5 of the receiving mon; and the residual gen-3 Berries (Sitrus / Oran / the pinch berries,
  `data/mods/gen3/items.ts`, order 10 sub 4) are not modelled.

Every column is alive-gated, and an opponent slot with no species (the belief-off ablation only; X5 always holds a
hypothesis) reads 0. EXCLUDED: partial trapping (−1/16 in gen 3, but its last turn ends without damage and the trap
timer is an OBS-FACTS-only fact), Future Sight / Doom Desire (priced at cast, not in the observation), Dig / Dive's
semi-invulnerable turn (no observation column), Baton Pass carrying Ingrain / Curse / Leech Seed (a choice, not this
turn's state).

ONE formula for both sides (every table is indexed by the mon's own ids), so the rule is equivariant over each side's
team slots by construction.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch

from agents.observation.constants import (BOOSTS_DIM, POKEMON_CONDITION_OFFSET, POKEMON_COUNTER_OFFSET,
                                          POKEMON_ITEMS_OFFSET, POKEMON_SLEEP_BELIEF_OFFSET, TEAM_SIZE,
                                          WEATHER_ONEHOT_DIM)
from agents.model.damage_kinds import (our_hp_points, our_max_hp, our_spread, our_team_spread, residual_points,
                                       their_max_hp, their_residual_points)
from agents.model.damage_op_pairwise import toxic_next_stage
from agents.model.extractor_ctx import ability_known, held_item_ids, revealed_ability1_ids
from agents.observation.gen3_effects import VOLATILE_SLOTS

if TYPE_CHECKING:
    from agents.model.extractor_ctx import ExtractorContext

#: (`--eot-residual` and `--g-ledger`, the flags that chose this rule beside the op's own older `g` ledger, were
#: DELETED by the legacy deletion's U1: the per-mon residual is built whenever the op is, and the `g` cell (and every
#: reader of it: `c4`'s nets, the static op content) reads THIS rule through `g_cells` — the model holds ONE
#: end-of-turn rule.)

#: The per-mon columns, in order (signed fractions of the mon's own max HP).
EOT_FACTS: Tuple[str, ...] = ("leftovers", "weather", "rain_dish", "status", "leech_drain", "leech_gain", "wish",
                              "ingrain", "curse", "nightmare", "net")
EOT_DIM = len(EOT_FACTS)

#: The abilities the rule reads, one column each of `ABILITY_EOT` (Cloud Nine and Air Lock share the suppression one).
_ABILITY_COLS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("sand_veil", ("sandveil",)), ("rain_dish", ("raindish",)), ("shed_skin", ("shedskin",)),
    ("liquid_ooze", ("liquidooze",)), ("weather_suppress", ("cloudnine", "airlock")),
)
_A_SAND_VEIL, _A_RAIN_DISH, _A_SHED_SKIN, _A_LIQUID_OOZE, _A_SUPPRESS = range(len(_ABILITY_COLS))

#: The condition one-hot is [None, BRN, PAR, SLP, FRZ, PSN, TOX].
_BRN, _SLP, _PSN, _TOX = 1, 3, 5, 6
#: The sleep-wake belief's P(wake) is its 2nd column. (The observation's Toxic counter is min(stage, 15) / 15
#: (`encoder/slot.rs`, `TOXIC_STAGE_MAX`); `damage_op_pairwise.toxic_next_stage` decodes it to the next tick's stage.)
_P_WAKE_COL = 1
#: The weather block: one-hot [NONE, SUN, RAIN, SAND, HAIL], permanent, turns remaining / 5 (`encoder/mod.rs`).
_W_RAIN, _W_SAND, _W_HAIL = 2, 3, 4
_W_PERMANENT, _W_TURNS = WEATHER_ONEHOT_DIM, WEATHER_ONEHOT_DIM + 1
_WEATHER_MAX_TURNS = 5.0
#: Whether a TIMED weather with `remaining` turns left still applies its residual this turn, by `remaining` 0..5: it
#: expires at this residual (no chip) when 1 remains; 0 never occurs for a present timed weather.
_TIMED_WEATHER_LIVE = (0.0, 0.0, 1.0, 1.0, 1.0, 1.0)

_VOL = {name: BOOSTS_DIM + VOLATILE_SLOTS.index(name) for name in ("leechseed", "ingrain", "curse", "nightmare")}


def leech_seed_drained(eighth: torch.Tensor, cur_hp: torch.Tensor) -> torch.Tensor:
    """The HP Leech Seed takes from the seeded mon at the residual, in HP points — and so what the seeder's slot
    heals (gen3_leech_seed_exact_v1). `data/moves.ts` leechseed ``onResidual``: ``this.damage(pokemon.baseMaxhp / 8,
    pokemon, target)`` → `sim/battle.ts` ``spreadDamage``: ``clampIntRange(damage, 1)`` = ``max(1, ⌊maxhp / 8⌋)``
    → `sim/pokemon.ts` ``damage``: the HP stops at 0, so the amount DEALT is ``min(hp, max(1, ⌊maxhp / 8⌋))``;
    ``if (damage) this.heal(damage, target, pokemon)`` heals that amount (`sim/battle.ts` ``heal`` truncates an
    integer to itself; `sim/pokemon.ts` ``heal`` caps it at the recipient's missing HP — the rule's ``net`` clamp).
    Big Root does not exist in gen 3 (`data/items.ts` bigroot ``gen: 4``). A max HP below 8 (Shedinja's 1) still
    loses 1. ``eighth`` is the mon's 1/8 residual amount in HP points — `damage_kinds.residual_points` of OUR exact
    max HP, `damage_kinds.their_residual_points` (the rule in expectation over their max-HP belief) for THEIR mon.
    ``cur_hp`` is the HP LEFT when the drain runs (the forward's HP chain: after Wish, the weather, Ingrain, Rain
    Dish and Leftovers)."""
    return torch.minimum(cur_hp, eighth)


def _residual_step(h: torch.Tensor, delta: torch.Tensor, maxhp: torch.Tensor) -> torch.Tensor:
    """One residual applied to HP ``h`` (HP points), in order: ``delta`` > 0 heals, < 0 chips; the HP stays in
    ``[0, maxhp]`` (`sim/pokemon.ts` ``damage`` stops at 0, ``heal`` at max HP), and a mon a chip has KO-ed heals
    nothing (`sim/battle.ts` ``heal``: ``if (!target?.hp) return false``). The KO gate is ``min(h, 1)`` — a live mon has
    >= 1 HP point, so it is exactly 0 / 1 on an integer chain and needs no comparison."""
    healed = torch.minimum(h + delta.clamp(min=0.0) * h.clamp(max=1.0), maxhp)
    return (healed + delta.clamp(max=0.0)).clamp(min=0.0)


def _num(table: Any, ident: str) -> int:
    rec = table.get(ident)
    if rec is None:
        raise ValueError(f"eot_residual: {ident!r} did not resolve in the gen-3 data — the rule would silently read "
                         "nothing for it. Fix the id or regenerate data/pokemon/.")
    return int(rec.num)


def build_eot_tables(n_species: int, n_items: int, n_abilities: int) -> Dict[str, torch.Tensor]:
    """The rule's constant tables (non-persistent buffers): the Leftovers indicator and per-species prior, the five
    ability columns and their per-species Smogon marginals, the sand / hail type immunities."""
    from agents import gen3_data
    from agents.model.belief_tables import build_item_prior
    from agents.model.damage_tables import N_TYPE_IDX, _T2I, build_species_ability_marginal
    lo = _num(gen3_data.items, "leftovers")
    is_lo = torch.zeros(n_items, dtype=torch.float32)
    is_lo[lo] = 1.0
    ability = torch.zeros(n_abilities, len(_ABILITY_COLS), dtype=torch.float32)
    for k, (_col, ids) in enumerate(_ABILITY_COLS):
        for ident in ids:
            ability[_num(gen3_data.abilities, ident), k] = 1.0
    sand_immune = torch.zeros(N_TYPE_IDX, dtype=torch.float32)
    for t in ("ROCK", "GROUND", "STEEL"):
        sand_immune[_T2I[t]] = 1.0
    hail_immune = torch.zeros(N_TYPE_IDX, dtype=torch.float32)
    hail_immune[_T2I["ICE"]] = 1.0
    # Shedinja's max HP is 1 whatever its spread (`species.maxHP`); it is the only gen-3 species with base HP 1.
    shedinja = torch.zeros(n_species, dtype=torch.float32)
    shedinja[_num(gen3_data.species, "shedinja")] = 1.0
    return {
        "IS_LEFTOVERS": is_lo,
        "SPECIES_LEFTOVERS_PRIOR": build_item_prior(n_species, n_items)[:, lo].contiguous(),
        "ABILITY_EOT": ability,
        "SPECIES_EOT_PRIOR": build_species_ability_marginal(n_species, ability, neutral=0.0),
        "TYPE_SAND_IMMUNE": sand_immune,
        "TYPE_HAIL_IMMUNE": hail_immune,
        "TIMED_WEATHER_LIVE": torch.tensor(_TIMED_WEATHER_LIVE, dtype=torch.float32),
        "SHEDINJA_HP_ONE": shedinja,
    }


class EotResidualRule(torch.nn.Module):
    """THE end-of-turn residual rule (module docstring). No parameters: only constant tables, so building it draws
    nothing and adds no state_dict key. Called with the context the op prices with (under X5 the hypothesis
    context) and the op (for its base-stat table)."""

    IS_LEFTOVERS: torch.Tensor
    SPECIES_LEFTOVERS_PRIOR: torch.Tensor
    ABILITY_EOT: torch.Tensor
    SPECIES_EOT_PRIOR: torch.Tensor
    TYPE_SAND_IMMUNE: torch.Tensor
    TYPE_HAIL_IMMUNE: torch.Tensor
    TIMED_WEATHER_LIVE: torch.Tensor
    SHEDINJA_HP_ONE: torch.Tensor

    def __init__(self, layout: Dict[str, Any]) -> None:
        super().__init__()
        from agents.model.board_tokens import board_offsets
        for name, t in build_eot_tables(layout["max_species"], layout["max_items"],
                                        layout["max_abilities"]).items():
            self.register_buffer(name, t, persistent=False)
        off = board_offsets(layout)
        self._wish_our, self._wish_opp = int(off.wish_our), int(off.wish_opp)

    def forward(self, ctx: 'ExtractorContext', op: Any, opp_concrete: Optional[torch.Tensor] = None,
                opp_alive: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[B, 12, EOT_DIM] (`EOT_FACTS`): ours slots 0–5, theirs 6–11. ``opp_concrete`` [B,6] (1 = the slot's
        species is known or hypothesised); None ⇒ revealed = not believed. ``opp_alive`` [B,6] (X5: the roster's
        `alive`, i.e. ADDRESSABILITY): their side's alive gate — a hypothesis slot's HP cell is not its liveness
        (`x5_opp_mon_axis_test`); None ⇒ the HP cell (> 0), both sides."""
        pp = ctx.pokemon_part
        dt = pp.dtype
        B = ctx.batch_size
        hp = ctx.hp_and_active[:, :, 0]                                                    # [B,12]
        active = ctx.hp_and_active[:, :, -1]                                               # [B,12] 0/1
        alive = (hp > 0).to(dt)
        if opp_alive is not None:
            alive = torch.cat([alive[:, :6], opp_alive.to(dt)], dim=1)
        if opp_concrete is None:
            opp_concrete = (~ctx.opp_believed_mask).to(dt)
        gate = alive * torch.cat([torch.ones_like(opp_concrete), opp_concrete.to(dt)], dim=1)   # [B,12]
        sp = ctx.species_ids                                                               # [B,12]

        # --- per-mon abilities: known exact, else the species' Smogon marginal (ONE formula, both sides) ---
        known = ability_known(ctx).to(dt)[..., None]                                        # [B,12,1]
        ab = known * self.ABILITY_EOT[revealed_ability1_ids(ctx)] + (1.0 - known) * self.SPECIES_EOT_PRIOR[sp]

        # --- each amount in HP POINTS (gen3_residual_exact_v1): Showdown's integer max(1, ⌊maxhp / d⌋) for OUR mons
        #     (`damage_kinds.residual_points`), its expectation over THEIR max-HP belief for theirs
        #     (`damage_kinds.their_residual_points`, gen3_their_max_hp_belief_v1; a Shedinja's 1 either way) ---
        their_ids = ctx.species_ids[:, TEAM_SIZE:]                                          # [B,6]
        their = their_max_hp(op, their_ids)
        maxhp = torch.cat([self._our_max_hp(ctx, op), their.maxhp.to(dt)], dim=1)           # [B,12]
        p16, p8, p4, p2 = (torch.cat([residual_points(maxhp[:, :TEAM_SIZE], d),
                                      their_residual_points(op, their_ids, d).to(dt)], dim=1)
                           for d in (16.0, 8.0, 4.0, 2.0))

        # --- Leftovers: the item where known (0 once consumed / removed), the species prior where not ---
        item_known = pp[..., POKEMON_ITEMS_OFFSET + 1]
        consumed = pp[..., POKEMON_ITEMS_OFFSET + 2]
        p_lo = (item_known * self.IS_LEFTOVERS[held_item_ids(ctx)] * (1.0 - consumed)
                + (1.0 - item_known) * self.SPECIES_LEFTOVERS_PRIOR[sp])
        leftovers = p_lo * p16

        # --- weather: live this residual (not on a timed weather's last turn) and not suppressed ---
        w = ctx.weather_feature.to(dt)                                                      # [B,7]
        remaining = (w[:, _W_TURNS] * _WEATHER_MAX_TURNS).round().long().clamp(0, 5)
        perm = w[:, _W_PERMANENT]
        live = perm + (1.0 - perm) * self.TIMED_WEATHER_LIVE[remaining]                     # [B]
        p_sup = ab[..., _A_SUPPRESS]                                                        # [B,12]
        ar = torch.arange(B, device=ctx.device)
        sup_our_act = p_sup[ar, ctx.our_active_idx]                                         # [B]
        sup_opp_act = p_sup[ar, TEAM_SIZE + ctx.opp_active_local]
        # mon i on the field at the end of the turn faces the OTHER side's current active
        sup_other = torch.cat([sup_opp_act[:, None].expand(B, TEAM_SIZE),
                               sup_our_act[:, None].expand(B, TEAM_SIZE)], dim=1)           # [B,12]
        unsup = (1.0 - p_sup) * (1.0 - sup_other)
        t1, t2 = ctx.type1_ids, ctx.type2_ids
        sand_immune = (self.TYPE_SAND_IMMUNE[t1] + self.TYPE_SAND_IMMUNE[t2]).clamp(max=1.0)
        hail_immune = (self.TYPE_HAIL_IMMUNE[t1] + self.TYPE_HAIL_IMMUNE[t2]).clamp(max=1.0)
        sand = (w[:, _W_SAND] * live)[:, None]
        hail = (w[:, _W_HAIL] * live)[:, None]
        rain = (w[:, _W_RAIN] * live)[:, None]
        weather = -(sand * (1.0 - sand_immune) * (1.0 - ab[..., _A_SAND_VEIL]) + hail * (1.0 - hail_immune)) \
            * unsup * p16
        rain_dish = rain * ab[..., _A_RAIN_DISH] * unsup * p16

        # --- status: the next tick (Toxic at its counter + 1; a benched mon's counter is 0), Shed Skin's cure first ---
        cond = pp[..., POKEMON_CONDITION_OFFSET:POKEMON_CONDITION_OFFSET + 7]
        tox_stage = toxic_next_stage(pp[..., POKEMON_COUNTER_OFFSET + 1])    # min(ticks taken + 1, 15)
        no_cure = 1.0 - ab[..., _A_SHED_SKIN] / 3.0
        status = -(cond[..., _BRN] * p8 + cond[..., _PSN] * p8 + cond[..., _TOX] * tox_stage * p16) * no_cure

        # --- the actives' volatiles (cleared by a switch: a benched mon reads 0) ---
        def _vol(name: str) -> torch.Tensor:
            v = torch.zeros(B, 2 * TEAM_SIZE, dtype=dt, device=ctx.device)
            v[ar, ctx.our_active_idx] = ctx.our_ctx_raw[:, _VOL[name]].to(dt)
            v[ar, TEAM_SIZE + ctx.opp_active_local] = ctx.opp_ctx_raw[:, _VOL[name]].to(dt)
            return v * active
        seeded = _vol("leechseed")
        ingrain = _vol("ingrain") * p16
        curse = -_vol("curse") * p4
        p_stay_asleep = 1.0 - pp[..., POKEMON_SLEEP_BELIEF_OFFSET + _P_WAKE_COL]
        nightmare = -_vol("nightmare") * cond[..., _SLP] * p_stay_asleep * no_cure * p4

        # --- Wish: half of the RECIPIENT's max HP, to whoever is in the slot (the board flag is 0.5 or 0) ---
        nmr = ctx.non_matchup_rest.to(dt)
        wish = torch.cat([nmr[:, self._wish_our, None].expand(B, TEAM_SIZE),
                          nmr[:, self._wish_opp, None].expand(B, TEAM_SIZE)], dim=1) * 2.0 * p2

        # --- THE HP CHAIN, in Showdown's residual order (module docstring): Wish, the weather, then this mon's
        #     Ingrain, Rain Dish, Leftovers; Leech Seed drains the HP LEFT after them; then the status, Nightmare,
        #     Curse. Ours in exact HP points, theirs the reported fraction × their max-HP belief's harmonic mean ---
        cur_hp = torch.cat([our_hp_points(hp[:, :TEAM_SIZE], maxhp[:, :TEAM_SIZE]),
                            their.frac_points(hp[:, TEAM_SIZE:]).to(dt)], dim=1)             # [B,12] HP points
        h = cur_hp
        for before_leech in (wish, weather, ingrain, rain_dish, leftovers):
            h = _residual_step(h, before_leech, maxhp)
        drained = leech_seed_drained(p8, h) * seeded                                        # [B,12] HP points
        ooze = 1.0 - 2.0 * ab[..., _A_LIQUID_OOZE]
        give = (drained * ooze)                                                             # [B,12]
        give_from_opp = give[:, TEAM_SIZE:].sum(1)                                          # [B] (one seeded active)
        give_from_our = give[:, :TEAM_SIZE].sum(1)
        recv = torch.cat([give_from_opp[:, None].expand(B, TEAM_SIZE),
                          give_from_our[:, None].expand(B, TEAM_SIZE)], dim=1)
        h = _residual_step(h, -drained, maxhp)
        h = _residual_step(h, recv, maxhp)
        for chip in (status, nightmare, curse):
            h = _residual_step(h, chip, maxhp)

        parts = [leftovers, weather, rain_dish, status, -drained, recv, wish, ingrain, curse, nightmare]
        parts = [x / maxhp for x in parts]
        net = (h - cur_hp) / maxhp
        out = torch.stack(parts + [net], dim=-1)                                            # [B,12,EOT_DIM]
        return out * gate[..., None]

    def _our_max_hp(self, ctx: 'ExtractorContext', op: Any) -> torch.Tensor:
        """[B,6] OUR max HP, EXACT from the observation's spread (our Shedinja, base HP 1, is 1). THEIR max HP is
        never computed here: it is `damage_kinds.their_max_hp`'s belief (the forward divides their amounts by its
        harmonic mean, so each fraction is the expected one; their Shedinja is exactly 1 there)."""
        our_ids = ctx.species_ids[:, :TEAM_SIZE]
        base_hp = op.BASE_STATS[our_ids][..., 0].to(ctx.pokemon_part.dtype)                  # [B,6]
        spread = our_team_spread(ctx)
        iv, ev_points, _nat = our_spread(spread)
        ours = our_max_hp(base_hp, iv[..., 0], ev_points[..., 0])                         # the one spelling
        shedinja = self.SHEDINJA_HP_ONE[our_ids]
        out: torch.Tensor = shedinja + (1.0 - shedinja) * ours
        return out


#: The `g` cell's four columns as sums of `EOT_FACTS` components (`g_cells`): every component lands in exactly one
#: column, so the four sum to the rule's unclamped total (`c4` sums them).
G_FROM_EOT: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("heal", ("leftovers", "rain_dish", "wish", "ingrain")),
    ("weather", ("weather",)),
    ("status", ("status", "curse", "nightmare")),
    ("leech", ("leech_drain", "leech_gain")),
)


def g_cells(eot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """the one end-of-turn rule (`gen3_endstate_facts_v1`): the `g` ledger's ``(our_cells [B,6,4], opp_cells [B,6,4])`` —
    ``[heal, weather, status, leech]`` per mon, signed max-HP fractions — from THE end-of-turn rule's per-mon
    components ``eot`` ``[B,12,EOT_DIM]`` (`EotResidualRule.forward`). The coarse ledger's four columns are
    Leftovers / weather / status / Leech Seed drain; here each column also carries the rule's other components of its
    kind (Rain Dish, Wish and Ingrain heal; Curse and Nightmare tick with the status; the seeder's heal with the
    drain), so `g`, `c4` and the static op content price the end of the turn by the SAME rule as the per-mon residual."""
    idx = {n: i for i, n in enumerate(EOT_FACTS)}
    cols = [torch.stack([eot[..., idx[c]] for c in comps], dim=-1).sum(-1) for _name, comps in G_FROM_EOT]
    g = torch.stack(cols, dim=-1)                                                       # [B,12,4]
    return g[:, :TEAM_SIZE], g[:, TEAM_SIZE:]
