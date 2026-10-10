"""THE END-OF-TURN RESIDUAL per mon — `--eot-residual on` (`gen3_static_recovery_v1`, config v150;
`designs/endstate/design_static_tokens.md` §13, `design_hand_computed_features.md` N4).

For EVERY mon on BOTH sides: the HP it gains or loses at the END OF THIS TURN **if it is the mon on its side's field at
that point** (the active if it stays in, a benched mon if it is the one that switches in), per component, as fractions
of its OWN max HP (signed: heal +, chip −), plus the HP-clamped net. Added to the mon's token as content (zero-init,
bias-free `IsolatedLinear`, built LAST: one-lever init), beside the static diagnostic's facts. Why: static loses most
on stall / semi-stall / Wish / Spikes teams (`measurements/static_diag_2026-10-09/`), the long residual race, and under
`static` the op's `g` ledger reaches a token only through the op content; the switch decision had no residual read.

THE COMPONENTS (`EOT_FACTS`), each VERIFIED in `deps/pokemon-showdown` at the pinned commit (gen 3 = the gen3 mod over
gen4 over gen5 … over the base; `measurements/static_recovery_2026-10-09/README.md` §2 carries every citation):

* ``leftovers`` +1/16 × P(holds Leftovers): the item is EXACT where it is known (our team; a revealed opponent item,
  0 once consumed / removed), the species' format-filtered Smogon prior where it is not (an X5 hidden slot: its
  hypothesis species). (`data/mods/gen4/items.ts` leftovers, order 10 sub 4.)
* ``weather`` −1/16 under sand (Rock / Ground / Steel immune, and Sand Veil) or hail (Ice immune; gen 3 has no ability
  immunity to hail). Read on the mon's types as they will be on the field (the type columns: an active's current types,
  a benched mon's base ones). Zero on the weather's LAST turn (a timed weather with 1 turn left expires at this
  residual, before the chip: `sim/battle.ts` `fieldEvent`'s duration branch; the Rust engine's `residuals.rs`), and
  scaled by P(no Cloud Nine / Air Lock on the field). (`data/conditions.ts` sandstorm / hail, order 8.)
* ``rain_dish`` +1/16 in rain × P(Rain Dish) × the same live / unsuppressed weather. (`data/mods/gen3/abilities.ts`.)
* ``status`` the next status tick: burn −1/8 and poison −1/8 (gen 3: `data/mods/gen6/conditions.ts` sets 1/8 for burn);
  Toxic −(n + 1)/16 where n is the ticks taken (the observation's counter, which resets on a switch, so a benched toxic
  mon switching in ticks −1/16: `data/conditions.ts` tox `onSwitchIn`); × (1 − P(Shed Skin)/3) (Shed Skin's 33 % cure
  resolves at sub-order 3, before the tick at 6).
* ``leech_drain`` −1/8 for the ACTIVE mon carrying Leech Seed (a volatile: cleared on switch-out).
* ``leech_gain`` for every mon of the side OPPOSITE a seeded active (the heal goes to whoever is in the seeder's slot):
  + min(the seeded mon's max HP / 8, its current HP) / this mon's max HP, × (1 − 2·P(the seeded mon has Liquid Ooze))
  (gen 3: the seeder TAKES that amount instead: `data/mods/gen4/abilities.ts` liquidooze). Max HP: ours EXACT (the
  spread in the observation), theirs the op's neutral estimate (2·base + 31 + 110), Shedinja 1 — the op's own
  convention. The one cross-mon quantity; it assumes the seeded active stays in.
* ``wish`` +1/2 for every mon of a side whose Wish heals at the end of THIS turn (gen 3 heals HALF THE RECIPIENT'S max
  HP: `data/mods/gen4/moves.ts` wish replaces the condition; base gen 5+ stores the wisher's). The observation's
  board flag is exactly that event (`trackers/history.rs`: a Wish used last turn) at 0.5.
* ``ingrain`` +1/16 for an active with Ingrain; ``curse`` −1/4 for an active with Curse (the Ghost one);
  ``nightmare`` −1/4 for an active with Nightmare × P(it does not wake this turn) (the observation's sleep-wake
  belief) × (1 − P(Shed Skin)/3) — Nightmare ends on waking.
* ``net`` the sum of the above, clamped to [−HP, 1 − HP] (a mon cannot lose more than it has nor heal past full): the
  clamp of the expected sum, not the expectation of a clamp.

Every component is the NOMINAL fraction (Showdown floors each amount, minimum 1). Every column is alive-gated, and an
opponent slot with no species (the belief-off ablation only; X5 always holds a hypothesis) reads 0. EXCLUDED: partial
trapping (−1/16 in gen 3, but its last turn ends without damage and the trap timer is an OBS-FACTS-only fact), Future
Sight / Doom Desire (priced at cast, not in the observation), Dig / Dive's semi-invulnerable turn (no observation
column), Baton Pass carrying Ingrain / Curse / Leech Seed (a choice, not this turn's state).

ONE formula for both sides (every table is indexed by the mon's own ids), so the rule is equivariant over each side's
team slots by construction.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch

from agents.observation.constants import (BOOSTS_DIM, POKEMON_CONDITION_OFFSET, POKEMON_COUNTER_OFFSET,
                                          POKEMON_ITEMS_OFFSET, POKEMON_SLEEP_BELIEF_OFFSET, POKEMON_SPREAD_OFFSET,
                                          TEAM_SIZE, WEATHER_ONEHOT_DIM)
from agents.model.extractor_ctx import ability_known, revealed_ability1_ids
from agents.observation.gen3_effects import VOLATILE_SLOTS

if TYPE_CHECKING:
    from agents.model.extractor_ctx import ExtractorContext

#: `--eot-residual`'s legal values (``off`` builds nothing).
EOT_RESIDUAL_MODES = ("off", "on")

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
#: The observation's Toxic counter is min(ticks, 8) / 8 (`encoder/slot.rs`), the sleep-wake belief's P(wake) its 2nd column.
_TOX_COUNTER_SCALE = 8.0
_P_WAKE_COL = 1
#: The weather block: one-hot [NONE, SUN, RAIN, SAND, HAIL], permanent, turns remaining / 5 (`encoder/mod.rs`).
_W_RAIN, _W_SAND, _W_HAIL = 2, 3, 4
_W_PERMANENT, _W_TURNS = WEATHER_ONEHOT_DIM, WEATHER_ONEHOT_DIM + 1
_WEATHER_MAX_TURNS = 5.0
#: Whether a TIMED weather with `remaining` turns left still applies its residual this turn, by `remaining` 0..5: it
#: expires at this residual (no chip) when 1 remains; 0 never occurs for a present timed weather.
_TIMED_WEATHER_LIVE = (0.0, 0.0, 1.0, 1.0, 1.0, 1.0)

_VOL = {name: BOOSTS_DIM + VOLATILE_SLOTS.index(name) for name in ("leechseed", "ingrain", "curse", "nightmare")}


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

    def forward(self, ctx: 'ExtractorContext', op: Any, opp_concrete: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[B, 12, EOT_DIM] (`EOT_FACTS`): ours slots 0–5, theirs 6–11. ``opp_concrete`` [B,6] (1 = the slot's
        species is known or hypothesised); None ⇒ revealed = not believed."""
        pp = ctx.pokemon_part
        dt = pp.dtype
        B = ctx.batch_size
        hp = ctx.hp_and_active[:, :, 0]                                                    # [B,12]
        active = ctx.hp_and_active[:, :, -1]                                               # [B,12] 0/1
        alive = (hp > 0).to(dt)
        if opp_concrete is None:
            opp_concrete = (~ctx.opp_believed_mask).to(dt)
        gate = alive * torch.cat([torch.ones_like(opp_concrete), opp_concrete.to(dt)], dim=1)   # [B,12]
        sp = ctx.species_ids                                                               # [B,12]

        # --- per-mon abilities: known exact, else the species' Smogon marginal (ONE formula, both sides) ---
        known = ability_known(ctx).to(dt)[..., None]                                        # [B,12,1]
        ab = known * self.ABILITY_EOT[revealed_ability1_ids(ctx)] + (1.0 - known) * self.SPECIES_EOT_PRIOR[sp]

        # --- Leftovers: the item where known (0 once consumed / removed), the species prior where not ---
        item_known = pp[..., POKEMON_ITEMS_OFFSET + 1]
        consumed = pp[..., POKEMON_ITEMS_OFFSET + 2]
        p_lo = (item_known * self.IS_LEFTOVERS[ctx.item_ids] * (1.0 - consumed)
                + (1.0 - item_known) * self.SPECIES_LEFTOVERS_PRIOR[sp])
        leftovers = p_lo / 16.0

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
            * unsup / 16.0
        rain_dish = rain * ab[..., _A_RAIN_DISH] * unsup / 16.0

        # --- status: the next tick (Toxic at its counter + 1; a benched mon's counter is 0), Shed Skin's cure first ---
        cond = pp[..., POKEMON_CONDITION_OFFSET:POKEMON_CONDITION_OFFSET + 7]
        tox_ticks = pp[..., POKEMON_COUNTER_OFFSET + 1] * _TOX_COUNTER_SCALE
        no_cure = 1.0 - ab[..., _A_SHED_SKIN] / 3.0
        status = -(cond[..., _BRN] / 8.0 + cond[..., _PSN] / 8.0 + cond[..., _TOX] * (tox_ticks + 1.0) / 16.0) * no_cure

        # --- the actives' volatiles (cleared by a switch: a benched mon reads 0) ---
        def _vol(name: str) -> torch.Tensor:
            v = torch.zeros(B, 2 * TEAM_SIZE, dtype=dt, device=ctx.device)
            v[ar, ctx.our_active_idx] = ctx.our_ctx_raw[:, _VOL[name]].to(dt)
            v[ar, TEAM_SIZE + ctx.opp_active_local] = ctx.opp_ctx_raw[:, _VOL[name]].to(dt)
            return v * active
        seeded = _vol("leechseed")
        leech_drain = -seeded / 8.0
        ingrain = _vol("ingrain") / 16.0
        curse = -_vol("curse") / 4.0
        p_stay_asleep = 1.0 - pp[..., POKEMON_SLEEP_BELIEF_OFFSET + _P_WAKE_COL]
        nightmare = -_vol("nightmare") * cond[..., _SLP] * p_stay_asleep * no_cure / 4.0

        # --- Leech Seed's heal: to whoever is in the slot opposite a seeded active (Liquid Ooze inverts it) ---
        maxhp = self._max_hp(ctx, op)                                                       # [B,12]
        drained = maxhp * torch.minimum(hp, hp.new_full((), 1.0 / 8.0)) * seeded            # [B,12] HP points
        ooze = 1.0 - 2.0 * ab[..., _A_LIQUID_OOZE]
        give = (drained * ooze)                                                             # [B,12]
        give_from_opp = give[:, TEAM_SIZE:].sum(1)                                          # [B] (one seeded active)
        give_from_our = give[:, :TEAM_SIZE].sum(1)
        recv = torch.cat([give_from_opp[:, None].expand(B, TEAM_SIZE),
                          give_from_our[:, None].expand(B, TEAM_SIZE)], dim=1)
        leech_gain = recv / maxhp

        # --- Wish: half of the RECIPIENT's max HP, to whoever is in the slot (the board flag is 0.5 or 0) ---
        nmr = ctx.non_matchup_rest.to(dt)
        wish = torch.cat([nmr[:, self._wish_our, None].expand(B, TEAM_SIZE),
                          nmr[:, self._wish_opp, None].expand(B, TEAM_SIZE)], dim=1)

        parts = [leftovers, weather, rain_dish, status, leech_drain, leech_gain, wish, ingrain, curse, nightmare]
        total = torch.stack(parts, dim=-1).sum(-1)
        net = torch.minimum(torch.maximum(total, -hp), 1.0 - hp)
        out = torch.stack(parts + [net], dim=-1)                                            # [B,12,EOT_DIM]
        return out * gate[..., None]

    def _max_hp(self, ctx: 'ExtractorContext', op: Any) -> torch.Tensor:
        """[B,12] max HP: ours EXACT from the observation's spread, theirs the op's neutral estimate
        (2·base + 31 + 110); Shedinja (base HP 1) is 1 on both sides. The op's own convention."""
        base_hp = op.BASE_STATS[ctx.species_ids][..., 0].to(ctx.pokemon_part.dtype)          # [B,12]
        spread = ctx.pokemon_part[:, :TEAM_SIZE, POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + 12]
        ours = 2.0 * base_hp[:, :TEAM_SIZE] + spread[..., 0] * 31.0 + spread[..., 6] * 252.0 / 4.0 + 110.0
        theirs = 2.0 * base_hp[:, TEAM_SIZE:] + 31.0 + 110.0
        formula = torch.cat([ours, theirs], dim=1)
        shedinja = self.SHEDINJA_HP_ONE[ctx.species_ids]
        out: torch.Tensor = shedinja + (1.0 - shedinja) * formula
        return out
