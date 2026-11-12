"""gen3_nonformula_damage_v1 — the DamageOperator's damage model for every move whose damage is
NOT the plain gen-3 base-power formula, applied identically by every kernel (outgoing AND incoming).

The tables are built and DECLARED in `damage_tables.DAMAGE_MODELS` (one row per such move, with its
Showdown citation); this module is the one place the kernels turn those tables into damage. Two
pure functions, no parameters, no state:

``effective_bp``
    `MOVE_BP` is the move's base power AT THE ATTACKER'S FULL HP. Two kinds depend on the
    attacker's current HP fraction ``f``:

    * ``bp_hp_scaled`` (Eruption / Water Spout): ``BP = 150 · f`` (min 1) —
      `data/moves.ts` eruption ``basePowerCallback``: ``move.basePower * pokemon.hp / pokemon.maxhp``.
    * ``bp_flail`` (Flail / Reversal): the gen-3 step table on
      ``ratio = max(floor(48 · f), 1)`` — `data/mods/gen3/moves.ts` flail / reversal
      ``basePowerCallback`` (200 / 150 / 100 / 80 / 40 / 20 at ratio < 2 / 5 / 10 / 17 / 33 / else).
      A ``1e-4`` nudge inside the floor makes an exact quotient land on its integer: with maxhp ≤ 714
      a non-integer ``48·hp/maxhp`` sits ≥ 1/714 from the next integer, so the nudge never moves a
      genuine value across a step (standing rule 8 — no decision is made within a rounding error).

``nonformula_rolls``
    The damage IN HP of the kinds the BP formula cannot express, each exact given the HP the op
    already carries (Showdown `sim/battle-actions.ts` ``getDamage``: immunity first, then ``ohko`` →
    ``target.hp`` in gen 3, then ``damageCallback``, then ``damage: 'level' | N``):

    * ``fixed`` — a constant: Seismic Toss / Night Shade = the user's level (100), Dragon Rage 40,
      Sonic Boom 20, Psywave its EXPECTATION 100 at level 100 (declared approximation).
    * ``target_hp_frac`` — a fraction of the TARGET's CURRENT HP: Super Fang ½; the OHKO moves 1
      (gen 3 OHKO deals ``target.hp``).
    * ``endeavor`` — ``max(0, target_hp − attacker_hp)`` (the gen-4 mod's ``onTry`` fails it when
      the user is not lower).

    They ignore Atk/Def, the roll, crits, screens and weather, and RESPECT type / ability immunity
    (``eff > 0``) — so all three rolls are one value and P(KO) is ``acc · [the hit KOs]``.

``beatup_*`` (gen3_beatup_exact_v1) — gen-3 Beat Up, the ONE kind that is still the BP formula, with the
    formula's inputs swapped (so it rides the kernels' own roll / screen / crit / P(KO) arithmetic, not
    `nonformula_rolls`). `data/mods/gen3/moves.ts` beatup: the move is TYPELESS (``'???'`` — no STAB, no
    effectiveness, no ability read) and SPECIAL (Light Screen halves it; Reflect and burn do not); it hits
    once per ally in ``side.pokemon.filter(a => !a.fainted && !a.status)`` (the USER counts iff healthy);
    hit *i* is the ordinary formula at BP 10 with ``A`` = ally *i*'s species BASE Atk and ``D`` = the
    TARGET's species BASE Def (``event.modifier = 1``: no boost, item or ability reaches either stat). Every
    hit's pre-roll damage is ``(42/50)·10·A_i/D + 2``, so the sum over the party is

        core = (42/50) · 10 · S / D_base + 2 · N       with  S = Σ ally base Atk,  N = ally count

    — `S` and `N` are all a kernel needs of the attacking side, and `D_base` all it needs of the target.
    The kernels then multiply by the same ``0.925`` mean roll, apply the same screen, crit ×2 and KO ramp
    as for any other move (a per-hit roll is independent in the sim; the op's one shared roll over the
    summed damage over-states the spread of the KO ramp by ≈ √N — the P(KO) edges are exact, the middle is
    smoother than the sim's).

    *Our side* is fully known (our party's HP + status are in the obs). *The opponent's side*: a revealed
    mon is eligible iff alive with no status; **an unrevealed slot is eligible with certainty** (a mon that
    never entered the battle cannot have fainted or been statused) and contributes its EXPECTED base Atk
    under the Species-Clause-filtered usage prior (`DamageOperator.unrevealed_species_probs` — the op's one
    convention for a hidden mon). The expectation is exact for the sum (linearity), so E[damage] is exact
    given those marginals; X5's U3 revisits hidden-mon handling. An unrevealed DEFENDER uses
    ``D_eff = 1 / E[1 / D_base]`` (the harmonic mean), which makes the same formula return E[damage]
    exactly. Declared limits: a forme reads its base species' stats (the op's num-keyed convention); Beat
    Up into a Substitute and the per-hit crit roll are not modelled.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from agents.model.damage_op_layout import _DMG_CHIP_CAP, _DMG_CRIT_CAP
from agents.model.extractor_ctx import our_rows
from agents.observation.constants import (CONDITION_DIM, POKEMON_CONDITION_OFFSET, POKEMON_SPREAD_DIM,
                                          POKEMON_SPREAD_OFFSET, TEAM_SIZE)
from agents.observation.types import TypeEncoder

#: The TypeEncoder index of '???' — the type Beat Up has in battle (neutral row/column of every table).
TYPELESS_TYPE_IDX = TypeEncoder.TYPE_TO_IDX["???"]
# Condition one-hot [None, BRN, PAR, SLP, FRZ, PSN, TOX] — columns 1.. are the six MAJOR statuses
# (`ally.status` in Showdown); the all-zero placeholder an unrevealed slot carries reads "no status".
_STATUS_COLS = slice(POKEMON_CONDITION_OFFSET + 1, POKEMON_CONDITION_OFFSET + CONDITION_DIM)
_BS_ATK_COL, _BS_DEF_COL = 1, 2             # BASE_STATS columns [hp, atk, def, spa, spd, spe]

# Flail / Reversal (gen 3): ratio thresholds and the BP each band takes (`data/mods/gen3/moves.ts`).
_FLAIL_RATIO_STEPS = (2.0, 5.0, 10.0, 17.0, 33.0)
_FLAIL_BP = (200.0, 150.0, 100.0, 80.0, 40.0, 20.0)
_FLAIL_FLOOR_NUDGE = 1e-4


def flail_bp(hp_frac: torch.Tensor) -> torch.Tensor:
    """Gen-3 Flail / Reversal base power from the attacker's current HP fraction."""
    ratio = torch.floor(hp_frac * 48.0 + _FLAIL_FLOOR_NUDGE).clamp(min=1.0)
    bp = torch.full_like(hp_frac, _FLAIL_BP[-1])
    # Walk the bands from the widest down so the narrowest (lowest HP) band wins.
    for thr, val in reversed(list(zip(_FLAIL_RATIO_STEPS, _FLAIL_BP[:-1]))):
        bp = torch.where(ratio < thr, torch.full_like(bp, val), bp)
    return bp


def effective_bp(bp: torch.Tensor, hp_scaled: torch.Tensor, flail: torch.Tensor,
                 attacker_hp_frac: torch.Tensor) -> torch.Tensor:
    """The BP the formula should use: `bp` (= MOVE_BP, the BP at full HP) with the two
    attacker-HP-dependent kinds resolved. `hp_scaled` / `flail` are the gathered 0/1 tables;
    all four BROADCAST (the caller passes the attacker axis with a 1-wide move axis)."""
    bp, hp_scaled, flail, f = torch.broadcast_tensors(bp, hp_scaled, flail, attacker_hp_frac)
    out = torch.where(hp_scaled > 0, (bp * f).clamp(min=1.0), bp)
    return torch.where(flail > 0, flail_bp(f), out)


#: A floor nudge for a heal: the heal fractions are 1/4, 1/2, 2/3 and 1, so with maxhp <= 714 a non-integer
#: ``frac · maxhp`` sits >= 1/4 from the next integer, while fp32 misses an integer product by < 1e-4 (2/3 is
#: not representable) — the nudge lands an exact product on its integer and never moves a genuine one.
_HEAL_FLOOR_NUDGE = 1e-3


#: The observation's spread block as `encoder::slot::spread` writes it (`POKEMON_SPREAD_DIM` = 18): IV / 31 (×6),
#: EV / 252 (×6), spread_known, the nature multiplier (×5: atk, def, spa, spd, spe).
_OBS_IV_SCALE, _OBS_EV_SCALE = 31.0, 252.0
_SPREAD_IV, _SPREAD_EV, _SPREAD_NATURE = slice(0, 6), slice(6, 12), slice(13, 18)


def our_spread(spread: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """OUR mon's spread decoded from the observation's block ``[..., >= 18]`` (gen3_stat_ev_points_v1, THE one
    decode): ``(iv [...,6], ev_points [...,6], nature [...,5])``. ``iv`` is the integer IV (``round(f32(iv/31) ·
    31)``: the read-back misses the integer by an ulp); ``ev_points`` is what the stat formula ADDS, Showdown's
    ``tr(ev / 4)`` (`sim/battle.ts` ``statModify``: ``tr(tr(2·base + iv + tr(evs / 4)) · level/100 + 5)``), so an
    EV that is not a multiple of 4 (85 → 21 points, not 21.25) counts as the game counts it; ``nature`` the
    multiplier column as observed."""
    iv = (spread[..., _SPREAD_IV] * _OBS_IV_SCALE).round()
    ev_points = ((spread[..., _SPREAD_EV] * _OBS_EV_SCALE).round() / 4.0).floor()
    return iv, ev_points, spread[..., _SPREAD_NATURE]


def our_team_spread(ctx: Any) -> torch.Tensor:
    """OUR six mons' spread block ``[B, 6, POKEMON_SPREAD_DIM]``, `our_stats` / `our_spread`'s input for the whole
    team: read off the REAL rows (`extractor_ctx.our_rows`), never as ``ctx.pokemon_part[:, :TEAM_SIZE, ...]``. The
    values are the same; through a hypothesis context's block each of the IV and EV reads is a three-load select
    chain, and the ``[B, 6]`` max-HP sum built on them is what Triton 3.4.0 fails to compile at ``XBLOCK = 16``
    (B = 2: `triton_xblock16_cat_read_test.py`; identity: `our_rows_spelling_test.py`)."""
    return our_rows(ctx)[..., POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]


def our_max_hp(base_hp: torch.Tensor, iv_hp: torch.Tensor, ev_points_hp: torch.Tensor) -> torch.Tensor:
    """OUR mon's max HP at L100 (gen3_our_hp_exact_v1: THE one spelling): ``2·base + IV + tr(EV/4) + 110``
    (`sim/battle.ts` ``statModify``'s HP branch at level 100), integer-exact; Shedinja (the one gen-3 species with base
    HP 1) is 1 whatever its spread (`sim/pokemon.ts` ``if (this.species.maxHP) stats.hp = this.species.maxHP``;
    `data/pokedex.ts` shedinja ``maxHP: 1``; gen3_nature_floor_v1). ``iv_hp`` / ``ev_points_hp`` come from
    `our_spread` (the EV already in stat points)."""
    return torch.where(base_hp == 1.0, torch.ones_like(base_hp), 2.0 * base_hp + iv_hp + ev_points_hp + 110.0)


def gen3_nature_stat(raw: torch.Tensor, nature_mult: torch.Tensor) -> torch.Tensor:
    """THE nature step of a non-HP stat (gen3_nature_floor_v1), every stat of OUR spread through it: `sim/battle.ts`
    ``statModify`` ``stat = tr(tr(stat · 110, 16) / 100)`` (``· 90`` for the hindered stat), so a boosted / hindered
    stat is an INTEGER (Timid 359 Speed → 394, not 394.9). ``raw`` = ``tr(2·base + IV + tr(EV/4)) + 5`` (an integer);
    ``nature_mult`` = the observation's 0.9 / 1.0 / 1.1, rounded to its percentage so the f32 product cannot land a hair
    under an integer. ``raw · pct`` is an integer < 2^24 and exact in f32; a non-integer quotient sits >= 0.01 from the
    next integer, so the floor is exact. The 16-bit truncation never binds in gen 3 (the largest stat, 559, · 110 <
    65,536)."""
    pct = (nature_mult * 100.0).round()
    return (raw * pct / 100.0).floor()


def our_stats(base: torch.Tensor, spread: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """OUR mon's L100 ``(max_hp [...], stats [...,5])`` from its species base stats ``[..., 6]`` (hp, atk, def, spa,
    spd, spe) and the observation's spread block, each Showdown's integer: max HP via `our_max_hp`, the five others
    `gen3_nature_stat` of ``2·base + IV + tr(EV/4) + 5`` (the EV floored as the game floors it, `our_spread`; the nature
    truncated as the game truncates it). Our Speed comes from here too (`damage_op_speed`)."""
    iv, ev_points, nature = our_spread(spread)
    max_hp = our_max_hp(base[..., 0], iv[..., 0], ev_points[..., 0])
    return max_hp, gen3_nature_stat(2.0 * base[..., 1:6] + iv[..., 1:6] + ev_points[..., 1:6] + 5.0, nature)


#: The gen-3 end-of-turn residual amounts, as the DENOMINATOR of the max-HP fraction Showdown passes to
#: ``this.damage`` / ``this.heal`` (`eot_residual` cites each): Leftovers, Rain Dish, Ingrain, sand / hail and a Toxic
#: stage 1/16; burn, poison and Leech Seed 1/8; Curse and Nightmare 1/4; Wish 1/2.
RESIDUAL_DENOMS: Tuple[float, ...] = (16.0, 8.0, 4.0, 2.0)


def residual_points(maxhp: torch.Tensor, denom: float) -> torch.Tensor:
    """THE gen-3 end-of-turn residual amount in HP points (gen3_residual_exact_v1), for a mon of EXACT max HP
    ``maxhp`` (OUR mon; a Shedinja's 1): `sim/battle.ts` ``spreadDamage`` ``clampIntRange(damage, 1)`` = ``max(1,
    ⌊maxhp / denom⌋)`` for a chip, and ``heal``'s ``if (damage && damage <= 1) damage = 1; damage = trunc(damage)`` —
    the same integer — for a heal (the HP clamps are the caller's: `sim/pokemon.ts` ``damage`` stops at 0, ``heal``
    at max HP). A 1-max-HP Shedinja loses its one HP to a 1/16 sand chip. THEIR mon's max HP is a belief, and its
    amount the expectation of this rule over it: `their_residual_points`. ``denom`` must be a power of two
    (`RESIDUAL_DENOMS`), so ``maxhp / denom`` of an integer is exact in f32."""
    if denom not in RESIDUAL_DENOMS:
        raise ValueError(f"residual_points: denominator {denom!r} is not a gen-3 residual's (one of {RESIDUAL_DENOMS}); "
                         "a non-power-of-two quotient is not exact in f32 and the floor would read it wrong.")
    return torch.floor(maxhp / denom).clamp(min=1.0)


# ------------------------------------------------------------------ gen3_their_max_hp_belief_v1 (backlog E13)
class TheirHp:
    """THE op's belief about an OPPONENT mon's max HP ``M`` (gen3_their_max_hp_belief_v1: THE one spelling of their
    max HP; `designs/model/op_contracts.md`). Their max HP is hidden: ``2·base + IV + tr(EV / 4) + 110`` with an HP
    EV the opponent chose. What the agent legitimately holds is the SPECIES and Smogon's usage spreads for it, so
    the belief is the species' discrete Smogon distribution of ``M`` (`belief_tables.build_their_max_hp`), carried
    as the three reads the op's arithmetic needs — each exact for the quantity it enters:

    * ``inv`` = ``E[1 / M]``: a damage of ``d`` HP is the expected FRACTION ``d · inv`` of their max HP (`maxhp` =
      ``1 / inv``, the harmonic mean, is the max HP to divide by; `frac_points` their HP on that scale);
    * ``mean`` = ``E[M]``: their HP in POINTS, where it is an operand in points (their own Endeavor), is
      ``fraction · mean`` (`points`);
    * ``atoms`` / ``weights`` ``[..., K]``: the distribution as a K-atom mixture, over which P(KO) is taken — a KO
      is a threshold in ``M``, so no single value prices it (`their_ko_given_hit`, `their_fixed_ko`).

    Built ONLY by `their_max_hp` (species ids) and `expected_their_max_hp` (a species distribution): nothing of the
    opponent's actual spread can reach it."""

    __slots__ = ("inv", "mean", "atoms", "weights")

    def __init__(self, inv: torch.Tensor, mean: torch.Tensor, atoms: torch.Tensor, weights: torch.Tensor) -> None:
        self.inv, self.mean, self.atoms, self.weights = inv, mean, atoms, weights

    @property
    def maxhp(self) -> torch.Tensor:
        """``1 / E[1 / M]``: the max HP a damage in HP divides by to give its EXPECTED fraction of their max HP."""
        out: torch.Tensor = 1.0 / self.inv
        return out

    def frac_points(self, hp_frac: torch.Tensor) -> torch.Tensor:
        """Their current HP on the scale of `maxhp` (``hp_frac · maxhp``): the operand of a damage read as a
        fraction of their max HP."""
        return hp_frac / self.inv

    def points(self, hp_frac: torch.Tensor) -> torch.Tensor:
        """Their current HP in HP POINTS, in expectation: ``hp_frac · E[M]``."""
        return hp_frac * self.mean

    def lift(self, *index: Any) -> "TheirHp":
        """The same belief indexed by ``index`` on its leading (non-atom) dims — the broadcast axes a kernel adds
        (``their.lift(slice(None), None)``: ``[B] → [B, 1]``, the atoms ``[B, K] → [B, 1, K]``)."""
        tail = index + (slice(None),)
        return TheirHp(self.inv[index], self.mean[index], self.atoms[tail], self.weights[tail])


def their_max_hp(op: Any, species_ids: torch.Tensor) -> TheirHp:
    """THEIR max-HP belief for the opponent mons of species ``species_ids`` (any shape, INTEGER dex nums): the
    op's `THEIR_HP_*` rows. A Shedinja is exactly 1 (species-determined). The ONLY input is the species: a
    non-integer tensor (a spread, a stat, an HP value) RAISES — their true spread must never price their max HP."""
    if species_ids.is_floating_point() or species_ids.dtype == torch.bool:
        raise TypeError(f"their_max_hp: species_ids must be integer dex nums, got {species_ids.dtype}. Their max HP "
                        "is a belief from the SPECIES (Smogon's spreads); no spread, stat or HP value may enter it.")
    return TheirHp(op.THEIR_HP_INV[species_ids], op.THEIR_HP_MEAN[species_ids],
                   op.THEIR_HP_ATOMS[species_ids], op.THEIR_HP_WEIGHTS[species_ids])


def expected_their_max_hp(op: Any, their: TheirHp, sp_probs: torch.Tensor, where: torch.Tensor) -> TheirHp:
    """``their`` ``[B, 6]`` with the slots ``where`` ``[B, 6]`` (bool) — a slot whose SPECIES is itself a
    distribution ``sp_probs`` (``[B, S]`` shared by the six slots, or ``[B, 6, S]``): the unrevealed defender under
    the usage prior, X5's OTHER — read as the expectation over species: ``E[1 / M]`` and ``E[M]`` are the mixture's
    (linear in the probabilities, so exact). The atoms there collapse to the one value ``1 / E[1 / M]``: the op
    takes no P(KO) against an averaged defender (it is nulled), and a point is the honest content of that slot."""
    if sp_probs.shape[-1] != op.THEIR_HP_INV.shape[0]:
        raise ValueError(f"expected_their_max_hp: sp_probs' last dim {sp_probs.shape[-1]} is not the species axis "
                         f"({op.THEIR_HP_INV.shape[0]}): only a distribution over SPECIES may average their max HP.")
    # Mass the distribution places on NO species (an all-zero row: a slot with nothing to average) reads as the
    # tables' row 0, the unowned num (a point mass) — so the belief is finite on every row, never 1 / 0.
    missing = (1.0 - sp_probs.sum(dim=-1)).clamp(min=0.0)
    inv_u = sp_probs @ op.THEIR_HP_INV.to(sp_probs.dtype) + missing * op.THEIR_HP_INV[0]
    mean_u = sp_probs @ op.THEIR_HP_MEAN.to(sp_probs.dtype) + missing * op.THEIR_HP_MEAN[0]
    if sp_probs.dim() == 2:                     # one marginal per battle: broadcast the RESULTS over the slots
        inv_u, mean_u = inv_u[:, None], mean_u[:, None]
    inv = torch.where(where, inv_u.to(their.inv.dtype), their.inv)
    mean = torch.where(where, mean_u.to(their.mean.dtype), their.mean)
    first = torch.zeros_like(their.weights)
    first[..., 0] = 1.0
    atoms = torch.where(where[..., None], (1.0 / inv)[..., None], their.atoms)
    return TheirHp(inv, mean, atoms, torch.where(where[..., None], first, their.weights))


def _their_averaged(op: Any, ctx: Any) -> torch.Tensor:
    """``[B, 6]`` bool: THEIR slots whose species is AVERAGED by the caller's species distribution — hidden, and not
    holding a concrete species (without X5 every hidden slot; under X5 only a slot holding OTHER: a hypothesis
    slot's species is in the context)."""
    believed: torch.Tensor = ctx.opp_believed_mask
    x5 = op.stash.x5
    return believed if x5 is None else believed & ~x5.concrete.bool()


def their_hp_slots(op: Any, ctx: Any, sp_probs: Optional[torch.Tensor] = None) -> TheirHp:
    """THEIR six slots' max-HP belief ``[B, 6]`` (atoms ``[B, 6, K]``) from the context the op prices with: each
    slot's species (a revealed mon; under X5 a hidden slot's hypothesis species), and in X5's OTHER-mode pass the
    OTHER slots as the tail's expectation (`expected_their_max_hp` over ``P_tail``, the speed mixture's rule).
    ``sp_probs`` (`unrevealed_species_probs`: the defender kernels' species distribution): the slots it AVERAGES
    (`_their_averaged`) read its expectation too."""
    their = their_max_hp(op, ctx.species_ids[:, TEAM_SIZE:2 * TEAM_SIZE])
    x5 = op.stash.x5
    if x5 is not None and x5.override is not None:
        their = expected_their_max_hp(op, their, x5.species_probs, x5.override)
    if sp_probs is not None:
        their = expected_their_max_hp(op, their, sp_probs, _their_averaged(op, ctx))
    return their


def _their_hp_interval(their: TheirHp, hp_frac: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(lo, hi)`` ``[..., K]``: per max-HP atom, the HP-point interval their reported fraction ``hp_frac`` leaves
    (`ko_exact.opp_hp_bounds`: HP Percentage Mod's (p − 1 %, p] of THAT max HP; a full mon exact)."""
    from agents.model.ko_exact import opp_hp_bounds
    f = hp_frac.unsqueeze(-1)
    return opp_hp_bounds(f * their.atoms, f, their.atoms, 1.0)


def _mix_atoms(weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``Σ_k w_k · x_k`` over the atom axis, written about the first atom (``x_0 + Σ_{k>0} w_k · (x_k − x_0)``, the
    same number since the weights sum to 1) so that atoms which AGREE return their value EXACTLY: a KO every max HP
    grants reads 1.0 and one none grants 0.0, where the plain sum returned the fp32 weights' own sum (1 ± 1.2e-7)."""
    first = x[..., :1]
    out: torch.Tensor = (first.squeeze(-1) + (weights[..., 1:] * (x[..., 1:] - first)).sum(-1)).clamp(0.0, 1.0)
    return out


def their_ko_given_hit(mean_dmg: torch.Tensor, crit_mean_dmg: torch.Tensor, their: TheirHp, hp_frac: torch.Tensor,
                       crit_p: Optional[torch.Tensor]) -> torch.Tensor:
    """P(KO | hit) on THEIR mon: `ko_exact.ko_given_hit` (the 16 rolls + the crit over the reported percent bin)
    for each max-HP atom, mixed by the atoms' weights — the KO probability over rolls, the percent bin AND their
    max HP. ``mean_dmg`` / ``crit_mean_dmg`` in HP points; ``their`` / ``hp_frac`` / ``crit_p`` broadcast to them
    (the atoms on one trailing axis)."""
    from agents.model.ko_exact import ko_given_hit
    lo, hi = _their_hp_interval(their, hp_frac)
    ko = ko_given_hit(mean_dmg.unsqueeze(-1), crit_mean_dmg.unsqueeze(-1), lo, hi,
                      None if crit_p is None else crit_p.unsqueeze(-1))
    return _mix_atoms(their.weights, ko)


def their_fixed_ko(their: TheirHp, hp_frac: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
    """P(a fixed damage of ``fixed`` HP KOs THEIR mon) = P(their remaining HP <= ``fixed``) (`sim/pokemon.ts`
    ``damage``: faint at ``hp <= 0``), their HP uniform on each atom's interval and mixed over the atoms: Seismic
    Toss into a 35 % Skarmory is P(max HP <= ~285), not the yes / no of one assumed max HP. Continuous in the
    fraction (no cut on a product of estimates); 0 where the kind deals no fixed damage."""
    lo, hi = _their_hp_interval(their, hp_frac)
    p = ((fixed.unsqueeze(-1) - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)
    out: torch.Tensor = _mix_atoms(their.weights, p) * (fixed > 0).to(p.dtype)
    return out


def _residual_col(denom: float) -> int:
    if denom not in RESIDUAL_DENOMS:
        raise ValueError(f"denominator {denom!r} is not a gen-3 residual's (one of {RESIDUAL_DENOMS}).")
    return RESIDUAL_DENOMS.index(denom)


def their_residual_frac(op: Any, species_ids: torch.Tensor, denom: float) -> torch.Tensor:
    """The EXPECTED fraction of THEIR mon's max HP its ``1/denom`` end-of-turn residual moves: ``E[max(1, ⌊M /
    denom⌋) / M]`` — Showdown's integer rule (`residual_points`) over the species' max-HP distribution (the op's
    `THEIR_HP_RESIDUAL`). A Shedinja's is 1 (its one HP), exactly. Species ids only (`their_max_hp`'s refusal)."""
    their_max_hp(op, species_ids)                               # the integer-ids refusal, in one place
    out: torch.Tensor = op.THEIR_HP_RESIDUAL[species_ids][..., _residual_col(denom)]
    return out


def their_residual_points(op: Any, species_ids: torch.Tensor, denom: float) -> torch.Tensor:
    """`their_residual_frac` in HP on the scale of `TheirHp.maxhp` (× ``1 / E[1 / M]``) — the amount an HP chain
    in points carries, so that dividing by `maxhp` returns the expected fraction exactly."""
    out: torch.Tensor = their_residual_frac(op, species_ids, denom) / their_max_hp(op, species_ids).inv
    return out


def their_residual_frac_slots(op: Any, ctx: Any, denom: float, sp_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """`their_residual_frac` for THEIR six slots ``[B, 6]``, under `their_hp_slots`' slot rule: the species where
    it is concrete, the expectation over species where the slot is averaged (X5's OTHER over the tail; with
    ``sp_probs``, a hidden slot over it; mass on no species reads as the unowned row 0)."""
    table: torch.Tensor = op.THEIR_HP_RESIDUAL[:, _residual_col(denom)]
    frac = their_residual_frac(op, ctx.species_ids[:, TEAM_SIZE:2 * TEAM_SIZE], denom)

    def _mix(p: torch.Tensor) -> torch.Tensor:
        e = p @ table.to(p.dtype) + (1.0 - p.sum(dim=-1)).clamp(min=0.0) * table[0]
        out: torch.Tensor = (e[:, None] if p.dim() == 2 else e).to(frac.dtype)
        return out
    x5 = op.stash.x5
    if x5 is not None and x5.override is not None:
        frac = torch.where(x5.override, _mix(x5.species_probs), frac)
    if sp_probs is not None:
        frac = torch.where(_their_averaged(op, ctx), _mix(sp_probs), frac)
    return frac


def our_hp_points(hp_frac: torch.Tensor, maxhp: torch.Tensor) -> torch.Tensor:
    """OUR mon's current HP in HP points, INTEGER-EXACT (gen3_our_hp_exact_v1). The obs carries
    ``f32(hp / maxhp)``, and ``f32(hp / maxhp) · maxhp`` misses ``hp`` by an ulp for ~8.5 % of (maxhp <= 714,
    hp) pairs (e.g. 100 / 714 -> 100.0000076): the fixed-damage KO cut ``fixed >= hp`` then read Seismic Toss as
    NOT KO-ing a mon at exactly 100 HP, which Showdown faints (`sim/pokemon.ts` ``damage``: ``hp -= d; if
    (hp <= 0) faint``). The product is a few ulps from the integer (<< 1/2), so ``round`` returns ``hp`` exactly.
    THEIR HP is a percent of a max-HP BELIEF (`their_max_hp`) — not an integer — and never comes through here."""
    return (hp_frac * maxhp).round()


def our_healed_hp(cur_hp: torch.Tensor, frac: torch.Tensor, maxhp: torch.Tensor) -> torch.Tensor:
    """OUR HP after a self-heal of ``frac`` of max HP, integer-exact: ``min(maxhp, hp + trunc(frac · maxhp))`` —
    gen-3 Showdown truncates every heal (`sim/battle-actions.ts` ``moveData.heal``: ``Math.floor`` below gen 5;
    the weather heals' ``this.heal(maxhp * 2 / 3)`` etc. through `sim/battle.ts` ``heal``'s ``trunc``).
    ``cur_hp`` comes from `our_hp_points`; ``frac`` = 0 is the no-heal world, exactly ``cur_hp``."""
    return torch.minimum(maxhp, cur_hp + torch.floor(frac * maxhp + _HEAL_FLOOR_NUDGE))


def _nonformula_fracs(nf_tabs: Tuple[torch.Tensor, ...], tgt_cur_hp: torch.Tensor, tgt_maxhp: torch.Tensor,
                      atk_cur_hp: torch.Tensor, eff: torch.Tensor, eps: float) -> Tuple[torch.Tensor, ...]:
    """`nonformula_rolls`' damage arithmetic, shared by both target sides: ``(is_nf, high, crit, not_immune)``."""
    fixed, target_frac, endeavor = nf_tabs
    is_nf = (fixed + target_frac + endeavor) > 0
    dmg = fixed + target_frac * tgt_cur_hp + endeavor * (tgt_cur_hp - atk_cur_hp).clamp(min=0.0)
    not_immune = (eff > 0).float()
    frac = dmg / (tgt_maxhp + eps)
    high = frac.clamp(max=_DMG_CHIP_CAP) * not_immune
    crit = frac.clamp(max=_DMG_CRIT_CAP) * not_immune
    return is_nf, high, crit, not_immune


def nonformula_rolls(nf_tabs: Tuple[torch.Tensor, ...],
                     tgt_cur_hp: torch.Tensor, tgt_maxhp: torch.Tensor, atk_cur_hp: torch.Tensor,
                     eff: torch.Tensor, acc: torch.Tensor,
                     eps: float = 1e-6) -> Tuple[torch.Tensor, ...]:
    """→ ``(is_nf, high, crit, ko)`` broadcast to a common shape, for a target whose HP is EXACT (OUR mon; THEIR
    target goes through `nonformula_rolls_their`). ``is_nf`` marks the cells whose
    rolls must be REPLACED; ``high`` (== low: no roll) and ``crit`` (no crit) are the damage as a
    fraction of the target's max HP, clamped like the formula rolls; ``ko = acc · [the hit KOs a live
    target]`` (an OHKO always; a fixed amount iff ≥ the remaining HP; Endeavor / Super Fang never). Every
    output is 0 where the target is immune (``eff <= 0``). ``nf_tabs`` = ``(fixed, target_frac,
    endeavor)`` as `gather_nonformula` returns them (each broadcastable to the output)."""
    fixed, target_frac, _endeavor = nf_tabs
    is_nf, high, crit, not_immune = _nonformula_fracs(nf_tabs, tgt_cur_hp, tgt_maxhp, atk_cur_hp, eff, eps)
    # KO: a full-current-HP kind (OHKO, target_frac 1) KOs any live target BY CONSTRUCTION — tested on
    # the table, never as `dmg >= hp` (an exact equality is not a near-tie to be margined); a fixed kind
    # KOs iff it covers the remaining HP (the one genuine threshold — a declared MARGIN site); Endeavor
    # leaves the target at the attacker's HP and Super Fang at half, so neither KOs. APPROXIMATION:
    # Super Fang's 1-HP floor (clampIntRange(hp/2, 1) KOs a 1-HP target) is not modelled.
    alive = (tgt_cur_hp > 0).float()
    full_hp_kind = target_frac >= 1.0                       # (one comparison per line: the
    covers = fixed >= tgt_cur_hp                            #  selection-site recorder reads lines)
    kills = full_hp_kind | covers
    ko = acc * kills.float() * alive * not_immune
    return is_nf, high, crit, ko


def nonformula_rolls_their(nf_tabs: Tuple[torch.Tensor, ...], their: "TheirHp", hp_frac: torch.Tensor,
                           atk_cur_hp: torch.Tensor, eff: torch.Tensor, acc: torch.Tensor,
                           eps: float = 1e-6) -> Tuple[torch.Tensor, ...]:
    """`nonformula_rolls` into THEIR mon (gen3_their_max_hp_belief_v1): ``their`` its max-HP belief, ``hp_frac`` its
    reported HP fraction (both broadcast to the cells). The damage fractions are the EXPECTED fractions of their
    max HP (a fixed 100 HP is ``100 · E[1 / M]``; Super Fang half the reported fraction; Endeavor the fraction left
    above OUR exact HP), and a fixed kind's KO is the PROBABILITY that it covers their remaining HP
    (`their_fixed_ko`) — no hard cut on a percent × an assumed max HP; an OHKO still KOs any live target."""
    fixed, target_frac, _endeavor = nf_tabs
    is_nf, high, crit, not_immune = _nonformula_fracs(nf_tabs, their.frac_points(hp_frac), their.maxhp, atk_cur_hp,
                                                      eff, eps)
    alive = (hp_frac > 0).float()
    full_hp_kind = (target_frac >= 1.0).float()
    kills = torch.maximum(full_hp_kind, their_fixed_ko(their, hp_frac, fixed))
    ko = acc * kills * alive * not_immune
    return is_nf, high, crit, ko


def gather_bp(op: Any, idx: torch.Tensor, attacker_hp_frac: torch.Tensor,
              bp: Optional[torch.Tensor] = None) -> torch.Tensor:
    """`effective_bp` over the op's tables at move nums `idx`. `bp` overrides the gathered MOVE_BP
    (the outgoing kernels' Hidden Power → 70 substitution); `attacker_hp_frac` broadcasts against
    `idx` (same rank, a 1-wide move axis)."""
    base: torch.Tensor = op.MOVE_BP[idx] if bp is None else bp
    return effective_bp(base, op.MOVE_BP_HP_SCALED[idx], op.MOVE_BP_FLAIL[idx], attacker_hp_frac)


def gather_nonformula(op: Any, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(fixed, target_frac, endeavor)`` at move nums `idx` — the `nonformula_rolls` inputs."""
    return op.MOVE_FIXED_DAMAGE[idx], op.MOVE_TARGET_HP_FRAC[idx], op.MOVE_ENDEAVOR[idx]


def is_priced(op: Any, idx: torch.Tensor, bp: torch.Tensor) -> torch.Tensor:
    """0/1: the op has a damage model for the move — a positive formula BP OR a non-formula kind.
    The `unmodelled` rows of `DAMAGE_MODELS` (and every Status move) read 0, by declaration."""
    nonformula: torch.Tensor = op.MOVE_NONFORMULA[idx]
    return ((bp > 0) | (nonformula > 0)).float()


def override_rolls(rolls: Tuple[torch.Tensor, ...], nf: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
    """Replace the formula ``(high, low, crit, ko)`` with `nonformula_rolls`'s ``(is_nf, high, crit, ko)``
    at the non-formula cells (low == high: no roll)."""
    is_nf, nf_high, nf_crit, nf_ko = nf
    high, low, crit, ko = rolls
    return (torch.where(is_nf, nf_high, high), torch.where(is_nf, nf_high, low),
            torch.where(is_nf, nf_crit, crit), torch.where(is_nf, nf_ko, ko))


# --------------------------------------------------------------------------- gen3_beatup_exact_v1
def gather_beatup(op: Any, idx: torch.Tensor) -> torch.Tensor:
    """0/1 ``MOVE_BEATUP`` at move nums `idx` — the cells whose formula inputs `beatup_swap` replaces."""
    flag: torch.Tensor = op.MOVE_BEATUP[idx]
    return flag


def typeless_move_type(op: Any, move_ids: torch.Tensor, move_ty: torch.Tensor) -> torch.Tensor:
    """OUR move's TypeEncoder index as the battle resolves it: '???' for Beat Up, else `move_ty` (the obs'
    resolved type — which is the DEX type, Dark, for Beat Up). The incoming kernels get the same fact from
    the `MOVE_TYPE_IDX` row `build_damage_buffers` writes; the outgoing kernels read the obs type, so they
    route it through here. → STAB, the chart, the ability multipliers and the weather / sport modifiers all
    read neutral."""
    return torch.where(op.MOVE_BEATUP[move_ids] > 0, torch.full_like(move_ty, TYPELESS_TYPE_IDX), move_ty)


def beatup_base_def(op: Any, species_ids: torch.Tensor) -> torch.Tensor:
    """The TARGET's species BASE Def (what Beat Up's ``onFoeModifySpD`` returns). ``clamp(min=1)`` keeps the
    sentinel species 0 (an empty / unrevealed slot — always gated downstream) off a 1/0; every real species
    has base Def ≥ 5, so the clamp never moves a real value."""
    d: torch.Tensor = op.BASE_STATS[species_ids][..., _BS_DEF_COL]
    return d.clamp(min=1.0)


def beatup_opp_target_def(op: Any, ctx: Any, sp_probs: torch.Tensor) -> torch.Tensor:
    """``[B,6]`` the base Def Beat Up meets at each of the six OPP slots. A revealed slot: that species'
    base Def. An UNREVEALED slot: ``1 / E[1 / D_base]`` under `sp_probs` (``unrevealed_species_probs`` —
    ``[B,S]`` prior or ``[B,6,S]`` override), the harmonic mean, so the kernel's ``… / D`` returns
    E[damage] exactly. The expectation runs over the SAME marginal the other expected-latent reads use."""
    revealed = beatup_base_def(op, ctx.species_ids[:, TEAM_SIZE:2 * TEAM_SIZE])              # [B,6]
    inv_all = 1.0 / op.BASE_STATS[:, _BS_DEF_COL].clamp(min=1.0)                             # [S]
    harmonic = 1.0 / (sp_probs @ inv_all).clamp(min=1e-12)                                   # [B] | [B,6]
    harmonic = harmonic[:, None] if harmonic.dim() == 1 else harmonic                        # [B,1] | [B,6]
    return torch.where(ctx.opp_believed_mask, harmonic, revealed)


def _healthy(ctx: Any, sl: slice) -> torch.Tensor:
    """``[B,6]`` 0/1: alive (HP fraction > 0) AND no major status — Beat Up's ally filter
    ``!ally.fainted && !ally.status``. (An unrevealed opp slot encodes HP 0: the caller handles it.)"""
    alive = (ctx.hp_and_active[:, sl, 0] > 0).float()
    status = ctx.pokemon_part[:, sl, _STATUS_COLS].sum(dim=-1)
    ok: torch.Tensor = alive * (status < 0.5).float()
    return ok


def beatup_party_ours(op: Any, ctx: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(S, N)`` ``[B]`` each for OUR party as Beat Up's attacker: ``S`` = Σ base Atk over our healthy
    mons, ``N`` = how many (the user included iff healthy — it is one of the six). Fully known."""
    ours = slice(0, TEAM_SIZE)
    ok = _healthy(ctx, ours)                                                      # [B,6]
    atk = op.BASE_STATS[ctx.species_ids[:, ours]][..., _BS_ATK_COL]               # [B,6]
    return (ok * atk).sum(dim=-1), ok.sum(dim=-1)


def beatup_party_opp(op: Any, ctx: Any, species_probs: Optional[torch.Tensor] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(S, N)`` ``[B]`` each for the OPPONENT's party as Beat Up's attacker. A REVEALED mon counts iff alive
    with no status; every UNREVEALED slot (`ctx.opp_believed_mask`) counts with certainty — a mon that never
    entered the battle cannot have fainted or been statused — at its EXPECTED base Atk under the op's one
    hidden-mon belief: `unrevealed_species_probs(ctx, species_probs)` — the Species-Clause usage prior, or the
    T0 species belief the extractor hands every pricing site (``[B,S]`` team-level, or a per-slot
    ``[B,6,S]``). The sum's expectation is exact (linear in the per-slot marginals).

    X5 fixed_mass (U3 part 3, `op.stash.x5`): the hidden slots' marginal is the hidden-TEAM belief
    ``π / k`` over EVERY candidate species (hypotheses AND the tail; `OpRoster.team_probs`), never the
    per-slot hypothesis one-hots the defender kernels read — so the hidden part is ``Σ_s π_s ·
    baseAtk(s)``, Beat Up's exact expectation under the fixed-mass belief (class E, linear)."""
    opp = slice(TEAM_SIZE, 2 * TEAM_SIZE)
    hidden = ctx.opp_believed_mask.float()                                        # [B,6]
    ok = _healthy(ctx, opp) * (1.0 - hidden)                                      # [B,6] revealed + eligible
    atk = op.BASE_STATS[ctx.species_ids[:, opp]][..., _BS_ATK_COL]                # [B,6]
    x5 = getattr(getattr(op, "stash", None), "x5", None)
    if x5 is not None:
        species_probs = x5.team_probs.to(op.BASE_STATS.dtype)                     # [B,S] π / k
    e_atk = op.unrevealed_species_probs(ctx, species_probs) @ op.BASE_STATS[:, _BS_ATK_COL]   # [B] | [B,6]
    hidden_atk = hidden.sum(dim=-1) * e_atk if e_atk.dim() == 1 else (hidden * e_atk).sum(dim=-1)   # [B]
    return (ok * atk).sum(dim=-1) + hidden_atk, ok.sum(dim=-1) + hidden.sum(dim=-1)


def beatup_swap(is_bu: torch.Tensor, A: torch.Tensor, D: torch.Tensor, atk_sum: torch.Tensor,
                defence: torch.Tensor, hits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The kernels' ``core = 42·bp·A/(D+eps)/50 + plus2`` inputs, with Beat Up's cells swapped: ``A`` → the
    party's Σ base Atk, ``D`` → the target's base Def, ``plus2`` → ``2·hits`` (the formula's ``+2`` is per
    hit). Every other cell keeps ``(A, D, 2.0)`` EXACTLY (a `where`, no arithmetic), so a table with no Beat
    Up in it is bit-identical to the kernel before. All args broadcast to `A` / `D`'s shape."""
    bu = is_bu > 0
    return (torch.where(bu, atk_sum, A), torch.where(bu, defence, D), torch.where(bu, 2.0 * hits, 2.0))
