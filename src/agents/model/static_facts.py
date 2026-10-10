"""Two NARROW per-entity facts for `--token-encoding static` (`gen3_static_port_v1`,
`designs/endstate/design_static_tokens.md` §12), from the static diagnostic
(`designs/research_state/measurements/static_diag_2026-10-09/`, H2 / H3): under `static` a board fact reaches a mon
token only through attention, and mostly at the trunk's LAST layer; two dynamic facts read worse than under legacy
(our side's Spikes on our mon tokens, R² 0.33 vs 0.46 and the one gap that GREW with training; our active's HP on its
move tokens, R² 0.59 vs 0.80). Each fact is a flag of its own, OFF by default:

* **`--mon-hazard-cost on`** — every mon's token (BOTH sides) gets, as D content (`mon_hazard_features`, [B,12,2]):
  its OWN side's Spikes layers (/3) and the fraction of max HP it would lose switching in. The fraction is the damage
  operator's ONE Spikes entry rule (`DamageOperator.spikes_entry`, which the `x` edge cell reads too): 1/8, 1/6, 1/4
  for 1–3 layers, 0 for a Flying type or Levitate (both read from the SPECIES: base types, and the species' Smogon
  P(Levitate), exactly 0 or 1 in gen 3 — never the current type / ability columns, which Color Change, Transform, Trace,
  Role Play and Skill Swap change while a switch-in reverts them). Under X5 a hidden slot is priced as its HYPOTHESIS (the op's context).
  An opponent slot with no species (the belief-off ablation only: X5 always holds a hypothesis) reads fraction 0 —
  its Spikes column still says the layers are there.
* **`--move-actor-state on`** — our active's 4 E3 move seats get its CURRENT HP fraction and its status one-hot
  (`move_actor_features`, [B, 1 + CONDITION_DIM]): the actor's state on the tokens of the moves it would use,
  which legacy's move network mixed in and static's move tokens lack. Only the E3 seats: the opponent's E4 threat
  seats never carried their active's state under either encoding, so the diagnostic's logic (a fact static
  REMOVED) does not apply to them.

Both arrive through a ZERO-INIT, bias-free `IsolatedLinear` built LAST (no global RNG draw, skipped by SB3's
orthogonal re-init), so every other parameter's initial bytes equal the flag-off build's and an ON build at init
adds exactly 0. Why ADDED to the token rather than fed into D's MLP: a hidden opponent slot's static token is a
GATHER from the dex table (a pure function of the species, `static_tokens.static_hypothesis_tokens`); a board-
dependent column inside D's MLP would end that. As token content the fact is in place at the trunk's FIRST layer,
which is what the diagnostic's H3 (board facts arrive only at the last layer) asks for.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from agents.observation.constants import CONDITION_DIM, POKEMON_CONDITION_OFFSET, POKEMON_HP_OFFSET, TEAM_SIZE

if TYPE_CHECKING:
    from agents.model.extractor_ctx import ExtractorContext

#: The legal values of the two flags. ``off`` builds nothing.
MON_HAZARD_COST_MODES = ("off", "on")
MOVE_ACTOR_STATE_MODES = ("off", "on")

#: [its side's Spikes layers / 3, the fraction of max HP lost switching in].
MON_HAZARD_DIM = 2
#: [current HP fraction] ⊕ the status one-hot [None, BRN, PAR, SLP, FRZ, PSN, TOX].
MOVE_ACTOR_DIM = 1 + CONDITION_DIM

#: `--effective-stats` / `--move-target-state` (`gen3_probe_facts_v1`, config v153): the legal values (``off`` builds nothing).
EFFECTIVE_STATS_MODES = ("off", "on")
MOVE_TARGET_STATE_MODES = ("off", "on")
#: The effective-stat columns of an ACTIVE mon: its Atk / Def / SpA / SpD / Spe after the stat stages (Spe also after
#: paralysis), ÷ `STATIC_STAT_SCALE`, then its accuracy and evasion stage multipliers.
EFFECTIVE_STATS_FACTS = ("atk", "def", "spa", "spd", "spe", "accuracy_mult", "evasion_mult")
EFFECTIVE_STATS_DIM = len(EFFECTIVE_STATS_FACTS)
#: [their active's current HP fraction] ⊕ its status one-hot — the TARGET of our 4 E3 seats.
MOVE_TARGET_DIM = 1 + CONDITION_DIM

#: `--switch-hazard-cost` (`gen3_static_recovery_v1`): the legal values (``off`` builds nothing) and the width of the
#: switch pointer cell's new trailing block, [our side's Spikes layers / 3, the switch-in HP fraction].
SWITCH_HAZARD_COST_MODES = ("off", "on")
SWITCH_HAZARD_DIM = MON_HAZARD_DIM


def mon_hazard_features(op: Any, ctx: 'ExtractorContext', opp_concrete: Optional[torch.Tensor] = None) -> torch.Tensor:
    """[B, 12, MON_HAZARD_DIM]: per mon, its OWN side's Spikes layers (/3: ours for slots 0–5, theirs for 6–11) and
    the HP fraction it would lose switching in (``op.spikes_entry``, the op's one rule). ``ctx`` is the context the
    op prices with (under X5 the hypothesis context). ``opp_concrete`` [B,6] (1 = the slot's species is known or
    hypothesised): None ⇒ revealed = not believed. A non-concrete opponent slot's fraction is 0 (unknown types)."""
    chip_our, chip_opp, _gr_i, _gr_j = op.spikes_entry(ctx)                              # [B,6] ×2
    if opp_concrete is None:
        opp_concrete = (~ctx.opp_believed_mask).to(chip_opp.dtype)
    sp = ctx.spikes_feature.to(chip_our.dtype)                                          # [B,2] /3
    ours = torch.stack([sp[:, 0:1].expand(-1, TEAM_SIZE), chip_our], dim=-1)            # [B,6,2]
    theirs = torch.stack([sp[:, 1:2].expand(-1, TEAM_SIZE), chip_opp * opp_concrete.to(chip_opp.dtype)], dim=-1)
    return torch.cat([ours, theirs], dim=1)


def switch_hazard_features(op: Any, ctx: 'ExtractorContext') -> torch.Tensor:
    """[B, 6, SWITCH_HAZARD_DIM] (`--switch-hazard-cost on`, `gen3_static_recovery_v1`): per switch target j (OUR team
    slot j), ``[our side's Spikes layers / 3, the HP fraction j loses switching in]`` — EXACTLY our half of
    `mon_hazard_features` (the op's ONE entry rule, ``op.spikes_entry``; our mons are always known), so the switch cell,
    the per-mon fact and the `x` edge cell can never disagree. Appended LAST to the switch pointer cell. Not alive-gated:
    a fainted or active target's switch logit is masked, and the rule is the mon's own (no other mon's state enters)."""
    return mon_hazard_features(op, ctx)[:, :TEAM_SIZE]


def move_actor_features(ctx: 'ExtractorContext') -> torch.Tensor:
    """[B, MOVE_ACTOR_DIM]: our ACTIVE mon's current HP fraction and status one-hot — the actor of the 4 E3 seats."""
    ar = torch.arange(ctx.batch_size, device=ctx.device)
    row = ctx.pokemon_part[ar, ctx.our_active_idx]                                       # [B, POKEMON_FULL_DIM]
    return torch.cat([row[:, POKEMON_HP_OFFSET:POKEMON_HP_OFFSET + 1],
                      row[:, POKEMON_CONDITION_OFFSET:POKEMON_CONDITION_OFFSET + CONDITION_DIM]], dim=-1)


# ===================================================================== gen3_probe_facts_v1 (config v153)
def _stat_stage_mult(stage: torch.Tensor) -> torch.Tensor:
    """Gen-3 Atk / Def / SpA / SpD / Spe stage multiplier (`sim/pokemon.ts` ``boostTable = [1, 1.5, 2, 2.5, 3, 3.5,
    4]``: +s → (2 + s) / 2, −s → 2 / (2 + s)) — the op's own `_boost_mult` arithmetic."""
    s = stage.clamp(-6.0, 6.0)
    return torch.where(s >= 0, (2.0 + s) / 2.0, 2.0 / (2.0 - s))


def _acc_stage_mult(stage: torch.Tensor) -> torch.Tensor:
    """Gen-3 accuracy / evasion stage multiplier (`data/mods/gen3/scripts.ts` ``boostTable = [1, 4/3, 5/3, 2, 7/3, 8/3,
    3]``: +s → (3 + s) / 3, −s → 3 / (3 + s))."""
    s = stage.clamp(-6.0, 6.0)
    return torch.where(s >= 0, (3.0 + s) / 3.0, 3.0 / (3.0 - s))


def effective_stat_features(op: Any, ctx: 'ExtractorContext', spread_belief: Optional[torch.Tensor] = None
                            ) -> torch.Tensor:
    """[B, 12, EFFECTIVE_STATS_DIM] (`--effective-stats on`, `gen3_probe_facts_v1`): each side's ACTIVE mon's
    STAGE-APPLIED stats — Atk / Def / SpA / SpD / Spe × the gen-3 stage multiplier (Spe × 1/4 more when paralysed,
    `data/mods/gen4/conditions.ts` par ``onModifySpe``), ÷ `STATIC_STAT_SCALE` — and its accuracy / evasion stage
    multipliers; a benched mon reads 0 (a switch resets the stages). OURS from the observed spread, exact (the op's
    level-100 formula with the nature); THEIRS the model's believed stats (``spread_belief``, the expectation the op
    prices with; the op's neutral 2 · base + 36 where the belief is off). The probe battery (`measurements/
    probe_battery_2026-10-09/`): a stage's SIZE reads at R² ≤ 0.10 while "is boosted" reads at AUC 0.97 — the role
    network squashes the stage columns; this hands the magnitude over in the stat's own unit."""
    from agents.model.arch_constants import STATIC_STAT_SCALE
    from agents.model.damage_op_layout import _SB_ATK, _SB_DEF, _SB_SPA, _SB_SPD, _SB_SPE
    from agents.observation.constants import BOOSTS_DIM, POKEMON_SPREAD_DIM, POKEMON_SPREAD_OFFSET
    B = ctx.batch_size
    ar = torch.arange(B, device=ctx.device)
    pp = ctx.pokemon_part
    dt = pp.dtype

    def _row(ctx_raw: torch.Tensor, base_stats: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        st = [(ctx_raw[:, 2 * i] - ctx_raw[:, 2 * i + 1]) * 6.0 for i in range(7)]      # atk def spa spd spe acc eva
        mult = torch.stack([_stat_stage_mult(x) for x in st[:5]], dim=-1)               # [B,5]
        par = cond[:, 2]                                                                 # [None, BRN, PAR, …]
        para = torch.ones_like(mult)
        para[:, 4] = 1.0 - 0.75 * par
        stats = base_stats * mult * para / STATIC_STAT_SCALE
        return torch.cat([stats, _acc_stage_mult(st[5])[:, None], _acc_stage_mult(st[6])[:, None]], dim=-1)

    assert ctx.our_ctx_raw.shape[-1] >= BOOSTS_DIM
    # ours: the observed spread, exact (the op's `_outgoing_attacker_matrix` formula)
    oa = ctx.our_active_idx
    a_base = op.BASE_STATS[ctx.species_ids[ar, oa]].to(dt)                               # [B,6] hp atk def spa spd spe
    spr = pp[ar, oa, POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
    iv, ev, nat = spr[:, 0:6] * 31.0, spr[:, 6:12] * 252.0, spr[:, 13:18]
    ours = (2.0 * a_base[:, 1:6] + iv[:, 1:6] + ev[:, 1:6] / 4.0 + 5.0) * nat              # [B,5]
    ours_row = _row(ctx.our_ctx_raw, ours, pp[ar, oa, POKEMON_CONDITION_OFFSET:POKEMON_CONDITION_OFFSET + CONDITION_DIM])
    # theirs: the believed stats (the op's own spread belief; neutral where it is off)
    ta = ctx.opp_active_local
    t_base = op.BASE_STATS[ctx.species_ids[ar, TEAM_SIZE + ta]].to(dt)
    if spread_belief is not None:
        sb = spread_belief[ar, ta].to(dt)
        theirs = torch.stack([sb[:, _SB_ATK], sb[:, _SB_DEF], sb[:, _SB_SPA], sb[:, _SB_SPD], sb[:, _SB_SPE]], -1)
    else:
        theirs = 2.0 * t_base[:, 1:6] + 31.0 + 5.0
    theirs_row = _row(ctx.opp_ctx_raw, theirs,
                      pp[ar, TEAM_SIZE + ta, POKEMON_CONDITION_OFFSET:POKEMON_CONDITION_OFFSET + CONDITION_DIM])
    out = torch.zeros(B, 2 * TEAM_SIZE, EFFECTIVE_STATS_DIM, dtype=dt, device=ctx.device)
    out[ar, oa] = ours_row
    out[ar, TEAM_SIZE + ta] = theirs_row
    alive = (ctx.hp_and_active[:, :, 0] > 0).to(dt)
    has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).to(dt)              # [B]
    side = torch.cat([torch.ones(B, TEAM_SIZE, dtype=dt, device=ctx.device),
                      has_opp[:, None].expand(B, TEAM_SIZE)], dim=1)
    return out * (alive * side)[..., None]


def move_target_features(ctx: 'ExtractorContext') -> torch.Tensor:
    """[B, MOVE_TARGET_DIM] (`--move-target-state on`, `gen3_probe_facts_v1`): THEIR active mon's current HP fraction
    and status one-hot — the TARGET of our 4 E3 move seats, `move_actor_features`' sibling (the probe battery: their
    HP reads at R² 0.85 at its own token but 0.26–0.34 at our decision tokens)."""
    ar = torch.arange(ctx.batch_size, device=ctx.device)
    row = ctx.pokemon_part[ar, TEAM_SIZE + ctx.opp_active_local]
    has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).to(row.dtype)       # [B] (no target: 0)
    return torch.cat([row[:, POKEMON_HP_OFFSET:POKEMON_HP_OFFSET + 1],
                      row[:, POKEMON_CONDITION_OFFSET:POKEMON_CONDITION_OFFSET + CONDITION_DIM]], dim=-1) * has_opp[:, None]
