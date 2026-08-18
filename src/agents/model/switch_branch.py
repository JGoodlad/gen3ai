"""gen3_switch_branch_v1 — the SWITCH-BRANCH move cell (OA2), the Rapid-Spin spinblock, and
Protect's α-conditioning.

`design_conditional_opponent_cells.md` §2 (OA2 — *punish the switch*), plus two owner-specified
mechanics that are the SAME contraction and therefore belong in the same vector. Phase B of the
conditional-mechanics substrate; Phase A (`pair_outcome.py`) unified the currency of what the
opponent does to US, and this module is the other direction — what OUR move is worth given what
they do.

## The one shape all three share

    Σ over THEIR options of  (usage probability) × (a property of the option)

Phase A contracts over their believed MOVE seats with α. Everything here contracts over the
SWITCH branch — either the whole branch as one scalar (`α_SWITCH`) or, where our move's value
depends on *which* mon arrives, over their team slots with **β**. β exists only under
`--opp-intent`, so this flag REQUIRES the intent head rather than inventing a fallback: there is
no "presence belief over arrivals" the way `belief_mean` is a presence belief over moves, and
substituting a uniform would be a claim about their switching we have no source for.

## 1. OA2 — the switch-branch move cell (§2)

Gen-3 is **simultaneous-move**: they commit without seeing our move, so `P(they switch)` is ONE
scalar for the turn, never per-move (§2.1). The *consequence* is per-move, because switches resolve
first — our move lands on the ARRIVAL, not on the mon we are looking at. So a Focus Punch and a
Spikes have the same `α_SWITCH` and completely different worlds behind it.

The design's §2.3 rule is followed literally: **keep the branches DECORRELATED.** The STAY branch
already rides the op's move cell (`[low, high, crit, pko]` vs their active). This module adds the
SWITCH branch raw (`e_high_switch` / `e_pko_switch` / `e_mult_switch`) plus the shared `a_switch`
scalar, and **never** the collapsed `(1−p)·stay + p·switch`. The head then learns its own effective
weighting, and the *difference* between branches — the actual strategic quantity ("this is only
good if they stay") — is one subtraction away in a linear layer.

The one derived coordinate is `wasted_ko = pko_stay(k) · α_SWITCH`, which §2.3 names explicitly:
*"don't click the KO into the obvious switch."* It is a product of two delivered numbers, so it
fails the naive derivability rule and passes the counter-rule: a thin `tanh` MLP over a shared cell
does not multiply two of its own inputs well, and this is precisely the interaction the design
built OA2 to express.

**§9a — two actions whose ordering it flips.** *click Earthquake vs click Spikes* against a
Skarmory at 30% that will obviously pivot: `pko_stay(Earthquake)` is high, so damage-vs-active
ranks it first forever, while `e_high_switch(Earthquake)` is ~0 against the Gengar/Zapdos arrival
β names (both Ground-immune) and Spikes lands whatever arrives. Second: *click Rock Slide vs click
Hidden Power Grass* with a Salamence active and β pointing at Swampert — Rock Slide wins the stay
branch, HP Grass is 4× on the arrival.

**The §4.1 HARD PREREQUISITE is CLOSED, and this module depends on that.** OA2 was blocked because
v34's outgoing matrix was REVEALED-gated, so a β that (correctly) puts mass on unrevealed slots
would read ≈0 there — *"my move always lands on their active"*, misleading exactly when switching
matters most, the typeless-HP "immune" GIGO class. `gen3_unrevealed_outgoing_prior_v1` fixed it:
an unrevealed slot is priced against the EXPECTED-LATENT defender (E[type mult] / E[bulk] /
E[maxhp] marginalised through the species posterior, forced-alive at full HP). One honest residue
remains and is NOT hidden: **`pko` is still NULLED at unrevealed slots** by the op's owner rule (a
full-HP switch-in is ~never OHKO'd), so `e_pko_switch` is systematically deflated in proportion to
β's hidden mass. `e_high_switch` carries the magnitude there, which is why both ship.

## 2. Rapid Spin — the spinblock (owner spec, first-class)

> *"Rapid Spin fails if they switch a Ghost in — the REVERSE of Pursuit trapping; they go hand in
> hand."*

    p_spin_blocked = is_ghost(their active) · a_stay  +  α_SWITCH · Σ_slots β_slot · P(slot is Ghost)

**The Pursuit mirror, explicitly.** v85's Pursuit cell is `is_pursuit · α_SWITCH` (+ the bonus
damage): a contraction of `α_SWITCH` against a property of the DEPARTING mon, with **positive**
valence — their switch is what makes our move good. This is `α_SWITCH` contracted through β against
a property of the ARRIVING mon, with **negative** valence — their switch is what makes our move
worthless. Pursuit needs no β because the sim strikes the departing mon before the switch resolves
(verified against `src/rust_sim/state.rs`); Rapid Spin needs β because it resolves *after*, onto
whoever arrived. Same operator, opposite sign, different resolution order — and that difference is
a fact about gen-3, not a modelling choice.

In gen 3 Rapid Spin is a **Normal** move, so a Ghost final defender means **no damage AND no hazard
removal**: the entire reason for clicking it dies. Both halves die together, which is why one
probability suffices.

`P(slot is Ghost)` is **leak-free**: revealed slots use their revealed types; unrevealed slots use
the hidden-team species posterior (the Species-Clause-filtered Smogon usage prior, or the model's
own `species_probs` when `--t0-species-prior` is on) marginalised through the species→type table.
Publications only — the same source every other unrevealed read uses.

**§9a:** *click Rapid Spin vs click Hydro Pump* on a Starmie with a Gengar on their bench and
`α_SWITCH` high — the spin's whole value is `1 − p_spin_blocked`, and nothing else in the move cell
knows the arrival's type. Second: *click Rapid Spin vs switch out* when the spin is our only hazard
removal and `p_spin_blocked` ≈ 0.9.

`spin_value_lost = p_spin_blocked · our_side_hazards` is the currency half, in the spirit of §2.1:
a probability and a stake must meet in one vector or they cannot be traded. With no hazards on our
side, a blocked spin costs nothing and the coordinate correctly reads 0.

## 3. Protect — the α-conditioning `c4` never had (`design_conditional_execution.md` §3.3)

Protect's cell today carries `p_success`, the mechanical consecutive-use decay, and never asks the
other question. A Protect that will certainly succeed against an opponent who will certainly set up
is a wasted turn; a Protect that will probably fail against a Choice-Band-locked attacker may still
be right. So:

    attack_mass = Σ_k α_k · is_damaging(k)      # = 1 − α_SWITCH − α(status/setup), less masked mass

and `protect_blocked_mass = attack_mass · p_success` — the conjunction, because a thin shared
scorer cannot form it from the two factors. `is_damaging` comes from the DATA facade (the
`_status_table` convention), not from `high > 0`: an immune damaging move reads 0 damage but is
still an attack, and conflating the two is exactly the confusion `intent_conditional` avoided when
it typed its own status predicate.

**This is decorrelated from, not redundant with, v85's `e_dmg_avoided`** (`Σ_k α_k · high_k`). That
is a MAGNITUDE — how much damage a successful Protect avoids; this is a MASS — whether there is an
attack at all. They come apart in both directions: a believed Spore has mass and no magnitude, and
a 4× resisted Hidden Power has magnitude near zero at full mass. `c4`'s edge family is untouched
(its deletion is a separately licensed decision).

**§9a:** *click Protect vs click Recover* at 60% HP against an opponent believed to be setting up —
the decay says Protect will work, `attack_mass ≈ 0.2` says there is nothing to block, and Recover
strictly dominates. Second: *click Protect vs click Swords Dance* against a Choice-Band-locked
attacker, where `attack_mass ≈ 1.0` and Protect banks a Leftovers/Toxic tick for free.

## Contract compliance (the four α-consumer rules + Phase A's two)

α and β are read from the PUBLICATIONS and **stop-grad unconditionally** (`pair_alpha_full` /
`.detach()` here), because a policy-side consumer whose PPO→head route exists only when
`--belief-grad-mode` says so is one training-flag change from silently reopening. Seat widths are
checked against the op's top-K and fail loud (the `op move-order` bug class). The projection is
zero-init and captured by `restore_identity_init` (M1). Seat-permutation invariant: every
seat-indexed computation is `Σ_k α_k · f_k`. Their-team-permutation invariant: every slot-indexed
computation is `Σ_j β_j · f_j` (§5 gate 5's "the contracted OA2 column is invariant to their bench
permutation").
"""
from __future__ import annotations

from typing import Tuple, cast

import torch

from agents.gen3_data.moves import MoveData

from agents.model.arch_constants import _SWITCH_BRANCH_RAW
from agents.model.damage_op_layout import (_DMG_OMX_IDX_HIGH, _DMG_OMX_IDX_MULT, _DMG_OMX_IDX_PKO)
from agents.model.pair_outcome import pair_alpha_full, rapid_spin_num

#: The raw coordinate names, in order — the contract the consumers and the tests read. Never
#: re-spell an index; `switch_branch_test` asserts this tuple against `_SWITCH_BRANCH_RAW`.
SWITCH_BRANCH_COORDS: Tuple[str, ...] = (
    # --- OA2 (design_conditional_opponent_cells.md §2.3), branches kept DECORRELATED ---
    "e_high_switch", "e_pko_switch", "e_mult_switch", "wasted_ko", "a_switch",
    # --- the Rapid Spin spinblock (the Pursuit mirror) ---
    "p_spin_blocked", "spin_value_lost",
    # --- Protect's α-conditioning (the c4 successor) ---
    "protect_attack_mass", "protect_blocked_mass",
)
SWITCH_BRANCH_IDX = {name: i for i, name in enumerate(SWITCH_BRANCH_COORDS)}

assert len(SWITCH_BRANCH_COORDS) == _SWITCH_BRANCH_RAW, \
    "SWITCH_BRANCH_COORDS and _SWITCH_BRANCH_RAW disagree — one was edited without the other."

#: Protect / Detect. Endure is deliberately EXCLUDED, matching v85's `_PROTECT_ONLY`: Endure does
#: not block an attack, it survives one, so "will they attack" is not the question its value asks
#: (that is the v84 `p_KO` branch).
_PROTECT_ONLY = (182, 197)


def _damaging_table() -> torch.Tensor:
    """`[n_move_nums]` 1.0 where the move num is a DAMAGING move, from the data facade.

    The complement of `intent_conditional._status_table`, and sourced the same way for the same
    reason: `high > 0` would read 0 for a damaging move the defender is immune to, so an
    immunity would masquerade as a status move and deflate `attack_mass` exactly when the opponent
    is most likely to be attacking into it."""
    from agents import gen3_data
    n = max(cast(MoveData, gen3_data.moves.get(mid)).num for mid in gen3_data.moves.raw()) + 1
    t = torch.zeros(n, dtype=torch.float32)
    for mid in gen3_data.moves.raw():
        md = gen3_data.moves.get(mid)
        if md is not None and md.is_damaging:
            t[md.num] = 1.0
    return t


class SwitchBranchMoveCell(torch.nn.Module):
    """`(published α/β, the op's outgoing grid + ghost marginal, request ids) → the extra pointer
    MOVE-cell block [B, 4, out_dim]`.

    Zero-init projection ⇒ ON-at-init contributes exactly zero to every action logit, so any
    measured effect on a trained run is something the run LEARNED (ledger M1;
    `restore_identity_init` captures the projection by OBSERVATION on the real `MaskablePPO`
    build)."""

    n_moves: int = 4

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = int(out_dim)
        self.proj = torch.nn.Linear(_SWITCH_BRANCH_RAW, self.out_dim)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)
        self.register_buffer("spin_num", torch.tensor([rapid_spin_num()], dtype=torch.long),
                             persistent=False)                                        # [1]
        self.register_buffer("protect_nums",
                             torch.tensor(_PROTECT_ONLY, dtype=torch.long),
                             persistent=False)                                        # [2]
        self.register_buffer("damaging_num", _damaging_table(), persistent=False)     # [n_nums]

    def forward(self, alpha_logits: torch.Tensor, beta_logits: torch.Tensor,
                seat_live: torch.Tensor, topk_nums: torch.Tensor,
                omx_cells: torch.Tensor, opp_p_ghost: torch.Tensor,
                opp_active_local: torch.Tensor, req_move_ids: torch.Tensor,
                protect_odds: torch.Tensor, our_side_hazards: torch.Tensor) -> torch.Tensor:
        """`alpha_logits` `[B,K+1]` (last class = SWITCH) · `beta_logits` `[B,6]` (the PUBLISHED β,
        `-inf`-masked to legal switch-ins) · `seat_live` `[B,K]` (the meaningful-K gate) ·
        `topk_nums` `[B,K]` (the seats' move NUMS) · `omx_cells` `[B,4,6,5]` (the op's outgoing
        matrix as `[low, high, crit, pko, type_mult]` per (our request move, their mon)) ·
        `opp_p_ghost` `[B,6]` (P(their slot j is Ghost) — revealed types where revealed, the
        species posterior where not) · `opp_active_local` `[B]` · `req_move_ids` `[B,4]` ·
        `protect_odds` `[B,1]` (our active's obs floored-doubling success odds) ·
        `our_side_hazards` `[B,1]` (the Spikes fraction on OUR side, which our spin would remove)
        → `[B,4,out]`.

        Fails loud on a seat-axis width mismatch — α's seats and the op's top-K are the SAME axis,
        and a silent broadcast would pair each α weight with the wrong opponent move while every
        shape check still passed (the named `op move-order` bug class)."""
        k = alpha_logits.shape[-1] - 1
        if topk_nums.shape[-1] != k or seat_live.shape[-1] != k:
            raise ValueError(
                f"alpha has {k} move seats but the op stashed {topk_nums.shape[-1]} candidate nums "
                f"/ {seat_live.shape[-1]} liveness flags. These must be the SAME axis "
                "(entity_topk_seats == damage_topk_k); a mismatch would pair each alpha weight "
                "with the wrong opponent move while every shape check still passed.")
        if omx_cells.shape[1] != self.n_moves:
            raise ValueError(
                f"the outgoing grid carries {omx_cells.shape[1]} attacker moves but the request "
                f"slot axis is {self.n_moves} — the cell is per-ACTION and a mismatch would score "
                "each move from another move's row.")
        dt = omx_cells.dtype
        B = omx_cells.shape[0]
        ar = torch.arange(B, device=omx_cells.device)
        # --- α: the three-way split, stop-grad, masked seats' mass NOT reassigned ---
        alpha, a_switch, a_stay = pair_alpha_full(alpha_logits, seat_live)
        alpha, a_switch, a_stay = alpha.to(dt), a_switch.to(dt), a_stay.to(dt)
        # --- β: the arrival distribution. `has_cand` is the honest zero for a board with no legal
        # switch-in (trapped, or five fainted): softmax over an all-`-inf` row is uniform garbage,
        # and a uniform arrival belief is a claim, not an absence. Stop-grad for the same reason α
        # is — this is a POLICY-side consumer of a supervised head.
        has_cand = torch.isfinite(beta_logits).any(-1, keepdim=True).to(dt)                # [B,1]
        beta = torch.softmax(beta_logits.detach().float().clamp(min=-1e9), dim=-1
                             ).to(dt) * has_cand                                           # [B,6]
        # --- OA2: the SWITCH branch of our own moves, contracted over the arrival ---
        pko_all = omx_cells[..., _DMG_OMX_IDX_PKO]                                         # [B,4,6]
        e_high = (omx_cells[..., _DMG_OMX_IDX_HIGH] * beta[:, None, :]).sum(-1)            # [B,4]
        e_pko = (pko_all * beta[:, None, :]).sum(-1)                                       # [B,4]
        e_mult = (omx_cells[..., _DMG_OMX_IDX_MULT] * beta[:, None, :]).sum(-1)            # [B,4]
        # The STAY branch's pko is already on the op's move cell; it appears here only inside the
        # §2.3 interaction, never as a duplicate column. (Same gather as v85's `pko_active`.)
        pko_stay = pko_all.gather(
            2, opp_active_local[:, None, None].expand(-1, self.n_moves, 1))[..., 0]        # [B,4]
        wasted_ko = pko_stay * a_switch                                                    # [B,4]
        # --- the spinblock: α_SWITCH × β × slot-property, the Pursuit mirror with opposite valence
        p_ghost_arrival = (opp_p_ghost.to(dt) * beta).sum(-1, keepdim=True)                # [B,1]
        ghost_active = opp_p_ghost.to(dt)[ar, opp_active_local][:, None]                   # [B,1]
        p_blocked = ghost_active * a_stay + a_switch * p_ghost_arrival                     # [B,1]
        is_spin = (req_move_ids[..., None] == self.spin_num).any(-1).to(dt)                # [B,4]
        p_spin_blocked = is_spin * p_blocked                                               # [B,4]
        # --- Protect: the mass the decay multiplies ---
        is_dmg_k = self.damaging_num[topk_nums.clamp(min=0,
                                                     max=self.damaging_num.shape[0] - 1)]  # [B,K]
        attack_mass = (alpha * is_dmg_k.to(dt)).sum(-1, keepdim=True)                      # [B,1]
        is_prot = (req_move_ids[..., None] == self.protect_nums).any(-1).to(dt)            # [B,4]
        prot_mass = is_prot * attack_mass                                                  # [B,4]
        raw = torch.stack([
            e_high,                                       # E[damage | they switch]
            e_pko,                                         # E[P(KO) | switch] — NULLED at hidden
            e_mult,                                        # E[type effectiveness | switch]
            wasted_ko,                                     # "don't click the KO into the switch"
            a_switch.expand(-1, self.n_moves),             # §2.1's ONE per-turn scalar, broadcast
            p_spin_blocked,                                # the spin dies to a Ghost final defender
            p_spin_blocked * our_side_hazards,             # ...and this much removal dies with it
            prot_mass,                                     # will they attack at all
            prot_mass * protect_odds,                      # ...and will the Protect be there
        ], dim=-1)                                                                # [B,4,_RAW]
        return self.proj(raw)  # type: ignore[no-any-return]  # [B,4,out]
