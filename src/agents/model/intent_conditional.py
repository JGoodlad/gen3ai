"""gen3_intent_conditional_v1 — the remaining α-conditioned mechanic cells (build steps 4 + 7).

`design_conditional_execution.md` §§3.1/3.6/3.7/3.8. The threshold operator (v84) landed the five
`p_thresh` mechanics; this lands the rest of the class-A set plus Explosion — each a per-request-
slot cell computed from tensors the op ALREADY stashes (the pair cells, the outgoing per-move
rolls, `p_outspeed`, the secondary columns) contracted with the published α. No new physics.

* **Counter / Mirror Coat** (§3.7 — "the purest read-the-opponent moves in gen3; literally
  unplayable without an intent model"): `E[counter] = Σ_k α_k · 1[k is PHYSICAL] · 2·damage(k,me)`,
  mirrored for Mirror Coat over the special-damaging candidates. Each fails against the other
  category, against status and against a switch — all of which the α-weighted category sum
  expresses and a presence belief cannot. The return is delivered in my-maxhp-fraction units
  (the head owns the scale; the ×2 is a constant it can learn).
* **Flinch** (§3.8): the raw flinch chance already rides the move cell's secondary columns —
  what is missing is the conditioning that makes it MEANINGFUL:
  `p_flinch_useful = p_outspeed · p_flinch · (1 − α_SWITCH)` — a flinch against a switching
  opponent is worth exactly nothing.
* **Explosion / Self-Destruct** (§3.1 — the ledger-H1 companion): the facts the valuation
  needs, decorrelated — `p_executes = 1 − Σ_k α_k·1[k is Protect/Detect]` (the worst branch, and
  it is α-visible) and `α_SWITCH` (detonating on an arrival we did not choose). ENDURE is not a
  blocker (gen3_op_ability_status_gigo_v1, 2026-10-07): it survives the hit at 1 HP and the user
  still faints, so the detonation lands; the shipped set counted it.
* **Protect / Detect** (§3.3, build step 5): `E[protect] = Σ_k α_k · damage_avoided(k) −
  tempo_cost`, where today's `c4` edge carries the mechanical `p_success` MULTIPLIER and omits
  the quantity it multiplies. Three decorrelated channels: the α-weighted incoming damage a
  successful Protect avoids, the obs floored-doubling `p_success` (the same scalar `c4` reads),
  and the α mass on STATUS seats (a successful Protect blocks the status too — the currency
  §2.1 named missing). Endure is deliberately NOT in this gate — its value is the v84 `p_KO`
  branch, not damage avoidance. The odds channel is `p_success · P(an action follows)`
  (gen3_op_ability_status_gigo_v1): Protect FAILS when no action follows it in the queue
  (`onPrepareHit: !!this.queue.willAct()`), so into a switch and when we move last —
  `Σ_k α_k·(1 − P(seat k first)) + a_un·(1 − P(an unnamed priority-0 seat first))`, the order
  from `move_order.p_seat_first` (the family's own rule).
* **Magic Coat** (§3.12, build step 6 — its G0 oracle ran FIRST, per the doc's own gate): the
  reflectable set was UNVERIFIED, so five constructed scenarios on the reference sim resolved
  it (`measurements/gen3_magiccoat_reflectable_oracle.json`): foe-targeting status (Toxic /
  Thunder Wave / Leech Seed / Will-O-Wisp) BOUNCES; side-targeting Spikes does NOT — it lands
  on the user's own side. The cell is `is_magiccoat · Σ_k α_k·is_reflectable_k`, the predicate
  the simulator's own `reflectable` flag set (`move_resolution_rules.REFLECTABLE`, re-derived from
  the gen-3 dex by `move_resolution_rules_integration_test`). The shipped predicate (status AND raw
  target 'normal') also bounced Taunt / Encore / Disable / Torment / Roar, which carry no flag
  (gen3_op_ability_status_gigo_v1, 2026-10-07); the oracle's five scenarios agree with both.
* **Explosion's β half** (§3.1, class B — the FIRST forward-side β consumer): the trade's
  target differs by branch, so the KO probability is
  `α_stay · pko(boom, their active) + α_SWITCH · Σ_j β_j · pko(boom, arrival j)` — β published
  through the same stop-grad boundary as α (label_only keeps cutting the PPO route), the
  per-(move, their mon) pko from the outgoing matrix (which prices an unrevealed arrival's
  P(KO) as NULLED — so unrevealed β mass honestly contributes zero rather than a guess).
* **Pursuit** (§3.6, CORRECTED): the doc's formula weighted the damage by a β-weighted
  switch-IN, but the sim (verified against `src/rust_sim/src/state.rs`'s pursuit interrupt, itself
  golden-gated against Showdown) strikes the DEPARTING mon at ×2 BP, never-miss, before the
  switch resolves. So no β enters: `E[pursuit] = dmg(active) + α_SWITCH · (2·dmg − dmg)`. The
  cell carries the trigger probability and the α-weighted bonus damage separately.

Delivery: one zero-init `Linear(_INTENT_COND_RAW, out_dim)` appended to the pointer MOVE cell
beside the v77/v84 blocks (per-action absolutes — the channel measured to work). ON-at-init is
bit-identical (ledger M1: the identity-init sweep captures the projection by observation).
Seat-permutation invariant: the only seat-indexed computation is `Σ_k α_k · f_k`.
"""
from __future__ import annotations

from typing import Optional, cast

import torch

from agents.gen3_data.moves import MoveData

from agents.model.arch_constants import _INTENT_COND_RAW
from agents.model.move_order import p_seat_first
from agents.model.pair_outcome import seat_in_set, seat_num_table

# Gate move NUMS (gen3_data.moves, read 2026-08-16): counter 68, mirrorcoat 243, explosion 153,
# selfdestruct 120, pursuit 228. Protect/Detect/Endure = the op's _PROTECT_NUMS (182/197/203).
def _gate_nums() -> torch.Tensor:
    from agents import gen3_data
    nums = []
    for mid in ("counter", "mirrorcoat", "explosion", "selfdestruct", "pursuit", "magiccoat"):
        md = gen3_data.moves.get(mid)
        if md is None:
            raise ValueError(f"intent_conditional: move {mid!r} missing from gen3_data.moves — "
                             "the mechanic gate would silently never fire.")
        nums.append(int(md.num))
    return torch.tensor(nums, dtype=torch.long)


_PROTECT_ONLY = (182, 197)            # the damage-avoidance pair (Endure's value is p_KO, v84)


def _reflectable_table() -> torch.Tensor:
    """[n_move_nums] 1.0 where gen3 Magic Coat bounces the move — the simulator's ``reflectable`` flag
    set (`move_resolution_rules.REFLECTABLE`, ONE declaration, re-derived from the gen-3 dex by its
    integration test). Fail-loud on an id the facade does not know (a silent drop reads as "nothing
    to bounce")."""
    from agents import gen3_data
    from agents.model.move_resolution_rules import REFLECTABLE
    raw = gen3_data.moves.raw()
    n = max(cast(MoveData, gen3_data.moves.get(mid)).num for mid in raw) + 1
    t = torch.zeros(n, dtype=torch.float32)
    for mid in REFLECTABLE:
        md = gen3_data.moves.get(mid)
        if md is None:
            raise ValueError(f"intent_conditional: reflectable move {mid!r} missing from gen3_data.moves")
        t[md.num] = 1.0
    return t


def _priority_table() -> torch.Tensor:
    """[n_move_nums] each move num's priority (the data facade's gen-3 value) — the ORDER input of the
    Protect rule (`move_order.p_seat_first`)."""
    from agents import gen3_data
    raw = gen3_data.moves.raw()
    n = max(cast(MoveData, gen3_data.moves.get(mid)).num for mid in raw) + 1
    t = torch.zeros(n, dtype=torch.float32)
    for mid in raw:
        md = gen3_data.moves.get(mid)
        if md is not None:
            t[md.num] = float(md.priority)
    return t


def _status_table() -> torch.Tensor:
    """[n_move_nums] 1.0 where the move num is a STATUS move (`not is_damaging`), from the data
    facade — so 'α mass on status seats' cannot be conflated with an immune damaging seat
    (both read high == 0 in the pair cells)."""
    from agents import gen3_data
    n = max(cast(MoveData, gen3_data.moves.get(mid)).num for mid in gen3_data.moves.raw()) + 1
    t = torch.zeros(n, dtype=torch.float32)
    for mid in gen3_data.moves.raw():
        md = gen3_data.moves.get(mid)
        if md is not None and not md.is_damaging:
            t[md.num] = 1.0
    return t


class IntentConditionalMoveCell(torch.nn.Module):
    """`(published α, op stashes, request ids) → the extra pointer-move-cell block [B,4,out]`.

    Zero-init projection ⇒ ON-at-init contributes exactly zero to every action logit."""

    gate_nums: torch.Tensor
    protect_only_nums: torch.Tensor
    priority_num: torch.Tensor
    status_num: torch.Tensor
    reflectable_num: torch.Tensor

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = int(out_dim)
        self.proj = torch.nn.Linear(_INTENT_COND_RAW, self.out_dim)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)
        self.register_buffer("gate_nums", _gate_nums(), persistent=False)          # [6]
        self.register_buffer("protect_only_nums",
                             torch.tensor(_PROTECT_ONLY, dtype=torch.long),
                             persistent=False)                                     # [2]
        self.register_buffer("status_num", _status_table(), persistent=False)      # [n_nums]
        self.register_buffer("reflectable_num", _reflectable_table(), persistent=False)
        self.register_buffer("priority_num", _priority_table(), persistent=False)  # [n_nums]

    def forward(self, alpha_logits: torch.Tensor, pair_cells: torch.Tensor,
                pair_gate: torch.Tensor, our_active_idx: torch.Tensor,
                topk_nums: torch.Tensor, out_high: torch.Tensor,
                p_outspeed: torch.Tensor, sec_flinch: torch.Tensor,
                req_move_ids: torch.Tensor, protect_odds: torch.Tensor,
                beta_logits: torch.Tensor, out_pko_mj: torch.Tensor,
                opp_active_local: torch.Tensor,
                seat_live: Optional[torch.Tensor] = None,
                other_u: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`alpha_logits` [B,K+1] (last class = SWITCH) · `pair_cells` [B,6,K,6] (the op's
        [low,high,crit,ko,acc,is_phys] per (defender, seat candidate)) · `pair_gate` [B,6,1] ·
        `our_active_idx` [B] · `topk_nums` [B,K] (the seat candidates' move NUMS) · `out_high`
        [B,4] (our request moves' max-roll damage vs their active) · `p_outspeed` [B,1] ·
        `sec_flinch` [B,4] (per-slot flinch chance) · `req_move_ids` [B,4] · `protect_odds`
        [B,1] (our active's obs floored-doubling success odds) · `beta_logits` [B,6] (the
        PUBLISHED β, -inf-masked to legal switch-ins) · `out_pko_mj` [B,4,6] (the outgoing
        matrix's per-(our move, their mon) pko) · `opp_active_local` [B] → [B,4,out].

        Fails loud on a seat-axis width mismatch (the `op move-order` bug class).

        X5 U4 (fixed_mass only; `gen3_x5_flat_pointer_v1`): α arrives re-expressed from the flat
        pointer — K seats + OTHER_move (a (K+1)-th seat, priced: `pair_cells` carries its tail
        column) + the total switch mass — and β over the six slots + OTHER_species (`out_pko_mj`
        carries OTHER's column). ``seat_live`` `[B,K+1]` masks α (F-X5-15); ``other_u`` `[B,M]` gives
        OTHER_move's num-table columns (`seat_num_table` / `seat_in_set`). Both None: blob, unchanged."""
        k = alpha_logits.shape[-1] - 1                                 # last class is SWITCH
        n_nums = topk_nums.shape[-1] + (1 if other_u is not None else 0)
        if pair_cells.shape[2] != k or n_nums != k:
            raise ValueError(
                f"alpha has {k} move seats but the op stashed {pair_cells.shape[2]} candidate "
                f"channels / {topk_nums.shape[-1]} candidate nums. These must be the SAME axis "
                "(entity_topk_seats == damage_topk_k); a mismatch would pair each alpha weight "
                "with the wrong opponent move while every shape check still passed.")
        B = pair_cells.shape[0]
        ar = torch.arange(B, device=pair_cells.device)
        cells = pair_cells[ar, our_active_idx]                         # [B,K,6] our ACTIVE's row
        gate = pair_gate[ar, our_active_idx]                           # [B,1] alive · has_opp
        high_k = cells[..., 1]                                         # [B,K]
        is_phys_k = cells[..., 5]                                      # [B,K]
        dmg_k = (high_k > 0).float()                                   # damaging (status/immune = 0)
        # The Explosion BLOCKERS: Protect / Detect only (Endure survives the hit; the user still faints).
        is_protect_k = (seat_in_set(self.protect_only_nums, topk_nums, other_u)
                        if other_u is not None
                        else (topk_nums[..., None] == self.protect_only_nums).any(-1).float())   # [B,K]
        # Full α (renormalized would be wrong everywhere here): the SWITCH mass carries meaning
        # in every one of these cells, which is exactly what the v77 c2 cell could not express.
        alpha_full = torch.softmax(alpha_logits.float(), dim=-1).to(pair_cells.dtype)
        alpha = alpha_full[:, :k]                                      # [B,K] move seats
        if seat_live is not None:
            alpha = alpha * seat_live.to(alpha.dtype)
        a_switch = alpha_full[:, -1:]                                  # [B,1]
        # --- the α-weighted category sums (Counter / Mirror Coat / Explosion) ---
        e_phys = (alpha * is_phys_k * dmg_k * high_k).sum(-1, keepdim=True) * gate      # [B,1]
        e_spec = (alpha * (1 - is_phys_k) * dmg_k * high_k).sum(-1, keepdim=True) * gate
        p_phys = (alpha * is_phys_k * dmg_k).sum(-1, keepdim=True) * gate
        p_spec = (alpha * (1 - is_phys_k) * dmg_k).sum(-1, keepdim=True) * gate
        p_blocked = (alpha * is_protect_k).sum(-1, keepdim=True) * gate                 # [B,1]
        # Protect's two α-weighted quantities: the damage a successful Protect avoids, and the
        # α mass on STATUS seats (data-typed, so an immune damaging seat cannot masquerade).
        is_status_k = (seat_num_table(self.status_num, topk_nums, other_u) if other_u is not None
                       else self.status_num[topk_nums.clamp(min=0, max=self.status_num.shape[0] - 1)])
        e_dmg_avoided = (alpha * high_k).sum(-1, keepdim=True) * gate                   # [B,1]
        e_status_avoided = (alpha * is_status_k).sum(-1, keepdim=True) * gate           # [B,1]
        is_refl_k = (seat_num_table(self.reflectable_num, topk_nums, other_u) if other_u is not None
                     else self.reflectable_num[
                         topk_nums.clamp(min=0, max=self.reflectable_num.shape[0] - 1)])
        e_reflect = (alpha * is_refl_k).sum(-1, keepdim=True) * gate                    # [B,1]
        # --- the boom trade's branch-dependent KO probability (the β half) ---
        a_stay = alpha.sum(-1, keepdim=True)                                            # [B,1]
        has_cand = torch.isfinite(beta_logits).any(-1, keepdim=True).float()            # [B,1]
        beta = torch.softmax(beta_logits.float().clamp(min=-1e9), dim=-1
                             ).to(pair_cells.dtype) * has_cand                          # [B,6]
        pko_active = out_pko_mj.gather(
            2, opp_active_local[:, None, None].expand(-1, out_pko_mj.shape[1], 1))[..., 0]  # [B,4]
        pko_arrival = (out_pko_mj * beta[:, None, :]).sum(-1)                           # [B,4]
        # --- the per-slot mechanic gates ---
        gates = (req_move_ids[..., None] == self.gate_nums).float()    # [B,4,6]
        is_counter, is_mc, is_expl, is_sd, is_pursuit, is_mcoat = gates.unbind(-1)
        is_boom = is_expl + is_sd
        is_protfam = (req_move_ids[..., None] == self.protect_only_nums).any(-1).float()  # [B,4]
        # Protect FAILS with no action after it (`onPrepareHit: !!this.queue.willAct()`): P(an action follows)
        # = the α mass of seats that move AFTER our slot's move, + the unnamed seat's (priority 0); a switch
        # resolves before any move, so α_SWITCH never follows. Order: `p_seat_first` (one rule, the family's).
        prio_m = self.priority_num[req_move_ids.clamp(min=0, max=self.priority_num.shape[0] - 1)]   # [B,4]
        prio_k = self.priority_num[topk_nums.clamp(min=0, max=self.priority_num.shape[0] - 1)]   # [B,K named]
        if other_u is not None:     # X5: OTHER_move (the (K+1)-th seat) is priced at priority 0 (named residual)
            prio_k = torch.cat([prio_k, torch.zeros_like(prio_k[:, :1])], dim=-1)
        p_out = p_outspeed.to(alpha.dtype)                                             # [B,1]
        first_k = p_seat_first(prio_m[:, :, None], prio_k[:, None, :], p_out[:, :, None])    # [B,4,K]
        first_un = p_seat_first(prio_m, torch.zeros_like(prio_m), p_out)               # [B,4]
        a_un = (1.0 - a_stay - a_switch).clamp(min=0.0)                                # [B,1]
        p_follow = (alpha[:, None, :] * (1.0 - first_k)).sum(-1) + a_un * (1.0 - first_un)   # [B,4]
        raw = torch.stack([
            is_counter * e_phys,                        # the return's operand (head scales the ×2)
            is_mc * e_spec,
            is_counter * p_phys + is_mc * p_spec,       # P(the category matches at all)
            p_outspeed * sec_flinch * (1.0 - a_switch), # flinch is worthless into a switch
            is_boom * (1.0 - p_blocked),                # P(the detonation lands)
            is_boom * a_switch,                         # detonating on an arrival we didn't pick
            is_pursuit * a_switch,                      # the ×2 never-miss trigger
            is_pursuit * a_switch * out_high,           # the α-weighted bonus damage (≈ +1× high)
            is_protfam * e_dmg_avoided,                 # what a successful Protect buys (damage)
            is_protfam * protect_odds * p_follow,       # the decay odds × P(an action follows it)
            is_protfam * e_status_avoided,              # ...and the status it blocks
            is_mcoat * e_reflect,                       # P(there is something to bounce) — the
                                                        # oracle-verified foe-status set only
            is_boom * (a_stay * pko_active
                       + (alpha_full[:, -1:]) * pko_arrival),   # P(the trade KOs its real target)
        ], dim=-1)                                                     # [B,4,13]
        return self.proj(raw)  # type: ignore[no-any-return]  # [B,4,out]
