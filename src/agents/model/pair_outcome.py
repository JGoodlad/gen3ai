"""gen3_pair_outcome_v1 — the UNIFIED PER-PAIR OUTCOME VECTOR and its α-weighted delivery.

`design_opponent_intent.md` §5.1 (component 1) + §5.3 (component 3), and
`design_pair_reduction.md` §2.1 (the missing coordinates) / §3.1 (Contract W) / §9a (the admission
test). This is **Phase A** — the MOVE-cell half. The switch cell and the β-conditioned cells are
Phase B and are deliberately NOT built here.

## The defect this closes

`design_pair_reduction.md` §2.1, verified against the live tree, states it exactly:

> The decision *"they will click Will-O-Wisp, so bring the Natural Cure mon"* is made at the switch
> logit. The switch logit's per-action cell contains **no status information at all**.

Ten damage numbers, one speed number, two belief-mass numbers — and status reaching the policy only
through the `s3` edge family, i.e. as a softmax-normalised **ratio**, never as a per-action
absolute. That is a **CURRENCY failure one level below the reduction failure**: you cannot trade
"35% of my HP" against "80% chance of burn" if the two never appear in the same vector in the same
units, and no reducer however expressive repairs it. It is also the most economical reading of the
G1 n=299 null — a 2800-dim SKYLINE over the un-collapsed pair grid could not beat the collapsed
summary, which says the quantity the decision needs was never in the grid at all.

So this module does the thing §2.1 says binds FIRST: **fix what the message carries, then reduce
it.**

## The coordinate table — and every one of them answers §9a

§9a's admission test is one line: *name two specific actions whose ordering it flips; if you cannot
produce the pair, it is decoration.* Phase A delivers the reduced row of our **ACTIVE** defender to
the pointer MOVE cell, so each answer below names a MOVE pair (the axis this phase actually moves);
where the canonical design example is a SWITCH pair it is given too, marked (Phase B).

| # | coordinate | currency | §9a — the two actions it flips |
|---|---|---|---|
| 0-5 | `p_par p_brn p_frz p_slp p_psn p_tox` | probability, per status IDENTITY | **click Swords Dance vs click Earthquake** under a believed Spore: setup is worthless if you may never act again, while the attack still banks damage — and the ten damage numbers are IDENTICAL in both branches, because Spore deals none. (Phase B: **switch Swampert vs switch Celebi** into a Gengar believed to hold Will-O-Wisp + Thunderbolt.) |
| 6 | `neutralization` | fraction of this mon's per-turn contribution destroyed WITHOUT a KO | **click Substitute vs click Calm Mind** on a physical attacker facing a likely Will-O-Wisp: the sub blocks the burn and preserves Atk, CM does nothing about it. Damage-only reads BOTH branches at 0.0 incoming. |
| 7 | `tempo_cost` | turns of OUR clock spent undoing it | **click Substitute vs click Toxic**: if the status will land and this mon carries a cure, the status costs a turn LATER, so spending a turn NOW to prevent it can be right. Without tempo, "it lands and I cure it" and "it never lands" are the SAME state, so the sub is never worth its turn. (Phase B: Milotic **Refreshes** the burn away — `neutralization` correctly reads ≈0 — but "absorbs it" and "absorbs it and falls a turn behind a setup sweeper" are otherwise indistinguishable.) |

Coordinates 0-5 hold the six columns the op's status physics already computes and then throws away;
6 and 7 are the two `design_pair_reduction.md` §2.1 names as MISSING. The damage half (indices
0..5 of the op's own `last_pair_cells` — `[low, high, crit, ko_ramp, acc, is_phys]`) is
concatenated in FRONT of them, so `pair_in` is one vector in one place, which is the whole point of
component 1: *"damage and status are computed in two functions with two reductions, and one α
cannot weight two tensors."*

### What §9a's derivability rule REMOVED from the design's sketch

§5.1's sketch lists `p_status_land` and `p_immobilize` as two of the coordinates. Both are
**linear functions of coordinates 0-5** (`p_major = Σ_s p_s`; `p_immobilize = p_par + p_frz +
p_slp`), and 0-5 pass through a `Linear` before anything else touches them — so the derivability
rule applies in its plain form (*"derivable from what IS delivered → do not add it"*) and they are
NOT shipped. The un-collapsed per-identity vector is strictly richer: burn and sleep are not
interchangeable to a physical attacker, and a single `p_major` scalar asserts they are.

### What is deliberately NOT here

* **Physics mutation** (§5.1, §2.1) — burning Milotic multiplies its Def by 1.5 and moves EVERY
  subsequent number in the matrix. That is a statement about the successor state, not an outcome
  coordinate; a one-ply reduction can learn "burn into Milotic is fine" but cannot represent "and
  here is the new matrix". Naming the limit beats pretending a richer φ closes it.
* **Status DURATION.** `neutralization` is a per-turn fraction; sleep lasting 1-5 turns and freeze
  lasting until a fire move are not modelled. Adding an expected-duration factor would mean choosing
  numbers that are not gen3 rules (see the constants note below).
* **A held berry's auto-cure.** A Lum Berry cures at 0 tempo AND ~0 neutralization; we model neither,
  so a berry-holding mon reads as if it had no answer. Folding it in would mean PRE-BLENDING two
  probabilistic branches into one column, which §9's anti-patterns forbid.

## The reduction — ONE α, every channel (Contract W)

> **Contract W** (`design_pair_reduction.md` §3.1). A weighted reducer emits `α ∈ Δ^K` — ONE
> distribution over the move axis, computed from the board and their moves ONLY, never per defender
> and never per channel — and the reduced cell is `Σ_k α_k · pair_in[k, j, :]`.

`reduce_pair_in` IS that one line, and the shape is the enforcement: α has no channel axis and no
defender axis, so per-channel maxima (defect **D2** — the flat block takes NINE independent maxima,
so up to nine different opponent moves describe one defender) and per-defender α (defect **D3**) are
both shape errors here rather than things a test hunts for. `pair_outcome_test` plants the D2
violation explicitly and asserts the contract catches it.

Two properties come free and both matter downstream: `Σα ≤ 1, α ≥ 0` ⇒ every coordinate stays a
convex combination of values in `[0,1]`, so units, range and scale survive, and the pointer head's
`tanh` sees the responsive region rather than the saturated tails.

## Where α comes from — and the R1 fallback that makes this flag independently enableable

Two rungs, and which one runs is decided by whether the intent head exists:

* **`--opp-intent` ON** → α is the softmax of the PUBLISHED α logits, **move slice only,
  UNRENORMALIZED** (the `IntentValueReduce` / v84 / v85 precedent). The missing SWITCH mass is the
  literally-correct statement that a switching opponent applies no outcome to us this turn, so a
  high `α_SWITCH` must shrink every coordinate toward zero; renormalizing would assert they
  attacked. Read from the PUBLICATION and additionally **stop-grad**: this is a policy-side
  consumer, and no PPO→`alpha_head` route may open through it (the v87 pattern — a route that only
  exists under `label_only` is a route that reopens the day someone changes a training flag).
* **`--opp-intent` OFF** → **R1 `belief_mean`**: `α := w / Σw` over the same top-K seats
  (`pair_reduce.alpha_belief_mean`, the shipped rung). ⚠️ **The two are NOT the same object and the
  difference is not cosmetic**: `w` is a PRESENCE belief (does this move exist in their set), α is a
  USAGE belief (will they click it this turn) — `α ≠ w` is the substantive modelling error the whole
  design names. The fallback also sums to **1** where the α path sums to `1 − α_SWITCH`, because
  without an intent head there is no switch belief to withhold mass for. So the fallback is the
  DELIVERY claim tested without the DISTRIBUTION claim (§7a.2's own suggestion), not a cheap α.

A seat that is not live — the meaningful-K gate, once all four of their moves are revealed the 5th+
top-K slot is closed — is MASKED to zero mass and, like SWITCH, **not renormalized away**: an
unmodeled seat's mass is simply not spent. That is the same rule §4.2 applies at the loss ("if we
can't name it, we don't train on it"), applied in the forward.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from agents.model.arch_constants import _PAIR_OUTCOME_DMG, _PAIR_OUTCOME_RAW

#: The reduced row's coordinate names, in order. Index 0.._PAIR_OUTCOME_DMG-1 are the op's own
#: pair-cell channels (its `last_pair_cells` F axis, unchanged); the rest are gen3_pair_outcome_v1's.
#: This tuple IS the contract the consumers and the tests read — never re-spell an index.
PAIR_OUTCOME_COORDS: Tuple[str, ...] = (
    # --- the op's existing per-(defender, seat) damage cell, verbatim ---
    "low", "high", "crit", "ko_ramp", "acc", "is_phys",
    # --- NEW: status, per IDENTITY (SECONDARY_COLS' major prefix order) ---
    "p_par", "p_brn", "p_frz", "p_slp", "p_psn", "p_tox",
    # --- NEW: the two coordinates design_pair_reduction.md §2.1 names as missing ---
    "neutralization", "tempo_cost",
)
#: Name → index over `PAIR_OUTCOME_COORDS`. The op reads `"high"` off it when building the vector,
#: so a coordinate reorder cannot silently feed the status physics the wrong damage column.
PAIR_OUTCOME_IDX = {name: i for i, name in enumerate(PAIR_OUTCOME_COORDS)}

assert len(PAIR_OUTCOME_COORDS) == _PAIR_OUTCOME_RAW, \
    "PAIR_OUTCOME_COORDS and _PAIR_OUTCOME_RAW disagree — one was edited without the other."
assert PAIR_OUTCOME_COORDS[_PAIR_OUTCOME_DMG - 1] == "is_phys", \
    "_PAIR_OUTCOME_DMG no longer names the op's damage-cell prefix."

_EPS = 1e-8


def alpha_belief_mean(w: torch.Tensor) -> torch.Tensor:
    """R1 — `α = w/Σw`, re-exported from `pair_reduce` so the fallback and the shipped rung are
    provably the same function (two spellings of a distribution is how they drift apart)."""
    from agents.model.pair_reduce import alpha_belief_mean as _r1
    return _r1(w)


def pair_alpha(alpha_logits: Optional[torch.Tensor], w_topk: torch.Tensor,
               seat_live: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Resolve the ONE distribution over the opponent's believed-move axis. `[B,K]`.

    `alpha_logits` `[B,K+1]` (last class = SWITCH) is the PUBLICATION when `--opp-intent` is on, or
    `None`; `w_topk` `[B,K]` is the belief's top-K presence weights (the R1 fallback's source AND
    the shape oracle); `seat_live` `[B,K]` is the meaningful-K gate (`None` = every seat live).

    Fails loud when α's seat count and the op's top-K disagree — they are the SAME axis
    (`intent_axis_alignment_test`), so a mismatch means one was reconfigured independently and a
    silent broadcast would pair each α weight with the WRONG opponent move while every shape check
    still passed (the named `op move-order` bug class).
    """
    k = w_topk.shape[-1]
    if alpha_logits is not None:
        if alpha_logits.shape[-1] - 1 != k:
            raise ValueError(
                f"alpha has {alpha_logits.shape[-1] - 1} move seats but the op stashed {k} top-K "
                "candidates. These must be the SAME axis (entity_topk_seats == damage_topk_k); a "
                "mismatch would pair each alpha weight with the wrong opponent move while every "
                "shape check still passed.")
        # UNRENORMALIZED move slice — the missing SWITCH mass reads "they apply no outcome to us
        # this turn". `.detach()` is the stop-grad: a POLICY-side consumer must not open a
        # PPO -> alpha_head route, and relying on `belief_grad_mode=label_only` for that would make
        # the route's existence a function of a TRAINING flag.
        alpha = torch.softmax(alpha_logits.detach().float(), dim=-1)[:, :k].to(w_topk.dtype)
    else:
        # R1 `belief_mean` — presence in, presence out. Documented in the module docstring as a
        # DIFFERENT quantity, not a cheap alpha. `w_topk` is already detached by the op's stash.
        alpha = alpha_belief_mean(w_topk.float()).to(w_topk.dtype)
    if seat_live is not None:
        # Mask, never renormalize: an unmodeled seat's mass is not spent, exactly like SWITCH's.
        alpha = alpha * seat_live.to(alpha.dtype)
    return alpha


def reduce_pair_in(alpha: torch.Tensor, pair_in: torch.Tensor, gate: torch.Tensor,
                   our_active_idx: torch.Tensor) -> torch.Tensor:
    """Contract W's one line, at our ACTIVE defender: `Σ_k α_k · pair_in[k, active, :]` → `[B,F]`.

    `alpha` `[B,K]` · `pair_in` `[B,J,K,F]` · `gate` `[B,J,1]` (alive × has_opp) ·
    `our_active_idx` `[B]`.

    **α is defender-independent and channel-independent BY SIGNATURE** — no `J` axis and no `F`
    axis exists on it — so D3 (a per-defender α: Skarmory's row assuming "they click Rock Slide"
    while Blissey's assumes "they click Thunderbolt") and D2 (a per-channel argmax) are shape
    errors here rather than properties a test has to hunt for.
    """
    if alpha.shape[-1] != pair_in.shape[2]:
        raise ValueError(
            f"alpha carries {alpha.shape[-1]} seats but pair_in has {pair_in.shape[2]} candidate "
            "columns — the SAME axis (the `op move-order` bug class).")
    ar = torch.arange(pair_in.shape[0], device=pair_in.device)
    rows = pair_in[ar, our_active_idx]                                   # [B,K,F]
    g = gate[ar, our_active_idx]                                         # [B,1]
    return torch.einsum("bk,bkf->bf", alpha, rows) * g                   # [B,F]


class PairOutcomeMoveCell(torch.nn.Module):
    """`reduced pair_in row [B,F] → the extra pointer-move-cell block [B, 4, out_dim]`.

    The row carries no per-request-slot dependence — it is *what is coming at me this turn, in
    every currency* — so it rides every move slot identically, exactly like `intent_threshold`'s
    `p_ko` context channel. That is **not** a no-op on the logits: the pointer scorer is an MLP over
    `(move token ⊕ move cell)`, so a constant cell modulates how each DIFFERENT move token is
    scored (and the switch logits, Phase B, do not receive it at all — so the move/switch balance
    moves too). A term that were merely additive on the logit would cancel in the softmax; this one
    does not.

    Zero-init projection ⇒ ON-at-init contributes exactly zero to every action logit, so any
    measured effect on a trained run is something the run LEARNED. `restore_identity_init` captures
    it BY OBSERVATION (ledger M1 — SB3's ortho pass re-initialises every extractor Linear on policy
    build and would otherwise silently falsify the claim).

    Seat-permutation invariant by construction: the only seat-indexed computation upstream is
    `Σ_k α_k · f_k`, and a joint permutation of (α's seats, `pair_in`'s candidate columns) leaves
    the sum unchanged.
    """

    n_moves: int = 4

    def __init__(self, out_dim: int, in_dim: int = _PAIR_OUTCOME_RAW):
        super().__init__()
        self.out_dim = int(out_dim)
        self.in_dim = int(in_dim)
        self.proj = torch.nn.Linear(self.in_dim, self.out_dim)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, row: torch.Tensor) -> torch.Tensor:
        """`row` `[B,F]` (the α-reduced outcome vector at our active) → `[B,4,out_dim]`."""
        if row.shape[-1] != self.in_dim:
            raise ValueError(
                f"PairOutcomeMoveCell was built for a {self.in_dim}-wide outcome vector but got "
                f"{row.shape[-1]} — PAIR_OUTCOME_COORDS and the op's producer have drifted.")
        cells = row[:, None, :].expand(row.shape[0], self.n_moves, self.in_dim)   # [B,4,F]
        return self.proj(cells)  # type: ignore[no-any-return]  # [B,4,out]
