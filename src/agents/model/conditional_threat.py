"""gen3_conditional_threat_v1 — OA1, the CONDITIONAL THREAT CELL (the defensive pivot).

`design_conditional_opponent_cells.md` §1 (Part 1 — OA1) and §0.2 (the MAGNITUDE rule that gives
Part 1 its shape). **Phase C** of the conditional-mechanics substrate; Phase A
(`pair_outcome.py`) unified the currency of what the opponent does to us, Phase B put that unified
row on the SWITCH cell and added the β half, and this is the last of the defensive coordinates —
the ones the row structurally cannot carry.

The design's own sentence for what it buys: *"they'll Ice Beam my Salamence; switch to the mon that
eats Ice Beam."*

## What the design asked for, and what it gets INSTEAD (the Phase-A/B substitutions)

§1.2 predates both shipped phases, so three of its five clauses are already law elsewhere. Where
the design's plumbing is superseded, the SHIPPED machinery is used and the substitution is recorded
here rather than silently taken:

| §1.2 clause | what ships | why |
|---|---|---|
| `w_k = softmax_k(λ·threat_k + log belief_k)`, `λ` a learned `nn.Parameter` init 0 | **`pair_alpha`** — the published α (or the R1 `belief_mean` rung) | The design's `w` is a hand-rolled opponent-rationality prior with one learned temperature; `α` is a SUPERVISED usage belief over the SAME seats. Building a second weighting would be a second α — exactly the D2/D3 family Contract W makes a shape error — and it would sit beside the shipped one disagreeing with it. **λ is not built.** |
| `Σ_k w_k · [high(k,j), pko(k,j), status_lands(k,j)]` | **already delivered** by `--pair-outcome-switch` (`reduce_pair_in_all`) | Phase B puts `Σ_k α_k · pair_in[k,j,:]` — which contains `high`, `ko_ramp`, `acc` and the six status identities — on mon j's own switch cell. Re-emitting them here would be duplicated delivery, and `status_lands = Σ_s p_s` is additionally barred by §9a's derivability rule. |
| `... type_mult(k,j)` | **BUILT here** (`e_type_mult`) | The one cell channel `pair_in` does not carry, and the only one that is not divided by the defender's own bulk. |
| `⊕ margin_j` (§0.2(3)) | **BUILT here**, twice (`margin_high`, `margin_crit`) | §0.2(3) is a standing rule — *"probabilities SATURATE; ship the MARGIN too"* — and §0.2(4) says keep the roll DISTRIBUTION, so the rule applies to the crit roll as well as the max roll. |
| §1.3 "also turn on `--damage-matrices-outgoing-all`" | **VOID** | That flag was deleted outright at v88 (`gen3_dead_flag_purge_v1`); its OAX block is gone and its kernel survives as `d2`'s engine. The switch cell's offensive half is not reachable through a flag any more, and re-adding one is not this phase's licence. |

So OA1 ships as **four coordinates**, not the design's five-wide vector: the three the reduced row
cannot express, plus the accuracy-folded KO probability §0.2(2) says the operator must precompute.

## The coordinates

    e_pko_acc[j]    = Σ_k α_k · ko_ramp(k,j) · acc(k)
    e_type_mult[j]  = Σ_k α_k · type_mult(k,j)
    margin_high[j]  = Σ_k α_k · high(k,j) − hp_frac(j)          # > 0 ⇒ dead
    margin_crit[j]  = Σ_k α_k · crit(k,j) − hp_frac(j)          # > 0 ⇒ dead to a crit

All four are contractions of the form `Σ_k α_k · f(k, j)` against ONE α with no `J` axis and no
channel axis, so Contract W holds by signature exactly as it does in Phase A/B; and `hp_frac` is
our OWN mon's observed HP, which enters after the contraction and therefore cannot make α
defender-dependent.

### §9a — the two actions each coordinate flips, that the reduced row cannot

`design_pair_reduction.md` §9a's admission test is one line: *name two specific actions whose
ordering it flips; if you cannot produce the pair, it is decoration.* This is a SWITCH cell, so
every pair below is switch-vs-switch — the axis OA1 exists to move.

**`e_pko_acc` — the product §0.2(2) says the op must form.** `ko_ramp` and `acc` ride the reduced
row DECORRELATED, and a thin `tanh` MLP over a shared cell does not multiply two of its own inputs.
The case is exact: their believed set is {Blizzard 70% acc, which OHKOs our Salamence; Thunderbolt
100% acc, which OHKOs our Starmie}, α = ½/½. **switch Salamence vs switch Starmie** — the two mons
are IDENTICAL in every decorrelated channel (`Σα·ko_ramp` = 0.5 for both; `Σα·acc` = 0.85 for both,
since `acc` has no defender axis at all) while the quantity the pivot needs, P(this mon dies), is
0.35 for Salamence and 0.50 for Starmie. Second: **switch a mon that dies to a believed Focus Punch
vs one that dies to a believed Hypnosis-then-anything** — same shape, different numbers.

**`e_type_mult` — the only channel not divided by the defender's own bulk.**
1. **switch Gengar vs switch Swampert into a believed Earthquake.** Swampert takes a real hit;
   Gengar's `high` is `0.0` — and so is the `high` of *any* mon on a turn the seat is a status move,
   or of a mon the roll simply cannot dent. `type_mult = 0.0` is the coordinate that separates a
   STRUCTURAL immunity ("this pivot is free forever") from an incidental zero ("this pivot is free
   once"), and bringing the immune mon rather than the merely-bulky one is the whole content of
   *"switch to the mon that eats Ice Beam."*
2. **switch Skarmory vs switch Blissey into a believed Fire Blast.** Blissey's max roll is a smaller
   FRACTION of its own enormous bar than Skarmory's is of a much smaller one, so `high` and
   `ko_ramp` can rank Blissey the safer switch-in — and they are right about *this* turn.
   `type_mult` (1× vs 2×) is the read that survives the mon's own HP moving: at Skarmory 40% the
   ordering of `high` flips and nothing else in the cell records why it was going to.

**`margin_high` / `margin_crit` — §0.2(3), at both ends of the saturation.**
1. **switch Swampert vs switch Milotic into a believed Hidden Power Grass that KOs neither.**
   `ko_ramp` is ~0 for both, so the probability channel ties them exactly; `margin_high` says
   Swampert survives with 4% of its bar and Milotic with 41% — i.e. one of them dies to the
   follow-up and the other does not.
2. **switch Tyranitar vs switch Metagross into a believed Hydro Pump that KOs both.** Now `ko_ramp`
   saturates at ~1 for both; `margin_high` says Tyranitar is dead by 60% of its bar and Metagross by
   2%, so a low roll saves one of them and neither probability channel can say so.
3. `margin_crit` is the same rule on the crit roll, and it separates a *safe* pivot from a
   *coinflip* one: **switch mon A vs switch mon B** where both survive the max roll (identical
   `ko_ramp` 0, similar `margin_high`) and only one survives a critical hit. Gen 3 crits are ~1/16
   per attack, so this is a real fraction of pivots, and it is invisible in every other coordinate.

### What is deliberately NOT here

* **A second α.** See the substitution table. `λ` is not built, and no rung of `pair_reduce` is
  reached for either — this consumer takes the same `pair_alpha` ladder every other one takes.
* **`status_lands`.** `Σ_s p_s` over coordinates already delivered; §9a's derivability rule.
* **The switch-in's own hazard/entry cost.** A pivot's price includes Spikes and the free turn the
  opponent gets. Neither is a *conditional threat* — they do not depend on what the opponent
  clicks — so folding them into an α-contraction would assert a dependence that is not there.
* **A KO-margin on the LOW roll.** `low` is in the reduced row; a third margin would be three
  spellings of one rule, and `high`/`crit` already bracket the distribution the pivot cares about.

### Contract compliance (the four α-consumer rules + Phase A's two)

α is read from the PUBLICATION and **stop-grad unconditionally** (`pair_alpha` detaches), because a
policy-side consumer whose PPO→head route exists only when `--belief-grad-mode` says so is one
training-flag change from silently reopening. Seat widths are checked against the op's top-K and
fail loud (the `op move-order` bug class). The projection is zero-init and captured by
`restore_identity_init` (M1). Seat-permutation invariant: every seat-indexed computation is
`Σ_k α_k · f_k`. Our-team-permutation EQUIvariant: row j rides mon j's own switch logit and α has
no `J` axis, so permuting our six permutes the cells with them.

**The fallback is meaningful here, and that is a judgement rather than a habit** (`model/CLAUDE.md`
records the v94 case where it was not). Every coordinate is a *what lands on me if they attack*
contraction, so the unrenormalized move slice's missing SWITCH mass is the literally-correct
statement that a switching opponent applies no threat this turn, and the R1 `belief_mean` rung
states a PRESENCE belief rather than nothing at all. So `--conditional-threat-cell` requires
`damage_op` and not `opp_intent`, and a run can test the DELIVERY claim apart from the
DISTRIBUTION claim — the same split Phases A and B ship.
"""
from __future__ import annotations

from typing import Tuple

import torch

from agents.model.arch_constants import _CONDITIONAL_THREAT_RAW
from agents.model.pair_outcome import PAIR_OUTCOME_IDX

#: The cell's coordinate names, in order — the contract the consumers and the tests read. Never
#: re-spell an index; `conditional_threat_test` asserts this tuple against `_CONDITIONAL_THREAT_RAW`.
CONDITIONAL_THREAT_COORDS: Tuple[str, ...] = (
    "e_pko_acc", "e_type_mult", "margin_high", "margin_crit",
)
CONDITIONAL_THREAT_IDX = {name: i for i, name in enumerate(CONDITIONAL_THREAT_COORDS)}

assert len(CONDITIONAL_THREAT_COORDS) == _CONDITIONAL_THREAT_RAW, \
    "CONDITIONAL_THREAT_COORDS and _CONDITIONAL_THREAT_RAW disagree — one was edited alone."


class ConditionalThreatCell(torch.nn.Module):
    """gen3_conditional_threat_v1 — `(the op's pair cells, the type multiplier, α, our HP) → the
    extra pointer-SWITCH-cell block [B, 6, out_dim]`.

    The SECOND module to widen the pointer switch cell (Phase B's `PairOutcomeSwitchCell` was the
    first), and it is deliberately independent of it: the two deliver different quantities to the
    same sink, and coupling them would make a measured result unattributable to either.

    Zero-init projection ⇒ ON-at-init contributes exactly zero to every switch logit, so any
    measured effect on a trained run is something the run LEARNED; `restore_identity_init` captures
    it BY OBSERVATION (ledger M1 — SB3's ortho pass re-inits every extractor Linear on the only
    construction path training uses, and would otherwise silently falsify the claim).
    """

    n_switch: int = 6

    def __init__(self, out_dim: int, in_dim: int = _CONDITIONAL_THREAT_RAW):
        super().__init__()
        self.out_dim = int(out_dim)
        self.in_dim = int(in_dim)
        self.proj = torch.nn.Linear(self.in_dim, self.out_dim)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, alpha: torch.Tensor, pair_in: torch.Tensor, type_mult: torch.Tensor,
                gate: torch.Tensor, hp_frac: torch.Tensor) -> torch.Tensor:
        """`alpha` `[B,K]` (the move slice, already seat-masked + stop-grad) · `pair_in`
        `[B,6,K,F]` (the unified outcome grid — read for `ko_ramp`, `acc`, `high`, `crit`) ·
        `type_mult` `[B,6,K]` (the op's per-(defender, seat) multiplier) · `gate` `[B,6,1]`
        (alive × has_opp) · `hp_frac` `[B,6]` (our mons' current HP fraction) → `[B,6,out_dim]`.

        Fails loud on a seat-axis width mismatch — α's seats, `pair_in`'s candidate columns and
        `type_mult`'s are ONE axis, and a silent broadcast would pair each α weight with the wrong
        opponent move while every shape check still passed (the named `op move-order` bug class).
        """
        if alpha.shape[-1] != pair_in.shape[2] or type_mult.shape[-1] != pair_in.shape[2]:
            raise ValueError(
                f"alpha carries {alpha.shape[-1]} seats, pair_in {pair_in.shape[2]} candidate "
                f"columns and type_mult {type_mult.shape[-1]} — these are the SAME axis "
                "(entity_topk_seats == damage_topk_k); the `op move-order` bug class.")
        if type_mult.shape[1] != pair_in.shape[1]:
            raise ValueError(
                f"type_mult carries {type_mult.shape[1]} defenders but pair_in has "
                f"{pair_in.shape[1]} — one of them was built over a different team axis.")
        dt = pair_in.dtype
        a = alpha.to(dt)
        # `pko` in the design's sense = acc · P(KO | hit). The op ships the two DECORRELATED, and
        # §0.2(2) is explicit that the product must be formed HERE rather than left to the head:
        # a thin tanh scorer over a shared cell does not multiply two of its own inputs.
        pko_acc = (pair_in[..., PAIR_OUTCOME_IDX["ko_ramp"]]
                   * pair_in[..., PAIR_OUTCOME_IDX["acc"]])                        # [B,6,K]
        e_pko_acc = torch.einsum("bk,bjk->bj", a, pko_acc)                         # [B,6]
        e_type = torch.einsum("bk,bjk->bj", a, type_mult.to(dt))                   # [B,6]
        e_high = torch.einsum("bk,bjk->bj", a, pair_in[..., PAIR_OUTCOME_IDX["high"]])
        e_crit = torch.einsum("bk,bjk->bj", a, pair_in[..., PAIR_OUTCOME_IDX["crit"]])
        hp = hp_frac.to(dt)                                                        # [B,6]
        raw = torch.stack([
            e_pko_acc,                                   # the accuracy-folded P(this mon dies)
            e_type,                                      # E[effectiveness], bulk-independent
            e_high - hp,                                 # §0.2(3): > 0 ⇒ dead on the max roll
            e_crit - hp,                                 # ...and on the crit roll
        ], dim=-1) * gate.to(dt)                                                   # [B,6,RAW]
        return self.proj(raw)  # type: ignore[no-any-return]  # [B,6,out]
