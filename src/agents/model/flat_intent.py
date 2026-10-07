"""X5's FLAT OPPONENT POINTER (`gen3_x5_flat_pointer_v1`, build unit U4;
`designs/endstate/design_x5_belief_tokens.md` §3.7). X5 only: the `blob` arm
keeps the α / β heads (`opp_intent.py`) and never calls anything here.

ONE candidate list, ONE softmax, in a fixed column layout for K move seats (`flat_layout`):

    [0, K)        the opponent ACTIVE's move seats (THE one order's E4 seats; `FixedMassMoves`)
    K             OTHER_move — the active's moves beyond the seats (its token: the active's E5 seat)
    K+1 .. K+6    switch → their slot j: a REVEALED bench mon, or the HYPOTHESIS that slot holds
    K+7           OTHER_species — a switch to a mon in the hypothesis set's tail

**Scoring.** One SHARED scorer over (candidate token ⊕ board context ⊕ the candidate's KIND, move or
switch), plus the candidate's DETACHED log-presence as a logit bias (M10): log π_m on an unrevealed
seat (0 revealed), OTHER_move's log-mass, log π of a hypothesis slot (0 revealed), OTHER_species'
log-mass. That bias is ToMe's proportional attention again (Bolya et al. 2023): a candidate of
presence w behaves as w copies of itself — **I2 for the pointer** (design §3.5): splitting a candidate
into two copies whose presences sum to w leaves the probability of the EVENT (the sum over its
copies) and of every other candidate unchanged; the per-candidate output vector changes shape. A
structurally masked candidate is −inf (never a threshold on a mass).

**Labels** (`flat_intent_targets`, built from the Rust intent label — no Rust change): a move in the
seats → its seat (a typed Hidden Power label matches a REVEALED Hidden Power's seat, num 237); a
move beyond the seats → OTHER_move; a switch to a revealed slot → that slot; a switch to a hidden mon
whose species a hypothesis slot holds → that slot; any other hidden switch-in in the tail →
OTHER_species. A belief miss is an OTHER LABEL, not a masked row. Masked: non-choices (the existing
semantics) and a choice outside every candidate's support (a move outside the presence
construction's candidate set — Struggle, a learnset gap — or a species outside V); those are
counted, never guessed.

**The consumers' re-expression** (`compat_intent_logits`): α over [K seats, OTHER_move] + the TOTAL
switch mass α_SWITCH = Σ of the switch candidates (a logsumexp, so the softmax is the flat one
exactly), and β over [six slots, OTHER_species] = α(switch → j) / α_SWITCH. β_OTHER is a column,
never renormalised away; OTHER_move is a priced seat (its op columns are the tail contraction,
`FixedMassMoves.other_u`), never mass that reads as SWITCH.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import torch

from agents.model.arch_constants import FLAT_INTENT_HIDDEN, FLAT_INTENT_INIT_SEED
from agents.model.hypothesis_set import HypothesisSet, IsolatedLinear
from agents.model.hypothesis_tokens import FixedMassMoves
from agents.model.opp_intent import INTENT_IGNORE
from agents.observation.constants import TEAM_SIZE

#: The label CLASS of a flat target (metrics only; `flat_intent_targets`' second output).
LABEL_SEAT, LABEL_OTHER_MOVE, LABEL_SLOT, LABEL_OTHER_SPECIES = 0, 1, 2, 3
LABEL_UNMODELED = -1     # a genuine choice outside every candidate's support (masked, counted)
LABEL_NONCHOICE = -2     # not the opponent's choice (the existing UNKNOWN semantics; masked)

#: The bare, typeless Hidden Power and its 16 typed nums (the Rust label resolves a click to the
#: attacker's TRUE typed num; a REVEALED HP's seat is the typeless 237 priced as its typed mixture).
_HP_NUM = 237
_TYPED_HP_LO, _TYPED_HP_HI = 355, 370


def flat_width(k: int) -> int:
    """The flat list's width for K move seats: K seats + OTHER_move + 6 slots + OTHER_species."""
    return int(k) + 2 + TEAM_SIZE


def other_move_col(k: int) -> int:
    return int(k)


def slot_col(k: int, j: int = 0) -> int:
    return int(k) + 1 + int(j)


def other_species_col(k: int) -> int:
    return int(k) + 1 + TEAM_SIZE


@dataclasses.dataclass(frozen=True)
class FlatIntentInputs:
    """The flat pointer's LABEL-side description of one forward (every tensor DETACHED; the
    `last_flat_intent` stash). The loss, the ride-along B head and the readers build targets from it
    with `flat_intent_targets`, so they can never disagree on which column a label names."""
    k: int
    live: torch.Tensor              # [B,F] bool — the candidate exists (structural)
    cand_ids: torch.Tensor          # [B,F] long — seat move nums · 0 · slot species nums · 0
    log_pi: torch.Tensor            # [B,F] the detached log-presence bias (masked: irrelevant)
    seat_nums: torch.Tensor         # [B,K] long
    seat_on: torch.Tensor           # [B,K] bool
    hp_seat: torch.Tensor           # [B,K] bool — a REVEALED Hidden Power's seat (num 237)
    beyond: torch.Tensor            # [B,M] bool — OTHER_move's members
    slot_species: torch.Tensor      # [B,6] long — the hypothesis num on a hidden slot, 0 revealed
    slot_is_hypothesis: torch.Tensor  # [B,6] bool
    in_tail: torch.Tensor           # [B,S] bool — OTHER_species' members


class FlatIntentHead(torch.nn.Module):
    """The flat pointer's ONE shared scorer (module docstring). Built from a PRIVATE seed out of
    `IsolatedLinear`s (SB3's orthogonal re-init skips them), so building it moves no non-X5 initial
    byte; its last layer is NOT zero (α's and β's were not either): a cold head is a learned-noise
    re-weighting of the presence prior, which the log-presence bias already carries.

    The last layer has NO BIAS (gen3_x5_version_break_v1 part 2, architecture audit F16b): one scorer
    feeds every candidate of ONE softmax, so its output bias is added identically to every logit and is
    shift-invariant — it changes no probability, and its gradient is Σ_c (p_c − y_c) = 0 up to rounding
    (measured over one K9 update: |g| ≤ 2.3e-10 against the weight's ~1e-3). Every consumer reads the
    flat logits through a softmax (the loss, `compat_intent_logits`' α / β, the readers)."""

    def __init__(self, token_dim: int, ctx_dim: int, hidden: int = FLAT_INTENT_HIDDEN) -> None:
        super().__init__()
        self.token_dim, self.ctx_dim = int(token_dim), int(ctx_dim)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(FLAT_INTENT_INIT_SEED)
            self.hidden = IsolatedLinear(self.token_dim + self.ctx_dim + 2, int(hidden))
            self.out = IsolatedLinear(int(hidden), 1, bias=False)

    def forward(self, tokens: torch.Tensor, ctx: torch.Tensor, log_pi: torch.Tensor,
                live: torch.Tensor, k: int) -> torch.Tensor:
        """`tokens` [B,F,D] · `ctx` [B,C] · `log_pi` [B,F] (DETACHED here, M10) · `live` [B,F] bool
        → logits [B,F]. A masked candidate is −inf; a row with NO live candidate (a padding row) is
        left all-zero (finite), so a cross-entropy over it — its target is always IGNORE — cannot be
        NaN (the deleted blob `BetaSwitchHead`'s rule)."""
        B, F, _ = tokens.shape
        if F != flat_width(k):
            raise ValueError(f"the flat pointer has {F} candidates, expected {flat_width(k)} for K={k}")
        is_move = (torch.arange(F, device=tokens.device) <= other_move_col(k)).to(tokens.dtype)
        kind = torch.stack([is_move, 1.0 - is_move], dim=-1)                   # [F,2] static
        x = torch.cat([tokens, ctx[:, None, :].expand(B, F, ctx.shape[-1]),
                       kind[None].expand(B, F, 2)], dim=-1)
        s = self.out(torch.relu(self.hidden(x))).squeeze(-1)                  # [B,F]
        s = s + torch.where(live, log_pi.detach().to(s.dtype), torch.zeros_like(s))
        s = s.masked_fill(~live, float("-inf"))
        dead = ~live.any(dim=-1, keepdim=True)                                 # [B,1] structural
        return torch.where(dead, torch.zeros_like(s), s)


def flat_candidates(seat_out: torch.Tensor, k: int, their_team_out: torch.Tensor,
                    other_out: torch.Tensor, fm: FixedMassMoves, hs: HypothesisSet,
                    opp_addressable: torch.Tensor, opp_active_flag: torch.Tensor,
                    opp_active_local: torch.Tensor, opp_species_ids: torch.Tensor,
                    ) -> Tuple[torch.Tensor, FlatIntentInputs]:
    """The candidate TOKENS [B,F,D] and the label-side description, from the forward's own handles:
    `seat_out` the trunk's extra-seat outputs (E3 [4], E4 [K], E5 [6], …), `their_team_out` [B,6,D]
    (a hidden slot's token IS its hypothesis's), OTHER_species' refined token `other_out` [B,D], the
    move axis `fm` (`FixedMassMoves`), the hypothesis set `hs` and the REAL masks. A switch target is
    live iff addressable (alive-and-revealed, or hidden — exact), not the active, and either revealed
    or holding a hypothesis (structural)."""
    B = seat_out.shape[0]
    ar = torch.arange(B, device=seat_out.device)
    e4 = seat_out[:, 4:4 + k, :]                                               # [B,K,D]
    om = seat_out[ar, 4 + k + opp_active_local, :].unsqueeze(1)                # [B,1,D] active's E5
    tokens = torch.cat([e4, om, their_team_out, other_out.unsqueeze(1).to(their_team_out.dtype)], dim=1)
    dt = their_team_out.dtype
    believed = hs.slot_is_hypothesis
    slot_live = (opp_addressable.bool() & (opp_active_flag < 0.5)
                 & (~believed | (hs.slot_species > 0)))
    live = torch.cat([fm.seat_on, fm.other_live.unsqueeze(-1), slot_live,
                      hs.other_live.unsqueeze(-1)], dim=-1)
    log_pi = torch.cat([fm.seat_logp.to(dt), fm.other_log_mass.to(dt).unsqueeze(-1),
                        hs.slot_log_pi.to(dt), hs.other_log_mass.to(dt).unsqueeze(-1)], dim=-1)
    zero = torch.zeros(B, 1, dtype=torch.long, device=seat_out.device)
    slot_sp = torch.where(believed, hs.slot_species.long(), opp_species_ids.long())
    cand_ids = torch.cat([fm.seat_nums.long(), zero, slot_sp, zero], dim=-1)
    mv = hs.moves
    assert mv is not None, "the flat pointer needs the active's move group (MoveBelief on)"
    hp_seat = mv.seat_revealed & (mv.seat_nums == _HP_NUM)
    sp = hs.species
    in_tail = sp.cand & (hs.rank >= sp.k.unsqueeze(-1)) & (sp.k > 0).unsqueeze(-1)
    fi = FlatIntentInputs(
        k=int(k), live=live.detach(), cand_ids=cand_ids.detach(), log_pi=log_pi.detach(),
        seat_nums=fm.seat_nums.detach(), seat_on=fm.seat_on.detach(),
        hp_seat=hp_seat.detach(), beyond=fm.beyond.detach(),
        slot_species=hs.slot_species.detach(), slot_is_hypothesis=believed.detach(),
        in_tail=in_tail.detach())
    return tokens, fi


class FlatConsumerOps(NamedTuple):
    """The α / β consumers' operands under the flat pointer (`ExtractorForward._flat_consumer_ops`):
    α [B,K+2] / β [B,7] re-expressed (`compat_intent_logits`), and every seat-axis or mon-axis op
    stash with OTHER's column appended — OTHER_move's from the tail contraction (`other_u`),
    OTHER_species' from the OTHER-mode D1 pass and the tail's P(Ghost). A stash whose consumer is
    off is None."""
    alpha: torch.Tensor                  # [B,K+2]  K seats · OTHER_move · log α_SWITCH
    beta: torch.Tensor                   # [B,7]    six slots · OTHER_species
    seat_live: torch.Tensor              # [B,K+1]  the seats' meaningful-K gate · OTHER_move live
    other_u: torch.Tensor                # [B,M]    OTHER_move's pricing weights
    topk_w: Optional[torch.Tensor]       # [B,K+1]  seat presence · OTHER_move's mass (shape oracle)
    pair_cells: Optional[torch.Tensor]   # [B,6,K+1,6]
    pair_in: Optional[torch.Tensor]      # [B,6,K+1,RAW]
    type_mult: Optional[torch.Tensor]    # [B,6,K+1]
    out_cells: Optional[torch.Tensor]    # [B,4,7,5]
    out_pko: Optional[torch.Tensor]      # [B,4,7]
    opp_p_ghost: Optional[torch.Tensor]  # [B,7]


def append_other(x: Optional[torch.Tensor], other: Optional[torch.Tensor], dim: int,
                 what: str) -> Optional[torch.Tensor]:
    """``x`` with OTHER's column ``other`` appended on ``dim`` (None stays None). A stash present
    WITHOUT its OTHER column is LOUD: a consumer would otherwise read a K-wide axis against a
    (K+1)-wide α — or, worse, OTHER priced as a zero row (`e_mult` 0 = IMMUNE)."""
    if x is None:
        return None
    if other is None:
        raise RuntimeError(f"fixed_mass: the op stashed {what} but not OTHER's column for it — the flat "
                           "pointer's OTHER candidate would read as a zero row (IMMUNE / no outcome).")
    return torch.cat([x, other.to(x.dtype)], dim=dim)


def compat_intent_logits(flat: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """The α-consumers' re-expression of the flat logits (§3.7): ``(alpha [B,K+2], beta [B,7])``.

    ``alpha`` = [the K seats + OTHER_move, log Σ_switch exp] — its softmax is the flat distribution's
    move columns and α_SWITCH EXACTLY (one softmax of the same numbers). ``beta`` = the seven switch
    columns — its softmax is α(switch → j) / α_SWITCH. A row with NO live switch candidate gets
    α_SWITCH = −inf (zero mass) through a guarded logsumexp: the ungarded one is −inf in value but
    NaN in gradient, and three consumers do not detach α."""
    sw = flat[:, k + 1:]                                                       # [B,7]
    any_sw = torch.isfinite(sw).any(dim=-1, keepdim=True)
    lse = torch.logsumexp(torch.where(any_sw, sw, torch.zeros_like(sw)), dim=-1, keepdim=True)
    lse = torch.where(any_sw, lse, torch.full_like(lse, float("-inf")))
    return torch.cat([flat[:, :k + 1], lse], dim=-1), sw


# ------------------------------------------------------------------------------------- labels
def flat_intent_targets(fi: FlatIntentInputs, kind: torch.Tensor, num: torch.Tensor,
                        switch_slot: torch.Tensor, switch_species: torch.Tensor,
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """The flat target [B] (a column, or `INTENT_IGNORE`) and its label class [B] (`LABEL_*`), from
    the Rust intent label (`opp_action_kind` 0 MOVE / 1 SWITCH / 2 UNKNOWN, `opp_action_num`,
    `opp_switch_slot` — the REVEALED slot or < 0 —, `opp_switch_species`). Static: no host read, no
    boolean-mask indexing (the learner micro-step compiles it). Every match is a structural equality
    on integer ids; a target whose candidate is not live is masked (the `reach` rule)."""
    K = fi.k
    F = fi.live.shape[1]
    kind = kind.long().reshape(-1)
    num = num.long().reshape(-1)
    slot = switch_slot.long().reshape(-1)
    spc = switch_species.long().reshape(-1)
    ign = torch.full_like(num, INTENT_IGNORE)
    # ---- MOVE: a seat (first match), else OTHER_move when it is a member, else unmodeled
    typed_hp = (num >= _TYPED_HP_LO) & (num <= _TYPED_HP_HI)
    hit = fi.seat_on & ((fi.seat_nums == num[:, None]) | (fi.hp_seat & typed_hp[:, None]))
    seat = hit.to(torch.int32).argmax(dim=-1)
    M = fi.beyond.shape[1]
    in_range = (num > 0) & (num < M)
    member = fi.beyond.gather(1, num.clamp(min=0, max=M - 1)[:, None]).squeeze(1) & in_range
    move_tgt = torch.where(hit.any(-1), seat,
                           torch.where(member, torch.full_like(num, other_move_col(K)), ign))
    # ---- SWITCH: a revealed slot, else the hypothesis slot holding that species, else the tail
    revealed = (slot >= 0) & (slot < TEAM_SIZE)
    hyp_hit = fi.slot_is_hypothesis & (fi.slot_species == spc[:, None]) & (spc > 0)[:, None]
    hyp_j = hyp_hit.to(torch.int32).argmax(dim=-1)
    S = fi.in_tail.shape[1]
    tail = (fi.in_tail.gather(1, spc.clamp(min=0, max=S - 1)[:, None]).squeeze(1)
            & (spc > 0) & (spc < S))
    sw_tgt = torch.where(revealed, slot.clamp(min=0, max=TEAM_SIZE - 1) + slot_col(K),
                         torch.where(hyp_hit.any(-1), hyp_j + slot_col(K),
                                     torch.where(tail, torch.full_like(num, other_species_col(K)), ign)))
    tgt = torch.where(kind == 0, move_tgt, torch.where(kind == 1, sw_tgt, ign))
    reach = fi.live.gather(1, tgt.clamp(min=0, max=F - 1)[:, None]).squeeze(1)
    tgt = torch.where((tgt >= 0) & reach, tgt, ign)
    choice = (kind == 0) | (kind == 1)
    cls = torch.where(tgt < 0, torch.where(choice, torch.full_like(tgt, LABEL_UNMODELED),
                                           torch.full_like(tgt, LABEL_NONCHOICE)),
                      torch.where(tgt < K, torch.full_like(tgt, LABEL_SEAT),
                                  torch.where(tgt == other_move_col(K), torch.full_like(tgt, LABEL_OTHER_MOVE),
                                              torch.where(tgt == other_species_col(K),
                                                          torch.full_like(tgt, LABEL_OTHER_SPECIES),
                                                          torch.full_like(tgt, LABEL_SLOT)))))
    return tgt, cls


def render_flat(probs: torch.Tensor, fi_row: Tuple[int, torch.Tensor, torch.Tensor],
                move_name: Callable[[int], Optional[str]], species_name: Callable[[int], Optional[str]],
                top: int = 6) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """The flat pointer as two ranked lists of NAMED options (the interpretability deliverable,
    `render_alpha`'s twin): ``(moves, switches)`` — moves = the named seats, ``"OTHER move"`` and
    ``"SWITCH"`` (the total switch mass); switches = the named slot targets and ``"OTHER species"``,
    each with its share of α_SWITCH. ``fi_row`` = (K, live [F], cand_ids [F]) for ONE row."""
    k, live, ids = fi_row
    p = [float(x) for x in probs]
    moves: List[Dict[str, Any]] = []
    sw: List[Dict[str, Any]] = []
    for c in range(k):
        if bool(live[c]):
            moves.append({"name": move_name(int(ids[c])) or f"move#{int(ids[c])}", "p": p[c]})
    if bool(live[other_move_col(k)]):
        moves.append({"name": "OTHER move", "p": p[other_move_col(k)]})
    a_sw = sum(p[slot_col(k):])
    moves.append({"name": "SWITCH", "p": a_sw})
    for j in range(TEAM_SIZE):
        c = slot_col(k, j)
        if bool(live[c]):
            sw.append({"slot": j, "species": species_name(int(ids[c])), "p": p[c] / a_sw if a_sw > 0 else 0.0})
    if bool(live[other_species_col(k)]):
        c = other_species_col(k)
        sw.append({"slot": None, "species": "OTHER species", "p": p[c] / a_sw if a_sw > 0 else 0.0})
    moves.sort(key=lambda r: -float(r["p"]))
    sw.sort(key=lambda r: -float(r["p"]))
    return moves[:top], sw[:4]


__all__ = ["FlatIntentHead", "FlatIntentInputs", "FlatConsumerOps", "append_other", "flat_candidates", "flat_intent_targets",
           "compat_intent_logits", "render_flat", "flat_width", "other_move_col", "slot_col",
           "other_species_col", "LABEL_SEAT", "LABEL_OTHER_MOVE", "LABEL_SLOT", "LABEL_OTHER_SPECIES",
           "LABEL_UNMODELED", "LABEL_NONCHOICE"]
