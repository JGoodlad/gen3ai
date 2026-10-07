"""X5's T0 HYPOTHESIS BUILDER — concrete opponent-species hypotheses with FIXED-MASS presence, plus
OTHER (`gen3_x5_hypothesis_set_v1`; `designs/endstate/design_x5_belief_tokens.md` §3.1–§3.3, build
unit U2).

WHAT IT BUILDS, at every decision, from the current observation (no recurrent state):

* **Species presence** (§3.2). Scores ``a_s = log P_T0(s | revealed) + δ_θ(s | ctx)`` over the
  candidate set ``V`` = the dex-row table's ``valid`` nums (the 386 base-form species at nums 1–386)
  MINUS the revealed nums (finding F-X5-21: the T0 prior's axis is ``[400]`` and floors the 13
  phantom nums 387–399 at 1e-4; they are not candidates). ``V`` is STRUCTURAL — a non-candidate has
  ``π = 0`` and no logit, never a finite Species-Clause logit.
  ``π_s = σ(a_s + τ)``, ``τ`` the root of ``Σ_{s∈V} σ(a_s + τ) = k``, ``k = 6 − r`` — the logistic
  fixed-size marginals (Amos, Koltun & Kolter 2019, LML, Eq. 3–4; the I-projection of independent
  Bernoulli presences onto "expected count = k", Csiszár 1975).
* **δ_θ**, the learned, state-dependent correction: a Deep-Sets sum-pool (Zaheer et al. 2017) over
  the REVEALED opponent role tokens ⊕ a projection of the global token's raw inputs → a 2-layer MLP
  → species logits. Its last layer is zero-initialised, so at a cold start ``π`` is the Smogon
  prior's fixed-size marginal exactly.
* **One stable ordering** (§3.1): a STABLE ascending argsort of ``−π`` over candidates laid out in
  num order, so equal ``π`` resolves to the LOWER number. ``torch.topk`` does not specify tie order
  and is not used for selection anywhere here.
* **The hypotheses**: the top-``k`` species by that order. Hypothesis rank ``j`` sits in the ``j``-th
  hidden opponent slot, in slot order (``slot_species``), and carries its dex-table row
  (``slot_rows``, U1's table) — a POPULATED mon of a KNOWN species (species-known 1, HP 1.0,
  recency 1.0). Only ``hypothesis_marker`` (U3) and ``ExtractorContext.opp_addressable`` tell it
  apart from a revealed mon; nothing may read the species-known or HP cells to decide.
* **OTHER_species** (§3.3): the tail's mass ``Σ_tail π`` (summed directly), its log-mass
  ``logsumexp_tail(logsigmoid(a + τ))`` (finite whenever the tail is non-empty, so no floor), and its
  embedding — a learned "rest" vector plus a linear map of the π-weighted mean SPECIES embedding over
  the WHOLE tail (U3, ORCHESTRATOR decision on F-X5-23: U2's "next 32" cut was a second selection
  boundary; the whole-tail mean removes it and matches M3 (c)'s physics, which also reads the whole
  renormalised tail ``P_tail = π·[tail] / Σ_tail π``, exposed as `other_tail_probs`). Masked
  STRUCTURALLY: iff ``k = 0`` (r = 6) or the tail is empty.
* **The opponent active's moves** (§3.1 / §3.2, "Moves"): the same construction at
  ``k_m = 4 − r_m`` over the active species' LEGAL moves minus its revealed moves; the ``K`` seats are
  the ``r_m`` revealed moves (pinned at 1) then the top ``K − r_m`` unrevealed by ``π_m``; OTHER_move
  is the mass beyond the seats (it re-uses the active's E5 tail seat, which U3 wires).

THE FIXED-SIZE CONSTRUCTION (`fixed_size_tau`).

* **Bracket (provable).** With ``n = |V|``: at ``τ_lo = log(k/(n−k)) − a_max`` every term is at most
  ``k/n`` so ``Σ ≤ k``; at ``τ_hi = log(k/(n−k)) − a_min`` every term is at least ``k/n`` so ``Σ ≥ k``.
  ``Σσ(a + τ)`` is strictly increasing in ``τ``, so the root is unique and inside.
* **Bisection: ``BISECTION_ITERS`` = 64 fixed iterations**, the same count in every dtype, under
  ``no_grad``, no data-dependent exit: a static graph and a deterministic result. 64 halvings of a
  bracket of width ``a_max − a_min`` reach ``width · 2^-64`` — below fp64's resolution of ``τ`` for
  any realistic width (the prior's is ≤ 45.2 nats, MEASURED), and fp32 stops moving at its own
  resolution after ~25 halvings.
* **Structural cases.** ``k = 0``: ``π ≡ 0``, nothing is read from the bisection, every seat and
  OTHER are masked. ``k = n``: ``π ≡ 1`` on ``V`` and the row is EXCLUDED from the BCE (it carries no
  information). The bisection runs on a SAFE stand-in (``k`` clamped into ``[1, n−1]``) for those
  rows so no NaN or inf is ever formed, and its answer is discarded.
* **τ needs no gradient.** ``∂BCE/∂τ = Σ_s (π_s − y_s) = k − k = 0`` whenever the label set counts
  exactly ``k`` true unseen species, so computing ``τ`` under ``no_grad`` gives the EXACT implicit
  gradient (LML's implicit backward is not needed). A row whose labels do not count ``k`` (or name a
  non-candidate) is excluded from the BCE and reported (`presence_label_mismatch`).

THE GRADIENT PATH (review M10). Every tensor a POLICY or CRITIC consumer may read — ``pi``,
``log_pi``, OTHER's mass, log-mass and embedding inputs, the hypothesis selection — is computed from
DETACHED logits. The ONLY graph-carrying output is ``species_logits`` (``a + τ``), and its only
reader is the presence BCE: δ_θ learns from the BCE alone. ``hypothesis_set_test`` pins that a loss
built from every consumer-facing output puts exactly zero gradient into δ_θ.

RULE 8 AT THE SELECTION BOUNDARY (§3.1). Which species takes the last seat is a discontinuous
function of ``π``. Every check that compares two computations of the selection EXCLUDES a row whose
``π`` at a selection boundary differs by less than ``SELECTION_TIE_EPS`` (1e-6, absolute) across it,
and reports the excluded count: `near_tie_rows`. Since U3 the boundaries are EVERY adjacent pair of the
order up to and across the seat boundary (``i``-th vs ``(i+1)``-th for ``i < k``): hypothesis rank
``j`` sits in the ``j``-th hidden slot, so a swap INSIDE the selection moves a token between slots —
also a log π change once the tokens are read (U3). Likewise the move seats (their seat order is the
E4 seat order). OTHER's tail mean no longer has a cutoff (the whole tail, U3).

WHERE IT IS READ (U3, `agents.model.hypothesis_tokens`). The species half (`species_set`) runs at T0
BEFORE the move belief: its dex rows are encoded by THE `pokemon_encoder` (+ ``hypothesis_marker``)
into the hidden opponent slots, so every T0 belief head reads a species-specific token; the move
group (`with_moves`) is added after the move belief. The log-π key bias, OTHER's token and the class-E
pools read ``slot_log_pi`` / ``other_log_mass`` / ``other_token``. The flat pointer is U4.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn.functional as F

from agents.model.arch_constants import D_MODEL, HYPOTHESIS_DELTA_HIDDEN, HYPOTHESIS_INIT_SEED
from agents.observation.constants import TEAM_SIZE
from agents.observation.moves import HIDDEN_POWER_MOVE_NUM

#: The bisection's FIXED iteration count (§3.2): the same in every dtype, no data-dependent exit.
BISECTION_ITERS = 64
#: §3.1's rule-8 exclusion: a row whose π at a selection boundary differs by less than this
#: (ABSOLUTE, π ∈ (0, 1)) across the boundary is excluded from every two-computation comparison.
SELECTION_TIE_EPS = 1e-6
#: The log-presence of a MASKED key — the key-padding addend the transformer already uses (§3.5).
MASKED_LOG_PRESENCE = -1e9
#: The 16 typed Hidden Power channels (`compose_typed_hp`): 355 + t.
TYPED_HP_NUMS = tuple(range(355, 371))
#: Seats per group: the hypothesis seats are the opponent's 6 team slots; the move seats are E4's K.
N_TEAM_SEATS = TEAM_SIZE
#: The opponent active's move group mass (§3.1: "mass 4").
MOVE_GROUP_MASS = 4


# ============================================================================ the pure construction

def fixed_size_tau(scores: torch.Tensor, cand: torch.Tensor, k: torch.Tensor,
                   n_iter: int = BISECTION_ITERS) -> torch.Tensor:
    """``τ`` [B] with ``Σ_{cand} σ(scores + τ) = k`` (LML Eq. 4), by a fixed-step bisection on the
    provable bracket in the module docstring. ``scores`` [B, N] float; ``cand`` [B, N] bool; ``k`` [B]
    integer. NO gradient (the exact implicit gradient is zero, module docstring). Rows with
    ``k == 0`` or ``k >= n`` (the structural cases) get a finite, MEANINGLESS ``τ`` — callers mask
    them; nothing here forms a NaN or an inf for them."""
    with torch.no_grad():
        s = scores.detach()
        dt = s.dtype
        n = cand.sum(-1)                                                        # [B]
        # The SAFE stand-in for the structural rows: k into [1, n-1] (n >= 2), else a dummy row.
        n_safe = n.clamp(min=2)
        k_safe = torch.minimum(k.clamp(min=1), n_safe - 1)
        has = n > 0
        a_max = torch.where(cand, s, torch.full_like(s, -math.inf)).amax(-1)
        a_min = torch.where(cand, s, torch.full_like(s, math.inf)).amin(-1)
        a_max = torch.where(has, a_max, torch.zeros_like(a_max))
        a_min = torch.where(has, a_min, torch.zeros_like(a_min))
        base = torch.log(k_safe.to(dt)) - torch.log((n_safe - k_safe).to(dt))  # log(k/(n-k))
        lo = base - a_max
        hi = base - a_min
        k_t = k_safe.to(dt)
        zero = torch.zeros((), dtype=dt, device=s.device)
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            total = torch.where(cand, torch.sigmoid(s + mid.unsqueeze(-1)), zero).sum(-1)
            above = total > k_t
            hi = torch.where(above, mid, hi)
            lo = torch.where(above, lo, mid)
        return 0.5 * (lo + hi)


@dataclass(frozen=True)
class Presence:
    """The fixed-size presence of one candidate group (species, or the active's moves).

    ``logits`` is the ONLY graph-carrying field (``a + τ``; the BCE's exact logit). Every other field
    is DETACHED — the M10 rule: a policy / critic consumer reads ``pi`` / ``log_pi``, never
    ``logits``."""

    logits: torch.Tensor      # [B,N] a + τ (graph: into δ_θ; for the presence BCE ONLY)
    pi: torch.Tensor          # [B,N] σ(a + τ) on live candidates, 1 on a k = n row's candidates, 0 off V
    log_pi: torch.Tensor      # [B,N] logsigmoid(a + τ); 0 where π = 1; MASKED_LOG_PRESENCE off V / k = 0
    cand: torch.Tensor        # [B,N] bool — V (structural)
    k: torch.Tensor           # [B] long — the group mass
    n: torch.Tensor           # [B] long — |V|
    live: torch.Tensor        # [B] bool — 0 < k < n: the construction ran and the BCE may read the row
    full: torch.Tensor        # [B] bool — k >= n > 0: π ≡ 1 on V (structural; excluded from the BCE)


def fixed_mass_presence(scores: torch.Tensor, cand: torch.Tensor, k: torch.Tensor,
                        n_iter: int = BISECTION_ITERS) -> Presence:
    """The logistic fixed-size marginals ``π = σ(scores + τ)``, Σ = k over ``cand``, with the
    structural k = 0 / k = n branches. ``scores`` may carry a graph (δ_θ); ``τ`` never does."""
    k = k.long()
    n = cand.sum(-1)
    tau = fixed_size_tau(scores, cand, k, n_iter)
    logits = scores + tau.unsqueeze(-1)                                         # graph → scores only
    live = (k > 0) & (k < n)
    full = (k >= n) & (n > 0)
    lg = logits.detach()
    live_c = cand & live.unsqueeze(-1)
    full_c = cand & full.unsqueeze(-1)
    one = torch.ones((), dtype=lg.dtype, device=lg.device)
    zero = torch.zeros((), dtype=lg.dtype, device=lg.device)
    pi = torch.where(live_c, torch.sigmoid(lg), torch.where(full_c, one, zero))
    masked = torch.full((), MASKED_LOG_PRESENCE, dtype=lg.dtype, device=lg.device)
    log_pi = torch.where(live_c, F.logsigmoid(lg), torch.where(full_c, zero, masked))
    return Presence(logits=logits, pi=pi, log_pi=log_pi, cand=cand, k=k, n=n, live=live, full=full)


class SetCuts(Tuple[int, ...]):
    """A caller's declaration that it reads the first ``max(cuts)`` positions of a `stable_order` as a SET
    at each cut — every consumer of the selected prefix is permutation-invariant over it, so only WHICH
    candidates sit before each cut can move anything, never their order inside it
    (`gen3_behaviour_tie_consumed_v1`; the blob arm's ``topk`` over the same consumers is judged the same
    way, at its k-th vs (k+1)-th value)."""

    def __new__(cls, cuts: "Tuple[int, ...]") -> "SetCuts":
        return super().__new__(cls, tuple(sorted({int(c) for c in cuts})))


def stable_order(key: torch.Tensor, cand: torch.Tensor,
                 consumed: "Optional[torch.Tensor | SetCuts]" = None) -> torch.Tensor:
    """[B,N] candidate indices by ``key`` DESCENDING, ties to the LOWER index, non-candidates last.

    A STABLE ascending argsort of ``−key`` (non-candidates keyed ``+inf``): stability is what makes
    "equal ``π`` → lower number first" a guarantee rather than an implementation detail. Negation is
    exact in floating point, so ``−key`` orders exactly as ``key`` reversed.

    ``consumed`` changes NOTHING this returns: it is the caller's declaration of how it READS the order,
    read by the K9(b) tie-margin recorder (`selection_sites.Rule.consumed`, `gen3_behaviour_tie_consumed_v1`)
    so that a tie only counts where it can move something:

    * a long tensor of shape ``key.shape[:-1]`` — that many leading positions are read IN ORDER, the
      rest only as a set (the adjacent pairs up to and across it are boundaries);
    * a `SetCuts` — the prefix is read as a SET at each cut (only the pair straddling a cut is one);
    * None — every adjacent pair of the recorder's head is a boundary (the conservative default)."""
    neg = torch.where(cand, -key, torch.full_like(key, math.inf))
    return torch.argsort(neg, dim=-1, stable=True)


def ranks_of(order: torch.Tensor) -> torch.Tensor:
    """The inverse permutation: ``rank[b, order[b, i]] = i`` [B,N] long."""
    pos = torch.arange(order.shape[-1], device=order.device).expand_as(order)
    return torch.empty_like(order).scatter_(-1, order, pos)


def order_gap(sorted_pi: torch.Tensor, at: torch.Tensor, n_avail: torch.Tensor,
              start: Optional[torch.Tensor] = None) -> torch.Tensor:
    """[B] the SMALLEST gap ``sorted_pi[i] − sorted_pi[i+1]`` over ``start <= i < at`` with ``i + 1 <
    n_avail`` — every adjacent pair of the order up to and across the selection boundary ``at`` (the
    count selected). ``+inf`` where there is no such pair. ``start`` (default 0) skips a structural
    prefix (the move group's revealed seats, pinned ahead of every π). U3: a swap inside the selection
    moves a hypothesis between slots (or a move between E4 seats), so those pairs are boundaries too."""
    N = sorted_pi.shape[-1]
    i = torch.arange(N - 1, device=sorted_pi.device).unsqueeze(0)              # [1,N-1]
    lo = torch.zeros_like(at) if start is None else start
    use = (i >= lo.unsqueeze(-1)) & (i < at.unsqueeze(-1)) & (i + 1 < n_avail.unsqueeze(-1))
    gaps = sorted_pi[:, :-1] - sorted_pi[:, 1:]
    gaps = torch.where(use, gaps, torch.full_like(gaps, math.inf))
    return gaps.amin(-1)


def boundary_gap(sorted_pi: torch.Tensor, at: torch.Tensor, n_avail: torch.Tensor) -> torch.Tensor:
    """[B] the gap ``sorted_pi[at-1] − sorted_pi[at]`` across the selection boundary ``at`` (the count
    selected), or ``+inf`` where there is no boundary (``at <= 0`` or ``at >= n_avail``: nothing on one
    side). ``sorted_pi`` [B,N] is π in the selection order."""
    N = sorted_pi.shape[-1]
    has = (at > 0) & (at < n_avail)
    i_hi = (at - 1).clamp(0, N - 1).unsqueeze(-1)
    i_lo = at.clamp(0, N - 1).unsqueeze(-1)
    gap = (sorted_pi.gather(-1, i_hi) - sorted_pi.gather(-1, i_lo)).squeeze(-1)
    return torch.where(has, gap, torch.full_like(gap, math.inf))


# ============================================================================ the output

@dataclass(frozen=True)
class HypothesisSet:
    """One forward's X5 hypothesis set (the `last_hypothesis` stash; U3's input). Every tensor is
    DETACHED except ``species.logits`` (the presence BCE's logit)."""

    species: Presence                   # over the [max_species] num axis
    order: torch.Tensor                 # [B,S] long — species nums in the ONE selection order
    rank: torch.Tensor                  # [B,S] long — the inverse of `order`
    hyp_species: torch.Tensor           # [B,6] long — hypothesis rank j's num (0 when j >= k)
    hyp_live: torch.Tensor              # [B,6] bool — j < k
    hyp_pi: torch.Tensor                # [B,6] its presence (0 when not live)
    slot_species: torch.Tensor          # [B,6] long — per OPP SLOT: the hypothesis num on a hidden slot, 0 on a revealed one
    slot_is_hypothesis: torch.Tensor    # [B,6] bool — the hidden slots (each holds exactly one hypothesis)
    slot_log_pi: torch.Tensor           # [B,6] the slot's key log-presence: hypothesis log π; 0 on a revealed slot
    slot_rows: torch.Tensor             # [B,6,POKEMON_FULL_DIM] the hypothesis's dex row; 0 on a revealed slot
    other_mass: torch.Tensor            # [B] Σ_tail π
    other_log_mass: torch.Tensor        # [B] logsumexp_tail log π; MASKED_LOG_PRESENCE when masked
    other_live: torch.Tensor            # [B] bool — k > 0 and the tail is non-empty (structural)
    other_tail_probs: torch.Tensor      # [B,S] the RENORMALISED tail P_tail = π·[tail] / Σ_tail π (0 when masked)
    other_tail_mean: torch.Tensor       # [B,species_emb] π-weighted mean species embedding over the WHOLE tail
    other_token: torch.Tensor           # [B,D_MODEL] the OTHER_species embedding (rest + map(tail mean))
    other_any: torch.Tensor             # [B] P(at least one tail species present) = 1 − Π_tail(1 − π) ∈ [0,1]
    species_tie_gap: torch.Tensor       # [B] smallest adjacent π gap up to and across the seat boundary (+inf: none)
    moves: Optional["MovePresence"]     # the opponent ACTIVE's move group (None: no move belief)
    # U3 part 3 (the op's opponent-MON axis): a slot's presence at a max-type site over MONS (§9 M2 = C)
    # — 1 on a revealed slot, the hypothesis's π on a hidden one (the same gather as `slot_log_pi`).
    slot_pi: Optional[torch.Tensor] = None          # [B,6]
    # U3 part 3: the smallest gap at a PER-MON move-selection boundary (`hypothesis_tokens.OpRoster`:
    # the attacker candidates of every live opponent mon and the bench E5 tail cut); +inf: none.
    slot_moves_tie_gap: Optional[torch.Tensor] = None   # [B]


@dataclass(frozen=True)
class MovePresence:
    """The opponent ACTIVE's move group: mass 4, ``K`` seats (revealed first, then the top unrevealed
    by π_m), OTHER_move beyond the seats. Every tensor DETACHED (π_m enters no loss in U2)."""

    presence: Presence                  # over the [max_moves] num axis (unrevealed candidates)
    seat_nums: torch.Tensor             # [B,K] long — the seat move nums (0 when not live)
    seat_live: torch.Tensor             # [B,K] bool
    seat_revealed: torch.Tensor         # [B,K] bool — a revealed move (π pinned at 1)
    seat_pi: torch.Tensor               # [B,K] 1 on a revealed seat, π_m on an unrevealed one, 0 off
    r: torch.Tensor                     # [B] long — the active's revealed (distinct) move count
    other_mass: torch.Tensor            # [B] Σ π_m beyond the seats
    other_log_mass: torch.Tensor        # [B] its logsumexp log-mass; MASKED_LOG_PRESENCE when masked
    other_live: torch.Tensor            # [B] bool — k_m > 0 and some unrevealed candidate is beyond the seats
    beyond: torch.Tensor                # [B,M] bool — OTHER_move's members (unrevealed candidates past the seats)
    tie_gap: torch.Tensor               # [B] smallest adjacent π_m gap among the unrevealed seats and across the seat boundary


def near_tie_rows(hs: HypothesisSet, eps: float = SELECTION_TIE_EPS) -> torch.Tensor:
    """[B] bool — the rows §3.1's rule-8 exclusion drops from every check that compares two
    computations of the selection (eager vs compiled, CPU vs CUDA, a golden, a reader): a π gap under
    ``eps`` between ANY adjacent pair of the order up to and across a selection boundary (the
    hypothesis seats and their slot order; the move seats and their seat order). Callers report
    ``int(near_tie_rows(hs).sum())`` beside the verdict."""
    bad = hs.species_tie_gap < eps
    if hs.moves is not None:
        bad = bad | (hs.moves.tie_gap < eps)
    if hs.slot_moves_tie_gap is not None:      # U3 part 3: the per-mon attacker / E5-tail cuts
        bad = bad | (hs.slot_moves_tie_gap < eps)
    return bad


# ============================================================================ candidates

def species_candidates(valid: torch.Tensor, opp_species_ids: torch.Tensor,
                       opp_believed_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(cand [B,S] bool, k [B] long). ``V`` = ``valid`` (the dex-row table's mask: nums 1–386) minus the
    REVEALED nums — a slot is revealed iff it is not believed AND carries a num > 0 (the T0 prior's own
    evidence rule, `species_team_prior_logits`). ``k`` = the hidden-slot count ``6 − r``. Formes share
    their base's num, so a revealed forme removes its base num exactly (the num-keyed rule)."""
    S = valid.shape[0]
    ids = opp_species_ids.clamp(0, S - 1).long()                                # [B,6]
    revealed = (ids > 0) & (~opp_believed_mask.bool())
    hit = torch.zeros(ids.shape[0], S, dtype=torch.bool, device=ids.device)
    hit.scatter_(1, ids, revealed)        # a hidden slot writes False at index 0 (never valid)
    cand = valid.unsqueeze(0) & ~hit
    k = opp_believed_mask.bool().sum(-1).long()
    return cand, k


def move_candidates(legal_row: torch.Tensor, valid_moves: torch.Tensor,
                    revealed_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(cand [B,M] bool, revealed [B,M] bool, r [B] long) for one mon's move group.

    ``V_m`` = the mon's LEGAL moves (the move prior's own legality, `build_move_legality`) ∩ the
    nums any species can learn, minus the revealed moves, minus the typeless Hidden Power channel
    237 (BP 0 — a bookkeeping channel, never a candidate; `mask_typeless_hp`), minus the 16 typed
    HP channels when Hidden Power is REVEALED (they are then that one revealed move's type
    distribution, not further candidates). All structural."""
    M = legal_row.shape[-1]
    ids = revealed_ids.clamp(0, M - 1).long()                                   # [B,4]
    valid_id = revealed_ids > 0
    revealed = torch.zeros(ids.shape[0], M, dtype=torch.bool, device=ids.device)
    revealed.scatter_(1, ids, valid_id)
    r = revealed.sum(-1).long()
    num = torch.arange(M, device=ids.device)
    hp_rev = revealed[:, HIDDEN_POWER_MOVE_NUM]                                  # [B]
    typed = (num >= TYPED_HP_NUMS[0]) & (num <= TYPED_HP_NUMS[-1])              # [M]
    cand = (legal_row & valid_moves.unsqueeze(0) & ~revealed
            & (num != HIDDEN_POWER_MOVE_NUM).unsqueeze(0)
            & ~(typed.unsqueeze(0) & hp_rev.unsqueeze(-1)))
    return cand, revealed, r


# ============================================================================ the module

class IsolatedLinear(torch.nn.Module):
    """``y = x Wᵀ + b`` that is DELIBERATELY not a ``torch.nn.Linear``.

    SB3's ``_build`` re-initialises every ``nn.Linear`` under the features extractor with an
    orthogonal init drawn from the GLOBAL RNG. X5's modules are built from a PRIVATE seed inside
    ``fork_rng`` (``HYPOTHESIS_INIT_SEED``) so the ``fixed_mass`` arm leaves every NON-X5 parameter's
    initial bytes equal to the ``blob`` arm's (design §6 item 2) — and an ``nn.Linear`` here would be
    re-drawn by SB3 from the global stream, shifting every later draw (the mlp_extractor, the heads).
    The init is ``nn.Linear``'s own (Kaiming-uniform weight, uniform bias), or exact zeros. ``bias=False``
    builds no bias (``nn.Linear``'s convention; the weight draws exactly as with one, the bias draw being
    the last)."""

    bias: Optional[torch.nn.Parameter]

    def __init__(self, in_features: int, out_features: int, zero: bool = False, bias: bool = True) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.empty(out_features)) if bias else None
        if zero:
            torch.nn.init.zeros_(self.weight)
            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)
        else:
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                bound = 1.0 / math.sqrt(in_features) if in_features > 0 else 0.0
                torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


def build_move_legality(n_species: int, n_moves: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """(legal [S,M] bool, valid_moves [M] bool) from the move prior's OWN legality rule
    (`belief_tables.build_move_prior_logits`: an illegal cell is exactly ``logit(_ILLEGAL_PROB)``, a
    legal one at least ``logit(_MIN_PRIOR_FLOOR)``). The comparison is on a CONSTANT TABLE against a
    cutoff halfway between the two classes, so it is exact. ``valid_moves`` = the nums some covered
    species can learn (the unknown-species sentinel row, which is flat ``floor`` everywhere, is
    excluded from that union — it would make every num 'valid')."""
    from agents.gen3_data import species as gs
    from agents.model.belief_tables import (_ILLEGAL_PROB, _MIN_PRIOR_FLOOR,
                                            build_move_prior_logits)

    logits = build_move_prior_logits(n_species, n_moves)
    cut = 0.5 * (math.log(_ILLEGAL_PROB / (1 - _ILLEGAL_PROB))
                 + math.log(_MIN_PRIOR_FLOOR / (1 - _MIN_PRIOR_FLOOR)))
    legal = logits > cut
    covered = torch.zeros(n_species, dtype=torch.bool)
    for sid in gs.base_form_ids():
        num = gs.species_data(sid).num
        if 0 < num < n_species:
            covered[num] = True
    valid_moves = (legal & covered.unsqueeze(-1)).any(0)
    valid_moves[0] = False
    return legal, valid_moves


class HypothesisBuilder(torch.nn.Module):
    """T0 RESOLVE under X5: δ_θ, the fixed-size presence, the hypothesis
    selection, OTHER, and the active's move group (module docstring). Built ONLY in that arm, from a
    private seed, appended after every other module (no parameter position moves)."""

    if TYPE_CHECKING:  # registered buffers — declared so every read is a Tensor (typing.md)
        dex_rows: torch.Tensor
        species_valid: torch.Tensor
        move_legal: torch.Tensor
        move_valid: torch.Tensor

    def __init__(self, layout: dict, global_input_dim: int, n_move_seats: int) -> None:
        super().__init__()
        from agents.model.hypothesis_dex_rows import load_hypothesis_dex_rows

        S = int(layout["max_species"])
        M = int(layout["max_moves"])
        self.n_species, self.n_moves = S, M
        # K seats (E4's `entity_topk_seats`; production 6). With K < 4 a fourth revealed move has no
        # seat — the move group still sums correctly, and U3's E4 re-wiring must refuse that config.
        self.n_move_seats = int(n_move_seats)
        table = load_hypothesis_dex_rows(S)
        # U1 hand-off: READ-ONLY arrays — copy before `from_numpy`; NON-persistent (data-derived,
        # never a saved weight; a pinned run isolates the committed artifact with its code).
        self.register_buffer("dex_rows", torch.from_numpy(table.rows.copy()), persistent=False)
        self.register_buffer("species_valid", torch.from_numpy(table.valid.copy()), persistent=False)
        legal, valid_moves = build_move_legality(S, M)
        self.register_buffer("move_legal", legal, persistent=False)
        self.register_buffer("move_valid", valid_moves, persistent=False)
        species_emb_dim = int(layout["species_embedding_dim"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(HYPOTHESIS_INIT_SEED)
            # δ_θ: Deep-Sets sum-pool over the REVEALED opponent role tokens ⊕ a projection of the
            # global token's raw inputs → 2-layer MLP → species logits (last layer ZERO: cold start ==
            # the Smogon prior's fixed-size marginal exactly).
            self.delta_global = IsolatedLinear(global_input_dim, D_MODEL)
            self.delta_norm = torch.nn.LayerNorm(2 * D_MODEL)
            self.delta_hidden = IsolatedLinear(2 * D_MODEL, HYPOTHESIS_DELTA_HIDDEN)
            self.delta_out = IsolatedLinear(HYPOTHESIS_DELTA_HIDDEN, S, zero=True)
            # OTHER_species: a learned rest vector + a linear map of the tail-mean species embedding.
            self.other_rest = torch.nn.Parameter(torch.randn(D_MODEL) * 0.02)
            self.other_map = IsolatedLinear(species_emb_dim, D_MODEL)
            # U3 (§3.4): the learned marker added to a hypothesis seat's encoded token, as E5's
            # `tail_marker` is (the token-type table is NOT grown — that would change every
            # state_dict). Drawn LAST in this private stream, so no earlier X5 init byte moves.
            self.hypothesis_marker = torch.nn.Parameter(torch.randn(D_MODEL) * 0.02)
        self.detach_read = False      # stamped by `_stamp_belief_grad_flags` (belief_grad_mode)

    # ------------------------------------------------------------------ δ_θ
    def delta(self, opp_role_tokens: torch.Tensor, revealed: torch.Tensor,
              global_input: torch.Tensor) -> torch.Tensor:
        """[B,S] the learned correction. ``opp_role_tokens`` [B,6,D] (pre-transformer, pre-belief),
        ``revealed`` [B,6] bool, ``global_input`` [B,G] (the TeamTransformer global token's raw input)."""
        tok = opp_role_tokens.detach() if self.detach_read else opp_role_tokens
        pooled = (tok * revealed.unsqueeze(-1).to(tok.dtype)).sum(1)                # [B,D] Deep Sets
        h = torch.cat([pooled, self.delta_global(global_input)], dim=-1)            # [B,2D]
        h = torch.relu(self.delta_hidden(self.delta_norm(h)))
        return self.delta_out(h)                                                     # type: ignore[no-any-return]

    # ------------------------------------------------------------------ forward
    def forward(self, t0_log_prior: torch.Tensor, opp_species_ids: torch.Tensor,
                opp_believed_mask: torch.Tensor, opp_role_tokens: torch.Tensor,
                global_input: torch.Tensor, species_embedding: torch.nn.Embedding,
                active_move_logits: Optional[torch.Tensor] = None,
                active_species: Optional[torch.Tensor] = None,
                active_revealed_moves: Optional[torch.Tensor] = None) -> HypothesisSet:
        """The whole set in one call (`species_set` then `with_moves`) — the readers' and the tests'
        entry. The extractor calls the two halves separately (U3): the species half must run BEFORE
        the move belief (its dex rows become the hidden slots' tokens), the move group AFTER it.
        ``t0_log_prior`` [B,S] log P_T0(s | revealed) (`species_team_prior_logits`);
        ``opp_species_ids`` [B,6]; ``opp_believed_mask`` [B,6] bool; ``opp_role_tokens`` [B,6,D];
        ``global_input`` [B,G]; the active's move group from ``active_move_logits`` [B,M] (the typed
        MoveBelief posterior at the opponent active), ``active_species`` [B] and
        ``active_revealed_moves`` [B,4] (all three or none)."""
        hs = self.species_set(t0_log_prior, opp_species_ids, opp_believed_mask, opp_role_tokens,
                              global_input, species_embedding)
        if active_move_logits is None:
            return hs
        assert active_species is not None and active_revealed_moves is not None
        return self.with_moves(hs, active_move_logits, active_species, active_revealed_moves)

    def with_moves(self, hs: HypothesisSet, active_move_logits: torch.Tensor,
                   active_species: torch.Tensor, active_revealed_moves: torch.Tensor) -> HypothesisSet:
        """``hs`` with the opponent active's move group attached (`move_group`)."""
        return dataclasses.replace(
            hs, moves=self.move_group(active_move_logits, active_species, active_revealed_moves))

    def species_set(self, t0_log_prior: torch.Tensor, opp_species_ids: torch.Tensor,
                    opp_believed_mask: torch.Tensor, opp_role_tokens: torch.Tensor,
                    global_input: torch.Tensor, species_embedding: torch.nn.Embedding) -> HypothesisSet:
        """The species half: δ_θ, the fixed-size presence, the one order, the hypotheses and their
        slots, OTHER (``moves`` is None)."""
        believed = opp_believed_mask.bool()
        cand, k = species_candidates(self.species_valid, opp_species_ids, believed)
        revealed_slots = (opp_species_ids > 0) & ~believed
        scores = t0_log_prior + self.delta(opp_role_tokens, revealed_slots, global_input)
        pres = fixed_mass_presence(scores, cand, k)
        pi = pres.pi                                                                 # detached
        # ranks >= k are OTHER's tail, read only as a set (`in_tail`, index-order sums): k positions are
        # consumed in order (gen3_behaviour_tie_consumed_v1)
        order = stable_order(pi, cand, consumed=k)
        rank = ranks_of(order)
        n_avail = pres.n
        sorted_pi = pi.gather(-1, order)
        # ---- the hypotheses: rank j < k, j < 6
        j = torch.arange(N_TEAM_SEATS, device=pi.device)
        hyp_live = j.unsqueeze(0) < k.unsqueeze(-1)                                  # [B,6]
        hyp_species = torch.where(hyp_live, order[:, :N_TEAM_SEATS], torch.zeros_like(order[:, :N_TEAM_SEATS]))
        hyp_pi = torch.where(hyp_live, sorted_pi[:, :N_TEAM_SEATS], torch.zeros_like(sorted_pi[:, :N_TEAM_SEATS]))
        hyp_log_pi = pres.log_pi.gather(-1, order[:, :N_TEAM_SEATS])
        # ---- hypothesis j → the j-th hidden slot, in slot order
        hid_rank = (torch.cumsum(believed.long(), dim=-1) - 1).clamp(min=0)          # [B,6]
        slot_species = torch.where(believed, hyp_species.gather(-1, hid_rank), torch.zeros_like(hid_rank))
        zero = torch.zeros((), dtype=pi.dtype, device=pi.device)
        slot_log_pi = torch.where(believed, hyp_log_pi.gather(-1, hid_rank), zero)
        slot_pi = torch.where(believed, hyp_pi.gather(-1, hid_rank), torch.ones_like(slot_log_pi))
        slot_rows = self.dex_rows[slot_species] * believed.unsqueeze(-1).to(self.dex_rows.dtype)
        # ---- OTHER_species: the tail (rank >= k) — structural
        in_tail = cand & (rank >= k.unsqueeze(-1)) & (k > 0).unsqueeze(-1)
        other_live = in_tail.any(-1)
        w_tail = torch.where(in_tail, pi, zero)                                      # [B,S] detached
        other_mass = w_tail.sum(-1)
        neg_inf = torch.full((), -math.inf, dtype=pi.dtype, device=pi.device)
        lse = torch.logsumexp(torch.where(in_tail, pres.log_pi, neg_inf), dim=-1)
        other_log_mass = torch.where(other_live, lse,
                                     torch.full_like(lse, MASKED_LOG_PRESENCE))
        # The renormalised tail (M3 (c), F-X5-23): the WHOLE tail, π-weighted — no second cutoff. A
        # live OTHER has Σ_tail π > 0 (π never reaches 0 on a live row), so the division is exact; a
        # masked one reads 0 (structural gate, `other_live`, never a float comparison).
        tiny = torch.finfo(pi.dtype).tiny
        p_tail = torch.where(other_live.unsqueeze(-1),
                             w_tail / other_mass.clamp(min=tiny).unsqueeze(-1), torch.zeros_like(w_tail))
        tail_mean = p_tail.to(species_embedding.weight.dtype) @ species_embedding.weight  # [B,E]
        other_token = (self.other_rest + self.other_map(tail_mean)) * other_live.unsqueeze(-1).to(tail_mean.dtype)
        # P(at least one tail species present) under the I-projection's independent Bernoullis
        # (ORCHESTRATOR, Tier-0 F4 (b)): 1 − Π(1 − π) = −expm1(Σ log1p(−π)) — in [0, 1], continuous,
        # and OTHER's MASS (an expected count, often > 1) never scales a threat directly.
        other_any = torch.where(other_live, -torch.expm1(torch.log1p(-w_tail).sum(-1)),
                                torch.zeros_like(other_mass))
        species_gap = order_gap(sorted_pi, k, n_avail)
        return HypothesisSet(
            species=pres, order=order, rank=rank, hyp_species=hyp_species, hyp_live=hyp_live,
            hyp_pi=hyp_pi, slot_species=slot_species, slot_is_hypothesis=believed,
            slot_log_pi=slot_log_pi, slot_rows=slot_rows, other_mass=other_mass,
            other_log_mass=other_log_mass, other_live=other_live, other_tail_probs=p_tail,
            other_tail_mean=tail_mean, other_token=other_token, other_any=other_any,
            species_tie_gap=species_gap, moves=None, slot_pi=slot_pi)

    def move_group(self, move_logits: torch.Tensor, species: torch.Tensor,
                   revealed_ids: torch.Tensor) -> MovePresence:
        """The opponent active's move group (module docstring). ``move_logits`` [B,M] is read
        DETACHED — π_m enters no loss in U2 (its consumers are U3's seats, M10)."""
        K = self.n_move_seats
        sp = species.clamp(0, self.n_species - 1).long()
        cand, revealed, r = move_candidates(self.move_legal[sp], self.move_valid, revealed_ids)
        k_m = (MOVE_GROUP_MASS - r).clamp(min=0)
        pres = fixed_mass_presence(move_logits.detach(), cand, k_m)
        pi = pres.pi
        # One order: revealed (key 2, structural — never a float compare with π) → unrevealed by π →
        # the rest last; ties to the lower num.
        key = torch.where(revealed, torch.full_like(pi, 2.0), pi)
        order = stable_order(key, cand | revealed)
        rank = ranks_of(order)
        n_seatable = (cand | revealed).sum(-1)
        jj = torch.arange(K, device=pi.device)
        seat_live = jj.unsqueeze(0) < torch.minimum(n_seatable, torch.full_like(n_seatable, K)).unsqueeze(-1)
        seat_nums = torch.where(seat_live, order[:, :K], torch.zeros_like(order[:, :K]))
        seat_revealed = jj.unsqueeze(0) < r.unsqueeze(-1)
        zero = torch.zeros((), dtype=pi.dtype, device=pi.device)
        one = torch.ones((), dtype=pi.dtype, device=pi.device)
        seat_pi = torch.where(seat_revealed, one,
                              torch.where(seat_live, pi.gather(-1, order[:, :K]), zero))
        beyond = cand & (rank >= K) & (k_m > 0).unsqueeze(-1)
        other_live = beyond.any(-1)
        other_mass = torch.where(beyond, pi, zero).sum(-1)
        neg_inf = torch.full((), -math.inf, dtype=pi.dtype, device=pi.device)
        lse = torch.logsumexp(torch.where(beyond, pres.log_pi, neg_inf), dim=-1)
        other_log_mass = torch.where(other_live, lse, torch.full_like(lse, MASKED_LOG_PRESENCE))
        # Every adjacent pair of unrevealed seats and the seat boundary (positions r .. K of the one
        # order; revealed moves fill positions < r, pinned ahead of every π — a structural prefix).
        gap = order_gap(key.gather(-1, order), torch.full_like(r, K), n_seatable, start=r)
        gap = torch.where((k_m > 0) & (r < K), gap, torch.full_like(gap, math.inf))
        return MovePresence(presence=pres, seat_nums=seat_nums, seat_live=seat_live,
                            seat_revealed=seat_revealed, seat_pi=seat_pi, r=r,
                            other_mass=other_mass, other_log_mass=other_log_mass,
                            other_live=other_live, beyond=beyond, tie_gap=gap)


# ============================================================================ the loss terms

def label_multi_hot(belief_species: torch.Tensor, n_species: int, like: torch.Tensor) -> torch.Tensor:
    """[B,S] float 0/1 — the TRUE unseen species set from ``belief_species`` [B,6] (−1 = revealed / pad).
    ``scatter_add`` of exact 0/1 counts then ``> 0``: order-independent, so deterministic."""
    lab = belief_species.long().to(like.device)
    ok = lab >= 0
    ids = lab.clamp(0, n_species - 1)
    cnt = torch.zeros(lab.shape[0], n_species, dtype=like.dtype, device=like.device)
    cnt.scatter_add_(1, ids, ok.to(like.dtype))
    # index 0 only ever receives the 0.0s of the −1 pads (num 0 is never a species)
    return (cnt > 0).to(like.dtype)


def set_bce(logits: torch.Tensor, pres: Presence, y: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The SET BCE (§3.2): one binary indicator per candidate species, on the EXACT logit (no clamp).
    Per row: Σ over ``V`` / k (per unseen mon — the per-slot CE's scale); averaged over the scored rows.

    A row is scored iff ``live`` (0 < k < n: k = 0 has nothing to predict, k = n carries no
    information) AND its labels are CONSISTENT: every true unseen species is a candidate and they
    count exactly k (what makes ``∂BCE/∂τ = 0`` hold). Returns (loss 0-d, n_scored 0-d, n_mismatch
    0-d) — ``n_mismatch`` counts live rows dropped for inconsistent labels (reported, never silent)."""
    in_v = (y > 0.5) & pres.cand
    consistent = ((y > 0.5) & ~pres.cand).sum(-1).eq(0) & in_v.sum(-1).eq(pres.k)
    rows = pres.live & consistent
    per = F.binary_cross_entropy_with_logits(logits, y, reduction="none")       # finite: exact logit
    per = torch.where(pres.cand & rows.unsqueeze(-1), per, torch.zeros_like(per))
    per_row = per.sum(-1) / pres.k.clamp(min=1).to(per.dtype)
    n = rows.sum().to(per.dtype)
    loss = per_row.sum() / n.clamp(min=1)
    return loss, n, (pres.live & ~consistent).sum().to(per.dtype)


def belief_head_team_scores(species_logits: torch.Tensor, believed: torch.Tensor) -> torch.Tensor:
    """BeliefHead's RE-TARGET (§3.2 "Supervision"): its per-slot species logits [B,6,S] (prior ⊕ delta
    under `species_prior_fusion`) reduced to ONE team-level score [B,S] — the mean over the hidden slots
    (the prior half is slot-independent, so this is prior ⊕ the mean delta). It goes through the same
    fixed-size construction and set BCE; the arm changes the representation, not the shaping signal."""
    w = believed.to(species_logits.dtype).unsqueeze(-1)                          # [B,6,1]
    return (species_logits * w).sum(1) / w.sum(1).clamp(min=1)


def hypothesis_moves_targets(slot_species: torch.Tensor, slot_is_hyp: torch.Tensor,
                             belief_species: torch.Tensor, belief_moves: torch.Tensor,
                             n_moves: int, like: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(multi-hot [B,6,M], supervised [B,6] bool) for a per-slot move head on HYPOTHESIS seats (§3.4's
    rule): the seat holding hypothesis species ``h`` is supervised IFF ``h`` IS on the true unseen team,
    against that mon's true moveset; a hypothesis not on the team is masked. Structural, no threshold.
    A seat whose matched mon has no labeled move is not supervised (never toward "no moves"). Shared by
    BeliefHead's moves term (`hypothesis_moves_bce`) and, since U3, MoveBelief's unrevealed population
    (`belief_bank.move_belief_loss` under fixed_mass — the hypothesis seat's move head now reads a
    species-specific token, so the blob's Hungarian slot matching would supervise one species with
    another's moveset)."""
    B, T = slot_species.shape
    lab = belief_species.long().to(like.device)
    mv = belief_moves.long().to(like.device)
    h = slot_species.long()
    match = (lab.unsqueeze(1) == h.unsqueeze(-1)) & (lab.unsqueeze(1) >= 0) & (h > 0).unsqueeze(-1) \
        & slot_is_hyp.unsqueeze(-1)                                              # [B,T,6]
    present = match.any(-1)                                                      # [B,T]
    # the FIRST matching label slot, deterministically (unique descending weights; F-X5-19's rule)
    wts = (T - torch.arange(lab.shape[1], device=lab.device)).view(1, 1, -1)
    first = (match.long() * wts).argmax(-1)                                      # [B,T]
    ids = mv.gather(1, first.unsqueeze(-1).expand(B, T, mv.shape[-1]))           # [B,T,4]
    okm = ids >= 0
    mh = torch.zeros(B, T, n_moves, dtype=like.dtype, device=like.device)
    mh.scatter_add_(2, ids.clamp(0, n_moves - 1), okm.to(like.dtype))
    mh = (mh > 0).to(like.dtype)
    return mh, present & okm.any(-1)


def hypothesis_moves_bce(move_logits: torch.Tensor, slot_species: torch.Tensor,
                         slot_is_hyp: torch.Tensor, belief_species: torch.Tensor,
                         belief_moves: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """BeliefHead's MOVES term on hypothesis seats (§3.4's rule, the Hungarian matching retired; the
    targets are `hypothesis_moves_targets`). ``move_logits`` [B,6,M]; labels [B,6] / [B,6,4]. Returns
    (per-slot-mean BCE 0-d, n_supervised 0-d)."""
    mh, sup = hypothesis_moves_targets(slot_species, slot_is_hyp, belief_species, belief_moves,
                                       move_logits.shape[-1], move_logits)
    per = F.binary_cross_entropy_with_logits(move_logits, mh, reduction="none").mean(-1)  # [B,T]
    per = torch.where(sup, per, torch.zeros_like(per))
    n = sup.sum().to(per.dtype)
    return per.sum() / n.clamp(min=1), n
