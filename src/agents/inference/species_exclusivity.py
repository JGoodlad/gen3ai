"""Species-clause EXCLUSIVITY over the hidden-team species belief — a READING AID, not the belief.

The gap this closes. ``BeliefHead`` publishes one INDEPENDENT species posterior per hidden
opponent slot (`belief_heads.BeliefHead.species_logits` → a per-slot softmax over the num axis).
Each row is a legitimate marginal; the JOINT reading of the rows is not. The reported case: three
hidden slots carrying P(Salamence) = 0.39 / 0.60 / 0.39 at one decision — an expected Salamence
count of 1.38 on a team the SPECIES CLAUSE caps at exactly one. Nothing in the head's
parameterization can prevent that: six independent softmaxes have no channel to say "at most one of
you".

**Measured** (`tmp/species_exclusivity_measure.py`, gen-15 `ai_v9_18` @2M and @6M, 1500 decisions
each with ≥1 hidden and ≥1 revealed opponent mon, posteriors re-computed from the exact eval
snapshot). Two DIFFERENT defects, and the common one is not the one that motivated this:

* the DISTRIBUTION is jointly illegal (peak E[count] > 1) on **2.5% → 3.5%** of decisions, peaking
  at 1.70;
* two hidden slots NAME the same mon — a legal expected count, an illegal DISPLAY — on
  **6.5% → 14.2%**, rising with training and reaching 29.9% at four hidden slots.

So the presentational defect is ~4x the distributional one and growing. It is also the one a reader
sees, which is why `coherent_team_hypothesis` exists beside the posterior operator rather than
being folded into it.

**What this module is.** A pure, post-hoc operator that takes the published marginals and returns
the nearest set of per-slot distributions that a gen3 team could actually realize:

  (a) zero mass on a species already REVEALED on the opponent's side,
  (b) per-species expected count over the hidden slots ≤ 1,
  (c) every row still a distribution (sums to 1).

**What this module is NOT — and the distinction is the whole contract.** The model's belief IS the
raw marginals. This is a display/aggregation transform applied at READ time, so a surface must show
BOTH: hiding the raw rows behind a "cleaner" adjusted view would substitute our arithmetic for the
model's actual state, which is the same dishonesty as reporting a calibrated number without its
provenance. Every consumer here is expected to carry the raw alongside.

**The method: Sinkhorn-style iterative proportional fitting on the species axis.** Constraint (c) is
an affine set (row sums = 1); constraint (b) is a convex set (column sums ≤ 1). Alternating the two
multiplicative I-projections — scale an over-full column by ``1/colsum``, then renormalize the rows —
is the standard capacity-constrained Sinkhorn iteration. Three properties earn it the job over a
bespoke rule:

* **It degrades to the IDENTITY when the input is already coherent.** No column is over-full, so the
  column step is a no-op; rows already sum to 1, so the row step is a no-op. An adjusted view that
  silently differs from the raw view on a coherent belief would be worse than no view at all.
* **It is differentiable**, so if a substrate consumer is ever wired (v94's aggregate reads), the
  same operator can sit in a forward pass without being rewritten from scratch.
* **It is the minimum-KL correction** among the feasible points, i.e. it changes the belief as
  little as the constraint allows rather than imposing a shape of our choosing.

**Feasibility.** The constraint set is non-empty whenever the rows have broad enough support (Hall's
condition on the support graph); with a softmax over ~400 species every entry is strictly positive,
so a real belief is always feasible. A hand-built degenerate input (two rows each pinned at 1.0 on
the SAME species, exact zeros elsewhere) has an EMPTY feasible set and cannot converge — the
iteration is bounded by ``max_iter`` and REPORTS non-convergence (``ExclusivityInfo.converged``)
rather than looping or silently returning a point that satisfies neither constraint.

Pure numpy, no torch, no model, no battle — unit-testable with hand-written arrays. Read at trace /
display time only, never on the obs-build or training hot path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

# The iteration's stopping rule: the largest per-species expected count may exceed 1 by at most this
# much. 1e-9 is ~7 orders below the smallest probability any surface renders (0.1%), so "converged"
# means converged for every purpose here and not merely to display precision.
DEFAULT_TOL = 1e-9

# Bounded so a degenerate/infeasible input terminates.
#
# ⚠️ The iteration is SUBLINEAR near the solution, and the number below is measured rather than
# guessed. Alternating I-projections converge at a rate that degrades toward 1 as the binding
# columns approach their cap, so the excess decays roughly harmonically in the tail: on 300 random
# feasible beliefs the worst case took 425 rounds to reach 1e-6 and 712 to reach `DEFAULT_TOL`
# (1e-9). Each round is one `[H, S]` scale plus one row renormalize with H ≤ 6 and S ≈ 400, so even
# the worst case is well under a millisecond and there is no reason to trade exactness for speed.
# An over-relaxed variant (raise the column factor to a power ~1.8) cuts it to 392 and was NOT
# adopted: it is no longer the KL projection, so it would buy ~1.8x on an already-negligible cost by
# giving up the "changes the belief as little as the constraint allows" property.
#
# On REAL beliefs it is far cheaper than that worst case — over 3000 gen-15 decisions the median was
# 0 rounds (the identity fast path) and the maximum 143, with zero non-convergences. The bound is
# sized for the random probe rather than the observed data on purpose: the reason to have a cap at
# all is the input nobody has seen yet.
DEFAULT_MAX_ITER = 2000


@dataclass(frozen=True)
class ExclusivityInfo:
    """Diagnostics for one :func:`exclusive_team_posterior` call.

    ``converged`` False means the feasible set was empty (or ``max_iter`` was too small) — the
    returned rows are still distributions but the column cap is NOT satisfied, and a surface must
    not present them as clause-consistent. ``max_expected_count_before`` / ``_after`` bracket what
    the operator actually changed; ``iterations`` is the work done; ``revealed_leak_before`` is the
    per-slot mass that sat on already-revealed species (see :func:`revealed_leak`) and is reported
    because it is a DIFFERENT defect from marginal-vs-joint incoherence — a hidden slot cannot be a
    mon standing on the field, full stop.
    """

    converged: bool
    iterations: int
    max_expected_count_before: float
    max_expected_count_after: float
    illegal_mass_before: float
    illegal_mass_after: float
    revealed_leak_before: NDArray[np.float64]  # [H] mass on revealed species, per hidden slot
    total_variation: NDArray[np.float64]       # [H] ½·Σ|adjusted − raw|, how far each row moved


def _as_rows(slot_probs: ArrayLike) -> NDArray[np.float64]:
    """``[H, S]`` float64 copy, validated. Raises on a non-2D input or a non-finite entry —
    a silently-reshaped belief would pair each slot with the wrong species axis."""
    rows = np.array(slot_probs, dtype=np.float64, copy=True)
    if rows.ndim != 2:
        raise ValueError(f"slot_probs must be [H, S]; got shape {rows.shape}")
    if not np.all(np.isfinite(rows)):
        raise ValueError("slot_probs contains a non-finite entry")
    if np.any(rows < 0.0):
        raise ValueError("slot_probs contains a negative entry")
    return rows


def _revealed_index(revealed_species: Optional[Iterable[int]], n_species: int) -> NDArray[np.int64]:
    """The revealed-species column indices as an int array, clipped to the axis. Out-of-range ids
    are DROPPED rather than clamped: clamping would zero an innocent column (index 0 is the UNKNOWN
    sentinel), which is a wrong answer where dropping is merely a missing constraint."""
    if revealed_species is None:
        return np.zeros(0, dtype=np.int64)
    idx = np.asarray(list(revealed_species), dtype=np.int64).reshape(-1)
    return idx[(idx >= 0) & (idx < n_species)]


def expected_counts(slot_probs: ArrayLike) -> NDArray[np.float64]:
    """``[S]`` — the expected number of hidden slots holding each species, ``Σ_h P[h, s]``.

    This is the quantity the species clause bounds by 1. Note ``Σ_s expected_counts == H`` exactly
    (each row sums to 1), so the vector redistributes a fixed budget; only its PEAK is a violation.
    """
    return cast(NDArray[np.float64], _as_rows(slot_probs).sum(axis=0))


def illegal_mass(slot_probs: ArrayLike) -> float:
    """``Σ_s max(0, E[count(s)] − 1)`` — total expected-count mass above what the clause allows.

    The scalar incoherence headline. Zero on any coherent belief; bounded above by ``H − 1`` (every
    hidden slot certain of the same species). Reported alongside the peak because the two say
    different things: a peak of 1.4 on one species is a single confident contradiction, while 0.4 of
    illegal mass spread over four species is diffuse over-commitment.
    """
    counts = expected_counts(slot_probs)
    return float(np.maximum(counts - 1.0, 0.0).sum())


def revealed_leak(slot_probs: ArrayLike,
                  revealed_species: Optional[Iterable[int]]) -> NDArray[np.float64]:
    """``[H]`` — per hidden slot, the posterior mass sitting on an ALREADY-REVEALED species.

    A strictly stronger defect than joint incoherence and worth measuring on its own: the marginals
    can be individually defensible while still being jointly illegal, but "this hidden bench mon
    might be the Starmie currently standing on the field" is wrong under any reading. The belief
    head's Smogon prior half already floors revealed species at ``SPECIES_CLAUSE_LOGIT``
    (``damage_tables``, ~1e-6) under ``--species-prior-fusion``, but the LEARNED delta is added on
    top of that floor and nothing bounds it — so a leak is possible even on a fusion run, and is a
    measurement rather than an assumption.
    """
    rows = _as_rows(slot_probs)
    idx = _revealed_index(revealed_species, rows.shape[1])
    if idx.size == 0:
        return np.zeros(rows.shape[0], dtype=np.float64)
    return cast(NDArray[np.float64], rows[:, idx].sum(axis=1))


def exclusive_team_posterior(
    slot_probs: ArrayLike,
    revealed_species: Optional[Iterable[int]] = None,
    *,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
) -> NDArray[np.float64]:
    """``[H, S]`` per-hidden-slot distributions consistent with the SPECIES CLAUSE.

    ``slot_probs``      ``[H, S]`` the published per-slot species posteriors, HIDDEN SLOTS ONLY
                        (a revealed slot is not a guess and must not be a row here).
    ``revealed_species`` species indices already on the opponent's side — the same index space as
                        the belief axis, i.e. national-dex nums.

    Returns rows that (a) carry zero mass on a revealed species, (b) have per-species column sums
    ≤ 1, and (c) still sum to 1. Byte-identical to the input (up to float64 round-trip) when the
    input already satisfies all three — see :func:`exclusive_team_posterior_info` when you need to
    know whether it converged or how far it moved.
    """
    adjusted, _ = exclusive_team_posterior_info(
        slot_probs, revealed_species, tol=tol, max_iter=max_iter)
    return adjusted


def exclusive_team_posterior_info(
    slot_probs: ArrayLike,
    revealed_species: Optional[Iterable[int]] = None,
    *,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
) -> Tuple[NDArray[np.float64], "ExclusivityInfo"]:
    """:func:`exclusive_team_posterior` plus the :class:`ExclusivityInfo` diagnostics.

    The two-return form is separate rather than a ``return_info=`` flag so each name has ONE return
    type — a surface that wants only the array cannot accidentally render a tuple.
    """
    raw = _as_rows(slot_probs)
    n_hidden, n_species = raw.shape
    leak_before = revealed_leak(raw, revealed_species)
    counts_before = raw.sum(axis=0)
    max_before = float(counts_before.max()) if n_species else 0.0
    illegal_before = float(np.maximum(counts_before - 1.0, 0.0).sum())

    if n_hidden == 0:
        return raw, ExclusivityInfo(
            converged=True, iterations=0,
            max_expected_count_before=max_before, max_expected_count_after=max_before,
            illegal_mass_before=illegal_before, illegal_mass_after=illegal_before,
            revealed_leak_before=leak_before,
            total_variation=np.zeros(0, dtype=np.float64))

    idx = _revealed_index(revealed_species, n_species)

    # THE IDENTITY FAST PATH, and it is a CORRECTNESS requirement rather than an optimization.
    # Dividing an already-normalized row by its own float sum is not the identity in binary floating
    # point (0.7 + 0.2 + 0.1 sums to 0.9999999999999999, so the "no-op" would perturb every entry),
    # and a surface comparing the adjusted view to the raw one would then draw a difference on a
    # belief that has none. An already-coherent belief is returned BIT-IDENTICAL.
    if idx.size == 0 or float(raw[:, idx].max(initial=0.0)) == 0.0:
        if max_before - 1.0 <= tol:
            return raw, ExclusivityInfo(
                converged=True, iterations=0,
                max_expected_count_before=max_before, max_expected_count_after=max_before,
                illegal_mass_before=illegal_before, illegal_mass_after=illegal_before,
                revealed_leak_before=leak_before,
                total_variation=np.zeros(n_hidden, dtype=np.float64))

    work = raw.copy()

    # (a) Zero the revealed columns ONCE. Nothing below re-adds mass to a zeroed column — the column
    # step only scales columns down and the row step scales rows — so this needs no re-application.
    if idx.size:
        work[:, idx] = 0.0

    work = _renormalize_rows(work)

    # (b)+(c) Alternating multiplicative I-projections: cap the over-full columns, then restore the
    # rows. Each half-step is the KL-closest point satisfying its own constraint, so the pair
    # converges to the KL-closest point of the intersection whenever the intersection is non-empty.
    iterations = 0
    converged = float(work.sum(axis=0).max()) - 1.0 <= tol
    while not converged and iterations < max_iter:
        counts = work.sum(axis=0)
        scale = np.minimum(1.0, 1.0 / np.maximum(counts, 1e-300))     # only shrink over-full columns
        work *= scale[None, :]
        work = _renormalize_rows(work)
        iterations += 1
        converged = float(work.sum(axis=0).max()) - 1.0 <= tol

    counts_after = work.sum(axis=0)
    info = ExclusivityInfo(
        converged=converged,
        iterations=iterations,
        max_expected_count_before=max_before,
        max_expected_count_after=float(counts_after.max()),
        illegal_mass_before=illegal_before,
        illegal_mass_after=float(np.maximum(counts_after - 1.0, 0.0).sum()),
        revealed_leak_before=leak_before,
        total_variation=0.5 * np.abs(work - raw).sum(axis=1),
    )
    return work, info


def _renormalize_rows(rows: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rescale each row to sum to 1.

    A row whose mass was ENTIRELY on revealed species sums to exactly 0 here. That cannot arise from
    a softmax (every entry is strictly positive), so it means the caller handed us a hard-zeroed
    hand-built row; the honest completion is UNIFORM over the species still allowed, which states
    only "not one of the revealed ones" — an absence, not a claim about which mon it is. A uniform
    row over the FULL axis would be the claim, so the zeroed columns stay zero.
    """
    sums = rows.sum(axis=1, keepdims=True)
    dead = (sums <= 0.0).reshape(-1)
    if np.any(dead):
        rows = rows.copy()
        # A column zeroed by the revealed mask is zero in EVERY row, so the still-ALLOWED set is
        # recoverable from any surviving row. With no surviving row there is nothing left to read it
        # from, and the uniform over the full axis is the only completion available.
        live = rows[~dead]
        if live.shape[0]:
            allowed_mask = live.sum(axis=0) > 0.0
        else:
            allowed_mask = np.ones(rows.shape[1], dtype=bool)
        if not allowed_mask.any():
            allowed_mask = np.ones(rows.shape[1], dtype=bool)
        fill = allowed_mask.astype(np.float64) / float(allowed_mask.sum())
        rows[dead] = fill
        sums = rows.sum(axis=1, keepdims=True)
    return cast(NDArray[np.float64], rows / sums)


def coherent_team_hypothesis(
    slot_probs: ArrayLike,
    revealed_species: Optional[Iterable[int]] = None,
    num_to_name: Optional[Dict[int, str]] = None,
    slot_ids: Optional[Sequence[int]] = None,
) -> List[dict]:
    """The single most likely hidden team CONSISTENT with the species clause — a POINT hypothesis.

    Greedy no-duplicates assignment: rank every ``(slot, species)`` cell by probability, walk the
    ranking, and take a cell whenever neither its slot nor its species is already spoken for.
    Revealed species are excluded up front.

    **Greedy, not Hungarian, and deliberately so.** The Hungarian assignment maximizes the SUM of
    log-probabilities — a different objective, whose answer can demote a slot's near-certain top-1 to
    buy a larger gain elsewhere. This is a display line answering "what single team is the model
    closest to naming", where per-slot recognizability beats joint optimality; the prober's
    belief-vs-truth panel already runs the Hungarian match for the question that genuinely is an
    assignment problem (scoring the belief against the TRUE hidden mons, matching the training aux
    loss's own cost). Two objectives, two functions, both named for what they do.

    ``slot_ids`` maps row index → the real opponent team-slot number (rows here are hidden slots
    only, so row 0 is rarely slot 0). Defaults to ``range(H)``.

    Returns ``[{"slot": int, "species": name, "prob": float, "raw_top1": name,
    "raw_top1_prob": float, "differs": bool}, ...]`` in slot order — carrying the RAW top-1 beside
    the assignment so a surface can show the disagreement, which is the only case worth drawing.
    """
    rows = _as_rows(slot_probs)
    n_hidden, n_species = rows.shape
    ids = list(slot_ids) if slot_ids is not None else list(range(n_hidden))
    if len(ids) != n_hidden:
        raise ValueError(f"slot_ids has {len(ids)} entries for {n_hidden} rows")

    def name_of(s: int) -> str:
        if num_to_name is not None:
            return num_to_name.get(int(s), f"num_{int(s)}")
        return f"num_{int(s)}"

    work = rows.copy()
    idx = _revealed_index(revealed_species, n_species)
    if idx.size:
        work[:, idx] = 0.0

    assigned: Dict[int, int] = {}
    used: set = set()
    # Rank every cell once. H ≤ 6 and S ≈ 400, so this is ~2400 cells — a full sort is cheaper than
    # any incremental structure and has no state to get wrong.
    order = np.argsort(work, axis=None)[::-1]
    for flat in order:
        h, s = divmod(int(flat), n_species)
        if h in assigned or s in used or work[h, s] <= 0.0:
            continue
        assigned[h] = s
        used.add(s)
        if len(assigned) == n_hidden:
            break

    out: List[dict] = []
    for h in range(n_hidden):
        top1 = int(np.argmax(work[h])) if work[h].max() > 0 else -1
        s = assigned.get(h, -1)
        out.append({
            "slot": int(ids[h]),
            "species": name_of(s) if s >= 0 else None,
            "prob": float(rows[h, s]) if s >= 0 else 0.0,
            "raw_top1": name_of(top1) if top1 >= 0 else None,
            "raw_top1_prob": float(rows[h, top1]) if top1 >= 0 else 0.0,
            "differs": bool(s >= 0 and top1 >= 0 and s != top1),
        })
    return out
