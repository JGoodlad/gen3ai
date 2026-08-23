"""The obs/model DECODING units — each pure, each fed the loaded model and the raw arrays.

Faithfulness, the per-move matchup table, the intervention sweep, the two saliency heads, the
threat matrices and the incoming-damage belief. Every torch call goes through the injected
`model`, so all of it is testable with a fake.
"""

from __future__ import annotations

import numpy as np

from agents.action.constants import MOVE_END, MOVE_START

from main.prober.engine.util import _multiplier_meaningful, _norm_species, parse_pct
from main.prober.engine.views import (ActionRow, IncomingBeliefView, InterventionRow,
    InterventionSweep, MatchupView, Saliency, SaliencyBlock, ThreatView)


# Sweep points for the intervention (×-multipliers); stored /4 into the obs.
_SWEEP_MULTIPLIERS = (0.0, 1.0, 2.0, 4.0)


# The matchup matrices are 6×4×6 (opp/our mon × move slot × the other side) — mirrors
# reactive.py's `their_matchups`/`our_matchups` 144-dim blocks.
_TEAM_SIZE = 6
_MATCHUP_DIM = _TEAM_SIZE * 4 * _TEAM_SIZE   # 144


# ---------------------------------------------------------------------------
# Analysis units (each pure; fed the loaded model + arrays)
# ---------------------------------------------------------------------------

def _faithfulness(probs: np.ndarray, labels: list, acts: dict, chosen: str,
                  mask: np.ndarray) -> "tuple[ActionRow, ...]":
    rows = []
    for j, k in enumerate(labels):
        rows.append(ActionRow(
            label=k,
            valid=bool(mask[j]),
            recorded=parse_pct(acts[k]["prob"]),
            rerun=float(probs[j]),
            is_chosen=(k == chosen),
        ))
    return tuple(rows)


def _display_hp(move_id: str, our_species: str, hp_map: "dict | None") -> str:
    """Render OUR move label's Hidden Power with its TYPE (``hiddenpower(grass)``), since we always
    know our own HP type. Handles BOTH trace vintages: the recorder's typed id ``hiddenpowergrass``
    (recovered straight from the id) AND a bare own ``hiddenpower`` from an OLDER trace (recovered
    from the reconstruction ``hp_map``, our side only — matches how the Board/move-belief retype).
    A non-HP label, or a bare HP with no reconstruction record, is returned unchanged (no leak: an
    opponent's un-revealed HP never reaches this — these are OUR action labels)."""
    if not move_id.startswith("hiddenpower"):
        return move_id
    if move_id != "hiddenpower":                       # typed id (new traces) → from the id itself
        return f"hiddenpower({move_id[len('hiddenpower'):]})"
    return (hp_map or {}).get(_norm_species(our_species)) or move_id   # bare own HP → reconstruction


def _matchups(obs: np.ndarray, labels: list, off, our_species: str = "",
              hp_map: "dict | None" = None) -> MatchupView:
    # gen3_cpu_damage_deleted_v1: mm_off==0 ⇒ the block is absent from this obs layout.
    mults = (tuple(float(x) * 4.0 for x in obs[off.mm_off:off.mm_off + 4])
             if off.mm_off > 0 else (0.0, 0.0, 0.0, 0.0))
    raw = tuple(labels[MOVE_START:MOVE_END])  # the 4 move actions, request order
    # Show OUR Hidden Power typed (hiddenpower(grass)); the multiplier IS already typed in the obs.
    move_labels = tuple(_display_hp(lbl, our_species, hp_map) for lbl in raw)
    # Flag which slots' multipliers are meaningful vs a phantom artefact on a status/self move (from
    # the RAW move id, so is_damaging is exact) — see MatchupView.applicable. _multiplier_meaningful
    # tolerates unknown/empty ids → False.
    applicable = tuple(_multiplier_meaningful(lbl) for lbl in raw)
    return MatchupView(multipliers=mults, move_labels=move_labels, applicable=applicable)


def _switch_prob_sum(p: np.ndarray, labels: list) -> float:
    return float(sum(p[j] for j, l in enumerate(labels) if l.startswith("switch")))


def _intervention_sweep(model, obs: np.ndarray, mask: np.ndarray, labels: list,
                        chosen: str, off) -> InterventionSweep:
    cj = labels.index(chosen)
    if not (MOVE_START <= cj < MOVE_END):
        return InterventionSweep(chosen, -1, 0.0, ())
    slot = cj - MOVE_START
    probs, _ = model.action_dist(obs, mask)
    baseline_switches = _switch_prob_sum(probs, labels)
    rows = []
    for mult in _SWEEP_MULTIPLIERS:
        o = obs.copy()
        if off.mm_off > 0:                      # no-op when the block was deleted from the obs
            o[off.mm_off + slot] = mult / 4.0
        p, _ = model.action_dist(o, mask)
        rows.append(InterventionRow(
            multiplier=mult,
            p_chosen=float(p[cj]),
            p_switches=_switch_prob_sum(p, labels),
        ))
    return InterventionSweep(chosen, slot, baseline_switches, tuple(rows))


def _saliency_from_grad(g: np.ndarray, off) -> Saliency:
    """Aggregate a per-dim gradient into the named obs blocks. Shared by the policy-logit
    saliency and the critic value saliency so both report the SAME regions (incl. the new
    `incoming_damage` block) — the only difference is which head's gradient is fed in."""
    def block(name: str, lo: int, hi: int) -> SaliencyBlock:
        seg = g[lo:hi]
        return SaliencyBlock(name=name, mean_abs=float(seg.mean()), total_abs=float(seg.sum()))

    blocks = [

        *((block("our_matchups(144)", off.om_off, off.om_off + 144),
           block("their_matchups(144)", off.tm_off, off.tm_off + 144))
          if off.om_off > 0 else ()),  # gen3_entity_rehome_v1: absent from re-homed layouts
    ]
    if off.incoming_dim > 0:  # incoming-damage / OHKO belief block (incoming_damage_v1)
        blocks.append(block(f"incoming_damage({off.incoming_dim})",
                            off.incoming_off, off.incoming_off + off.incoming_dim))
    blocks += [
        block("our active pokemon block(99)", 0, off.active_block_dim),
        block("turn-history block", off.turn_history_offset,
              off.turn_history_offset + off.turn_history_dim),
    ]
    return Saliency(overall_mean_abs=float(g.mean()), blocks=tuple(blocks))


def history_slot_saliency(g: np.ndarray, off) -> "list[float]":
    """Per-turn-slot mean|grad| within the turn-history block — splits the one 'turn-history block'
    saliency into its N_HISTORY_TURNS TurnDelta slots, so we can see whether the OLDER turns carry
    little signal (a candidate to shorten N_HISTORY_TURNS and reclaim obs/attention compute). ``g``
    is an already-abs per-dim gradient (policy-logit or value). Slot i is the i-th TurnDelta in obs
    order; the transformer's positional embedding learns recency, so the caller labels recent/old."""
    td = getattr(off, "turn_delta_dim", 0) or 0
    if off.turn_history_dim <= 0 or td <= 0:
        return []
    base = off.turn_history_offset
    n = off.turn_history_dim // td
    return [float(g[base + i * td: base + (i + 1) * td].mean()) for i in range(n)]


def _saliency(model, obs: np.ndarray, mask: np.ndarray, chosen_idx: int, off) -> Saliency:
    """Policy saliency: |d logit(chosen) / d obs| aggregated per block (what the ACTOR reads)."""
    return _saliency_from_grad(model.logit_grad(obs, mask, chosen_idx), off)


def _value_saliency(model, obs: np.ndarray, mask: np.ndarray, off) -> "Saliency | None":
    """Critic saliency: |d V(s) / d obs| aggregated per block (what the VALUE head reads) — the
    relevant lens for OHKO tail-blindness. None when the model can't expose a value gradient."""
    vg = getattr(model, "value_grad", None)
    if vg is None:
        return None
    try:
        return _saliency_from_grad(vg(obs, mask), off)
    except Exception:  # noqa: BLE001 — value-grad is best-effort (stub models / odd heads)
        return None


def _threats(obs: np.ndarray, off) -> "ThreatView | None":
    """Decode the opponent's incoming type-effectiveness from `their_matchups`.

    Returns None if the obs is too short to hold the block (tiny synthetic test obs) —
    so the engine never crashes on a malformed/old trace."""
    if off.tm_off <= 0:
        return None  # gen3_entity_rehome_v1: the block no longer exists in the live layout
    seg = obs[off.tm_off:off.tm_off + _MATCHUP_DIM]
    if seg.shape[0] != _MATCHUP_DIM:
        return None
    m = seg.reshape(_TEAM_SIZE, 4, _TEAM_SIZE) * 4.0   # [opp_mon, move_slot, our_mon], ×4-denormalised
    return ThreatView(
        present=bool((m > 0).any()),
        revealed_frac=float((m > 0).mean()),
        max_incoming=float(m.max()),
        per_our_slot_max=tuple(float(m[:, :, j].max()) for j in range(_TEAM_SIZE)),
    )


def _active_slot(obs: np.ndarray, off) -> "int | None":
    """Our on-field mon's team-slot index, read from the per-mon active flag (the last dim of each
    `pokemon_full_dim`-wide our-team block). The incoming-damage block is slot-aligned to the same
    team list, so this index selects the active mon's belief. None if no flag is set / obs too short."""
    stride = off.pokemon_full_dim
    best = None
    best_v = 0.5
    for i in range(_TEAM_SIZE):
        idx = i * stride + stride - 1
        if idx < obs.shape[0] and float(obs[idx]) > best_v:
            best_v, best = float(obs[idx]), i
    return best


def decode_incoming_belief(obs: np.ndarray, off) -> "IncomingBeliefView | None":
    """Decode the incoming-damage / OHKO belief block from a raw obs vector.

    Pure + model-free (reads the saved obs the trace recorded), so it is exact for the model that
    produced the trace and is reusable by both `analyze_invocation` and the model-free `scan`.

    ``off`` (the obs offsets) is resolved from the CURRENT encoder, so it is only valid for traces of
    the current arch. A trace whose obs length differs (e.g. an archived old-arch run) would be
    **mis-sliced** by these offsets — so we REFUSE it (return None) on a length mismatch rather than
    silently decoding garbage. ``pm>=8`` is the crit-split layout (the live arch); the ``pm==5`` branch
    serves explicit/synthetic 5-field ObsOffsets (tests) — it is NOT a path for archived old-arch traces
    (``resolve()`` always yields the current 8-field offsets, and the length guard rejects the old obs).
    Returns None when the block is absent (dim 0) or the obs length doesn't match the current arch."""
    if off.incoming_dim <= 0:
        return None
    if getattr(off, "total_dim", 0) and obs.shape[0] != off.total_dim:
        return None   # wrong-length trace (old/foreign arch) — refuse, don't mis-slice
    seg = obs[off.incoming_off:off.incoming_off + off.incoming_dim]
    if seg.shape[0] != off.incoming_dim:
        return None
    pm, rec_dim = off.incoming_per_mon, off.incoming_recovery
    n_slots = (off.incoming_dim - rec_dim) // pm
    per_slot_pko, per_slot_exp, outspeeds, per_slot_nc, reveals = [], [], [], [], []
    for i in range(n_slots):
        b = i * pm
        if pm >= 8:   # crit-split: phys/spec exp, phys/spec pko_nocrit, phys/spec CRIT_DELTA, outspeed, revealed
            phys_exp, spec_exp, phys_nc, spec_nc, phys_d, spec_d, outspeed, revealed = (
                float(x) for x in seg[b:b + 8])
            # reconstruct the crit-inclusive P(KO) (= nocrit + delta) so active_pko keeps its meaning
            per_slot_pko.append(max(phys_nc + phys_d, spec_nc + spec_d))
            per_slot_nc.append(max(phys_nc, spec_nc))
            reveals.append(revealed)
        else:         # explicit/synthetic 5-field: phys_exp, spec_exp, phys_pko, spec_pko, outspeed
            phys_exp, spec_exp, phys_pko, spec_pko, outspeed = (float(x) for x in seg[b:b + 5])
            per_slot_pko.append(max(phys_pko, spec_pko))
        per_slot_exp.append(max(phys_exp, spec_exp))
        outspeeds.append(outspeed)
    rec = seg[n_slots * pm:]
    a = _active_slot(obs, off)
    in_range = a is not None and a < n_slots
    split = pm >= 8
    return IncomingBeliefView(
        present=bool(any(v > 0 for v in per_slot_pko) or any(v > 0 for v in per_slot_exp)),
        max_pko=max(per_slot_pko) if per_slot_pko else 0.0,
        active_pko=per_slot_pko[a] if in_range else None,
        active_exp=per_slot_exp[a] if in_range else None,
        active_outspeed=outspeeds[a] if in_range else None,
        per_slot_pko=tuple(per_slot_pko),
        recovery_rate=float(rec[0]) if rec.shape[0] > 0 else 0.0,
        cures_status=float(rec[1]) if rec.shape[0] > 1 else 0.0,
        recovery_known=float(rec[2]) if rec.shape[0] > 2 else 0.0,
        active_pko_nocrit=(per_slot_nc[a] if (split and in_range) else None),
        threat_revealed=(reveals[a] if (split and in_range) else None),
    )


# NOTE: a former `_reorder_move_labels` helper re-sorted the recorded move labels to the per-mon block's
# MOVESET order — it was REMOVED because the recorded `actions` dict is already in ACTION-INDEX / request-slot
# order (see `analyze_invocation`); reordering it scrambled correct labels. The recorder↔prober alignment
# invariant is pinned by `engine_test.test_recorded_actions_are_action_index_aligned`.
