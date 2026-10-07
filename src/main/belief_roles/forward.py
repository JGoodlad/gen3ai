"""ONE checkpoint on the bank: CPU forward passes → the per-decision belief quantities every read needs.

The load goes through THE strict loader (``load_checkpoint_strict`` + ``historical_load_kwargs``,
`gen3_strict_checkpoint_load_v1`); the forwards run inside ``policy_spectrum.reader.inference_globals``
(fixed thread count, restored on exit), batched in the bank's FIXED order. The checkpoint must carry
X5's hypothesis tokens (``features_extractor.hypothesis_builder``) — the only belief representation
since the X5 version break (config v144); its arm is recorded as ``fixed_mass`` (the name the X5 A/B
reads carry). A PRE-BREAK checkpoint (a blob one, or a pre-break fixed_mass one) is REFUSED before any
forward (:class:`ReadRefused`, the loader's typed reason: run this reader PINNED to the checkpoint's own
commit) — the blob read arm (BeliefHead's hidden-slot posterior, α / β) was deleted with the blob path.

Per decision, two COLUMNS of belief (each a per-species presence π over the structural candidate set
V = the dex-row table's valid nums minus the revealed ones, Σ_V π = k = the hidden count):

* ``arm`` — the checkpoint's OWN presence, ``hs.species.pi`` (σ(log P_T0 + δ_θ + τ));
* ``prior`` — the Smogon T0 prior alone (``species_team_prior_logits`` → the same fixed-size
  construction), the comparison column (§4.2), identical for every checkpoint.

The opponent-intent read (the COMMON EVENT SPACE, `bank_rows`) is the FLAT opponent pointer's
(:func:`_flat_event_logp`, X5 U4 `gen3_x5_flat_pointer_v1`): one softmax over [K move seats · OTHER_move ·
six switch targets · OTHER_species]; a seat's content is its move (a Hidden Power seat — 237 or a typed
channel — is the one HP event), a slot's its species (revealed, or the hypothesis it holds), OTHER's the
renormalised tail (π_m / Σ π_m over OTHER_move's members; ``other_tail_probs`` for OTHER_species).

An event no candidate can produce (a move outside the seats and outside OTHER_move's members; a
switch-in no slot's content supports) is a MISS: its own column, never floored into the log loss (§7.4).

Amendment 3(b) (§7.7(b), :mod:`.eset`): the same pointer also gives the FULL distribution on the dense
event space (checked equal to the scored probability at the realised event), and its mass on E_row =
a blob run's named set, passed as ``references`` — the BANKED ``<label>.erow.npz`` sets the X5 A/B's blob
reads wrote (EVERY blob run of the look; the run's value is the mean over them).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from main.belief_roles.bank_rows import SWITCH_BASE, BankRows
from main.belief_roles.eset import ERow, ESet, event_index, flat_event_dist
from main.belief_roles.roles import HP_NUM, RoleSet

BATCH = 256
TEAM = 6
#: The 16 typed Hidden Power channels (`hypothesis_set.TYPED_HP_NUMS`).
TYPED_HP = tuple(range(355, 371))
#: Rule-8 half-width at a selection boundary (`hypothesis_set.SELECTION_TIE_EPS`).
TIE_EPS = 1e-6


@dataclass
class Columns:
    """Per-decision arrays, bank order. ``*_arm`` is the checkpoint's own belief; ``*_prior`` the
    Smogon T0 column."""

    arm: str
    k: np.ndarray                    # [N] hidden-mon count (6 − r)
    cand: np.ndarray                 # [N,S] bool — V
    revealed_nums: np.ndarray        # [N,6] int — revealed opponent species nums (0 = hidden / none)
    pi_arm: np.ndarray               # [N,S] float64
    pi_prior: np.ndarray             # [N,S] float64
    sel_tie_arm: np.ndarray          # [N] bool — a near-tie at the arm's species selection boundary
    sel_tie_prior: np.ndarray        # [N] bool
    read_tie_arm: np.ndarray         # [N] bool — `near_tie_rows(hs)`
    other_mass_model: Optional[np.ndarray]   # [N] the model's own OTHER mass (None in a synthetic Columns)
    intent_logp: np.ndarray          # [N] float64 — log P(realised event); NaN where unlabeled or a miss
    intent_covered: np.ndarray       # [N] bool — the event is in the arm's support
    opp_active_species: np.ndarray   # [N] int — the opponent active's num (the miss breakdown)
    role_M_arm: np.ndarray           # [N,R] summed presence mass per role
    role_V_arm: np.ndarray           # [N,R] Σ p(1 − p) over the contributing indicators
    role_M_prior: np.ndarray
    role_V_prior: np.ndarray
    # ---- Amendment 3(b) (`eset`): E_row = a blob run's named set, from EVERY banked reference blob run of
    # the look (one entry each); empty when none was given
    esets: List[ESet] = field(default_factory=list)
    eset_logp_event: Optional[np.ndarray] = None  # [N] log P_arm(realised event), dense; −inf at 0


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class ReadRefused(ValueError):
    """A checkpoint this reader cannot read at this commit: a PRE-BREAK one (its ``model_config.json``
    below ``MIGRATION_FLOOR``, or a pickled ``belief_tokens='blob'``), carrying the loader's own typed
    reason and the pinned fix. The CLI prints it and exits; never a traceback."""


def refuse_pre_break(zip_path: Path) -> None:
    """Raise :class:`ReadRefused` for a PRE-BREAK checkpoint, BEFORE anything is loaded (the CLI runs it
    on every ``--ckpt`` before the bank's re-encode): its run-level ``model_config.json`` through
    ``version_break.check_post_break`` (every blob or pre-break fixed_mass run), then its pickled kwargs
    through the strict loader's own ``refuse_deleted_pickled_kwargs`` (a blob zip with no config beside
    it). A zip with neither verdict passes (the strict load reports anything else)."""
    from agents.model.model_version.constants import ModelVersionError
    from agents.model.model_version.shaped_reward import saved_config_path
    from agents.model.model_version.version_break import PreBreakCheckpointError, check_post_break
    from agents.model.snapshot import refuse_deleted_pickled_kwargs

    try:
        check_post_break(saved_config_path(str(zip_path)))
        refuse_deleted_pickled_kwargs(str(zip_path))
    except (PreBreakCheckpointError, ModelVersionError) as e:
        raise ReadRefused(f"{zip_path}: {e}") from e


def load_strict(zip_path: Path) -> Any:
    """The checkpoint through the strict loader, on the CPU, eval mode (a bare run dir is REFUSED: it
    would resolve to the run's last snapshot, which moves; a pre-break checkpoint is REFUSED first,
    :func:`refuse_pre_break`)."""
    from agents.model.oracle_reveal import refuse_if_revealed
    from agents.model.snapshot import historical_load_kwargs, load_checkpoint_strict

    refuse_if_revealed(str(zip_path), tool="main.belief_roles",
                       reason="The Lane S bank's rows are encoded without the reveal.")
    zip_path = Path(zip_path)
    if zip_path.is_dir() or zip_path.suffix != ".zip":
        raise ValueError(f"{zip_path}: name the checkpoint .zip — a bare run directory resolves to the "
                         "run's LAST snapshot and moves")
    refuse_pre_break(zip_path)
    m = load_checkpoint_strict(str(zip_path), device="cpu", **historical_load_kwargs(str(zip_path)))
    m.policy.eval()
    return m


def arm_of(model: Any) -> str:
    """``fixed_mass`` — the X5 hypothesis tokens, the only belief representation this code builds (the
    name the X5 A/B's reads carry). A model built with the belief family OFF has no belief to read."""
    fe = model.policy.features_extractor
    if getattr(fe, "hypothesis_builder", None) is None:
        raise ValueError("the checkpoint builds no X5 hypothesis tokens (the belief family is OFF: "
                         "--opp-belief-aux-coef / --opp-intent-coef 0) — there is no belief to read")
    return "fixed_mass"


def _boundary_tie(pi, cand, k):
    """[B] bool — the k-th and (k+1)-th π (in the one stable order) differ by < TIE_EPS."""
    from agents.model.hypothesis_set import boundary_gap, stable_order

    order = stable_order(pi, cand)
    return boundary_gap(pi.gather(-1, order), k, cand.sum(-1)) < TIE_EPS


def _hp_collapse(nums):
    """Move nums on the common event space: 237 and every typed Hidden Power → 237."""
    import torch

    typed = (nums >= TYPED_HP[0]) & (nums <= TYPED_HP[-1])
    return torch.where(typed | (nums == HP_NUM), torch.full_like(nums, HP_NUM), nums)


def _flat_event_logp(events, flat_logits, fi, pi_m, p_tail):
    """``(logp [B] (−inf on a miss), covered [B])`` of each row's realised event under U4's FLAT
    pointer (`agents.model.flat_intent`: columns [K seats · OTHER_move · six slots · OTHER_species]).
    Each live candidate's CONTENT over the event space: a seat = its move (HP collapsed); OTHER_move =
    π_m / Σ π_m over its members (``fi.beyond``; HP channels collapsed into one event); a slot = its
    species (``fi.cand_ids``: revealed, or the hypothesis it holds); OTHER_species = the renormalised
    tail ``p_tail`` over ``fi.in_tail``. Support is decided by MEMBERSHIP (structural), never by a
    value; a member's π is never 0 (§3.2)."""
    import torch

    from agents.model.flat_intent import other_move_col, other_species_col, slot_col

    K = int(fi.k)
    B, F = flat_logits.shape
    live = fi.live.bool()
    lf = torch.log_softmax(flat_logits.double().masked_fill(~live, -math.inf), dim=-1)
    neg = torch.full((), -math.inf, dtype=torch.float64)
    is_move = (events >= 0) & (events < SWITCH_BASE)
    is_sw = events >= SWITCH_BASE
    # ---- moves: the seats, then OTHER_move's members mapping to the event
    seat_ev = _hp_collapse(fi.cand_ids[:, :K].long())
    hit = live[:, :K] & (seat_ev == events.unsqueeze(-1)) & is_move.unsqueeze(-1)
    lp_seat = torch.logsumexp(torch.where(hit, lf[:, :K], neg), dim=-1)
    M = fi.beyond.shape[1]
    mem_ev = _hp_collapse(torch.arange(M).unsqueeze(0).expand(B, M))
    w = torch.where(fi.beyond, pi_m.double(), torch.zeros((), dtype=torch.float64))
    sel = fi.beyond & (mem_ev == events.unsqueeze(-1)) & is_move.unsqueeze(-1)
    om = other_move_col(K)
    om_ok = live[:, om] & sel.any(-1)
    w_sel = torch.where(sel, w, torch.zeros_like(w)).sum(-1)
    lc_om = torch.log(w_sel.clamp(min=1e-300)) - torch.log(w.sum(-1).clamp(min=1e-300))
    lp_om = torch.where(om_ok, lf[:, om] + lc_om, neg)
    lp_move = torch.logaddexp(lp_seat, lp_om)
    cov_move = is_move & (hit.any(-1) | om_ok)
    # ---- switches: the slots, then OTHER_species' tail
    S = fi.in_tail.shape[1]
    sp = (events - SWITCH_BASE).clamp(0, S - 1)
    c0 = slot_col(K)
    sl_hit = (live[:, c0:c0 + TEAM] & (fi.cand_ids[:, c0:c0 + TEAM].long() == sp.unsqueeze(-1))
              & is_sw.unsqueeze(-1))
    lp_slot = torch.logsumexp(torch.where(sl_hit, lf[:, c0:c0 + TEAM], neg), dim=-1)
    os_ = other_species_col(K)
    os_ok = live[:, os_] & fi.in_tail.gather(1, sp.unsqueeze(-1)).squeeze(-1) & is_sw
    pt = p_tail.double().gather(1, sp.unsqueeze(-1)).squeeze(-1)
    lp_os = torch.where(os_ok, lf[:, os_] + torch.log(pt.clamp(min=1e-300)), neg)
    lp_sw = torch.logaddexp(lp_slot, lp_os)
    cov_sw = is_sw & (sl_hit.any(-1) | os_ok)
    covered = cov_move | cov_sw
    lp = torch.where(is_move, lp_move, torch.where(is_sw, lp_sw, neg))
    return torch.where(covered, lp, neg), covered


def read_columns(model: Any, br: BankRows, roles: RoleSet, threads: int = 4,
                 batch: int = BATCH, references: Sequence[ERow] = ()) -> Columns:
    """The batched forward (module docstring). ``references``: banked blob runs' named sets (``eset``) —
    the E_rows the run's conditional intent read is scored on, one read each."""
    import torch

    from agents.model.extra_obs_keys import zero_extra_obs
    from agents.model.hypothesis_dex_rows import load_hypothesis_dex_rows
    from agents.model.hypothesis_set import fixed_mass_presence, near_tie_rows, species_candidates
    from agents.model.t0_species import species_team_prior_logits
    from main.belief_roles.roles import prior_move_probs
    from main.policy_spectrum.reader import inference_globals

    arm = arm_of(model)
    fe = model.policy.features_extractor
    S = int(fe.layout["max_species"])
    D = SWITCH_BASE + S
    references = list(references)
    for reference in references:
        if reference.n != br.n or reference.switch_ok.shape[1] != S:
            raise ValueError(f"the reference E_row is {reference.n} rows × {reference.switch_ok.shape[1]} "
                             f"species, this bank read is {br.n} × {S}")
    valid = torch.from_numpy(load_hypothesis_dex_rows(S).valid.copy())
    P = torch.from_numpy(prior_move_probs(S))                                   # [S,M] float64
    role_nums = torch.tensor(roles.nums, dtype=torch.long)
    R = len(roles.nums)
    P_roles = P[:, role_nums]                                                   # [S,R]
    hp_col = (role_nums == HP_NUM)
    t0 = fe.t0_species_prior
    out: Dict[str, List[np.ndarray]] = {}

    def put(name, x):
        out.setdefault(name, []).append(x.detach().cpu().numpy() if torch.is_tensor(x) else x)

    with inference_globals(threads), torch.no_grad():
        for i in range(0, br.n, batch):
            rows = torch.tensor(br.rows[i:i + batch])
            masks = torch.tensor(br.masks[i:i + batch].astype(np.float32))
            B = rows.shape[0]
            ob = {"observation": rows, "action_mask": masks}
            ob.update(zero_extra_obs(fe, batch=B))
            ctx = fe.unpack(ob)
            model.policy.get_distribution(ob)
            st = fe.stash
            opp_ids = ctx.species_ids[:, TEAM:2 * TEAM].long()
            believed = ctx.opp_believed_mask.bool()
            rev = (opp_ids > 0) & ~believed
            cand, k = species_candidates(valid, opp_ids, believed)
            # ---- the prior column (Smogon T0 → the fixed-size construction, fp64)
            t0lp = species_team_prior_logits(t0.species_prior_log_marginal, t0.species_prior_log_lift,
                                             opp_ids, believed).double()
            pi_prior = fixed_mass_presence(t0lp, cand, k).pi
            # ---- the arm's own presence
            hs = st.hypothesis
            assert hs is not None, "an X5 forward stashed no hypothesis set"
            pi_arm = hs.species.pi.double()
            read_tie = near_tie_rows(hs)
            put("other_mass_model", hs.other_mass.double())
            put("k", k)
            put("cand", cand)
            put("revealed_nums", torch.where(rev, opp_ids, torch.zeros_like(opp_ids)))
            put("pi_arm", pi_arm)
            put("pi_prior", pi_prior)
            put("sel_tie_arm", _boundary_tie(pi_arm, cand, k) | read_tie)
            put("sel_tie_prior", _boundary_tie(pi_prior, cand, k))
            put("read_tie_arm", read_tie)
            # ---- opponent intent on the common event space: the FLAT pointer (X5 U4)
            events = torch.tensor(br.event[i:i + batch], dtype=torch.long)
            assert (hs.moves is not None and st.flat_intent is not None
                    and st.flat_intent_logits is not None), "an X5 forward stashed no flat pointer"
            lp, cov = _flat_event_logp(events, st.flat_intent_logits, st.flat_intent,
                                       hs.moves.presence.pi, hs.other_tail_probs)
            Pd = flat_event_dist(st.flat_intent_logits, st.flat_intent, hs.moves.presence.pi,
                                 hs.other_tail_probs)
            # ---- Amendment 3(b): the arm's mass on each reference E_row (the dense distribution, `eset`)
            idx, ok = event_index(events, D)
            p_ev = torch.where(ok, Pd.gather(1, idx.unsqueeze(-1)).squeeze(-1),
                               torch.zeros((), dtype=torch.float64))
            lp_ev = torch.log(p_ev)                                               # −inf at 0
            gap = (lp_ev[cov] - lp[cov]).abs()
            if gap.numel() and float(gap.max()) > 1e-9:
                raise AssertionError(f"the dense event distribution disagrees with the scored log "
                                     f"probability by {float(gap.max()):.3g} nats")
            for j, r in enumerate(references):
                emask = r.mask(i, i + B, D)
                put(f"eset_mass_{j}", (Pd * emask).sum(-1))
                put(f"eset_in_{j}", emask.gather(1, idx.unsqueeze(-1)).squeeze(-1) & ok)
                put(f"eset_tie_{j}", torch.from_numpy(r.tie[i:i + B]))
            put("eset_logp_event", lp_ev)
            put("intent_logp", torch.where(cov, lp, torch.full_like(lp, math.nan)))
            put("intent_covered", cov)
            put("opp_active_species", opp_ids.gather(1, ctx.opp_active_local.long().clamp(0, TEAM - 1)
                                                     .unsqueeze(-1)).squeeze(-1))
            # ---- roles: summed presence mass per role (§4.2)
            mb = st.move_belief_logits.double()                                  # [B,6,M] typed
            ph = torch.sigmoid(mb[..., role_nums])                               # [B,6,R]
            hp_p = torch.sigmoid(mb[..., list(TYPED_HP)]).sum(-1).clamp(max=1.0)  # Σ_t P(HP_t)
            ph = torch.where(hp_col.view(1, 1, R), hp_p.unsqueeze(-1), ph)
            opp_mv = ctx.all_move_ids[:, TEAM:2 * TEAM, :].long()                # [B,6,4] revealed
            hit = (opp_mv.unsqueeze(-1) == role_nums.view(1, 1, 1, R)).any(2)    # [B,6,R]
            ph = torch.where(hit, torch.ones_like(ph), ph)                       # revealed: pinned 1
            revf = rev.double().unsqueeze(-1)
            M_rev = (revf * ph).sum(1)
            V_rev = (revf * ph * (1 - ph)).sum(1)
            sp = hs.slot_pi.double()
            q = believed.double().unsqueeze(-1) * sp.unsqueeze(-1) * ph
            tail = cand & (hs.rank >= k.unsqueeze(-1)) & (k > 0).unsqueeze(-1)
            w = torch.where(tail, pi_arm, torch.zeros_like(pi_arm))
            put("role_M_arm", M_rev + q.sum(1) + w @ P_roles)
            put("role_V_arm", V_rev + (q * (1 - q)).sum(1) + w @ P_roles - (w * w) @ (P_roles * P_roles))
            # prior column: revealed mons by Smogon P(m | s) (revealed moves 1; 4 revealed ⇒ 0)
            qp = P_roles[opp_ids.clamp(0, S - 1)]                                 # [B,6,R]
            four = ((opp_mv > 0).sum(-1) >= 4).unsqueeze(-1)
            qp = torch.where(hit, torch.ones_like(qp), torch.where(four, torch.zeros_like(qp), qp))
            M_p = (revf * qp).sum(1) + pi_prior @ P_roles
            V_p = ((revf * qp * (1 - qp)).sum(1) + pi_prior @ P_roles
                   - (pi_prior * pi_prior) @ (P_roles * P_roles))
            put("role_M_prior", M_p)
            put("role_V_prior", V_p)
    cat = {k_: np.concatenate(v, 0) for k_, v in out.items()}
    return Columns(arm=arm, k=cat["k"], cand=cat["cand"], revealed_nums=cat["revealed_nums"],
                   pi_arm=cat["pi_arm"], pi_prior=cat["pi_prior"], sel_tie_arm=cat["sel_tie_arm"],
                   sel_tie_prior=cat["sel_tie_prior"], read_tie_arm=cat["read_tie_arm"],
                   other_mass_model=cat.get("other_mass_model"),
                   intent_logp=cat["intent_logp"], intent_covered=cat["intent_covered"],
                   opp_active_species=cat["opp_active_species"],
                   role_M_arm=cat["role_M_arm"], role_V_arm=cat["role_V_arm"],
                   role_M_prior=cat["role_M_prior"], role_V_prior=cat["role_V_prior"],
                   esets=[ESet(mass=cat[f"eset_mass_{j}"], e_in=cat[f"eset_in_{j}"].astype(bool),
                               tie=cat[f"eset_tie_{j}"].astype(bool),
                               label=str(references[j].meta.get("label")),
                               checkpoint_sha256=str(references[j].meta.get("checkpoint_sha256")))
                          for j in range(len(references))],
                   eset_logp_event=cat.get("eset_logp_event"))
