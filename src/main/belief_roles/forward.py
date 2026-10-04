"""ONE checkpoint on the bank: CPU forward passes → the per-decision belief quantities every read needs.

The load goes through THE strict loader (``load_checkpoint_strict`` + ``historical_load_kwargs``,
`gen3_strict_checkpoint_load_v1`); the forwards run inside ``policy_spectrum.reader.inference_globals``
(fixed thread count, restored on exit), batched in the bank's FIXED order. The arm is read off the
model (``features_extractor.hypothesis_builder`` is built iff ``--belief-tokens fixed_mass``), never
assumed.

Per decision, three COLUMNS of belief (each a per-species presence π over the structural candidate set
V = the dex-row table's valid nums minus the revealed ones, Σ_V π = k = the hidden count):

* ``fixed_mass`` — the arm's OWN presence, ``hs.species.pi`` (σ(log P_T0 + δ_θ + τ));
* ``blob`` — the blob's own species belief, BeliefHead's per-slot species logits reduced to the
  hidden-slot mean and put through the SAME fixed-size construction (``belief_head_team_scores`` —
  §4.2 R3's "the fixed-size marginal of BeliefHead's posterior for the blob, so the two are comparable");
* ``prior`` — the Smogon T0 prior alone (``species_team_prior_logits`` → the same construction), the
  third column (§4.2), identical for every checkpoint.

The opponent-intent read (the COMMON EVENT SPACE, `bank_rows`): each arm's heads induce a probability
on the realised event,

* a move event m:   P = Σ_k α_k · [seat k's event = m]  (a Hidden Power seat — 237 or a typed channel —
  is the one HP event);
* a switch to s:    P = α_SWITCH · Σ_j β_j · c_j(s) over β's legal slots, where c_j is slot j's CONTENT:
  a revealed slot is its species; a hidden slot is, under ``blob``, BeliefHead's species posterior for
  that slot (softmax over V — the content β's training target is addressed by,
  ``resolve_believed_slot_by_content``), under ``fixed_mass`` the hypothesis species the seat holds.

Under ``fixed_mass`` (X5 U4, `gen3_x5_flat_pointer_v1`) α / β are retired and the FLAT pointer is
read instead (:func:`_flat_event_logp`): one softmax over [K move seats · OTHER_move · six switch
targets · OTHER_species], OTHER's content the renormalised tail (π_m / Σ π_m over OTHER_move's members;
``other_tail_probs`` for OTHER_species). A checkpoint whose forward stashes no flat logits takes the
α / β read.

An event no candidate can produce (a move outside the seats — and, under the flat pointer, outside
OTHER_move's members; a switch-in no slot's content supports) is a MISS: its own column, never floored
into the log loss (§7.4).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from main.belief_roles.bank_rows import SWITCH_BASE, BankRows
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
    read_tie_arm: np.ndarray         # [N] bool — fixed_mass: `near_tie_rows(hs)`; blob: the E4 seat cut
    other_mass_model: Optional[np.ndarray]   # [N] fixed_mass only: the model's own OTHER mass
    intent_logp: np.ndarray          # [N] float64 — log P(realised event); NaN where unlabeled or a miss
    intent_covered: np.ndarray       # [N] bool — the event is in the arm's support
    opp_active_species: np.ndarray   # [N] int — the opponent active's num (the miss breakdown)
    role_M_arm: np.ndarray           # [N,R] summed presence mass per role
    role_V_arm: np.ndarray           # [N,R] Σ p(1 − p) over the contributing indicators
    role_M_prior: np.ndarray
    role_V_prior: np.ndarray


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_strict(zip_path: Path) -> Any:
    """The checkpoint through the strict loader, on the CPU, eval mode (a bare run dir is REFUSED: it
    would resolve to the run's last snapshot, which moves)."""
    from agents.model.oracle_reveal import refuse_if_revealed
    from agents.model.snapshot import historical_load_kwargs, load_checkpoint_strict

    refuse_if_revealed(str(zip_path), tool="main.belief_roles",
                       reason="The Lane S bank's rows are encoded without the reveal.")
    zip_path = Path(zip_path)
    if zip_path.is_dir() or zip_path.suffix != ".zip":
        raise ValueError(f"{zip_path}: name the checkpoint .zip — a bare run directory resolves to the "
                         "run's LAST snapshot and moves")
    m = load_checkpoint_strict(str(zip_path), device="cpu", **historical_load_kwargs(str(zip_path)))
    m.policy.eval()
    return m


def arm_of(model: Any) -> str:
    fe = model.policy.features_extractor
    if getattr(fe, "hypothesis_builder", None) is not None:
        return "fixed_mass"
    if getattr(fe, "belief_head", None) is None or getattr(fe, "t0_species_prior", None) is None:
        raise ValueError("the checkpoint has no BeliefHead / T0 species prior — neither X5 arm "
                         "(`--belief-tokens blob` needs both for the presence read)")
    return "blob"


def _boundary_tie(pi, cand, k):
    """[B] bool — the k-th and (k+1)-th π (in the one stable order) differ by < TIE_EPS."""
    from agents.model.hypothesis_set import boundary_gap, stable_order

    order = stable_order(pi, cand)
    return boundary_gap(pi.gather(-1, order), k, cand.sum(-1)) < TIE_EPS


def _event_logp(events, alpha_logits, seat_nums, beta_logits, beta_ok, content_logp):
    """``(logp [B] float64 (−inf on a miss), covered [B] bool)`` of each row's realised event under one
    arm's heads (module docstring). ``content_logp`` [B,6,S] — log c_j(s), −inf off the support."""
    import torch

    B, K1 = alpha_logits.shape
    K = K1 - 1
    la = torch.log_softmax(alpha_logits.double(), dim=-1)                     # −inf on masked seats
    ev = _hp_collapse(seat_nums.long())
    seat_ok = torch.isfinite(la[:, :K]) & (seat_nums > 0)
    is_move = (events >= 0) & (events < SWITCH_BASE)
    is_sw = events >= SWITCH_BASE
    hit = seat_ok & (ev == events.unsqueeze(-1)) & is_move.unsqueeze(-1)
    neg = torch.full((), -math.inf, dtype=torch.float64)
    lp_move = torch.logsumexp(torch.where(hit, la[:, :K], neg), dim=-1)
    lb = torch.log_softmax(beta_logits.double().masked_fill(~beta_ok, -math.inf), dim=-1)
    S = content_logp.shape[-1]
    sp = (events - SWITCH_BASE).clamp(0, S - 1)
    lc = content_logp.double().gather(-1, sp.view(B, 1, 1).expand(B, TEAM, 1)).squeeze(-1)  # [B,6]
    term = torch.where(beta_ok & torch.isfinite(lc), lb + lc, neg)
    lp_sw = la[:, K] + torch.logsumexp(term, dim=-1)
    covered = (hit.any(-1) & is_move) | ((beta_ok & torch.isfinite(lc)).any(-1) & is_sw)
    lp = torch.where(is_move, lp_move, torch.where(is_sw, lp_sw, neg))
    return torch.where(covered, lp, neg), covered


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
                 batch: int = BATCH) -> Columns:
    """The batched forward (module docstring)."""
    import torch

    from agents.model.extra_obs_keys import zero_extra_obs
    from agents.model.hypothesis_dex_rows import load_hypothesis_dex_rows
    from agents.model.hypothesis_set import (belief_head_team_scores, fixed_mass_presence,
                                             near_tie_rows, species_candidates)
    from agents.model.t0_species import species_team_prior_logits
    from main.belief_roles.roles import prior_move_probs
    from main.policy_spectrum.reader import inference_globals

    arm = arm_of(model)
    fe = model.policy.features_extractor
    S = int(fe.layout["max_species"])
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
            if arm == "fixed_mass":
                assert hs is not None, "fixed_mass forward stashed no hypothesis set"
                pi_arm = hs.species.pi.double()
                read_tie = near_tie_rows(hs)
                put("other_mass_model", hs.other_mass.double())
            else:
                sl = st.belief_logits["species"].double()                       # [B,6,S]
                pi_arm = fixed_mass_presence(belief_head_team_scores(sl, believed), cand, k).pi
                # the blob's E4 seats are `torch.topk(w_all, K)` (F-X5-13): a near-tie at the cut
                w_all = fe.damage_op.last_w_all.double()
                Kseat = int(st.alpha_seat_nums.shape[-1])
                top = w_all.topk(min(Kseat + 1, w_all.shape[-1]), dim=-1).values
                read_tie = (top[:, Kseat - 1] - top[:, Kseat]) < TIE_EPS
            sel_tie_arm = _boundary_tie(pi_arm, cand, k)
            if arm == "fixed_mass":
                sel_tie_arm = sel_tie_arm | read_tie
            put("k", k)
            put("cand", cand)
            put("revealed_nums", torch.where(rev, opp_ids, torch.zeros_like(opp_ids)))
            put("pi_arm", pi_arm)
            put("pi_prior", pi_prior)
            put("sel_tie_arm", sel_tie_arm)
            put("sel_tie_prior", _boundary_tie(pi_prior, cand, k))
            put("read_tie_arm", read_tie)
            # ---- opponent intent on the common event space
            events = torch.tensor(br.event[i:i + batch], dtype=torch.long)
            active = ctx.hp_and_active[:, TEAM:2 * TEAM, -1] > 0.5
            beta_ok = ctx.opp_addressable.bool() & ~active
            content = torch.full((B, TEAM, S), -math.inf, dtype=torch.float64)
            content.scatter_(-1, opp_ids.clamp(0, S - 1).unsqueeze(-1),
                             torch.where(rev, 0.0, -math.inf).double().unsqueeze(-1))
            if arm == "fixed_mass":
                hyp = torch.full((B, TEAM, S), -math.inf, dtype=torch.float64)
                hyp.scatter_(-1, hs.slot_species.long().clamp(0, S - 1).unsqueeze(-1),
                             torch.where(hs.slot_is_hypothesis & (hs.slot_species > 0), 0.0,
                                         -math.inf).double().unsqueeze(-1))
            else:
                sl = st.belief_logits["species"].double()
                hyp = torch.log_softmax(sl.masked_fill(~cand.unsqueeze(1), -math.inf), dim=-1)
            content = torch.where(believed.unsqueeze(-1), hyp, content)
            if getattr(st, "flat_intent_logits", None) is not None:          # U4: fixed_mass
                assert hs is not None and hs.moves is not None and st.flat_intent is not None
                lp, cov = _flat_event_logp(events, st.flat_intent_logits, st.flat_intent,
                                           hs.moves.presence.pi, hs.other_tail_probs)
            else:
                lp, cov = _event_logp(events, st.alpha_logits, st.alpha_seat_nums, st.beta_logits,
                                      beta_ok, content)
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
            if arm == "fixed_mass":
                sp = hs.slot_pi.double()
                q = believed.double().unsqueeze(-1) * sp.unsqueeze(-1) * ph
                tail = cand & (hs.rank >= k.unsqueeze(-1)) & (k > 0).unsqueeze(-1)
                w = torch.where(tail, pi_arm, torch.zeros_like(pi_arm))
                M_arm = M_rev + q.sum(1) + w @ P_roles
                V_arm = V_rev + (q * (1 - q)).sum(1) + w @ P_roles - (w * w) @ (P_roles * P_roles)
            else:
                hidf = believed.double().unsqueeze(-1)
                M_arm = M_rev + (hidf * ph).sum(1)
                V_arm = V_rev + (hidf * ph * (1 - ph)).sum(1)
            put("role_M_arm", M_arm)
            put("role_V_arm", V_arm)
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
                   role_M_prior=cat["role_M_prior"], role_V_prior=cat["role_V_prior"])
