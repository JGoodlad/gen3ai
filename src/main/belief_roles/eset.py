"""Amendment 3(b) (`design_x5_belief_tokens.md` §7.7(b)): blob's NAMED SET E_row and each arm's full
distribution on the COMMON EVENT SPACE, so purpose metric (1) can be scored on the same rows and the same
support for both arms.

**E_row** = a blob checkpoint's named candidate set on a row: the HP-collapsed move events of its E4
seats that α could name (a finite α logit and a seat num > 0), and the switch-ins its β could name. It
was exactly the support of the blob's induced distribution, decided STRUCTURALLY (membership, never a
value). The blob reads of the X5 A/B wrote one per blob run beside their JSON (``<label>.erow.npz``,
:class:`ERow`); the blob read arm that computed them was DELETED with the blob path at the X5 version
break (config v144) — a blob checkpoint is read PINNED to its own commit — so this module now only
LOADS those banked sets.

**An X5 run has no blob heads**, so it is scored on EVERY blob run of the look, each set read on the same
bank rows, and its value is the MEAN of the conditional log loss over those sets (orchestrator decision
2026-10-06, the Decision record of §7.7: "seeds do not pair runs", §7.4, and the mean mirrors the cross
design); a read takes them through ``--reference``. ``infer`` refuses a conditional-metric comparison in
which any fixed_mass read does not reference exactly the control group's blob runs.

The event space is DENSE: index e < :data:`SWITCH_BASE` is a move num (every Hidden Power → 237), index
``SWITCH_BASE + s`` a switch-in by species s, so ``D = SWITCH_BASE + S``.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np

from main.belief_roles.bank_rows import SWITCH_BASE

TEAM = 6
#: Rule 8 (§7.7(b)): a renormalisation denominator within this of 0 excludes the row (counted).
DENOM_EPS = 1e-12
ERow_SCHEMA = "gen3_x5_blob_named_set_v1"


def hp_collapse(nums):
    """Move nums on the common event space: 237 and every typed Hidden Power (355–370) → 237."""
    import torch

    from main.belief_roles.roles import HP_NUM

    typed = (nums >= 355) & (nums <= 370)
    return torch.where(typed | (nums == HP_NUM), torch.full_like(nums, HP_NUM), nums)


@dataclass
class ERow:
    """A blob checkpoint's named set on every bank row (module docstring) — loaded from a banked
    ``.erow.npz``."""

    seat_events: np.ndarray          # [N,K] int64 — the namable seats' move events, 0 = no seat
    switch_ok: np.ndarray            # [N,S] bool — the switch-ins β can name
    tie: np.ndarray                  # [N] bool — the E4 seat cut is a rule-8 near-tie (set ill-determined)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.seat_events.shape[0])

    def content_sha256(self) -> str:
        h = hashlib.sha256()
        for a in (self.seat_events.astype(np.int64), np.packbits(self.switch_ok, axis=1),
                  self.tie.astype(np.uint8)):
            h.update(str(a.shape).encode())
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def save(self, path: Path) -> None:
        np.savez_compressed(path, seat_events=self.seat_events.astype(np.int64),
                            switch_ok=np.packbits(self.switch_ok, axis=1),
                            n_species=np.int64(self.switch_ok.shape[1]), tie=self.tie,
                            meta=np.array(json.dumps(self.meta, sort_keys=True)))

    @classmethod
    def load(cls, path: Path) -> "ERow":
        z = np.load(path, allow_pickle=False)
        S = int(z["n_species"])
        return cls(seat_events=z["seat_events"], tie=z["tie"].astype(bool),
                   switch_ok=np.unpackbits(z["switch_ok"], axis=1, count=S).astype(bool),
                   meta=json.loads(str(z["meta"])))

    def mask(self, lo: int, hi: int, D: int):
        """``[B,D]`` torch bool — rows ``lo:hi`` of E_row on the dense event space."""
        import torch

        return set_mask(torch.from_numpy(self.seat_events[lo:hi].astype(np.int64)),
                        torch.from_numpy(self.switch_ok[lo:hi]), D)


def set_mask(seat_events, switch_ok, D: int):
    """``[B,D]`` bool on the dense event space from a named set's seat events (0 = none) and switch-ins."""
    import torch

    se = seat_events.long()
    m = torch.zeros(se.shape[0], D, dtype=torch.bool)
    m.scatter_(1, se, torch.ones_like(se, dtype=torch.bool))
    m[:, 0] = False                                                  # 0 = an empty seat, never an event
    S = switch_ok.shape[1]
    m[:, SWITCH_BASE:SWITCH_BASE + S] = switch_ok.bool()
    return m


@dataclass
class ESet:
    """One E_row's per-row quantities for one arm (``forward.read_columns``)."""

    mass: np.ndarray                 # [N] Σ_{e ∈ E_row} P_arm(e) — the renormalisation denominator
    e_in: np.ndarray                 # [N] bool — the realised event is in E_row
    tie: np.ndarray                  # [N] bool — E_row's own seat cut is a rule-8 near-tie
    label: str = "own"               # the reference blob run's label
    checkpoint_sha256: str = ""      # the reference blob checkpoint


def flat_event_dist(flat_logits, fi, pi_m, p_tail):
    """``P [B,D] float64`` — U4's FLAT pointer on the dense event space: a seat → its move (HP
    collapsed); OTHER_move → π_m / Σ π_m over its members; a slot → its species; OTHER_species → the tail
    ``p_tail`` over ``fi.in_tail`` (the same contents :func:`forward._flat_event_logp` scores)."""
    import torch

    from agents.model.flat_intent import other_move_col, other_species_col, slot_col

    K = int(fi.k)
    B = flat_logits.shape[0]
    S = fi.in_tail.shape[1]
    M = fi.beyond.shape[1]
    if M > SWITCH_BASE:
        raise ValueError(f"the move axis ({M}) overlaps the switch events (SWITCH_BASE {SWITCH_BASE})")
    D = SWITCH_BASE + S
    zero = torch.zeros((), dtype=torch.float64)
    live = fi.live.bool()
    pf = torch.log_softmax(flat_logits.double().masked_fill(~live, -math.inf), dim=-1).exp()
    pf = torch.where(live, pf, zero)
    P = torch.zeros(B, D, dtype=torch.float64)
    seat_ev = torch.where(live[:, :K], hp_collapse(fi.cand_ids[:, :K].long()),
                          torch.zeros_like(fi.cand_ids[:, :K].long()))
    P.scatter_add_(1, seat_ev, pf[:, :K])
    om = other_move_col(K)
    w = torch.where(fi.beyond, pi_m.double(), zero)
    wn = w / w.sum(-1, keepdim=True).clamp(min=1e-300)
    mem_ev = hp_collapse(torch.arange(M).unsqueeze(0).expand(B, M))
    P.scatter_add_(1, mem_ev, pf[:, om:om + 1] * wn)
    c0 = slot_col(K)
    slot_sp = fi.cand_ids[:, c0:c0 + TEAM].long().clamp(0, S - 1)
    sl_live = live[:, c0:c0 + TEAM]
    P.scatter_add_(1, SWITCH_BASE + torch.where(sl_live, slot_sp, torch.zeros_like(slot_sp)),
                   pf[:, c0:c0 + TEAM])
    os_ = other_species_col(K)
    P[:, SWITCH_BASE:] += pf[:, os_:os_ + 1] * torch.where(fi.in_tail, p_tail.double(), zero)
    P[:, 0] = 0.0                                                    # 0 is never an event
    P[:, SWITCH_BASE] = 0.0
    return P


def event_index(events, D: int):
    """``(idx [B] long, ok [B] bool)`` — each realised event's dense index; ``ok`` False when the row is
    unlabelled or its event lies outside the dense space."""
    import torch

    ok = (events > 0) & (events < D)
    return torch.where(ok, events, torch.zeros_like(events)), ok
