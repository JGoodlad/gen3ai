"""gen3_op_status_rules_v1 — the ONE declaration of the gen-3 side / clause rules that decide whether an
INCOMING status can land on one of our six, shared by the damage operator (its production source,
`DamageOperatorBlocks._incoming_status_mask`) and the move-resolution family (`move_resolution.
incoming_status_correction`, which re-applies it as a no-op on the op's already-masked grid so its own contract
stays self-contained).

Every rule verified at the source (`deps/pokemon-showdown`, gen 3 resolved through gen 4 → base). WHETHER a clause
is in force is the FORMAT SPEC's call, never a constant here (`agents.gen3_data.format_spec`: `sleep_clause_mod()` /
`freeze_clause_mod()`, read at call time — `design_format_spec.md`); the mechanics below are what the clause does
when it is (`main.format_drift` fails if master's clause body ever differs from the pinned one verified here):

* **our Safeguard** (`data/moves.ts` safeguard: `onSetStatus` returns null for any status a FOE sets — a
  secondary included; `onTryAddVolatile` stops their Yawn): all six columns, every defender (a side condition).
* **incoming Sleep Clause** (`data/rulesets.ts:1378-1402`): a LIVE mon of ours asleep from a non-Rest source
  blocks their sleep — the slp column, every defender.
* **Freeze Clause** (`data/rulesets.ts:1451-1471`): ANY frozen mon of ours (no HP check) blocks a freeze — the
  frz column, every defender.
* **our Substitute** (`gen4/moves.ts:1283-1320`: their status move fails on it, their secondary hits the
  substitute): our ACTIVE's row only — a bench mon switching in has none.
"""
from __future__ import annotations

import torch

from agents.gen3_data import format_spec

#: The six major-status columns, in `SECONDARY_COLS[:6]` / `MOVE_STATUS_IDENT` order.
STATUS_COLS = ("par", "brn", "frz", "slp", "psn", "tox")
COL_FRZ = STATUS_COLS.index("frz")
COL_SLP = STATUS_COLS.index("slp")


def incoming_status_mask(sg_ours: torch.Tensor, our_slp: torch.Tensor, our_frz: torch.Tensor,
                         our_alive: torch.Tensor, our_rest: torch.Tensor, our_sub: torch.Tensor,
                         our_active: torch.Tensor) -> torch.Tensor:
    """``[B,6,6]`` the 0/1 multiplier on P(their status lands) per (OUR defender, status column).

    ``sg_ours`` [B] our Safeguard flag · ``our_slp`` / ``our_frz`` / ``our_alive`` / ``our_rest`` [B,6] our six's
    sleep / freeze flags, alive (HP > 0) and the Rest-sleep flag · ``our_sub`` [B] our active's Substitute flag ·
    ``our_active`` [B] long, our active's slot. Every input is public and binary, so the mask is exact and
    idempotent (applying it twice is applying it once)."""
    dt = sg_ours.dtype
    clause_slp = ((our_slp * our_alive * (1.0 - our_rest)).sum(-1) > 0.5).to(dt)       # [B]
    clause_frz = (our_frz.sum(-1) > 0.5).to(dt)                                       # [B]
    if not format_spec.sleep_clause_mod():                                            # the format spec's call
        clause_slp = torch.zeros_like(clause_slp)
    if not format_spec.freeze_clause_mod():
        clause_frz = torch.zeros_like(clause_frz)
    one = torch.ones_like(sg_ours)
    cols = [one] * len(STATUS_COLS)
    cols[COL_FRZ] = 1.0 - clause_frz
    cols[COL_SLP] = 1.0 - clause_slp
    col = torch.stack(cols, dim=-1)                                                   # [B,6]
    act = torch.nn.functional.one_hot(our_active, our_slp.shape[-1]).to(dt)          # [B,6]
    mask: torch.Tensor = ((1.0 - sg_ours)[:, None, None] * col[:, None, :]
                          * (1.0 - act * our_sub[:, None])[:, :, None])               # [B,6,6]
    return mask
