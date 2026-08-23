"""Per-invocation FLAGS (`switch` / `uncertain` / `faint` / `opp-switch` / `cure-skipped`).

Model-free: everything here reads the recorded summary dict. The cure helpers are data-driven off
the `gen3_data` facade rather than a hardcoded move-id list — see the "heal ≠ cure" trap in
`main/prober/CLAUDE.md`.
"""

from __future__ import annotations

from agents import gen3_data

from main.prober.engine.timeline import opp_voluntary_switch
from main.prober.engine.util import parse_pct


# A decision is "uncertain" when even the chosen (top) action's recorded prob is below
# this — a genuine tossup (≈3-way), not merely a non-dominant pick (common in Pokémon).
UNCERTAIN_THRESHOLD = 0.34


# A real STATUS (curable by Refresh / Heal Bell), as opposed to a VOLATILE. The recorder bundles both
# into one display string ("TOX(2)|TAUNT"), so a cure check must split them — Taunt is not curable, and
# it also makes every status move ILLEGAL, which the action-validity check below picks up on its own.
_CURABLE_STATUSES = frozenset({"TOX", "PSN", "BRN", "PAR", "SLP", "FRZ"})


def has_curable_status(status: str) -> bool:
    """True when a recorder status string ("TOX(2)|TAUNT", "PAR", "") carries a real status (not just
    volatiles). Pure — the recorder's own bundling format is the only contract."""
    return any(tok.split("(")[0].strip().upper() in _CURABLE_STATUSES
               for tok in str(status or "").split("|"))


def is_status_cure(move_id: str) -> bool:
    """True for a move that CLEARS status — Refresh (self) or Heal Bell / Aromatherapy (team).
    Read from the DATA facade's `curesSelfStatus` / `curesTeamStatus`, never a hardcoded id list
    (`data/` is the source of truth). Rest is NOT one: it cures by *inflicting* sleep."""
    md = gen3_data.moves.get(str(move_id or "").split(":")[0])
    return bool(md and (md.cures_self_status or md.cures_team_status))


def self_cure_options(inv: dict) -> "tuple[str, ...]":
    """The LEGAL status-cure moves at this decision, but ONLY when our active actually carries a
    curable status — i.e. the cures that would have *done something*. Empty otherwise.

    Model-free (summary only). This is the "was a cure on the table" question the raw action list
    can't answer: an illegal (Taunted / no-PP) cure and a cure with nothing to cure both read as a
    move label, and neither is a real option."""
    if not has_curable_status((inv.get("our") or {}).get("status")):
        return ()
    return tuple(lbl for lbl, a in (inv.get("actions") or {}).items()
                 if a.get("valid") and is_status_cure(lbl))


def summary_flags(inv: dict, uncertain_threshold: float = UNCERTAIN_THRESHOLD) -> "tuple[str, ...]":
    """Cheap, model-free per-invocation flags (used by the TUI list + agent API to
    jump to interesting decisions): a switch, an uncertain (low top-prob) call, or a
    turn whose recorded events include a faint."""
    flags = []
    chosen = inv.get("chosen", "")
    if chosen.startswith("switch"):
        flags.append("switch")
    acts = inv.get("actions", {})
    if chosen in acts and parse_pct(acts[chosen].get("prob", "0%")) < uncertain_threshold:
        flags.append("uncertain")
    events = (inv.get("outcome") or {}).get("events") or []
    if any("faint" in str(e).lower() for e in events):
        flags.append("faint")
    # The OPPONENT voluntarily pivoted this turn → our move RESOLVED against a switch-IN, not the active we
    # computed damage against (the "computed-vs ≠ resolved-vs" trap — see `opp_voluntary_switch`).
    if opp_voluntary_switch(inv):
        flags.append("opp-switch")
    # Statused, a LEGAL cure was on the table, and we did something else (Recover heals HP but never
    # clears status — a distinction that is invisible in a move list).
    if self_cure_options(inv) and not is_status_cure(chosen):
        flags.append("cure-skipped")
    return tuple(flags)
