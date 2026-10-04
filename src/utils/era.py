"""The ERA label: ONE table, the run-name prefix, and the run-record readers.

Owner decision 2026-10-03: research eras are named after Hoenn towns in journey order, the first
being **Rustboro**. A run carries its era's two-letter code as a name prefix (``rb_x26_s1001``) and
records it in ``metadata.json`` as an IMMUTABLE ``era`` block written at creation. Everything that
needs an era (the naming rule, the metadata writer, the readers, the cross-era warning) imports it
from HERE — no other module spells a code or a name.

The Rustboro era starts with the X5 A/B and X26. A run without an ``era`` block is **pre-era**
history (``PRE_ERA``): reading it never guesses, and a resume never stamps one onto it.

Journey order (the main-story order in Pokemon Ruby / Sapphire / Emerald; Littleroot, Oldale and
Petalburg precede Rustboro and are not eras): Rustboro -> Dewford -> Slateport -> Mauville ->
Verdanturf -> Fallarbor -> Lavaridge -> Fortree -> Lilycove -> Mossdeep -> Sootopolis ->
Pacifidlog -> Ever Grande. Codes are two letters and unique (a code is only ever read as ``<code>_`` at the start of a run
name).

To start the next era: move ``CURRENT_ERA`` down this table (one edit) and record the boundary in
``designs/endstate/era_plan_post_m5.md``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class Era:
    order: int   # 1-based position in the journey
    name: str
    code: str    # two lowercase letters: the run-name prefix and the recorded ``era.code``

    def block(self) -> dict:
        """The immutable record written into ``metadata.json``."""
        return {"code": self.code, "name": self.name, "order": self.order}


#: THE era table — the single source of truth (name, code, order).
ERAS: Sequence[Era] = (
    Era(1, "Rustboro", "rb"),
    Era(2, "Dewford", "dw"),
    Era(3, "Slateport", "sp"),
    Era(4, "Mauville", "mv"),
    Era(5, "Verdanturf", "vt"),
    Era(6, "Fallarbor", "fb"),
    Era(7, "Lavaridge", "lv"),
    Era(8, "Fortree", "ft"),
    Era(9, "Lilycove", "lc"),
    Era(10, "Mossdeep", "md"),
    Era(11, "Sootopolis", "st"),
    Era(12, "Pacifidlog", "pd"),
    Era(13, "Ever Grande", "eg"),
)

#: The era every NEW run belongs to. Moving to the next era is this one line.
CURRENT_ERA: Era = ERAS[0]

#: What a run with no recorded ``era`` block reads as. A label, never a guess.
PRE_ERA = "pre-era"


def era_by_code(code: str) -> Optional[Era]:
    return next((e for e in ERAS if e.code == code), None)


def era_of_name(run_name: str) -> Optional[Era]:
    """The era a run NAME's prefix names (``rb_x26_s1001`` -> Rustboro), else None.

    Only a leading ``<code>_`` counts; ``rbx26`` or ``x26_rb`` name no era. A convenience read of
    the name — the RECORD (``read_run_era``) is the truth."""
    head, sep, _ = run_name.partition("_")
    return era_by_code(head) if sep else None


def prefixed(leaf: str, era: Optional[Era] = None) -> str:
    """``leaf`` with the era prefix — for a DEFAULT-minted name. Never double-prefixes.

    ``era`` defaults to ``CURRENT_ERA`` read AT CALL TIME (never bound at import)."""
    era = era or CURRENT_ERA
    return leaf if era_of_name(leaf) is not None else f"{era.code}_{leaf}"


def run_name_warning(run_name: str, era: Optional[Era] = None) -> Optional[str]:
    """The one-line warning for an EXPLICIT ``--run-name``, or None when it is fine.

    Decision (2026-10-03): an explicit name is ACCEPTED UNCHANGED. Silently prefixing would make
    the directory differ from what the operator typed (scripts, TUI badges, `--model models/<name>`
    refs and the ledger all quote the typed name); refusing would break every established
    convention. The era is recorded in ``metadata.json`` whatever the name says, so the prefix is
    a convenience for the eye, not the record — a warning is the proportionate response."""
    era = era or CURRENT_ERA
    named = era_of_name(run_name)
    if named is None:
        return (f"[Era] --run-name {run_name!r} has no era prefix; the current era is "
                f"{era.name} (``{era.code}_``). Accepted as typed — the run RECORDS era "
                f"{era.code} in metadata.json regardless. Prefer {era.code}_{run_name}.")
    if named != era:
        return (f"[Era] --run-name {run_name!r} carries era {named.name} (``{named.code}_``) but "
                f"the current era is {era.name} (``{era.code}_``). Accepted as typed; the run "
                f"RECORDS era {era.code}.")
    return None


# --- the run record ----------------------------------------------------------------------------

def creation_era_block(era: Optional[Era] = None) -> dict:
    """The block a NEW run records — ``CURRENT_ERA`` read at call time."""
    return (era or CURRENT_ERA).block()


def read_era_block(metadata: dict) -> Optional[dict]:
    """The ``era`` block of an already-loaded metadata dict, or None (pre-era / malformed)."""
    blk = metadata.get("era") if isinstance(metadata, dict) else None
    if isinstance(blk, dict) and isinstance(blk.get("code"), str) and blk.get("name"):
        return blk
    return None


def read_run_era(run_dir: str) -> Optional[dict]:
    """``<run_dir>/metadata.json``'s recorded ``era`` block, or None for a pre-era / unreadable run."""
    try:
        with open(os.path.join(run_dir, "metadata.json")) as fh:
            return read_era_block(json.load(fh))
    except (OSError, ValueError):
        return None


def era_label(block: Optional[dict]) -> str:
    """``rb Rustboro`` for a recorded era, ``pre-era`` for none."""
    return PRE_ERA if block is None else f"{block['code']} {block['name']}"


def cross_era_warning(run_a: str, run_b: str) -> Optional[str]:
    """A warning when two run dirs' recorded eras differ (pre-era counts as its own era), else None.

    Era boundaries are where something that makes ratings incomparable changed (the Rustboro
    boundary is the fixed bots + the Rustboro anchor base), so a cross-era delta is NOT a like-for-
    like reading and must say so."""
    a, b = read_run_era(run_a), read_run_era(run_b)
    la, lb = era_label(a), era_label(b)
    if la == lb:
        return None
    return (f"⚠ CROSS-ERA COMPARISON: {os.path.basename(os.path.normpath(run_a))} is {la}, "
            f"{os.path.basename(os.path.normpath(run_b))} is {lb}. Ratings, anchors and opponent "
            f"regimes are not defined across an era boundary — read this delta as descriptive, "
            f"never as an effect.")
