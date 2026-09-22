"""The ONE normalization both replays share — so a mismatch is a parser disagreement, not a spelling.

🚨 This module is imported by BOTH sides of the parity check, and the two sides run under DIFFERENT
interpreters with DIFFERENT ``poke_env`` packages (`gen3ai_stable` + our vendored fork on one side,
the `metamon` env + upstream poke-env 0.8.3.3 on the other — root ``CLAUDE.md``'s two-packages
hazard, anchors H11). It therefore imports NOTHING but the standard library. A shared helper that
pulled in either package would make one of the two replays impossible to run.

The comparison schema is deliberately the INTERSECTION of what the two parsers both claim to know,
canonicalised so that a difference in spelling (``BRN`` vs ``brn``, ``Tyranitar`` vs ``tyranitar``)
can never be reported as a difference in state.
"""

import json
import re

_NON_ALNUM = re.compile(r"[^a-z0-9]")


def canon(value) -> str:
    """Lowercase, strip every non-alphanumeric character. ``None`` becomes ``""``."""
    if value is None:
        return ""
    return _NON_ALNUM.sub("", str(value).lower())


def hp(fraction) -> float:
    """HP as a fraction, rounded to 4 dp.

    Both sides store a float ratio; rounding kills the last-bit noise of
    ``current_hp / max_hp`` computed in a different order without hiding a real difference
    (1/48 of a bar is 0.0208, three orders of magnitude above this).
    """
    if fraction is None:
        return -1.0
    return round(float(fraction), 4)


def boosts(mapping) -> dict:
    """The seven boost counters, always all seven, always ints."""
    mapping = mapping or {}
    out = {}
    for key in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion"):
        out[key] = int(mapping.get(key, 0) or 0)
    return out


def moveset(names) -> list:
    """Canonical, SORTED move ids. Order is a presentation rule; membership is not."""
    return sorted({canon(n) for n in (names or []) if canon(n)})


def conditions(mapping) -> dict:
    """Side conditions as ``{canonical name: int layer/turn counter}``."""
    return {canon(k): int(v) for k, v in (mapping or {}).items()}


def dump(path, rows) -> None:
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def load(path) -> list:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


#: Which keyword starts a new decision point. A ``|request|`` whose JSON carries ``wait`` is the
#: server telling a client "the other side is thinking" — poke-env does not ask the player for an
#: action on it, so it is not a decision point on either side.
def is_decision_request(request: dict) -> bool:
    if not request:
        return False
    if request.get("wait"):
        return False
    if request.get("teamPreview"):
        return False
    return True
