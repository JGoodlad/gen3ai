"""Opponent-NAME ordering: sentinels first, strongest first, then everything else."""

from __future__ import annotations

import re


#: Self-play pool opponents are named `sentinel_<k>` and the index is a STRENGTH RANK, not a
#: creation order: **sentinel_0 is the strongest**. The labels float — a promotion re-seats every
#: sentinel — so the number means "k-th best self right now" and nothing about age.
_SENTINEL_RE = re.compile(r"^sentinel_(\d+)$")


def opponent_rank(name: str, tiebreak: int = 0) -> "tuple[int, int, int]":
    """Sort key putting the SENTINELS first, strongest first, then everything else unchanged.

    A sentinel is the trainee's own recent self, so it is the opponent whose games say most about
    where the model is now — and on a run with five of them they were scattered alphabetically
    among nine bots, which is a scanning task rather than a choice. Bots keep their incoming order
    (`tiebreak`) rather than being re-sorted: this key answers "which of these matters most", and
    inventing a strength order for the fixed bots is a different claim, which the ELO ladder owns.

    Not a presentation detail smuggled into a view: the CLI and the browser should agree about
    which opponent leads a list, so the rule lives here with the reason attached.
    """
    m = _SENTINEL_RE.match(str(name or ""))
    if m:
        return (0, int(m.group(1)), tiebreak)      # sentinel_0 first — it is the STRONGEST self
    return (1, 0, tiebreak)


def sort_opponents(names: "list[str] | tuple[str, ...]") -> "list[str]":
    """`opponent_rank` applied to a list of opponent names, preserving the original order within
    the non-sentinel group."""
    return [n for _k, n in sorted(((opponent_rank(n, i), n) for i, n in enumerate(names or ())),
                                  key=lambda pair: pair[0])]
