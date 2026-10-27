"""``check-spec``: validate a spec, and say where its PIN stands (``designs/tools/reads.md`` §3, "The registration flow").

A spec's ``pin`` is the commit the PLAYING checkout must be at. It can never be "the commit that adds this spec" (a
commit cannot name its own hash), so the pin is a commit that ALREADY EXISTS: the one that lands the code the read plays,
an ancestor of the registration commit (PREREG + spec(s), one commit). The belief-prior-mix read (71c76dfb FINDING 8) landed
its PREREG and its specs in two commits because it pinned the PREREG's own commit. Two modes make the one-commit flow
checkable:

* ``check-spec`` (default): everything :func:`main.reads.spec.load` checks, plus, for a spec that carries a ``pin``, that the
  pin NAMES A COMMIT of this repo (a pin that cannot be played is refused here, not at ``plan``), with how far the play
  path at HEAD has moved from it (``play_path_differs_at_head``: 0 = the code you review at HEAD plays exactly as the
  pin does).
* ``check-spec --pin-pending``: review a spec BEFORE its pin is decided. The spec must carry no ``pin`` yet; all else is
  validated and the output is marked ``PENDING`` with ``sha256_final: false`` (adding the pin changes the spec's
  sha256, which a PREREG may quote), plus the checkout's HEAD as the candidate.

This module is OFF the play path (:mod:`main.reads.playpath`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from main.reads import playpath as PP
from main.reads.spec import SpecError, load as load_spec

SHOWN_DIFFS = 12


def _head(repo: Optional[Path]) -> Optional[str]:
    try:
        return PP.resolve("HEAD", repo)
    except PP.PlayPathError:
        return None


def check_spec(path: "str | Path", *, pin_pending: bool = False, repo: Optional[Path] = None) -> Dict[str, Any]:
    """The spec's summary plus its pin standing (module docstring). Refusals are :class:`SpecError`."""
    s = load_spec(path)
    out: Dict[str, Any] = {"read": s.read, "rule": s.rule, "models": sorted(s.models), "anchor": sorted(s.anchor),
                           "s": sorted(s.s), "sha256": s.sha256, "allow_partial_at_stop": s.allow_partial_at_stop}
    head = _head(repo)
    if pin_pending:
        if s.pin is not None:
            raise SpecError(f"{path}: the spec already carries pin {s.pin}; --pin-pending is for a spec that has none yet "
                            "(check it without the flag)")
        out.update(pin={"state": "PENDING", "candidate_head": head,
                        "next": "land the code the read plays, then add \"pin\": \"<that commit>\" (an existing commit, "
                                "never the one that adds the spec); the sha256 below changes when you do"},
                   sha256_final=False)
        return out
    out["sha256_final"] = True
    if s.pin is None:
        out["pin"] = None                   # unpinned: `plan` records the checkout it runs at, and a run holds to that
        return out
    try:
        full = PP.resolve(s.pin, repo)
    except PP.PlayPathError:
        raise SpecError(f"{path}: pin {s.pin} names no commit in this repo: a pin must exist BEFORE the spec (the commit "
                        "that lands the code the read plays, not the one that adds the spec); to review a spec whose pin "
                        "is not decided yet, leave `pin` out and use --pin-pending") from None
    info: Dict[str, Any] = {"commit": full, "head": head, "is_head": full == head}
    if head is not None and head != full:
        try:
            diffs, _dig, _n = PP.compare(full, head, repo)
            info.update(play_path_differs_at_head=len(diffs), files=diffs[:SHOWN_DIFFS])
        except PP.PlayPathError as exc:
            info.update(play_path_differs_at_head=None, error=str(exc))
    else:
        info.update(play_path_differs_at_head=0 if head == full else None, files=[])
    out["pin"] = info
    return out
