"""A fork's OWN TensorBoard rows, one SEGMENT per process (the X35 amendments' shared reader, 2026-10-19).

A fork INHERITS its parent's TensorBoard rows at steps <= the fork step (``agents.training.tb_inherit``,
``gen3_tb_inherit_v1``: one file named :data:`~agents.training.tb_inherit.INHERITED_EVENTS_BASENAME`, provenance in
``tb/INHERITED_FROM.json``). Every X35 read of an in-run quantity (K9(b) at arm end, AM-6; X47's effective learn time,
AM-4) must read the fork's OWN rows only, so this module:

* lists the run's event files EXCEPT the inherited one, each file one process (one SEGMENT), ordered by the file's
  first event wall time (a restart writes a new file);
* drops every row at a step <= the fork step (belt and braces: the inherited file is already excluded);
* never fills an absent tag with zero (absence is not a measurement; the caller decides what it means).

Read-only. ``tensorboard``'s ``EventAccumulator`` is imported lazily.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from agents.training.tb_inherit import INHERITED_EVENTS_BASENAME, PROVENANCE_BASENAME

Rows = List[Tuple[int, float]]


@dataclass(frozen=True)
class Segment:
    """One process's rows: ``{tag: [(step, value), ...]}`` sorted by step, from one event file."""

    path: str
    first_wall_time: float
    rows: Dict[str, Rows]


def tb_dir(run_dir: str) -> str:
    return os.path.join(run_dir, "tb")


def fork_step(run_dir: str) -> Optional[int]:
    """The step the fork inherited up to: ``tb/INHERITED_FROM.json``'s ``fork_step``, else the run's recorded lineage
    block's ``fork_step``; ``None`` when neither says (the caller refuses: an unknown fork step would let the
    parent's rows in)."""
    p = os.path.join(tb_dir(run_dir), PROVENANCE_BASENAME)
    if os.path.isfile(p):
        try:
            v = json.loads(open(p, encoding="utf-8").read()).get("fork_step")
            if v is not None:
                return int(v)
        except (OSError, ValueError, TypeError):
            pass
    from agents.training.lineage import read_block

    block = read_block(run_dir) or {}
    v = block.get("fork_step")
    return int(v) if v is not None else None


def own_event_files(run_dir: str) -> List[str]:
    """Every event file of the run's ``tb/`` except the inherited prefix (name order; the caller orders by time)."""
    return sorted(f for f in glob.glob(os.path.join(tb_dir(run_dir), "*tfevents*"))
                  if os.path.basename(f) != INHERITED_EVENTS_BASENAME and ".inherited." not in os.path.basename(f))


def load_segments(run_dir: str, tags: Sequence[str], *, after_step: Optional[int]) -> List[Segment]:
    """The run's own segments in chronological order, each with every requested tag's rows at steps > ``after_step``
    (``None`` keeps every row). A tag a file does not carry is absent from that segment's ``rows``."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    segs: List[Segment] = []
    for f in own_event_files(run_dir):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        avail = set(ea.Tags().get("scalars", []))
        rows: Dict[str, Rows] = {}
        first = float("inf")
        for t in tags:
            if t not in avail:
                continue
            evs = ea.Scalars(t)
            first = min([first] + [e.wall_time for e in evs])
            rows[t] = sorted((int(e.step), float(e.value)) for e in evs
                             if after_step is None or int(e.step) > after_step)
        if rows:
            segs.append(Segment(f, first, rows))
    segs.sort(key=lambda s: (s.first_wall_time, s.path))
    return segs
