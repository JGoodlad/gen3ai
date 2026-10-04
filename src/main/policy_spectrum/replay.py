"""RE-ENCODE a banked battle: its input log through the Rust core → every decision's obs row, mask
and legal-action tokens, per viewer (`gen3_policy_spectrum_replay_v1`, M5 Lane S gate ①).

The bank stores INPUTS (the sim's seed, both players' packed teams, the command log), never obs
vectors. This module turns them back into rows with the port's ``core_events --trackers --obs``
(the same replay the Rust Core Program's slice O gates against the Python encoder, and whose rows
Lane 0's gate ① pins byte-equal to ``sim_bridge``'s ``__OBS__`` frames). A future architecture reads
the SAME turns by re-encoding through ITS encoder at ITS commit.

A decision is a tracker entry that carries an ``obs`` row; entries are in decision order per viewer,
so a decision's index ``n`` is stable for a given input log.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_TURN_RE = re.compile(r"^\|turn\|(\d+)")


@dataclass
class ReplayDecision:
    """One viewer's decision, re-derived from the input log."""

    n: int                      # decision index within the viewer (obs-bearing entries only)
    turn: int                   # the latest `|turn|N` the viewer has seen (0 before turn 1)
    mask: np.ndarray            # int8 [11]
    tokens: Dict[int, str]      # action index -> choice token, legal actions only
    choice: Optional[str]       # the token the log played here
    row: np.ndarray             # float32 [obs_dim], read-only view over the wire bytes
    request: dict               # the viewer's `|request|` JSON for this decision
    opp_fainted: int            # the viewer's opponent's faints seen so far
    #: the trackers' intent label at THIS entry: what the opponent did at the viewer's PREVIOUS
    #: decision (``trackers::IntentLabel``, the α/β label's source; `opp_intent_labels`).
    label: Optional[dict] = None
    #: the label of the NEXT obs-bearing entry (an unanswered final request included): what the
    #: opponent did at THIS decision — the label training aligns to this row
    #: (``align_labels_to_predictions``); None when this is the viewer's last entry.
    next_label: Optional[dict] = None


@dataclass
class ReplayBattle:
    ok: bool
    error: Optional[str]
    winner: Optional[str]
    decisions: Tuple[List[ReplayDecision], List[ReplayDecision]]   # (p1, p2)


def _viewer_decisions(res: dict, viewer: int) -> List[ReplayDecision]:
    from agents.battle.core_obs import wrap_row

    chunks = res["chunks"]
    opp_tag = f"|faint|p{2 - viewer}a:"
    out: List[ReplayDecision] = []
    for entry in res["trackers"][viewer]:
        if "obs" not in entry:
            continue
        after = int(entry["after"])
        request, turn, fainted = None, 0, 0
        for side, lines in chunks[: after + 1]:
            if side != viewer:
                continue
            for line in lines:
                if line.startswith("|request|"):
                    request = json.loads(line[len("|request|"):])
                elif line.startswith(opp_tag):
                    fainted += 1
                else:
                    m = _TURN_RE.match(line)
                    if m:
                        turn = int(m.group(1))
        if request is None:
            raise RuntimeError(f"{res['label']} viewer {viewer} decision {len(out)}: no |request| "
                               f"in the viewer's chunks up to {after}")
        mask = np.asarray(entry["mask"], dtype=np.int8)
        tokens = {int(k): v for k, v in entry["tokens"].items()}
        if sorted(tokens) != [int(i) for i in np.nonzero(mask)[0]]:
            raise RuntimeError(f"{res['label']} viewer {viewer} decision {len(out)}: tokens "
                               f"{sorted(tokens)} disagree with the mask {mask.tolist()}")
        out.append(ReplayDecision(n=len(out), turn=turn, mask=mask, tokens=tokens,
                                  choice=entry.get("choice"), row=wrap_row(entry["obs"]),
                                  request=request, opp_fainted=fainted,
                                  label=(entry.get("trackers") or {}).get("label")))
    for a, b in zip(out, out[1:]):
        a.next_label = b.label
    return out


def _run_chunk(battles) -> List[dict]:
    from agents.battle.rust_core_parity import run_core

    return run_core(battles, trackers=True, obs=True)


def replay(recorded: Sequence, workers: int = 2, chunk: int = 32) -> List[ReplayBattle]:
    """Replay ``recorded`` (``RecordedBattle``s) through ``core_events``; one result per battle, in
    order. ``workers`` core processes run concurrently, ``chunk`` battles each. Only ANSWERED
    decisions are returned (a forfeited or end-of-battle request is dropped)."""
    parts = [list(recorded[i:i + chunk]) for i in range(0, len(recorded), chunk)]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        results = [r for part in ex.map(_run_chunk, parts) for r in part]
    out = []
    for res in results:
        if not res["ok"]:
            out.append(ReplayBattle(False, str(res.get("error")), None, ([], [])))
            continue
        views = []
        for v in (0, 1):
            ds = _viewer_decisions(res, v)
            # A request the log never answered (the stall FORFEIT's `forcelose`, or a request open
            # when the battle ended) is not a decision anyone took: it may only be the LAST one.
            unplayed = [d.n for d in ds if d.choice is None]
            if unplayed and unplayed != [ds[-1].n]:
                raise RuntimeError(f"{res['label']} viewer {v}: unanswered requests at {unplayed} "
                                   "(only the final request may be unanswered)")
            views.append([d for d in ds if d.choice is not None])
        out.append(ReplayBattle(True, None, res.get("winner"), (views[0], views[1])))
    return out


def core_events_identity() -> dict:
    """What produced the rows: the resolved ``core_events`` binary and its content hash (a row is a
    property of the encoder that wrote it)."""
    import hashlib

    from utils.bridge.sim_bridge_bin import resolve_core_events_bin

    path = resolve_core_events_bin()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return {"core_events": path, "core_events_sha256": h.hexdigest()}
