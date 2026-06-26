"""Phase 0 — model-free selection: rank ``(trace, turn)`` candidates worth spending search on.

The two-stage funnel (§3): cheap, model-free crater ranking → ``falsify``-gate to *reducible mistakes*
(not aleatoric luck — don't teach against dice) → expand to the crater ±window (the cause is usually
1–2 turns BEFORE the value crater — the Spore demo). Pure orchestration of the prober engine
(``ProbeSession.scan`` + ``falsifier.falsify_battle``); no model loaded, no training touched.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Candidate:
    summary_path: str   # absolute *_summary.json (the battle id)
    recon_path: str     # absolute *_reconstruction.json sibling (verified to exist)
    inv_index: int      # the move_selection invocation to search
    turn: int
    opponent: str       # the trace's opponent label (sentinel_N / heuristic / ext_<run>)
    step: int           # the eval step that produced the trace
    anchor_delta: float # |δ| crater magnitude — the priority weight (P(mistake) ≈ 1 post-gate)
    verdict: str        # 'MISTAKE' (proven) or 'CRATER' (un-falsified, when falsify_gate=False)


def _recon_for(summary_path: str) -> str:
    return summary_path[: -len("_summary.json")] + "_reconstruction.json"


def _move_inv_at_turn(summary: dict, turn: int) -> Optional[int]:
    """The move_selection invocation index at ``turn`` (for the ±window expansion), or None."""
    for i, inv in enumerate(summary.get("invocations", [])):
        if inv.get("phase") == "move_selection" and int(inv.get("turn", -1)) == int(turn):
            return i
    return None


def select_candidates(
    run_dir: str,
    *,
    budget: int = 200,
    scan_limit: int = 60,
    step: Optional[int] = None,
    falsify_gate: bool = True,
    worst: int = 2,
    n_seeds: int = 16,
    n_alts: int = 2,
    window: int = 2,
) -> List[Candidate]:
    """Rank up to ``budget`` candidates from the run's LATEST eval-cycle losses (the cycle whose opponent
    mapping is resolvable — see ``opponent_resolver``). ``falsify_gate`` keeps only proven MISTAKEs
    (expensive — re-rolls each crater); off → cheaper, every crater is a candidate. ``window`` also
    queues the crater's preceding turns (the mistake is often upstream of the crater)."""
    from main.prober.session import ProbeSession
    from main.prober.falsifier import falsify_battle
    from utils.bridge.reconstruction import ReconstructionRecord

    sess = ProbeSession(run_dir)
    if step is None:
        steps = [sg.step for sg in sess.tree.steps]
        if not steps:
            return []
        step = max(steps)

    rows = sess.scan(outcome="loss", step=step, limit=scan_limit, metric="value_drop")
    seen = set()
    out: List[Candidate] = []
    for row in rows:
        summary_path = row["id"]
        recon = _recon_for(summary_path)
        if not os.path.exists(recon):
            continue                         # falsify/better-line need the reconstruction sibling
        b = sess._battle(summary_path)
        summary, npz = sess._summary(b), sess._npz(b)
        worst_inv = int(row["worst"]["inv"])
        crater_delta = abs(float(row["worst"].get("delta_v") or 0.0))

        # the (inv, delta, verdict) anchors to queue for this battle
        anchors = []
        if falsify_gate:
            try:
                record = ReconstructionRecord.load(recon)
                fb = falsify_battle(record, summary, npz, worst=worst, gamma=sess._gamma,
                                    n_seeds=n_seeds, n_alts=n_alts)
            except (FileNotFoundError, RuntimeError, ValueError) as e:  # noqa: F841 — skip on a wedge
                continue
            for d in fb.get("decisions", []):
                if d.get("verdict") in ("MISTAKE", "MIXED"):           # a reducible mistake (not LUCK)
                    anchors.append((int(d["inv"]), abs(float(d.get("anchor_delta") or crater_delta)),
                                    d["verdict"]))
        else:
            anchors.append((worst_inv, crater_delta, "CRATER"))

        # crater ± window: also queue the move_selection invs of the preceding turns (the cause).
        expanded = list(anchors)
        for inv_i, dlt, verd in anchors:
            inv = summary["invocations"][inv_i] if inv_i < len(summary["invocations"]) else None
            t = int(inv.get("turn", 0)) if inv else 0
            for back in range(1, window + 1):
                wi = _move_inv_at_turn(summary, t - back)
                if wi is not None:
                    expanded.append((wi, dlt * 0.9 ** back, verd))     # decay the window's priority

        for inv_i, dlt, verd in expanded:
            key = (summary_path, inv_i)
            if key in seen:
                continue
            inv = summary["invocations"][inv_i] if inv_i < len(summary["invocations"]) else None
            if not inv or inv.get("phase") != "move_selection":
                continue
            seen.add(key)
            out.append(Candidate(
                summary_path=summary_path, recon_path=recon, inv_index=inv_i,
                turn=int(inv.get("turn", 0)), opponent=row.get("opponent", "?"),
                step=int(row.get("step", step)), anchor_delta=dlt, verdict=verd))

    out.sort(key=lambda c: c.anchor_delta, reverse=True)
    return out[:budget]
