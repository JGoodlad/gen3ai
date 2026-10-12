"""Gate-conditioned world sets per state and arm (the 8 accepted worlds a state's playouts run in).

``build(prep, arm)`` draws worlds for ``arm`` (``builder.assemble``; for F / F+M the seen mons' item, ability,
spread and bare-HP type come from ``screen.screen``'s per-mon gate-conditioned pools), rejects a world team that
breaks a gen3ou rule (``sampling.legality``), and accepts a world only if the full gate passes (lines AND rows),
until 8 are accepted or 60 draws are spent (spec §6.1). The accepted worlds are taken in draw order.
"""
from __future__ import annotations

import time
from typing import Dict, List

from . import builder as B
from . import sampling as S


def build(prep: B.Prep, arm: str) -> dict:
    """Up to 8 accepted worlds for ``arm`` (``MAX_TRIES`` draws in all). Returns the worlds and the bookkeeping."""
    t0 = time.process_time()
    rec: dict = {"arm": arm, "id": prep.sid}
    if prep.error:
        rec.update({"ok": False, "reason": "prep:" + prep.error})
        return rec
    scr = None
    if arm in B.SCREENED:
        from . import screen as SC
        scr = SC.screen(prep, arm)
        rec["screen"] = {k: v for k, v in scr.items()
                         if k not in ("pools", "context", "fixed_moves", "u_world", "pool_hp_raw")}
        if scr.get("empty_pool"):
            rec.update({"ok": False, "reason": "empty_pool", "cpu_s": time.process_time() - t0})
            return rec
        if not scr["context_ok"]:
            rec.update({"ok": False, "reason": "no_consistent_context", "cpu_s": time.process_time() - t0})
            return rec
    rng = B.rng_for("worlds", arm, prep.sid)
    accepted: List[dict] = []
    tries = 0
    reasons: Dict[str, int] = {}
    while len(accepted) < B.N_WORLDS and tries < B.MAX_TRIES:
        need = B.N_WORLDS - len(accepted)
        batch = []
        for _ in range(min(need * 2, B.MAX_TRIES - tries)):
            tries += 1
            if arm in B.SCREENED:
                dr = SC.draw_from_pools(scr, rng)
                w = B.assemble(prep, arm, rng, seen_draws=dr)
            else:
                w = B.assemble(prep, arm, rng)
            probs = S.legality(w, prep.true_opp)
            if probs:
                reasons["illegal"] = reasons.get("illegal", 0) + 1
                rec.setdefault("illegal_examples", []).append(probs[:3]) if len(rec.get("illegal_examples", [])) < 5 \
                    else None
                continue
            batch.append(w)
        if not batch:
            continue
        for w, g in zip(batch, B.gate_many(prep, batch, rows=True)):
            if g.ok and len(accepted) < B.N_WORLDS:
                cmds, changed = B.render(prep, w)
                accepted.append({"team": S.pack_team(w), "rendered": changed, "cmds": cmds,
                                 "origin": [x.origin for x in w]})
            elif not g.ok:
                key = "gate_rows" if g.lines_equal else ("gate_error" if g.error else "gate_lines")
                reasons[key] = reasons.get(key, 0) + 1
    rec.update({"ok": len(accepted) == B.N_WORLDS, "worlds": accepted, "tries": tries, "rejects": reasons,
                "reason": None if len(accepted) == B.N_WORLDS else f"only_{len(accepted)}_worlds",
                "pending_dropped": prep.pending is not None, "notes": sorted(set(prep.notes)),
                "cpu_s": round(time.process_time() - t0, 3)})
    return rec
