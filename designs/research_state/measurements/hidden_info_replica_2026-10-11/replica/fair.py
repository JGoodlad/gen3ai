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


def screen_hp_om(prep: B.Prep) -> Dict[str, Dict[str, float]]:
    """O-M: the type of a revealed bare Hidden Power is class M (sampled); screen each type in the prior's support
    with every other fact TRUE (spec §6.1's per-candidate screen), and draw from the prior restricted to the
    consistent types (the full prior when none is consistent: the gate then rejects)."""
    out: Dict[str, Dict[str, float]] = {}
    for name, m in prep.seen.mons.items():
        if "hiddenpower" not in m.moves:
            continue
        slot = prep.seen_slot[name]
        prior = S.hp_type_prior(m.species)
        worlds = []
        types = sorted(prior)
        for t in types:
            w = B.assemble(prep, "ID", B.rng_for("om_hp", prep.sid))
            x = w[slot]
            x.moves = [("hiddenpower" + t) if S.is_hp(mv) else mv for mv in x.moves]
            x.ivs = S.ivs_for(t, "min" if prep.true_opp[slot].ivs[1] <= 3 else "max")
            worlds.append(w)
        res = B.gate_many(prep, worlds, rows=False)
        ok = {t: prior[t] for t, r in zip(types, res) if r.lines_equal}
        out[name] = ok or prior
    return out


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
        rec["screen"] = {k: v for k, v in scr.items() if k not in ("pools", "context", "fixed_moves", "u_world")}
        if scr.get("empty_pool"):
            rec.update({"ok": False, "reason": "empty_pool", "cpu_s": time.process_time() - t0})
            return rec
        if not scr["context_ok"]:
            rec.update({"ok": False, "reason": "no_consistent_context", "cpu_s": time.process_time() - t0})
            return rec
    hp_allowed = screen_hp_om(prep) if arm == "O-M" else None
    if hp_allowed is not None:
        rec["hp_allowed"] = hp_allowed
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
                w = B.assemble(prep, arm, rng, hp_allowed=hp_allowed)
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
