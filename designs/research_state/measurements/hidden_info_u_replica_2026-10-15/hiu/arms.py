"""The world sets of the arms this build adds or re-runs, on top of the replica's builder.

====== ============================================ ===============================================================
arm    sampled                                      true / kept
====== ============================================ ===============================================================
O      nothing (the true battle; the replica's)     everything, incl. the opponent's pending root choice
F      M, U, I, A, S (``replica.fair.build``)       nothing hidden; the pending choice DROPPED (replica A3)
F+M    U, I, A, S (``replica.fair.build``)          M; pending DROPPED
F+U    M, I, A, S = F world j's draws, verbatim     U: every never-seen slot is the TRUE packed entry (realized
                                                    gender) in its TRUE slot; pending DROPPED (as F)
O-M    M (``replica.builder.assemble``)             U, I, A, S; the pending choice KEPT (rendered for the world)
O-U    U: species from the teammate prior, a whole  every seen mon verbatim; the pending choice KEPT
       set from the Smogon prior, placed uniformly
====== ============================================ ===============================================================

The PENDING choice (the opponent's root-turn choice the log recorded before the searched side's, not yet executed):
the replica drops it in every constructed world (its A3). Here the arms built from the TRUE team (O-M, O-U) KEEP
it, rendered for the world, exactly as O keeps it: an "O minus x" arm differs from O only by x, and the registered
identity O-U = O at k = 6 then holds on every state, the pending ones included. The arms built from F (F, F+M, F+U)
drop it, as the replica's F does, so F+U = F at k = 6 holds too. Spec ambiguity, reported (README, U1).

Every arm except O is gated exactly as the replica gates F: the searched side's protocol lines AND observation rows
(mask and legal tokens included) byte-equal to the truth's at every one of its decisions, no gen3ou rule broken
(``sampling.legality``), and the playout tool replays the world (``validate``).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from replica import builder as B
from replica import fair
from replica import sampling as S

from . import guard

Validate = Optional[Callable[[str, list], bool]]


def assemble_ou(prep: B.Prep, rng) -> List[S.MonSet]:
    """O-U's world: every seen mon the TRUE entry (realized gender) in its slot, the never-seen slots re-drawn."""
    world: List[Optional[S.MonSet]] = [None] * 6
    for name in prep.seen.mons:
        i = prep.seen_slot[name]
        t = prep.true_opp[i].copy()
        t.gender = prep.opp_real_gender[i]
        t.origin = {"M": "true", "I": "true", "A": "true", "S": "true"}
        world[i] = t
    B.draw_never_seen(prep, world, rng)              # hiu.fix's uniformly placed draw
    return world                                     # type: ignore[return-value]


def build_true_family(prep: B.Prep, arm: str, validate: Validate) -> dict:
    """Up to 8 accepted worlds for O-M or O-U (``B.MAX_TRIES`` draws), the pending choice kept."""
    t0 = time.process_time()
    rec: dict = {"arm": arm, "id": prep.sid, "pending_kept": prep.pending is not None}
    if prep.error:
        rec.update({"ok": False, "reason": "prep:" + prep.error})
        return rec
    rng = B.rng_for("hiu_v1", "worlds", arm, prep.sid)
    accepted: List[dict] = []
    tries = 0
    reasons: Dict[str, int] = {}
    while len(accepted) < B.N_WORLDS and tries < B.MAX_TRIES:
        need = B.N_WORLDS - len(accepted)
        batch = []
        for _ in range(min(need * 2, B.MAX_TRIES - tries)):
            tries += 1
            w = B.assemble(prep, "O-M", rng) if arm == "O-M" else assemble_ou(prep, rng)
            probs = S.legality(w, prep.true_opp)
            if probs:
                reasons["illegal"] = reasons.get("illegal", 0) + 1
                if len(rec.get("illegal_examples", [])) < 5:
                    rec.setdefault("illegal_examples", []).append(probs[:3])
                continue
            batch.append(w)
        if not batch:
            continue
        for w, g in zip(batch, B.gate_many(prep, batch, rows=True, keep_pending=True)):
            if g.ok and len(accepted) < B.N_WORLDS:
                cmds, changed = B.render(prep, w, keep_pending=True)
                if validate is not None and not validate(S.pack_team(w), cmds):
                    reasons["replay_refused"] = reasons.get("replay_refused", 0) + 1
                    continue
                accepted.append({"team": S.pack_team(w), "rendered": changed, "cmds": cmds,
                                 "origin": [x.origin for x in w]})
            elif not g.ok:
                key = "gate_rows" if g.lines_equal else ("gate_error" if g.error else "gate_lines")
                reasons[key] = reasons.get(key, 0) + 1
    rec.update({"ok": len(accepted) == B.N_WORLDS, "worlds": accepted, "tries": tries, "rejects": reasons,
                "reason": None if len(accepted) == B.N_WORLDS else f"only_{len(accepted)}_worlds",
                "notes": sorted(set(prep.notes)), "cpu_s": round(time.process_time() - t0, 3)})
    return rec


def build_fair(prep: B.Prep, arm: str, validate: Validate) -> dict:
    """F or F+M: the replica's own builder (with ``hiu.fix`` installed)."""
    rec = fair.build(prep, arm, validate=validate)
    rec["pending_kept"] = False
    return rec


def build_fpu(prep: B.Prep, f_worlds: Optional[List[str]], validate: Validate) -> dict:
    """F+U: F's world j with every never-seen slot replaced by the TRUE packed entry at its TRUE slot. Gated as F;
    any world that fails fails the state (no re-draw: F+U is F's worlds or nothing)."""
    t0 = time.process_time()
    rec: dict = {"arm": "F+U", "id": prep.sid, "pending_kept": False}
    if prep.error:
        rec.update({"ok": False, "reason": "prep:" + prep.error})
        return rec
    if not f_worlds or len(f_worlds) != B.N_WORLDS:
        rec.update({"ok": False, "reason": "F_not_ok"})
        return rec
    us = guard.u_slots(prep)
    worlds: List[List[S.MonSet]] = []
    for team in f_worlds:
        sets = S.unpack(team)
        for i in us:
            t = prep.true_opp[i].copy()
            t.gender = prep.opp_real_gender[i]
            t.origin = {"U": "true"}
            sets[i] = t
        worlds.append(sets)
    accepted: List[dict] = []
    for j, w in enumerate(worlds):
        probs = S.legality(w, prep.true_opp)
        if probs:
            rec.update({"ok": False, "reason": f"illegal_world_{j}:{probs[:3]}"})
            return rec
    for j, (w, g) in enumerate(zip(worlds, B.gate_many(prep, worlds, rows=True))):
        if not g.ok:
            key = "rows" if g.lines_equal else ("error" if g.error else f"lines@{g.first_bad_line}")
            rec.update({"ok": False, "reason": f"gate_world_{j}:{key}", "cpu_s": round(time.process_time() - t0, 3)})
            return rec
        cmds, changed = B.render(prep, w)
        if validate is not None and not validate(S.pack_team(w), cmds):
            rec.update({"ok": False, "reason": f"replay_refused_world_{j}"})
            return rec
        accepted.append({"team": S.pack_team(w), "rendered": changed, "cmds": cmds,
                         "origin": [x.origin for x in w]})
    rec.update({"ok": True, "worlds": accepted, "tries": B.N_WORLDS, "reason": None, "n_true_u": len(us),
                "cpu_s": round(time.process_time() - t0, 3)})
    return rec
