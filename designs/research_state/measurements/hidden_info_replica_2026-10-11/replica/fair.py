"""Gate-conditioned world sets per state and arm (the 8 accepted worlds a state's playouts run in).

For F and F+M the seen mons' I / A / S are SCREENED (spec §6.1: "each candidate in the prior's support is screened
by a replay with the other mons at their F draws, then the world's combination passes the full gate (≤ 60 tries)").
The Smogon spread support runs to thousands of spreads per species (Tyranitar 11,230), so the screen is a
MONTE-CARLO screen of the support: joint prior draws of a mon's unrevealed (item, ability, nature + EVs, attack-IV
variant, bare-HP type), each replayed with the other mons at a CONSISTENT context, the consistent ones kept. Kept
draws are iid samples of the prior restricted to the consistent set (rejection sampling), which is what a world
draws from. How the context is found: a coordinate search from each class's prior MODE, mons in order of first
appearance, each step keeping the candidate whose replay matches the searched side's stream longest (ties to the
current context, then to the higher prior), until the whole stream and every row match (≤ 4 sweeps).
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import builder as B
from . import sampling as S

BATCH = 64
FINAL_BATCHES = 8          # ≤ 512 prior draws per mon in the final screen
POOL_TARGET = 24           # stop drawing once this many consistent draws are pooled
SWEEPS = 4


def _prior_logp(prep: B.Prep, name: str, d: B.SeenDraw, arm: str) -> float:
    m = prep.seen.mons[name]
    sp = m.species
    lp = 0.0
    if m.item is None:
        ip = S.item_prior(sp)
        lp += np.log(ip.get(d.item, 1e-300) / sum(ip.values()))
    if m.ability is None:
        ap = S.ability_prior(sp)
        lp += np.log(ap.get(d.ability, 1e-300) / sum(ap.values()))
    w = {(n, e): x for n, e, x in S.spread_prior(sp)}
    lp += np.log(w.get((d.nature, tuple(d.evs)), 1e-300))
    if d.hp_t is not None and arm == "F":
        hp = S.hp_type_prior(sp)
        lp += np.log(hp.get(d.hp_t, 1e-300) / sum(hp.values()))
    return float(lp)


def screen(prep: B.Prep, arm: str) -> dict:
    """The context and the per-mon consistent pools for a screened arm."""
    rng = B.rng_for("screen", arm, prep.sid)
    names = [m.name for m in prep.seen.order()]
    fixed_moves = {n: B.seen_moves(prep, prep.seen.mons[n], prep.seen_slot[n], arm, rng,
                                   B.seen_mode(prep, prep.seen.mons[n], arm).hp_t) for n in names}
    u_world: List[Optional[S.MonSet]] = [None] * 6
    for n in names:
        u_world[prep.seen_slot[n]] = prep.true_opp[prep.seen_slot[n]].copy()     # placeholder, replaced below
    B.draw_never_seen(prep, u_world, rng)
    for n in names:
        u_world[prep.seen_slot[n]] = None
    ctx = {n: B.seen_mode(prep, prep.seen.mons[n], arm) for n in names}
    replays = 0

    def worlds_for(cands: Dict[str, List[B.SeenDraw]], who: str) -> List[List[S.MonSet]]:
        out = []
        for c in cands[who]:
            dr = dict(ctx)
            dr[who] = c
            out.append(B.assemble(prep, arm, rng, seen_draws=_with_hp(dr, fixed_moves),
                                  fixed_moves=_moves_for(prep, dr, fixed_moves, arm), fixed_u=u_world))
        return out

    def full_ok() -> Tuple[bool, B.GateResult]:
        nonlocal replays
        w = B.assemble(prep, arm, rng, seen_draws=ctx, fixed_moves=_moves_for(prep, ctx, fixed_moves, arm),
                       fixed_u=u_world)
        replays += 1
        g = B.gate_many(prep, [w], rows=True)[0]
        return g.ok, g

    ok, g = full_ok()
    sweeps = 0
    while not ok and sweeps < SWEEPS:
        sweeps += 1
        for n in names:
            cands = [ctx[n]] + [B.seen_prior_draw(prep, prep.seen.mons[n], arm, rng) for _ in range(BATCH - 1)]
            ws = worlds_for({n: cands}, n)
            res = B.gate_many(prep, ws, rows=False)
            replays += len(ws)
            score = [r.first_bad_line for r in res]
            best = max(score)
            winners = [i for i, s in enumerate(score) if s == best]
            if 0 not in winners:
                winners.sort(key=lambda i: -_prior_logp(prep, n, cands[i], arm))
                ctx[n] = cands[winners[0]]
        ok, g = full_ok()
    out = {"context_ok": ok, "sweeps": sweeps, "first_bad_line": g.first_bad_line,
           "rows_equal": g.rows_equal, "n_lines": len(prep.truth_lines), "fixed_moves": fixed_moves,
           "u_world": u_world}
    if not ok:
        out["replays"] = replays
        return out
    pools: Dict[str, List[B.SeenDraw]] = {}
    tried: Dict[str, int] = {}
    for n in names:
        pool: List[B.SeenDraw] = []
        k = 0
        while k < FINAL_BATCHES and len(pool) < POOL_TARGET:
            k += 1
            cands = [B.seen_prior_draw(prep, prep.seen.mons[n], arm, rng) for _ in range(BATCH)]
            res = B.gate_many(prep, worlds_for({n: cands}, n), rows=False)
            replays += len(cands)
            pool += [c for c, r in zip(cands, res) if r.lines_equal]
        tried[n] = k * BATCH
        pools[n] = pool if pool else [ctx[n]]
        if not pool:
            out.setdefault("context_only", []).append(n)
    out.update({"pools": pools, "pool_sizes": {n: len(p) for n, p in pools.items()}, "tried": tried,
                "replays": replays, "context": ctx})
    return out


def _with_hp(dr: Dict[str, B.SeenDraw], fixed_moves) -> Dict[str, B.SeenDraw]:
    return dr


def _moves_for(prep: B.Prep, dr: Dict[str, B.SeenDraw], fixed_moves: Dict[str, List[str]], arm: str
               ) -> Dict[str, List[str]]:
    """The screening context's moves with each mon's bare-HP type set from its draw."""
    out = {}
    for n, mv in fixed_moves.items():
        hp_t = dr[n].hp_t
        out[n] = [("hiddenpower" + hp_t) if (S.is_hp(x) and hp_t and "hiddenpower" in prep.seen.mons[n].moves
                                              and arm == "F") else x for x in mv]
    return out


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
