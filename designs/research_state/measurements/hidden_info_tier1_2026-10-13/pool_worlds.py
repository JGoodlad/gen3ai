"""POOL, HEAD SIDE: worlds drawn from the TRAINING-POOL TEAMS consistent with the observation (`design_hidden_info_binding.md`
§6.4: "a world's opponent team = a pool team drawn uniformly among those whose species include every revealed species
and whose sets contain every revealed move, then the gate (the true team is in the pool, so POOL can be the oracle when
the observations identify the team)"). The Bayes posterior of the eval distribution — a MEASUREMENT only (the pool
never ships as a prior).

THE DRAW (per decision, deterministic in ``key``):

1. ``root = worlds.prepare_root(...)`` — the builder's own truth replay, the ONE log source (``agents.info_levers.
   revealed``), the viewer's rows and protocol, and the choice TEMPLATE;
2. CANDIDATES: the pool teams (``pool_teams.load_pool()``, the trainer's packed pool, duplicates kept: the eval
   opponent's draw) whose species include every revealed species (``root.kept``) and whose set of each holds every
   move play revealed for it (a revealed bare Hidden Power matches any typed one);
3. a uniformly random ORDER of the candidates (``sha256("hib_pool_v1:<key>")``); EVERY candidate (up to
   ``MAX_GATED``, a cost bound) is BUILT and GATED, ``max_tries`` per batched core replay (gating only a 60-candidate
   prefix EXCLUDED a k = 1 state whose single passing team lay outside it — the first validation run): each
   candidate's SEEN mons are its sets placed at the seen mons' TRUE slots
   (its moves aligned to the true slots exactly as the builder aligns a seen mon's drawn moves: the revealed moves at
   their slots, ``world_draws.align_to_truth`` for the rest; its item, ability, nature, EVs, IVs; the name, gender,
   level and misc fields stay the truth's, as the builder keeps them for a seen mon), its OTHER mons go to the
   never-seen slots and ``worlds.place_genders`` (the engine pad) places them, exactly as the builder places its drawn
   never-seen mons; then the builder's checks in the builder's order: legality (``worlds.team_problems`` against the
   true team's), the STRICT gate (``run_core(..., trackers=True, obs=True)``: the searched side's obs rows AND protocol
   lines byte-equal to the truth's at every one of its decisions up to the root — :func:`strict_verdict`, the builder's
   ``gate`` with no exact-HP condition, proved equal to it by ``pool_worlds_test.py`` / the validation run) and the
   playout tool's open check (``WorldSampler.playout_refusal``);
4. world j (j = 0..J−1) = a uniform draw (``sha256("hib_pool_v1:<key>:<j>")``) among the candidates that passed —
   with replacement (the posterior: uniform over consistent pool teams; a random prefix, when ``MAX_GATED`` cuts,
   keeps it uniform).

A decision with no passing candidate is an EXCLUSION: ``PoolExcluded`` with ``reason`` ``no_candidate`` (no pool team
fits the reveals), ``build`` (every gated candidate failed construction), ``gate`` (none passed the gate / playout).

Interface (the harness)::

    from pool_worlds import pool_sample, PoolExcluded
    from pool_teams import load_pool
    pool = load_pool()                                          # once per process
    pw = pool_sample(log, at, side, 8, key=did, pool=pool, max_tries=60, root_choice_in_log=..., ws=<a WorldSampler,
                     optional: its playout core is reused>, screen_cache=cache)   # cache: optional, verdicts shared
    pw.teams, pw.cmds, pw.logs, pw.meta    # meta: candidates, gated, passed, true_in_*, true_drawn, fails, …
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pool_teams as PT  # noqa: E402

from main.value_search import world_draws as WD  # noqa: E402
from main.value_search import worlds as W  # noqa: E402

SALT = "hib_pool_v1:"
#: every candidate is gated (the exact posterior support) up to this many (a cost bound; the largest candidate set of
#: the validation states was far below it — see the report); ``max_tries`` is the core batch size
MAX_GATED = 1000


class PoolExcluded(W.WorldError):
    def __init__(self, msg: str, reason: str, meta: dict):
        super().__init__(msg)
        self.reason = reason
        self.meta = meta


@dataclass
class PoolWorlds:
    teams: List[str]
    cmds: List[List[list]]
    logs: List[dict]
    meta: dict = field(default_factory=dict)


def _rng(tag: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(tag.encode()).hexdigest()[:16], 16))


def holds(f: List[str], moves: Set[str]) -> bool:
    """Does packed set ``f`` hold every move id in ``moves`` (a bare ``hiddenpower`` = any typed one)?"""
    tm = W.typed_moves(f)
    for m in moves:
        if m in tm:
            continue
        if m == "hiddenpower" and any(W.is_hp(x) for x in tm):
            continue
        return False
    return True


def candidates(pool: PT.Pool, kept: Dict[str, Set[str]]) -> List[int]:
    """Pool indices whose species include every revealed species and whose sets hold every revealed move."""
    out = []
    for c, t in enumerate(pool.teams):
        fs = {W.toid(f[1] or f[0]): f for f in (W.parse_set(e) for e in t.split("]"))}
        if all(s in fs and holds(fs[s], set(m)) for s, m in kept.items()):
            out.append(c)
    return out


def build_world(root: "W.Root", team: str, rng: np.random.Generator, pad: Optional[str]) -> Tuple[Optional[str], str]:
    """The world team of pool team ``team`` at this root (module doc step 3), or (None, why)."""
    psets = [W.parse_set(e) for e in team.split("]")]
    pby = {W.toid(f[1] or f[0]): f for f in psets}
    if len(psets) != len(root.entries):
        return None, "team_size"
    out = list(root.entries)
    seen_slots = [i for i, s in enumerate(root.sids) if s in root.kept]
    never = [i for i, s in enumerate(root.sids) if s not in root.kept]
    for i in seen_slots:
        sid = root.sids[i]
        g = pby[sid]
        gm = W.typed_moves(g)
        tm = root.moves_of[i]
        kept = root.kept[sid]
        keep = {s: m for s, m in enumerate(tm) if m in kept or (W.is_hp(m) and "hiddenpower" in kept)}
        if any(not (m in tm or (m == "hiddenpower" and any(W.is_hp(x) for x in tm))) for m in kept):
            return None, "kept_not_in_truth"
        g_hp = [x for x in gm if W.is_hp(x)]
        slots: Dict[int, str] = {}
        for s, m in keep.items():
            if W.is_hp(m) and m not in kept:      # a revealed BARE Hidden Power: the candidate's own typed one
                slots[s] = g_hp[0]
            else:
                slots[s] = m
        used = set(slots.values())
        rest = [m for m in gm if m not in used]
        free = [s for s in range(len(tm)) if s not in keep]
        if len(rest) != len(free):
            return None, "move_count"
        for s, m in zip(free, rest):
            slots[s] = m
        mv = WD.align_to_truth([slots[s] for s in range(len(tm))], tm, set(keep))
        f = list(root.sets[i])
        f[2], f[3], f[4], f[5], f[6], f[8] = g[2], g[3], ",".join(mv), g[5], g[6], g[8]
        out[i] = "|".join(f)
    others = [f for f in psets if W.toid(f[1] or f[0]) not in root.kept]
    if len(others) != len(never):
        return None, "never_count"
    for i, f in zip(never, others):
        out[i] = "|".join(f)
    draw_slots = W.gender_draw_slots(root.team) & set(never)
    placed = W.place_genders(rng, out, never, draw_slots, set(), pad)
    if placed is None:
        return None, "gender_count"
    tw = "]".join(placed)
    if W.team_problems(tw) - W.team_problems(root.team):
        return None, "clause"
    return tw, ""


def strict_verdict(root: "W.Root", res: dict) -> bool:
    """The builder's STRICT gate (``WorldSampler.sample``'s ``gate`` with no exact-HP condition): the replay ran, and
    the searched side's obs rows at every one of its decisions up to the root AND every protocol line it was sent are
    byte-equal to the truth's."""
    if not res["ok"]:
        return False
    try:
        _, rows = W.reveals(res, root.viewer, root.n)
    except W.WorldError:
        return False
    return rows == root.ref and W.protocol_lines(res, root.viewer, root.n) == root.ref_proto


def strict_gate(root: "W.Root", teams: List[str], key: str, verdicts: Optional[dict] = None) -> List[bool]:
    """Strict verdicts for opponent teams (one batched ``run_core``), cached in ``verdicts`` under the builder's own
    cache key (``text + "#" + "" + "#strict"``) — the same dict the builder fills, so a team either judged is judged
    once."""
    from agents.battle.core_replay import run_core

    verdicts = verdicts if verdicts is not None else {}
    todo = [t for t in dict.fromkeys(teams) if (t + "##strict") not in verdicts]
    if todo:
        out = run_core([root.rec(t, f"{key}#pool{i}", W.render(root.tmpl, t)) for i, t in enumerate(todo)],
                       trackers=True, obs=True)
        for t, r in zip(todo, out):
            verdicts[t + "##strict"] = (strict_verdict(root, r), -1.0, None)
    return [bool(verdicts[t + "##strict"][0]) for t in teams]


def world_log(root: "W.Root", tw: str) -> dict:
    """The world's full core input log, exactly as the builder writes it."""
    log, oi, at = root.log, root.oi, root.at
    pre = [f"FORCELOSE {tok}" if s_ == "forcelose" else f"CHOOSE {s_} {tok}" for s_, tok in W.render(root.tmpl, tw)]
    return dict(log, teams=[tw if s_ == oi else t_ for s_, t_ in enumerate(log["teams"])],
                cmds=pre + list(log["cmds"][at:]))


def pool_sample(log: dict, at: int, side: str, J: int, key: str, pool: Optional[PT.Pool] = None,
                max_tries: int = 60, root_choice_in_log: bool = False, ws: Optional[W.WorldSampler] = None,
                screen_cache: Optional[dict] = None, max_gated: Optional[int] = None) -> PoolWorlds:
    max_gated = MAX_GATED if max_gated is None else max_gated
    pool = pool or PT.load_pool()
    ws = ws or W.WorldSampler(reveal="off")
    root = W.prepare_root(log, at, side, key, root_choice_in_log)
    verdicts = screen_cache.setdefault(("verdict", key), {}) if screen_cache is not None else {}
    kept = {s: set(m) for s, m in root.kept.items()}
    cands = candidates(pool, kept)
    true_idx = set(PT.true_indices(pool, root.team))
    tkey = PT.team_key(root.team)
    meta: dict = {"candidates": len(cands), "distinct_candidates": len({pool.teams[c] for c in cands}),
                  "k": len(kept), "true_in_pool": bool(true_idx), "true_in_candidates": bool(true_idx & set(cands)),
                  "pool_sha256": pool.sha256, "fails": {}}
    if not cands:
        raise PoolExcluded(f"{key}: no pool team holds the revealed species and moves", "no_candidate", meta)
    order = [cands[i] for i in _rng(SALT + key).permutation(len(cands))][:max_gated]
    built: List[Tuple[int, str]] = []
    for c in order:
        tw, why = build_world(root, pool.teams[c], _rng(f"{SALT}{key}:cand:{c}"), ws.pad)
        if tw is None:
            meta["fails"][why] = meta["fails"].get(why, 0) + 1
        else:
            built.append((c, tw))
    meta["gated"] = len(order)
    if not built:
        raise PoolExcluded(f"{key}: no gated candidate could be built ({meta['fails']})", "build", meta)
    oks: List[bool] = []
    step = max(1, int(max_tries))                 # candidates replayed per core batch
    for s0 in range(0, len(built), step):
        oks += strict_gate(root, [tw for _, tw in built[s0:s0 + step]], key, verdicts)
    passed: List[Tuple[int, str]] = []
    refusals: List[str] = []
    for (c, tw), ok in zip(built, oks):
        if not ok:
            meta["fails"]["gate"] = meta["fails"].get("gate", 0) + 1
            continue
        why = ws.playout_refusal(world_log(root, tw), at, side)
        if why is not None:
            meta["fails"]["playout_refused"] = meta["fails"].get("playout_refused", 0) + 1
            refusals.append(why)
            continue
        passed.append((c, tw))
    meta["passed"] = len(passed)
    meta["passed_distinct_teams"] = len({PT.team_key(tw) for _, tw in passed})
    meta["true_in_passed"] = any(PT.team_key(tw) == tkey for _, tw in passed)
    meta["true_gated"] = any(c in true_idx for c in order)
    if refusals:
        meta["playout_refusal_example"] = refusals[0][:300]
    if not passed:
        raise PoolExcluded(f"{key}: none of {len(built)} built candidates passed the gate ({meta['fails']})",
                           "gate", meta)
    pick = [passed[int(_rng(f"{SALT}{key}:{j}").integers(len(passed)))] for j in range(J)]
    teams = [tw for _, tw in pick]
    meta["drawn_pool_index"] = [c for c, _ in pick]
    meta["true_drawn"] = [PT.team_key(tw) == tkey for tw in teams]
    meta["revealed"] = {s: sorted(m) for s, m in kept.items()}
    return PoolWorlds(teams=teams, cmds=[W.render(root.tmpl, t) for t in teams],
                      logs=[world_log(root, t) for t in teams], meta=meta)


class PoolSampler:
    """The harness's interface: ``PoolSampler(max_tries=60).sample(log, at, side, J, key=did)`` → a
    ``worlds.Worlds`` (``logs``, ``teams``, ``tries``, ``fails``, ``leak``, ``revealed``, ``spec``, ``facts``,
    ``screens`` = the candidates replayed, ``stats``), with the draw's record in ``.extra["pool"]`` (= ``.pool_meta``).
    Raises :class:`PoolExcluded` (a ``WorldError``; ``.reason`` / ``.meta``) for an exclusion."""

    def __init__(self, max_tries: int = 60, pool: Optional[PT.Pool] = None, ws: Optional[W.WorldSampler] = None):
        self.max_tries = max_tries
        self.pool = pool or PT.load_pool()
        self.ws = ws or W.WorldSampler(reveal="off")

    def sample(self, log: dict, at: int, side: str, J: int, key: str, root_choice_in_log: bool = False,
               screen_cache: Optional[dict] = None) -> "W.Worlds":
        pw = pool_sample(log, at, side, J, key, self.pool, self.max_tries, root_choice_in_log, self.ws, screen_cache)
        m = pw.meta
        kept = m["revealed"]
        facts = []
        for t in pw.teams:
            rows = []
            for i, e in enumerate(t.split("]")):
                f = W.parse_set(e)
                sid = W.toid(f[1] or f[0])
                rows.append({"slot": i, "species": sid, "seen": sid in kept, "source": "pool_team"})
            facts.append(rows)
        w = W.Worlds(logs=pw.logs, teams=pw.teams, tries=[int(m["gated"])] * J, leak={"pool": True},
                     revealed=kept, stats=[{"pool_index": c} for c in m["drawn_pool_index"]],
                     fails=dict(m["fails"]), facts=facts,
                     spec={"arm": "POOL", "reveal": "off", "salt": SALT, "max_tries": self.max_tries},
                     screens=int(m["gated"]))
        w.pool_meta = m            # type: ignore[attr-defined]
        w.extra = {"pool": m}      # type: ignore[attr-defined]
        return w
