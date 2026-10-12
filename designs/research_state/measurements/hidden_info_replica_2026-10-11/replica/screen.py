"""The F / F+M screen of a seen mon's unrevealed item, ability, bare-Hidden-Power type and spread (spec §6.1).

Why not a plain per-candidate replay: a species' Smogon spread support runs to thousands of spreads (Tyranitar
11,230), and consistent ones are rare (our side's EXACT HP after a hit pins the attacker's Attack to a few
values). So the spread is screened through its REALIZED STATS, which is what the battle reads:

1. **Context search** (find ONE world consistent with the searched side's stream). Start every seen mon at its
   classes' prior modes. Repeat: replay; find the first line where the world's stream departs from the truth's;
   the opponent mon on the field at that line is IMPLICATED; try single-coordinate changes of that mon (each item
   / ability / bare-HP type in the prior support; each of its six stats moved to values the support realizes, a
   coarse grid plus the neighbours of its current value) and keep the change whose replay matches the stream
   longest (ties to the higher prior). A stat value is realized in the replay by a SYNTHETIC spread (any nature,
   EVs 0-252, IVs that keep the Hidden Power bits) with every other stat unchanged. ≤ 24 rounds.
2. **Windows** (once the context matches every line and row): for each seen mon and stat, the support's realized
   values whose replay still matches (scanned outward from the context value until two consecutive failures; a
   stat whose support extremes both match is IRRELEVANT and all its values pass); for each discrete class, every
   candidate in the support that matches.
3. **Posterior pool**: every support tuple (spread, attack-IV variant, item, ability, HP type) whose realized
   stats lie in the windows and whose discrete facts passed, weighted by the product of the Smogon marginals.
   Worlds draw from this pool; the full gate (lines + rows) then rejects the combinations that only fail jointly.
"""
from __future__ import annotations

import functools
import itertools
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.battle.core_replay import run_core
from agents.gen3_data import natures as natures_dex
from agents.gen3_data import species as species_dex

from . import builder as B
from . import sampling as S

STATS = S.STATS
ROUNDS = 40
VERBOSE = bool(__import__("os").environ.get("HIR_VERBOSE"))
GRID = 12


# ------------------------------------------------------------------------------------------ stats

def stat_value(k: str, base: int, ev: int, iv: int, mult: float) -> int:
    if k == "hp":
        return 2 * base + iv + ev // 4 + 110
    x = 2 * base + iv + ev // 4 + 5
    if mult > 1.0:
        return x * 11 // 10
    if mult < 1.0:
        return x * 9 // 10
    return x


def mults(nature: str) -> Dict[str, float]:
    nd = natures_dex.get(nature.lower())
    return {k: (nd.multipliers.get(k, 1.0) if nd is not None and k != "hp" else 1.0) for k in STATS}


def realized(species: str, nature: str, evs: Sequence[int], ivs: Sequence[int]) -> Tuple[int, ...]:
    return _realized(species, nature, tuple(evs), tuple(ivs))


@functools.lru_cache(maxsize=1_000_000)
def _realized(species: str, nature: str, evs: Tuple[int, ...], ivs: Tuple[int, ...]) -> Tuple[int, ...]:
    base = species_dex.get(species).base_stats
    m = mults(nature)
    return tuple(stat_value(k, base[k], evs[i], ivs[i], m[k]) for i, k in enumerate(STATS))


_NATURES = sorted(natures_dex.multipliers())


@functools.lru_cache(maxsize=None)
def _solve(k: str, base: int, target: int, mult: float, bits: int) -> Optional[Tuple[int, int]]:
    """(EV, IV) with stat_value(k, base, EV, IV, mult) == target and IV = bits (mod 4), the largest IV first."""
    for x in range(int(target / max(mult, 0.5)) - 3, int(target / max(mult, 0.5)) + 4):
        if k == "hp":
            if x != target:
                continue
            s = target - 2 * base - 110
        else:
            if stat_value(k, base, 0, 0, mult) is None:
                continue
            fx = x * 11 // 10 if mult > 1.0 else (x * 9 // 10 if mult < 1.0 else x)
            if fx != target:
                continue
            s = x - 2 * base - 5
        for iv in range(31, -1, -1):
            if iv % 4 != bits % 4 or iv > s:
                continue
            ev4 = s - iv
            if 0 <= ev4 <= 63:
                return ev4 * 4, iv
    return None


def synthesize(species: str, target: Sequence[int], iv_bits: Sequence[int], prefer: str
               ) -> Optional[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]:
    """A (nature, EVs, IVs) realizing ``target`` exactly, IVs keeping ``iv_bits`` (each IV's two low bits, which
    fix the Hidden Power type and power). None when no nature reaches it."""
    base = species_dex.get(species).base_stats
    order = [prefer.lower()] + [n for n in _NATURES if n != prefer.lower()]
    for nat in order:
        m = mults(nat)
        evs, ivs = [], []
        for i, k in enumerate(STATS):
            got = _solve(k, base[k], int(target[i]), m[k], int(iv_bits[i]) & 3)
            if got is None:
                break
            evs.append(got[0])
            ivs.append(got[1])
        else:
            return nat.capitalize(), tuple(evs), tuple(ivs)
    return None


def synthesize_relaxed(species: str, target: Sequence[int], k: int, iv_bits: Sequence[int]
                       ) -> Optional[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]:
    """When no nature reaches ``target`` exactly (a nature raises one stat and lowers one, so moving stat ``k``
    past its neutral range forces another to move): stat ``k`` exact, every other stat at its nearest reachable
    value, the nature with the smallest total move of the others."""
    base = species_dex.get(species).base_stats
    best = None
    for nat in _NATURES:
        m = mults(nat)
        got_k = _solve(STATS[k], base[STATS[k]], int(target[k]), m[STATS[k]], int(iv_bits[k]) & 3)
        if got_k is None:
            continue
        evs, ivs, cost = [], [], 0
        for i, s in enumerate(STATS):
            if i == k:
                evs.append(got_k[0])
                ivs.append(got_k[1])
                continue
            hit = None
            for d in range(0, 400):
                for v in ((target[i] - d, target[i] + d) if d else (target[i],)):
                    g = _solve(s, base[s], int(v), m[s], int(iv_bits[i]) & 3)
                    if g is not None:
                        hit = (g, d)
                        break
                if hit:
                    break
            if hit is None:
                break
            evs.append(hit[0][0])
            ivs.append(hit[0][1])
            cost += hit[1]
        else:
            if best is None or cost < best[0]:
                best = (cost, nat.capitalize(), tuple(evs), tuple(ivs))
    return None if best is None else best[1:]


# ------------------------------------------------------------------------------------------ the mon state

@dataclass
class MonState:
    item: str
    ability: str
    hp_t: Optional[str]
    nature: str
    evs: Tuple[int, ...]
    ivs: Tuple[int, ...]

    def draw(self) -> B.SeenDraw:
        return B.SeenDraw(self.item, self.ability, self.nature, self.evs, "max", self.hp_t, ivs=self.ivs)


def support(prep: B.Prep, name: str, arm: str) -> dict:
    """The prior support of every screened class of one seen mon."""
    m = prep.seen.mons[name]
    sp = m.species
    out: dict = {}
    out["item"] = {m.item: 1.0} if m.item is not None else S.item_prior(sp)
    ap = S.ability_prior(sp)
    out["ability"] = {m.ability: 1.0} if m.ability is not None else (ap or {"": 1.0})
    if "hiddenpower" in m.moves:
        if arm == "F":
            out["hp_t"] = S.hp_type_prior(sp)
        else:
            slot = prep.seen_slot[name]
            t = next(S.hp_type(x) for x in prep.true_opp[slot].moves if S.is_hp(x))
            out["hp_t"] = {t: 1.0}
    else:
        out["hp_t"] = {None: 1.0}
    out["spreads"] = S.spread_prior(sp)
    return out


def _norm(p: Dict) -> Dict:
    t = sum(p.values())
    return {k: v / t for k, v in p.items()} if t > 0 else p


# ------------------------------------------------------------------------------------------ scoring

def _world(prep: B.Prep, arm: str, states: Dict[str, MonState], fixed_moves, fixed_u) -> List[S.MonSet]:
    draws = {n: s.draw() for n, s in states.items()}
    mv = {}
    for n, moves in fixed_moves.items():
        t = states[n].hp_t
        mv[n] = [("hiddenpower" + t) if (S.is_hp(x) and t) else x for x in moves]
    return B.assemble(prep, arm, B.rng_for("unused"), seen_draws=draws, fixed_moves=mv, fixed_u=fixed_u)


def score_many(prep: B.Prep, worlds: Sequence[List[S.MonSet]], upto: Optional[int] = None) -> List[Tuple[int, int]]:
    """How far each world's searched-side stream matches the truth's: the first mismatching line index
    (len + 1 = every line matches). A world the core refuses part-way (a desync) is scored by the longest
    TRUNCATED command prefix that replays (3/4, 1/2, 1/4 of the commands)."""
    recs = []
    for w in worlds:
        cmds, _ = B.render(prep, w)
        if upto is not None:
            cmds = cmds[:upto]
        recs.append((B._recorded(prep.battle, prep.opp, S.pack_team(w), cmds, prep.own_team, prep.world_seed), cmds))
    tl = prep.truth_lines
    res = _run([r for r, _ in recs])
    out = []
    retry = []
    for i, r in enumerate(res):
        if r.get("ok"):
            out.append(_first_bad(B.lines_of(r, prep.v), tl, full=upto is None))
        else:
            out.append((-1, -BIG))
            retry.append(i)
    if retry:
        lo = {i: 0 for i in retry}
        hi = {i: len(recs[i][1]) for i in retry}
        best: Dict[int, dict] = {}
        while True:
            act = [i for i in retry if hi[i] - lo[i] > 1]
            if not act:
                break
            sub = []
            for i in act:
                rec, cmds = recs[i]
                mid = (lo[i] + hi[i]) // 2
                sub.append(replace(rec, commands=[list(c) for c in cmds[:mid]]))
            for i, r in zip(act, _run(sub)):
                mid = (lo[i] + hi[i]) // 2
                if r.get("ok"):
                    lo[i] = mid
                    best[i] = r
                else:
                    hi[i] = mid
        for i in retry:
            if i in best:
                out[i] = _first_bad(B.lines_of(best[i], prep.v), tl, full=False)
    return out


BIG = 10 ** 6


def _first_bad(wl: List[str], tl: List[str], full: bool) -> Tuple[int, int]:
    """(index of the first mismatching line, minus a DISTANCE at that line): two ``-damage`` / ``-heal`` lines on
    the same target that differ only in the HP number are |HP_world - HP_truth| apart, so a search can walk a stat
    toward the value that reproduces the observed hit; any other mismatch is BIG apart."""
    n = min(len(wl), len(tl))
    first = next((i for i in range(n) if wl[i] != tl[i]), n)
    if first == n:
        if full and len(wl) == len(tl):
            return (len(tl) + 1, 0)
        return (first, -BIG)
    return (first, -_dist(wl[first], tl[first]))


def _hp(tok: str) -> Optional[Tuple[int, int]]:
    try:
        a, b = tok.split(" ")[0].split("/")
        return int(a), int(b)
    except ValueError:
        return None


def _dist(a: str, b: str) -> int:
    pa, pb = a.split("|"), b.split("|")
    if len(pa) >= 4 and len(pb) >= 4 and pa[1] == pb[1] and pa[1] in ("-damage", "-heal") and pa[2] == pb[2]:
        ha, hb = _hp(pa[3]), _hp(pb[3])
        if ha and hb and ha[1] == hb[1]:
            return abs(ha[0] - hb[0]) + (0 if pa[4:] == pb[4:] else 1000)
    return BIG


def _run(recs) -> List[dict]:
    out: List[dict] = []
    for c0 in range(0, len(recs), 128):
        part = recs[c0:c0 + 128]
        try:
            out += run_core(part, trackers=False, obs=False)
        except RuntimeError:
            for r in part:
                try:
                    out += run_core([r], trackers=False, obs=False)
                except RuntimeError as e:
                    out.append({"ok": False, "error": str(e)[:200]})
    return out


def active_at(prep: B.Prep, line: int) -> Optional[str]:
    """The opponent mon on the field at truth line ``line`` (the last switch / drag before it)."""
    tag = f"|switch|{prep.opp}a: "
    tag2 = f"|drag|{prep.opp}a: "
    who = None
    for ln in prep.truth_lines[:max(line, 0) + 1]:
        if ln.startswith(tag) or ln.startswith(tag2):
            who = ln.split("|")[2].split(": ", 1)[1]
    return who


# ------------------------------------------------------------------------------------------ the screen

NEAR = 300           # support spreads proposed per round: the nearest in the stats the mismatch does NOT move
RANDOM = 32          # plus this many prior draws
POOL_BATCH = 32
POOL_BATCHES = 8
PERTURB = 8          # a stat moved by ± this many points that leaves the stream unchanged is IRRELEVANT


@dataclass
class Ctx:
    item: str
    ability: str
    hp_t: Optional[str]
    nature: str
    evs: Tuple[int, ...]
    variant: str
    ivs: Optional[Tuple[int, ...]] = None      # a synthetic perturbation's IVs (relevance tests only)

    def draw(self) -> B.SeenDraw:
        return B.SeenDraw(self.item, self.ability, self.nature, self.evs, self.variant, self.hp_t, ivs=self.ivs)

    def iv(self) -> Tuple[int, ...]:
        return self.ivs if self.ivs is not None else tuple(S.ivs_for(self.hp_t, self.variant))


def screen(prep: B.Prep, arm: str) -> dict:
    """The context search, the relevance tests and the per-mon posterior pools (module docstring)."""
    t0 = time.process_time()
    rng = B.rng_for("screen", arm, prep.sid)
    names = [m.name for m in prep.seen.order()]
    sup = {n: support(prep, n, arm) for n in names}
    fixed_moves = {}
    for n in names:
        hp_mode = max(sup[n]["hp_t"], key=lambda t: sup[n]["hp_t"][t])
        fixed_moves[n] = B.seen_moves(prep, prep.seen.mons[n], prep.seen_slot[n], arm, rng, hp_mode)
    u_world: List[Optional[S.MonSet]] = [None] * 6
    for n in names:
        u_world[prep.seen_slot[n]] = prep.true_opp[prep.seen_slot[n]].copy()
    B.draw_never_seen(prep, u_world, rng)
    for n in names:
        u_world[prep.seen_slot[n]] = None
    spreads = {n: sup[n]["spreads"] for n in names}
    sw = {n: np.array([r[2] for r in spreads[n]], dtype=float) for n in names}

    def stats_arr(n: str, hp_t: Optional[str], variant: str) -> np.ndarray:
        sp = prep.seen.mons[n].species
        iv = tuple(S.ivs_for(hp_t, variant))
        return np.array([realized(sp, nat, evs, iv) for nat, evs, _ in spreads[n]], dtype=np.int64)

    arr_cache: Dict[tuple, np.ndarray] = {}

    def arr(n: str, hp_t, variant) -> np.ndarray:
        key = (n, hp_t, variant)
        if key not in arr_cache:
            arr_cache[key] = stats_arr(n, hp_t, variant)
        return arr_cache[key]

    def mode(n: str) -> Ctx:
        s = sup[n]
        best = max(s["spreads"], key=lambda r: r[2])
        return Ctx(max(s["item"], key=lambda k: s["item"][k]), max(s["ability"], key=lambda k: s["ability"][k]),
                   max(s["hp_t"], key=lambda k: s["hp_t"][k]), best[0], tuple(best[1]), "max")

    ctx = {n: mode(n) for n in names}

    def world_of(states: Dict[str, Ctx]) -> List[S.MonSet]:
        draws = {n: c.draw() for n, c in states.items()}
        mv = {}
        for n, moves in fixed_moves.items():
            t = states[n].hp_t
            mv[n] = [("hiddenpower" + t) if (S.is_hp(x) and t) else x for x in moves]
        return B.assemble(prep, arm, B.rng_for("unused"), seen_draws=draws, fixed_moves=mv, fixed_u=u_world)

    replays = 0
    # how many of the RENDERED commands produce the truth's first L lines (to replay only what a mismatch needs)
    w0 = world_of(ctx)
    cmds0, _ = B.render(prep, w0)
    from agents.battle.core_replay import RecordedBattle as _RB  # noqa: F401
    pref = [B._recorded(prep.battle, prep.opp, S.pack_team(B.assemble(prep, "ID", B.rng_for("x"))), cmds0[:k],
                        prep.own_team, prep.world_seed) for k in range(len(cmds0) + 1)]
    pres = _run(pref)
    n_lines_at = [len(B.lines_of(r, prep.v)) if r.get("ok") else -1 for r in pres]

    def upto_for(line: int) -> Optional[int]:
        for k, nl in enumerate(n_lines_at):
            if nl > line + 1:
                return min(len(cmds0), k + 2)
        return None

    def score(cands: List[Dict[str, Ctx]], upto: Optional[int] = None) -> List[Tuple[int, int]]:
        nonlocal replays
        replays += len(cands)
        return score_many(prep, [world_of(c) for c in cands], upto)

    moving: set = set()

    def sensitivity(n: str, upto: Optional[int], cur_score) -> set:
        """The stats of mon n whose ±PERTURB move changes the score at the current mismatch."""
        c = ctx[n]
        sp = prep.seen.mons[n].species
        base = list(realized(sp, c.nature, c.evs, c.iv()))
        tests, which = [], []
        for k in range(6):
            for dlt in (-PERTURB, PERTURB):
                t = list(base)
                t[k] += dlt
                syn = synthesize(sp, t, c.iv(), c.nature) or synthesize_relaxed(sp, t, k, c.iv())
                if syn is None:
                    continue
                st = dict(ctx)
                st[n] = replace(c, nature=syn[0], evs=syn[1], ivs=syn[2])
                tests.append(st)
                which.append(k)
        res = score(tests, upto) if tests else []
        return {k for k, s in zip(which, res) if s != cur_score}

    def proposals(n: str) -> List[Tuple[str, Ctx, float]]:
        c = ctx[n]
        s = sup[n]
        out = []
        for cls in ("item", "ability", "hp_t"):
            if len(s[cls]) <= 1:
                continue
            p = _norm(s[cls])
            for v, w in p.items():
                if v != getattr(c, cls):
                    new = replace(c, **{cls: v})
                    if cls == "hp_t":
                        new = replace(new, variant=c.variant.split("|")[0])
                    out.append((f"{cls}={v}", new, float(np.log(max(w, 1e-300)))))
        for key in S.iv_keys(c.hp_t):
            if key != c.variant and not (c.variant in ("max", "min") and key == f"{c.variant}|"
                                         + ",".join(map(str, S.iv_patterns(c.hp_t)[0]))):
                out.append((f"iv={key}", replace(c, variant=key), float(np.log(S.iv_key_weight(c.hp_t)))))
        a = arr(n, c.hp_t, c.variant)
        sp = prep.seen.mons[n].species
        cur = np.array(realized(sp, c.nature, c.evs, c.iv()))
        keep = [k for k in range(6) if k not in moving]
        mv = sorted(moving)
        d = np.abs(a[:, keep] - cur[keep]).sum(axis=1) if keep else np.zeros(len(a))
        if mv:
            # every distinct value of the stats the mismatch moves, each by the support row closest to the current
            # spread in the other stats (so the move changes what matters and little else)
            reps: Dict[tuple, int] = {}
            for i in np.argsort(d, kind="stable"):
                key = tuple(int(x) for x in a[i, mv])
                if key not in reps:
                    reps[key] = int(i)
            keys_ = list(reps)
            if len(keys_) > NEAR:
                dd = np.array([sum(abs(x - int(cur[k])) for x, k in zip(kk, mv)) for kk in keys_])
                near = [keys_[j] for j in np.argsort(dd, kind="stable")[:NEAR // 2]]
                rest = [kk for kk in keys_ if kk not in set(near)]
                pick = rng.choice(len(rest), size=min(NEAR // 2, len(rest)), replace=False)
                keys_ = near + [rest[j] for j in pick]
            idx = [reps[kk] for kk in keys_]
        else:
            idx = list(np.argsort(d, kind="stable")[:NEAR + 1])
        idx += list(rng.choice(len(a), size=min(RANDOM, len(a)), p=sw[n] / sw[n].sum()))
        seen_ix = set()
        for i in idx:
            i = int(i)
            if i in seen_ix:
                continue
            seen_ix.add(i)
            nat, evs, w = spreads[n][i]
            if nat == c.nature and tuple(evs) == c.evs:
                continue
            out.append((f"spread={nat}:{evs}", replace(c, nature=nat, evs=tuple(evs)), float(np.log(max(w, 1e-300)))))
        return out

    full = (len(prep.truth_lines) + 1, 0)
    cur = score([ctx])[0]
    rounds = 0
    stuck: set = set()
    while cur < full and rounds < ROUNDS:
        rounds += 1
        who = active_at(prep, max(cur[0], 0))
        order = ([who] if who in ctx and who not in stuck else []) + [n for n in names if n != who and n not in stuck]
        improved = False
        upto = upto_for(max(cur[0], 0))
        cur_t = score([ctx], upto)[0]
        for n in order:
            moving.clear()
            moving.update(sensitivity(n, upto, cur_t))
            props = proposals(n)
            cands = []
            for _lab, c, _lp in props:
                st = dict(ctx)
                st[n] = c
                cands.append(st)
            sc = score(cands, upto)
            ranked = sorted(range(len(sc)), key=lambda i: (sc[i], props[i][2]), reverse=True)
            if VERBOSE:
                print(f"  round {rounds} {n} cur {cur} trunc {cur_t} best {sc[ranked[0]]} {props[ranked[0]][0]} "
                      f"moving {sorted(STATS[k] for k in moving)} upto {upto}")
            if sc[ranked[0]] > cur_t:
                for i in ranked[:4]:
                    if sc[i] <= cur_t:
                        break
                    st = dict(ctx)
                    st[n] = props[i][1]
                    full_sc = score([st])[0]
                    if full_sc > cur:
                        ctx[n] = props[i][1]
                        cur = full_sc
                        improved = True
                        break
                if improved:
                    stuck.clear()
                    break
            stuck.add(n)
        if not improved:
            break
    out: dict = {"rounds": rounds, "score": list(cur), "n_lines": len(prep.truth_lines),
                 "fixed_moves": fixed_moves, "u_world": u_world}
    ok = False
    if cur >= full:
        g = B.gate_many(prep, [world_of(ctx)], rows=True)[0]
        replays += 1
        ok = g.ok
        out["rows_equal"] = g.rows_equal
    out["context_ok"] = ok
    if not ok:
        out.update({"replays": replays, "cpu_s": round(time.process_time() - t0, 2)})
        return out

    pools: Dict[str, list] = {}
    for n in names:
        c = ctx[n]
        sp = prep.seen.mons[n].species
        # discrete classes: every support value tested at the context
        win: Dict[str, set] = {}
        for cls in ("item", "ability", "hp_t"):
            vals_ = list(sup[n][cls])
            if len(vals_) <= 1:
                win[cls] = set(vals_)
                continue
            cands = []
            for v in vals_:
                st = dict(ctx)
                new = replace(c, **{cls: v})
                if cls == "hp_t":
                    new = replace(new, variant=c.variant.split("|")[0])
                st[n] = new
                cands.append(st)
            win[cls] = {v for v, s in zip(vals_, score(cands)) if s >= full}
        keys = S.iv_keys(c.hp_t)
        cands = []
        for key in keys:
            st = dict(ctx)
            st[n] = replace(c, variant=key)
            cands.append(st)
        win["variant"] = {k for k, s in zip(keys, score(cands)) if s >= full} or {c.variant}
        # relevance: each stat moved by ±PERTURB with every other stat fixed (exact synthetic spreads)
        cur_st = list(realized(sp, c.nature, c.evs, c.iv()))
        relevant = []
        tests, which = [], []
        for k in range(6):
            for dlt in (-PERTURB, PERTURB):
                t = list(cur_st)
                t[k] += dlt
                syn = synthesize(sp, t, c.iv(), c.nature)
                if syn is None:
                    continue
                st = dict(ctx)
                st[n] = replace(c, nature=syn[0], evs=syn[1], ivs=syn[2])
                tests.append(st)
                which.append(k)
        res = score(tests) if tests else []
        bad = {k for k, s in zip(which, res) if s < full}
        tested = set(which)
        relevant = [k for k in range(6) if k in bad or k not in tested]
        # group the support (x attack-IV variant in the window) by realized RELEVANT stats; test one per group,
        # nearest groups first, until a whole batch fails
        groups: Dict[tuple, list] = {}
        groups_keys = keys
        for var in sorted(win["variant"]):
            a = arr(n, c.hp_t, var)
            for i, row in enumerate(a):
                key = tuple(int(row[k]) for k in relevant)
                groups.setdefault(key, []).append((i, var))
        cur_key = tuple(cur_st[k] for k in relevant)
        keys = sorted(groups, key=lambda kk: sum(abs(x - y) for x, y in zip(kk, cur_key)))
        _ = groups_keys
        good_keys = set()
        b = 0
        pos = 0
        while pos < len(keys) and b < POOL_BATCHES:
            batch = keys[pos:pos + POOL_BATCH]
            pos += len(batch)
            b += 1
            cands = []
            for kk in batch:
                i, var = groups[kk][0]
                nat, evs, _w = spreads[n][i]
                st = dict(ctx)
                st[n] = replace(c, nature=nat, evs=tuple(evs), variant=var)
                cands.append(st)
            res = score(cands)
            hits = [kk for kk, s in zip(batch, res) if s >= full]
            good_keys.update(hits)
            if not hits and b >= 2:
                break
        pool = []
        for kk in good_keys:
            for i, var in groups[kk]:
                nat, evs, w = spreads[n][i]
                pool.append(((nat, tuple(evs), var), w * S.iv_key_weight(c.hp_t)))
        items = {k: v for k, v in _norm(sup[n]["item"]).items() if k in win["item"]}
        abil = {k: v for k, v in _norm(sup[n]["ability"]).items() if k in win["ability"]}
        hpt = {k: v for k, v in _norm(sup[n]["hp_t"]).items() if k in win["hp_t"]}
        if not pool or not items or not abil or not hpt:
            out.setdefault("empty_pool", []).append(n)
            continue
        pools[n] = pool
        out.setdefault("pool_items", {})[n] = items
        out.setdefault("pool_abilities", {})[n] = abil
        out.setdefault("pool_hp", {})[n] = {str(k): v for k, v in hpt.items()}
        out.setdefault("pool_hp_raw", {})[n] = hpt
        out.setdefault("pool_sizes", {})[n] = {"spreads": len(pool), "groups_ok": len(good_keys),
                                               "groups": len(keys), "relevant": [STATS[k] for k in relevant],
                                               "items": len(items), "abilities": len(abil), "hp_types": len(hpt)}
    out.update({"pools": pools, "context": ctx, "replays": replays, "cpu_s": round(time.process_time() - t0, 2)})
    return out


def draw_from_pools(scr: dict, rng: np.random.Generator) -> Dict[str, B.SeenDraw]:
    out = {}
    for n, pool in scr["pools"].items():
        w = np.array([x[1] for x in pool], dtype=float)
        nat, evs, var = pool[int(rng.choice(len(pool), p=w / w.sum()))][0]
        item = S.weighted_pick(scr["pool_items"][n], rng)
        ab = S.weighted_pick(scr["pool_abilities"][n], rng)
        hp = scr["pool_hp_raw"][n]
        keys = list(hp)
        pw = np.array([hp[k] for k in keys], dtype=float)
        t = keys[int(rng.choice(len(keys), p=pw / pw.sum()))]
        if t != scr["context"][n].hp_t:
            var = var.split("|")[0]          # the pattern belonged to the context's type
        out[n] = B.SeenDraw(item, ab, nat, evs, var, t)
    return out
