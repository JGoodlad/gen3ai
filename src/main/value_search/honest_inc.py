"""THE INCREMENTAL LEAK-HONEST BUILDER (``r0s4_honest_inc``; ``designs/endstate/design_sprime_honest.md`` §9): the
worlds of :mod:`main.value_search.honest_worlds`, CARRIED from one searched decision of a battle to the next instead
of rebuilt from turn 0 at each. Offline / evaluation use only.

**Why.** ``r0s4_honest`` re-matches the whole of I at every decision and restarts its chain cold, so a battle's world
building grows with the square of its length (S′-HONEST stage 1b: late decisions cost up to ≈ 330 CPU-s).

**Everything stays a function of I.** I at a decision extends I at every earlier one (its lines and commands are a
prefix). The CHECKPOINTS of I are the earlier decisions the live searcher searches (``info_set.checkpoints``: turn >=
its first turn, >= 2 legal actions), read off I. The CARRY at a checkpoint is a function of the carry at the one before
and of its own I; the first is built cold. So the carry at any decision, and the decision's record, is a function of
its I alone, whatever the process saw before: :class:`HonestIncSampler` keeps carries in a per-process cache keyed by
I's digest and, on a miss, REBUILDS the missing checkpoints in order from the latest cached one (from the first
checkpoint when none is), the same computation. A live process plays a battle's decisions in order, so it hits; a unit
retry (a new process replaying the battle) or an offline re-score of one decision rebuilds, and gets the same bytes.

**A step at decision k** (:meth:`HonestIncSampler.step`), from the carry of the previous checkpoint:

1. *Extend.* Every carried world is re-rendered under this I's scaffold (a mon newly seen: its coordinate drawn and the
   never-seen mons redrawn; a move newly revealed that its set lacks: its unrevealed moves redrawn; a coordinate this
   I rules out: redrawn; keyed draws, :data:`REVEAL_ATTEMPTS` of them) and matched with its CARRIED TRAJECTORY
   (``pokesim::fresh_dice`` ``prefix``): the segments of the I it last matched replayed exactly up to the first whose
   header (the opponent's intent, the counters' onsets, the hidden-HP hits) this I changes, the rest on this
   decision's fresh dice (tag ``x<world>:<step>:<attempt>``).
2. *Fresh.* :data:`FRESH_MIN` worlds at least (more to fill J) from the WARM chain (after a failure, deferred on the
   stuck backoff while the carried worlds alone fill the decision: :func:`chain_due`): ``honest_worlds``' joint chain
   started at the carried chain state, sampling at once when it fits, on ONE set of chain dice for the battle (keyed by
   the I of the checkpoint the chain's lineage began at, so a state that fit before replays its kept tries), each new
   world on its own fresh dice (tag ``w<world id>:<attempt>``), as the cold builder's.
3. *Shortfall.* The worlds that failed step 1 once more on rarer dice (``CHAIN_TRIES_MAX`` tries).
4. *Compose and retire.* The fresh worlds (max(FRESH_MIN, J - the survivors)), then the youngest survivors, to J: the
   oldest retire, so a world serves at most ⌈J / FRESH_MIN⌉ decisions while the battle draws fresh worlds every
   decision; after a stretch that drew none, or with too few survivors, older ones stay (an OVERAGE, recorded): the
   CORRELATION BOUND, soft (§9 of the design, C17, S'H-15).
5. *Gate.* Every returned world passes the STRICT gate (one lean ``core_events`` replay over all of them,
   ``info_set.extract_many``) and the playout tool. Fewer than J worlds: an EXCLUSION (:class:`WorldError`), whose
   carry keeps EVERY world: the extended ones at this I, the rest PENDING at the I they last matched, so the next
   decision retries a hard segment on its own dice instead of rebuilding the battle's history (a carry with no world at
   all, a battle whose first decision failed, is rebuilt cold: ``r0s4_honest``'s own build). After an exclusion the
   battle is STUCK: its next decisions first PROBE (the carried worlds' extensions, one redraw each) and go on only
   when one extended; the full attempt comes back at the :data:`STUCK_FULL_EVERY`-th stuck decision and at every
   doubling after it; so a segment no world can match costs a few CPU-s a decision.

The record's ``honest.inc`` block: the step, the carried and returned carries' digests (``carry_in``, ``carry_out``:
a battle's records chain, each ``carry_in`` the previous ``carry_out``), the mode, the boundary, the worlds pending,
extended, failed, escalated, new and retired, their ages and the overage, the chain's fate and its failures, how long
the battle is stuck. A RE-SCORE-SAMPLED record also keeps the carry it started from (``inc_carry_in``,
:func:`pack`): a re-score seeds a fresh searcher with it (:meth:`HonestIncSampler.seed`) and re-runs that decision
alone, its digest checked by the record's ``carry_in``. Never the cache's history (whether a carry was cached or
rebuilt is :attr:`HonestIncSampler.last_rebuilt`, process state, outside the record).
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import math
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Union

from main.value_search import honest_worlds as HW
from main.value_search import info_set as IS
from main.value_search import world_screen as WS
from main.value_search.search import cpu_all
from main.value_search.world_draws import WorldError

CARRY_SCHEMA = "gen3_honest_inc_carry_v1"
#: each decision draws max(FRESH_MIN, J - the survivors) worlds fresh (when the chain yields them) and the oldest retire:
#: a world's life <= ceil(J / FRESH_MIN) decisions while every decision draws fresh ones (else an overage, recorded)
FRESH_MIN = 2
#: a carried world's tries a new segment (one attempt: a failure is cheaper to replace from the chain than to retry
#: on rarer dice)
EXT_TRIES = HW.WORLD_TRIES
#: keyed redraws of what a newly seen mon / a newly revealed move changed in a carried world
REVEAL_ATTEMPTS = 4
#: a fresh world's draws of its never-seen mons on its own dice (the cold builder's ATTEMPTS is B3's 8: a fresh world
#: that cannot re-solve the battle's history in two is better replaced by a carried one than paid for)
FRESH_ATTEMPTS = 2
#: chain samples beyond the shortfall (a new world that fails its own replay takes the next)
SPARES_INC = 2
#: the warm chain's budget while a surviving world can be CLONED (stage 1's base budget: a chain that cannot fit then
#: costs ≈ 30 CPU-s, not B3's ≈ 200); with no survivor it is the only source of worlds and takes the builder's own (B3)
WARM_STEPS, WARM_FEEDS = 300, 600_000
#: carries a process keeps (one a live battle is enough; LRU)
CACHE = 64
#: after an EXCLUSION the battle is STUCK: its next decisions first PROBE (the carried worlds' extensions, one redraw of
#: what a reveal changed) and go on only when a world extended; the full attempt (every redraw, the warm chain with no
#: survivor, the escalation) comes back at the STUCK_FULL_EVERY-th stuck decision and at every doubling after it (4, 8,
#: 16, …). A cold rebuild while no world was ever built retries in full at the first STUCK_FULL_EVERY - 1 decisions,
#: then on the same doublings. A segment no world can match (S'H-10) then costs a few CPU-s a decision and a logarithmic
#: number of full attempts (≈ 100-200 CPU-s each), not one every few decisions
STUCK_FULL_EVERY = 4


def chain_due(fails: int, short: bool) -> bool:
    """Whether the warm chain runs: always when the carried worlds cannot fill the decision (``short``), else on the
    same backoff after a failure (``fails``: decisions since the chain last failed and did not fit since; 0 = it fits).
    In a hard battle the chain re-matches the whole history on its common dice and can fail every decision (≈ 30-45
    CPU-s each) while every carried world extends: rejuvenation waits, feasibility does not."""
    return short or full_attempt(fails)


def full_attempt(stuck: int) -> bool:
    """Whether a battle stuck for ``stuck`` decisions makes the full attempt (:data:`STUCK_FULL_EVERY`)."""
    return stuck == 0 or (stuck >= STUCK_FULL_EVERY and stuck & (stuck - 1) == 0)

Coord = Tuple[WS.State, List[str]]


def digest(text: Optional[str]) -> Optional[str]:
    return None if text is None else hashlib.sha256(text.encode()).hexdigest()


def _at(w: dict, carry: dict) -> Tuple[int, int]:
    """The I (its line and command counts) a carried world's trajectory matched."""
    return int(w.get("lines", carry["lines"])), int(w.get("icmds", carry["cmds"]))


def _child_cpu(m: Optional[HW.Matcher]) -> float:
    """A live matcher child's CPU-s so far (0 when it has exited or ``/proc`` is unreadable)."""
    if m is None or m.p.returncode is not None:
        return 0.0
    try:
        f = open(f"/proc/{m.p.pid}/stat").read().rsplit(")", 1)[1].split()
    except OSError:
        return 0.0
    return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")


def pack(carry: str) -> str:
    """A carry as a record field (gzip, base64); :func:`unpack` inverts it."""
    return base64.b64encode(gzip.compress(carry.encode(), mtime=0)).decode()


def unpack(field: str) -> str:
    return gzip.decompress(base64.b64decode(field)).decode()


def _state_json(st: WS.State) -> list:
    item, ab, nat, evs, hpt = st
    return [item, ab, nat, [int(x) for x in evs], hpt]


def _state_of(x: Sequence) -> WS.State:
    return (str(x[0]), str(x[1]), str(x[2]), tuple(int(v) for v in x[3]), None if x[4] is None else str(x[4]))


def coords_json(sc: HW.Scaffold, st: Sequence[Coord]) -> list:
    return [[m.nick, _state_json(c[0]), list(c[1])] for m, c in zip(sc.seen, st)]


def coords_of(x: Optional[Sequence]) -> Dict[str, Coord]:
    return {} if x is None else {str(n): (_state_of(s), [str(v) for v in mv]) for n, s, mv in x}


def boundary(prev: HW.Scaffold, cur: HW.Scaffold) -> int:
    """The first segment of the earlier I whose header this I changes (the opponent's intent — a choice first seen
    in a later segment —, the onsets a failure may re-roll, the hidden-HP hits); the earlier I's length when none
    does. A carried trajectory is replayed up to it."""
    a, b = json.loads(prev.header), json.loads(cur.header)
    n = len(a["segments"])
    if b["segments"][:n] != a["segments"]:
        raise RuntimeError("the earlier I is not a prefix of this one")
    for s in range(n):
        if any(a[k][s] != b[k][s] for k in ("intents", "backtrack", "hp_last")):
            return s
    return n


class HonestIncSampler(HW.HonestSampler):
    """``HonestIncSampler(from_turn=10).sample_info(info, J, root_row, points=…)`` -> :class:`HW.HonestWorlds`
    (module docstring); ``points``: I's earlier decisions (``info_set.extract_points``)."""

    dice = "fresh_inc"

    def __init__(self, *a, from_turn: int = 10, fresh_min: int = FRESH_MIN, cache: int = CACHE, **k) -> None:
        super().__init__(*a, **k)
        self.from_turn = int(from_turn)
        self.fresh_min = int(fresh_min)
        self.cache = int(cache)
        self.store: "OrderedDict[str, str]" = OrderedDict()
        #: PROCESS HISTORY, never in a record: checkpoints the last call rebuilt (0 when the predecessor was cached)
        self.last_rebuilt = 0
        #: PROCESS HISTORY, never in a record: the last step's CPU by phase (``search.cpu_all`` deltas: the process and
        #: its reaped children)
        self.last_phases: Dict[str, float] = {}
        #: the carry the last call's decision started from (None: built cold); the live searcher packs it into a
        #: re-score-sampled record (:func:`pack`), so an offline re-score of that decision alone needs no rebuild
        self.last_carry_in: Optional[str] = None

    def seed(self, carry: str) -> None:
        """Put a carry into the cache under its own I's digest (an offline re-score handing the decision the carry
        its record names: the decision then runs from it, and its ``carry_in`` digest checks it)."""
        self._put(str(json.loads(carry)["key"]), carry)

    # -- the cache ------------------------------------------------------------------------------------------
    def _put(self, key: str, carry: str) -> None:
        self.store[key] = carry
        self.store.move_to_end(key)
        while len(self.store) > self.cache:
            self.store.popitem(last=False)

    def sample_info(self, info: IS.InfoSet, J: int, root_row: Optional[str] = None,
                    points: Optional[Sequence[IS.Point]] = None) -> HW.HonestWorlds:
        if points is None:
            raise ValueError("the incremental builder needs I's earlier decisions (info_set.extract_points)")
        cps = IS.checkpoints(points, self.from_turn)
        pre = [IS.prefix(info, p.lines, p.cmds) for p in cps]
        start, carry = 0, None
        for i in range(len(pre) - 1, -1, -1):
            got = self.store.get(pre[i].key())
            if got is not None:
                start, carry = i + 1, got
                break
        self.last_rebuilt = len(pre) - start
        for i in range(start, len(pre)):
            _w, carry = self.step(pre[i], J, cps[i].row, carry)
        self.last_carry_in = carry
        w, _out = self.step(info, J, root_row, carry)
        if isinstance(w, WorldError):
            raise w
        return w

    # -- one step -------------------------------------------------------------------------------------------
    def step(self, info: IS.InfoSet, J: int, root_row: Optional[str],
             carry_in: Optional[str]) -> Tuple[Union[HW.HonestWorlds, WorldError], str]:
        """The decision at ``info`` from the previous checkpoint's carry (None: the first, built cold): its worlds (or
        the exclusion) and its carry, which the cache keeps under I's digest."""
        key = info.key()
        sc = HW.Scaffold.of(info)
        rec: dict = {"seen": len(sc.seen), "size": sc.size, "segments": len(sc.segments), "evaluations": 0,
                     "feeds": 0, "steps": 0}
        carry = json.loads(carry_in) if carry_in is not None else None
        inc: dict = {"step": 0 if carry is None else int(carry["step"]) + 1, "carry_in": digest(carry_in)}
        self.last_phases = {}
        with HW.Matcher(sc.header) as m:
            self._m = m
            self._t = cpu_all()
            if carry is None:
                res, worlds, chain, key0, ctries, nwid = self._cold(info, sc, key, J, root_row, rec, m, inc)
            else:
                res, worlds, chain, key0, ctries, nwid = self._warm(info, sc, key, J, root_row, rec, m, carry, inc)
        stuck = 0 if isinstance(res, HW.HonestWorlds) else int((carry or {}).get("stuck") or 0) + 1
        cf = int((carry or {}).get("chain_fail") or 0)
        cf = 0 if inc.get("chain") in ("fit", "refit", "cold") else (cf + 1 if (cf or inc.get("chain") == "failed") else 0)
        out = json.dumps({"schema": CARRY_SCHEMA, "key": key, "lines": len(info.lines), "cmds": len(info.cmds),
                          "step": inc["step"], "key0": key0, "chain_tries": ctries, "next_wid": nwid,
                          "worlds": worlds, "chain": chain, "stuck": stuck, "chain_fail": cf},
                         sort_keys=True, separators=(",", ":"))
        inc["carry_out"] = digest(out)
        self._put(key, out)
        res.inc = inc                                    # the exclusion carries it too (the record's ``honest.inc``)
        if isinstance(res, HW.HonestWorlds):
            res.chain = rec
        return res, out

    def _cold(self, info, sc, key, J, root_row, rec, m, inc, born: int = 0):
        """The first checkpoint (or a battle with no world carried yet): ``r0s4_honest``'s own build (the same worlds,
        byte for byte), its worlds born at step ``born``."""
        inc["mode"] = "cold"
        try:
            w = self._build(info, sc, key, J, root_row, rec, m)
        except WorldError as e:
            inc["chain"] = "failed"
            best = getattr(e, "best", None)
            return (e, [], (coords_json(sc, best) if best else None), key, int(rec.get("chain_tries", HW.CHAIN_TRIES)),
                    J)
        worlds = [self._carried(sc, j, born, st, team, r, info)
                  for j, (st, r, team) in enumerate(zip(w.samples, w.reports, w.teams))]
        inc.update(ages=[0] * J, retired=0, extended=[0, 0], new=J, overage=0, chain="cold")
        return w, worlds, coords_json(sc, w.samples[-1]), key, int(rec.get("chain_tries", HW.CHAIN_TRIES)), J

    @staticmethod
    def _carried(sc: HW.Scaffold, wid: int, born: int, st: Sequence[Coord], team: str, r: dict,
                 info: IS.InfoSet) -> dict:
        """A world as the carry keeps it: its seen mons' coordinates, its never-seen entries, its team, its trajectory
        and the I it matched (``lines``, ``cmds``)."""
        ents = team.split("]")
        return {"wid": int(wid), "born": int(born), "coords": coords_json(sc, st), "never": ents[len(sc.seen):],
                "team": team, "seed": r["seed"], "trail": [list(x) for x in r["trail"]], "cmds": list(r["cmds"]),
                "reseed": [list(x) for x in r["reseed"]], "rolls": [list(x) for x in r["rolls"]],
                "log_w": r.get("log_w"), "lines": len(info.lines), "icmds": len(info.cmds)}

    # -- projection onto this I ------------------------------------------------------------------------------
    def _project(self, sc: HW.Scaffold, key: str, tag: str, coords: Dict[str, Coord]) -> Tuple[List[Coord], int]:
        """Carried coordinates re-read under this I's scaffold: a mon kept as carried when this I admits its state and
        its moves hold every revealed one; its moves redrawn when they lack one; its whole coordinate drawn when it is
        newly seen or its state ruled out (keyed by ``tag``). ``(coordinates, how many changed)``."""
        rng = HW.rng_of(key, "inc_project", tag)
        out: List[Coord] = []
        changed = 0
        for i, mon in enumerate(sc.seen):
            ms = self._sampler(f"{key}:{tag}", i, mon)
            c = coords.get(mon.nick)
            if c is not None and ms.logp(c[0]) > -math.inf:
                st, mv = c
                if all(x in mv for x in mon.moves):
                    out.append((st, list(mv)))
                    continue
                out.append((st, self._fit(rng, mon, st, None)))
            else:
                st = ms.draw()
                out.append((st, self._fit(rng, mon, st, None)))
            changed += 1
        return out, changed

    def _phase(self, name: str) -> None:
        """Book the CPU since the last phase to ``name``: this process and its reaped children, plus the live matcher
        child's own (read from ``/proc``: it is reaped only at the step's end)."""
        t = cpu_all() + _child_cpu(getattr(self, "_m", None))
        self.last_phases[name] = round(self.last_phases.get(name, 0.0) + t - self._t, 3)
        self._t = t

    def _render(self, sc: HW.Scaffold, st: Sequence[Coord]) -> List[str]:
        return [self._entry(mon, st[i][0], st[i][1]) for i, mon in enumerate(sc.seen)]

    def _legal(self, sc: HW.Scaffold, ents: Sequence[str], team: str) -> bool:
        from main.value_search.worlds import team_problems

        return (all(self.legality.set_ok(e) for e in ents) and not team_problems(team)
                and not self.legality.team_problems([team])[0])

    # -- the warm step -----------------------------------------------------------------------------------------
    def _warm(self, info, sc, key, J, root_row, rec, m, carry, inc):
        prev = IS.prefix(info, int(carry["lines"]), int(carry["cmds"]))
        if prev.key() != carry["key"]:
            raise RuntimeError(f"{key}: the carry is of another I than this one's prefix")
        stuck = int(carry.get("stuck") or 0)
        full = full_attempt(stuck)
        inc["stuck"] = stuck
        if not carry["worlds"]:
            # no world was ever built (the battle's first decisions were excluded): r0s4_honest's own cold build at this
            # I (early, cheap and mostly successful), then only on the stuck doublings
            if not (full or stuck < STUCK_FULL_EVERY):
                inc["mode"] = "stuck"
                return (WorldError(f"{key}: no world built yet; stuck {stuck}, the cold build waits"), [], carry["chain"],
                        "", HW.CHAIN_TRIES, int(carry["next_wid"]))
            res, worlds, chain, key0, ctries, _n = self._cold(info, sc, key, J, root_row, rec, m, inc,
                                                              born=int(carry["step"]) + 1)
            inc["mode"] = "recold"
            nxt = max(int(carry["next_wid"]), J)
            return res, worlds, chain, key0, ctries, nxt
        scs: Dict[Tuple[int, int], HW.Scaffold] = {}

        def scaffold_at(lines: int, cmds: int) -> HW.Scaffold:
            if (lines, cmds) not in scs:
                scs[(lines, cmds)] = HW.Scaffold.of(IS.prefix(info, lines, cmds))
            return scs[(lines, cmds)]

        b0 = boundary(scaffold_at(int(carry["lines"]), int(carry["cmds"])), sc)
        self._phase("setup")
        step = int(carry["step"]) + 1
        # the battle's chain dice: the key of the checkpoint the chain's lineage started at (reset after a failure), and
        # the tries its verdicts were last judged on
        key0 = str(carry["key0"]) if carry.get("key0") else key
        ctries = int(carry.get("chain_tries") or HW.CHAIN_TRIES)
        pending = sum(1 for w in carry["worlds"] if _at(w, carry) != (int(carry["lines"]), int(carry["cmds"])))
        inc.update(mode="warm" if full else "stuck", boundary=b0,
                   segments_prev=len(scaffold_at(int(carry["lines"]), int(carry["cmds"])).segments), pending=pending)
        fails: Dict[str, int] = {}
        # 1. EXTEND every carried world, each from the I its trajectory matched (a world an exclusion left PENDING lags)
        old = sorted(carry["worlds"], key=lambda w: int(w["wid"]))
        survivors: Dict[int, tuple] = {}                          # wid -> (carried world, report, team, coordinates)
        revised = 0
        for cw in old:
            got = self._extend(info, sc, key, step, cw, scaffold_at(*_at(cw, carry)), rec, fails, m,
                               attempts=REVEAL_ATTEMPTS if full else 1)
            if got is not None:
                survivors[int(cw["wid"])] = (cw,) + got[:3]
                revised += int(got[3] > 0)
        inc["extended"] = [len(survivors), len(old) - len(survivors)]
        inc["revised"] = revised
        self._phase("extend")
        if not full and not survivors:
            # the stuck PROBE found no world past the segment that stopped the battle: excluded at once
            inc.update(mode="stuck", chain="skipped", new=0, retired=0, ages=[], overage=0)
            return (WorldError(f"{key}: stuck {stuck}: no carried world extends (the probe; {fails})"),
                    sorted(carry["worlds"], key=lambda w: (-int(w["born"]), int(w["wid"]))), carry["chain"], key0,
                    ctries, int(carry["next_wid"]))
        # 2. FRESH worlds from the warm chain, on the battle's one set of chain dice (each on its own fresh dice)
        need = max(self.fresh_min, J - len(survivors))
        next_wid = int(carry["next_wid"])
        if ctries > HW.CHAIN_TRIES:
            rec["chain_tries"] = ctries
        init_c = self._project(sc, key, "chain", coords_of(carry["chain"]))[0] if carry["chain"] else None
        samples: List[list] = []
        chain_state: Optional[list] = None
        saved = (self.max_steps, self.chain_feeds, self.attempts)
        if survivors:
            self.max_steps, self.chain_feeds = min(WARM_STEPS, saved[0]), min(WARM_FEEDS, saved[1])
        cfails = int(carry.get("chain_fail") or 0)
        inc["chain_fail"] = cfails
        try:
            if not chain_due(cfails, len(survivors) < J):
                inc["chain"] = "deferred"
            else:
                samples = self._chain(sc, key, need + SPARES_INC, rec, m, init=init_c, dice_key=key0, tries0=ctries)
                chain_state = samples[-1] if samples else None
                inc["chain"] = rec.get("warm", "cold")
        except WorldError:
            inc["chain"] = "failed"
        finally:
            self.max_steps, self.chain_feeds = saved[0], saved[1]
        ctries = int(rec.get("chain_tries", ctries))
        self._phase("chain")
        new: List[tuple] = []                                       # (wid, coordinates, built)
        self.attempts = min(FRESH_ATTEMPTS, saved[2])
        try:
            for st in samples:
                if len(new) >= need:
                    break
                wid = next_wid
                next_wid += 1
                built = self._one_world(info, sc, key, wid, st, root_row, rec, fails, m, strict=False)
                if built is not None:
                    new.append((wid, st, built))
        finally:
            self.attempts = saved[2]
        self._phase("new_worlds")
        # 3. a SHORTFALL: the worlds that failed the cheap extension, once more on rarer dice
        if full and len(new) + len(survivors) < J:
            esc = 0
            for cw in old:
                if int(cw["wid"]) in survivors:
                    continue
                esc += 1
                got = self._extend(info, sc, key, step, cw, scaffold_at(*_at(cw, carry)), rec, fails, m,
                                   tries=HW.CHAIN_TRIES_MAX, tag="e", attempts=1)
                if got is not None:
                    survivors[int(cw["wid"])] = (cw,) + got[:3]
                    revised += int(got[3] > 0)
                if len(new) + len(survivors) >= J:
                    break
            inc["escalated"] = [esc, len(survivors) - inc["extended"][0]]
            self._phase("escalate")
        # 4. COMPOSE: the fresh worlds, then the youngest survivors (the oldest retire; an old one stays only when no
        # fresh world replaced it: an OVERAGE, recorded)
        surv = sorted(survivors.values(), key=lambda x: (-int(x[0]["born"]), int(x[0]["wid"])))
        keep = surv[:max(0, J - len(new))]
        worlds: List[dict] = [self._carried(sc, int(cw["wid"]), int(cw["born"]), st, team, r, info)
                              for cw, r, team, st in sorted(keep, key=lambda x: int(x[0]["wid"]))]
        worlds += [self._carried(sc, wid, step, st, b[1], b[5], info) for wid, st, b in new]
        life = -(-J // max(1, self.fresh_min))
        inc.update(new=len(new), retired=len(old) - len(keep), ages=[step - int(w["born"]) for w in worlds],
                   overage=sum(1 for w in worlds if step - int(w["born"]) >= life))
        if chain_state is None and inc["chain"] == "deferred":
            chain_json = carry["chain"]
        elif chain_state is None:
            # a chain that found no fit restarts next time from the last state that fit, on NEW chain dice (the next
            # decision's key: dice no candidate met stay unmet on the same seeds)
            chain_json = carry["chain"]
            key0, ctries = "", HW.CHAIN_TRIES
        else:
            chain_json = coords_json(sc, chain_state)
        if len(worlds) < J:
            # an EXCLUSION keeps every world: the extended ones at this I, the rest PENDING at their own (the next
            # decision retries them over every segment they lack, on its own dice), youngest first, at most J + spares
            pend = [w for w in old if int(w["wid"]) not in survivors]
            keepall = worlds + sorted(pend, key=lambda w: (-int(w["born"]), int(w["wid"])))
            return (WorldError(f"{key}: {J - len(worlds)} of {J} incremental worlds not built ({len(pend)} carried "
                               f"failed, chain {inc['chain']}, {fails})"), keepall[:J + SPARES_INC], chain_json, key0,
                    ctries, next_wid)
        # 5. THE STRICT GATE over every returned world, in one replay
        logs = [self._world_log(info, sc, w["team"], w) for w in worlds]
        for j, (got_j, row) in enumerate(IS.extract_many(logs, info.side, info.lines)):
            if got_j.lines != info.lines or got_j.cmds != info.cmds:
                raise RuntimeError(f"{key}: incremental world {j}'s strict replay ships other lines than I")
            if root_row is not None and row != root_row:
                raise RuntimeError(f"{key}: incremental world {j}'s root row is not the live row")
        self._phase("strict")
        ats = {len(lg["cmds"]) for lg in logs}
        if len(ats) != 1:
            raise RuntimeError(f"{key}: the worlds' roots sit at different commands {sorted(ats)}")
        stats: List[dict] = []
        tries: List[int] = []
        for cw, r, _team, _st in sorted(keep, key=lambda x: int(x[0]["wid"])):
            stats.append({"kind": "carried", "wid": int(cw["wid"]), "born": int(cw["born"]), "fixed": int(r["fixed"]),
                          "attempt": str(r["_attempt"]), "rerolls": r["rerolls"], "feeds": r["feeds"],
                          "rolls": len(r["rolls"])})
            tries.append(1)
        for wid, _st, b in new:
            stats.append({"kind": "new", "wid": wid, "born": step, **b[3]})
            tries.append(int(b[2]))
        leak = {"builder": "honest_inc", "key": key, "residual": [], "partial": []}
        res = HW.HonestWorlds(logs=logs, teams=[w["team"] for w in worlds], at=ats.pop(), tries=tries, leak=leak,
                              revealed={mon.sid: list(mon.moves) for mon in sc.seen}, stats=stats, fails=fails,
                              chain=rec, log_w=[float(w["log_w"]) if w["log_w"] is not None else float("nan")
                                                for w in worlds])
        return res, worlds, chain_json, key0, ctries, next_wid

    def _extend(self, info, sc, key, step, cw, sc_w, rec, fails, m, tries: int = EXT_TRIES, tag: str = "",
                attempts: int = REVEAL_ATTEMPTS) -> Optional[tuple]:
        """A carried world extended to this I (module docstring, step 2) from the I its trajectory matched (``sc_w``,
        its scaffold): ``(report, team, coordinates, changed)`` or None. Its trajectory is replayed up to the first
        segment of that I whose header this one changes; its never-seen mons are redrawn only when a mon was newly
        seen since (one of them is now seen)."""
        carried = coords_of(cw["coords"])
        new_seen = len(sc.seen) - len(carried)
        prefix = {"seed": cw["seed"], "trail": cw["trail"], "cmds": cw["cmds"], "upto": int(boundary(sc_w, sc))}
        for a in range(attempts):
            t = f"x{int(cw['wid'])}:{step}:{tag}{a}"
            st, changed = self._project(sc, key, t, carried)
            if new_seen > 0:
                never = self._never(HW.rng_of(key, "inc_never", int(cw["wid"]), f"{tag}{a}"), sc, legal=True)
                if never is None:
                    fails["never_seen"] = fails.get("never_seen", 0) + 1
                    continue
            else:
                never = list(cw["never"])
            ents = self._render(sc, st)
            team = "]".join(ents + never)
            if team != cw["team"] and not self._legal(sc, ents, team):
                fails["illegal_team"] = fails.get("illegal_team", 0) + 1
                continue
            job = self._job(sc, team, t, key, True, tries)
            job["prefix"] = prefix
            r = m([job])[0]
            rec["feeds"] += int(r.get("feeds", 0))
            if r["ok"] and self._refusal(info, sc, team, r) is None:
                r["_attempt"] = f"{tag}{a}"
                return r, team, st, changed + new_seen
            touched = changed > 0 or new_seen > 0
            fails[("extend_revised" if touched else "extend") + ("_escalated" if tag else "")] = \
                fails.get(("extend_revised" if touched else "extend") + ("_escalated" if tag else ""), 0) + 1
            if not touched:
                break                                    # nothing to redraw: another attempt is the same match
        return None

    def _refusal(self, info, sc, team, r) -> Optional[str]:
        return self._playout_refusal(self._world_log(info, sc, team, r), info.side)
