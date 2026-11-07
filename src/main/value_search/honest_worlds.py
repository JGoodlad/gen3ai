"""THE LEAK-HONEST WORLD BUILDER: determinized worlds that are a function of the searched side's information set I
alone (``r0s4_honest``; ``designs/endstate/design_sprime_honest.md`` §2, D1-D8). Offline / evaluation use only.

The input is I (:class:`main.value_search.info_set.InfoSet`), never the battle's log: no seed, no opponent team, no
opponent command enters. Every draw is keyed by I's digest (D7), so an alternative truth that ships the searched side
the same lines yields the same worlds, byte for byte (the leak test, ``honest_worlds_inv_test``).

A world is the opponent's team, a command list and a RESEED SCHEDULE:

* **The scaffold (D2).** The opponent's team size from ``|teamsize|``; each SEEN mon's nickname, species, level,
  gender and shininess from its switch details; its revealed moves, item and ability from the ONE log source
  (``agents.info_levers.revealed``); the party in first-appearance order (the lead first), never-seen mons last; every
  gender NAMED (ours from our own request), so no world's construction draws one; default happiness (0 when the set
  holds Frustration). A seen mon's unrevealed moves are drawn from the prior with the four slots permuted uniformly,
  and are part of the joint chain's state (below); a never-seen mon is drawn whole, as today.
* **Fresh dice (D3, D4, D6).** ``pokesim::fresh_dice`` (``core_events`` ``INFOSET`` / ``MATCH``) replays a world on
  dice keyed by (I, the world, the segment, the try) and keeps it iff every line the searched side was shipped is
  reproduced; the opponent's commands are rebuilt from I's lines (``info_set.intents``), an unseen choice a stand-in
  the replay resolves; a running hidden counter is re-rolled at its onset, the opponent's hidden exact HP at its hits.
* **The joint chain (D5).** One chain over every seen mon's item, ability, spread (nature, EVs, the IVs a Hidden
  Power type fixes) AND unrevealed moves (a spread can fit I only through a move the mon held and never showed: a
  fast mon that chose a negative-priority move), with no true background: its score is the matched length under
  fresh dice (common dice across
  candidates: the ``chain`` tag); a search step moves the mon the first unmatched segment implicates (the opponent's
  active mon there, ``worlds.divergence_hint``'s stat) with ``world_screen.MonSampler``'s own kernels; once I fits,
  Metropolis-Hastings over a uniformly chosen mon, thinned. A candidate the prior rules out is never replayed.
* **Each world (D3).** World j takes the chain's j-th sample (its seen mons whole, their moves included) and its own
  never-seen mons, and is replayed on its OWN fresh dice (tag ``w<j>:<attempt>``), more tries and re-rolls than a
  chain step (a later attempt 1,024 tries: rare dice); a world whose replay fails redraws its never-seen mons, then
  takes a spare chain sample. Every returned world then passes the STRICT
  gate: ``core_events`` replays it on its schedule with every check, its lines must be I's and its root row the live
  row (a difference is a defect, raised), and the playout tool must open it.
* **Exclusions (D8).** No fit within the chain's budget, or too few worlds: :class:`WorldError` (the live searcher
  excludes the decision and π plays). A function of I alone.

FINDINGs it carries (named in the design doc): the per-segment rejection weights a world's dice path by how easy its
segments were to match (a normaliser per segment, a bias not a leak); a pinned damage site (Beat Up, a Future Sight
cast, Focus Band / Endure / False Swipe when lethal) is matched on fresh dice alone.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.info_levers import revealed as RV
from agents.info_levers.consumers import SEARCH
from main.value_search import info_set as IS
from main.value_search import world_draws as WD
from main.value_search import world_legality as WL
from main.value_search import world_screen as WS
from main.value_search.world_draws import Prior, WorldError, is_hp, parse_set, typed_moves

KEY_TAG = "sprime_honest_v1"
#: fresh-dice settings: a CHAIN step (a cheap verdict on common dice) and each returned WORLD (its own dice)
CHAIN_TRIES, CHAIN_REROLLS = 64, 8
WORLD_TRIES, WORLD_REROLLS = 128, 24
#: draws of a world's unseen parts before it takes a spare chain sample; spare samples the chain keeps
ATTEMPTS, SPARES = 8, 4
#: the joint chain: independent prior rounds, then search steps of SEARCH_WIDTH proposals; MAX_STEPS in all.
#: THE BUDGET (MAX_STEPS, CHAIN_FEEDS_MAX, ATTEMPTS) is setting B3 of stage 1's budget curve
#: (``measurements/sprime_honest_stage1_2026-11-05/``): the smallest measured setting that holds every U panel to
#: <= 10 % exclusions at <= 20 projected CPU-min a searched game (the coordinator's rule, 2026-11-05)
REJECT_ROUNDS = 2
SEARCH_WIDTH = 8
MAX_STEPS = 1800
STALL = 15
BURN, THIN = 6, 2
#: the share of search proposals that move the IMPLICATED mon (the rest move a uniformly chosen seen mon), and of
#: proposals that redraw a mon's unrevealed moves
P_IMPLICATED = 0.8
P_MOVES = 0.15
#: a chain that restarts stuck at the same segment may face dice no candidate meets on its common seeds (two crits,
#: a five-strike Beat Up): its tries are multiplied by ESCALATE, up to CHAIN_TRIES_MAX (its verdicts re-judged)
ESCALATE, CHAIN_TRIES_MAX = 4, 1024
#: a chain's whole budget in battle feeds (≈ 50 µs each): past it, no fit is an EXCLUSION (a segment whose dice are
#: rare jointly — a 10 % secondary, a crit and a 70 % miss of another secondary — can need far more; B3, above)
CHAIN_FEEDS_MAX = 4_000_000


def rng_of(key: str, *tag: object) -> np.random.Generator:
    t = ":".join([KEY_TAG, key] + [str(x) for x in tag])
    return np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:16], 16))


def dice_base(key: str, tag: str) -> str:
    """The base of a world's try seeds (``pokesim::fresh_dice::try_seed``)."""
    return f"{KEY_TAG}:{key}:{tag}"


# ------------------------------------------------------------------------------------------------ the scaffold
@dataclass
class SeenMon:
    """A seen opponent mon (D2): what its switch details and play revealed."""

    nick: str
    species: str
    sid: str
    level: str
    gender: str
    shiny: bool
    moves: List[str]                     # revealed move ids (a gen-3 Hidden Power named bare: ``hiddenpower``)
    item: Optional[str]
    ability: Optional[str]

    def base(self) -> List[str]:
        f = [""] * 12
        f[0] = self.nick
        f[1] = self.species if WD.toid(self.species) != WD.toid(self.nick) else ""
        f[2] = self.item or ""
        f[3] = self.ability or ""
        f[7] = self.gender
        f[9] = "S" if self.shiny else ""
        f[10] = "" if self.level == "100" else self.level
        return f


def _details(det: str) -> Tuple[str, str, str, bool]:
    parts = [x.strip() for x in det.split(",")]
    level, gender, shiny = "100", "", False
    for x in parts[1:]:
        if x.startswith("L") and x[1:].isdigit():
            level = x[1:]
        elif x in ("M", "F"):
            gender = x
        elif x == "shiny":
            shiny = True
    return parts[0], level, gender, shiny


@dataclass
class Scaffold:
    info: IS.InfoSet
    seen: List[SeenMon]
    size: int
    our_team: str
    segments: List[List[str]]
    header: str                          # the INFOSET line's JSON
    #: per segment, the opponent's active nickname at its start
    active: List[Optional[str]] = field(default_factory=list)

    @staticmethod
    def of(info: IS.InfoSet) -> "Scaffold":
        from agents.gen3_data import species as SP

        opp = info.opp
        rv = RV.from_lines(list(info.lines), info.viewer)
        seen: Dict[str, SeenMon] = {}
        segs = info.segments()
        active: List[Optional[str]] = []
        cur: Optional[str] = None
        for seg in segs:
            active.append(cur)
            for ln in seg:
                p = ln.split("|")
                if len(p) < 4 or p[1] not in ("switch", "drag", "replace") or not p[2].startswith(f"{opp}a: "):
                    continue
                nick = p[2].split(": ", 1)[1]
                cur = nick
                if nick in seen:
                    continue
                sp, level, gender, shiny = _details(p[3])
                sid = WD.toid(sp)
                d = SP.get(sid)
                if not gender:
                    gender = "N" if d is None or d.gender in (None, "N") else d.gender
                mr = rv.mons.get(sid)
                seen[nick] = SeenMon(nick=nick, species=sp, sid=sid, level=level, gender=gender, shiny=shiny,
                                     moves=sorted(mr.moves) if mr else [], item=mr.item if mr else None,
                                     ability=mr.ability if mr else None)
        size = IS.team_size(info, opp)
        if size is None:
            raise WorldError("the information set names no opponent team size (|teamsize|)")
        if len(seen) > size:
            raise WorldError(f"{len(seen)} opponent mons seen of a team of {size}")
        hdr = {"viewer": info.viewer, "segments": segs, "ours": list(info.cmds), "intents": IS.intents(info),
               "backtrack": IS.backtrack(info), "hp_last": IS.hp_last(info)}
        return Scaffold(info=info, seen=list(seen.values()), size=size, our_team=IS.own_genders(info),
                        segments=segs, header=json.dumps(hdr), active=active)

    def implicated(self, seg: int, k: int) -> Optional[int]:
        """The seen mon (its index) a failure at line ``k`` of segment ``seg`` implicates: the opponent's active mon
        there."""
        cur = self.active[seg] if seg < len(self.active) else None
        for ln in self.segments[seg][:k + 1] if seg < len(self.segments) else []:
            p = ln.split("|")
            if len(p) > 3 and p[1] in ("switch", "drag", "replace") and p[2].startswith(f"{self.info.opp}a: "):
                cur = p[2].split(": ", 1)[1]
        names = [m.nick for m in self.seen]
        return names.index(cur) if cur in names else None


# ------------------------------------------------------------------------------------------------ the matcher
class Matcher:
    """ONE persistent ``core_events`` process per decision (``pokesim::fresh_dice``): I is sent once (``INFOSET``),
    each world is one ``MATCH`` line and one answer. Closed (and reaped, so its CPU is counted) by :meth:`close`."""

    def __init__(self, header: str):
        from utils.bridge.sim_bridge_bin import resolve_core_events_bin

        self.p = subprocess.Popen([resolve_core_events_bin()], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, bufsize=1)
        self._send("INFOSET " + header)

    def _send(self, line: str) -> None:
        assert self.p.stdin is not None
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def __call__(self, jobs: Sequence[dict]) -> List[dict]:
        assert self.p.stdout is not None
        out = []
        for j in jobs:
            self._send("MATCH " + json.dumps(j))
            ans = self.p.stdout.readline()
            if not ans:
                self.close(check=False)
                raise RuntimeError(f"core_events MATCH died: {self._err[-2000:]}")
            out.append(json.loads(ans))
        return out

    _err = ""

    def close(self, check: bool = True) -> None:
        if self.p.returncode is None:
            _o, self._err = self.p.communicate()
        if check and self.p.returncode != 0:
            raise RuntimeError(f"core_events MATCH failed (exit {self.p.returncode}): {self._err.strip()[-2000:]}")

    def __enter__(self) -> "Matcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close(check=exc[0] is None)


def match(header: str, jobs: Sequence[dict]) -> List[dict]:
    """``pokesim::fresh_dice`` verdicts for ``jobs`` in a one-shot process (the leak test's builders)."""
    with Matcher(header) as m:
        return m(jobs)


def _with_happiness(f: List[str], moves: Sequence[str]) -> List[str]:
    if "frustration" in moves:
        misc = f[11].split(",") if f[11] else [""]
        misc[0] = "0"
        f[11] = ",".join(misc)
    return f


@dataclass
class HonestWorlds:
    """J worlds and the build's record (``worlds.Worlds``' fields the search reads, plus ``at``)."""

    logs: List[dict]
    teams: List[str]
    at: int
    tries: List[int]
    leak: dict
    revealed: Dict[str, List[str]]
    stats: List[dict]
    fails: Dict[str, int]
    chain: dict
    log_w: List[float] = field(default_factory=list)
    #: NOT in the decision record: the chain's samples (per seen mon, its (state, moves)) the worlds took, in world
    #: order, and each world's matcher report (its trajectory: ``seed``, ``trail``, ``cmds``). The incremental builder
    #: (``main.value_search.honest_inc``) carries them to the next decision
    samples: List[list] = field(default_factory=list)
    reports: List[dict] = field(default_factory=list)
    #: the incremental builder's record (``honest_inc``: the step, the carries' digests, the worlds' fates); empty for
    #: the cold builder, whose record is unchanged
    inc: dict = field(default_factory=dict)


class HonestSampler:
    """``HonestSampler().sample_info(info, J, root_row)`` -> :class:`HonestWorlds` (module docstring); raises
    :class:`WorldError` when I admits no world in its budget (an EXCLUSION)."""

    dice = "fresh"

    def __init__(self, prior: Optional[Prior] = None, legality: Optional[WL.Legality] = None, rolls: bool = True,
                 playout_check: bool = True, playout_lib=None, max_steps: int = MAX_STEPS,
                 chain_feeds: int = CHAIN_FEEDS_MAX, attempts: int = ATTEMPTS):
        self.P = prior or Prior()
        self.legality = legality if legality is not None else WL.Legality(pad=None)
        self.rolls = rolls
        #: THE BUDGET: chain steps, the chain's feeds, a world's attempts. A larger budget replays a smaller one's
        #: trajectory exactly and only goes on where the smaller one stopped (the builder is deterministic)
        self.max_steps = max_steps
        self.chain_feeds = chain_feeds
        self.attempts = attempts
        self.spec = SEARCH.validate("off")
        self._spreads: Dict[str, WS.SpreadPrior] = {}
        self.playout_check = playout_check
        self._playout_lib = playout_lib
        self._playout_core = None

    # -- pieces -------------------------------------------------------------------------------------------
    def _spread_prior(self, sid: str) -> WS.SpreadPrior:
        if sid not in self._spreads:
            self._spreads[sid] = WS.SpreadPrior(self.P.spreads(sid))
        return self._spreads[sid]

    def _sampler(self, key: str, i: int, m: SeenMon) -> WS.MonSampler:
        fixed = []
        if m.item:
            fixed.append(("item", m.item))
        if m.ability:
            fixed.append(("ability", m.ability))
        hp = "must" if any(is_hp(x) for x in m.moves) else "free"
        k = WS.ScreenKey(slot=i, hp=hp, fixed=tuple(sorted(fixed)))
        return WS.MonSampler(key=k, sid=m.sid, f_true=m.base(), items=self.P.items(m.sid),
                             abilities=self.P.ability(m.sid), spreads=self._spread_prior(m.sid),
                             hp_prior=self.P.hidden_power(m.sid),
                             hp_any=min(1.0, sum(v for mv, v in self.P.moves(m.sid).items() if is_hp(mv))),
                             rng=rng_of(key, "mon", i), needed=0)

    def _moves(self, rng, m: SeenMon, hpt: Optional[str]) -> Optional[List[str]]:
        """A seen mon's four slots: its revealed moves and a legality-free prior draw of the rest, uniformly permuted
        (D2). ``hpt``: the Hidden Power type its IVs carry (a set holding one gets one)."""
        keep = {i: mv for i, mv in enumerate(m.moves)}
        n = max(4, len(keep))
        got = WD.draw_moves(rng, self.P.moves(m.sid), keep, n, None, [], hpt,
                            hp_rule="force" if hpt is not None else "none")
        if got is None:
            return None
        mv = list(got[0])
        return [mv[k] for k in rng.permutation(len(mv))]

    def _entry(self, m: SeenMon, st: WS.State, moves: List[str]) -> str:
        """A seen mon's packed entry: its scaffold, the chain state's item, ability and spread (the IVs its Hidden Power
        type fixes, ``world_screen.ivs_of``), its moves with a bare revealed Hidden Power typed by those IVs."""
        item, ab, nat, evs, hpt = st
        f = m.base()
        f[2] = m.item or item
        f[3] = m.ability or ab
        ivs = WS.ivs_of(hpt)
        WD.Spread(nat[:1].upper() + nat[1:], tuple(evs), ivs).write(f)
        f[4] = ",".join(("hiddenpower" + WD.hp_type_of(ivs)) if x == "hiddenpower" else x for x in moves)
        return "|".join(_with_happiness(f, typed_moves(f)))

    def _never(self, rng, sc: Scaffold, legal: bool) -> Optional[List[str]]:
        """The never-seen mons (an honest prior draw, as today; every gender named)."""
        from agents.gen3_data import species as SP

        slots = list(range(len(sc.seen), sc.size))
        if not slots:
            return []
        got = WD.never_seen_species(rng, self.P, slots, {i: "" for i in slots}, {m.sid for m in sc.seen}, self.spec)
        if got is None:
            return None
        out = []
        for i in slots:
            tally: Dict[str, int] = {}
            ns = WD.new_set(rng, self.P, got[0][i], None, self.spec, legal=self.legality.set_ok if legal else None,
                            tally=tally)
            if ns is None:
                return None
            f = parse_set(ns[0])
            d = SP.get(got[0][i])
            g = d.gender if d is not None else "N"
            f[7] = ("M" if rng.random() < 0.5 else "F") if g is None else g
            out.append("|".join(_with_happiness(f, typed_moves(f))))
        return out

    def _job(self, sc: Scaffold, team_opp: str, tag: str, key: str, world: bool, tries: Optional[int] = None) -> dict:
        names = sc.info.names
        teams = [sc.our_team, team_opp] if sc.info.viewer == 0 else [team_opp, sc.our_team]
        quick = any(WD.item_id(parse_set(e)) == "quickclaw" for t in teams for e in t.split("]"))
        return {"label": tag, "formatid": sc.info.format_id, "p1": {"name": names[0], "team": teams[0]},
                "p2": {"name": names[1], "team": teams[1]}, "dice": dice_base(key, tag),
                "tries": tries or (WORLD_TRIES if world else CHAIN_TRIES),
                "rerolls": WORLD_REROLLS if world else CHAIN_REROLLS,
                "rolls": self.rolls, "prev_backtrack": quick}

    def _score(self, sc: Scaffold, r: dict) -> Tuple[bool, float, Optional[int], Optional[Tuple[str, int]]]:
        from main.value_search.worlds import divergence_hint, divergence_score

        if r["ok"]:
            return True, float(len(sc.segments)), None, None
        f = r.get("fail") or {}
        s = int(r["matched"])
        if s >= len(sc.segments):
            return False, float(s), None, None
        lines, ref = list(f.get("lines") or []), sc.segments[s]
        frac = divergence_score(lines, ref) / (len(ref) + 1.0)
        return False, s + frac, sc.implicated(s, int(f.get("lcp", 0))), divergence_hint(lines, ref, sc.info.opp)

    # -- the joint chain (D5) ------------------------------------------------------------------------------
    def _hpt(self, m: SeenMon, st: WS.State) -> Optional[str]:
        """The Hidden Power type a state's IVs put on an UNREVEALED slot (None: no unrevealed Hidden Power; a revealed
        one is typed at the entry)."""
        return None if any(is_hp(x) for x in m.moves) else WS.hp_split(st[4])[0]

    def _fit(self, rng, m: SeenMon, st: WS.State, mv: Optional[List[str]]) -> List[str]:
        """The mon's moves kept consistent with a (new) state's IVs: an unrevealed Hidden Power retyped, added (a redraw)
        or removed (a redraw) as the IVs say."""
        want = self._hpt(m, st)
        if mv is not None:
            unrev = [i for i, x in enumerate(mv) if is_hp(x) and x != "hiddenpower"]
            if want is None and not unrev:
                return mv
            if want is not None and unrev:
                out = list(mv)
                out[unrev[0]] = "hiddenpower" + want
                return out
        # the unrevealed moves drawn from the prior CONDITIONED ON LEGALITY (F2's rule: Showdown's validator on the set
        # with this state), redrawn up to SET_REDRAWS times; the last draw stands (the world's own check rejects it)
        got = None
        for _r in range(WD.SET_REDRAWS):
            got = self._moves(rng, m, want)
            if got is None:
                break
            if self.legality.set_ok(self._entry(m, st, got)):
                return got
        if got is None:
            raise WorldError(f"no prior move draw for {m.sid}")
        return got

    def _chain(self, sc: Scaffold, key: str, need: int, rec: dict, match, init: Optional[list] = None,
               dice_key: Optional[str] = None, tries0: int = CHAIN_TRIES) -> List[List[Tuple[WS.State, List[str]]]]:
        """The joint chain (D5). A seen mon's coordinate is (item, ability, spread) AND its unrevealed moves (in slot
        order): a spread can be consistent with I only through a move it holds (a fast mon that chose a
        negative-priority move it never showed), so the two are sampled jointly. The never-seen mons are one keyed
        background draw (they reach I only through a die over the party).

        ``init`` (the incremental builder's WARM start: the previous decision's chain state, one coordinate a seen mon):
        no prior rounds; the chain starts there, sampling at once when it fits I, else searching from it. None: the
        cold chain, unchanged. ``dice_key``: the key of the chain's common dice (default ``key``; the incremental
        builder keeps ONE per battle, so a state that fit the earlier I replays its kept tries there); ``tries0`` its
        tries to start from. A chain that finds no fit raises a :class:`WorldError` carrying its best state
        (``.best``)."""
        rng = rng_of(key, "chain")
        ms = [self._sampler(key, i, m) for i, m in enumerate(sc.seen)]
        bg_rng = rng_of(key, "chain_background")
        bg_never = self._never(bg_rng, sc, legal=False)
        if bg_never is None:
            raise WorldError(f"{key}: no never-seen draw for the chain background")
        Coord = Tuple[WS.State, List[str]]

        def team(states: Sequence[Coord]) -> str:
            ents = [self._entry(m, states[i][0], states[i][1]) for i, m in enumerate(sc.seen)]
            return "]".join(ents + bg_never)

        def fresh(i: int) -> Coord:
            st = ms[i].draw()
            return st, self._fit(rng, sc.seen[i], st, None)

        def moved(i: int, c: Coord, st: WS.State) -> Coord:
            return st, self._fit(rng, sc.seen[i], st, c[1])

        verdicts: Dict[str, Tuple[bool, float, Optional[int], Optional[Tuple[str, int]]]] = {}
        tries = [tries0]
        dkey = key if dice_key is None else dice_key

        def judge(cands: List[List[Coord]]) -> List[Tuple[bool, float, Optional[int], Optional[Tuple[str, int]]]]:
            teams = [team(c) for c in cands]
            # a set Showdown's validator rejects (an item, ability or spread change can break a legal move set) is a
            # failed candidate, never replayed: the chain samples the prior conditioned on legality
            for t, c in zip(teams, cands):
                if t not in verdicts and not all(self.legality.set_ok(self._entry(m, c[i][0], c[i][1]))
                                                 for i, m in enumerate(sc.seen)):
                    verdicts[t] = (False, -1.0, None, None)
                    rec["illegal"] = rec.get("illegal", 0) + 1
            todo = sorted({t for t in teams if t not in verdicts})
            for t, r in zip(todo, match([self._job(sc, t, "chain", dkey, False, tries[0]) for t in todo])):
                verdicts[t] = self._score(sc, r)
                rec["feeds"] += int(r.get("feeds", 0))
            rec["evaluations"] += len(todo)
            return [verdicts[t] for t in teams]

        def logp(c: Sequence[Coord]) -> float:
            return float(sum(ms[i].logp(c[i][0]) for i in range(len(ms))))

        if not ms:                                  # no seen opponent mon: nothing hidden shapes I
            return [[] for _ in range(need)]
        samples: List[List[Coord]] = []
        best: Optional[Tuple[float, List[Coord], Optional[int], Optional[Tuple[str, int]]]] = None
        warm = None
        if init is not None:
            w_ok, w_sc, w_imp, w_hint = judge([list(init)])[0]
            rec["warm"] = "fit" if w_ok else "refit"
            warm = (list(init), w_ok, float(len(sc.segments)) if w_ok else w_sc, w_imp, w_hint)
        else:
            for _ in range(REJECT_ROUNDS):
                cands = [[fresh(i) for i in range(len(ms))] for _ in range(need - len(samples))]
                for c, (ok, sc_, imp, hint) in zip(cands, judge(cands)):
                    if ok and len(samples) < need:
                        samples.append(c)
                    elif not ok and (best is None or sc_ > best[0]):
                        best = (sc_, c, imp, hint)
                if len(samples) >= need:
                    rec["phase"] = "rejection"
                    return samples
        # the chain: hill-climb on the matched length until I fits, then Metropolis-Hastings, thinned
        if warm is not None:
            cur, cur_ok, cur_score, imp, hint = warm
        elif samples:
            cur, cur_ok, cur_score, imp, hint = samples[-1], True, float(len(sc.segments)), None, None
        else:
            assert best is not None
            cur, cur_ok, cur_score, imp, hint = best[1], False, best[0], best[2], best[3]
        stall, burn, since = 0, (0 if cur_ok else BURN), 0
        best_ever = (cur_score, cur)
        best_hint = hint
        for step in range(self.max_steps):
            rec["steps"] = step + 1
            if rec["feeds"] > self.chain_feeds:
                rec["phase"] = "budget"
                break
            if cur_ok:
                i = int(rng.integers(len(ms)))
                c = list(cur)
                u = rng.random()
                if u < P_MOVES:                        # the unrevealed moves, redrawn (an independence move: ratio 1)
                    c[i], sym = (cur[i][0], self._fit(rng, sc.seen[i], cur[i][0], None)), False
                elif u < 0.25 + P_MOVES:
                    c[i], sym = fresh(i), False
                else:
                    st, sym = ms[i].local(cur[i][0])
                    c[i] = moved(i, cur[i], st)
                if logp(c) > -math.inf:
                    ok, _s, _i, _h = judge([c])[0]
                    if ok and (not sym or math.log(max(rng.random(), 1e-300)) < logp(c) - logp(cur)):
                        cur = c
                if burn > 0:
                    burn -= 1
                else:
                    since += 1
                    if since >= THIN:
                        since = 0
                        samples.append(list(cur))
                        if len(samples) >= need:
                            rec["phase"] = "chain"
                            return samples
                continue
            cands = []
            for _ in range(SEARCH_WIDTH):
                i = imp if (imp is not None and rng.random() < P_IMPLICATED) else int(rng.integers(len(ms)))
                c = list(cur)
                u = rng.random()
                if u < P_MOVES:
                    c[i] = (cur[i][0], self._fit(rng, sc.seen[i], cur[i][0], None))
                elif hint is not None and u < 0.5:
                    c[i] = moved(i, cur[i], ms[i].guided(cur[i][0], hint))
                elif u < 0.75:
                    c[i] = fresh(i)
                else:
                    c[i] = moved(i, cur[i], ms[i].local(cur[i][0])[0])
                cands.append(c)
            live = [c for c in cands if logp(c) > -math.inf]
            got = judge(live) if live else []
            top = max(zip(got, live), key=lambda x: (x[0][0], x[0][1]), default=None)
            if top is not None and (top[0][0] or top[0][1] > cur_score):
                (ok, sc_, imp2, hint2), c = top
                cur, cur_ok, cur_score, imp, hint, stall = c, ok, sc_, imp2, hint2, 0
                if cur_ok:
                    burn, since = BURN, 0
                    samples.append(list(cur))
                    if len(samples) >= need:
                        rec["phase"] = "search"
                        return samples
                if cur_score > best_ever[0]:
                    best_ever, best_hint = (cur_score, cur), hint
            else:
                stall += 1
            if not cur_ok and stall >= STALL:
                rec["restarts"] = rec.get("restarts", 0) + 1
                stall = 0
                seg_now = int(best_ever[0])
                # an HP NUMBER that differs (a ``bulk`` / ``power`` hint) is a spread or a hidden-HP fact the chain and
                # the matcher's re-rolls move; only another line (a crit, a miss, an order) may want rarer dice
                dice_bound = best_hint is None or best_hint[0] == "speed"
                if seg_now == rec.get("stuck_at") and dice_bound and tries[0] < CHAIN_TRIES_MAX:
                    tries[0] = min(CHAIN_TRIES_MAX, tries[0] * ESCALATE)    # rarer dice: re-judge on more tries
                    rec["chain_tries"] = tries[0]
                    verdicts.clear()
                rec["stuck_at"] = seg_now
                if rec["restarts"] % 2 == 1:           # from the best state, shaken
                    cur = [moved(i, best_ever[1][i], ms[i].local(best_ever[1][i][0])[0]) for i in range(len(ms))]
                else:                                  # from a fresh prior draw
                    cur = [fresh(i) for i in range(len(ms))]
                cur_score, imp, hint = -1.0, None, None
        rec.setdefault("phase", "exhausted")
        if rec["phase"] not in ("budget",):
            rec["phase"] = "exhausted"
        if samples:
            return samples
        err = WorldError(f"{key}: no honest world fits I in {rec['steps']} chain steps ({rec['phase']}; best: "
                         f"segment {best_ever[0]:.2f} of {len(sc.segments)})")
        err.best = list(best_ever[1])                # type: ignore[attr-defined]
        raise err

    # -- the strict gate ----------------------------------------------------------------------------------
    def _strict(self, sc: Scaffold, lg: dict, root_row: Optional[str], key: str) -> None:
        """THE STRICT GATE: the world replayed on its schedule through every ``core_events`` check reproduces I's
        lines (and, given, the live root row). A difference is a DEFECT of the matcher (raised)."""
        from main.value_search.info_set import extract

        try:
            got, row = extract(lg, len(lg["cmds"]), sc.info.side)
        except ValueError as e:
            raise RuntimeError(f"{key}: a matched world does not replay strictly: {e}") from e
        if got.lines != sc.info.lines or got.cmds != sc.info.cmds:
            raise RuntimeError(f"{key}: a matched world's strict replay ships other lines than I")
        if root_row is not None and row != root_row:
            raise RuntimeError(f"{key}: a matched world's root row is not the live row")

    def _playout_refusal(self, lg: dict, side: str) -> Optional[str]:
        if not self.playout_check:
            return None
        if self._playout_core is None:
            from utils.rust_env.successors import SearchCore
            from main.value_search.playout import release_lib

            self._playout_core = SearchCore(lib=self._playout_lib or release_lib())
        req = {"log": lg, "at": len(lg["cmds"]), "side": side, "actions": None, "seeds": [None], "stall": None,
               "max_turns": 999, "keep_cmds": False}
        try:
            self._playout_core._cstr(self._playout_core.lib.rust_env_playout_open, json.dumps(req).encode())
        except Exception as e:                       # the tool's typed refusal: the world is rejected, named
            return f"{type(e).__name__}: {str(e)[:300]}"
        return None

    # -- the build --------------------------------------------------------------------------------------------
    def sample_info(self, info: IS.InfoSet, J: int, root_row: Optional[str] = None) -> HonestWorlds:
        key = info.key()
        sc = Scaffold.of(info)
        rec: dict = {"seen": len(sc.seen), "size": sc.size, "segments": len(sc.segments), "evaluations": 0,
                     "feeds": 0, "steps": 0}
        with Matcher(sc.header) as m:
            return self._build(info, sc, key, J, root_row, rec, m)

    def _one_world(self, info: IS.InfoSet, sc: Scaffold, key: str, j: int, st: list, root_row: Optional[str],
                   rec: dict, fails: Dict[str, int], match, strict: bool = True) -> Optional[tuple]:
        """World ``j`` on chain sample ``st`` (its seen mons whole): up to ``attempts`` draws of its never-seen mons,
        each replayed on the world's OWN fresh dice (tag ``w<j>:<attempt>``). ``(log, team, attempts, stats, log_w,
        report)``, or None (each failure counted in ``fails``). ``strict``: the strict gate here (the incremental
        builder runs it once over a decision's worlds)."""
        from main.value_search.worlds import team_problems

        ents = [self._entry(m, st[i][0], st[i][1]) for i, m in enumerate(sc.seen)]
        if not all(self.legality.set_ok(e) for e in ents):
            fails["illegal_set"] = fails.get("illegal_set", 0) + 1
            return None
        for a in range(self.attempts):
            rng = rng_of(key, "world", j, a)
            never = self._never(rng, sc, legal=True)
            if never is None:
                fails["never_seen"] = fails.get("never_seen", 0) + 1
                continue
            tw = "]".join(ents + never)
            if team_problems(tw) or self.legality.team_problems([tw])[0]:
                fails["illegal_team"] = fails.get("illegal_team", 0) + 1
                continue
            # a world faces the dice its chain faced (the chain's escalated tries, at least WORLD_TRIES); a
            # later attempt allows rarer dice (two exact Speed ties and a crit in one segment: ≈ 1 in 128)
            base_t = max(WORLD_TRIES, int(rec.get("chain_tries", CHAIN_TRIES)))
            r = match([self._job(sc, tw, f"w{j}:{a}", key, True,
                                 base_t if a == 0 else max(base_t, CHAIN_TRIES_MAX))])[0]
            rec["feeds"] += int(r.get("feeds", 0))
            if not r["ok"]:
                fails["world_replay"] = fails.get("world_replay", 0) + 1
                continue
            lg = self._world_log(info, sc, tw, r)
            if strict:
                self._strict(sc, lg, root_row, key)
            why = self._playout_refusal(lg, info.side)
            if why is not None:
                fails["playout_refused"] = fails.get("playout_refused", 0) + 1
                continue
            return (lg, tw, a + 1, {"attempt": a, "rerolls": r["rerolls"], "feeds": r["feeds"],
                                    "rolls": len(r["rolls"])}, r.get("log_w"), r)
        return None

    def _world_log(self, info: IS.InfoSet, sc: Scaffold, team_opp: str, r: dict) -> dict:
        """A matched world's log: both teams, its START seed, commands, reseed schedule and damage-roll table."""
        our = sc.our_team
        lt = [our, team_opp] if info.viewer == 0 else [team_opp, our]
        lg = {"format_id": info.format_id, "seed": r["seed"], "names": list(info.names), "teams": lt,
              "cmds": list(r["cmds"]), "reseed": [list(x) for x in r["reseed"]]}
        if r["rolls"]:
            lg["rolls"] = [list(x) for x in r["rolls"]]
        return lg

    def _build(self, info: IS.InfoSet, sc: Scaffold, key: str, J: int, root_row: Optional[str], rec: dict,
               match) -> HonestWorlds:
        samples = self._chain(sc, key, J + SPARES, rec, match)
        fails: Dict[str, int] = {}
        logs: List[dict] = []
        teams: List[str] = []
        tries: List[int] = []
        stats: List[dict] = []
        log_w: List[float] = []
        took: List[list] = []
        reports: List[dict] = []
        pool = list(range(len(samples)))
        for j in range(J):
            built = None
            while built is None and pool:
                st = samples[pool.pop(0)]            # world j takes the next chain sample (a spare after a failure)
                built = self._one_world(info, sc, key, j, st, root_row, rec, fails, match)
                if built is not None:
                    took.append(st)
            if built is None:
                err = WorldError(f"{key}: {J - len(logs)} of {J} worlds not built ({fails})")
                err.best = list(samples[-1])         # type: ignore[attr-defined]
                raise err
            lg, tw, ntry, stt, lw, r = built
            reports.append(r)
            logs.append(lg)
            teams.append(tw)
            tries.append(ntry)
            stats.append(stt)
            log_w.append(float(lw) if lw is not None else float("nan"))
        ats = {len(lg["cmds"]) for lg in logs}
        if len(ats) != 1:
            raise RuntimeError(f"{key}: the worlds' roots sit at different commands {sorted(ats)}")
        leak = {"builder": "honest", "key": key, "residual": [], "partial": []}
        return HonestWorlds(logs=logs, teams=teams, at=ats.pop(), tries=tries, leak=leak,
                            revealed={m.sid: list(m.moves) for m in sc.seen}, stats=stats, fails=fails, chain=rec,
                            log_w=log_w, samples=took, reports=reports)
