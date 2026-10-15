"""THE WORLD SAMPLER: determinized worlds for a decision of a core input log, HONEST BY DEFAULT, with the extra
information an INFORMATION-LEVER spec grants (``agents.info_levers``; ``designs/endstate/design_information_levers.md``
§7). Offline / evaluation use only.

A world is the battle's input log with the OPPONENT's hidden team replaced. Each hidden fact is the truth's if play
revealed it (``agents.info_levers.revealed``: the ONE log source), the lever's target if a lever fires for it, else a
draw from the Smogon prior (``world_draws``). The world is kept only if, replayed through the core (``core_events
--trackers --obs``) to the root, the searched side's observation row at EVERY one of its decisions AND every protocol
line it was sent are BYTE-EQUAL to the truth's (the hidden-info study's audit: the rows alone missed gender tokens, a
Focus Punch tightening line and a Beat Up hit, `design_hidden_info_binding.md` §16.4).

HONEST BY DEFAULT (``reveal="off"``): every unrevealed fact is drawn, a SEEN mon's item, ability and spread (nature,
EVs, IVs: its exact stats, its Hidden Power type and the exact HP its history leaves) included. Those shaped the
history the searched side saw, so they are SCREENED one mon at a time: each proposal is replayed with every other
opponent mon at its true facts and kept only if the gate passes; the accepted per-mon facts are then combined and the
whole world passes the gate again (§7.2: exact when each observable event depends on one opponent mon's hidden
facts; an event coupling two is caught by the joint gate). A seen mon's unrevealed moves and every never-seen mon are
drawn directly (they never acted). The value-discrimination pilot's builder (v1 / v2: a seen mon's item, ability and
spread kept TRUE) is the ``pilot`` preset.

THE RESIDUAL (named, reported in ``Worlds.leak["residual"]``): the world replays the prefix on the TRUE battle seed, so
a hidden duration rolled before the root (sleep, confusion, Encore, a lock) is the truth's in every world (``counters``).
The PARTIAL leaks (``Worlds.leak["partial"]``, design §7.5, measured): for the same reason a hidden fact that spends a
die without printing a line — a speed tie, King's Rock, Focus Band, a contact ability's roll — must spend the truth's
dice. Every world is also OPENED by the playout tool before it is returned (the gate's ``core_events`` replay accepts
some choice logs the playout tool refuses).

The template: a past opponent move choice is attributed to the mon that made it (the opponent's ACTIVE mon at that
command, from its own request); when the searched side never saw that mon use it, it is rewritten to its SLOT
(rendered as the world's move there), and a past switch to a never-seen mon to the world's mon in that slot; the
opponent's ROOT-turn choice stays when the log already holds it.

GENDERS: the engine draws a gender at team construction for every mon of a gender-ratio species whose packed set
names none (one uniform M / F sample, in team order), so the pre-root dice depend on which never-seen slots draw. The
world's DRAWN never-seen mons (an exchangeable sample) are placed so the draws happen at exactly the true team's draw
slots (`place_genders`); a mon pinned to its slot by a lever keeps it. Without the engine's gender draw pad a world
needs at least as many ratio'd never-seen mons as the true draws (else rejected: ``gender_count``, the residual).

LEGALITY: a world team breaking a gen3ou rule (`agents.gen3_data.team_legality`) that the true team does not break is
rejected (``clause``). Only moves and items the engine runs are drawn (``BANNED_MOVES`` / ``BANNED_ITEMS``).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from agents.info_levers import revealed as RV
from agents.info_levers.consumers import SEARCH
from agents.info_levers.spec import InfoReveal
from main.value_search.world_draws import (  # noqa: F401  (re-exported: the sampler's public helpers)
    BANNED_ITEMS, BANNED_MOVES, HP_TYPES, STATS, Prior, Spread, WorldError, fires, hp_ivs, hp_type_of, is_hp,
    never_seen_species, new_set, parse_set, set_ivs, toid, typed_moves)
from main.value_search import world_draws as WD
from main.value_search import world_guard as WG
from main.value_search import world_screen as WS

MAX_TRIES = 30
#: move draws per seen mon before a world's try fails on a set the format bans
MOVE_REDRAWS = 10


def log_commands(log: dict, upto: Optional[int] = None) -> List[List[str]]:
    """A core input log's ``cmds`` as ``[side, token]`` pairs (``FORCELOSE pN`` → ``[forcelose, pN]``)."""
    out = []
    for c in log["cmds"][:upto]:
        if c.startswith("FORCELOSE "):
            out.append(["forcelose", c.split(" ", 1)[1]])
        else:
            _, side, tok = c.split(" ", 2)
            out.append([side, tok])
    return out


def sample_moves(rng, P, sid: str, keep: Dict[int, str], hp_type: Optional[str]) -> Optional[List[str]]:
    """The four move slots: ``keep`` stays; the rest an HONEST draw from the species' move usage (the CP presence
    design, ``world_draws.draw_moves`` with no lever); at most one Hidden Power, and only of ``hp_type`` when given."""
    got = WD.draw_moves(rng, P.moves(sid), keep, 4, None, [], hp_type)
    return None if got is None else got[0]


def reveals(res: dict, viewer: int, n: int) -> Tuple[Dict[str, Set[str]], List[str]]:
    """{opponent species id seen: its own moves play named} up to the viewer's n-th decision (inclusive; the ONE log
    source, ``agents.info_levers.revealed``), and the viewer's obs frames (base64) at decisions 0..n."""
    ent = [e for e in res["trackers"][viewer] if "obs" in e]
    if len(ent) <= n:
        raise WorldError(f"the viewer has {len(ent)} decisions, the root is decision {n}")
    r = RV.from_lines(RV.viewer_lines(res, viewer, int(ent[n]["after"])), viewer)
    return r.moves(), [e["obs"]["b64"] for e in ent[: n + 1]]


def protocol_lines(res: dict, viewer: int, n: int) -> List[str]:
    """Every protocol line the viewer was sent up to its n-th decision (inclusive), in order."""
    ent = [e for e in res["trackers"][viewer] if "obs" in e]
    return RV.viewer_lines(res, viewer, int(ent[n]["after"]))


def opp_actives(res: dict, side_idx: int, n_cmds: int) -> List[Optional[str]]:
    """The species id of side ``side_idx``'s ACTIVE mon at each of its first ``n_cmds`` commands (the k-th
    command answers the side's k-th request asking for a choice), from that side's own request lines."""
    out: List[Optional[str]] = []
    for rq in RV.own_team_request(res, side_idx):
        if rq.get("wait") or rq.get("teamPreview"):
            continue
        act = [p for p in rq.get("side", {}).get("pokemon", []) if p.get("active")]
        out.append(toid(act[0]["details"].split(",")[0]) if act else None)
    if len(out) < n_cmds:
        raise WorldError(f"side {side_idx}: {len(out)} choice requests for {n_cmds} commands")
    return out[:n_cmds]


def opp_conditions(res: dict, side_idx: int, upto_chunk: int) -> Dict[str, str]:
    """{species id: its exact ``condition`` (``hp/maxhp status``)} from the LAST request side ``side_idx`` was sent at
    or before chunk ``upto_chunk``: the opponent's own exact HP (the ``hp`` lever's target)."""
    last: Optional[dict] = None
    for i, (side, lines) in enumerate(res["chunks"][: upto_chunk + 1]):
        if side != side_idx:
            continue
        for ln in lines:
            if ln.startswith("|request|") and ln[len("|request|"):].strip():
                rq = json.loads(ln[len("|request|"):])
                if rq.get("side"):
                    last = rq
    if last is None:
        return {}
    return {toid(p["details"].split(",")[0]): str(p.get("condition", "")) for p in last["side"]["pokemon"]}


def _hp_number(line: str) -> Optional[Tuple[str, int]]:
    """(the line's kind + subject, the HP numerator it shows) for an HP-bearing line, else None."""
    p = line.split("|")
    if len(p) < 4 or p[1] not in ("-damage", "-heal", "-sethp", "switch", "drag"):
        return None
    tok = p[4] if p[1] in ("switch", "drag") and len(p) > 4 else p[3]
    num = tok.split("/", 1)[0].split(" ", 1)[0]
    return (p[1] + "|" + p[2], int(num)) if num.isdigit() else None


def divergence_score(proto: List[str], ref: List[str]) -> float:
    """How far ``proto`` matches ``ref``: the number of leading equal lines, plus, when the first differing lines are
    the same HP line of the same mon, a closeness 1 / (1 + |Δ HP shown|) in [0, 1) (the screen's search climbs it)."""
    k = next((i for i, (a, b) in enumerate(zip(proto, ref)) if a != b), min(len(proto), len(ref)))
    if k < len(proto) and k < len(ref):
        a, b = _hp_number(proto[k]), _hp_number(ref[k])
        if a is not None and b is not None and a[0] == b[0]:
            return k + 1.0 / (2.0 + abs(a[1] - b[1]))
    return float(k)


def divergence_hint(proto: List[str], ref: List[str], opp: str) -> Optional[Tuple[str, int]]:
    """Which stat the first differing line implicates, for the screen's directed search: an HP line of the
    OPPONENT's mon after a hit → its BULK (sign +1: the world's mon shows more HP left than the truth's, so it is too
    bulky); one of OUR mons → the opponent's POWER (sign +1: ours shows more HP left, so the opponent hits too
    softly); two lines of DIFFERENT sides (a move, an end-of-turn residual: gen 3 orders both by Speed) → SPEED
    (sign +1: the world's opponent acted first where the truth's did not, so it is too fast); anything else → SPEED,
    unsigned. ``opp``: the opponent's side tag (``p2``)."""
    k = next((i for i, (a, b) in enumerate(zip(proto, ref)) if a != b), None)
    if k is None:
        return None
    a, b = _hp_number(proto[k]), _hp_number(ref[k])
    if a is not None and b is not None and a[0] == b[0]:
        sign = 1 if a[1] > b[1] else -1
        side = a[0].split("|", 1)[1][:2]
        return ("bulk", sign) if side == opp else ("power", sign)
    pa, pb = proto[k].split("|"), ref[k].split("|")
    sa = pa[2][:2] if len(pa) > 2 and pa[2][:2] in ("p1", "p2") else None
    sb = pb[2][:2] if len(pb) > 2 and pb[2][:2] in ("p1", "p2") else None
    if sa is not None and sb is not None and sa != sb:      # two sides in another order: moves, end-of-turn residuals
        return ("speed", 1 if sa == opp else -1)
    return ("speed", 0)


def species_draws_gender(sid: str) -> bool:
    """The engine draws a gender for a mon iff its packed set names none AND its species has no fixed gender."""
    from agents.gen3_data import species

    sd = species.get(sid)
    return sd is not None and sd.gender is None


def _draws(entry: str) -> bool:
    """Does this packed entry spend a turn-0 gender draw? (no gender named on a ratio'd species, or the engine's pad
    marker on a named one: ``utils.team_packing.GENDER_DRAW_PAD``)"""
    f = parse_set(entry)
    pad = _gender_pad()
    if pad and f[7].endswith(pad):
        return True
    return not f[7] and species_draws_gender(toid(f[1] or f[0]))


def gender_draw_slots(team: str) -> Set[int]:
    return {i for i, e in enumerate(team.split("]")) if _draws(e)}


def _gender_pad() -> Optional[str]:
    """The engine's gender-draw PAD marker (a named gender that still consumes the turn-0 draw), if this build has it."""
    try:
        from utils import team_packing

        return getattr(team_packing, "GENDER_DRAW_PAD", None)
    except ImportError:
        return None


def place_genders(rng, entries: List[str], never_seen: List[int], draw_slots: Set[int],
                  pinned: Set[int] = frozenset(), pad: Optional[str] = None,
                  src_of: Optional[Dict[int, int]] = None) -> Optional[List[str]]:
    """The world's DRAWN never-seen mons (an exchangeable sample: ``never_seen`` minus ``pinned``) permuted among their
    slots so the true team's draw slots hold mons that draw (a gender-ratio mon with no gender named: the value is the
    true slot's fair coin) and every other never-seen mon names a gender from the engine's prior (uniform M / F, or its
    fixed gender), so it draws nothing. A PINNED slot keeps its mon (a lever put it there). With the engine's ``pad``
    marker a mon that cannot draw sits on a draw slot by naming its gender and padding the draw; without it, None when
    the world cannot match the true draws (fewer ratio'd mons than draws: the residual ``gender_count``).

    ``src_of`` (a dict the caller passes): filled with {never-seen slot: the slot whose mon now sits there}, the
    PERMUTATION, so a caller that keeps per-slot records moves them with their mons (the 2026-10-15 GIGO fix: the
    builder's ``facts`` stayed at the slot each mon was drawn for). It changes no draw and no output."""
    from agents.gen3_data import species

    ns = sorted(never_seen)
    free = [i for i in ns if i not in pinned]
    sets = {i: parse_set(entries[i]) for i in ns}

    def named(f: List[str]) -> List[str]:
        sd = species.get(toid(f[1] or f[0]))
        g = sd.gender if sd is not None else "N"
        f = list(f)
        f[7] = ("M" if rng.random() < 0.5 else "F") if g is None else ("" if g == "N" else g)
        return f

    def padded(f: List[str]) -> Optional[List[str]]:
        if pad is None:
            return None
        sd = species.get(toid(f[1] or f[0]))
        g = (f[7] or "")[:1] or (sd.gender if sd is not None and sd.gender is not None else "M")
        f = list(f)
        f[7] = ("N" if g in ("", "N") else g) + pad
        return f

    out = list(entries)
    need = sorted(s for s in draw_slots if s in free)
    ratio = [i for i in free if species_draws_gender(toid(sets[i][1] or sets[i][0]))]
    if len(ratio) < len(need) and pad is None:
        return None
    pick = [ratio[k] for k in rng.permutation(len(ratio))[: len(need)]]
    rest = [i for i in free if i not in pick]
    rest = [rest[k] for k in rng.permutation(len(rest))]
    srcs = pick + rest[: max(0, len(need) - len(pick))]
    rest = rest[max(0, len(need) - len(pick)):]
    moved: Dict[int, int] = {slot: slot for slot in ns if slot in pinned}
    for slot, src in zip(need, srcs):
        moved[slot] = src
        f = list(sets[src])
        if species_draws_gender(toid(f[1] or f[0])):
            f[7] = ""
        else:
            g = padded(f)
            if g is None:
                return None
            f = g
        out[slot] = "|".join(f)
    for slot, src in zip([s for s in free if s not in need], rest):
        moved[slot] = src
        out[slot] = "|".join(named(sets[src]))
    for slot in sorted(pinned & set(ns)):
        f = list(sets[slot])
        draws = _draws("|".join(f))
        if slot in draw_slots and not draws:
            if species_draws_gender(toid(f[1] or f[0])):
                f[7] = ""                            # a ratio'd mon with a named gender: let it draw
                out[slot] = "|".join(f)
                continue
            g = padded(f)
            if g is None:
                return None
            out[slot] = "|".join(g)
        elif slot not in draw_slots and draws:
            out[slot] = "|".join(named(f))
    if sorted(moved) != ns or sorted(moved.values()) != ns:
        raise RuntimeError(f"place_genders: not a permutation of the never-seen slots {ns}: {moved}")
    if src_of is not None:
        src_of.clear()
        src_of.update(moved)
    return out


def mon_problems(entry: str) -> Set[str]:
    """The gen3ou rules ONE packed set breaks on its own (a per-set clause or ban)."""
    return {r for r, _ in team_problems(entry)}


def team_problems(team: str) -> Set[Tuple[str, int]]:
    """(rule, slot) of every gen3ou format-spec violation of a packed team (bans and clauses)."""
    from agents.gen3_data import team_legality as tl

    pad = _gender_pad()
    if pad:
        team = "]".join("|".join((g.replace(pad, "") if i == 7 else g) for i, g in enumerate(e.split("|")))
                        for e in team.split("]"))
    return {(p.rule, int(p.slot)) for p in tl.validate_team(tl.parse_packed(team))}


@dataclass
class Worlds:
    """J gate-passed worlds: each a full core input log (same ``at``) and its opponent team."""

    logs: List[dict]
    teams: List[str]
    tries: List[int]
    leak: dict
    revealed: Dict[str, List[str]]
    stats: List[dict] = field(default_factory=list)
    fails: Dict[str, int] = field(default_factory=dict)   # rejections: gender_count, clause, sample, screen, joint
    #: per world, per opponent slot OF THE RETURNED TEAM: each fact's provenance (a never-seen mon's record moves with
    #: it through the gender placement, ``drawn_slot`` naming where it was drawn; checked by ``world_guard``)
    facts: List[List[dict]] = field(default_factory=list)
    spec: dict = field(default_factory=dict)              # the information-lever spec's record block
    screens: int = 0                                       # screening replays run
    #: per seen-mon sampler: its slot, species, draws needed / made, chain steps, failure, proposal honesty
    samplers: List[dict] = field(default_factory=list)


def key_rng(key: str, j: int, attempt: int = 0):
    tag = f"value_search_world_v1:{key}:{j}" + (f":{attempt}" if attempt else "")
    return np.random.default_rng(int(hashlib.sha256(tag.encode()).hexdigest()[:16], 16))


@dataclass
class Root:
    """The TRUTH at a decision, everything a world is built and gated against (``prepare_root``)."""

    log: dict
    at: int
    side: str
    viewer: int
    oi: int
    n: int
    cmds: List[List[str]]
    true: dict
    rv: "RV.Reveals"
    ref: List[str]
    ref_proto: List[str]
    ref_cond: Dict[str, str]
    team: str
    entries: List[str]
    sets: List[List[str]]
    sids: List[str]
    moves_of: List[List[str]]
    kept: Dict[str, Set[str]]
    tmpl: List[list]
    leak: dict

    def rec(self, team_opp: Optional[str], label: str, commands):
        from agents.battle.core_replay import RecordedBattle

        teams = list(self.log["teams"])
        if team_opp is not None:
            teams[self.oi] = team_opp
        return RecordedBattle(label=label, format_id=self.log["format_id"], seed=self.log["seed"],
                              p1={"name": self.log["names"][0], "team": teams[0]},
                              p2={"name": self.log["names"][1], "team": teams[1]}, commands=commands)


def render(tmpl: List[list], team_w: str) -> List[list]:
    """The prefix with every SLOT-tagged choice rendered against a world's team: a switch to the world's mon in that
    slot, a move as the world's move ID at that slot of that mon (the playout core resolves a replayed choice against
    the offered tokens BY NAME: a numeric ``move k`` is refused there)."""
    sets_w = [parse_set(e) for e in team_w.split("]")]

    def one(t):
        if not isinstance(t, tuple):
            return t
        if t[0] == "SLOT":
            return f"switch {sets_w[t[1]][0]}"
        return f"move {typed_moves(sets_w[t[1]])[t[2]]}"      # ("MOVE", mon, slot)
    return [[s_, one(t)] for s_, t in tmpl]


def prepare_root(log: dict, at: int, side: str, key: str, root_choice_in_log: bool = False) -> Root:
    """Replay the TRUE prefix and derive what the searched side saw (the ONE log source), its rows, protocol and the
    opponent's exact conditions, and the choice TEMPLATE (module docstring)."""
    from agents.battle.core_replay import run_core

    viewer = 0 if side == "p1" else 1
    oi = 1 - viewer
    opp_key = "p2" if side == "p1" else "p1"
    cmds = log_commands(log, at)
    n = sum(1 for s, _ in cmds if s == side)
    root = Root(log=log, at=at, side=side, viewer=viewer, oi=oi, n=n, cmds=cmds, true={}, rv=RV.Reveals(), ref=[],
                ref_proto=[], ref_cond={}, team="", entries=[], sets=[], sids=[], moves_of=[], kept={}, tmpl=[],
                leak={})
    true = run_core([root.rec(None, f"{key}#true", cmds)], trackers=True, obs=True)[0]
    if not true["ok"]:
        raise WorldError(f"{key}: the true prefix does not replay: {true.get('error')}")
    ent = [e for e in true["trackers"][viewer] if "obs" in e]
    if len(ent) <= n:
        raise WorldError(f"{key}: the viewer has {len(ent)} decisions, the root is decision {n}")
    root_chunk = int(ent[n]["after"])
    root.true = true
    root.rv = RV.from_lines(RV.viewer_lines(true, viewer, root_chunk), viewer)
    _, root.ref = reveals(true, viewer, n)
    root.ref_proto = protocol_lines(true, viewer, n)
    root.ref_cond = opp_conditions(true, oi, root_chunk)
    team = log["teams"][oi]
    entries = team.split("]")
    sets = [parse_set(e) for e in entries]
    sids = [toid(f[1] or f[0]) for f in sets]
    moves_of = [typed_moves(f) for f in sets]
    kept: Dict[str, Set[str]] = root.rv.moves()
    leak = {"attribution_miss": 0, "root_choice": None, "rewritten_moves": 0, "rewritten_switches": 0}
    last_opp = max((i for i, c in enumerate(cmds) if c[0] == opp_key), default=None)
    actives = opp_actives(true, oi, sum(1 for c in cmds if c[0] == opp_key))
    n_opp = 0
    tmpl: List[list] = []
    for i, (s_, t) in enumerate(cmds):
        if s_ == opp_key:
            n_opp += 1
        if s_ == opp_key and t.startswith("move ") and not t[5:].strip().isdigit():
            m = toid(t[5:])
            a_ = actives[n_opp - 1]            # the mon that made the choice
            k = sids.index(a_) if a_ in sids else None
            if k is None or m not in moves_of[k]:
                leak["attribution_miss"] += 1
                tmpl.append([s_, t])
            elif i == last_opp and root_choice_in_log:
                kept.setdefault(a_, set()).add(m)
                leak["root_choice"] = m
                tmpl.append([s_, t])
            elif m in kept.get(a_, set()):
                tmpl.append([s_, t])
            elif is_hp(m) and "hiddenpower" in kept.get(a_, set()):
                # a revealed Hidden Power: its TYPE is a spread fact the world draws, so the choice is the
                # world's typed id at that slot (keeping the true id would reject every other type)
                tmpl.append([s_, ("MOVE", k, moves_of[k].index(m))])
            else:
                tmpl.append([s_, ("MOVE", k, moves_of[k].index(m))])
                leak["rewritten_moves"] += 1
        elif s_ == opp_key and t.startswith("switch ") and not t[7:].strip().isdigit() \
                and toid(t[7:]) not in kept and toid(t[7:]) in sids:
            tmpl.append([s_, ("SLOT", sids.index(toid(t[7:])))])
            leak["rewritten_switches"] += 1
        else:
            tmpl.append([s_, t])
    root.team, root.entries, root.sets, root.sids, root.moves_of = team, entries, sets, sids, moves_of
    root.kept, root.tmpl, root.leak = kept, tmpl, leak
    return root


class WorldSampler:
    """``WorldSampler(reveal="off").sample(log, at, side, J, key)`` → :class:`Worlds` (raises :class:`WorldError` when
    a world cannot be built in ``max_tries``). ``reveal``: an information-lever spec (a preset name or clause text;
    validated against the SEARCH consumer); ``off`` is honest. ``key`` seeds every draw (deterministic)."""

    def __init__(self, prior: Optional[Prior] = None, max_tries: int = MAX_TRIES, reveal="off",
                 playout_check: bool = True, playout_lib=None):
        self.P = prior or Prior()
        self.max_tries = max_tries
        self.spec: InfoReveal = SEARCH.validate(reveal)
        self.pad = _gender_pad()
        self._spreads: Dict[str, WS.SpreadPrior] = {}
        #: every world is also OPENED by the playout tool (``rust_env_playout_open``) before it is returned: the
        #: core_events replay the gate runs accepts some choice logs the playout tool refuses (a numeric ``move k``, a
        #: Struggle the world does not force; the blind replica's finding), and a world search cannot play is no world
        self.playout_check = playout_check
        self._playout_lib = playout_lib
        self._playout_core = None
        #: set to a list to record why joint gates fail (the first divergent line), for diagnosis
        self.debug: Optional[List[dict]] = None

    def playout_refusal(self, log: dict, at: int, side: str) -> Optional[str]:
        """None when the playout tool opens ``log`` at ``at`` for ``side`` (the search's own replay), else its refusal."""
        if not self.playout_check:
            return None
        if self._playout_core is None:
            from utils.rust_env import ffi as F
            from utils.rust_env.build import ensure_built
            from utils.rust_env.successors import SearchCore

            lib = self._playout_lib
            if lib is None:
                import os

                profile = "selfcheck" if os.environ.get("POKESIM_EMISSION_SELFCHECK") == "1" else "release"
                lib = F.load(ensure_built(profile)[1])
            self._playout_core = SearchCore(lib=lib)
        req = {"log": log, "at": int(at), "side": side, "actions": None, "seeds": [None], "stall": None,
               "max_turns": 999, "keep_cmds": False}
        try:
            self._playout_core._cstr(self._playout_core.lib.rust_env_playout_open, json.dumps(req).encode())
        except Exception as e:                       # the tool's typed refusal: the world is rejected, named
            return f"{type(e).__name__}: {str(e)[:300]}"
        return None

    def _spread_prior(self, sid: str) -> WS.SpreadPrior:
        if sid not in self._spreads:
            self._spreads[sid] = WS.SpreadPrior(self.P.spreads(sid))
        return self._spreads[sid]

    def sample(self, log: dict, at: int, side: str, J: int, key: str,
               root_choice_in_log: bool = False, screen_cache: Optional[dict] = None) -> Worlds:
        """J worlds of the decision ``side`` takes at command ``at`` of ``log``. ``screen_cache``: a dict the caller
        keeps per run and passes to every spec it builds (the seen-mon samplers and gate verdicts, keyed by ``key``)."""
        from agents.battle.core_replay import run_core

        spec = self.spec
        root = prepare_root(log, at, side, key, root_choice_in_log)
        viewer, oi, n = root.viewer, root.oi, root.n
        rec, rv, ref, ref_proto, ref_cond = root.rec, root.rv, root.ref, root.ref_proto, root.ref_cond
        team, entries, sets, sids, moves_of = root.team, root.entries, root.sets, root.sids, root.moves_of
        kept, tmpl = root.kept, root.tmpl
        leak = dict(root.leak, residual=["%s:%s" % k for k in SEARCH.residual_of(spec)],
                    partial=["%s:%s" % k for k in SEARCH.partial_of(spec)])

        def world_cmds(team_w: str) -> List[list]:
            return render(tmpl, team_w)

        seen_slots = [i for i, s in enumerate(sids) if s in kept]
        never = [i for i, s in enumerate(sids) if s not in kept]
        draw_slots = gender_draw_slots(team) & set(never)
        true_problems = team_problems(team)
        fails: Dict[str, int] = {}
        verdict: Dict[str, tuple] = (screen_cache if screen_cache is not None else {}).setdefault(
            ("verdict", key), {})                     # screening-team text → (passed, score), cached
        screens = [0]

        opp_tag = f"p{oi + 1}"

        def gate(res: dict, cond_need: Set[str]) -> Tuple[bool, float, Optional[Tuple[str, int]]]:
            """(passed, how far the searched side's protocol matches the truth's (``divergence_score``), the stat the
            first difference implicates (``divergence_hint``))."""
            if not res["ok"]:
                return False, -1, None
            try:
                _, rows = reveals(res, viewer, n)
            except WorldError:                 # the world ended or stalled before the root: score what it showed
                part = RV.viewer_lines(res, viewer)
                return False, divergence_score(part, ref_proto), divergence_hint(part, ref_proto, opp_tag)
            proto = protocol_lines(res, viewer, n)
            ln = divergence_score(proto, ref_proto)
            hint = divergence_hint(proto, ref_proto, opp_tag)
            if rows != ref or proto != ref_proto:
                return False, ln, hint
            if cond_need:
                ec = [e for e in res["trackers"][viewer] if "obs" in e]
                got = opp_conditions(res, oi, int(ec[n]["after"]))
                if any(got.get(s) != ref_cond.get(s) for s in cond_need):
                    return False, ln, None
            return True, ln, None

        def run_gate(jobs: List[Tuple[str, FrozenSet[str]]], lenient: bool = False
                     ) -> List[Tuple[bool, float, Optional[Tuple[str, int]]]]:
            """Gate verdicts for (opponent team text, species whose exact HP must equal the truth's), cached.
            ``lenient`` (the SCREEN only): the prefix is fed with ``CHOOSEIF`` — a recorded choice the diverged world
            no longer asks for is skipped instead of refusing the replay, so a wrong candidate still yields the
            protocol its search scores; the joint gate of every returned world is STRICT (``CHOOSE``)."""
            tag = "#lenient" if lenient else "#strict"
            todo: Dict[str, Tuple[str, FrozenSet[str]]] = {}
            for t, c in jobs:
                kk = t + "#" + ",".join(sorted(c)) + tag
                if kk not in verdict:
                    todo[kk] = (t, c)
            if todo:
                def cmds_of(t: str) -> List[list]:
                    cs = world_cmds(t)
                    return [c + ["if_open"] if lenient and c[0] != "forcelose" else c for c in cs]
                out = run_core([rec(t, f"{key}#s{screens[0] + i}", cmds_of(t)) for i, (t, _) in enumerate(todo.values())],
                               trackers=True, obs=True)
                screens[0] += len(todo)
                for (kk, (_, c)), r in zip(todo.items(), out):
                    verdict[kk] = gate(r, set(c))
            return [verdict[t + "#" + ",".join(sorted(c)) + tag] for t, c in jobs]

        def screen_gate(jobs: List[Tuple[str, int, bool]]) -> List[Tuple[bool, float, Optional[Tuple[str, int]]]]:
            return run_gate([(t, frozenset({sids[i]}) if need else frozenset()) for t, i, need in jobs], lenient=True)

        # one sampler per (seen mon, constraint), shared by the worlds of this decision AND, through
        # ``screen_cache``, by every other spec built at it: arms that differ only in other levers take the same
        # seen-mon draws (common random numbers for their paired contrasts) and pay the screen once
        cache = screen_cache if screen_cache is not None else {}
        samplers: Dict[WS.ScreenKey, WS.MonSampler] = cache.setdefault(("samplers", key), {})

        def sampler_for(k: WS.ScreenKey, i: int, rng) -> WS.MonSampler:
            if k not in samplers:
                sid = sids[i]
                samplers[k] = WS.MonSampler(
                    key=k, sid=sid, f_true=list(sets[i]), items=self.P.items(sid), abilities=self.P.ability(sid),
                    spreads=self._spread_prior(sid), hp_prior=self.P.hidden_power(sid),
                    hp_any=min(1.0, sum(v for mv, v in self.P.moves(sid).items() if is_hp(mv))),
                    rng=np.random.default_rng(int(hashlib.sha256(f"world_screen_v1:{key}:{k}".encode())
                                                  .hexdigest()[:16], 16)), needed=0)
                if k.moves_told:
                    samplers[k].legal = (lambda e, i=i: not (mon_problems(e) - mon_problems(entries[i])))
            return samplers[k]

        def build_base(j: int, attempt: int):
            """Draw everything of world j that needs no screen, and the screen key of each seen mon."""
            rng = key_rng(key, j, attempt)
            out = list(entries)
            facts: Dict[int, dict] = {}
            keys: Dict[int, WS.ScreenKey] = {}
            need_hp: Set[str] = set()
            pinned: Set[int] = set()
            hp_lv = spec.lever("hp", "seen")
            for i in seen_slots:
                f = list(sets[i])
                sid = sids[i]
                mr = rv.mons.get(sid)
                tm = moves_of[i]
                keep = {s: m for s, m in enumerate(tm)
                        if m in kept[sid] or (is_hp(m) and "hiddenpower" in kept[sid])}
                rev_hp_slot = next((s for s, m in keep.items() if is_hp(m) and m not in kept[sid]), None)
                rev_hp[i] = rev_hp_slot
                revd = {"item": mr.item if mr is not None else None, "ability": mr.ability if mr is not None else None}
                # a reveal that names a fact the true set does not hold is the PARSER's error: counted, and the
                # fact is drawn (never kept true on a reveal that did not happen)
                for cls, fi in (("item", 2), ("ability", 3)):
                    if revd[cls] is not None and revd[cls] != toid(f[fi]):
                        leak["reveal_mismatch"] = leak.get("reveal_mismatch", 0) + 1
                        revd[cls] = None
                fixed: Dict[str, object] = {}
                excl: Dict[str, object] = {}
                maps: List[str] = []
                how: Dict[str, Tuple[str, str]] = {}
                truth_of = {"item": toid(f[2]), "ability": toid(f[3]), "spread": "TRUE"}
                for cls in ("item", "ability", "spread"):
                    lv = spec.lever(cls, "seen")
                    if cls != "spread" and revd[cls] is not None:
                        fixed[cls] = truth_of[cls]
                        how[cls] = ("revealed", "")
                    elif fires(rng, lv):
                        assert lv is not None
                        how[cls] = ("told", lv.source)
                        if lv.source == "truth":
                            fixed[cls] = truth_of[cls]
                        elif lv.source == "placebo":
                            sp_t = Spread.of(f)
                            excl[cls] = truth_of[cls] if cls != "spread" else (toid(sp_t.nature), tuple(sp_t.evs))
                        else:
                            maps.append(cls)
                    else:
                        how[cls] = ("drawn", "")
                # the MOVE lever's mass is the presence mixture itself (§6.4 of the study): never a per-world coin
                mv_lv = spec.lever("moves", "seen")
                true_hp = any(is_hp(m) for m in tm)
                if fixed.get("spread") == "TRUE" or (mv_lv is not None and mv_lv.source == "truth"
                                                      and mv_lv.mass_bp >= 10_000):
                    hp_rule = "must" if true_hp else "none"     # the told fact says whether it carries one
                elif rev_hp_slot is not None:
                    hp_rule = "must"
                else:
                    hp_rule = "free"
                need = fires(rng, hp_lv)
                if need:
                    need_hp.add(sid)
                told_moves = mv_lv is not None and mv_lv.source == "truth" and mv_lv.mass_bp >= 10_000
                keys[i] = WS.ScreenKey(slot=i, hp=hp_rule, moves_told=told_moves, fixed=tuple(sorted(fixed.items())),
                                       exclude=tuple(sorted(excl.items())), map=tuple(maps), need_hp=need)
                facts[i] = {"slot": i, "species": sid, "seen": True, "_how": how, "_plan": (keep, mv_lv)}
                if need:
                    facts[i]["hp"] = {"how": "told", "source": "truth", "match": True}
            sp = never_seen_species(rng, self.P, never, {i: sids[i] for i in never}, set(kept), spec)
            if sp is None:
                return None, "sample"
            picks, sprov = sp
            true_never = {sids[i]: i for i in never}
            for i in never:
                s = picks[i]
                f_true = sets[true_never[s]] if s in true_never else None
                got2 = new_set(rng, self.P, s, f_true, spec)
                if got2 is None:
                    return None, "sample"
                ent, nf = got2
                if f_true is not None and s == sids[i] and all(v["match"] for v in nf.values()):
                    ent = entries[i]                 # every fact the truth's: the true entry verbatim
                out[i] = ent
                facts[i] = {"slot": i, "species": s, "seen": False, "species_fact": sprov[i], **nf,
                            "true_mon": f_true is not None}
                if sprov[i]["how"] == "told":
                    pinned.add(i)
            # the never-seen mons' genders and the team's legality are fixed BEFORE the (costly) seen-mon screen
            src_of: Dict[int, int] = {}
            placed = place_genders(rng, out, never, draw_slots, pinned, self.pad, src_of)
            if placed is None:
                return None, "gender_count"
            # each never-seen mon's record MOVES WITH IT (the placement permutes them; GIGO 2026-10-15), and its
            # species flags are read at the slot it now holds: ``match`` = a TRUE never-seen species (set-based: the
            # drawn slots are exchangeable), ``slot_match`` = the truth's species at this slot
            drawn = {i: facts[i] for i in never}
            true_never_sp = {sids[i] for i in never}
            for slot, src in src_of.items():
                fa = dict(drawn[src], slot=slot, drawn_slot=src)
                fa["species_fact"] = dict(fa["species_fact"], match=fa["species"] in true_never_sp,
                                          slot_match=fa["species"] == sids[slot])
                facts[slot] = fa
            if team_problems("]".join(placed)) - true_problems:
                return None, "clause"
            return (rng, placed, facts, keys, need_hp, pinned), None

        rev_hp: Dict[int, Optional[int]] = {}
        refusals: List[str] = []
        used: Dict[WS.ScreenKey, int] = {}
        teams: List[Optional[str]] = [None] * J
        stats: List[dict] = [{} for _ in range(J)]
        wfacts: List[List[dict]] = [[] for _ in range(J)]
        tries = [0] * J
        attempt = 0
        while attempt < self.max_tries and any(t is None for t in teams):
            todo = [j for j in range(J) if teams[j] is None]
            base = {}
            for j in todo:
                tries[j] += 1
                b, why = build_base(j, attempt)
                if b is None:
                    fails[why] = fails.get(why, 0) + 1
                    continue
                base[j] = b
            # ---- one shared posterior sampler per (seen mon, constraint): every world takes the next draw
            want: Dict[WS.ScreenKey, int] = {}
            for j, b in base.items():
                for i, k in b[3].items():
                    want[k] = want.get(k, 0) + 1
                    sampler_for(k, i, b[0])
            for k, c in want.items():
                samplers[k].needed = max(samplers[k].needed, used.get(k, 0) + c)
            WS.run([samplers[k] for k in want], rev_hp, entries, screen_gate)
            dead = [samplers[k] for k in want if samplers[k].failed]
            if dead:                            # a seen mon no honest draw fits: every further try fails the same way
                fails["screen"] = fails.get("screen", 0) + len(base)
                raise WorldError(f"{key}: no honest draw of " + ", ".join(f"{m.sid} (slot {m.key.slot})" for m in dead)
                                 + f" fits the log in {WS.MAX_STEPS} chain steps ({fails})")
            final: Dict[int, Tuple[str, List[dict], Set[str]]] = {}
            for j, (rng, out, facts, keys, need_hp, pinned) in base.items():
                ok_all = True
                for i, k in keys.items():
                    m = samplers[k]
                    if used.get(k, 0) >= len(m.samples):
                        ok_all = False
                        break
                    st = m.samples[used.get(k, 0)]
                    used[k] = used.get(k, 0) + 1
                    # the moves AFTER the screen: the drawn IVs say whether the set holds a Hidden Power, and its type
                    keep, mv_lv = facts[i].pop("_plan")
                    hpt = WS.hp_split(st[4])[0]                  # the Hidden Power TYPE the screened IVs carry
                    e_w = None
                    for _r in range(MOVE_REDRAWS):         # a set the format bans (Speed Pass, …) is redrawn
                        got = WD.draw_moves(rng, self.P.moves(sids[i]), keep, len(moves_of[i]), mv_lv, moves_of[i],
                                            hpt, hp_rule="force" if hpt is not None else "none")
                        if got is None:
                            break
                        mv, facts[i]["moves"] = got
                        f_w = list(sets[i])
                        f_w[4] = ",".join(WD.align_to_truth(mv, moves_of[i], set(keep)))
                        e_w = m.entry(st, rev_hp.get(i), f_w)
                        if not (mon_problems(e_w) - mon_problems(entries[i])):
                            break
                        e_w = None
                    if e_w is None:
                        ok_all = False
                        break
                    out[i] = e_w
                    facts[i].update(WS.facts_of(st, sets[i], facts[i].pop("_how")))
                if not ok_all:
                    fails["screen"] = fails.get("screen", 0) + 1
                    continue
                tw = "]".join(out)
                if team_problems(tw) - true_problems:
                    fails["clause"] = fails.get("clause", 0) + 1
                    continue
                final[j] = (tw, [facts[i] for i in range(len(entries))], need_hp)
            oks = run_gate([(t, frozenset(c)) for t, _, c in final.values()])
            for (j, (tw, fl, _)), (ok, _ln, _h) in zip(final.items(), oks):
                if ok:
                    lw = dict(log, teams=[tw if s_ == oi else t_ for s_, t_ in enumerate(log["teams"])],
                              cmds=[f"FORCELOSE {tok}" if c0 == "forcelose" else f"CHOOSE {c0} {tok}"
                                    for c0, tok in world_cmds(tw)] + list(log["cmds"][at:]))
                    why = self.playout_refusal(lw, at, side)
                    if why is not None:
                        fails["playout_refused"] = fails.get("playout_refused", 0) + 1
                        refusals.append(why)
                        continue
                    teams[j] = tw
                    wfacts[j] = fl
                    stats[j] = {"attempt": attempt}
                else:
                    fails["joint"] = fails.get("joint", 0) + 1
                    if self.debug is not None and len(self.debug) < 20:
                        r = run_core([rec(tw, f"{key}#dbg", world_cmds(tw))], trackers=True, obs=True)[0]
                        pr = protocol_lines(r, viewer, n) if r["ok"] else []
                        k_ = next((i for i, (a, b) in enumerate(zip(pr, ref_proto)) if a != b), None)
                        self.debug.append({"key": key, "world": j, "ok": r["ok"], "error": r.get("error"),
                                           "at": k_, "world_line": pr[k_] if k_ is not None else None,
                                           "true_line": ref_proto[k_] if k_ is not None else None,
                                           "rows_equal": r["ok"] and reveals(r, viewer, n)[1] == ref,
                                           "before": ref_proto[max(0, (k_ or 0) - 6):(k_ or 0)],
                                           "team": tw, "true_team": team})
            attempt += 1
        if any(t is None for t in teams):
            if refusals:
                fails["playout_refusal_example"] = refusals[0]          # type: ignore[assignment]
            raise WorldError(f"{key}: {sum(t is None for t in teams)} of {J} worlds not built in {self.max_tries} "
                             f"tries ({fails})")
        # THE THROWING GUARD: slot <-> species <-> facts <-> the engine's view of every returned world
        WG.check_worlds(root, [str(t) for t in teams], wfacts, key)
        logs = []
        for t in teams:
            tw = list(log["teams"])
            tw[oi] = t
            pre = [f"FORCELOSE {tok}" if s_ == "forcelose" else f"CHOOSE {s_} {tok}" for s_, tok in world_cmds(t)]
            logs.append(dict(log, teams=tw, cmds=pre + list(log["cmds"][at:])))
        return Worlds(logs=logs, teams=[str(t) for t in teams], tries=tries, leak=leak,
                      revealed={k: sorted(v) for k, v in kept.items()}, stats=stats, fails=fails, facts=wfacts,
                      spec=spec.record(), screens=screens[0],
                      samplers=[{"slot": k.slot, "species": m.sid, "needed": m.needed, "samples": len(m.samples),
                                 "proposals": m.proposals, "steps": m.steps, "failed": m.failed,
                                 "prop_stats": m.prop_stats, "need_hp": k.need_hp} for k, m in samplers.items() if k in used])
