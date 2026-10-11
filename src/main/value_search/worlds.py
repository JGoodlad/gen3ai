"""THE WORLD SAMPLER: honest-information (DETERMINIZED) worlds for a decision of a core input log.

A world is the battle's input log with the OPPONENT's hidden team replaced by a draw from the Smogon
prior (``agents.gen3_data.priors``), CONDITIONED on what the searched side observed by rejection: the
world's log, cut at the root, is replayed through the core (``core_events --trackers --obs``) and is
kept only if the searched side's observation row at EVERY one of its decisions up to and including the
root is BYTE-EQUAL to the true log's. The value-discrimination pilot measured this sampler
(`designs/endstate/design_value_discrimination.md` §7.10 Amendment 2; its pinned twin is
`research_state/measurements/value_disc_pilot_2026-10-11/determinize.py`).

What is drawn: every opponent slot the searched side never saw (species ∝ the Smogon teammate
co-occurrence with the revealed species, else species usage; never a species already on the team;
moves / item / ability / nature + EVs ∝ that species' usage; a Hidden Power sets its IVs), and every
move slot of a seen mon whose move it never saw used.

What stays TRUE (the named leaks, counted per call in ``Worlds.leak``): a seen mon's species, item,
ability, nature, EVs and IVs (they shaped the observed history, so the gate would reject a redraw); a
past opponent move choice never seen executed is rewritten to its SLOT (``move k``) and a past switch to
a never-seen mon to the world's mon in that slot, except an AMBIGUOUS move (in several seen mons' sets)
kept by name; the opponent's ROOT-turn choice when the log already holds it; and the TRUE pre-root dice
(a world must reproduce the true team's construction draws, e.g. each slot's gender-draw class).

Only moves and items the engine runs are drawn (``BANNED_MOVES`` / ``BANNED_ITEMS``, the engine's
``scan_move_probe`` FAIL-LOUD lists, measured 2026-10-11); a FAIL-LOUD move panics the core.
Offline / evaluation use only.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

HP_TYPES = ("fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel", "fire", "water",
            "grass", "electric", "psychic", "ice", "dragon", "dark")
STATS = ("hp", "atk", "def", "spa", "spd", "spe")       # the packed-team order
MAX_TRIES = 30

#: the engine's FAIL-LOUD moves and items (``scan_move_probe``, PROBE_KIND=move|item, 2026-10-11, at
#: 62085750 and 95d014fa): drawing one would panic the core.
BANNED_MOVES = frozenset("""attract bide blastburn bounce camouflage dig dive flatter fly followme foresight
frenzyplant grudge helpinghand hydrocannon ingrain lockon magiccoat magnitude metronome mindreader mirrormove
mist mudsport naturepower nightmare odorsleuth present psywave razorwind roleplay sketch skullbash skyattack
snore spite spitup stockpile swallow teleport triplekick watersport""".split())
BANNED_ITEMS = frozenset({"machobrace", "mail", "mentalherb", "shellbell"})


class WorldError(RuntimeError):
    pass


def toid(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def hp_type_of(ivs: Sequence[int]) -> str:
    """Gen-3 Hidden Power type from IVs given in the packed order (HP, Atk, Def, SpA, SpD, Spe)."""
    iv = dict(zip(STATS, ivs))
    v = sum((iv[k] & 1) << i for i, k in enumerate(("hp", "atk", "def", "spe", "spa", "spd")))
    return HP_TYPES[v * 15 // 63]


def hp_ivs(t: str) -> List[int]:
    """30/31 IVs (Hidden Power at 70 power) giving type ``t``, with the most 31s, in the packed order."""
    best: Optional[List[int]] = None
    for mask in range(64):
        b = [(mask >> i) & 1 for i in range(6)]          # hp, atk, def, spe, spa, spd
        v = sum(x << i for i, x in enumerate(b))
        if HP_TYPES[v * 15 // 63] == t and (best is None or sum(b) > sum(best)):
            best = b
    if best is None:
        raise WorldError(f"no Hidden Power IVs for type {t!r}")
    iv = dict(zip(("hp", "atk", "def", "spe", "spa", "spd"), (30 + x for x in best)))
    return [iv[k] for k in STATS]


def parse_set(entry: str) -> List[str]:
    f = entry.split("|")
    return f + [""] * (12 - len(f)) if len(f) < 12 else f


def set_ivs(f: Sequence[str]) -> List[int]:
    raw = f[8].split(",") if f[8] else [""] * 6
    return [int(x) if x else 31 for x in raw]


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


class Prior:
    """The Smogon prior through the ``agents.gen3_data`` facade."""

    def __init__(self):
        from agents.gen3_data import priors, species

        self._p, self._s = priors, species
        self.usage = dict(priors.species_usage())
        self.pool = sorted(s for s in self.usage if priors.moves(s) and species.get(s) is not None)

    def name(self, sid: str) -> str:
        return self._s.get(sid).name

    def moves(self, sid):
        return self._p.moves(sid)

    def items(self, sid):
        return self._p.items(sid)

    def ability(self, sid):
        return self._p.ability(sid)

    def spreads(self, sid):
        return self._p.spreads(sid)

    def mates(self, sid):
        return self._p.teammates(sid)


def _draw(rng, keys: List[str], w) -> Optional[str]:
    w = np.asarray(w, dtype=np.float64)
    if not keys or w.sum() <= 0:
        return None
    return keys[int(rng.choice(len(keys), p=w / w.sum()))]


def sample_moves(rng, P: Prior, sid: str, keep: Dict[int, str], hp_type: Optional[str]) -> Optional[List[str]]:
    """The four move slots: ``keep`` stays; the rest drawn ∝ the species' move usage, without
    replacement; at most one Hidden Power, and only of ``hp_type`` when given."""
    pri = P.moves(sid)
    if not pri:
        return None
    slots = [keep.get(i) for i in range(4)]
    have = {m for m in slots if m}
    for i in range(4):
        if slots[i]:
            continue
        has_hp = any(m.startswith("hiddenpower") for m in have)
        keys = [m for m in pri if m not in have and m not in BANNED_MOVES
                and not (m.startswith("hiddenpower") and (has_hp or (hp_type is not None and m != f"hiddenpower{hp_type}")))]
        m = _draw(rng, keys, [pri[k] for k in keys])
        if m is None:
            return None
        slots[i] = m
        have.add(m)
    return [str(m) for m in slots]


def sample_new_set(rng, P: Prior, sid: str) -> Optional[str]:
    moves = sample_moves(rng, P, sid, {}, None)
    if moves is None:
        return None
    it = {k: v for k, v in P.items(sid).items() if k not in BANNED_ITEMS}
    item = _draw(rng, list(it), list(it.values())) if it else None
    ab = P.ability(sid)
    ability = _draw(rng, list(ab), list(ab.values())) if ab else None
    sp = P.spreads(sid) or [["Hardy", [84] * 6, 1.0]]
    k = int(rng.choice(len(sp), p=np.array([x[2] for x in sp]) / sum(x[2] for x in sp)))
    hp = [m for m in moves if m.startswith("hiddenpower")]
    ivs = ",".join(str(x) for x in hp_ivs(hp[0][len("hiddenpower"):])) if hp else ""
    evs = ",".join(str(int(x)) if int(x) else "" for x in sp[k][1])
    return "|".join([P.name(sid), "", "" if item in (None, "nothing") else item, ability or "", ",".join(moves),
                     sp[k][0], evs, "", ivs, "", "", ""])


def redraw_seen(rng, P: Prior, entry: str, seen_moves: Set[str]) -> Tuple[str, int]:
    f = parse_set(entry)
    sid = toid(f[1] or f[0])
    mv = f[4].split(",")
    keep = {i: m for i, m in enumerate(mv)
            if m in seen_moves or (m.startswith("hiddenpower") and "hiddenpower" in seen_moves)}
    if len(keep) == len(mv) or not P.moves(sid):
        return entry, 0
    out = sample_moves(rng, P, sid, keep, hp_type_of(set_ivs(f)))
    if out is None:
        return entry, 0
    f[4] = ",".join(out)
    return "|".join(f), len(mv) - len(keep)


def sample_team(rng, P: Prior, team: str, kept: Dict[str, Set[str]]) -> Tuple[str, dict]:
    entries = team.split("]")
    ids = [toid(parse_set(e)[1] or parse_set(e)[0]) for e in entries]
    present = {s for s in ids if s in kept}
    out, new, redrawn = [], 0, 0
    for e, sid in zip(entries, ids):
        if sid in kept:
            e2, k = redraw_seen(rng, P, e, kept[sid])
            out.append(e2)
            redrawn += k
            continue
        cand = [s for s in P.pool if s not in present]
        w = (np.array([np.mean([P.mates(r).get(s, 0.0) for r in kept]) for s in cand]) if kept
             else np.array([P.usage.get(s, 0.0) for s in cand]))
        if w.sum() <= 0:
            w = np.array([P.usage.get(s, 0.0) for s in cand])
        s = _draw(rng, cand, w)
        ent = sample_new_set(rng, P, s) if s else None
        if ent is None:
            out.append(e)
            continue
        present.add(s)
        out.append(ent)
        new += 1
    return "]".join(out), {"new_slots": new, "redrawn_move_slots": redrawn}


def reveals(res: dict, viewer: int, n: int) -> Tuple[Dict[str, Set[str]], List[str]]:
    """{opponent species id seen: move ids seen used} up to the viewer's n-th decision (inclusive),
    and the viewer's obs frames (base64) at decisions 0..n."""
    opp = f"p{2 - viewer}a: "
    ent = [e for e in res["trackers"][viewer] if "obs" in e]
    if len(ent) <= n:
        raise WorldError(f"the viewer has {len(ent)} decisions, the root is decision {n}")
    after = int(ent[n]["after"])
    nick: Dict[str, str] = {}
    seen: Dict[str, Set[str]] = {}
    for side, lines in res["chunks"][: after + 1]:
        if side != viewer:
            continue
        for line in lines:
            p = line.split("|")
            if len(p) < 4 or not p[2].startswith(opp):
                continue
            if p[1] in ("switch", "drag"):
                nick[p[2]] = toid(p[3].split(",")[0])
                seen.setdefault(nick[p[2]], set())
            elif p[1] == "move" and p[2] in nick:
                seen[nick[p[2]]].add(toid(p[3]))
    return seen, [e["obs"]["b64"] for e in ent[: n + 1]]


@dataclass
class Worlds:
    """J gate-passed worlds: each a full core input log (same ``at``) and its opponent team."""

    logs: List[dict]
    teams: List[str]
    tries: List[int]
    leak: dict
    revealed: Dict[str, List[str]]
    stats: List[dict] = field(default_factory=list)


def key_rng(key: str, j: int):
    return np.random.default_rng(int(hashlib.sha256(f"value_search_world_v1:{key}:{j}".encode()).hexdigest()[:16], 16))


class WorldSampler:
    """``WorldSampler().sample(log, at, side, J, key)`` → :class:`Worlds` (raises :class:`WorldError`
    when a world cannot be built in ``max_tries``). ``key`` seeds every draw (deterministic)."""

    def __init__(self, prior: Optional[Prior] = None, max_tries: int = MAX_TRIES):
        self.P = prior or Prior()
        self.max_tries = max_tries

    def sample(self, log: dict, at: int, side: str, J: int, key: str,
               root_choice_in_log: bool = False) -> Worlds:
        from agents.battle.core_replay import RecordedBattle, run_core

        viewer = 0 if side == "p1" else 1
        oi = 1 - viewer
        opp_key = "p2" if side == "p1" else "p1"
        cmds = log_commands(log, at)
        n = sum(1 for s, _ in cmds if s == side)

        def rec(team_opp: Optional[str], label: str, commands):
            teams = list(log["teams"])
            if team_opp is not None:
                teams[oi] = team_opp
            return RecordedBattle(label=label, format_id=log["format_id"], seed=log["seed"],
                                  p1={"name": log["names"][0], "team": teams[0]},
                                  p2={"name": log["names"][1], "team": teams[1]}, commands=commands)

        true = run_core([rec(None, f"{key}#true", cmds)], trackers=True, obs=True)[0]
        if not true["ok"]:
            raise WorldError(f"{key}: the true prefix does not replay: {true.get('error')}")
        revealed, ref = reveals(true, viewer, n)
        team = log["teams"][oi]
        sets = [parse_set(e) for e in team.split("]")]
        sids = [toid(f[1] or f[0]) for f in sets]
        moves_of = [f[4].split(",") for f in sets]
        kept = {k: set(v) for k, v in revealed.items()}
        leak = {"ambiguous_moves": [], "root_choice": None, "rewritten_moves": 0, "rewritten_switches": 0}
        last_opp = max((i for i, c in enumerate(cmds) if c[0] == opp_key), default=None)
        tmpl: List[list] = []
        for i, (s_, t) in enumerate(cmds):
            if s_ == opp_key and t.startswith("move ") and not t[5:].strip().isdigit():
                m = toid(t[5:])
                cand = [k for k in range(len(sets)) if sids[k] in kept and m in moves_of[k]]
                seen_for = [k for k in cand if m in kept[sids[k]]
                            or (m.startswith("hiddenpower") and "hiddenpower" in kept[sids[k]])]
                if i == last_opp and root_choice_in_log:
                    for k in cand:
                        kept[sids[k]].add(m)
                    leak["root_choice"] = m
                    tmpl.append([s_, t])
                elif seen_for:
                    tmpl.append([s_, t])
                elif len(cand) == 1:
                    tmpl.append([s_, ("MOVE", cand[0], moves_of[cand[0]].index(m))])
                    leak["rewritten_moves"] += 1
                else:
                    for k in cand:
                        kept[sids[k]].add(m)
                        leak["ambiguous_moves"].append(f"{sids[k]}:{m}")
                    tmpl.append([s_, t])
            elif s_ == opp_key and t.startswith("switch ") and not t[7:].strip().isdigit() \
                    and toid(t[7:]) not in kept and toid(t[7:]) in sids:
                tmpl.append([s_, ("SLOT", sids.index(toid(t[7:])))])
                leak["rewritten_switches"] += 1
            else:
                tmpl.append([s_, t])

        def world_cmds(team_w: str) -> List[list]:
            """The prefix with every SLOT-tagged choice rendered against the world's team: a switch to the
            world's mon in that slot, a move as the world's move ID at that slot of that mon (the playout
            core resolves a replayed choice against the offered tokens BY NAME: a numeric ``move k``
            is refused there, though ``core_events`` accepts it)."""
            sets_w = [parse_set(e) for e in team_w.split("]")]

            def render(t):
                if not isinstance(t, tuple):
                    return t
                if t[0] == "SLOT":
                    return f"switch {sets_w[t[1]][0]}"
                return f"move {sets_w[t[1]][4].split(',')[t[2]]}"      # ("MOVE", mon, slot)
            return [[s_, render(t)] for s_, t in tmpl]

        rngs = [key_rng(key, j) for j in range(J)]
        teams: List[Optional[str]] = [None] * J
        stats: List[dict] = [{} for _ in range(J)]
        tries = [0] * J
        for _ in range(self.max_tries):
            todo = [j for j in range(J) if teams[j] is None]
            if not todo:
                break
            cand_ = {}
            for j in todo:
                cand_[j] = sample_team(rngs[j], self.P, team, kept)
                tries[j] += 1
            out = run_core([rec(cand_[j][0], f"{key}#w{j}", world_cmds(cand_[j][0])) for j in todo],
                           trackers=True, obs=True)
            for j, r in zip(todo, out):
                if not r["ok"]:
                    continue
                try:
                    _, rows = reveals(r, viewer, n)
                except WorldError:
                    continue
                if rows == ref:
                    teams[j], stats[j] = cand_[j]
        if any(t is None for t in teams):
            raise WorldError(f"{key}: {sum(t is None for t in teams)} of {J} worlds not built in {self.max_tries} tries")
        logs = []
        for t in teams:
            tw = list(log["teams"])
            tw[oi] = t
            pre = [f"FORCELOSE {tok}" if s_ == "forcelose" else f"CHOOSE {s_} {tok}" for s_, tok in world_cmds(t)]
            logs.append(dict(log, teams=tw, cmds=pre + list(log["cmds"][at:])))
        return Worlds(logs=logs, teams=[str(t) for t in teams], tries=tries, leak=leak,
                      revealed={k: sorted(v) for k, v in kept.items()}, stats=stats)
