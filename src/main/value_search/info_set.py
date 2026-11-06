"""THE SEARCHED SIDE'S INFORMATION SET I, and what the leak-honest builder reads from it alone
(``designs/endstate/design_sprime_honest.md`` D1, D6, D7, D4; the builder is :mod:`main.value_search.honest_worlds`).

I is everything the searched side (the VIEWER) knows at a decision: the format, both names, its own packed team, every
protocol line it was shipped up to the decision (its requests included) and its own commands. :func:`extract` reads I
off a battle's core input log with ONE replay (the live searcher holds the true log; nothing else of that replay is
returned), and everything here is a function of I alone:

* :meth:`InfoSet.segments` — I's lines split after each ``|request|``: the lines ONE write shipped the viewer
  (segment 0 is the START write). The fresh-dice matcher (``pokesim::fresh_dice``) reproduces them one by one.
* :meth:`InfoSet.key` — sha256 of I's canonical text (D7): every draw the builder and the search make derives from it.
* :func:`intents` — per segment, the OPPONENT's choice in the update that resolves into it, read from the segment's
  own lines (D6): its first ``|switch|`` / own ``|move|`` (never one CALLED or LOCKED: a ``[from]`` line), Focus
  Punch's tightening, the move a ``cant`` line names; ``None`` (a STAND-IN the matcher resolves to the world's first
  offered choice that reproduces the segment) when the opponent was seen to make none (asleep, paralysed, KO'd first).
* :func:`backtrack` — per segment, the earlier segments whose dice a failure there may RE-ROLL (D4): the onsets, in
  I's lines, of a hidden counter still running (a non-Rest sleep, confusion, Encore, Disable, a partial trap, an
  Outrage-family lock, Uproar, a pending Future Sight / Doom Desire).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from agents.info_levers import revealed as RV

SCHEMA = "gen3_info_set_v1"
#: the partial-trapping moves of gen 3 (a ``-activate|X|move: <name>`` onset)
TRAPS = ("Wrap", "Bind", "Fire Spin", "Whirlpool", "Sand Tomb", "Clamp")
#: the Outrage-family locks (2-3 turns, drawn at the first use)
LOCKS = ("outrage", "thrash", "petaldance")


def toid(s: str) -> str:
    return RV.toid(s)


@dataclass(frozen=True)
class InfoSet:
    """The searched side's information set at one decision (module docstring)."""

    format_id: str
    names: Tuple[str, str]
    side: str
    team: str
    lines: Tuple[str, ...]
    cmds: Tuple[str, ...]

    @property
    def viewer(self) -> int:
        return 0 if self.side == "p1" else 1

    @property
    def opp(self) -> str:
        return "p2" if self.side == "p1" else "p1"

    def segments(self) -> List[List[str]]:
        out: List[List[str]] = [[]]
        for ln in self.lines:
            out[-1].append(ln)
            if ln.startswith("|request|"):
                out.append([])
        if not out[-1]:
            out.pop()
        return out

    def canonical(self) -> str:
        return json.dumps({"schema": SCHEMA, "format_id": self.format_id, "names": list(self.names),
                           "side": self.side, "team": self.team, "lines": list(self.lines),
                           "cmds": list(self.cmds)}, sort_keys=True, separators=(",", ":"))

    def key(self) -> str:
        """D7: the digest every honest draw derives from."""
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    def as_dict(self) -> dict:
        return json.loads(self.canonical())


def extract(log: dict, at: int, side: str) -> Tuple[InfoSet, str]:
    """``(I, the viewer's root row)`` at command ``at`` of ``log``: ONE replay of the log through the core, of which
    only the viewer's own lines, its own commands and its root observation row (a function of them) are kept. A log
    that does not replay to a decision of ``side`` there is a :class:`ValueError`."""
    from agents.battle.core_replay import RecordedBattle, run_core
    from main.value_search.worlds import log_commands

    cmds = log_commands(log, at)
    viewer = 0 if side == "p1" else 1
    rec = RecordedBattle(label="info_set", format_id=log["format_id"], seed=log["seed"],
                         p1={"name": log["names"][0], "team": log["teams"][0]},
                         p2={"name": log["names"][1], "team": log["teams"][1]}, commands=cmds,
                         rolls=({"viewer": viewer, "mode": "force", "force": log["rolls"]} if log.get("rolls") else None),
                         reseed=log.get("reseed"))
    res = run_core([rec], trackers=True, obs=True)[0]
    if not res["ok"]:
        raise ValueError(f"the log does not replay: {res.get('error')}")
    ours = [t for s, t in cmds if s == side]
    ent = [e for e in res["trackers"][viewer] if "obs" in e]
    if len(ent) <= len(ours):
        raise ValueError(f"the viewer has {len(ent)} decisions, the root is decision {len(ours)}")
    lines = RV.viewer_lines(res, viewer, int(ent[len(ours)]["after"]))
    if not lines or not lines[-1].startswith("|request|"):
        raise ValueError("the viewer's lines do not end at its root request")
    info = InfoSet(format_id=str(log["format_id"]), names=(str(log["names"][0]), str(log["names"][1])), side=side,
                   team=str(log["teams"][viewer]), lines=tuple(lines), cmds=tuple(ours))
    return info, ent[len(ours)]["obs"]["b64"]


def _mon(tok: str) -> Optional[Tuple[str, str]]:
    """``p2a: Nick`` -> (``p2``, ``Nick``)."""
    head, sep, name = tok.partition(": ")
    if not sep or len(head) < 3 or head[0] != "p":
        return None
    return head[:2], name


def opp_chooses_after(seg: Sequence[str], opp: str) -> bool:
    """Whether the OPPONENT makes a new choice in the update that follows segment ``seg`` (read from ``seg``): a new
    turn (``|turn|``), the viewer waiting (``"wait"``: the opponent switches alone), or the opponent's active
    fainted (it switches beside the viewer). A mid-turn request of the viewer alone (its Baton Pass, its own faint, a
    re-request after a refused choice) continues the SAME update for the opponent."""
    if any(ln.startswith("|turn|") for ln in seg):
        return True
    if seg and seg[-1].startswith("|request|") and '"wait":true' in seg[-1]:
        return True
    return any(ln.startswith(f"|faint|{opp}a:") for ln in seg)


def intent(segs: Sequence[Sequence[str]], opp: str) -> Optional[str]:
    """The opponent's choice in the update resolving into ``segs[0]`` (module docstring), as a core token; None = a
    stand-in. ``segs``: that segment and every later one. A MID-TURN request (a Baton Pass, a faint, both replaced
    mid-turn in gen 3) splits the opponent's turn across segments, so the scan continues into the next segment while
    the opponent makes no new choice there (:func:`opp_chooses_after`)."""
    for k, seg in enumerate(segs):
        got = _intent_in(seg, opp)
        if got is not None or (k + 1 < len(segs) and opp_chooses_after(seg, opp)):
            return got if got != "" else None
    return None


def _intent_in(seg: Sequence[str], opp: str) -> Optional[str]:
    """The opponent's action in one segment's lines: a token, ``""`` (it was seen NOT to act: a ``cant``), or None
    (nothing about it)."""
    for ln in seg:
        p = ln.split("|")
        if len(p) < 3:
            continue
        who = _mon(p[2])
        if who is None or who[0] != opp:
            continue
        kind = p[1]
        tags = [t for t in p[3:] if t.startswith("[")]
        if kind == "switch":
            return f"switch {who[1]}"
        if kind == "move" and len(p) > 3:
            if any(t.startswith("[from]") for t in tags):
                continue                          # a called move, a lock's continuation: not this turn's choice
            return f"move {toid(p[3])}"
        if kind == "-singleturn" and len(p) > 3 and toid(p[3]) == "movefocuspunch":
            return "move focuspunch"
        if kind == "cant":
            if len(p) > 4 and p[4].strip():
                return f"move {toid(p[4].replace('move: ', ''))}"   # the move it chose and could not use
            return ""                            # asleep, paralysed, flinched, recharging, Truant: a stand-in
    return None


def intents(info: InfoSet) -> List[Optional[str]]:
    segs = info.segments()
    return [None] + [intent(segs[s:], info.opp) for s in range(1, len(segs))]


def backtrack(info: InfoSet) -> List[List[int]]:
    """Per segment, the onsets (earlier segments) of every hidden counter still running there (module docstring)."""
    segs = info.segments()
    #: (side, nick, counter) -> its onset segment, while running
    live: Dict[Tuple[str, str, str], int] = {}
    out: List[List[int]] = []
    pos: Dict[str, Tuple[str, str]] = {}          # position (p1a) -> (side, nick)
    lock_left: Dict[Tuple[str, str], int] = {}
    for s, seg in enumerate(segs):
        out.append(sorted({v for v in live.values() if v < s}))
        for ln in seg:
            p = ln.split("|")
            if len(p) < 3:
                continue
            who = _mon(p[2])
            if who is None:
                continue
            kind, rest = p[1], p[3:]
            tags = [t for t in rest if t.startswith("[")]

            def drop(name: str, w: Tuple[str, str] = who) -> None:
                live.pop((w[0], w[1], name), None)

            if kind in ("switch", "drag"):
                old = pos.get(p[2].split(":")[0])
                if old is not None:
                    for c in ("confusion", "encore", "disable", "trap", "lock", "uproar"):
                        live.pop((old[0], old[1], c), None)
                pos[p[2].split(":")[0]] = who
            elif kind == "faint":
                for c in ("slp", "confusion", "encore", "disable", "trap", "lock", "uproar"):
                    drop(c)
            elif kind == "-status" and rest and rest[0] == "slp" and not any("Rest" in t for t in tags):
                live[(who[0], who[1], "slp")] = s
            elif kind == "-curestatus" and rest and rest[0] == "slp":
                drop("slp")
            elif kind == "-start" and rest:
                what = toid(rest[0])
                if what == "confusion":
                    live[(who[0], who[1], "confusion")] = s
                    drop("lock")
                elif what in ("encore", "disable", "uproar"):
                    live[(who[0], who[1], what)] = s
                elif what in ("movefuturesight", "movedoomdesire", "futuresight", "doomdesire"):
                    live[(who[0], who[1], "future")] = s
            elif kind == "-end" and rest:
                what = toid(rest[0])
                if what in ("confusion", "encore", "disable", "uproar"):
                    drop(what)
                elif what in ("movefuturesight", "movedoomdesire", "futuresight", "doomdesire"):
                    drop("future")
                elif any(toid(t) == what or toid("move: " + t) == what for t in TRAPS):
                    drop("trap")
            elif kind == "-activate" and rest and any(toid(rest[0]) == toid("move: " + t) for t in TRAPS):
                live.setdefault((who[0], who[1], "trap"), s)
            elif kind == "move" and len(rest) > 0 and toid(rest[0]) in LOCKS and not any(t.startswith("[from]") for t in tags):
                live[(who[0], who[1], "lock")] = s
                lock_left[who] = 3
        for w in list(lock_left):
            lock_left[w] -= 1
            if lock_left[w] <= 0:
                live.pop((w[0], w[1], "lock"), None)
                del lock_left[w]
    return out


def team_size(info: InfoSet, side: str) -> Optional[int]:
    """``|teamsize|<side>|N`` (the gen-3 protocol names each team's size at START)."""
    for ln in info.lines:
        p = ln.split("|")
        if len(p) >= 4 and p[1] == "teamsize" and p[2] == side:
            return int(p[3])
    return None


def hp_last(info: InfoSet) -> List[List[List]]:
    """Per segment, ``[ident, [s, …]]`` for each OPPONENT mon: the earlier segments in which it took a damage roll (a
    ``-damage`` line with no ``[from]``, or a confusion self-hit), latest first, since its exact HP was last KNOWN (a
    line showing it at ``100/100`` — the percent fold shows 100 only at full HP — or fainted). The viewer sees that
    mon's HP as a percent, so its exact HP is a die the matcher may re-roll (``pokesim::fresh_dice``); a hit before it
    was last known cannot move it."""
    segs = info.segments()
    last: Dict[str, List[int]] = {}
    out: List[List[List]] = []
    for s, seg in enumerate(segs):
        out.append([[k, sorted(v, reverse=True)] for k, v in sorted(last.items()) if v])
        for ln in seg:
            p = ln.split("|")
            if len(p) < 4 or p[1] not in ("-damage", "-heal", "-sethp", "switch", "drag", "faint"):
                continue
            who = _mon(p[2])
            if who is None or who[0] != info.opp:
                continue
            hp = p[4] if p[1] in ("switch", "drag") and len(p) > 4 else p[3]
            if p[1] == "faint" or hp.split(" ")[0] in ("100/100", "0"):
                last[p[2]] = []                  # its exact HP is known here: earlier hits no longer move it
                continue
            frm = [t for t in p[4:] if t.startswith("[from]")]
            if p[1] == "-damage" and (not frm or toid(frm[0]) == "fromconfusion") and s not in last.get(p[2], []):
                last.setdefault(p[2], []).append(s)
    return out


def own_genders(info: InfoSet) -> str:
    """The viewer's packed team with every gender NAMED from its own first request (``side.pokemon[].details``), so a
    world's construction draws none (D2). A team that already names them all is returned unchanged."""
    rq = next((json.loads(ln[len("|request|"):]) for ln in info.lines
               if ln.startswith("|request|") and '"side"' in ln), None)
    if rq is None:
        raise ValueError("the information set holds no request naming the viewer's team")
    by_name: Dict[str, str] = {}
    for p in rq["side"]["pokemon"]:
        parts = [x.strip() for x in p["details"].split(",")]
        by_name[p["ident"].split(": ", 1)[1]] = next((x for x in parts[1:] if x in ("M", "F")), "N")
    out = []
    for e in info.team.split("]"):
        f = e.split("|")
        f += [""] * max(0, 8 - len(f))
        if not f[7]:
            if f[0] not in by_name:
                raise ValueError(f"the viewer's request names no mon {f[0]!r}")
            f[7] = by_name[f[0]]
        out.append("|".join(f))
    return "]".join(out)
