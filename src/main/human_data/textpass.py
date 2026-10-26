"""The rowset's TEXT PASS over one public log: who played, how it ended, and what each side REVEALED (no reader).

Everything a battle-level filter of the plan's §3 needs, computed from the log's text alone, before anything is
completed or labelled (``designs/endstate/design_human_replay_lever.md`` §3; the data card ``hrq.py``'s ``parse``,
whose fields and constants this mirrors so the two count the same things):

* :func:`record` — players, ratings, the turn count, the ending (``ko`` / ``forfeit`` / ``timer`` / ``tie`` / ``none``)
  and the quitter, the first timestamp, a room-line-free body hash (duplicates), and per seat the REVEALED facts
  (``agents.info_levers.revealed``, each side read from the other seat, as ``main.x50_r1.memory`` does).
* :class:`PanelMatcher` — the ONE panel-exclusion rule (plan §3.1, REVIEW A B1 / B2): a side is NEAR a held-out panel
  team when >= :data:`NEAR` of its mons with at least one revealed move FIT that team's sets (``memory._fits``: the
  species, every revealed move, the revealed original item and ability). It runs on revealed facts, before C-GAME
  fills anything, over EVERY panel the registry holds out from training (the sealed ``forum216b`` included: its
  content is read for the exclusion only, which ``heldout_panels.panel_team_texts`` allows).
* :func:`bot_flags` — the card's bot rule (a bot-like name; >= 100 games on one Los Angeles date; games started in
  >= 20 distinct clock hours of one date).
* :func:`acct` — an account's id in every committed artefact: a salted hash, never the name (REVIEW C F13).
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: the panel-exclusion threshold (plan §3.1): >= 3 move-revealed mons fitting ONE held-out team
NEAR = 3
#: the card's bot rule (``hrq.py``; REVIEW A): names, volume, round-the-clock hours
BOT_NAME = re.compile(r"(^pac[- ]|^h2h[- ]|metamon|pastamittens|exeggcute|kakuna|foul ?play|gen3ai|\bbot\b|bot$|^bot)",
                      re.I)
BOT_GAMES_PER_DAY = 100
BOT_HOURS_PER_DAY = 20
_PLAYER = re.compile(r"^\|player\|(p[12])\|([^|]*)\|([^|]*)\|(\d*)")
_FORFEIT = re.compile(r"^\|-message\|(.*) forfeited\.$")
_TIMER = re.compile(r"^\|-message\|(.*) lost due to inactivity\.$")
_SKIP_BODY = ("|init|", "|title|", "|j|", "|J|", "|l|", "|L|", "|c|", "|raw|", "|inactive|", "|n|", "|N|", "|t:|")
SALT_PATH = Path.home() / "gen3ai_archive" / "human_snapshots" / "salt"


def toid(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def acct(name: str, salt: bytes) -> str:
    """The salted account id (``hrq.acct``'s definition: the same id for the same name and salt)."""
    return "acct_" + hashlib.sha256(salt + b":" + toid(name).encode()).hexdigest()[:12]


def load_salt(path: Path = SALT_PATH) -> bytes:
    if not path.exists():
        raise FileNotFoundError(f"no account salt at {path}: committed artefacts never carry account names")
    return path.read_text().strip().encode()


def la_date_hour(t: Optional[int]) -> Tuple[Optional[str], Optional[int]]:
    """Los Angeles date and hour (PDT across the snapshot's window, as the card)."""
    if t is None:
        return None, None
    d = dt.datetime.fromtimestamp(t, dt.timezone(dt.timedelta(hours=-7)))
    return d.date().isoformat(), d.hour


def record(text: str) -> Optional[Dict[str, Any]]:
    """One battle's text record (module docstring); None when the log has no ``|player|`` line."""
    from agents.info_levers import revealed

    lines = text.split("\n")
    rec: Dict[str, Any] = {"rated": False, "turns": 0, "end": "none", "win": None, "quit": None, "t0": None}
    names: Dict[str, str] = {}
    rating: Dict[str, Optional[int]] = {"p1": None, "p2": None}
    body = hashlib.sha1()
    winner = None
    for ln in lines:
        if not ln.startswith("|"):
            continue
        if ln.startswith("|t:|"):
            if rec["t0"] is None:
                try:
                    rec["t0"] = int(ln[4:])
                except ValueError:
                    pass
            continue
        if ln.startswith("|c|"):
            continue
        if not ln.startswith(_SKIP_BODY):
            body.update(ln.encode())
        if ln.startswith("|turn|"):
            try:
                rec["turns"] = int(ln[6:])
            except ValueError:
                pass
        elif ln.startswith("|player|"):
            m = _PLAYER.match(ln)
            if m and m.group(2) and m.group(1) not in names:
                names[m.group(1)] = m.group(2)
                rating[m.group(1)] = int(m.group(4)) if m.group(4) else None
        elif ln.startswith("|rated|"):
            rec["rated"] = True
        elif ln.startswith("|-message|"):
            m = _FORFEIT.match(ln) or _TIMER.match(ln)
            if m:
                rec["end"] = "forfeit" if "forfeited" in ln else "timer"
                q = toid(m.group(1))
                rec["quit"] = next((s for s, n in names.items() if toid(n) == q), None)
        elif ln.startswith("|win|"):
            winner = ln[5:]
            if rec["end"] == "none":
                rec["end"] = "ko"
            break
        elif ln == "|tie" or ln.startswith("|tie|"):
            rec["end"] = "tie"
            break
    if not names:
        return None
    if rec["end"] in ("forfeit", "timer") and winner is None:
        rec["end"], rec["quit"] = "none", None
    rec["win"] = next((s for s, n in names.items() if n == winner), None) if winner else None
    rec["names"] = names
    rec["rating"] = rating
    rec["body"] = body.hexdigest()[:16]
    rec["reveals"] = {}
    for seat, viewer in (("p1", 1), ("p2", 0)):               # the OTHER seat's view of this one (memory.py)
        rv = revealed.from_lines(lines, viewer)
        rec["reveals"][seat] = {revealed.toid(k): (sorted(revealed.toid(x) for x in v.moves), v.item, v.ability)
                                for k, v in rv.mons.items()}
    return rec


def fits(moves: Sequence[str], item: Optional[str], ability: Optional[str], ms: Any) -> bool:
    """``main.x50_r1.memory._fits`` with the species already matched: every revealed move in the set (a bare Hidden
    Power fits any typed one; Struggle is no reveal), the revealed original item and ability equal."""
    for mv in moves:
        if mv == "struggle":
            continue
        if mv == "hiddenpower":
            if not any(x.startswith("hiddenpower") for x in ms.moves):
                return False
        elif mv not in ms.moves:
            return False
    if item is not None and item != ms.item:
        return False
    if ability is not None and ability != ms.ability:
        return False
    return True


class PanelMatcher:
    """The held-out panels' teams as fit references (module docstring). ``panels`` defaults to every panel the
    registry holds out from training."""

    def __init__(self, panels: Optional[Sequence[str]] = None):
        from main import heldout_panels as HP
        from main.anchors.team_overlap import _species_id, parse_team

        self._sid = _species_id
        self.panels = sorted(panels if panels is not None else HP.panel_names(held_out_only=True))
        #: canonical species -> [(panel, team index, MonSet)]
        self.index: Dict[str, List[Tuple[str, int, Any]]] = collections.defaultdict(list)
        self.sizes: Dict[str, int] = {}
        for name in self.panels:
            items = HP.panel_team_texts(name)
            self.sizes[name] = len(items)
            for i, (lab, txt) in enumerate(items):
                for ms in parse_team(txt, lab).mons:
                    self.index[ms.species].append((name, i, ms))
        self._canon: Dict[str, str] = {}

    def canon(self, sid: str) -> str:
        c = self._canon.get(sid)
        if c is None:
            c = self._canon[sid] = self._sid(sid)
        return c

    def side_max_fits(self, mons: Dict[str, Tuple[List[str], Optional[str], Optional[str]]]) -> Dict[str, int]:
        """Per panel, the most move-revealed mons of this side that fit ONE of its teams (0 when none)."""
        count: Dict[Tuple[str, int], int] = collections.Counter()
        for sid, (moves, item, ability) in mons.items():
            if not [m for m in moves if m != "struggle"]:
                continue                                      # only mons with a revealed move count (§3.1)
            for (panel, i, ms) in self.index.get(self.canon(sid), ()):
                if fits(moves, item, ability, ms):
                    count[(panel, i)] += 1
        out = {p: 0 for p in self.panels}
        for (panel, _), n in count.items():
            out[panel] = max(out[panel], n)
        return out


def accounts(recs: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per account (``toid`` of the name): its name, game count, games per LA date and clock hours per date."""
    per: Dict[str, Dict[str, Any]] = {}
    for r in recs:
        d, h = la_date_hour(r.get("t0"))
        for seat, n in r["names"].items():
            a = per.setdefault(toid(n), {"name": n, "games": 0, "days": collections.Counter(),
                                         "hours": collections.defaultdict(set)})
            a["games"] += 1
            if d:
                a["days"][d] += 1
                a["hours"][d].add(h)
    return per


def bot_flags(per: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """The card's bot rule per account key: ``name`` / ``volume`` / ``hours`` (empty = clean, omitted)."""
    out: Dict[str, List[str]] = {}
    for k, a in per.items():
        why = []
        if BOT_NAME.search(a["name"]):
            why.append("name")
        if a["days"] and max(a["days"].values()) >= BOT_GAMES_PER_DAY:
            why.append("volume")
        if a["hours"] and max(len(v) for v in a["hours"].values()) >= BOT_HOURS_PER_DAY:
            why.append("hours")
        if why:
            out[k] = why
    return out
