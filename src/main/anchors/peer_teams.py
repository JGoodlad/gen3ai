"""WHICH TEAM A METAMON PEER PLAYED IN EACH GAME: the per-game ``our_team`` of an anchor-vs-anchor cell.

🚨 **METAMON'S PER-BATTLE CSV "Team File" COLUMN IS NOT THE TEAM PLAYED** (H-CTX review F1, 2026-10-27). Its
``PokeEnvWrapper.step`` writes the row from ``TeamSet.most_recent_team_file`` the moment the battle ends, and poke-env's
challenge loop has by then drawn the team for the NEXT battle on another thread. So the column is a race between two
draws: the next one (it won the race in 1,886 of 2,302 captured pair games, 82 %) or the played one (the rest, and
every battle's last row, which has no next draw). It is never an offset that holds: row ``i`` is draw ``i + 1`` or draw
``i``. 🚨 Before this module ``session.read_peer_battles`` stored that column as ``our_team`` for a cell where BOTH
sides are external peers (Kakuna against Kakuna, Kakuna against PastaMittens: the ``kcal`` kind), so a per-team
attribution read from such a cell's ``games.jsonl`` was off by one in most games.

**The played team is the peer's own draw log** (``team_draws`` in its report: one entry per ``yield_team`` that
returned, in the order the battles were dealt, one battle per draw). That log equalled the captured six on all 2,302
captured pair games it was checked on (the team-strength Kakuna lane, X37a's ``kcal``, X51's pair cell). Here:

* :func:`attribute_played_teams` sets each record's ``our_team`` from the draw log, and THROWS (a named
  :class:`SeriesFailure`) when the log cannot be the record: fewer draws than games, or a CSV row that is neither
  draw ``i`` nor draw ``i + 1`` (a previous attempt's rows appended to the file, a draw that never became a battle);
* :func:`check_against_captures` is the PLAYED-MONS guard: when the front end captured every battle of the half, the
  six species of the peer's side in each START command must equal the six of the team file the log names, or it
  THROWS. A missing capture is not a disagreement, so an incomplete capture set is skipped and the report says so.

Pinned reads play at their pin and are unaffected; the labels of a pair cell played before this module are the CSV's.
"""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from main.anchors.session import BattleRecord, SeriesFailure
from utils.showdown_id import to_id_str

#: ``SeriesFailure.cause`` of a draw log or a team file that disagrees with what was played.
CAUSE = "peer_team_attribution"
#: ``BattleRecord.team_from`` / the row's ``our_team_from``: the label came from the peer's own draw log.
FROM_DRAW_LOG = "peer_draw_log"


def attribute_played_teams(records: Sequence[BattleRecord], draws: Optional[Sequence[str]]) -> List[BattleRecord]:
    """``records`` with ``our_team`` = the team the peer PLAYED, game by game, from its draw log ``draws``.

    ``records`` are :func:`session.read_peer_battles`' (``csv_team`` carries Metamon's racy column). THROWS
    :class:`SeriesFailure` (:data:`CAUSE`) when the log cannot be the record: fewer draws than games, or a CSV row that
    is neither draw ``i`` (the played team) nor draw ``i + 1`` (the next one, drawn first). A peer with no draw log
    (an unseeded run by hand) leaves every record UNATTRIBUTED (``our_team`` None): unknown, never wrong."""
    n = len(records)
    if not draws or n == 0:
        return list(records)
    log = list(draws)
    if len(log) < n:
        raise SeriesFailure(CAUSE, f"{n} games recorded but the peer logged {len(log)} team draws: the draw log cannot "
                                   "be matched to the games, so no per-game team is attributed")
    out: List[BattleRecord] = []
    for i, rec in enumerate(records):
        csv_team = rec.csv_team
        allowed = [log[i]] + ([log[i + 1]] if i + 1 < len(log) else [])
        if csv_team is not None and csv_team not in allowed:
            raise SeriesFailure(
                CAUSE, f"game {i + 1}: Metamon's CSV names {csv_team!r}, which is neither draw {i + 1} ({log[i]!r}, "
                       f"the team played) nor draw {i + 2} ({allowed[-1]!r}, the next): the CSV and the draw log are "
                       "not the same series (a previous attempt's rows appended, or a draw that never became a battle)")
        out.append(replace(rec, our_team=log[i], team_from=FROM_DRAW_LOG))
    return out


def packed_species(packed: str) -> FrozenSet[str]:
    """The six species of a PACKED team (``Name|Species|item|...]...``; an empty species field means the name)."""
    from main.anchors.team_overlap import _species_id

    return frozenset(_species_id(m.split("|")[1] or m.split("|")[0]) for m in packed.split("]") if m)


def captured_sixes(capture_dir: Path, player: str, opponent: str) -> List[FrozenSet[str]]:
    """``player``'s six species in each capture of the battles between ``player`` and ``opponent``, in battle order
    (the captures' numbers). A capture that cannot be read or holds other players is not counted."""
    want = {to_id_str(player), to_id_str(opponent)}
    found: List[Tuple[int, FrozenSet[str]]] = []
    for path in Path(capture_dir).glob("*.json"):
        try:
            start = json.loads(json.loads(path.read_text())["commands"][0].split(" ", 1)[1])
            sides = {to_id_str(start[s]["name"]): s for s in ("p1", "p2")}
            if set(sides) != want:
                continue
            found.append((int((re.search(r"(\d+)$", path.stem) or [0, "0"])[1]),
                          packed_species(start[sides[to_id_str(player)]]["team"])))
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
            continue
    return [six for _n, six in sorted(found, key=lambda x: x[0])]


def check_against_captures(records: Sequence[BattleRecord], sixes: Sequence[FrozenSet[str]],
                           species_of: Callable[[str], Optional[FrozenSet[str]]]) -> None:
    """THE PLAYED-MONS GUARD: game ``i``'s attributed team file must hold the six species the peer's side had in that
    battle's capture (``sixes[i]``). THROWS :class:`SeriesFailure` (:data:`CAUSE`) on the first game where it does not,
    and when a named team file cannot be read (a team that cannot be compared is not a pass)."""
    for i, (rec, played) in enumerate(zip(records, sixes)):
        if rec.our_team is None:
            continue
        have = species_of(rec.our_team)
        if have is None:
            raise SeriesFailure(CAUSE, f"game {i + 1}: team file {rec.our_team!r} cannot be read, so the draw log's "
                                       "team cannot be checked against the captured six")
        if have != played:
            raise SeriesFailure(
                CAUSE, f"game {i + 1}: the draw log says the peer played {rec.our_team!r} "
                       f"({sorted(have)}), but the capture's START holds {sorted(played)}")


def attribute_pair_half(records: Sequence[BattleRecord], *, draws: Optional[Sequence[str]], our_name: str,
                        peer_name: str, capture_dir: Optional[Path], team_dir: Optional[Path],
                        ) -> Tuple[List[BattleRecord], Dict[str, Any]]:
    """One pair-cell half: ``(records with the played team, report fields)``; THROWS as the two guards do.

    The report fields (``our_team_attribution`` / ``our_team_capture_check``) say how each label was obtained and
    whether the played-mons guard ran, so a summary never leaves the question to the reader."""
    out = attribute_played_teams(records, draws)
    if not out:
        return out, {"our_team_attribution": "no games", "our_team_capture_check": "skipped: no games"}
    if not out[0].team_from:
        return out, {"our_team_attribution": "unattributed: the peer logged no team draws",
                     "our_team_capture_check": "skipped: no attributed team"}
    info: Dict[str, Any] = {"our_team_attribution": FROM_DRAW_LOG}
    if capture_dir is None or team_dir is None:
        info["our_team_capture_check"] = "skipped: no capture dir" if capture_dir is None else "skipped: no team dir"
        return out, info
    sixes = captured_sixes(capture_dir, our_name, peer_name)
    if len(sixes) != len(out):
        info["our_team_capture_check"] = f"skipped: {len(sixes)} captures for {len(out)} games"
        return out, info
    from main.anchors.team_overlap import parse_team

    def species_of(label: str) -> Optional[FrozenSet[str]]:
        try:
            return parse_team((Path(team_dir) / label).read_text(), label).species6
        except OSError:
            return None

    check_against_captures(out, sixes, species_of)
    info["our_team_capture_check"] = f"checked {len(out)} of {len(out)} games"
    return out, info
