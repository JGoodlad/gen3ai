"""The BOARD read-model — decoding the recorder's per-side strings into `BoardView`."""

from __future__ import annotations

import re

from agents.action.constants import MOVE_END, MOVE_START

from main.prober.engine.util import _norm_species
from main.prober.engine.views import BoardView, MonState, SideBoard


# Recorder bench format: "species(hp%)" / "species(hp%,STATUS)" / "species(faint)"
# where STATUS bundles status+volatiles (e.g. "TOX(5)", "PAR|SUB") — see battle_recorder.
_BENCH_RE = re.compile(r"^(.+?)\((.+)\)$")          # "metagross(100%)" / "tyranitar(faint)"
_MOVE_PLACEHOLDER_RE = re.compile(r"move\d$")        # "move0".."move3" filler labels


def _team_entry(team: dict, species: str) -> dict:
    """Per-mon {item, moves} from a leniently-keyed team map ({} when absent)."""
    return team.get(_norm_species(species), {}) if team else {}


def build_our_hp_types(team_details: "list[dict] | None") -> "dict[str, str]":
    """`{norm_species: 'hiddenpower(bug)'}` from a reconstruction `team_details` list — the typed
    Hidden Power display id for each own mon, so a bare own HP (all the request carries before reveal)
    can be typed. Pure; the caller does the reconstruction-file IO (mirrors the opp-team load)."""
    out: "dict[str, str]" = {}
    for m in team_details or []:
        for mv in m.get("moves", ()) or ():
            s = str(mv)
            if s.startswith("hiddenpower") and s != "hiddenpower":
                out[_norm_species(m.get("species", ""))] = f"hiddenpower({s[len('hiddenpower'):]})"
    return out


def _retype_hp(moves: "tuple[str, ...]", species: str, hp_map: "dict | None") -> "tuple[str, ...]":
    """Replace a bare ``hiddenpower`` in a KNOWN-OWN moveset with its true typed display form
    (``hiddenpower(bug)``) from ``hp_map`` (norm-species → formatted id, built from the reconstruction
    record). Showdown's request carries only the bare id for an UNREVEALED own Hidden Power (the type
    is IV-derived, not in the request), so without this our own mons show an untyped HP until they use
    it. ``hp_map`` is None for the OPPONENT side and for non-reconstruction traces — an opponent's
    un-revealed HP MUST stay bare (no leak), so the retype only ever runs on our own team."""
    typed = (hp_map or {}).get(_norm_species(species))
    if not typed:
        return tuple(moves)
    return tuple(typed if str(m) == "hiddenpower" else m for m in moves)


def _parse_bench(s: str, team: "dict | None" = None, hp_map: "dict | None" = None) -> "tuple[MonState, ...]":
    team = team or {}
    out = []
    for chunk in (s or "").split(", "):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = _BENCH_RE.match(chunk)
        if m:
            species, inside = m.group(1), m.group(2)
            e = _team_entry(team, species)
            item, moves = e.get("item", ""), _retype_hp(tuple(e.get("moves", ())), species, hp_map)
            if "faint" in inside.lower():
                out.append(MonState(species, "faint", True, "", item, moves))
            else:
                # "hp%[,STATUS]" — the status tail (incl. any volatiles) is comma-separated.
                hp, _, status = inside.partition(",")
                out.append(MonState(species, hp.strip(), False, status.strip(), item, moves))
        else:
            e = _team_entry(team, chunk)
            out.append(MonState(chunk, "?", False, "", e.get("item", ""),
                                _retype_hp(tuple(e.get("moves", ())), chunk, hp_map)))
    return tuple(out)


def _side_board(side: dict, moves: "tuple[str, ...]", team: "dict | None" = None,
                hp_map: "dict | None" = None) -> SideBoard:
    team = team or {}
    species = side.get("species", "")
    e = _team_entry(team, species)
    return SideBoard(
        active_species=species,
        active_hp=side.get("hp", "?"),
        status=side.get("status", "") or "",
        boosts=side.get("boosts", "") or "",
        # our active's moves come from the trace actions; the opp active's (and any side with no
        # trace moves) fall back to the obs-decoded revealed moveset. `hp_map` (our side only) types
        # a bare own Hidden Power the request couldn't.
        moves=_retype_hp(moves or tuple(e.get("moves", ())), species, hp_map),
        bench=_parse_bench(side.get("bench", ""), team, hp_map),
        item=e.get("item", ""),
    )


def _our_items(summary: dict) -> "dict[str, dict]":
    """species → {item, moves:()} from the summary's top-level teams block (our side only; opp
    items aren't in it; moves aren't in it at all). Items recorded at battle end — held items
    (Choice Band / Leftovers) are exact; a consumed berry reads 'none' (dropped to ''). The obs
    decode (`describe_team`) supersedes this per-turn when a state is captured — no-state fallback."""
    out = {}
    for m in (summary.get("teams", {}) or {}).get("ours", []) or []:
        item = str(m.get("item", "") or "")
        if item and item.lower() != "none":
            out[str(m.get("species", ""))] = {"item": item, "moves": ()}
    return out


def _merge_team(base: dict, obs_team: dict) -> "dict[str, dict]":
    """Overlay the per-turn obs decode onto the summary-derived team, keyed leniently by species.
    The obs only OVERRIDES a field when it carries info — an empty obs item must NOT erase a known
    item from the summary teams block (the bug where our own bench mon showed no item because its
    obs slot decoded blank). Returns ``{norm_species: {item, moves}}``."""
    out: "dict[str, dict]" = {}
    for sp, e in (base or {}).items():
        out[_norm_species(sp)] = {"item": e.get("item", ""), "moves": tuple(e.get("moves", ()))}
    for sp, e in (obs_team or {}).items():
        cur = out.setdefault(_norm_species(sp), {"item": "", "moves": ()})
        if e.get("item"):
            cur["item"] = e["item"]
        if e.get("moves"):
            cur["moves"] = tuple(e["moves"])
    return out


def build_board(inv: dict, team: "dict | None" = None,
                our_hp_types: "dict | None" = None) -> BoardView:
    """Board state at a decision — model-free, parsed from the summary invocation. ``team``
    (species → {item, moves}) annotates BOTH sides; keys are matched leniently (:func:`_norm_species`)
    so an obs-decoded name resolves against a board id. ``our_hp_types`` (norm-species → typed HP
    display id, from the reconstruction record) types a bare own Hidden Power — OUR side only."""
    norm = {_norm_species(k): v for k, v in (team or {}).items()}
    labels = list(inv.get("actions", {}).keys())
    moves = tuple(k for k in labels[MOVE_START:MOVE_END] if not _MOVE_PLACEHOLDER_RE.fullmatch(k))
    return BoardView(ours=_side_board(inv.get("our", {}), moves, norm, our_hp_types),
                     opp=_side_board(inv.get("opp", {}), (), norm))
