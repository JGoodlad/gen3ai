"""ASSEMBLE: the played units and ledger rows -> CELLS, each with its regime id (``designs/tools/reads.md`` §4).

THE REGIME ID IS ALWAYS RECORDED. Five S drivers wrote ``regime_ids: [""]`` because the ledger's ``Cell`` carries no
regime and they read it with ``getattr(c, "regime_id", "")`` (x26c FINDING 4, g1 FINDING 3); the regime was then
checked by hand on the raw rows. Here:

* an S group's regime is the eval-ledger READ's own (``LedgerRead.regime_id``: one family = one regime, a mixed
  family is the reader's typed refusal), recorded with the read's digest; a spec that registers a ``regime_id`` and a
  ledger that holds another, or a panel whose team set is not the rows', makes the group INCONCLUSIVE;
* an anchor unit's regime is :func:`anchor_regime_id` over the cell identity its ``summary.json`` stamps (opponent
  and version, both regimes, transport, server, format, forfeit limit, both team-content digests, the declared team
  cell, and the ENGINE, :mod:`main.reads.engine`): every counted unit of a (group, team set) must share ONE, else the
  group is INCONCLUSIVE, and the group's id is :func:`group_regime_id` over its team sets' ids. The playing commit
  (``server_version``) and file paths are recorded per unit, not part of the id; the engine the commit played on is;
* a group may hold models with DIFFERENT load options (``spec.models.<label>.load``, e.g. a belief prior mix λ): each load
  stamps its own ``our_regime`` (``greedy+belief_prior_mix=0.25``), so the group's id is then taken per (load, team set)
  AND over the BASE id (``our_regime`` without the load's stamp) that every model of a team set must share (see
  :func:`anchor_group`). A group whose models share one load keeps the id it always had.

THE ENGINE (REVIEW F F-M2, 2026-10-26). Every unit row and group carries its ``engine`` (the Rust build-stamp
identity it played on) and a group whose counted units span two engines is INCONCLUSIVE, as an S group whose rows'
core stamps span two is. A summary that stamps its engine (``cell.engine_stamp``) carries it in the regime id unless
it is an engine a banked read played on (:data:`main.reads.engine.LEGACY_ENGINE`), so every banked id is unchanged
(the LEGACY rule, as the weights pins did in ff54265e) and a new read on a banked engine keeps that engine's id. Ids
alone cannot tell two banked engines apart (they were computed without one), so every import compares the engines
themselves (:func:`main.reads.engine.require_same`).

THE SIDES (TECH_DEBT P1 (b), 2026-10-27): the Metamon peer's code and our policy's load mode, by the same LEGACY rule
(:mod:`main.reads.side_identity`). Every unit row and group carries ``peer_code`` and ``policy_mode``; a group whose
counted units span two is INCONCLUSIVE, and every import compares them.

A cell or node with no regime id is never conclusive.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agents.training import eval_ledger as L
from main.reads import attempts as AT
from main.reads import engine as EN
from main.reads import side_identity as SI

ANCHOR_REGIME_SCHEMA = "gen3_read_anchor_regime_v1"
#: The cell fields of an anchor summary that ARE the regime (everything else is recorded, not identity).
ANCHOR_REGIME_KEYS = ("battle_format", "forfeit_turn_limit", "mirrored_pairs", "opponent", "opponent_kind",
                      "opponent_version", "our_regime", "their_regime", "regime_matched", "our_transport",
                      "server_impl", "our_side", "model_loader", "our_teamset", "teamset", "our_team_count",
                      "their_team_count", "our_team_digest", "their_team_digest", "search_time_ms",
                      "search_parallelism", "showdown_pin")
CELLS_SCHEMA = "gen3_read_cells_v1"
PROTOCOL = "gen3_eval_protocol_v1_h2h"

#: The Metamon weights EVERY anchor read banked before the pins existed (FR-1 FINDING 3, 2026-10-25) played: the one
#: hub commit in the box's cache, ``{agent: weights_sha256}`` (``main.anchors.peer_scripts.weight_pins.weights_digest``
#: of the agent's ``{path: sha256}``, each file hashed against its blob). Those summaries stamp no weights, so their
#: regime ids were computed without them. A NEW summary whose verified weights equal this entry keeps that SAME id (the
#: weights are the ones the id always meant); any other revision or digest enters the id (:func:`anchor_regime_id`),
#: so different weights are a different regime. FROZEN: a repin never edits it.
LEGACY_HF_REVISION = "ac8e88e7da8dc2a2b85176d88f81fd8854b1445b"
LEGACY_WEIGHTS: Dict[str, str] = {
    "SmallRL": "b8849c52b443d7a1607c34e806dd959baa89c254aa0737c8bbdb8160a0f96ce2",
    "SyntheticRLV2": "3f00c9cbabe6e6bc776f3bbecdc9c134537704c7cc45c7f628fdda4d1e045961",
    "Kakuna": "bbfa8f2d9425341f0299c721ea847b7d74170cb72b80440951861efb4643992b",
    "Kakuna28": "7e4c71c28905d31c56cd8bb933c19eaac8641ade81fb3b64eb178181a38cf288",
    "LargeRL": "b26296d4963eb3d642ee7edcd16f069175bbc22083fa9e71a77addb8b7e82098",
    "LargeIL": "2167e509ef6e6123b8dbe8b69fe86a966d9781fc9d4c9c250d55d442a83aa02d",
    "PastaMittens": "b9f2c07e051c9a69d199484f919babc8b160ef7486cac578874fa22aa982e303",
}

#: S's look read: an A/B family's mirrored greedy-vs-greedy rows at the h2h protocol (``main.h2h.cross``'s terms).
S_FAMILY_READ = L.ReaderDecl(
    name="main.reads.assemble.s_family", purposes=frozenset({"ab"}),
    regime=L.RegimeFilter(protocol="gen3_eval_protocol_v1_h2h", play="greedy", opponent_play="greedy", mirrored=True),
    requests="family", selection="include", flags_ok=frozenset(), inference="across_runs",
    decision_kind="ab_verdict")


class AssembleError(RuntimeError):
    """The state cannot be assembled as this read (named)."""


# ------------------------------------------------------------------------------------------------ anchor
def anchor_regime_id(summary: Mapping[str, Any], *, strip_load: str = "") -> Optional[str]:
    """The regime id of one anchor unit (16 hex), or ``None`` when its summary stamps no cell. ``strip_load``: the
    ``our_regime`` suffix the unit's model-load options add (``spec.load_stamp``); given, the id is the unit's BASE id,
    the one it would have without them (``None`` when the unit's ``our_regime`` does not end with it: a unit that
    declared a load and did not stamp it)."""
    cell = summary.get("cell")
    if not isinstance(cell, dict) or not cell.get("opponent_version"):
        return None
    ident = {k: cell.get(k) for k in ANCHOR_REGIME_KEYS}
    if strip_load:
        ours = str(cell.get("our_regime") or "")
        if not ours.endswith(strip_load):
            return None
        ident["our_regime"] = ours[:-len(strip_load)]
    ident.update(_weights_identity(cell))
    ident.update(peer_pack_identity(cell))
    ident.update(SI.regime_identity(cell))                 # the peer's code, our load mode: each only when not legacy
    stamped = cell.get("engine_stamp")
    if stamped and not EN.is_legacy(str(stamped)):
        ident["engine"] = str(stamped)                     # a banked engine keeps its id (the LEGACY rule)
    ident.update(schema=ANCHOR_REGIME_SCHEMA, declared_team_cell=summary.get("declared_team_cell"),
                 team_digest_schema=summary.get("team_digest_schema"))
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]


def peer_pack_identity(cell: Mapping[str, Any]) -> Dict[str, Any]:
    """The team-pack inputs of a regime id, each only when True (``EXTERNAL_ANCHORS_SOP.md`` H22):
    ``peer_pack`` / ``our_peer_pack`` = ``PEER_PACK_SCHEMA`` when a Metamon peer's team set holds a file that plays
    differently from the legacy poke-env pack (gen3_peer_showdown_pack_v1); ``frontend_pack`` / ``our_frontend_pack`` =
    ``FRONTEND_PACK_SCHEMA`` when a Foul Play side's team set holds a file the front end now plays differently, because it
    plays the validator's re-pack (gen3_frontend_validated_pack_v1). ``{}`` otherwise, so a team set played identically
    keeps its banked id, and a changed one (forum72 / away / the pool for Kakuna; Foul Play's ``gen3ai_pool`` and ``away``)
    never shares an id with its banked reads."""
    from main.anchors.frontend_pack import FRONTEND_PACK_SCHEMA
    from main.anchors.peer_scripts.team_pack import PEER_PACK_SCHEMA

    fields = (("peer_pack", "peer_pack_changed", PEER_PACK_SCHEMA),
              ("our_peer_pack", "our_peer_pack_changed", PEER_PACK_SCHEMA),
              ("frontend_pack", "frontend_pack_changed", FRONTEND_PACK_SCHEMA),
              ("our_frontend_pack", "our_frontend_pack_changed", FRONTEND_PACK_SCHEMA))
    return {key: schema for key, field, schema in fields if cell.get(field) is True}


def peer_pack_state(cell: Mapping[str, Any]) -> str:
    """A unit's team-pack regime, for the import check (``main.reads.imports``): ``"changed"`` when its Metamon peer stamps
    a team set the legacy pack played differently, ``"frontend_changed"`` when its Foul Play side does (both, joined by
    ``+``), else ``"legacy"`` (the same play as every banked unit; a unit before the pack, or one whose files are
    unchanged)."""
    keys = peer_pack_identity(cell)
    parts = [name for name, hit in (("changed", any(k in keys for k in ("peer_pack", "our_peer_pack"))),
                                    ("frontend_changed", any(k in keys for k in ("frontend_pack", "our_frontend_pack"))))
             if hit]
    return "+".join(parts) if parts else "legacy"


def _weights_identity(cell: Mapping[str, Any]) -> Dict[str, Any]:
    """The weight-identity inputs of a regime id: ``{"their_weights": {...}}`` / ``{"our_weights": {...}}`` for each
    Metamon side whose verified (revision, sha256) is NOT :data:`LEGACY_WEIGHTS`; ``{}`` for a summary that stamps none
    (a pre-pin read) or stamps the legacy weights, so every banked id is unchanged and a re-read of the same weights
    is comparable to it."""
    out: Dict[str, Any] = {}
    sides: Tuple[Tuple[str, Any, Any, Any], ...] = (
        ("their", cell.get("opponent_agent"), cell.get("hf_revision"), cell.get("weights_sha256")),
        ("our", str(cell.get("our_side") or "").partition("metamon:")[2], cell.get("our_hf_revision"),
         cell.get("our_weights_sha256")))
    for prefix, agent, revision, digest in sides:
        if not (revision and digest):
            continue
        if revision == LEGACY_HF_REVISION and LEGACY_WEIGHTS.get(agent) == digest:
            continue
        out[f"{prefix}_weights"] = {"hf_revision": revision, "weights_sha256": digest}
    return out


def group_regime_id(by_teamset: Mapping[str, Optional[str]]) -> str:
    """An anchor GROUP's one id: the digest of its per-team-set regime ids (each team set is its own cell, with its
    own team-content digests, so a group spans one regime per team set)."""
    return hashlib.sha256(json.dumps(dict(by_teamset), sort_keys=True).encode()).hexdigest()[:16]


MIXED_GROUP_SCHEMA = "gen3_read_anchor_group_regime_mixed_v1"


def mixed_group_regime_id(base_by_teamset: Mapping[str, str], full_by_load: Mapping[str, Mapping[str, str]]) -> str:
    """The id of an anchor group whose models carry DIFFERENT load options: the digest of its per-team-set BASE ids
    (shared by every model) and of each load's per-team-set full ids (``{load stamp or "none": {team set: id}}``)."""
    doc = {"schema": MIXED_GROUP_SCHEMA, "base": dict(base_by_teamset),
           "by_load": {k: dict(v) for k, v in full_by_load.items()}}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]


#: A unit's per-game identity (``games.jsonl``): what a determinism guard compares game for game.
GAME_KEYS = ("half", "index", "our_team", "result", "turns")


def unit_games(kakuna_state: Path, name: str) -> Optional[List[List[Any]]]:
    """``[[half, index, our_team, result, turns], ...]`` of one played unit, in ``(half, index)`` order; ``None`` when
    the unit wrote no ``games.jsonl``."""
    p = kakuna_state / "cells" / name / "games.jsonl"
    try:
        rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    except (OSError, ValueError):
        return None
    return sorted(([r.get(k) for k in GAME_KEYS] for r in rows), key=lambda g: (str(g[0]), int(g[1])))


#: The stricter game-for-game identity (``measurements/kakuna_determinism_2026-10-27/`` §6): per game, the digest of
#: each player's CHOOSE sequence in the front end's capture. It holds OUR choices, so a choice that differs but leaves
#: the 5-tuple equal (a different move that ends in the same result and turn count, or one that never ran) is caught:
#: 15 divergences where the 5-tuple sees 11 (E1 vs E3). The two sides' choices are kept apart and their interleaving
#: is ignored, so an ORDER-ONLY game (the same choices racing to the server) stays equal.
CHOICES_SCHEMA = "gen3_game_choices_v1"


def game_choices(capture: Mapping[str, Any]) -> str:
    """The choices digest of one captured battle (``{tag, seed, commands, chunks}``): sha256 over each player's
    ``CHOOSE`` strings, in order, per player."""
    per: Dict[str, List[str]] = {"p1": [], "p2": []}
    for c in capture.get("commands") or []:
        parts = str(c).split(" ", 2)
        if parts[0] == "CHOOSE" and len(parts) == 3:
            per.setdefault(parts[1], []).append(parts[2])
    doc = {"schema": CHOICES_SCHEMA, "choose": per}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]


def cell_choices(cell: Path) -> Tuple[Optional[List[List[Any]]], Optional[List[str]], str]:
    """``(games, choices, why)`` of ONE played cell dir: its :data:`GAME_KEYS` rows and, aligned with them in
    ``(half, index)`` order, each game's :func:`game_choices` from ``capture/<battle_tag>.json``. ``choices`` is None
    (and ``why`` says why) when the cell has no ``games.jsonl``, or any game has no tag or no capture: a guard never
    compares a partial choice record."""
    try:
        rows = [json.loads(ln) for ln in (cell / "games.jsonl").read_text().splitlines() if ln.strip()]
    except (OSError, ValueError):
        return None, None, f"no readable games.jsonl in {cell}"
    rows.sort(key=lambda r: (str(r.get("half")), int(r.get("index") or 0)))
    games = [[r.get(k) for k in GAME_KEYS] for r in rows]
    out: List[str] = []
    for r in rows:
        tag = r.get("battle_tag")
        cap = cell / "capture" / f"{tag}.json"
        if not tag or not cap.is_file():
            return games, None, f"no capture for game {r.get('half')}/{r.get('index')} in {cell}"
        try:
            out.append(game_choices(json.loads(cap.read_text())))
        except (OSError, ValueError) as exc:
            return games, None, f"unreadable capture {cap}: {exc}"
    return games, out, ""


#: :func:`compare_choices`' result when a capture the comparison EXPECTS cannot be read (review H2): the guard is
#: INCONCLUSIVE, never a silent fall back to the 5-tuple. ``unavailable`` is kept for a bank that names no single
#: capture source (a property of the registration, e.g. X51's rows) and for a spec's ``choices_guard: five_tuple``.
CHOICES_INCONCLUSIVE = "inconclusive"


def compare_choices(here: Optional[Path], there: Optional[Path], want_games: Sequence[Sequence[Any]],
                    unavailable: str = "", *, five_tuple_ok: bool = False) -> Dict[str, Any]:
    """The choices half of a game-for-game guard: ``{choices: equal | differ | inconclusive | unavailable, differing,
    why}``.

    ``here`` is the re-played unit's cell dir, ``there`` the banked unit's (None: the bank names no capture source,
    ``unavailable`` says why; the 5-tuple then decides alone). The banked cell must hold exactly ``want_games`` (the
    bank's own rows), so a capture source that is not the banked unit is never compared. ``differing`` lists each game
    ``[half, index]`` whose choices differ.

    FAIL CLOSED (review H2): once the bank names its source, a capture missing on EITHER side (the source moved or
    pruned, ``$GEN3AI_ARCHIVE_DIR`` elsewhere, the re-played unit's capture off or lost), or a source that holds other
    games, is :data:`CHOICES_INCONCLUSIVE`, so one registered decision never passes or fails by the environment.
    ``five_tuple_ok`` (the spec's ``choices_guard: five_tuple``) turns that into ``unavailable``, recorded."""
    def missing(why: str) -> Dict[str, Any]:
        if five_tuple_ok:
            return {"choices": "unavailable", "differing": [],
                    "why": f"{why} (the spec registers choices_guard = five_tuple: the 5-tuple decides alone)"}
        return {"choices": CHOICES_INCONCLUSIVE, "differing": [],
                "why": f"{why}: the choice comparison expects this capture (INCONCLUSIVE, never the 5-tuple alone)"}

    if there is None:
        return {"choices": "unavailable", "differing": [], "why": unavailable or "the bank names no capture source"}
    if here is None:
        return missing("the re-played unit has no cell dir")
    g_here, c_here, why_here = cell_choices(here)
    g_there, c_there, why_there = cell_choices(there)
    if g_there is not None and [list(g) for g in g_there] != [list(g) for g in want_games]:
        return missing(f"the capture source {there} does not hold the banked unit's games")
    if c_here is None or c_there is None:
        return missing(why_here or why_there)
    if g_here is None or [g[:2] for g in g_here] != [g[:2] for g in g_there or []]:
        return {"choices": "differ", "differing": [], "why": "the two units do not hold the same games"}
    diff = [list(g[:2]) for g, a, b in zip(g_here, c_here, c_there) if a != b]
    return {"choices": "differ" if diff else "equal", "differing": diff, "why": ""}


def five_tuple_ok(cells: Mapping[str, Any]) -> bool:
    """Whether the read's spec registered ``choices_guard: five_tuple`` (recorded in the cells by :func:`assemble`)."""
    from main.reads.spec import CHOICES_FIVE_TUPLE

    return (cells.get("guards") or {}).get("choices") == CHOICES_FIVE_TUPLE


def unit_row(u: Any, summary: Mapping[str, Any], games: Optional[List[List[Any]]] = None, *,
             cell_dir: Optional[Path] = None, rederived: Optional[str] = None) -> Dict[str, Any]:
    """One played unit's row (x26c's ``kakuna_units.jsonl`` fields, plus its regime, its BASE regime (the id without its
    model-load options' stamp), reasons, model-load options and, when given, its per-game identities
    :data:`GAME_KEYS`).

    ``cell_dir`` + ``rederived`` (a :data:`main.reads.rederive.MODES` mode): the forfeit-win guard and the H21 peer-fault
    check RE-DERIVED from the unit's captures and peer logs (:mod:`main.reads.rederive`, review H1) are recorded as
    ``rederived``; in ``count`` mode a hit makes the unit not count (``rederived_uncounted``), in ``report`` mode the
    counting is unchanged."""
    from main.anchors.results import unit_counts
    from main.reads import rederive as RD
    from main.reads.spec import load_stamp

    uc = unit_counts(summary, games=int(u.games))
    counted, reasons = uc.counts, list(uc.reasons)
    extra: Dict[str, Any] = {}
    if cell_dir is not None and rederived is not None:
        chk = RD.unit_check(cell_dir, summary)
        extra["rederived"] = chk
        if rederived == RD.COUNT and chk["hits"]:
            extra["rederived_uncounted"] = counted
            counted, reasons = False, reasons + [RD.hit_reason(chk)]
    cell = summary.get("cell") or {}
    stamp = load_stamp({k: v for k, v in (("belief_prior_mix", getattr(u, "belief_prior_mix", None)),
                                          ("search", getattr(u, "search", None))) if v is not None})
    return {"unit": u.name, "label": u.label, "panel": u.teamset, "unit_seed": int(u.unit_seed),
            "game": getattr(u, "game", None), "search": getattr(u, "search", None),
            "belief_prior_mix": getattr(u, "belief_prior_mix", None), "our_regime": cell.get("our_regime"),
            "games": games,
            "model_sha256": u.model_sha256, "status": summary.get("status"), "n": summary.get("n"),
            "wins": summary.get("wins"), "losses": summary.get("losses"), "ties": summary.get("ties", 0),
            "regime_verified_decisions": summary.get("regime_verified_decisions"),
            "peer_clean": summary.get("peer_clean"), "team_source_asymmetry": summary.get("team_source_asymmetry"),
            "counted": counted, "reasons": reasons, "regime_id": anchor_regime_id(summary),
            "base_regime_id": anchor_regime_id(summary, strip_load=stamp), "engine": EN.unit_engine(cell),
            "server_version": cell.get("server_version"), "model_step": cell.get("model_step"),
            "peer_pack": peer_pack_state(cell), **SI.unit_sides(cell), **_search_played(summary), **extra}


#: The searched half's record a ``main.anchors --our-search`` summary carries (``provenance.peer_report.search``, the
#: searcher's own counters): what ``search_headroom`` checks against the registered preset and reports.
SEARCH_PLAYED_KEYS = ("search", "preset", "searched", "deviations", "excluded", "search_cpu_s")


def _search_played(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """``{"search_played": {...}}`` when the unit's summary carries a search record (or ``our_search``), else ``{}``:
    a raw or banked unit's row is unchanged."""
    peer = (summary.get("provenance") or {}).get("peer_report") or {}
    if "search" not in peer and "our_search" not in peer:
        return {}
    found = peer.get("search")
    rec: Dict[str, Any] = found if isinstance(found, dict) else {}
    return {"search_played": dict({k: rec.get(k) for k in SEARCH_PLAYED_KEYS}, our_search=peer.get("our_search"))}


def anchor_units(kakuna_state: Path, *, rederived: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every PLAYED unit of one ``main.x35.kakuna`` state (a unit with no summary is unplayed, not a row);
    ``rederived``: the spec's mode of the re-derived guards (:func:`unit_row`; None = not checked)."""
    from main.x35 import kakuna as KA

    rows = []
    for u in KA.load_plan(kakuna_state):
        s = KA.read_summary(kakuna_state, u.name)
        if s is not None:
            cell = kakuna_state / "cells" / u.name
            rows.append(dict(unit_row(u, s, unit_games(kakuna_state, u.name), cell_dir=cell, rederived=rederived),
                             cell_dir=str(cell)))
    return rows


def anchor_group(rows: Sequence[Mapping[str, Any]], *, labels: Sequence[str], teamsets: Sequence[str],
                 unit_seeds: Sequence[int], shas: Optional[Mapping[str, str]] = None,
                 loads: Optional[Mapping[str, Mapping[str, Any]]] = None, per_game: bool = False,
                 exhausted: Sequence[str] = ()) -> Dict[str, Any]:
    """One anchor group's cells ``{label: {teamset: {wins, ties, n, units, planned, skipped}}}`` over COUNTED units,
    its ONE regime id and every reason it is not conclusive. ``shas``: the resolved models' sha256 (a unit played
    on other bytes refuses).

    ``loads``: ``{label: the model's load options}``. When every model shares one load (or none is given), a team set's
    counted units must share ONE regime id and the group's id is :func:`group_regime_id`, as it always was. When the
    models' loads DIFFER (each load stamps its own ``our_regime``), that rule would call every such group a mix, so
    instead: (1) the counted units of one (load, team set) share one regime id; (2) across ALL models a team set shares one
    BASE id (the id without the load's stamp, so the opponent, the teams, the transport and the rest are the same for every
    model); (3) a unit whose model declares a load but did not stamp it is not conclusive. The group's id is then
    :func:`mixed_group_regime_id`, and the group carries ``regime_ids_by_load`` and ``loads``.

    ``per_game`` (S-U): each unit is ONE game of its seed's series (its row's ``game``), so a cell plans
    ``len(unit_seeds) × UNIT_GAMES`` units and a unit is keyed by (label, set, seed, game).

    ``exhausted``: the names of this group's own units whose retries are spent (``main.reads.attempts``: not counted,
    ``MAX_ATTEMPTS`` attempts ended uncounted). The group carries them as ``exhausted_units`` and those of them with no
    row at all (no summary) as ``unplayed_exhausted_units``, a part of ``unplayed``: the rest of ``unplayed`` is what the
    lane's stop cut off (``stop_cut``)."""
    from main.reads.spec import load_stamp
    from main.x35.kakuna import UNIT_GAMES

    per_seed = UNIT_GAMES if per_game else 1

    stamps = {lab: load_stamp((loads or {}).get(lab) or {}) for lab in labels}
    mixed = len(set(stamps.values())) > 1
    why: List[str] = []
    cells: Dict[str, Dict[str, Dict[str, Any]]] = {
        lab: {ts: {"wins": 0, "ties": 0, "n": 0, "units": 0, "planned": len(unit_seeds) * per_seed, "skipped": []}
              for ts in teamsets} for lab in labels}
    regimes: Dict[str, set] = {ts: set() for ts in teamsets}
    by_load: Dict[Tuple[str, str], set] = {}
    bases: Dict[str, set] = {ts: set() for ts in teamsets}
    seen = set()
    for r in rows:
        key = (r["label"], r["panel"], int(r["unit_seed"]), r.get("game"))
        if r["label"] not in cells or r["panel"] not in cells[r["label"]] or int(r["unit_seed"]) not in unit_seeds:
            raise AssembleError(f"unit {r['unit']} is not a unit of this group")
        if per_game != (r.get("game") is not None) or per_game and not 0 <= int(r["game"]) < UNIT_GAMES:
            raise AssembleError(f"unit {r['unit']} (game {r.get('game')!r}) is not a "
                                f"{'per-game' if per_game else 'whole-series'} unit of this group")
        if key in seen:
            raise AssembleError(f"unit {key} appears twice")
        seen.add(key)
        if shas is not None and r.get("model_sha256") and r["model_sha256"] != shas[r["label"]]:
            raise AssembleError(f"unit {r['unit']} played sha {str(r['model_sha256'])[:12]}, not "
                                f"{shas[r['label']][:12]}")
        c = cells[r["label"]][r["panel"]]
        if not r["counted"]:
            c["skipped"].append(r["unit"])
            continue
        if r.get("regime_id") is None:
            why.append(f"unit {r['unit']}: no regime id (its summary stamps no cell)")
        regimes[r["panel"]].add(r.get("regime_id"))
        if mixed:
            stamp = stamps[r["label"]]
            by_load.setdefault((stamp, r["panel"]), set()).add(r.get("regime_id"))
            base = r.get("base_regime_id", r.get("regime_id")) if not stamp else r.get("base_regime_id")
            if base is None and r.get("regime_id") is not None:
                why.append(f"unit {r['unit']}: our_regime {r.get('our_regime')!r} does not end with {stamp!r}, the stamp "
                           f"its model's load options declare")
            bases[r["panel"]].add(base)
        c["units"] += 1
        c["wins"] += int(r["wins"])
        c["ties"] += int(r.get("ties", 0))
        c["n"] += int(r["n"])
    by_ts: Dict[str, Optional[str]] = {}
    for ts in teamsets:
        named = sorted(x for x in (bases if mixed else regimes)[ts] if x is not None)
        by_ts[ts] = named[0] if len(named) == 1 else None
        what = "BASE regimes (load stamps stripped)" if mixed else "regimes"
        if len(named) > 1:
            why.append(f"{ts}: counted units span {len(named)} {what} {named}")
        elif not named:
            why.append(f"{ts}: no counted unit carries a {'base ' if mixed else ''}regime id")
    out_extra: Dict[str, Any] = {}
    if mixed:
        full: Dict[str, Dict[str, Optional[str]]] = {}
        for stamp in sorted(set(stamps.values())):
            for ts in teamsets:
                named = sorted(x for x in by_load.get((stamp, ts), set()) if x is not None)
                full.setdefault(stamp or "none", {})[ts] = named[0] if len(named) == 1 else None
                if len(named) > 1:
                    why.append(f"{ts}: counted units of load {stamp or 'none'!r} span {len(named)} regimes {named}")
                elif not named:
                    why.append(f"{ts}: no counted unit of load {stamp or 'none'!r} carries a regime id")
        ok = all(by_ts.values()) and all(v for d in full.values() for v in d.values())
        rid = mixed_group_regime_id(by_ts, full) if ok else None    # type: ignore[arg-type]
        out_extra = {"regime_ids_by_load": full, "loads": stamps}
    else:
        rid = group_regime_id(by_ts) if all(by_ts.values()) else None
    engine, engine_why = EN.one(r.get("engine") for r in rows if r["counted"])
    if engine_why:
        why.append(f"counted units {engine_why}")
    sides, sides_why = SI.group_sides(rows)
    why += sides_why
    unplayed = len(labels) * len(teamsets) * len(unit_seeds) * per_seed - len(seen)
    have = {str(r["unit"]) for r in rows}
    spent = sorted(set(exhausted))
    return {"regime_id": rid, "regime_ids": by_ts, "engine": engine, **sides, "teamsets": list(teamsets),
            "unit_seeds": list(unit_seeds),
            "cells": cells, "units": sorted((dict(r) for r in rows), key=lambda r: r["unit"]),
            "unplayed": unplayed, "exhausted_units": spent,
            "unplayed_exhausted_units": [n for n in spent if n not in have],
            "inconclusive": sorted(set(why)), **out_extra}


# ------------------------------------------------------------------------------------------------ S
def members_of(refpop: Path, *, verify: bool = True, want_content: Optional[str] = None) -> Dict[str, str]:
    """``{label: sha256}`` of a FROZEN R (``main.x35.d1.refpop_members``: every file re-hashed first), its
    ``content_sha256`` checked against ``want_content`` (R moved under a read: the drift pre-read's hazard 1)."""
    from main.x35 import d1 as D

    doc = json.loads(Path(refpop).read_text())
    if want_content is not None and doc.get("content_sha256") != want_content:
        raise AssembleError(f"{refpop}: content_sha256 {str(doc.get('content_sha256'))[:12]} is not the spec's "
                            f"{want_content[:12]}")
    return D.refpop_members(str(refpop), verify=verify)


def s_node(cells: Sequence[Any], sha: str, members: Mapping[str, str], *, min_pairs: int = 500) -> Dict[str, Any]:
    """One node's S against R: ``readout.population_score`` plus the per-member SE, se(S) and its 95 % interval."""
    from main.reads.rules import Z95, cell_se, s_se
    from main.x35 import readout as R

    label_of = {v: k for k, v in members.items()}
    ps = R.population_score(cells, sha, members, min_pairs=min_pairs)
    mine = [c for c in cells if c.player == sha and c.opponent in label_of]
    ses = {label_of[c.opponent]: 100.0 * cell_se(c.pairs) for c in mine if c.pairs and sum(c.pairs)}
    node: Dict[str, Any] = {"sha256": sha, "s_pp": ps.s_pp, "per_member_pp": dict(ps.per_member_pp),
                            "per_member_se_pp": ses, "inconclusive": list(ps.inconclusive),
                            "n_pairs": {label_of[c.opponent]: c.n_pairs for c in mine},
                            "population_score": json.loads(json.dumps(ps.as_dict())), "se_pp": None,
                            "ci95_pp": None}
    if ps.s_pp is not None and len(ses) == len(members):
        se = s_se(list(ses.values()))
        node.update(se_pp=se, ci95_pp=[ps.s_pp - Z95 * se, ps.s_pp + Z95 * se])
    return node


def cell_pairs(root: Path, family: str, request: str, sha: str, members: Mapping[str, str]) -> Dict[str, int]:
    """``{member label: completed pairs}`` of node ``sha``'s cells in one request so far (the S lane's resume)."""
    got = L.read(S_FAMILY_READ, root=root, family=family)
    label_of = {v: k for k, v in members.items()}
    cells = dict(L.looks(got)).get(request, [])
    return {label_of[c.opponent]: c.n_pairs for c in cells if c.player == sha and c.opponent in label_of}


def s_group(root: Path, family: str, requests: Mapping[str, str], shas: Mapping[str, str],
            members: Mapping[str, str], *, panel: Optional[str] = None, want_regime: Optional[str] = None,
            refpop_content_sha256: Optional[str] = None, min_pairs: int = 500) -> Dict[str, Any]:
    """One S group from its eval ledger: ``requests`` = ``{label: request id}`` (a request may hold several
    nodes), ``shas`` = ``{label: sha256}``. The regime and digest are the family READ's own (module docstring)."""
    why: List[str] = []
    try:
        got = L.read(S_FAMILY_READ, root=root, family=family)
    except L.ReaderDeclError as exc:
        raise AssembleError(f"family {family} at {root}: {exc}") from None
    except L.MixedRegimeError as exc:
        return {"family": family, "regime_id": None, "engine": None, "ledger_digest": None, "panel": panel,
                "team_set": None,
                "rows": 0, "nodes": {lab: {"sha256": sha, "s_pp": None, "se_pp": None,
                                           "inconclusive": [f"mixed regimes: {exc}"]} for lab, sha in shas.items()},
                "inconclusive": [f"family {family}: mixed regimes ({exc})"], "refpop_content_sha256":
                    refpop_content_sha256}
    team_sets = sorted({str(r["regime"].get("team_set")) for r in got.rows})
    want_ts = None
    if panel is not None:
        from main.h2h import panel as PN

        want_ts = PN.panel_team_set_id(panel)
        off = [t for t in team_sets if t != want_ts]
        if off:
            why.append(f"rows on team set(s) {off}, not panel {panel}'s {want_ts}")
    if not got.rows:
        why.append(f"family {family}: no row")
    if got.rows and got.regime_id is None:
        why.append(f"family {family}: the read carries no regime id")
    if want_regime is not None and got.regime_id is not None and got.regime_id != want_regime:
        why.append(f"regime {got.regime_id} is not the registered {want_regime}")
    engine, mixed = EN.one(EN.row_engine(r) for r in got.rows)
    if mixed:
        why.append(f"family {family}: rows {mixed}")
    looks = dict(L.looks(got))
    for req in sorted(set(requests.values())):
        if req not in looks:
            why.append(f"request {req} of family {family}: no row")
    nodes = {lab: s_node(looks.get(requests[lab], []), sha, members, min_pairs=min_pairs) for lab, sha in shas.items()}
    return {"family": family, "regime_id": got.regime_id, "engine": engine, "ledger_digest": got.digest, "panel": panel,
            "team_set": want_ts if panel is not None else (team_sets[0] if len(team_sets) == 1 else team_sets),
            "rows": len(got.rows), "nodes": nodes, "inconclusive": why,
            "refpop_content_sha256": refpop_content_sha256}


# ------------------------------------------------------------------------------------------------ the read
def assemble(spec: Any, state: Path) -> Dict[str, Any]:
    """Every cell group of ``spec`` from its state (``main.reads.layout``), the models as resolved."""
    from main.reads import layout as LY
    from main.reads.models import spec_path

    mf = LY.models_file(state)
    models = json.loads(mf.read_text()) if mf.exists() else {}
    out: Dict[str, Any] = {"schema": CELLS_SCHEMA, "read": spec.read, "spec_sha256": spec.sha256,
                           "models": {lab: dict(models[lab], arm=m.arm, seed=m.seed) if lab in models
                                      else {"arm": m.arm, "seed": m.seed, "unresolved": True}
                                      for lab, m in spec.models.items()},
                           "anchor": {}, "s": {},
                           # the guard policy the spec registers, read by the rules' game-for-game guards (H2) and
                           # main.reads.rederive (H1)
                           "guards": {"rederived": spec.rederived_guards, "choices": spec.choices_guard}}
    shas = {lab: e["sha256"] for lab, e in models.items()}
    from main.reads import imports as IM
    from main.reads.spec import imported

    for g, a in spec.anchor.items():
        rows: List[Dict[str, Any]] = []
        spent: List[str] = []
        for lab in a.models:
            st = LY.anchor_state(state, g, lab)
            if lab not in imported(a) and (st / "plan.json").exists():
                rows += anchor_units(st, rederived=spec.rederived_guards)
                spent += AT.exhausted_units(st)
        meta = None
        if a.imports is not None:                         # the earlier read's units: sha-, regime- and engine-checked
            got, meta = IM.anchor_rows(spec, g, shas, this_state=state, own=rows)
            rows += got
        grp = anchor_group(rows, labels=a.models, teamsets=a.teamsets, unit_seeds=a.unit_seeds, shas=shas,
                           loads={lab: spec.models[lab].load for lab in a.models}, per_game=a.per_game,
                           exhausted=spent)
        if a.per_game and any("search" in spec.models[lab].load for lab in a.models):
            # G-PREFIX (main.reads.prefix): a pair whose protocol diverged before the search's first turn is a
            # nondeterminism that corrupts the pairing; the group is not conclusive (the read VOID) until it is resolved
            from main.reads import prefix as PX

            grp["g_prefix"] = PX.check_group(spec, state, g)
            if grp["g_prefix"]["mismatches"]:
                grp["inconclusive"] = sorted(set(grp["inconclusive"]) | {
                    f"G-PREFIX: deal {m['deal']} diverged before the search's first turn ({m['why']})"
                    for m in grp["g_prefix"]["mismatches"]})
        missing = [lab for lab in a.models if lab not in shas]
        if missing:
            grp["inconclusive"] = sorted(set(grp["inconclusive"]) | {f"model(s) {missing} not resolved"})
        if meta is not None:
            grp["imports"] = meta
        out["anchor"][g] = grp
    for g, x in spec.s.items():
        mem = members_of(spec_path(x.refpop), want_content=x.refpop_content_sha256)
        have = {lab: shas[lab] for lab in x.models if lab in shas and lab not in imported(x)}
        sgrp: Optional[Dict[str, Any]] = None
        if have or x.imports is None:
            sgrp = s_group(LY.s_ledger(state, g), x.family, {lab: x.request(lab) for lab in have}, have, mem,
                           panel=x.panel, want_regime=x.regime_id, refpop_content_sha256=x.refpop_content_sha256,
                           min_pairs=x.pairs)
        if x.imports is not None:
            ig, meta = IM.s_group(spec, g, shas, mem, this_state=state)
            if sgrp is None:
                sgrp = dict(ig)                               # every node imported: the earlier ledger's read
            else:
                sgrp["nodes"] = dict(sgrp["nodes"], **ig["nodes"])
                sgrp["inconclusive"] = list(sgrp["inconclusive"]) + [f"imported: {w}" for w in ig["inconclusive"]]
                if sgrp.get("rows"):                          # this ledger played too: one engine for both
                    IM.check_engine(f"s.{g} import", sgrp.get("engine"), ig.get("engine"))
                if ig.get("regime_id") != sgrp.get("regime_id"):
                    sgrp["inconclusive"].append(f"imported regime {ig.get('regime_id')} is not this ledger's "
                                                f"{sgrp.get('regime_id')}")
            sgrp["imports"] = meta
        assert sgrp is not None       # `have or no imports` computed it; otherwise the imports branch set it
        missing = [lab for lab in x.models if lab not in shas]
        if missing:
            sgrp["inconclusive"] = list(sgrp["inconclusive"]) + [f"model(s) {missing} not resolved"]
        out["s"][g] = sgrp
    return out


def group_reasons(lane: str, g: str, grp: Mapping[str, Any], *, allow_unplayed: bool = False) -> List[str]:
    """Every reason ONE assembled group is not complete and regime-stamped (empty = it is). ``allow_unplayed``: an anchor
    group's unplayed units are not a reason (a reduced read the spec registered, ``record.completeness``)."""
    why = [f"{lane}.{g}: {w}" for w in grp.get("inconclusive") or ()]
    if not grp.get("regime_id"):
        why.append(f"{lane}.{g}: no regime id")
    if lane == "anchor" and grp.get("unplayed") and not allow_unplayed:
        why.append(f"anchor.{g}: {grp['unplayed']} unit(s) unplayed")
    if lane == "s":
        for lab, n in (grp.get("nodes") or {}).items():
            if n.get("s_pp") is None or n.get("se_pp") is None or not math.isfinite(float(n["s_pp"])):
                why.append(f"s.{g}.{lab}: no S ({'; '.join(n.get('inconclusive') or ['incomplete'])})")
    return why


def conclusive(cells: Mapping[str, Any], *, reported: Sequence[Tuple[str, str]] = (),
               allow_unplayed: Sequence[Tuple[str, str]] = ()) -> List[str]:
    """Every reason the assembled cells are not a complete, regime-stamped read (empty = record-ready). The groups in
    ``reported`` (``(lane, group)``: the rule's REPORTED-only groups, which no verdict reads) are left out: they are
    :func:`reported_incomplete`'s, and never block the record of the decided primary (N1-P FINDING 7). The anchor groups in
    ``allow_unplayed`` may have units unplayed (:func:`unplayed_groups`); every other reason still stands."""
    skip, allow = set(reported), set(allow_unplayed)
    why: List[str] = []
    for lane in ("anchor", "s"):
        for g, grp in (cells.get(lane) or {}).items():
            if (lane, g) not in skip:
                why += group_reasons(lane, g, grp, allow_unplayed=(lane, g) in allow)
    return why


def unplayed_groups(cells: Mapping[str, Any], reported: Sequence[Tuple[str, str]] = ()) -> Dict[str, Dict[str, Any]]:
    """``{"anchor.<group>": {planned, played, unplayed, counted, unplayed_exhausted, stop_cut, unplayed_exhausted_units}}``
    of each DECIDED (not reported-only) anchor group that has units unplayed: what a reduced read at a registered stop
    leaves out. ``counted`` = units that entered the cells. ``unplayed`` is TWO things: ``unplayed_exhausted`` units whose
    retries were spent without a summary (a defect of the play, which a longer stop would not have fixed) and ``stop_cut``
    units the lane's registered stop never reached (``main.reads.attempts``)."""
    skip = set(reported)
    out: Dict[str, Dict[str, Any]] = {}
    for g, grp in (cells.get("anchor") or {}).items():
        n = int(grp.get("unplayed") or 0)
        if n and ("anchor", g) not in skip:
            planned = sum(int(c["planned"]) for by in grp["cells"].values() for c in by.values())
            lost = list(grp.get("unplayed_exhausted_units") or ())
            out[f"anchor.{g}"] = {"planned": planned, "played": planned - n, "unplayed": n,
                                  "counted": sum(int(c["units"]) for by in grp["cells"].values() for c in by.values()),
                                  "unplayed_exhausted": len(lost), "stop_cut": n - len(lost),
                                  "unplayed_exhausted_units": lost}
    return out


def reported_incomplete(cells: Mapping[str, Any], reported: Sequence[Tuple[str, str]]) -> Dict[str, List[str]]:
    """``{"<lane>.<group>": [reasons]}`` of each REPORTED-only group that is not complete (recorded INCOMPLETE)."""
    out: Dict[str, List[str]] = {}
    for lane, g in reported:
        grp = (cells.get(lane) or {}).get(g)
        why = [f"{lane}.{g}: not assembled"] if grp is None else group_reasons(lane, g, grp)
        if why:
            out[f"{lane}.{g}"] = why
    return out
