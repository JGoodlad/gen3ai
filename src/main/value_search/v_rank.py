"""V-RANK's driver (``designs/endstate/design_v_rank_on_u.md``; the estimator is :mod:`main.value_search.v_rank_stat`):
does the search's value estimate Q̂ rank the moves of ONE position as well on unfamiliar teams as on home72, with the
truth a REAL game against Kakuna after every legal action?

It is the deviation-only live read's pipeline (:mod:`main.value_search.live_branch`) with EVERY legal action branched
instead of S′'s pick alone. The raw games are a ``main.reads`` anchor read (a pinned P*, ``load.input_log``, whole
20-game units); the branches are ``main.x35.kakuna`` states of one-game forced units (``main.anchors --our-force``).

* ``select``: per raw game (deal) and per STRATUM (A = turns 10-19, B = turns 20-30), the kept decision of ours with
  >= 2 legal actions and the smallest ``sha256("vrank_v1|<deal>|<turn>|<n>")``, from the COUNTED units of every named
  raw state. Outcome-blind; written once, frozen.
* ``score`` (the SHADOW ARMS, offline on the kept honest input log, the live searcher's own ``_search`` and root
  guards through :class:`sign_gate.GateScorer`): Q̂(a) for every legal action with the PER-WORLD leaf values (a tap on
  the leaf engine; refused unless each action's mean is the search's own Q̂), log π, an exclusion when no world is
  consistent; and the SELF-REFEREE ``oracle_leaf8`` (:data:`ORACLE`): the TRUE world, every legal action played
  :data:`ORACLE_SEEDS` times to the end with P* on both sides (REPORTED only). Resumable; ``--part i/K`` splits the
  states by their draw hash so K scorers run side by side, each into its own file of ``--out-dir``. Prints no value.
* ``plan``: one forced one-game unit per legal action other than π's at every SEARCHED scored state (kind ``branch``)
  and one NULL control per raw unit with a scored state (π's own action forced at that unit's selected state with the
  smallest hash), dealt over ``--shards`` kakuna states ``<out>/s<k>`` by raw unit (one supervisor is serial). The plan
  is a function of the selection and the scores only, so it may be EXTENDED after more states are scored: every unit
  already planned must be reproduced exactly, or it refuses. Play each shard with
  ``python -m main.x35.kakuna run --state <out>/s<k>``.
* ``classify``: the peer-divergence classifier (:mod:`main.reads.peer_divergence`) over every attempt of every null
  and of every branch that is not counted or recorded a G-BRANCH failure -> ``<shard>/peer_drops.jsonl``. It opens
  captures only, never a result.
* ``stage0``: the outcome-free look (the design's §5.4): s̄ = the mean Σ q_c² a searched position, the exclusion rate,
  the legal-action count, the CPU of every part, the teeth, and the sizing rule's block count. It reads no y.
* ``read``: the registered decision (:func:`read`); ``main.reads`` records it through the rule ``v_rank``
  (:mod:`main.reads.rules_v_rank`).

y(a) for π's own action is the RAW game's outcome; for every other legal action its forced branch's. A position enters
iff the search valued it and every one of its branches counted.

THE TEETH (each a VOID): G-BRANCH failed in an attempt that is not a classified peer drop; a counted branch's capture
is not the raw game's through the branch turn; a null control that changed its game (peer drops excepted); a branch
that forced another action; counted games on another engine than the registered one; a searched state on another
preset; a searched state without its per-world values or its ``oracle_leaf8`` record; the planned branches are not
exactly the other legal actions of the searched scored states; a selected state whose hash or stratum is not its
draw's, or two of one (deal, stratum); more than ``max_drop_frac`` of a panel's searched positions incomplete.

    python -m main.value_search.v_rank select --raw-state DIR [--raw-state DIR] --out selected.jsonl
    python -m main.value_search.v_rank score  --selected selected.jsonl --model ZIP --search r0s4_honest@10
        --engine STAMP --out-dir SCORED [--part i/K] [--games LO-HI] [--unit-seeds A,B] [--budget-s S]
    python -m main.value_search.v_rank plan   --selected selected.jsonl --scored-dir SCORED --raw-state DIR [...]
        --out BRANCHES --shards K
    python -m main.value_search.v_rank classify --branch-state BRANCHES
    python -m main.value_search.v_rank stage0 --selected selected.jsonl --scored-dir SCORED --branch-state BRANCHES
    python -m main.value_search.v_rank read   --selected selected.jsonl --scored-dir SCORED --branch-state BRANCHES
        --params P.json

``--branch-state`` takes a plan root (every shard of it) or one shard ``BRANCHES/s<k>``, and repeats.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from main.value_search import v_rank_stat as ST
from main.value_search.v_rank_stat import VRankError

SELECT_SALT = "vrank_v1"
#: the strata's turn windows (inclusive); one position a stratum a deal
STRATA: Dict[str, Tuple[int, int]] = {"A": (10, 19), "B": (20, 30)}
#: the self-referee's name and size: the true world, every legal action x this many seeds to the end, P* both sides.
#: A REFEREE configuration, never a playable preset (it reads the true opponent team): it is not in
#: ``main.anchors.search_presets.PRESETS``, so no ``--our-search`` / ``load.search`` can name it.
ORACLE, ORACLE_SEEDS = "oracle_leaf8", 8
SELECT_SCHEMA = "gen3_v_rank_selected_v1"
STATE_SCHEMA = "gen3_v_rank_state_v1"
PLAN_SCHEMA = "gen3_v_rank_plan_v1"
VERDICT_SCHEMA = "gen3_v_rank_verdict_v1"
PEER_DROPS_NAME = "peer_drops.jsonl"
GAMES_NAME = "games.jsonl"
VOID = "VOID"
#: the design's §5.4 sizing rule: final blocks = BLOCKS x min(2, max(1, S_BAR_REF / s̄)), up to a multiple of BLOCK_STEP
BLOCKS, S_BAR_REF, BLOCK_STEP = 200, 0.030, 20
_ROW_KEYS = ("panel", "cell", "unit_seed", "index", "deal", "battle", "team", "turn", "n", "n_legal", "key",
             "select_hash", "stratum")


def _jsonl(p: Path) -> List[Dict[str, Any]]:
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _atomic(p: Path, text: str) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, p)


def scored_files(scored_dir: Path) -> List[Path]:
    return sorted(Path(scored_dir).glob("*.jsonl"))


def load_scored(files: Sequence[Path]) -> List[Dict[str, Any]]:
    """Every scored row of ``files``; a state scored twice (in one file or two) is refused."""
    rows: List[Dict[str, Any]] = []
    for f in files:
        rows += _jsonl(Path(f))
    keys = [r["key"] for r in rows]
    if len(set(keys)) != len(keys):
        raise VRankError("a state is scored twice")
    return rows


def order_key(r: Mapping[str, Any]) -> Tuple[int, int, str, str]:
    """The registered order: unit seed, game, panel, draw hash."""
    return int(r["unit_seed"]), int(r["index"]), str(r["panel"]), str(r["select_hash"])


# --------------------------------------------------------------------------------------------------- select
def stratum_of(turn: int) -> Optional[str]:
    for name, (lo, hi) in STRATA.items():
        if lo <= int(turn) <= hi:
            return name
    return None


def select_hash(deal: str, turn: int, n: int) -> str:
    return hashlib.sha256(f"{SELECT_SALT}|{deal}|{int(turn)}|{int(n)}".encode()).hexdigest()


def select(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Per (deal, stratum) the eligible kept decision with the smallest :func:`select_hash`, in (deal, stratum) order.
    A duplicate (deal, turn, n) is a defect."""
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    seen = set()
    for r in records:
        k = (str(r["deal"]), int(r["turn"]), int(r["n"]))
        if k in seen:
            raise VRankError(f"deal {r['deal']}: decision (turn {k[1]}, n {k[2]}) kept twice")
        seen.add(k)
        st = stratum_of(k[1])
        if st is None or int(r["n_legal"]) < 2:
            continue
        h = select_hash(*k)
        cur = best.get((k[0], st))
        if cur is None or h < cur["select_hash"]:
            best[(k[0], st)] = {**r, "stratum": st, "select_hash": h}
    return [best[k] for k in sorted(best)]


def parse_range(text: Optional[str]) -> Optional[Tuple[int, int]]:
    if text is None:
        return None
    lo, sep, hi = str(text).partition("-")
    try:
        out = (int(lo), int(hi if sep else lo))
    except ValueError:
        raise VRankError(f"range {text!r}: 'LO-HI' or one integer") from None
    if not 0 <= out[0] <= out[1]:
        raise VRankError(f"range {text!r}: 0 <= LO <= HI")
    return out


def select_states(raw_states: Sequence[Path], games: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
    """The draw over the COUNTED units of every raw kakuna state (``<state>/anchor/<group>/<label>``), each state
    tagged with where it came from and with the TEAM we piloted in its game (the cluster of every interval; a unit
    deals each of its teams twice). ``games``: only these games of each unit (a smoke; the registered read draws from
    every game)."""
    from main.value_search import sign_gate as SG

    out: List[Dict[str, Any]] = []
    for st in raw_states:
        st = Path(st)
        label, group = st.name, st.parent.name
        try:
            cells = SG.counted_cells(st.parent.parent.parent, group, label)
            teams = {(c["cell"], g["battle_tag"]): g.get("our_team") for c in cells
                     for g in _jsonl(Path(c["dir"]) / GAMES_NAME)}
            recs = (r for r in SG.kept_records(cells) if games is None or games[0] <= int(r["index"]) <= games[1])
            for r in select(recs):
                team = teams.get((r["cell"], r["battle"]))
                if not team:
                    raise VRankError(f"{r['key']}: its game {r['battle']} of {r['cell']} names no team of ours")
                out.append({**r, "schema": SELECT_SCHEMA, "raw_state": str(st), "team": str(team)})
        except (SG.GateError, OSError, ValueError, KeyError) as exc:
            raise VRankError(f"raw state {st}: {type(exc).__name__}: {exc}") from None
    keys = [r["key"] for r in out]
    if len(set(keys)) != len(keys):
        raise VRankError("a state is drawn from two raw states")
    return out


# ---------------------------------------------------------------------------------------------------- score
def parse_part(text: str) -> Tuple[int, int]:
    i, sep, k = str(text).partition("/")
    try:
        out = (int(i), int(k))
    except ValueError:
        raise VRankError(f"part {text!r}: 'i/K'") from None
    if not sep or not 0 <= out[0] < out[1]:
        raise VRankError(f"part {text!r}: 'i/K' with 0 <= i < K")
    return out


def in_part(rec: Mapping[str, Any], part: Tuple[int, int]) -> bool:
    return int(str(rec["select_hash"])[:12], 16) % part[1] == part[0]


def to_score(selected: Sequence[Mapping[str, Any]], done: Iterable[str], *, part: Tuple[int, int] = (0, 1),
             games: Optional[Tuple[int, int]] = None, unit_seeds: Optional[Sequence[int]] = None
             ) -> List[Dict[str, Any]]:
    """The selected states still to score, in the registered order: this part's, inside the filters, not in ``done``."""
    done = set(done)
    seeds = None if unit_seeds is None else {int(x) for x in unit_seeds}
    short = sorted({k for r in selected for k in _ROW_KEYS if k not in r})
    if short:
        raise VRankError(f"the selection lacks {short} (another schema: draw it again)")       # before any search
    out = []
    for r in sorted(selected, key=order_key):
        if r["key"] in done or not in_part(r, part):
            continue
        if games is not None and not games[0] <= int(r["index"]) <= games[1]:
            continue
        if seeds is not None and int(r["unit_seed"]) not in seeds:
            continue
        out.append(dict(r))
    return out


def world_values(taps: Sequence[Tuple[int, int, float]], q: Mapping[str, float], key: str) -> Dict[str, List[float]]:
    """``{action: [leaf value per world, in world order]}`` from the leaf engine's tapped ``(world, action, value)``
    answers. Refused unless every legal action was valued once in each of the SAME >= 2 worlds and its mean is the
    search's own Q̂: the per-world values are then exactly what Q̂ averaged."""
    by: Dict[int, Dict[int, float]] = {}
    for w, a, v in taps:
        if int(w) in by.setdefault(int(a), {}):
            raise VRankError(f"{key}: action {a} was valued twice in world {w}")
        by[int(a)][int(w)] = float(v)
    acts = sorted(int(a) for a in q)
    if sorted(by) != acts:
        raise VRankError(f"{key}: the leaves valued actions {sorted(by)}, the search {acts}")
    worlds = sorted(by[acts[0]])
    if len(worlds) < 2:
        raise VRankError(f"{key}: {len(worlds)} world(s): the per-world noise needs >= 2")
    for a in acts:
        if sorted(by[a]) != worlds:
            raise VRankError(f"{key}: action {a} was not valued in the same worlds as action {acts[0]}")
        mean = float(np.mean([by[a][w] for w in worlds]))
        if abs(mean - float(q[str(a)])) > 1e-5:
            raise VRankError(f"{key}: action {a}'s per-world mean {mean:.6f} is not the search's Q̂ "
                             f"{float(q[str(a)]):.6f} (a weighted or halved search has no common-world values)")
    return {str(a): [round(by[a][w], 6) for w in worlds] for a in acts}


class Scorer:
    """The shadow arms on one kept decision: the search (Q̂ and its per-world values) and ``oracle_leaf8``."""

    def __init__(self, model_zip: str, search: str, *, gate: Any = None, model_sha256: Optional[str] = None) -> None:
        if gate is None:
            from main.value_search.sign_gate import GateScorer
            from utils.hashing import sha256_file

            gate, model_sha256 = GateScorer(model_zip, search), sha256_file(Path(model_zip))
        #: the bytes Q̂ and the referee were computed with, on every row: the rule VOIDs on another net than the raw arm's
        self.g, self.model_sha256 = gate, model_sha256

    def close(self) -> None:
        self.g.close()

    def search(self, rec: Mapping[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, List[float]]]]:
        """``(the live searcher's record, the per-world values or None on an exclusion)``."""
        from agents.battle.core_obs import wrap_row

        _truth, d = self.g.root(rec)
        row = np.array(wrap_row(d["frame"]), dtype=np.float32)
        mask = np.asarray(rec["mask"], dtype=np.int8)
        eng = self.g.s.engine
        orig = eng.leaves
        taps: List[Tuple[int, int, float]] = []

        def tapped(logs: Any, at: Any, side: Any, root_turn: Any, requests: Any, depth: Any = None) -> Any:
            vals = orig(logs, at, side, root_turn, requests, depth=depth)
            taps.extend((int(r.world), int(r.action), float(v)) for r, v in zip(requests, vals))
            return vals

        eng.leaves = tapped
        try:
            s = self.g.s._search(rec["log"], int(rec["at"]), rec["side"], row, mask, d, rec["key"],
                                 int(rec["pi_argmax"]))
        finally:
            eng.leaves = orig
        return s, (world_values(taps, s["q"], rec["key"]) if s.get("searched") else None)

    def referee(self, rec: Mapping[str, Any], legal: Sequence[int]) -> Dict[str, Any]:
        from main.value_search.playout import referee
        from main.value_search.sign_gate import _children_cpu, referee_seeds

        c0, k0 = time.thread_time(), _children_cpu()
        seeds = referee_seeds(rec["key"])[:ORACLE_SEEDS]
        if len(seeds) != ORACLE_SEEDS:
            raise VRankError(f"{rec['key']}: {len(seeds)} referee seeds, {ORACLE} plays {ORACLE_SEEDS}")
        res = referee(self.g.s.forward, rec["log"], int(rec["at"]), rec["side"], seeds, actions=list(legal),
                      lib=self.g.s.engine.lib)                    # the search's own engine library, one engine a state
        out = {str(a): str(res.out[a]) for a in legal}
        return {"preset": ORACLE, "seeds": ORACLE_SEEDS, "out": out,
                "q_self": {a: round(float(np.mean([ST.OUTCOME[ch] for ch in v])), 6) for a, v in out.items()},
                "cpu_s": round(time.thread_time() - c0 + _children_cpu() - k0, 3)}

    def score(self, rec: Mapping[str, Any]) -> Dict[str, Any]:
        t0 = time.monotonic()
        s, worlds = self.search(rec)
        out: Dict[str, Any] = {k: rec[k] for k in _ROW_KEYS}
        out.update(schema=STATE_SCHEMA, pi_argmax=int(rec["pi_argmax"]), model_sha256=self.model_sha256, search=s)
        if s.get("searched"):
            out["worlds"] = worlds
            out["referee"] = self.referee(rec, sorted(int(a) for a in s["q"]))
        out["wall_s"] = round(time.monotonic() - t0, 3)
        return out


# ----------------------------------------------------------------------------------------------------- plan
def planned(selected: Sequence[Mapping[str, Any]], scored: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The forced units the scores call for: ``{"kind", "rec", "action"}`` (module docstring)."""
    sel = {r["key"]: r for r in selected}
    rows: List[Dict[str, Any]] = []
    cells = set()
    for row in sorted(scored, key=order_key):
        rec = sel.get(row["key"])
        if rec is None:
            raise VRankError(f"{row['key']}: scored, never selected")
        cells.add((rec["panel"], rec["cell"]))
        if not row["search"].get("searched"):
            continue
        top = int(rec["pi_argmax"])
        acts = sorted(int(a) for a in row["search"]["q"])
        if top not in acts:
            raise VRankError(f"{row['key']}: π's action {top} is not among the valued actions {acts}")
        rows += [{"kind": "branch", "rec": rec, "action": a} for a in acts if a != top]
    nulls: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for rec in selected:
        k = (rec["panel"], rec["cell"])
        if k in cells and (k not in nulls or rec["select_hash"] < nulls[k]["select_hash"]):
            nulls[k] = rec
    rows += [{"kind": "null", "rec": rec, "action": int(rec["pi_argmax"])} for _k, rec in sorted(nulls.items())]
    return rows


def cell_shards(selected: Sequence[Mapping[str, Any]], shards: int) -> Dict[str, int]:
    """Raw unit -> shard: its rank among the selection's raw units by (unit seed, panel, name), modulo ``shards``."""
    cells = sorted({(int(r["unit_seed"]), str(r["panel"]), str(r["cell"])) for r in selected})
    return {c[2]: i % int(shards) for i, c in enumerate(cells)}


def plan_branches(selected: Sequence[Mapping[str, Any]], scored: Sequence[Mapping[str, Any]],
                  raw_states: Sequence[Path], out: Path, shards: int = 1) -> Dict[str, Any]:
    """Write or EXTEND the branch states ``<out>/s<k>`` (``plan.json``, ``branch_plan.jsonl``, ``specs/``). A unit
    already planned that the selection and the scores no longer call for, or call for differently, refuses; so does a
    shard whose supervisor is running, or another shard count than the first plan's."""
    import dataclasses

    from main.anchors.forced_branch import spec_from_kept
    from main.value_search.live_branch import _raw_units
    from utils.filelock import try_exclusive

    out = Path(out)
    if int(shards) < 1:
        raise VRankError("shards: >= 1")
    meta_p = out / "shards.json"
    if meta_p.exists() and int(json.loads(meta_p.read_text())["shards"]) != int(shards):
        raise VRankError(f"{meta_p} records another shard count than {shards}")
    raw = _raw_units(raw_states)
    shard = cell_shards(selected, shards)
    per: Dict[int, List[Tuple[Any, Dict[str, Any], Dict[str, Any], Path]]] = {k: [] for k in range(int(shards))}
    for x in planned(selected, scored):
        rec = x["rec"]
        if rec["cell"] not in raw:
            raise VRankError(f"{rec['key']}: raw cell {rec['cell']} is in no raw state's plan")
        st, ru = raw[rec["cell"]]
        if ru.game is not None or ru.search is not None:
            raise VRankError(f"raw unit {ru.name}: a whole raw 20-game unit is the branches' source")
        game = int(rec["index"])
        spec = spec_from_kept(rec, x["action"], game=game)
        name = f"{ru.name}__g{game:02d}__{spec['kind']}__t{spec['turn']}n{spec['n']}a{spec['action']}"
        k = shard[rec["cell"]]
        sp = out / f"s{k}" / "specs" / f"{name}.json"
        u = dataclasses.replace(ru, name=name, games=1, game=game, input_log=None, force=str(sp.resolve()))
        m = {"schema": PLAN_SCHEMA, "unit": name, "kind": spec["kind"], "key": spec["key"], "panel": rec["panel"],
             "cell": rec["cell"], "raw_state": str(st), "unit_seed": int(ru.unit_seed), "game": game,
             "deal": rec["deal"], "battle": spec["battle"], "turn": spec["turn"], "n": spec["n"],
             "action": spec["action"], "pi_argmax": spec["pi_argmax"], "spec": str(sp), "shard": k}
        per[k].append((u, m, spec, sp))
    report: Dict[str, Any] = {"shards": int(shards), "units": 0, "new": 0, "per_shard": {}, "per_panel": {}}
    writes = []
    for k, items in per.items():
        sdir = out / f"s{k}"
        # the queue: the registered order (unit seed, game), a unit's branches before its null control
        items.sort(key=lambda t: (t[1]["unit_seed"], t[1]["game"], t[1]["panel"], t[1]["kind"] == "null",
                                  t[1]["key"], t[1]["action"]))
        new = {t[1]["unit"]: t for t in items}
        if len(new) != len(items):
            raise VRankError("two branch units share a name")
        old = {m["unit"]: m for m in _jsonl(sdir / "branch_plan.jsonl")}
        gone = sorted(set(old) - set(new))
        moved = sorted(n for n in old if n in new and old[n] != new[n][1])
        if gone or moved:
            raise VRankError(f"{sdir}: the selection and scores no longer reproduce {len(gone)} planned unit(s) and "
                             f"change {len(moved)} (first {(gone + moved)[0]}): a plan is only ever extended")
        writes.append((sdir, items, [n for n in new if n not in old]))
    for sdir, items, fresh in writes:
        (sdir / "specs").mkdir(parents=True, exist_ok=True)
        lock = try_exclusive(sdir / "supervisor.lock", "w")
        if lock is None:
            raise VRankError(f"{sdir}: its supervisor is running: extend a plan between runs")
        try:
            for _u, _m, spec, sp in items:
                txt = json.dumps(spec, indent=1, sort_keys=True) + "\n"
                if sp.exists():
                    if sp.read_text() != txt:
                        raise VRankError(f"{sp}: the spec on disk is not the planned one")
                else:
                    sp.write_text(txt)
            _atomic(sdir / "plan.json", json.dumps({"units": [t[0].__dict__ for t in items]}, indent=1) + "\n")
            _atomic(sdir / "branch_plan.jsonl", "".join(json.dumps(t[1], sort_keys=True) + "\n" for t in items))
        finally:
            lock.close()
        report["per_shard"][sdir.name] = {"units": len(items), "new": len(fresh)}
        report["units"] += len(items)
        report["new"] += len(fresh)
        for _u, m, _s, _p in items:
            c = report["per_panel"].setdefault(m["panel"], {"branch": 0, "null": 0})
            c[m["kind"]] += 1
    _atomic(meta_p, json.dumps({"shards": int(shards)}) + "\n")
    return report


# ------------------------------------------------------------------------------------------------- classify
def classify_state(state: Path, everything: bool = False) -> List[Dict[str, Any]]:
    """Every attempt of every NULL, and of every branch that is not counted or that recorded a G-BRANCH failure (every
    unit with ``everything``), classified against its raw game's capture (:func:`main.reads.peer_divergence.classify`,
    the dlive read's AM-DL-PEER rule). A null is always classified: a counted null can still change its game after
    its branch turn. Captures only: no game row, summary or result line is opened."""
    from main.reads.peer_divergence import classify, is_peer_drop
    from main.value_search.live_branch import _g_branch_fails
    from main.x35 import kakuna as KA

    state = Path(state)
    units = {u.name: u for u in KA.load_plan(state)}
    out: List[Dict[str, Any]] = []
    for m in _jsonl(state / "branch_plan.jsonl"):
        u = units[m["unit"]]
        done, gb = KA.is_done(state, u), _g_branch_fails(state, u.name)
        if not everything and m["kind"] == "branch" and done and not gb:
            continue
        raw_p = Path(m["raw_state"]) / "cells" / m["cell"] / "capture" / f"{m['battle']}.json"
        dirs = [d for d in [state / "cells" / u.name, *sorted((state / "cells").glob(f"{u.name}.partial.*"))]
                if d.is_dir()]
        for d in dirs:
            row = {"unit": u.name, "kind": m["kind"], "panel": m["panel"], "attempt": d.name}
            caps = sorted((d / "capture").glob("*.json"))
            if len(caps) != 1:
                out.append({**row, "class": "unattributable", "why": f"{len(caps)} captures", "peer_drop": False})
                continue
            try:
                c = classify(json.loads(raw_p.read_text()), json.loads(caps[0].read_text()), int(m["turn"]))
            except (OSError, ValueError) as exc:
                raise VRankError(f"{u.name}: unreadable capture ({exc})") from None
            out.append({**row, "counted": done, "g_branch_fails": len(gb),
                        "peer_drop": is_peer_drop(m["kind"], c["class"]), **c})
    return out


def shard_states(out: Path) -> List[Path]:
    """The kakuna states ``<out>/s<k>`` of a plan root (``shards.json``, written by :func:`plan_branches`)."""
    meta = Path(out) / "shards.json"
    if not meta.is_file():
        raise VRankError(f"{meta}: missing (not a V-RANK plan root)")
    return [Path(out) / f"s{k}" for k in range(int(json.loads(meta.read_text())["shards"]))]


def expand_shards(states: Sequence[Path]) -> List[Path]:
    """Each path as its kakuna states: a plan root (it holds ``shards.json``) is its shards, anything else is one
    shard state as named."""
    out: List[Path] = []
    for st in (Path(x) for x in states):
        out += shard_states(st) if (st / "shards.json").is_file() else [st]
    return out


def load_branches(branch_states: Sequence[Path], *, need_peer: bool = True
                  ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    """``(rows, the peer-drop report, the artefacts' sha256)`` over every shard: ``live_branch.collect``'s rows with
    the classified peer drops applied (``rules_deviation_live.apply_peer_drops``). ``need_peer``: every shard must
    hold its ``peer_drops.jsonl`` (``classify`` ran)."""
    from main.reads.rules_deviation_live import apply_peer_drops
    from main.value_search.live_branch import LiveBranchError, collect
    from utils.hashing import sha256_file

    rows: List[Dict[str, Any]] = []
    cls: List[Dict[str, Any]] = []
    arts: Dict[str, Any] = {}
    have = 0
    for st in (Path(x) for x in branch_states):
        try:
            rows += collect(st)
        except (LiveBranchError, OSError, ValueError, KeyError) as exc:
            raise VRankError(f"branch state {st}: {type(exc).__name__}: {exc}") from None
        arts[str(st)] = {n: sha256_file(st / n) for n in ("plan.json", "branch_plan.jsonl")}
        pp = st / PEER_DROPS_NAME
        if pp.is_file():
            have += 1
            cls += _jsonl(pp)
            arts[str(st)][PEER_DROPS_NAME] = sha256_file(pp)
        elif need_peer:
            raise VRankError(f"{pp}: missing (run 'classify' on every shard once its branches are played)")
    names = [r["unit"] for r in rows]
    if len(set(names)) != len(names):
        raise VRankError("a branch unit is planned in two shards")
    if have != len(list(branch_states)):
        return rows, None, arts
    try:
        rows, peer = apply_peer_drops(rows, cls)
    except Exception as exc:        # noqa: BLE001 - its RuleInputError is main.reads'; the CLI names it the same way
        raise VRankError(str(exc)) from None
    return rows, peer, arts


# ---------------------------------------------------------------------------------------------------- stage 0
def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(np.mean(xs)) if len(xs) else None


def sizing(s_bar: float) -> Dict[str, Any]:
    """The design's §5.4 rule on s̄ (outcome-free): the final block count, and whether the cap binds."""
    if not s_bar > 0:
        raise VRankError("s̄ is not positive: no searched position")
    blocks = BLOCKS * min(2.0, max(1.0, S_BAR_REF / s_bar))
    return {"s_bar": s_bar, "final_blocks": int(math.ceil(blocks / BLOCK_STEP - 1e-9) * BLOCK_STEP),
            "cap_binds": bool(s_bar < S_BAR_REF / 2), "rule": f"{BLOCKS} x min(2, max(1, {S_BAR_REF} / s̄)), up to a "
                                                              f"multiple of {BLOCK_STEP}"}


def stage0(selected: Sequence[Mapping[str, Any]], scored: Sequence[Mapping[str, Any]],
           rows: Sequence[Mapping[str, Any]], params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The outcome-free look: no y, no slope. Q̂'s spread, the exclusion rate, the legal count, the CPU of every part,
    the branches' statuses and G-BRANCH. With ``params`` (the registered ones) every tooth is also evaluated."""
    per: Dict[str, Any] = {}
    spread_all: List[float] = []
    for panel in sorted({r["panel"] for r in selected}):
        sc = [r for r in scored if r["panel"] == panel]
        se = [r for r in sc if r["search"].get("searched")]
        spread = [float(((v - v.mean()) ** 2).sum()) for v in
                  (np.asarray(list(r["search"]["q"].values()), dtype=np.float64) for r in se)]
        spread_all += spread
        br = [r for r in rows if r["panel"] == panel]
        cpu = [float(r["cpu_s"]) for r in br if r.get("cpu_s") is not None]
        per[panel] = {
            "selected": sum(1 for r in selected if r["panel"] == panel), "scored": len(sc), "searched": len(se),
            "exclusion_rate": (len(sc) - len(se)) / len(sc) if sc else None,
            "legal_actions_mean": _mean([len(r["search"]["q"]) for r in se]), "sum_qc2_mean": _mean(spread),
            "search_cpu_s_mean": _mean([float(r["search"]["cpu_s"]) for r in sc if r["search"].get("cpu_s") is not None]),
            "referee_cpu_s_mean": _mean([float(r["referee"]["cpu_s"]) for r in se if r.get("referee")]),
            "units": len(br), "branches": sum(1 for r in br if r["kind"] == "branch"),
            "status": {s: sum(1 for r in br if r["status"] == s) for s in ("counted", "dropped", "pending")},
            "attempts": int(sum(int(r["attempts"]) for r in br)),
            "g_branch_fail_units": sum(1 for r in br if r["g_branch_fails"]),
            "g_branch_peer_units": sum(1 for r in br if r.get("g_branch_peer")),
            "unit_cpu_s_mean": _mean(cpu), "unit_cpu_s_total": float(np.sum(cpu)) if cpu else 0.0}
    out: Dict[str, Any] = {"schema": "gen3_v_rank_stage0_v1", "per_panel": per,
                           "searched_positions": len(spread_all), "reads_no_outcome": True}
    if spread_all:
        out["sizing"] = sizing(float(np.mean(spread_all)))
        tot_s = sum(float(r["search"].get("cpu_s") or 0.0) + float((r.get("referee") or {}).get("cpu_s") or 0.0)
                    for r in scored)
        tot_b = sum(float(r["cpu_s"]) for r in rows if r.get("cpu_s") is not None)
        out["cpu_s_per_searched_position"] = {"shadow_arms": tot_s / len(spread_all),
                                              "branches_and_nulls": tot_b / len(spread_all),
                                              "all": (tot_s + tot_b) / len(spread_all)}
    if params is not None:
        out["teeth"] = void_reasons(selected, scored, rows, params)
    return out


# ------------------------------------------------------------------------------------------------------ read
def void_reasons(selected: Sequence[Mapping[str, Any]], scored: Sequence[Mapping[str, Any]],
                 rows: Sequence[Mapping[str, Any]], params: Mapping[str, Any]) -> List[str]:
    """The teeth (module docstring), each a named VOID. ``rows``: :func:`load_branches`'s. ``params``: ``engine``,
    ``preset``, ``u_panels``, ``home_panel``."""
    why: List[str] = []
    panels = {*params["u_panels"], params["home_panel"]}
    off = sorted({r["panel"] for r in selected} - panels)
    if off:
        why.append(f"panels: selected states on {off}, the registered panels are {sorted(panels)}")
    wrong = [r["key"] for r in selected if r["select_hash"] != select_hash(r["deal"], r["turn"], r["n"])
             or r["stratum"] != stratum_of(r["turn"])]
    if wrong:
        why.append(f"selection: {len(wrong)} state(s) whose hash or stratum is not their draw's (first {wrong[0]})")
    slots = [(r["deal"], r["stratum"]) for r in selected]
    if len(set(slots)) != len(slots):
        why.append("selection: two states of one (deal, stratum)")
    teams = {(r["deal"], r.get("team")) for r in selected}
    if any(not t for _, t in teams) or len(teams) != len({d for d, _ in teams}):
        why.append("selection: a state without the team we piloted, or a deal on two teams (the team is the cluster "
                   "of every interval)")
    team_of = {r["key"]: r.get("team") for r in selected}
    moved = [r["key"] for r in scored if r["key"] in team_of and r.get("team") != team_of[r["key"]]]
    if moved:
        why.append(f"scored: {len(moved)} scored state(s) on another team than their selected state's (first "
                   f"{moved[0]})")
    sel = {r["key"] for r in selected}
    stray = [r["key"] for r in scored if r["key"] not in sel]
    if stray:
        why.append(f"scored: {len(stray)} scored state(s) never selected (first {stray[0]})")
    searched = [r for r in scored if r["search"].get("searched")]
    preset = params.get("preset")
    bad = [r["key"] for r in searched if (r["search"].get("config") or {}).get("name") != preset]
    if bad:
        why.append(f"search: {len(bad)} searched state(s) not on the registered preset {preset!r} (first {bad[0]})")

    def shadow_ok(r: Mapping[str, Any]) -> bool:
        acts = sorted(r["search"]["q"])
        ref, w = r.get("referee") or {}, r.get("worlds") or {}
        return (ref.get("preset") == ORACLE and ref.get("seeds") == ORACLE_SEEDS and sorted(ref.get("out", {})) == acts
                and all(len(v) == ORACLE_SEEDS for v in ref["out"].values()) and sorted(w) == acts
                and len({len(v) for v in w.values()}) == 1)

    bad = [r["key"] for r in searched if not shadow_ok(r)]
    if bad:
        why.append(f"shadow arms: {len(bad)} searched state(s) without their per-world values or their {ORACLE} "
                   f"record of {ORACLE_SEEDS} seeds an action (first {bad[0]})")
    want = {(r["key"], a) for r in searched for a in (int(x) for x in r["search"]["q"]) if a != int(r["pi_argmax"])}
    have = [(r["key"], int(r["action"])) for r in rows if r["kind"] == "branch"]
    if set(have) != want or len(set(have)) != len(have):
        why.append(f"plan: {len(want - set(have))} legal action(s) of searched states without a branch, "
                   f"{len(set(have) - want)} branch(es) the scores do not call for, {len(have) - len(set(have))} "
                   "planned twice")
    gb = [r["unit"] for r in rows if r["g_branch_fails"]]
    if gb:
        why.append(f"G-BRANCH: {len(gb)} unit(s) failed the live prefix check outside a classified peer drop (first "
                   f"{gb[0]})")
    counted = [r for r in rows if r["status"] == "counted"]
    cap = [r["unit"] for r in counted if not r["capture"]["ok"]]
    if cap:
        why.append(f"capture: {len(cap)} counted branch(es) whose protocol is not the raw game's through the branch "
                   f"turn (first {cap[0]})")
    nul = [r["unit"] for r in counted if r["kind"] == "null"
           and (r["y_branch"] != r["y_raw"] or r["branch_turns"] != r["raw_turns"])]
    if nul:
        why.append(f"null control: {len(nul)} replay(s) of π's own action changed the outcome or length (first "
                   f"{nul[0]})")
    forced = [r["unit"] for r in counted if r.get("forced") != r["action"] or r.get("spec_mismatch")
              or r.get("branch_index") != r["game"]]
    if forced:
        why.append(f"forced action: {len(forced)} unit(s) did not play their planned action in their raw game's "
                   f"series index (first {forced[0]})")
    reg = params.get("engine")
    eng = {r.get("raw_engine") for r in counted} | {r.get("branch_engine") for r in counted}
    if None in eng or "" in eng:
        why.append("engine: a counted game on an unknown engine")
    known = sorted(str(e) for e in eng if e)
    if len(known) > 1 or (reg and known and known != [reg]):
        why.append(f"engine: the counted games played on {known}, the registered one is {reg}")
    return why


def _extras(selected: Sequence[Mapping[str, Any]], scored: Sequence[Mapping[str, Any]],
            rows: Sequence[Mapping[str, Any]], panels: Sequence[str]) -> Dict[str, Any]:
    """The §4.4 rows that are counts, not slopes: the world quality, the stratum-B share, the null controls, CPU."""
    out: Dict[str, Any] = {}
    for p in panels:
        se = [r for r in scored if r["panel"] == p and r["search"].get("searched")]
        deals = {r["deal"] for r in selected if r["panel"] == p}
        b = {r["deal"] for r in selected if r["panel"] == p and r["stratum"] == "B"}
        nl = [r for r in rows if r["panel"] == p and r["kind"] == "null"]
        cpu = [float(r["cpu_s"]) for r in rows if r["panel"] == p and r.get("cpu_s") is not None]
        hits = [float(np.mean(r["search"]["species_hit"])) for r in se if r["search"].get("species_hit")]
        out[p] = {"deals_with_a_position": len(deals), "share_with_stratum_B": len(b) / len(deals) if deals else None,
                  "species_hit_mean": _mean(hits),
                  "search_cpu_s": float(sum(float(r["search"].get("cpu_s") or 0.0) for r in scored if r["panel"] == p)),
                  "referee_cpu_s": float(sum(float(r["referee"]["cpu_s"]) for r in se if r.get("referee"))),
                  "branch_cpu_s": float(np.sum(cpu)) if cpu else 0.0,
                  "nulls": len(nl), "nulls_counted": sum(1 for r in nl if r["status"] == "counted"),
                  "nulls_dropped": sum(1 for r in nl if r["status"] == "dropped")}
    return out


def read(selected: Sequence[Mapping[str, Any]], scored: Sequence[Mapping[str, Any]],
         rows: Sequence[Mapping[str, Any]], params: Mapping[str, Any]) -> Dict[str, Any]:
    """The registered decision (:func:`v_rank_stat.read`) behind the teeth. ``params``: ``u_panels``, ``home_panel``,
    ``look``, ``null_bar``, ``margin``, ``rule8_eps``, ``bootstrap`` {B, seed}, ``max_drop_frac``, ``engine``,
    ``preset``. An unscored selected state or a pending branch REFUSES; a failing tooth returns VOID."""
    done = {r["key"] for r in scored}
    todo = [r["key"] for r in selected if r["key"] not in done]
    if todo:
        raise VRankError(f"{len(todo)} selected state(s) not yet scored (first {todo[0]})")
    pend = [r["unit"] for r in rows if r["status"] == "pending"]
    if pend:
        raise VRankError(f"{len(pend)} branch unit(s) not yet played to a verdict (first {pend[0]})")
    why = void_reasons(selected, scored, rows, params)
    if why:
        return {"schema": VERDICT_SCHEMA, "verdict": VOID, "void": why}
    positions, counts = ST.build_positions(scored, rows)
    cap = float(params["max_drop_frac"])
    for p, c in sorted(counts.items()):
        if c["searched"] and c["incomplete"] / c["searched"] > cap:
            why.append(f"drops: {c['incomplete']} of {c['searched']} searched {p} positions have a branch that never "
                       f"counted (> {cap:g})")
    if why:
        return {"schema": VERDICT_SCHEMA, "verdict": VOID, "void": why, "counts": counts}
    res = ST.read(positions, params)
    panels = [*params["u_panels"], params["home_panel"]]
    for p, c in counts.items():
        c["exclusion_rate"] = c["excluded"] / c["scored"] if c["scored"] else None
    return {"schema": VERDICT_SCHEMA, "verdict": res["class"], "counts": counts,
            "extras": _extras(selected, scored, rows, panels), **res}


# ------------------------------------------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.value_search.v_rank", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="verb", required=True)
    a = sub.add_parser("select")
    a.add_argument("--raw-state", action="append", required=True, help="a raw kakuna state <state>/anchor/<g>/<label>")
    a.add_argument("--games", default=None, help="LO-HI: draw from these games of each unit only (a smoke)")
    a.add_argument("--out", required=True)
    b = sub.add_parser("score")
    b.add_argument("--selected", required=True)
    b.add_argument("--model", required=True)
    b.add_argument("--search", required=True)
    b.add_argument("--engine", required=True, help="the registered engine stamp: the scorer refuses another")
    b.add_argument("--out-dir", required=True, help="the scored dir: this part appends to part_<i>of<K>.jsonl in it")
    b.add_argument("--part", default="0/1", help="i/K: this scorer's share of the states, by draw hash")
    b.add_argument("--games", default=None, help="LO-HI: only these games (stage 0)")
    b.add_argument("--unit-seeds", default=None, help="a comma list: only these unit seeds (stage 0)")
    b.add_argument("--budget-s", type=float, default=None,
                   help="start no new state after this many seconds (a bounded chunk ends cleanly; resumable)")
    c = sub.add_parser("plan")
    c.add_argument("--selected", required=True)
    c.add_argument("--scored-dir", required=True)
    c.add_argument("--raw-state", action="append", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--shards", type=int, default=1)
    d = sub.add_parser("classify")
    d.add_argument("--branch-state", action="append", required=True)
    d.add_argument("--all", action="store_true", help="classify every unit, not only nulls and uncounted branches")
    root = "a plan root (every shard of it) or one shard state; repeatable"
    for name in ("stage0", "read"):
        e = sub.add_parser(name)
        e.add_argument("--selected", required=True)
        e.add_argument("--scored-dir", required=True)
        e.add_argument("--branch-state", action="append", required=True, help=root)
        e.add_argument("--params", required=(name == "read"), default=None)
        e.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    try:
        return _run(args)
    except VRankError as exc:
        print(f"[v_rank] REFUSED: {exc}", file=sys.stderr)
        return 2


def _emit(doc: Any, out: Optional[str]) -> int:
    txt = json.dumps(doc, indent=1, sort_keys=True, default=str)
    if out:
        Path(out).write_text(txt + "\n")
    print(txt)
    return 0


def _run(args: Any) -> int:
    if args.verb == "select":
        out = Path(args.out)
        if out.exists():
            raise VRankError(f"{out}: a selection is written once (frozen)")
        sel = select_states([Path(x) for x in args.raw_state], parse_range(args.games))
        _atomic(out, "".join(json.dumps(r, sort_keys=True) + "\n" for r in sel))
        by: Dict[str, Dict[str, int]] = {}
        for r in sel:
            c = by.setdefault(r["panel"], {"A": 0, "B": 0})
            c[r["stratum"]] += 1
        print(json.dumps({"states": len(sel), "per_panel": by}))
        return 0
    if args.verb == "score":
        return _score(args)
    if args.verb == "plan":
        rep = plan_branches(_jsonl(Path(args.selected)), load_scored(scored_files(Path(args.scored_dir))),
                            [Path(x) for x in args.raw_state], Path(args.out), args.shards)
        print(json.dumps(rep, sort_keys=True))
        return 0
    if args.verb == "classify":
        tot: Dict[str, int] = {}
        for st in expand_shards(args.branch_state):
            cls = classify_state(st, everything=args.all)
            _atomic(st / PEER_DROPS_NAME, "".join(json.dumps(r, sort_keys=True) + "\n" for r in cls))
            for r in cls:
                tot[r["class"]] = tot.get(r["class"], 0) + 1
        print(json.dumps({"classified_attempts": sum(tot.values()), "classes": tot}, sort_keys=True))
        return 0
    sel = _jsonl(Path(args.selected))
    scored = load_scored(scored_files(Path(args.scored_dir)))
    params = json.loads(Path(args.params).read_text()) if args.params else None
    rows, peer, _arts = load_branches(expand_shards(args.branch_state), need_peer=(args.verb == "read"))
    if args.verb == "stage0":
        return _emit(stage0(sel, scored, rows, params), args.out)
    res = read(sel, scored, rows, params)
    res["peer_drops"] = peer
    return _emit(res, args.out)


def _score(args: Any) -> int:
    from main.anchors.search_presets import canonical
    from main.anchors.server import engine_stamp

    search = canonical(args.search)
    part = parse_part(args.part)
    here = engine_stamp()
    if here != args.engine:
        raise VRankError(f"this checkout's engine is {here}, the registered one {args.engine}: no state is scored")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = out_dir / "engine.txt"
    if stamp.exists() and stamp.read_text().strip() != here:
        raise VRankError(f"{stamp} records {stamp.read_text().strip()}, this checkout's engine is {here}")
    stamp.write_text(here + "\n")
    out = out_dir / f"part_{part[0]}of{part[1]}.jsonl"
    done = [r["key"] for r in load_scored(scored_files(out_dir))]
    seeds = None if args.unit_seeds is None else [int(x) for x in str(args.unit_seeds).split(",") if x.strip()]
    todo = to_score(_jsonl(Path(args.selected)), done, part=part, games=parse_range(args.games), unit_seeds=seeds)
    t0 = time.monotonic()
    sc: Optional[Scorer] = None
    n = 0
    try:
        for r in todo:
            if args.budget_s is not None and time.monotonic() - t0 > args.budget_s:
                print(f"[v_rank] budget {args.budget_s:g} s spent: {n} scored this chunk", flush=True)
                break
            if sc is None:
                sc = Scorer(args.model, search)
            rec = sc.score(r)
            with open(out, "a") as f:
                f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
                f.flush()
            n += 1
            # progress only: no value is printed while scoring
            print(f"[v_rank] {n}/{len(todo)} {r['panel']} search cpu {rec['search'].get('cpu_s')} referee cpu "
                  f"{(rec.get('referee') or {}).get('cpu_s')}", flush=True)
        print(json.dumps({"scored_this_chunk": n, "left": len(todo) - n, "out": str(out)}))
        return 0
    finally:
        if sc is not None:
            sc.close()


if __name__ == "__main__":
    sys.exit(main())
