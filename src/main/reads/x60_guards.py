r"""X60-T look 2's GAIN-to-ADOPT guards (PREREG ``x60t_h_teams_2026-11-06`` §11, §23.4, §24 AM-X60T-3): the filled
guards artefact rule ``x60`` reads, and the RE-DECISION spec that reads it. Run ONLY on a look-2 Δ_F GAIN, after the
look-2 state is assembled; nothing here plays a game or decides a verdict.

    python -m main.reads.x60_guards gf-cells --state <look-2 state> --out <cells.json>
    python -m main.reads.x60_guards build --state <look-2 state> --gf-ledger <root> --gf-family <id> \
        --gf-request <id> --gc <gc.json> --out <guards.json>
    python -m main.reads.x60_guards redecide-spec --state <look-2 state> --spec <look-2 spec> --guards <guards.json> \
        --read <id> --out <spec.json>

THE ARTEFACT (schema ``rules_x60.GUARDS_SCHEMA``, status ``PLAYED``). Each block is what ``rules_x60`` re-checks
against the read's own resolved finals (``models.json`` sha256s); a block it cannot tie to them reads INCONCLUSIVE:

* ``g_f`` -- X35's G-F (``design_compute_efficiency.md`` §5.6.4; launch sheet §4.5's settings: ``main.h2h play-many
  --pairs 500 --purpose ab``, greedy vs greedy, schedule seed 0, both sides on the default eval team builder):
  :func:`gf_cells` lists the 64 cells (rows the ON finals, columns the OFF finals, each in seed order, by the state's
  resolved files); :func:`gf_block` reads the request's ledger cells (``cross.look_cells``), builds ``h_pp`` with
  ``cross.matrix_from_cells(..., min_pairs=readout.MIN_PAIRS)`` (500; F-AMX-2) and keeps every cell's pentanomial,
  so the rule re-derives ``h_pp``;
* ``r2`` -- G-E's ``main.x35.ge.r2_input("x60t")`` (``ge_x60t`` / ``ge_c0``, §23.5), plus each reader's target file
  hashed (``targets``);
* ``g_c`` -- G-C's ``python -m main.x35.gc pool --side production`` record over the eight seed pairs (§21.5), which
  carries each read's ``checkpoint_sha256``.

THE RE-DECISION (§23.4). A state is one spec, so the filled artefact is read in a NEW state: :func:`redecide_spec`
turns look 2's spec into one that IMPORTS every cell look 2 played (a group that already imported keeps its import
for those models and gets a ``<group>_l2`` twin for the models look 2 played: an import is never transitive), RE-PLAYS
only the determinism guard's unit (``determinism_guard.group`` takes no import), pins every final by the sha256 the
look-2 state resolved (FINDING 24), and registers the filled artefact. Nothing else changes.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: The protocol G-F's cells are played at (``main.h2h play-many``, mirrored, greedy vs greedy).
GF_PROTOCOL = "gen3_eval_protocol_v1_h2h"
#: §23.5: the G-E target seed and its readers' arm (``main.x35.ge``).
GE_ARM = "x60t"
#: the suffix of a group split off a group that already imported (§23.4).
L2_SUFFIX = "_l2"
#: the suffix of an S group's eval-ledger family in the re-decision spec (one family per S group).
FAMILY_SUFFIX = "_guards"


class GuardsError(ValueError):
    """An input the artefact or the re-decision cannot be built from (named; nothing is written)."""


# ------------------------------------------------------------------------------------------------ the look-2 state
def state_read(state: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(spec, models)`` of a look-2 state: its ``spec.json`` (rule ``x60``, look 2) and ``models.json``."""
    from main.reads import layout as LY

    spec = json.loads(LY.spec_file(state).read_text())
    models = json.loads(LY.models_file(state).read_text())
    dec = spec.get("decision") or {}
    if dec.get("rule") != "x60" or (dec.get("params") or {}).get("look") != 2:
        raise GuardsError(f"{state}: not a rule x60 look-2 state (rule {dec.get('rule')!r})")
    return spec, models


def sides(spec: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    """The ON / OFF labels in seed order."""
    p = spec["decision"]["params"]
    seed = {lab: int(spec["models"][lab]["seed"]) for lab in list(p["on"]) + list(p["off"])}
    return sorted(p["on"], key=seed.__getitem__), sorted(p["off"], key=seed.__getitem__)


def finals(spec: Mapping[str, Any], models: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Every ON / OFF final as the state resolved it: ``{label: {seed, sha256, source}}``; a final the state has not
    resolved yet refuses."""
    on, off = sides(spec)
    out: Dict[str, Dict[str, Any]] = {}
    for lab in on + off:
        m = models.get(lab) or {}
        if not m.get("sha256") or not m.get("source"):
            raise GuardsError(f"{lab}: the state has not resolved its final (models.json holds {sorted(m)})")
        out[lab] = {"seed": int(spec["models"][lab]["seed"]), "sha256": str(m["sha256"]), "source": str(m["source"])}
    return out


# ------------------------------------------------------------------------------------------------ G-F
def gf_cells(spec: Mapping[str, Any], models: Mapping[str, Any]) -> List[List[str]]:
    """G-F's ``play-many --cells`` list: ``[ON final, OFF final]`` for every pair, rows then columns in seed order (the
    resolved source files, whose run dirs carry the ``metadata.json`` ``main.h2h`` checks the team source against)."""
    on, off = sides(spec)
    fin = finals(spec, models)
    return [[fin[r]["source"], fin[c]["source"]] for r in on for c in off]


def gf_block_from_cells(cells: Sequence[Any], spec: Mapping[str, Any], models: Mapping[str, Any], *,
                        family: str, request: str) -> Dict[str, Any]:
    """The ``g_f`` block from ``eval_ledger`` cells (module docstring)."""
    from main.h2h import cross as CR
    from main.x35 import readout as R

    on, off = sides(spec)
    fin = finals(spec, models)
    by_sha = {v["sha256"]: lab for lab, v in fin.items()}
    mat = CR.matrix_from_cells(cells, [fin[r]["sha256"] for r in on], [fin[c]["sha256"] for c in off],
                               min_pairs=R.MIN_PAIRS)
    rows = []
    for c in cells:
        rows.append({"row": by_sha.get(c.player), "col": by_sha.get(c.opponent), "player_sha256": c.player,
                     "opponent_sha256": c.opponent, "player_id": c.player_id, "opponent_id": c.opponent_id,
                     "pairs": list(c.pairs) if c.pairs is not None else None, "w": c.w, "l": c.l, "d": c.d,
                     "aborted": c.aborted, "verdict": c.verdict, "reasons": list(c.reasons)})
    return {"family": family, "request": request, "protocol": GF_PROTOCOL, "min_pairs": R.MIN_PAIRS,
            "rows": on, "cols": off, "h_pp": [list(r) for r in mat.h] if mat.h is not None else None,
            "inconclusive": list(mat.inconclusive), "cells": rows}


def gf_block(ledger: Path, family: str, request: str, spec: Mapping[str, Any], models: Mapping[str, Any]
             ) -> Dict[str, Any]:
    from main.h2h import cross as CR

    return gf_block_from_cells(CR.look_cells(ledger, family, request, GF_PROTOCOL), spec, models,
                               family=family, request=request)


# ------------------------------------------------------------------------------------------------ G-E, G-C
def r2_block(models_dir: str, *, reader: Any = None) -> Dict[str, Any]:
    """``main.x35.ge.r2_input("x60t")`` (it refuses unmatched or unfinished readers) plus each reader's target file
    hashed, so the rule ties G-E to the read's seed-3501 finals."""
    from main.x35 import ge as GE
    from utils.hashing import sha256_file

    try:
        r2 = GE.r2_input(GE_ARM, GE.TARGET_SEEDS[0], models_dir=models_dir, reader=reader)
    except GE.ReaderError as exc:
        raise GuardsError(f"G-E: {exc}") from exc
    targets = {}
    for side, rd in r2["readers"].items():
        f = rd.get("target_file")
        if not f or not os.path.isfile(f):
            raise GuardsError(f"G-E: the {side} reader's target {f!r} is not a file")
        targets[side] = {"file": str(f), "sha256": sha256_file(Path(f))}
    return dict(r2, targets=targets)


def gc_block(path: Path) -> Dict[str, Any]:
    from main.x35 import gc as GC

    rec = json.loads(Path(path).read_text())
    if not isinstance(rec, dict) or rec.get("schema") != GC.GC_SCHEMA:
        raise GuardsError(f"G-C: {path} is not a {GC.GC_SCHEMA} record (python -m main.x35.gc pool)")
    return rec


# ------------------------------------------------------------------------------------------------ the artefact
def _commit() -> Optional[str]:
    from utils.paths import repo_path

    try:
        return subprocess.run(["git", "-C", str(repo_path(".")), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def artefact(state: Path, spec: Mapping[str, Any], models: Mapping[str, Any], *, g_f: Mapping[str, Any],
             r2: Mapping[str, Any], g_c: Mapping[str, Any]) -> Dict[str, Any]:
    """The filled artefact (schema ``rules_x60.GUARDS_SCHEMA``, status PLAYED)."""
    from main.reads import layout as LY
    from main.reads import rules_x60 as X
    from utils.hashing import sha256_file

    fin = finals(spec, models)
    return {"schema": X.GUARDS_SCHEMA, "status": X.PLAYED, "read": spec.get("read"),
            "look2_state": {"path": str(state), "spec_sha256": sha256_file(LY.spec_file(state))},
            "finals": {lab: {"seed": v["seed"], "sha256": v["sha256"]} for lab, v in fin.items()},
            "built_at": _commit(),
            "note": "AM-X60T-3 (PREREG §24): G-F / G-E / G-C of look 2's finals, produced only on a look-2 Δ_F GAIN; "
                    "rule x60 re-checks every block against the re-decision state's resolved finals.",
            "g_f": dict(g_f), "r2": dict(r2), "g_c": dict(g_c)}


# ------------------------------------------------------------------------------------------------ the re-decision
def _pinned(source: str, sha: str) -> Dict[str, Any]:
    from utils.paths import main_models_dir

    root = main_models_dir()
    path = source
    if root is not None:
        try:
            path = "models/" + Path(source).resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            path = source
    return {"kind": "pinned", "path": path, "sha256": sha}


def redecide_spec(spec: Mapping[str, Any], models: Mapping[str, Any], *, state: str, spec_sha256: str,
                  guards: Mapping[str, str], read: str) -> Dict[str, Any]:
    """Look 2's spec as the re-decision (§23.4; module docstring). ``models`` is the look-2 state's ``models.json``,
    ``state`` / ``spec_sha256`` name the look-2 state as the imports will, ``guards`` the filled artefact."""
    p = spec["decision"]["params"]
    det = (p.get("determinism_guard") or {}).get("group")
    out = copy.deepcopy(dict(spec))
    out["read"] = read
    out["note"] = (f"X60-T look 2's RE-DECISION with the filled guards (AM-X60T-3, PREREG §24; built by "
                   f"main.reads.x60_guards from {spec.get('read')}): every cell look 2 played or imported is imported "
                   f"(groups that imported keep that import; their look-2-played models are <group>{L2_SUFFIX}); "
                   f"the determinism unit is re-played; every final pinned by its look-2 sha256.")
    new_models: Dict[str, Any] = {}
    for lab, m in spec["models"].items():
        got = models.get(lab) or {}
        if not got.get("sha256") or not got.get("source"):
            raise GuardsError(f"{lab}: the look-2 state has not resolved its final")
        new_models[lab] = dict(m, ref=_pinned(str(got["source"]), str(got["sha256"])))
    out["models"] = new_models
    split: Dict[str, List[str]] = {}
    for lane in ("anchor", "s"):
        groups: Dict[str, Any] = {}
        for g, d in (spec.get(lane) or {}).items():
            if lane == "anchor" and g == det:
                groups[g] = copy.deepcopy(d)
                split[g] = [g]
                continue
            imp = d.get("import")
            played = [m for m in d["models"] if m not in ((imp or {}).get("models") or ())]
            names: List[str] = []
            if imp is not None:
                kept = dict(copy.deepcopy(d), models=list(imp["models"]))
                if lane == "s":
                    kept["family"] = d["family"] + FAMILY_SUFFIX
                groups[g] = kept
                names.append(g)
            if played:
                name = g + L2_SUFFIX if imp is not None else g
                if name in (spec.get(lane) or {}) and name != g:
                    raise GuardsError(f"{lane}.{name}: the split name of {g} is taken")
                twin = dict(copy.deepcopy(d), models=played,
                            **{"import": {"state": state, "spec_sha256": spec_sha256, "group": g, "models": played}})
                if lane == "s":
                    twin["family"] = d["family"] + (L2_SUFFIX if imp is not None else "") + FAMILY_SUFFIX
                groups[name] = twin
                names.append(name)
            split[g] = names
        out[lane] = groups
    q = copy.deepcopy(dict(p))

    def expand(gs: Sequence[str]) -> List[str]:
        return [n for g in gs for n in split[g]]

    for key in ("primary", "goal_a", "p", "i_f", "r72", "contrast"):
        if q.get(key) is not None:
            q[key]["groups"] = expand(q[key]["groups"])
    if q.get("reported"):
        q["reported"] = expand(q["reported"])
    q["guards"] = {"path": str(guards["path"]), "sha256": str(guards["sha256"])}
    out["decision"] = dict(spec["decision"], params=q)
    return out


def render(doc: Mapping[str, Any]) -> str:
    """A spec's committed bytes (indent 1, UTF-8; the look-2 spec's form)."""
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


# ------------------------------------------------------------------------------------------------ CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.reads.x60_guards", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("gf-cells", help="G-F's play-many --cells file (ON x OFF finals, seed order)")
    c.add_argument("--state", required=True)
    c.add_argument("--out", required=True)
    b = sub.add_parser("build", help="the filled guards artefact (exit 2 = REFUSED, nothing written)")
    b.add_argument("--state", required=True)
    b.add_argument("--gf-ledger", required=True)
    b.add_argument("--gf-family", required=True)
    b.add_argument("--gf-request", required=True)
    b.add_argument("--gc", required=True, help="the main.x35.gc pool record")
    b.add_argument("--models-dir", default=None)
    b.add_argument("--out", required=True)
    r = sub.add_parser("redecide-spec", help="the re-decision spec (§23.4) over the filled artefact")
    r.add_argument("--state", required=True)
    r.add_argument("--spec", required=True, help="look 2's committed spec (its bytes must be the state's spec.json)")
    r.add_argument("--guards", required=True, help="the filled artefact, repo-relative as the spec will name it")
    r.add_argument("--read", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from utils.hashing import sha256_file

    try:
        state = Path(os.path.expanduser(a.state))
        spec, models = state_read(state)
        if a.cmd == "gf-cells":
            cells = gf_cells(spec, models)
            Path(a.out).write_text(json.dumps(cells, indent=1) + "\n")
            print(f"[x60 guards] {len(cells)} G-F cells -> {a.out}")
            return 0
        if a.cmd == "build":
            from utils.paths import main_models_dir

            md = a.models_dir or str(main_models_dir() or "")
            doc = artefact(state, spec, models,
                           g_f=gf_block(Path(a.gf_ledger), a.gf_family, a.gf_request, spec, models),
                           r2=r2_block(md), g_c=gc_block(Path(a.gc)))
            Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
            print(f"[x60 guards] artefact -> {a.out} sha256 {sha256_file(Path(a.out))}"
                  + (f"; G-F inconclusive: {doc['g_f']['inconclusive'][:3]}" if doc["g_f"]["inconclusive"] else ""))
            return 0
        from main.reads import layout as LY
        from main.reads.models import spec_path

        if Path(a.spec).read_bytes() != LY.spec_file(state).read_bytes():
            raise GuardsError(f"{a.spec} is not the state's spec.json")
        doc = redecide_spec(spec, models, state=a.state, spec_sha256=sha256_file(LY.spec_file(state)),
                            guards={"path": a.guards, "sha256": sha256_file(spec_path(a.guards))}, read=a.read)
        Path(a.out).write_text(render(doc))
        print(f"[x60 guards] re-decision spec -> {a.out} sha256 {sha256_file(Path(a.out))}")
        return 0
    except (GuardsError, OSError, ValueError, KeyError) as exc:
        print(f"[x60 guards] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
