"""PILOT B's lane: Metamon's Kakuna pilots BOTH sides, through the anchors' B-D9 asymmetric PAIR CELL.

THE CELL. Side A (the reported side, ``--our-teamset away``) is Kakuna on a rated set's registered SUBSET; side B
(``--teamset home``) is Kakuna on the REFERENCE panel. Both are installed as Metamon team sets (byte copies of the
registered files, :func:`install`), and each unit's config names them; a unit is one ``python -m main.anchors``
invocation of 20 role-balanced games (``--declare-asymmetric-teams``, greedy vs greedy, CPU, ``--team-seed =
--seed-base =`` the plan's seed, ``--capture-dir``), run from a PINNED checkout root under ``nice -n 19 taskset -c
5-7,13-15 scripts/ops/mem_cap.sh`` (rule 5b), one at a time.

WHO PLAYED WHICH TEAM. Metamon draws each side's team itself, so the record of a game is the front end's CAPTURE: its
START command carries both packed teams and the two player names (side A's carries ``Away``, side B's ``Home``), and
its battle log names the winner (``|win|<name>``). Each packed team is exported to paste text by Showdown's own
``Teams.export`` and matched to its side's declared set by identity (``panel.team_id``), falling back to the
species + moves key when that key is UNIQUE in the side's set (the tier is recorded). A unit COUNTS iff
``main.anchors.results.unit_counts`` says so against the registered per-side content digests, AND every capture
matches on both sides, AND the captures' winners agree game for game with the unit's ``games.jsonl`` results.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from main.team_strength import panel as PN

GAMES = 20
CORES = "5-7,13-15"
MEM_GB = "12"
METAMON_FILE = ".gen3ou_team"
_WIN_RE = re.compile(r"\|win\|([^\n|\"\\]+)")
_EXPORT_JS = ("const {Teams}=require(process.argv[1]);const ls=JSON.parse(require('fs').readFileSync(0,'utf8'));"
              "process.stdout.write(JSON.stringify(ls.map(p=>Teams.export(Teams.unpack(p)))));")


class KakunaLaneError(RuntimeError):
    pass


def ref_files(reg: Mapping[str, Any]) -> Dict[str, Path]:
    """``{metamon file name: source path}`` of the reference panel."""
    out = {}
    for m in reg["reference"]["members"]:
        out[f"{m['set']}__{m['stem']}{METAMON_FILE}"] = Path(reg["sets"][m["set"]]["dir"]) / m["file"]
    return out


def subset_files(reg: Mapping[str, Any], set_name: str) -> Dict[str, Path]:
    blk = reg["sets"][set_name]
    want = set(reg["pilots"]["kakuna"]["subsets"][set_name])
    out: Dict[str, Path] = {}
    seen = set()
    for stem, tid in sorted(blk["team_ids"].items()):
        if tid in want and tid not in seen:
            seen.add(tid)
            fname = blk["file_names"][stem]
            out[f"{set_name}__{stem}{METAMON_FILE}"] = Path(blk["dir"]) / fname
    if seen != want:
        raise KakunaLaneError(f"subset {set_name}: {len(want - seen)} registered ids have no file")
    return out


def metamon_set_dir(teams_root: Path, name: str) -> Path:
    return Path(teams_root) / name / "gen3ou"


def install_one(dst: Path, files: Mapping[str, Path]) -> str:
    """Byte-copy ``files`` into ``dst`` (created); an EXISTING dst must already hold exactly those bytes (never
    overwritten). Returns the dir's ``gen3_team_set_digest_v1``."""
    from main.anchors.session import team_set_digest

    if dst.exists():
        have = sorted(p.name for p in dst.iterdir())
        if have != sorted(files) or any((dst / n).read_bytes() != src.read_bytes() for n, src in files.items()):
            raise KakunaLaneError(f"{dst} exists and is not the registered set — refusing to overwrite it")
    else:
        dst.mkdir(parents=True)
        for n, src in files.items():
            shutil.copyfile(src, dst / n)
    return team_set_digest(dst)


def install(reg: Mapping[str, Any], teams_root: Path) -> Dict[str, str]:
    """Install the reference set and every subset as Metamon team sets; returns ``{metamon set: digest}``."""
    k = reg["pilots"]["kakuna"]
    out = {k["ref_set"]: install_one(metamon_set_dir(teams_root, k["ref_set"]), ref_files(reg))}
    for s, name in k["subset_sets"].items():
        out[name] = install_one(metamon_set_dir(teams_root, name), subset_files(reg, s))
    return out


def config_for(reg: Mapping[str, Any], template: Mapping[str, Any], set_name: str, teams_root: Path
               ) -> Dict[str, Any]:
    """The anchors config of ``set_name``'s cell: the template with side B (``home``) = the reference set and side A
    (``away``) = the subset."""
    k = reg["pilots"]["kakuna"]
    cfg = json.loads(json.dumps(template))
    cfg["_README"] = [f"team-strength meter, pilot B: Kakuna on the {set_name} subset (--our-teamset away) vs Kakuna "
                      f"on the reference panel (--teamset home). A DECLARED asymmetric pair cell (B-D9)."]
    mm = cfg["opponents"]["metamon"]
    mm["team_sets"] = {"home": k["ref_set"], "away": k["subset_sets"][set_name]}
    cfg["our_team_sources"] = {
        "home": {"kind": "dir", "path": str(metamon_set_dir(teams_root, k["ref_set"]))},
        "away": {"kind": "dir", "path": str(metamon_set_dir(teams_root, k["subset_sets"][set_name]))}}
    return cfg


def unit_dir(state: Path, unit: str) -> Path:
    return Path(state) / "kakuna" / "cells" / unit


def is_done(state: Path, unit: str) -> bool:
    p = unit_dir(state, unit) / "summary.json"
    if not p.exists():
        return False
    try:
        s = json.loads(p.read_text())
    except ValueError:
        return False
    return s.get("status") == "OK" and s.get("n") == GAMES


def argv_for(py: str, tree: Path, out: Path, seed: int, port: int) -> List[str]:
    a = [py, "-m", "main.anchors", "--opponent-a", "metamon:Kakuna", "--opponent-b", "metamon:Kakuna",
         "--teamset", "home", "--our-teamset", "away", "--declare-asymmetric-teams", "--progress-timeout", "1800",
         "--regime", "greedy", "--games", str(GAMES), "--team-seed", str(seed), "--seed-base", str(seed),
         "--device", "cpu", "--port", str(port), "--nice", "19", "--out", str(out), "--capture-dir",
         str(out / "capture")]
    return ["nice", "-n", "19", "taskset", "-c", CORES, str(tree / "scripts" / "ops" / "mem_cap.sh"),
            "--name", "tsr", MEM_GB, *a]


def env_for(tree: Path, config: Path) -> Dict[str, str]:
    env = dict(os.environ)
    for k in ("POKESIM_SIM_BRIDGE_BIN", "POKESIM_SEARCH_DRIVER_BIN", "POKESIM_CORE_EVENTS_BIN",
              "CARGO_TARGET_DIR", "GEN3AI_ANCHORS_CONFIG"):
        env.pop(k, None)
    env.update({"PYTHONPATH": str(tree / "src"), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1", "GEN3AI_ANCHORS_CONFIG": str(config)})
    return env


def export_packed(packed: Sequence[str], showdown_sim: Path) -> List[str]:
    if not packed:
        return []
    p = subprocess.run(["node", "-e", _EXPORT_JS, str(showdown_sim)], input=json.dumps(list(packed)),
                       capture_output=True, text=True, check=True)
    return list(json.loads(p.stdout))


def match_team(text: str, ids: Mapping[str, str], mon_keys: Mapping[Any, List[str]]) -> Tuple[Optional[str], str]:
    """``(team id, tier)`` of a captured team in its side's set: by identity, else by a UNIQUE species + moves key;
    ``(None, reason)`` otherwise. ``ids`` = the set's team ids (as a set-like mapping), ``mon_keys`` = mon key ->
    the set's team ids holding it."""
    try:
        tid = PN.team_id(text)
    except PN.PanelError:
        return None, "unparseable"
    if tid in ids:
        return tid, "strict"
    hits = sorted(set(mon_keys.get(PN.mon_key(text), [])))
    if len(hits) == 1:
        return hits[0], "mon"
    return None, f"no strict match; {len(hits)} species+moves matches"


def winners(capture: Mapping[str, Any]) -> List[str]:
    txt = "\n".join(c if isinstance(c, str) else json.dumps(c) for c in capture.get("chunks", []))
    return sorted(set(_WIN_RE.findall(txt)))


def parse_unit(udir: Path, *, side_a: Mapping[str, str], side_b: Mapping[str, str], texts: Mapping[str, str],
               showdown_sim: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """``(games, reasons)``: one record per captured battle — side A's and side B's team ids, side A's score, the
    match tiers — and every reason the unit does not count (module docs). ``side_a`` / ``side_b`` map each side's
    team ids to themselves; ``texts`` maps every id to its paste (for the species + moves fallback)."""
    reasons: List[str] = []
    caps = []
    for f in sorted((udir / "capture").glob("*.json"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
        c = json.loads(f.read_text())
        start = json.loads(c["commands"][0].split(" ", 1)[1])
        caps.append((c, start))
    if len(caps) != GAMES:
        reasons.append(f"capture holds {len(caps)} battles, not {GAMES}")
    keys_a: Dict[Any, List[str]] = {}
    keys_b: Dict[Any, List[str]] = {}
    for tid in side_a:
        keys_a.setdefault(PN.mon_key(texts[tid]), []).append(tid)
    for tid in side_b:
        keys_b.setdefault(PN.mon_key(texts[tid]), []).append(tid)
    exported = export_packed([st[s]["team"] for _, st in caps for s in ("p1", "p2")], showdown_sim)
    rows = []
    for i, (c, st) in enumerate(caps):
        names = {s: st[s]["name"] for s in ("p1", "p2")}
        a_slot = [s for s in ("p1", "p2") if "away" in names[s].lower()]
        b_slot = [s for s in ("p1", "p2") if "home" in names[s].lower()]
        if len(a_slot) != 1 or len(b_slot) != 1:
            reasons.append(f"{c['tag']}: cannot tell the sides apart from {names}")
            continue
        ta = exported[2 * i + ("p1", "p2").index(a_slot[0])]
        tb = exported[2 * i + ("p1", "p2").index(b_slot[0])]
        ida, tier_a = match_team(ta, side_a, keys_a)
        idb, tier_b = match_team(tb, side_b, keys_b)
        if ida is None or idb is None:
            reasons.append(f"{c['tag']}: side A {tier_a if ida is None else 'ok'}, side B "
                           f"{tier_b if idb is None else 'ok'}")
            continue
        w = winners(c)
        if len(w) > 1:
            reasons.append(f"{c['tag']}: two winners {w}")
            continue
        s = 0.5 if not w else (1.0 if w[0] == names[a_slot[0]] else 0.0 if w[0] == names[b_slot[0]] else None)
        if s is None:
            reasons.append(f"{c['tag']}: winner {w[0]!r} is neither side")
            continue
        rows.append({"tag": c["tag"], "i": i, "a": ida, "b": idb, "s": s, "tier_a": tier_a, "tier_b": tier_b,
                     "a_seat": a_slot[0]})
    gj = udir / "games.jsonl"
    rr = [json.loads(line) for line in gj.read_text().splitlines()] if gj.exists() else []
    rr.sort(key=lambda r: int(r.get("index", 0)))
    if len(rr) != len(caps):
        reasons.append(f"games.jsonl holds {len(rr)} rows, the capture {len(caps)} battles")
    else:
        pts = {"win": 1.0, "loss": 0.0}
        for row, r in zip(rows, [rr[x["i"]] for x in rows]):
            want = pts.get(r.get("result"), 0.5)
            # Metamon's CSV books a tie as a LOSS of the reported side (X37a FINDING 2): a captured draw must be one
            if not (want == row["s"] or (row["s"] == 0.5 and want == 0.0)):
                reasons.append(f"{row['tag']}: the capture scores side A {row['s']}, games.jsonl {r.get('result')}")
    return rows, reasons


def run_unit(*, py: str, tree: Path, state: Path, unit: str, config: Path, seed: int, port: int) -> int:
    out = unit_dir(state, unit)
    if out.exists():
        out.rename(out.with_name(f"{unit}.partial.{int(time.time())}"))
    out.parent.mkdir(parents=True, exist_ok=True)
    logs = Path(state) / "kakuna" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with open(logs / f"{unit}.log", "ab") as log:
        proc = subprocess.Popen(argv_for(py, tree, out, seed, port), cwd=str(tree), env=env_for(tree, config),
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
        _journal(state, {"event": "start", "unit": unit, "pid": proc.pid, "seed": seed})
        rc = proc.wait()
    _journal(state, {"event": "end", "unit": unit, "rc": rc, "done": is_done(state, unit)})
    return rc


def _journal(state: Path, obj: Mapping[str, Any]) -> None:
    p = Path(state) / "kakuna" / "units.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps({"t": time.time(), **obj}) + "\n")
        f.flush()
        os.fsync(f.fileno())
