"""X72's EXPLOITABILITY PROBE at a fifth target (PREREG §10): the responders' argv, their record check, the read.

X71's design, unchanged, on ONE new target, tag ``plat``: the read round's final of lineage 1 (seed 7341), taken as
its ARCHIVE COPY through :func:`main.x72.finals.load_final`. Two responders (seeds 7111 / 7112, runs
``rb_x35_x71_plat_s<seed>``, +5M at the training pin) fork from it and train against it alone. The read plays THREE
new cells in X71's reader's regime (the Rust eval core on CPU, greedy against greedy, seed 20261020, the five pinned
offense teams): ``k0`` the target against itself, 8,000 games; ``r7111`` / ``r7112`` each responder's final against
it, 4,000 games. X71's twelve banked cells (``c0``, ``x67``, ``swap``, ``x70``) are REUSED from its record, never
re-played.

**Frozen records are restated, never imported.** ``x71_rule.py`` and ``read_x71.py`` freeze X71's ``TARGETS`` and
``READS``; this module restates their statistic, classes, responder check and chunk shape with the target list as an
argument, and ``probe_test.py`` pins every restated value to the frozen files: this :func:`classify` reproduces
X71's two banked verdicts field for field and equals the frozen ``classify`` at five targets on seeded cells.

**The statistic** (X71's): per target ``L = mean of the two responders' rates - the target's own rate against
itself``; ``D = L(plat) - L(c0)``; ``SE = sqrt(V_plat + V_c0 + s2r)`` with the responder variance ``s2r`` pooled over
the FIVE targets; the interval is ``D +- t(0.975, 5) * SE`` (2.571). Classes: ``MORE-EXPLOITABLE`` (lower end
> +0.05 pp), ``LESS-EXPLOITABLE`` (upper end < -0.05 pp), ``NOT-MORE-BY-BAR`` (upper end < +5 - 0.05 pp),
``UNRESOLVED``, ``INCONCLUSIVE`` (a cell missing, without a finished game or > 25 % unfinished; or the control's
responders did not respond). Rule 8: an end within 0.05 pp of a boundary satisfies neither side
(``near_boundary``). Also reported, NO class: ``L(plat) - L(swap)`` (``reported``).

**The banked cells** (:func:`banked`) are read from X71's record: each of the twelve chunks is re-hashed against the
record's ``SHA256SUMS``, must be X71's chunk (schema, seed, regime, game count, one team list, one transport), and
the sums must equal the cells X71's read-2 verdict was classed on.

**The same game.** The three new cells are comparable with the banked twelve only when played by the same code, so
:func:`drive` HOLDs unless the play checkout is AT the commit X71's ledger records (``record/read/play_commit``,
read from the file) with clean ``src/`` and ``data/``, and the player (``probe_play.py``, run by path on that
checkout's code) refuses unless its team files are the banked ones (index, key, sha1). This is the TOOL's guard: the
registration names no play commit for the probe's read.

    python -m main.x72.probe argv --round K --seed 7111              # one responder's trainer argv (launcher flags excluded)
    python -m main.x72.probe record-check --seed 7111                # X71's G6 on a responder that wrote its record
    python -m main.x72.probe plan --round K                          # the read's files, judged; nothing played
    python -m main.x72.probe drive --round K --play-dir <checkout at X71's play commit> --work-dir W

``K`` is the READ round (1..4; when the read round is 0 no probe is played). ``drive``: plan, one cell a heavy slot
(``mem_cap.sh --max-run-s``, nice 19, the agent cores, CUDA hidden, one OpenMP thread, as X71's read ran), then the
verdict ``W/x72_probe_verdict.json`` written ONCE. It resumes: a banked cell is never replayed. Exit codes (X71's
driver's): 0 the verdict is written with a class, 2 usage, 3 the plan or a cell REFUSED, 4 a cell failed, 5 HOLD (the
venue; the work dir is locked, inside a checkout or ``models/``, or another plan's), 6 the verdict is
``INCONCLUSIVE``, 8 a cell ran past its bound or banked nothing. A second ``drive`` on a finished work dir plays and
writes nothing and returns the banked verdict's code. ``record-check``: 0 OK, 3 a problem, 5 no record yet.

``--rehearsal-games N`` is a REHEARSAL of the tool path (N games a team, 2N for ``k0``): its cells go to
``W/rehearsal_ledger`` and ``W/rehearsal.json`` holds counts only; it never writes a verdict and no rehearsal cell
can be collected into one.

The plan reads the responders' records with ``agents.training.best_response_gap``, which resolves their recorded
repo-relative team paths (and judges their recorded fingerprints) from the working directory: ``drive`` reads them
in the PLAY checkout, the files the cells are played on; ``plan`` there too with ``--play-dir``, else in this
checkout.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from main.x35 import x71 as X
from main.x72 import finals as F
from utils.hashing import sha256_file
from utils.paths import repo_path

PREREG = F.PREREG
#: the fifth target: the read round's final of lineage 1 (the lowest seed, chosen by rule at registration)
TAG = "plat"
LINEAGE = F.SEEDS[0]
#: X71's four targets, whose banked cells are reused; the control; the arm the no-class difference is reported against
BANKED = ("c0", "x67", "swap", "x70")
TARGETS = BANKED + (TAG,)
CONTROL = "c0"
REPORTED = "swap"

#: the responders: X71's (``main.x35.x71``, itself pinned to the frozen rule)
SEEDS = X.SEEDS
L_RESP = X.L_RESP
PIN = X.PIN
FORK_LR = X.FORK_LR
RECIPE = X.RECIPE
RUN_FMT = "rb_x35_x71_plat_s{seed}"
CELLS = ("k0",) + tuple(f"r{s}" for s in SEEDS)

#: X71's read and rule (``x71_rule.py``; restated, pinned by ``probe_test.py``)
READ_SEED = 20261020
G = 4_000
G0 = 8_000
N_TEAMS = 5
TIMEOUT_BAR = 0.25
RULE8 = 0.0005
BAR = 0.05
Z_RESPONDED = 1.96
REGIME = "greedy-vs-greedy"

X71_RECORD = "designs/research_state/measurements/x71_exploitability_probe_2026-11-16/record"
X71_CHUNK_SCHEMA = "x71_chunk_v1"
CHUNK_SCHEMA = "x72_probe_chunk_v1"
REHEARSAL_SCHEMA = "x72_probe_rehearsal_chunk_v1"
PLAN_SCHEMA = "x72_probe_plan_v1"
VERDICT_SCHEMA = "x72_probe_verdict_v1"
VERDICT = "x72_probe_verdict.json"
#: one cell's heavy slot: X71's driver's cap and bound (its cells peaked at 0.81 GB; k0 took ~49 min)
MEM_GB = 8
CHUNK_MAX_S = 7200
REHEARSAL_MAX_S = 600

EXIT_OK, EXIT_USAGE, EXIT_REFUSED, EXIT_CHUNK, EXIT_HOLD, EXIT_INCONCLUSIVE, EXIT_BOUND = 0, 2, 3, 4, 5, 6, 8
MEM_CAP_BOUND_RC = 89
PLAYER_REFUSED_RC = 3


class ProbeRefused(Exception):
    def __init__(self, code: int, why: str) -> None:
        super().__init__(why)
        self.code = code


@contextlib.contextmanager
def _cwd(path: Optional[Path]) -> Iterator[None]:
    """Run a block from ``path`` (``None``: where we are): recorded team paths are repo-relative."""
    if path is None:
        yield
        return
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def run_name(seed: int) -> str:
    if int(seed) not in SEEDS:
        raise ProbeRefused(EXIT_USAGE, f"seed {seed} is not a registered responder seed ({list(SEEDS)})")
    return RUN_FMT.format(seed=int(seed))


def target_final(k: int, *, archive_root: Optional[Path] = None, models_dir: Optional[Path] = None) -> F.Final:
    """The probe's target: the ARCHIVE COPY of lineage 1's round-``k`` final, verified. Round 0 is refused."""
    try:
        return F.load_final(F.check_round(k, lo=1), LINEAGE, root=archive_root, models_dir=models_dir)
    except F.FinalsError as exc:
        raise ProbeRefused(EXIT_REFUSED, f"{exc} (the probe's target is the read round's final of lineage "
                                         f"{LINEAGE}; when the read round is 0 no probe is played)") from exc


def responder_argv(seed: int, target: str, target_step: int, *, teams: Optional[str] = None) -> List[str]:
    """One responder's trainer argv (the launcher's own flags excluded): X71's, on this target."""
    if teams is None:
        from main.x35.ge import offense_teams

        teams = offense_teams()
    f = str(target)
    return ["--model", f, "--exploiter", f, "--no-self-play", "--trainee-teams", teams,
            "--fork-lr", FORK_LR, "--fork-lr-freeze", "--behaviour-check", "warn",
            "--steps", str(int(target_step) + L_RESP), "--checkpoint-every-steps", "1000000",
            "--device", "cuda", *RECIPE, "--seed", str(int(seed)), "--run-name", run_name(seed)]


# ------------------------------------------------------------------------------------------------- the records
def _metadata(run_dir: Path) -> Optional[Dict[str, Any]]:
    f = Path(run_dir) / "metadata.json"
    if not f.is_file():
        return None
    md = json.loads(f.read_text(encoding="utf-8"))
    return md if isinstance(md, dict) else {}


def recipe_problems(md: Mapping[str, Any], name: str) -> List[str]:
    """X71's G6: a responder's record holds the TYPED recipe (``ent_coef`` 0.05, no magnet)."""
    probs: List[str] = []
    ent, mag = md.get("ent_coef"), float(md.get("magnet_kl_coef") or 0.0)
    if ent is None or abs(float(ent) - float(RECIPE[1])) > 1e-12:
        probs.append(f"{name}: recorded ent_coef {ent!r}, not the typed {RECIPE[1]}")
    if mag != 0.0:
        probs.append(f"{name}: recorded magnet_kl_coef {mag:g}, not 0 (the responders train without a magnet)")
    return probs


def check_responder(run_dir: Path, target: F.Final, seed: int, run: Any) -> List[str]:
    """Why ``run_dir`` is NOT the registered responder of ``target`` (``[]`` = it is): X71's reader's check
    (``read_x71.py`` ``check_responder``), plus the fork parent's recorded sha256. ``run`` is
    ``best_response_gap.read_exploiter``'s record of it."""
    run_dir = Path(run_dir)
    name = run_dir.name
    if not (run_dir / "final_model.zip").is_file():
        return [f"{name}: no final_model.zip (a responder is read FINISHED only)"]
    md = _metadata(run_dir)
    if md is None:
        return [f"{name}: no metadata.json"]
    probs: List[str] = []
    steps = int(md.get("num_timesteps") or 0)
    if steps < target.step + L_RESP:
        probs.append(f"{name}: final at {steps:,} < {target.step + L_RESP:,} (not +{L_RESP:,})")
    if not run.target_file or os.path.realpath(run.target_file) != os.path.realpath(target.path):
        probs.append(f"{name}: recorded target {run.target_file!r} is not {target.path}")
    if run.fork_step is None or int(run.fork_step) != int(target.step):
        probs.append(f"{name}: forked at {run.fork_step}, not at the target's step {target.step:,}")
    if run.archetype != "offense":
        probs.append(f"{name}: trainee teams are {run.archetype!r} ({run.membership}), not the offense pin")
    probs += recipe_problems(md, name)
    got_seed = (md.get("cli_args") or {}).get("seed")
    if got_seed is None or int(got_seed) != int(seed):
        probs.append(f"{name}: recorded seed {got_seed!r}, not {seed}")
    pins = sorted({str(p.get("git_hash")) for p in md.get("pin_history") or []})
    if pins != [PIN]:
        probs.append(f"{name}: trained at pin(s) {pins}, not {PIN[:12]} alone")
    parent = ((md.get("lineage") or {}).get("fork_parent") or {}).get("sha256")
    if parent is not None and parent != target.sha256:
        probs.append(f"{name}: forked from a file with sha256 {str(parent)[:12]}, not the target's "
                     f"{target.sha256[:12]}")
    return probs


def record_check(seed: int, *, models_dir: Optional[Path] = None) -> Tuple[int, List[str]]:
    """X71's G6 on a responder that may still be training: ``(0, [])`` OK, ``(5, [why])`` no record yet,
    ``(3, problems)``."""
    rd = _models(models_dir) / run_name(seed)
    try:
        md = _metadata(rd)
    except ValueError as exc:
        return EXIT_REFUSED, [f"{rd.name}: metadata.json is unreadable ({exc})"]
    if md is None:
        return EXIT_HOLD, [f"{rd.name}: no metadata.json yet under {rd.parent}"]
    probs = recipe_problems(md, rd.name)
    return (EXIT_REFUSED if probs else EXIT_OK), probs


def _models(models_dir: Optional[Path]) -> Path:
    try:
        return F._models(models_dir)
    except F.FinalsError as exc:
        raise ProbeRefused(EXIT_REFUSED, str(exc)) from exc


def _record(record: Optional[Path]) -> Path:
    return Path(record) if record is not None else repo_path(*X71_RECORD.split("/"))


def _sums(cells: Mapping[str, Mapping[str, Any]]) -> Dict[str, int]:
    return {k: sum(int(c[k]) for c in cells.values()) for k in ("wins", "finished", "attempted", "ties")}


def banked(record: Optional[Path] = None) -> Dict[str, Any]:
    """X71's twelve banked cells, VERIFIED (module docstring): ``{cells, by_team, teams, team_paths, transport,
    play_commit, chunks}``. Any mismatch is a refusal that names every problem."""
    rec = _record(record)
    probs: List[str] = []
    try:
        sums = {line.split()[1]: line.split()[0] for line in (rec / "SHA256SUMS").read_text().splitlines()
                if len(line.split()) == 2}
    except OSError as exc:
        raise ProbeRefused(EXIT_REFUSED, f"X71's record is unreadable: {exc}") from exc

    def verified(rel: str) -> Optional[Path]:
        f = rec / rel
        if rel not in sums:
            probs.append(f"{rel}: not listed in X71's SHA256SUMS")
        elif not f.is_file():
            probs.append(f"{rel}: missing from X71's record")
        elif sha256_file(f) != sums[rel]:
            probs.append(f"{rel}: no longer matches X71's SHA256SUMS (the banked record changed)")
        else:
            return f
        return None

    cells: Dict[str, Dict[str, Dict[str, int]]] = {}
    by_team: Dict[str, Dict[str, Dict[str, List[int]]]] = {}
    chunks: Dict[str, str] = {}
    teams: Optional[List[List[Any]]] = None
    transports = set()
    for tag in BANKED:
        for cell in CELLS:
            rel = f"read/ledger/{tag}__{cell}.json"
            f = verified(rel)
            if f is None:
                continue
            got = json.loads(f.read_text())
            per = (G0 if cell == "k0" else G) // N_TEAMS
            want = {"schema": X71_CHUNK_SCHEMA, "target_tag": tag, "cell": cell, "seed": READ_SEED,
                    "regime": REGIME, "games_per_team": per}
            off = {k: got.get(k) for k, v in want.items() if got.get(k) != v}
            if off:
                probs.append(f"{rel}: is not X71's chunk of this cell: {off}")
                continue
            teams = got["teams"] if teams is None else teams
            if got["teams"] != teams or sorted(got["counts"]) != sorted(t[1] for t in teams):
                probs.append(f"{rel}: its teams are not the ledger's one team list")
                continue
            agg = _sums(got["counts"])
            if agg["attempted"] != per * N_TEAMS:
                probs.append(f"{rel}: {agg['attempted']} games attempted, registered {per * N_TEAMS}")
            transports.add(got.get("transport"))
            cells.setdefault(tag, {})[cell] = agg
            by_team.setdefault(tag, {})[cell] = {k: [c["wins"], c["finished"]] for k, c in got["counts"].items()}
            chunks[rel] = sums[rel]
    out: Dict[str, Any] = {"cells": cells, "by_team": by_team, "teams": teams, "chunks": chunks}
    for key, rel in (("verdict", "read/x71_verdict_read2.json"), ("plan", "read/plan_read2.json"),
                     ("play_commit", "read/play_commit")):
        f = verified(rel)
        if f is not None:
            out[key] = f.read_text().strip() if key == "play_commit" else json.loads(f.read_text())
    if not probs:
        if len(transports) != 1 or len(teams or []) != N_TEAMS:
            probs.append(f"X71's ledger names transports {sorted(map(str, transports))} and {len(teams or [])} teams")
        if out["verdict"].get("cells") != cells:
            probs.append("the banked chunks do not sum to the cells X71's read-2 verdict was classed on")
        paths = {tuple(out["plan"][t]["teams"]) for t in BANKED}
        if len(paths) != 1 or len(next(iter(paths))) != N_TEAMS:
            probs.append("X71's read-2 plan does not name ONE list of five team files")
    if probs:
        raise ProbeRefused(EXIT_REFUSED, "X71's banked cells do not verify:\n  " + "\n  ".join(probs))
    return {"cells": cells, "by_team": by_team, "teams": teams, "team_paths": list(out["plan"][CONTROL]["teams"]),
            "transport": transports.pop(), "play_commit": out["play_commit"], "chunks": chunks}


# ------------------------------------------------------------------------------------------------- the statistic
def mult(n_targets: int) -> float:
    from main.h2h.cross import t_ppf

    return float(t_ppf(0.975, int(n_targets)))


def _cell(c: Mapping[str, Any], where: str) -> Tuple[int, int, float]:
    try:
        w, n = int(c["wins"]), int(c["finished"])
        a = int(c.get("attempted", n))
    except (KeyError, TypeError, ValueError) as exc:
        raise ProbeRefused(EXIT_REFUSED, f"{where}: not a cell ({c!r})") from exc
    if not 0 <= w <= n <= a:
        raise ProbeRefused(EXIT_REFUSED, f"{where}: wins {w} / finished {n} / attempted {a} is not a count")
    return w, n, ((a - n) / a if a else 1.0)


def target_stats(cells: Mapping[str, Any], tag: str) -> Dict[str, Any]:
    """One target's L, V, d, e from its three cells; ``bad`` names what makes it unreadable (X71's)."""
    bad: List[str] = []
    out: Dict[str, Any] = {"target": tag, "bad": bad}
    got: Dict[str, Tuple[int, int, float]] = {}
    for key in CELLS:
        c = cells.get(key)
        if c is None:
            bad.append(f"{tag}/{key}: missing")
            continue
        w, n, tf = _cell(c, f"{tag}/{key}")
        if n <= 0:
            bad.append(f"{tag}/{key}: no finished game")
        elif tf > TIMEOUT_BAR:
            bad.append(f"{tag}/{key}: {tf:.0%} of its games did not finish (> {TIMEOUT_BAR:.0%})")
        else:
            got[key] = (w, n, tf)
    if bad:
        return out
    p0 = got["k0"][0] / got["k0"][1]
    ps = [got[f"r{s}"][0] / got[f"r{s}"][1] for s in SEEDS]
    ns = [got[f"r{s}"][1] for s in SEEDS]
    game = [p * (1 - p) / n for p, n in zip(ps, ns)]
    d = ps[0] - ps[1]
    out.update(g0=p0 - 0.5, g={str(s): p - 0.5 for s, p in zip(SEEDS, ps)},
               L=sum(ps) / len(ps) - p0, V=sum(game) / len(ps) ** 2 + p0 * (1 - p0) / got["k0"][1],
               d=d, e=d * d - sum(game), n0=got["k0"][1], n=ns)
    return out


def classify(cells: Mapping[str, Mapping[str, Any]], targets: Sequence[str] = TARGETS,
             arms: Sequence[str] = (TAG,)) -> Dict[str, Any]:
    """X71's verdict over ``targets`` (every one required; the control among them) for ``arms``. The registered
    call is the default: five targets, the arm ``plat``."""
    tags = tuple(targets)
    stats = {t: target_stats(cells.get(t) or {}, t) for t in tags}
    out: Dict[str, Any] = {"targets": stats, "bar": BAR, "rule8": RULE8,
                           "games": {"G": G, "G0": G0, "seed": READ_SEED}}
    bad = [b for t in tags for b in stats[t]["bad"]]
    if bad:
        out.update(cls={a: "INCONCLUSIVE" for a in arms}, why="; ".join(bad))
        return out
    s2r = max(0.0, sum(stats[t]["e"] for t in tags) / len(tags)) / 2.0
    m = mult(len(tags))
    c0 = stats[CONTROL]
    c0_lo = c0["L"] - Z_RESPONDED * math.sqrt(c0["V"])
    out.update(s2r=s2r, sigma_r=math.sqrt(s2r), mult=m, control_L=c0["L"], control_L_lo=c0_lo)
    if not c0_lo > RULE8:
        out.update(cls={a: "INCONCLUSIVE" for a in arms},
                   why=f"the control's responders did not respond: L_c0 = {c0['L']:+.4f}, lower end {c0_lo:+.4f} "
                       f"<= +{RULE8}")
        return out
    cls: Dict[str, str] = {}
    table: Dict[str, Any] = {}
    near: List[str] = []
    for a in arms:
        d = stats[a]["L"] - c0["L"]
        se = math.sqrt(stats[a]["V"] + c0["V"] + s2r)
        lo, hi = d - m * se, d + m * se
        for name, v, edge in (("lower", lo, 0.0), ("upper", hi, 0.0), ("upper", hi, BAR)):
            if abs(v - edge) <= RULE8:
                near.append(f"{a}: the {name} end {v:+.5f} is within {RULE8} of {edge:+.2f}")
        if lo > RULE8:
            c = "MORE-EXPLOITABLE"
        elif hi < -RULE8:
            c = "LESS-EXPLOITABLE"
        elif hi < BAR - RULE8:
            c = "NOT-MORE-BY-BAR"
        else:
            c = "UNRESOLVED"
        cls[a] = c
        table[a] = {"D": d, "se": se, "lo": lo, "hi": hi, "L": stats[a]["L"]}
    out.update(cls=cls, arms=table, near_boundary=near, why="")
    return out


def reported(doc: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """``L(plat) - L(swap)`` with the same SE and multiplier: REPORTED, never classed. ``None`` when a target is
    unreadable."""
    st = doc["targets"]
    if "s2r" not in doc or any(st[t]["bad"] for t in (TAG, REPORTED)):
        return None
    d = st[TAG]["L"] - st[REPORTED]["L"]
    se = math.sqrt(st[TAG]["V"] + st[REPORTED]["V"] + doc["s2r"])
    return {"what": f"L({TAG}) - L({REPORTED}): reported, NO class", "D": d, "se": se,
            "lo": d - doc["mult"] * se, "hi": d + doc["mult"] * se}


# ------------------------------------------------------------------------------------------------- plan and ledger
def plan(k: int, *, archive_root: Optional[Path] = None, models_dir: Optional[Path] = None,
         record: Optional[Path] = None, rehearsal_games: Optional[int] = None,
         teams_root: Optional[Path] = None) -> Dict[str, Any]:
    """The read's files, each judged: the target (the archive copy), the two FINISHED responders by their records,
    X71's banked cells. A refusal names every problem at once. ``teams_root`` is the checkout the responders'
    recorded team paths are read in (``None``: the working directory)."""
    from agents.training import best_response_gap as E

    if rehearsal_games is not None and not 1 <= int(rehearsal_games) < G // N_TEAMS:
        raise ProbeRefused(EXIT_USAGE, f"--rehearsal-games {rehearsal_games}: 1..{G // N_TEAMS - 1} games a team")
    t = target_final(k, archive_root=archive_root, models_dir=models_dir)
    b = banked(record)
    md = _models(models_dir)
    with _cwd(teams_root):
        try:
            teamsets = E.load_teamsets()
        except E.RunReadError as exc:
            raise ProbeRefused(EXIT_REFUSED, str(exc)) from exc
    per = {c: (G0 if c == "k0" else G) // N_TEAMS for c in CELLS}
    if rehearsal_games is not None:
        per = {c: int(rehearsal_games) * (2 if c == "k0" else 1) for c in CELLS}
    cells: Dict[str, Any] = {"k0": {"path": t.path, "sha256": t.sha256, "games_per_team": per["k0"]}}
    probs: List[str] = []
    for s in SEEDS:
        rd = md / run_name(s)
        if not rd.is_dir():
            probs.append(f"{rd.name}: no such run under {md}")
            continue
        try:
            with _cwd(teams_root):
                run = E.read_exploiter(str(rd), teamsets)
        except E.BestResponseGapError as exc:        # an unreadable record, or one whose evals name another target
            probs.append(f"{rd.name}: {exc}")
            continue
        found = check_responder(rd, t, s, run)
        if list(run.teams) != b["team_paths"]:
            found.append(f"{rd.name}: its recorded trainee teams {list(run.teams)} are not the files X71's cells "
                         f"were played on {b['team_paths']}")
        probs += found
        if not found:
            cells[f"r{s}"] = {"path": str(rd / "final_model.zip"), "sha256": sha256_file(rd / "final_model.zip"),
                              "games_per_team": per[f"r{s}"]}
    if probs:
        raise ProbeRefused(EXIT_REFUSED, "the read's files are not the registered ones:\n  " + "\n  ".join(probs))
    return {"schema": PLAN_SCHEMA, "round": int(k), "tag": TAG, "target_run": t.run, "target": t.path,
            "target_step": t.step, "target_sha256": t.sha256, "team_paths": b["team_paths"], "teams": b["teams"],
            "play_commit": b["play_commit"], "rehearsal": rehearsal_games is not None, "cells": cells}


def want(p: Mapping[str, Any], cell: str) -> Dict[str, Any]:
    """What the chunk of ``cell`` must say (and what the player is asked to play): X71's chunk identity plus the
    round, the team paths and the playing commit."""
    c = p["cells"][cell]
    return {"schema": REHEARSAL_SCHEMA if p["rehearsal"] else CHUNK_SCHEMA, "target_tag": TAG, "cell": cell,
            "round": p["round"], "path": c["path"], "sha256": c["sha256"], "target": p["target"],
            "target_sha256": p["target_sha256"], "team_paths": list(p["team_paths"]),
            "teams": [list(t) for t in p["teams"]], "seed": READ_SEED, "games_per_team": c["games_per_team"],
            "regime": REGIME, "play_commit": p["play_commit"]}


def chunk_file(ledger: Path, cell: str) -> Path:
    return Path(ledger) / f"{TAG}__{cell}.json"


def banked_chunk(ledger: Path, w: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The banked chunk of ``w``'s cell, or ``None``; one banked for another read is a refusal, never replayed
    over."""
    f = chunk_file(ledger, w["cell"])
    if not f.is_file():
        return None
    got: Dict[str, Any] = json.loads(f.read_text())
    seen = {k: got.get(k) for k in w}
    if seen != dict(w):
        diff = {k: (seen[k], w[k]) for k in w if seen[k] != w[k]}
        raise ProbeRefused(EXIT_REFUSED, f"{f}: banked for another read (banked, wanted): {diff}")
    return got


def collect(p: Mapping[str, Any], ledger: Path, transport: str) -> Dict[str, Any]:
    """The target's three cells from its banked chunks (X71's ``collect``): ``{cells, by_team}``."""
    cells: Dict[str, Dict[str, int]] = {}
    by_team: Dict[str, Dict[str, List[int]]] = {}
    keys = sorted(t[1] for t in p["teams"])
    for cell in CELLS:
        w = want(p, cell)
        got = banked_chunk(ledger, w)
        f = chunk_file(ledger, cell)
        if got is None:
            raise ProbeRefused(EXIT_REFUSED, f"{f.name} is not banked (play first)")
        if sorted(got["counts"]) != keys:
            raise ProbeRefused(EXIT_REFUSED, f"{f}: its teams are {sorted(got['counts'])}")
        if got.get("transport") != transport:
            raise ProbeRefused(EXIT_REFUSED, f"{f}: played on {got.get('transport')!r}, X71's banked cells on "
                                             f"{transport!r}")
        agg = _sums(got["counts"])
        if agg["attempted"] != w["games_per_team"] * N_TEAMS:
            raise ProbeRefused(EXIT_REFUSED, f"{f}: {agg['attempted']} games attempted, planned "
                                             f"{w['games_per_team'] * N_TEAMS}")
        cells[cell] = agg
        by_team[cell] = {k: [c["wins"], c["finished"]] for k, c in got["counts"].items()}
    return {"cells": cells, "by_team": by_team}


def verdict(p: Mapping[str, Any], ledger: Path, *, record: Optional[Path] = None) -> Dict[str, Any]:
    """The probe's verdict document from the three banked chunks and X71's banked cells. A rehearsal plan is
    refused: its cells are not the registered counts."""
    if p["rehearsal"]:
        raise ProbeRefused(EXIT_REFUSED, "a rehearsal has no verdict")
    b = banked(record)
    got = collect(p, ledger, b["transport"])
    cells = dict(b["cells"], **{TAG: got["cells"]})
    doc = classify(cells)
    doc.update(schema=VERDICT_SCHEMA, prereg=PREREG, round=p["round"], cells=cells, reported=reported(doc),
               by_team=dict(b["by_team"], **{TAG: got["by_team"]}), transport=b["transport"], regime=REGIME,
               plan=dict(p), banked={"record": X71_RECORD, "play_commit": b["play_commit"], "chunks": b["chunks"]})
    return doc


def exit_of(doc: Mapping[str, Any]) -> int:
    return EXIT_INCONCLUSIVE if doc["cls"].get(TAG) == "INCONCLUSIVE" else EXIT_OK


# ------------------------------------------------------------------------------------------------- the driver
def _venue_problems(play_dir: Path, commit: str, run: Callable[..., Any]) -> List[str]:
    head = run(["git", "-C", str(play_dir), "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode != 0 or head.stdout.strip() != commit:
        return [f"{play_dir} is at {head.stdout.strip() or 'no commit'}, and X71's banked cells were played at "
                f"{commit} (one playing commit for the five targets)"]
    dirty = run(["git", "-C", str(play_dir), "status", "--porcelain", "--untracked-files=no", "--", "src", "data"],
                capture_output=True, text=True)
    if dirty.returncode != 0 or dirty.stdout.strip():
        return [f"{play_dir} has tracked changes under src/ or data/: {dirty.stdout.strip()[:200]!r}"]
    return []


def _inside(path: Path, root: Path) -> bool:
    return (os.path.realpath(path) + os.sep).startswith(os.path.realpath(root) + os.sep)


def play_argv(py: str, play_dir: Path, want_file: Path, out: Path, cpus: str, *, rehearsal: bool,
              mem_gb: int = MEM_GB, max_s: int = CHUNK_MAX_S) -> List[str]:
    """One cell's command: the by-path player on the play checkout's code, in rule 5b's shape."""
    bound = ["--max-s", str(REHEARSAL_MAX_S)] if rehearsal else ["--max-run-s", str(int(max_s))]
    return ["nice", "-n", "19", "taskset", "-c", cpus, str(repo_path("scripts", "ops") / "mem_cap.sh"),
            "--name", f"x72_probe_{out.stem}", *bound, str(int(mem_gb)), py, "-P",
            str(Path(__file__).with_name("probe_play.py")), "--want", str(want_file), "--out", str(out),
            "--checkout", str(play_dir)]


def drive(k: int, play_dir: Path, work: Path, *, archive_root: Optional[Path] = None,
          models_dir: Optional[Path] = None, record: Optional[Path] = None, rehearsal_games: Optional[int] = None,
          run: Callable[..., Any] = subprocess.run, py: Optional[str] = None, cpus: Optional[str] = None,
          mem_gb: int = MEM_GB, max_s: int = CHUNK_MAX_S) -> int:
    """Plan, play the cells not banked yet (one a heavy slot), write the verdict ONCE. Returns the exit code
    (module docstring). ``run`` is ``subprocess.run`` (the tests pass a recorder)."""
    work, play_dir = Path(work), Path(play_dir)
    lock = None

    def say(msg: str) -> None:
        print(msg, flush=True)
        with open(work / "run.log", "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    try:
        try:
            for root, what in ((repo_path(), "this checkout"), (play_dir, "the play checkout"),
                               (_models(models_dir), "models/")):
                if _inside(work, root):
                    print(f"[x72 probe] HOLD (exit {EXIT_HOLD}): the work dir {work} is inside {what}", flush=True)
                    return EXIT_HOLD
            work.mkdir(parents=True, exist_ok=True)
            lock = open(work / "lock", "w")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ProbeRefused(EXIT_HOLD, f"another driver holds {work / 'lock'} (one lane a work dir)") from exc
            out = work / VERDICT
            if out.is_file():
                doc = json.loads(out.read_text())
                say(f"[x72 probe] DONE before: {out} exists (a verdict is written ONCE); nothing was played or "
                    f"read; class {doc['cls'].get(TAG)}")
                return exit_of(doc)
            probs = _venue_problems(play_dir, banked(record)["play_commit"], run)
            if probs:
                raise ProbeRefused(EXIT_HOLD, "; ".join(probs))
            p = plan(k, archive_root=archive_root, models_dir=models_dir, record=record,
                     rehearsal_games=rehearsal_games, teams_root=play_dir)
            pf = work / ("rehearsal_plan.json" if p["rehearsal"] else "plan.json")
            text = json.dumps(p, indent=1, sort_keys=True) + "\n"
            if pf.is_file() and pf.read_text() != text:
                raise ProbeRefused(EXIT_HOLD, f"{pf} is another plan (a file moved, or another round's work dir)")
            pf.write_text(text)
            ledger = work / ("rehearsal_ledger" if p["rehearsal"] else "ledger")
            ledger.mkdir(exist_ok=True)
            py = py or os.environ.get("GEN3AI_PYTHON") or sys.executable
            env = dict(os.environ, PYTHONPATH=str(play_dir / "src"), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
            env.pop("GEN3AI_TEST_ALLOW_GPU", None)
            if cpus is None:
                from utils.core_split import agent_cpus

                cpus = agent_cpus()
            say(f"[x72 probe] round {k}{' REHEARSAL' if p['rehearsal'] else ''}: target {p['target']} at "
                f"{p['target_step']:,}; play commit {p['play_commit'][:8]} at {play_dir}; ledger {ledger}")
            for cell in CELLS:
                w = want(p, cell)
                if banked_chunk(ledger, w) is not None:
                    say(f"[x72 probe] {cell}: banked before")
                    continue
                wf = ledger / f"want_{cell}.json"
                wf.write_text(json.dumps(w, indent=1, sort_keys=True) + "\n")
                argv = play_argv(py, play_dir, wf, chunk_file(ledger, cell), cpus, rehearsal=p["rehearsal"],
                                 mem_gb=mem_gb, max_s=max_s)
                say(f"[x72 probe] {cell}: " + " ".join(argv))
                with open(work / "run.log", "a", encoding="utf-8") as lf:
                    rc = run(argv, cwd=str(play_dir), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
                if rc == PLAYER_REFUSED_RC:
                    raise ProbeRefused(EXIT_REFUSED, f"{cell}: the player REFUSED (see {work / 'run.log'})")
                if rc == MEM_CAP_BOUND_RC:
                    raise ProbeRefused(EXIT_BOUND, f"{cell}: ran past its bound and banked nothing (an indivisible "
                                                   f"cell: measure the rate before raising --max-chunk-s)")
                if rc != 0:
                    raise ProbeRefused(EXIT_CHUNK, f"{cell}: the player exited {rc} (see {work / 'run.log'}); "
                                                   f"re-run to resume")
                if banked_chunk(ledger, w) is None:
                    raise ProbeRefused(EXIT_BOUND, f"{cell}: the player exited 0 and banked nothing")
            if p["rehearsal"]:
                got = collect(p, ledger, banked(record)["transport"])
                (work / "rehearsal.json").write_text(json.dumps(
                    {"schema": "x72_probe_rehearsal_v1", "note": "a REHEARSAL of the tool path: counts only, no "
                     "class, never a verdict", "round": p["round"], "cells": got["cells"],
                     "by_team": got["by_team"]}, indent=1, sort_keys=True) + "\n")
                say(f"[x72 probe] REHEARSAL: {got['cells']} (no verdict)")
                return EXIT_OK
            doc = verdict(p, ledger, record=record)
            with open(out, "x", encoding="utf-8") as f:
                f.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        except ProbeRefused as exc:
            msg = f"[x72 probe] {'HOLD' if exc.code == EXIT_HOLD else 'REFUSED'} (exit {exc.code}): {exc}"
            say(msg) if work.is_dir() else print(msg, flush=True)
            return exc.code
        say(_summary(doc))
        return exit_of(doc)
    finally:
        if lock is not None:
            lock.close()


def _summary(doc: Mapping[str, Any]) -> str:
    a = (doc.get("arms") or {}).get(TAG)
    r = doc.get("reported")
    lines = [f"[x72 probe] round {doc['round']}: {doc['cls'].get(TAG)}" + (f" ({doc['why']})" if doc["why"] else "")]
    if a:
        lines.append(f"  D = L({TAG}) - L({CONTROL}) = {100 * a['D']:+.2f} pp [{100 * a['lo']:+.2f}, "
                     f"{100 * a['hi']:+.2f}] (t(0.975, {len(TARGETS)}) = {doc['mult']:.3f}; L({TAG}) "
                     f"{100 * a['L']:+.2f} pp, L({CONTROL}) {100 * doc['control_L']:+.2f} pp)")
    if r:
        lines.append(f"  reported, no class: L({TAG}) - L({REPORTED}) = {100 * r['D']:+.2f} pp "
                     f"[{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}]")
    return "\n".join(lines + [f"  - near a boundary: {n}" for n in doc.get("near_boundary") or []])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x72.probe", description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("argv", "record-check", "plan", "drive"):
        q = sub.add_parser(name)
        if name != "record-check":
            q.add_argument("--round", type=int, required=True, help="the READ round (1..4)")
            q.add_argument("--archive-root", type=Path, default=None, help="the finals archive (default: the "
                                                                              "registered one)")
        if name in ("argv", "record-check"):
            q.add_argument("--seed", type=int, required=True, choices=SEEDS)
        q.add_argument("--models-dir", type=Path, default=None, help="where the runs are (default: main's models/)")
        if name == "plan":
            q.add_argument("--play-dir", type=Path, default=None, help="read the recorded team files in this "
                                                                       "checkout (default: this one)")
        if name in ("plan", "drive"):
            q.add_argument("--x71-record", type=Path, default=None, help="X71's record (default: this checkout's)")
            q.add_argument("--rehearsal-games", type=int, default=None, help="REHEARSAL: games a team; no verdict")
        if name == "drive":
            q.add_argument("--play-dir", type=Path, required=True, help="a checkout AT X71's play commit, Rust "
                                                                        "eval core built")
            q.add_argument("--work-dir", type=Path, required=True, help="outside every checkout and models/")
            q.add_argument("--mem-gb", type=int, default=MEM_GB)
            q.add_argument("--max-chunk-s", type=int, default=CHUNK_MAX_S)
    a = ap.parse_args(argv)
    paths = {k: (None if v is None else Path(os.path.abspath(v))) for k, v in vars(a).items()
             if k in ("archive_root", "models_dir", "x71_record", "play_dir", "work_dir")}
    try:
        if a.cmd == "argv":
            t = target_final(a.round, archive_root=paths["archive_root"], models_dir=paths["models_dir"])
            with _cwd(repo_path()):      # the committed teamsets file is repo-relative
                print(" ".join(responder_argv(a.seed, t.path, t.step)))
            return EXIT_OK
        if a.cmd == "record-check":
            code, probs = record_check(a.seed, models_dir=paths["models_dir"])
            print(f"[x72 probe] {run_name(a.seed)}: " + ("the record holds the typed recipe (G6)" if not probs
                                                         else "; ".join(probs)))
            return code
        if a.cmd == "plan":
            print(json.dumps(plan(a.round, archive_root=paths["archive_root"], models_dir=paths["models_dir"],
                                  record=paths["x71_record"], rehearsal_games=a.rehearsal_games,
                                  teams_root=paths["play_dir"] or repo_path()), indent=1, sort_keys=True))
            return EXIT_OK
        play_dir, work = paths["play_dir"], paths["work_dir"]
        assert play_dir is not None and work is not None
        return drive(a.round, play_dir, work, archive_root=paths["archive_root"],
                     models_dir=paths["models_dir"], record=paths["x71_record"],
                     rehearsal_games=a.rehearsal_games, mem_gb=a.mem_gb, max_s=a.max_chunk_s)
    except ProbeRefused as exc:
        print(f"[x72 probe] REFUSED: {exc}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
