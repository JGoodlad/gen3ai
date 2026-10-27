"""X35's EXTERNAL-ANCHOR unit runner, G-K's instrument (``design_compute_efficiency.md`` §5.6.6;
``designs/ops/EXTERNAL_ANCHORS_SOP.md``). The X26 read's ``xq.py`` pattern
(``measurements/x26_metamon_anchor_2026-10-17/scripts/xq.py``), made a tested module.

THE CELL. ``metamon:Kakuna`` (ckpt 34), GREEDY vs GREEDY (SOP rule 2), our side greedy on CPU, the in-process Rust
front end on a 95xx port this tool picks itself. BOTH team sets: ``home72`` and ``away``, through ONE COMMITTED
config, ``<stage>/anchors_home72_kakuna.json`` -- byte-identical to the X26 read's (sha256 ``e8a064db…``, F21) --
whose every team file is pinned by ``<stage>/home72_manifest.json`` (our 72 committed pastes under
``measurements/n0_endofrun_2026-09-27/teams_home72/`` AND Metamon's copy of them in its cache). The manifest is
re-checked before EVERY unit: a moved team is a refusal, never a different read.

A UNIT is one ``python -m main.anchors`` invocation of 20 role-balanced games with ``--team-seed = --seed-base =``
the unit seed; the seeds 0-6 are SHARED across arms. It runs from the checkout ROOT (the pool source's CWD hazard,
X26 read FINDING 1), under ``nice -n 19 taskset -c 5-7,13-15`` inside ``scripts/ops/mem_cap.sh`` (rule 5b), one lane.
It starts on the heavy slots' SHORT lane; one the lane kills past its bound (exit 88) and that is not counted is played
ONCE more as a normal heavy-slot job (:func:`run_unit`). ``run`` is ONE supervisor per state PATH: the shared path lock
outside the state, the ``/proc`` scan and the dir-identity stop of :mod:`main.reads.supervisors`, beside the in-state
``supervisor.lock`` (:func:`cmd_run`).
A unit COUNTS iff ``status == OK``, ``n == 20``, ``regime_verified_decisions``, no ``team_source_asymmetry`` and
``peer_clean`` (§5.6.6). A T28 halt marker refuses every unit (``main.live.halt``).

    python -m main.x35.kakuna plan   --state DIR --finals finals.json   # [{label, arm, seed, path}]: the unit queue
    python -m main.x35.kakuna run    --state DIR [--stop-at-epoch E]    # work the queue (resumable, detached-safe)
    python -m main.x35.kakuna status --state DIR
    python -m main.x35.kakuna gk     --state DIR --on A1 --off C0       # G-K's input for main.x35.readout
    python -m main.x35.kakuna smoke  --state DIR --model ZIP [--games 2] # ONE unit vs metamon:SmallRL, home72
    python -m main.x35.kakuna manifest --check | --write
    python -m main.x35.kakuna panels                                     # check the unfamiliar panels' cells
    python -m main.x35.kakuna d1h    --state DIR --on X38 --off C0       # D1's PRIMARY H input (main.x35.d1)

THE UNFAMILIAR PANELS (X38 / D1's PRIMARY H, ``design_diversity_teams_opponents.md`` §4.2, §5.6). ``plan
--teamsets unfamiliar`` (= ``heldout72_hpfix,forum72,p72``; any comma list of :data:`TEAMSETS` and
:data:`PANEL_CELLS` names) plans the same 7 shared-seed 20-game units per final on each named panel. A panel unit
runs on that panel's COMMITTED anchors config (:data:`PANEL_CELLS`: X37a's ``anchors_x37a_forum72.json`` /
``anchors_x37a_p72.json`` and ``anchors_heldout72_hpfix_kakuna.json``, each pinned by sha256), whose ``home`` team
set is the panel on BOTH sides (a SYMMETRIC cell). Before EVERY panel unit (:func:`check_panel_cell`): the config's
sha256, the panel's registry entry and its committed copy (:func:`main.heldout_panels.panel_team_texts`), and the
CONTENT digest (``gen3_team_set_digest_v1``) and count of BOTH sides' team dirs the config names (our byte copy,
Metamon's cache set) against the registry -- a moved team is a refusal, never a different read. ``home72`` / ``away``
units are exactly as before (the home72 manifest; :data:`CONFIG`), and G-K reads only them. A panel unit also passes
``--capture-dir <unit>/capture`` (both packed teams of every game, F-DV-17), which changes no stamp and no play.

OTHER METAMON OPPONENTS (``plan_units(opponent=...)``; :data:`OPPONENTS`). A unit may face ``metamon:SmallRL``,
``PastaMittens``, ``LargeRL`` or ``LargeIL`` on the same committed cells. An agent the cell's config already names plays
under that config (its pins filled from the registry, ff54265e). An agent it lacks gets a DERIVED config, written once
at plan (:func:`opponent_config`): the committed config with ONE agent added, exactly the registry's entry
(``designs/ops/anchors.json``, its hub revision and weight sha256s), as the FR reads' ``make_configs.py`` did. The unit
records the derived file's sha256, and before every unit :func:`check_derived_config` re-checks it: the file's bytes,
the committed config it extends (unchanged but for that agent), and the pin. The peer runs offline on the pinned
revision (``HF_HUB_OFFLINE``, a260e9c3); the weights enter the regime id only when they differ from the banked ones.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from utils import ports
from utils.filelock import try_exclusive
from utils.hashing import sha256_file
from utils.paths import repo_path, repo_root

STAGE = repo_path("designs/research_state/measurements/x35_breakers_stage_2026-10-17")
CONFIG = STAGE / "anchors_home72_kakuna.json"
MANIFEST = STAGE / "home72_manifest.json"
CONFIG_SHA256 = "e8a064db845b058dee5ab2512e7f8d0ba636f10df58510e30b8cac9b30422fac"
OUR_TEAMS = repo_path("designs/research_state/measurements/n0_endofrun_2026-09-27/teams_home72")
OPPONENT = "metamon:Kakuna"
SMOKE_OPPONENT = "metamon:SmallRL"
#: Every Metamon opponent a unit may face on the committed cells (each pinned in the registry, :data:`REGISTRY`).
OPPONENTS = ("metamon:Kakuna", "metamon:SmallRL", "metamon:PastaMittens", "metamon:LargeRL", "metamon:LargeIL")
REGISTRY = repo_path("designs/ops/anchors.json")
TEAMSETS = {"home72": "home", "away": "away"}
#: The plan's default team sets (X35's G-K cell: home72 and away, through :data:`CONFIG`).
DEFAULT_TEAMSETS = ("home72", "away")
_MEAS = "designs/research_state/measurements"


@dataclass(frozen=True)
class PanelCell:
    """A registered panel's SYMMETRIC Kakuna cell: the committed anchors config whose ``teamset`` side pilots the
    panel on both sides, pinned by sha256."""

    config: str
    sha256: str
    teamset: str = "home"

    @property
    def path(self) -> Path:
        return repo_path(self.config)


#: D1's UNFAMILIAR cells (X37a's committed configs; registry ``heldout_panels.json``).
PANEL_CELLS: Dict[str, PanelCell] = {
    "heldout72_hpfix": PanelCell(f"{_MEAS}/heldout72_hpfix_kakuna_2026-10-18/anchors_heldout72_hpfix_kakuna.json",
                                 "dcd993083c46764415dad7fec04ca8efe778403726ee1bdbaf78a5aa742c3916"),
    "forum72": PanelCell(f"{_MEAS}/x37a_eval_matrix_2026-10-18/configs/anchors_x37a_forum72.json",
                         "0546e7c34c8083503dfae7dab2c45b3826b801e45cdc5a90e02e711fa2d15ea6"),
    "p72": PanelCell(f"{_MEAS}/x37a_eval_matrix_2026-10-18/configs/anchors_x37a_p72.json",
                     "71bd9bad2d69cdc51649c345428cb6b6605db7f262ab6c16f31a1934f88072ac"),
}
#: D1's PRIMARY H panels, in the registered order.
UNFAMILIAR = ("heldout72_hpfix", "forum72", "p72")
UNIT_SEEDS = tuple(range(7))
UNIT_GAMES = 20
CORES = "5-7,13-15"
MEM_GB = "12"
PORT_RANGE = (9540, 9599)
#: The address a port probe binds (this queue's rule since FR-3: every interface, no ``SO_REUSEADDR``).
PORT_HOST = "0.0.0.0"
#: Every supervisor's port claims (one ``<port>.lock`` flock per port, held for the unit's life): two supervisors
#: never pick the same port, which bind-and-release alone could not promise (D1 look 1's 9540 collision).
PORT_LOCK_DIR = Path.home() / ".claude" / "jobs" / "x35_kakuna_ports"
#: A unit DECLARES the heavy slots' SHORT LANE (``mem_cap.sh --max-s``): it plays in 2-5 min, so it need not wait
#: out the 40-min S chunks ahead of it (D1 look 1's starvation). An overrun is killed at ``UNIT_MAX_S`` + 30 s and
#: exits :data:`SHORT_LANE_OVERRUN`; :func:`run_unit` then plays the unit ONCE more as a normal heavy-slot (long) job.
UNIT_MAX_S = 600
#: ``mem_cap.sh``'s exit for a declared-short job that overran its hard timeout (its work is incomplete).
SHORT_LANE_OVERRUN = 88
#: The long retry's own bound, inside its heavy slot (``timeout``; the slots' chunk rule is <= 45 min).
UNIT_LONG_S = 2700
MANIFEST_SCHEMA = "x35_home72_manifest_v1"
MAX_ATTEMPTS = 2


class KakunaError(RuntimeError):
    """A precondition of a unit does not hold (named; the unit does not run)."""


# ---------------------------------------------------------------------------------------------- the manifest
def _team_hashes(d: Path, suffix: str) -> Dict[str, str]:
    return {p.name[: -len(suffix)]: sha256_file(p) for p in sorted(d.iterdir()) if p.name.endswith(suffix)}


def build_manifest(config: Path = CONFIG) -> Dict[str, Any]:
    cfg = json.loads(config.read_text())
    ours = Path(cfg["our_team_sources"]["home"]["path"])
    mm = cfg["opponents"]["metamon"]
    theirs = Path(mm["cache_dir"]) / "teams" / mm["team_sets"]["home"] / "gen3ou"
    return {"schema": MANIFEST_SCHEMA, "config": str(config.relative_to(repo_root())), "config_sha256": sha256_file(config),
            "our_team_dir": str(ours), "our_teams": _team_hashes(ours, ".txt"),
            "metamon_team_dir": str(theirs), "metamon_teams": _team_hashes(theirs, ".gen3ou_team")}


def check_manifest(manifest: Path = MANIFEST, config: Path = CONFIG, *,
                   our_dir: Optional[Path] = None, their_dir: Optional[Path] = None) -> List[str]:
    """Every way the home72 set on disk differs from the committed manifest (empty = it is the X26 read's)."""
    m = json.loads(manifest.read_text())
    probs: List[str] = []
    if m.get("schema") != MANIFEST_SCHEMA:
        probs.append(f"manifest schema {m.get('schema')!r}")
    if sha256_file(config) != m["config_sha256"] or m["config_sha256"] != CONFIG_SHA256:
        probs.append(f"{config}: sha256 is not the X26 read's {CONFIG_SHA256[:12]}")
    ours = our_dir or Path(m["our_team_dir"])
    theirs = their_dir or Path(m["metamon_team_dir"])
    for name, d, suffix, want in (("our", ours, ".txt", m["our_teams"]),
                                  ("Metamon's", theirs, ".gen3ou_team", m["metamon_teams"])):
        if not d.is_dir():
            probs.append(f"{name} team dir {d} is missing")
            continue
        got = _team_hashes(d, suffix)
        if got != want:
            diff = sorted(set(got) ^ set(want)) + sorted(k for k in set(got) & set(want) if got[k] != want[k])
            probs.append(f"{name} home72 teams differ from the manifest: {diff[:5]}{' …' if len(diff) > 5 else ''}")
    if m["our_teams"] != m["metamon_teams"]:
        probs.append("the manifest's two copies of home72 are not byte-identical")
    return probs


# ------------------------------------------------------------------------------------------------- the units
@dataclass(frozen=True)
class Unit:
    name: str
    label: str
    arm: str
    seed: int          # the final's training seed
    teamset: str       # home72 | away
    unit_seed: int
    model: str
    model_sha256: str
    opponent: str = OPPONENT
    games: int = UNIT_GAMES
    #: a model-LOAD option: ``main.anchors --our-belief-prior-mix`` (``agents.model.belief_prior_mix``); None = none
    belief_prior_mix: Optional[float] = None
    #: a DERIVED config (:func:`opponent_config`) and its sha256 at plan; None = the cell's committed config
    config: Optional[str] = None
    config_sha256: Optional[str] = None


def parse_teamsets(spec: str) -> Tuple[str, ...]:
    """``--teamsets``: a comma list of :data:`TEAMSETS` / :data:`PANEL_CELLS` names; ``unfamiliar`` = D1's three."""
    out: List[str] = []
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        for ts in (UNFAMILIAR if tok == "unfamiliar" else (tok,)):
            if ts not in TEAMSETS and ts not in PANEL_CELLS:
                raise KakunaError(f"team set {ts!r} is none of {sorted(TEAMSETS) + sorted(PANEL_CELLS)}")
            if ts in out:
                raise KakunaError(f"team set {ts!r} named twice")
            out.append(ts)
    if not out:
        raise KakunaError("no team set named")
    return tuple(out)


def parse_unit_seeds(spec: str) -> Tuple[int, ...]:
    """``"0-19"`` or ``"0,1,2"`` (or a mix) -> the sorted, distinct unit seeds; a read that wants more games per
    cell than the registered 7 (X26c's 20, X51's seeds) names them, and seeds 0-6 stay the shared D1 subset."""
    out: List[int] = []
    for part in (p.strip() for p in str(spec).split(",")):
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        try:
            a, b = (int(lo), int(hi)) if sep else (int(part), int(part))
        except ValueError:
            raise KakunaError(f"unit seeds {spec!r}: {part!r} is not an integer or a range") from None
        if a < 0 or b < a:
            raise KakunaError(f"unit seeds {spec!r}: bad range {part!r}")
        out.extend(range(a, b + 1))
    if not out or len(set(out)) != len(out):
        raise KakunaError(f"unit seeds {spec!r}: empty or repeated")
    return tuple(sorted(out))


def _agent(opponent: str) -> str:
    if opponent not in OPPONENTS:
        raise KakunaError(f"opponent {opponent!r} is none of {OPPONENTS}")
    return opponent.partition("metamon:")[2]


def _registry_agent(agent: str) -> Dict[str, Any]:
    ent = json.loads(REGISTRY.read_text())["opponents"]["metamon"]["agents"].get(agent)
    if not isinstance(ent, dict) or not ent.get("weights"):
        raise KakunaError(f"metamon agent {agent!r} carries no weight pin in {REGISTRY} (EXTERNAL_ANCHORS_SOP)")
    return ent


def derive_config(base: Mapping[str, Any], agent: str, entry: Mapping[str, Any]) -> Dict[str, Any]:
    """The committed config ``base`` with ONE agent, ``entry``, added (the FR reads' ``make_configs.py``)."""
    cfg = json.loads(json.dumps(base))
    cfg["opponents"]["metamon"]["agents"][agent] = dict(entry)
    return cfg


def _config_bytes(cfg: Mapping[str, Any]) -> bytes:
    return (json.dumps(cfg, indent=1, sort_keys=True) + "\n").encode()


def opponent_config(teamset: str, opponent: str, out_dir: Optional[Path]) -> Tuple[Optional[Path], Optional[str]]:
    """``(config, sha256)`` a unit on ``teamset`` against ``opponent`` runs under: ``(None, None)`` when the cell's
    committed config names the agent (it runs there unchanged), else a DERIVED config written to ``out_dir``
    (module docstring), its pin validated through ``main.anchors.config``."""
    from main.anchors.config import AnchorConfigError, load_config

    agent = _agent(opponent)
    base = json.loads(config_of(teamset).read_text())
    if agent in base["opponents"]["metamon"]["agents"]:
        return None, None
    if out_dir is None:
        raise KakunaError(f"{opponent} on {teamset}: its config is derived, and no config dir was given")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"anchors_{teamset}__{agent}.json"
    b = _config_bytes(derive_config(base, agent, _registry_agent(agent)))
    if p.exists() and p.read_bytes() != b:
        raise KakunaError(f"{p} exists with other bytes: a derived config is written once per plan")
    p.write_bytes(b)
    try:
        load_config(p).metamon.pin(agent)
    except AnchorConfigError as exc:
        raise KakunaError(f"{p}: {opponent}'s pin does not validate: {exc}") from None
    return p, hashlib.sha256(b).hexdigest()


def check_derived_config(u: "Unit") -> List[str]:
    """Every way a unit's DERIVED config differs from its plan (empty = it holds): the file's sha256, the committed
    config it extends (equal but for the one added agent), the agent's entry carrying a weight pin."""
    if u.config is None:
        return []
    p = Path(u.config)
    try:
        b = p.read_bytes()
    except OSError as exc:
        return [f"{p}: unreadable ({exc})"]
    probs: List[str] = []
    if hashlib.sha256(b).hexdigest() != u.config_sha256:
        probs.append(f"{p}: sha256 is not the plan's {str(u.config_sha256)[:12]}")
    agent = _agent(u.opponent)
    cfg = json.loads(b)
    entry = cfg["opponents"]["metamon"]["agents"].get(agent) or {}
    if not entry.get("weights"):
        probs.append(f"{p}: {agent} carries no weight pin")
    if _config_bytes(derive_config(json.loads(config_of(u.teamset).read_text()), agent, entry)) != b:
        probs.append(f"{p}: not the committed {config_of(u.teamset).name} plus the one agent {agent}")
    return probs


def plan_units(finals: Sequence[Mapping[str, Any]], *, hash_models: bool = True,
               teamsets: Sequence[str] = DEFAULT_TEAMSETS, unit_seeds: Sequence[int] = UNIT_SEEDS,
               opponent: str = OPPONENT, config_dir: Optional[Path] = None) -> List[Unit]:
    """Every (final × team set × unit seed) unit, unit seeds outermost so a stopped queue is role- and seed-balanced
    across finals. ``teamsets`` defaults to X35's G-K cell (home72, away); D1's PRIMARY H names the unfamiliar
    panels (:data:`UNFAMILIAR`). ``unit_seeds`` defaults to the registered 0-6 (:data:`UNIT_SEEDS`). A final may
    carry ``belief_prior_mix`` (a model-load option every one of its units passes). ``opponent`` is one of
    :data:`OPPONENTS`; one its cell's committed config lacks gets a derived config in ``config_dir``."""
    teamsets = parse_teamsets(",".join(teamsets))
    cfgs = {ts: opponent_config(ts, opponent, config_dir) for ts in teamsets}
    seen = set()
    for f in finals:
        if f["label"] in seen:
            raise KakunaError(f"final label {f['label']!r} appears twice")
        seen.add(f["label"])
    shas = {f["label"]: (sha256_file(f["path"]) if hash_models else "") for f in finals}
    return [Unit(name=f"{f['label']}__{ts}__{s:02d}", label=str(f["label"]), arm=str(f["arm"]), seed=int(f["seed"]),
                 teamset=ts, unit_seed=s, model=str(f["path"]), model_sha256=shas[f["label"]], opponent=opponent,
                 belief_prior_mix=(None if f.get("belief_prior_mix") is None else float(f["belief_prior_mix"])),
                 config=None if cfgs[ts][0] is None else str(cfgs[ts][0]), config_sha256=cfgs[ts][1])
            for s in unit_seeds for ts in teamsets for f in finals]


def counts(summary: Mapping[str, Any], games: int) -> bool:
    """§5.6.6's rule: a unit COUNTS iff OK, complete, regime-verified, symmetric-sourced and peer-clean — THE ONE
    rule, :func:`main.anchors.results.unit_counts`, undeclared (these cells are symmetric home72 / away reads; a
    summary whose two sides' content digests differ no longer counts on equal file counts alone, F-DV-5)."""
    from main.anchors.results import unit_counts

    return unit_counts(summary, games=games).counts


def read_summary(state: Path, name: str) -> Optional[Dict[str, Any]]:
    p = state / "cells" / name / "summary.json"
    try:
        return dict(json.loads(p.read_text()))
    except (OSError, ValueError):
        return None


def is_done(state: Path, u: Unit) -> bool:
    s = read_summary(state, u.name)
    return s is not None and counts(s, u.games)


def choose_port(lo: int = PORT_RANGE[0], hi: int = PORT_RANGE[1]) -> int:
    """The first 95xx port nobody holds (bound and released at once; 8000 / 8001 are never in range)."""
    p = ports.first_bindable(lo, hi, host=PORT_HOST, reuse_addr=False)
    if p is None:
        raise KakunaError(f"no free port in {lo}-{hi}")
    return p


def claim_port(lo: int = PORT_RANGE[0], hi: int = PORT_RANGE[1], lock_dir: Path = PORT_LOCK_DIR) -> Tuple[int, Any]:
    """``(port, held lock file)``: the first 95xx port no OTHER supervisor has claimed (an exclusive flock on
    ``<lock_dir>/<port>.lock``) and nobody holds (bound and released at once; :func:`utils.ports.claim`). The caller
    keeps the file open until its unit ends; closing it releases the claim."""
    got = ports.claim(lo, hi, lock_dir, host=PORT_HOST, reuse_addr=False)
    if got is None:
        raise KakunaError(f"no free, unclaimed port in {lo}-{hi}")
    return got


def config_of(teamset: str) -> Path:
    """The anchors config a unit on ``teamset`` runs under (:data:`CONFIG` for home72 / away)."""
    return PANEL_CELLS[teamset].path if teamset in PANEL_CELLS else CONFIG


def config_side(teamset: str) -> str:
    """The config's team-set key a unit on ``teamset`` passes as ``--teamset``."""
    return PANEL_CELLS[teamset].teamset if teamset in PANEL_CELLS else TEAMSETS[teamset]


def unit_argv(u: Unit, out: Path, port: int, py: str = sys.executable, *, short: bool = True) -> List[str]:
    """One unit's command. ``short``: the heavy slots' short lane (``--max-s``, hard-killed past it, exit
    :data:`SHORT_LANE_OVERRUN`); otherwise a normal heavy-slot job bounded by ``timeout`` :data:`UNIT_LONG_S`."""
    a = [py, "-m", "main.anchors", "--model", u.model, "--opponent", u.opponent, "--regime", "greedy",
         "--teamset", config_side(u.teamset), "--games", str(u.games), "--team-seed", str(u.unit_seed),
         "--seed-base", str(u.unit_seed), "--device", "cpu", "--port", str(port), "--nice", "19", "--out", str(out)]
    if u.belief_prior_mix is not None:
        a += ["--our-belief-prior-mix", repr(float(u.belief_prior_mix))]
    if u.teamset in PANEL_CELLS:
        # the front end's per-battle repro record (its START command carries BOTH packed teams): which team each side
        # piloted in each game (F-DV-6 / F-DV-17), so a matchup-clustered sensitivity read stays possible (F-DV-18).
        # It changes no stamp and no play (the seed comes from --seed-base); home72 / away units are unchanged.
        a += ["--capture-dir", str(out / "capture")]
    cap = ["--max-s", str(UNIT_MAX_S), MEM_GB] if short else [MEM_GB, "timeout", str(UNIT_LONG_S)]
    return ["nice", "-n", "19", "taskset", "-c", CORES, str(repo_path("scripts/ops/mem_cap.sh")), "--name",
            "x35kakuna", *cap, *a]


def unit_env(base: Optional[Mapping[str, str]] = None, config: Path = CONFIG) -> Dict[str, str]:
    env = dict(os.environ if base is None else base)
    env.update({"PYTHONPATH": str(repo_path("src")), "CUDA_VISIBLE_DEVICES": "", "GEN3AI_ANCHORS_CONFIG": str(config),
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    return env


def check_panel_cell(teamset: str, *, config: Optional[Path] = None, want_sha256: Optional[str] = None) -> List[str]:
    """Every way a registered panel's Kakuna cell on disk differs from its registration (empty = it holds): the
    config's sha256, the panel's committed copy against the registry, and BOTH sides' team dirs (our byte copy, and
    Metamon's cache set) by count and ``gen3_team_set_digest_v1`` content digest against the registry."""
    from main import heldout_panels as HP
    from main.anchors.session import load_team_texts, team_set_digest

    cell = PANEL_CELLS[teamset]
    cfg_path = config or cell.path
    probs: List[str] = []
    want = want_sha256 or cell.sha256
    if sha256_file(cfg_path) != want:
        probs.append(f"{cfg_path}: sha256 is not the registered {want[:12]}")
    try:
        ent = HP.panel_entry(teamset)
        HP.panel_team_texts(teamset)
    except (KeyError, HP.PanelRegistryError) as exc:
        return probs + [f"panel {teamset}: {exc}"]
    cfg = json.loads(cfg_path.read_text())
    mm = cfg["opponents"]["metamon"]
    sides = {"our": Path(cfg["our_team_sources"][cell.teamset]["path"]),
             "Metamon's": Path(mm["cache_dir"]) / "teams" / mm["team_sets"][cell.teamset] / "gen3ou"}
    for name, d in sides.items():
        if not d.is_dir():
            probs.append(f"{name} {teamset} team dir {d} is missing")
            continue
        n = len(load_team_texts(d))
        if n != int(ent["n"]):
            probs.append(f"{name} {teamset} team dir {d}: {n} teams, registered {ent['n']}")
        try:
            got = team_set_digest(d)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            probs.append(f"{name} {teamset} team dir {d}: unreadable ({exc})")
            continue
        if got != ent["content_digest"]:
            probs.append(f"{name} {teamset} team dir {d}: content digest {got[:12]} != registered "
                         f"{ent['content_digest'][:12]}")
    return probs


def preflight(u: Unit) -> None:
    """Refuse a unit on a T28 halt, a moved home72 set (or panel cell), or a model whose bytes changed since the
    plan."""
    from main.live.halt import read_halt

    if read_halt() is not None:
        raise KakunaError("a T28 live-parse HALT marker is set (python -m main.live.halt status): no play")
    if u.teamset in PANEL_CELLS:
        probs = check_panel_cell(u.teamset)
        if probs:
            raise KakunaError(f"the {u.teamset} panel cell does not hold: {probs}")
    else:
        probs = check_manifest()
        if probs:
            raise KakunaError(f"the home72 manifest does not hold: {probs}")
    probs = check_derived_config(u)
    if probs:
        raise KakunaError(f"the derived {u.opponent} config does not hold: {probs}")
    if u.model_sha256 and sha256_file(u.model) != u.model_sha256:
        raise KakunaError(f"{u.model}: sha256 changed since the plan")


def _jl(state: Path, obj: Mapping[str, Any]) -> None:
    with open(state / "units.jsonl", "a") as f:
        f.write(json.dumps({"t": time.time(), **obj}) + "\n")


def _overran(state: Path, u: Unit) -> bool:
    """Has this unit already overrun the short lane (a ``short_lane_overrun`` event in the state's log)? Then every
    later attempt plays it as a long job at once, never short again."""
    try:
        lines = (state / "units.jsonl").read_text().splitlines()
    except OSError:
        return False
    for line in lines:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("event") == "short_lane_overrun" and ev.get("unit") == u.name:
            return True
    return False


def _play(state: Path, u: Unit, *, short: bool) -> int:
    """ONE attempt of a unit on the short lane or as a long job: its partial dir (a killed earlier attempt's) is moved
    aside, its port is claimed with a held flock, and the child's exit code is returned."""
    preflight(u)
    out = state / "cells" / u.name
    if out.exists():
        out.rename(out.with_name(f"{u.name}.partial.{int(time.time())}"))
    port, claim = claim_port()
    try:
        argv = unit_argv(u, out, port, short=short)
        _jl(state, {"event": "start", "unit": u.name, "port": port, "short": short, "argv": argv,
                    "load": os.getloadavg()})
        (state / "logs").mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(state / "logs" / f"{u.name}.log", "ab") as log:
            cfg = Path(u.config) if u.config is not None else config_of(u.teamset)
            proc = subprocess.Popen(argv, cwd=str(repo_root()), env=unit_env(config=cfg),
                                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
            _jl(state, {"event": "pid", "unit": u.name, "pid": proc.pid})
            rc = proc.wait()
    finally:
        claim.close()
    _jl(state, {"event": "end", "unit": u.name, "rc": rc, "counted": is_done(state, u), "short": short,
                "wall_s": round(time.time() - t0, 1)})
    return rc


def run_unit(state: Path, u: Unit) -> int:
    """Play ``u`` (one ATTEMPT). It starts on the short lane; if that overran (exit :data:`SHORT_LANE_OVERRUN`) and the
    unit is not counted, it is played ONCE more as a normal heavy-slot job (``short_lane_overrun`` is logged), as FR-3's
    queue did, so an overrun is never left uncounted for want of a second chance. A unit that has overrun before goes
    straight to the long job on every later attempt. Returns the last child's exit code."""
    short = not _overran(state, u)
    rc = _play(state, u, short=short)
    if short and rc == SHORT_LANE_OVERRUN and not is_done(state, u):
        _jl(state, {"event": "short_lane_overrun", "unit": u.name, "rc": rc})
        rc = _play(state, u, short=False)
    return rc


def load_plan(state: Path) -> List[Unit]:
    return [Unit(**d) for d in json.loads((state / "plan.json").read_text())["units"]]


def check_stop_epoch(e: Optional[int], now: Optional[float] = None, horizon_s: int = 7 * 86400) -> None:
    """An ABSOLUTE epoch with a sanity bound (TRAINING_RUN_SOP process rule 1), never a clock string."""
    if e is None:
        return
    t = time.time() if now is None else now
    if not t < e <= t + horizon_s:
        raise KakunaError(f"--stop-at-epoch {e} is not in (now, now + {horizon_s // 86400} d] (now {int(t)})")


def cmd_run(state: Path, stop_at: Optional[int]) -> int:
    """The standalone queue: ONE supervisor per kakuna state PATH. It takes the shared path lock OUTSIDE the state plus
    the ``/proc`` scan (:class:`main.reads.supervisors.Hold`; a lock inside the state moves with a renamed dir, the D3
    reader's duplicate, 2026-10-26) and the in-state ``supervisor.lock`` too (a supervisor of older code takes only
    that one). It stops, writing nothing, when its path no longer names the dir it started on."""
    from main.reads import supervisors as SV

    check_stop_epoch(stop_at)
    try:
        hold = SV.Hold(state, SV.KAKUNA_LANE)
    except SV.SupervisorError as exc:
        raise KakunaError(str(exc)) from None
    lock = try_exclusive(state / "supervisor.lock", "w")
    if lock is None:
        hold.close()
        raise KakunaError(f"another supervisor holds {state / 'supervisor.lock'}")
    try:
        _jl(state, {"event": "supervisor_start", "pid": os.getpid()})
        fails: Dict[str, int] = {}
        for u in load_plan(state):
            while not is_done(state, u) and fails.get(u.name, 0) < MAX_ATTEMPTS:
                try:
                    hold.check()                     # the path still names the dir this supervisor started on
                except SV.SupervisorError as exc:
                    raise KakunaError(str(exc)) from None
                if stop_at is not None and time.time() >= stop_at:
                    _jl(state, {"event": "supervisor_stop_at"})
                    return 0
                run_unit(state, u)
                if not is_done(state, u):
                    fails[u.name] = fails.get(u.name, 0) + 1
        _jl(state, {"event": "supervisor_done",
                    "not_counted": sorted(n for n, k in fails.items() if k >= MAX_ATTEMPTS)})
        return 0
    finally:
        lock.close()
        hold.close()


def pool(state: Path, units: Sequence[Unit]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """``{label: {teamset: {wins, ties, n, units, skipped}}}`` over COUNTED units only."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for u in units:
        c = out.setdefault(u.label, {}).setdefault(u.teamset, {"arm": u.arm, "seed": u.seed, "wins": 0, "ties": 0,
                                                               "n": 0, "units": 0, "skipped": []})
        s = read_summary(state, u.name)
        if s is None:
            continue
        if not counts(s, u.games):
            c["skipped"].append(u.name)
            continue
        c["units"] += 1
        c["wins"] += int(s["wins"])
        c["ties"] += int(s.get("ties", 0))
        c["n"] += int(s["n"])
    return out


def gk_input(pooled: Mapping[str, Mapping[str, Mapping[str, Any]]], on: str, off: str) -> Dict[str, Any]:
    """G-K's ``{teamset: {"on": [wins, n], "off": [wins, n]}}``, each side pooled over its arm's seed finals."""
    out: Dict[str, Any] = {}
    for ts in TEAMSETS:
        side: Dict[str, List[int]] = {"on": [0, 0], "off": [0, 0]}
        for cells in pooled.values():
            c = cells.get(ts)
            if c is None:
                continue
            key = "on" if c["arm"] == on else "off" if c["arm"] == off else None
            if key:
                side[key][0] += int(c["wins"])
                side[key][1] += int(c["n"])
        out[ts] = side
    return out


def d1h_input(pooled: Mapping[str, Mapping[str, Mapping[str, Any]]], on: str, off: str,
              panels: Sequence[str] = UNFAMILIAR) -> Dict[str, Any]:
    """D1's PRIMARY H input (:func:`main.x35.d1.read_d1`'s ``h``): ``{"on": {seed: {panel: [wins, n]}}, "off":
    ...}`` over COUNTED units, each seed final's own counts per panel (a WIN is the success; ties are listed beside,
    never scored), plus the units counted and skipped per cell."""
    out: Dict[str, Any] = {"on": {}, "off": {}, "ties": {"on": {}, "off": {}}, "units": {"on": {}, "off": {}}}
    for lab, cells in sorted(pooled.items()):
        for ts in panels:
            c = cells.get(ts)
            if c is None:
                continue
            key = "on" if c["arm"] == on else "off" if c["arm"] == off else None
            if key is None:
                continue
            seed = str(c["seed"])
            if ts in out[key].get(seed, {}):
                raise KakunaError(f"two finals of arm {c['arm']} share seed {seed} on {ts} ({lab})")
            out[key].setdefault(seed, {})[ts] = [int(c["wins"]), int(c["n"])]
            out["ties"][key].setdefault(seed, {})[ts] = int(c["ties"])
            out["units"][key].setdefault(seed, {})[ts] = {"counted": int(c["units"]), "skipped": list(c["skipped"])}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x35.kakuna", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("plan", "run", "status", "gk", "d1h", "smoke"):
        sp = sub.add_parser(n)
        sp.add_argument("--state", required=True, help="the read's state dir (outside the repo and models/)")
        if n == "plan":
            sp.add_argument("--finals", required=True, help="JSON [{label, arm, seed, path}]")
            sp.add_argument("--teamsets", default=",".join(DEFAULT_TEAMSETS),
                            help="comma list of team sets / registered panels (default home72,away = G-K's cell); "
                                 "'unfamiliar' = D1's heldout72_hpfix,forum72,p72")
            sp.add_argument("--unit-seeds", default="0-6",
                            help="unit seeds, a comma list of integers / ranges (default 0-6, the registered 7)")
        if n == "run":
            sp.add_argument("--stop-at-epoch", type=int, default=None)
        if n in ("gk", "d1h"):
            sp.add_argument("--on", required=True)
            sp.add_argument("--off", required=True)
        if n == "smoke":
            sp.add_argument("--model", required=True)
            sp.add_argument("--games", type=int, default=2)
    sub.add_parser("panels", help="check every unfamiliar panel cell (config sha256, both sides' content)")
    m = sub.add_parser("manifest")
    m.add_argument("--write", action="store_true", help="(re)write the committed manifest from the files on disk")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "manifest":
            if a.write:
                MANIFEST.write_text(json.dumps(build_manifest(), indent=1, sort_keys=True) + "\n")
            probs = check_manifest()
            print(f"[x35 kakuna] manifest: {'OK' if not probs else probs}")
            return 0 if not probs else 1
        if a.cmd == "panels":
            bad = 0
            for ts in PANEL_CELLS:
                probs = check_panel_cell(ts)
                bad += bool(probs)
                print(f"[x35 kakuna] panel {ts}: {'OK' if not probs else probs}")
            return 0 if not bad else 1
        state = Path(a.state)
        state.mkdir(parents=True, exist_ok=True)
        if a.cmd == "plan":
            if (state / "plan.json").exists():
                raise KakunaError(f"{state / 'plan.json'} exists: a plan is written once per read")
            units = plan_units(json.loads(Path(a.finals).read_text()), teamsets=parse_teamsets(a.teamsets),
                               unit_seeds=parse_unit_seeds(a.unit_seeds))
            (state / "plan.json").write_text(json.dumps({"units": [u.__dict__ for u in units]}, indent=1) + "\n")
            print(f"[x35 kakuna] planned {len(units)} units")
            return 0
        if a.cmd == "run":
            return cmd_run(state, a.stop_at_epoch)
        if a.cmd == "smoke":
            u = Unit(name="smoke__home72__00", label="smoke", arm="smoke", seed=0, teamset="home72", unit_seed=0,
                     model=a.model, model_sha256=sha256_file(a.model), opponent=SMOKE_OPPONENT, games=a.games)
            rc = run_unit(state, u)
            s = read_summary(state, u.name) or {}
            keys = ("status", "n", "wins", "losses", "ties", "regime_verified_decisions", "team_source_asymmetry",
                    "peer_clean")
            print(json.dumps({"rc": rc, "counted": counts(s, u.games), **{k: s.get(k) for k in keys}}))
            return 0 if counts(s, u.games) else 1
        units = load_plan(state)
        pooled = pool(state, units)
        if a.cmd == "status":
            planned: Dict[Tuple[str, str], int] = {}
            for u in units:
                planned[(u.label, u.teamset)] = planned.get((u.label, u.teamset), 0) + 1
            for lab, cells in pooled.items():
                for ts, c in cells.items():
                    n_plan = planned.get((lab, ts), 0)
                    print(f"{lab:24s} {ts:7s} {c['units']}/{n_plan} units  {c['wins']}/{c['n']} wins  "
                          f"skipped {len(c['skipped'])}")
            return 0
        if a.cmd == "d1h":
            print(json.dumps(d1h_input(pooled, a.on, a.off), sort_keys=True))
            return 0
        print(json.dumps(gk_input(pooled, a.on, a.off)))
        return 0
    except KakunaError as exc:
        print(f"[x35 kakuna] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
