"""THE UNTAUGHT METER — the engine behind ``python -m main.untaught_meter``.

WHAT IT MEASURES. The win rate of a checkpoint **piloting** a fixed set of teams against ONE fixed
opponent, cluster-bootstrapped over TEAMS. When the team set is the reuse batch's **untaught 8** —
teams no teacher in the fleet ever trained on — that number is the off-slice competence every fold
verdict in the ledger rests on. Point it at the taught-16 manifest instead and the same instrument
reads the on-slice half.

WHY IT IS IN-TREE. It existed only as per-batch probe scripts copied between measurement
directories (``teacher_content_2x2_2026-09-04/untaught_probe.py``,
``reuse_batch_2026-09-03/offline_collateral_kl/``) and as uncommitted job-dir scripts. Every copy
carried its own seed convention, its own aggregation and its own idea of what the baseline is; two
of them disagreed about whether the levels were even reproducible. One meter, one recipe.

🚨 **THE CONTINUATION CONTROL IS NOT OPTIONAL AND THE METER SAYS SO.** Ledger 2026-09-06 (cell 2):
a plain +1.08M-step continuation of v8's parent — no teacher, no distillation term, no stable
opponents — moved the untaught meter **+3.45pp [+0.46, +6.48]** all by itself. An untaught delta
measured against a **frozen** parent therefore credits a fold with progress the parent would have
made anyway: re-based on a continuation control, v8's celebrated +4.64pp becomes ≈ +1.2pp and is
not significant. So this meter reports TWO delta columns whenever ``--control`` is given — vs the
frozen baseline, and vs the continuation arms at matched depth — and applies the verdict vocabulary
to both. A single column is a fold's *apparent* gift; the second is what is left after the parent's
own free progress is removed.

THE GAMES ARE PLAYED ON THE RUST EVAL CORE (poke-env retirement P2, ``agents.training.untaught_rust``).
One (ref, team) cell is one eval cycle of the Rust eval core with the T2 inference service on the CPU,
on the training encoder's rows — the poke-env ``RLPlayer`` pair over the bridge (the Python encoder)
is gone from the meter. 🚨 **That switch is a REGIME BOUNDARY**: every cell / artifact carries
``transport = "rust_eval"`` (:data:`TRANSPORT_RUST`); one without it is ``"python_bridge"``
(:data:`TRANSPORT_LEGACY`, pre-boundary), and every reader that puts reads side by side
(:func:`aggregate`, :func:`merge_cells`, ``--from-rows``) REFUSES a mix unless told
(``allow_transport_mix``).

REPRODUCIBILITY. Every game is a pure function of its key — (the cell's cycle seed, a function of
``--seed`` and the team index only; the game index) — through the per-game seed rule
(``rust_eval.seeds``) and the KEYED DRAW, never of the env count, the threads or the shard process,
except that a decision within a rounding error of a tie can flip with the forward's batch shape (the
env count). At a fixed compute a cell replays bit for bit, so **sharding over TEAMS** (``--workers``)
cannot move a number. ``concurrency`` is RETIRED (:func:`check_concurrency`): battles in flight are the
core's envs, and they no longer share a stream.

CRN. For a given (team ``ti``, battle ``j``) the battle seed, the opponent's team and both sides'
sampling-stream keys are identical across every ref — only the ref's weights change — so every
ref-vs-ref difference is a PAIRED difference on the same games. Each game's draw is its own, so the
schedule is prefix-consistent: a ref measured at 12 games/team plays the first 12 of another's 200.

THE TURN LIMIT. The core ends a game at the declared turn limit as a DRAW, as training does, so the
TIMEOUT bucket below is 0 on a Rust read and those games are TIES (``Cell.turn_limit_draws``).

AGGREGATION. Team is the cluster (between-team variance dominates and more games per team does not
shrink it). **One fixed resampling index set is shared by every ref and every contrast**, so a
ref-vs-ref difference is paired on the same team draws instead of each arm carrying independent
noise. A pooled level is the equal-weight mean of the per-team rates — never the state/game-weighted
pool, which can disagree in SIGN (measured: C1−B2 was +0.0309 pooled and −0.0188 clustered).

VERDICT VOCABULARY, as ruled: ``WITHIN FLOOR`` (|Δ| below the replicate floor — the CI may still
exclude zero, which says the games are consistent, not that the arm differs), ``NOT DETECTED``
(|Δ| clears the floor but the CI spans zero), ``SIGNIFICANT``. A run whose timeouts exceed 25% of
attempted battles is ``INCONCLUSIVE`` and reports no verdict at all — a timeout is never a semantic
outcome (a pre-boundary ``python_bridge`` read could carry timeouts; a Rust read carries none).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from agents.training import baselines
from agents.training.fixed_opponent_pool import resolve_model_ref
from agents.training.team_archetypes import team_sha
from utils.paths import main_models_dir, repo_path
from utils.torch_state_guard import restores_torch_globals

# --------------------------------------------------------------------------------------------
# Defaults — the recipe the banked artifacts were produced under
# --------------------------------------------------------------------------------------------

#: The reuse batch's UNTAUGHT 8, IN THE ORDER THE SEEDS ARE CONSUMED. The order is part of the
#: measurement (index = team seed offset); never sort it.
DEFAULT_TEAMS_MANIFEST = repo_path(
    "designs/research_state/measurements/reuse_batch_2026-09-03/offline_collateral_kl",
    "untaught_teams.json")

#: The taught-16 slice — every 2×2 arm's own (deleted) ``--distill-teacher`` expanded to exactly this set.
DEFAULT_TAUGHT_MANIFEST = repo_path(
    "designs/research_state/measurements/teacher_content_2x2_2026-09-04", "taught_teams.json")

#: The fixed opponent, BY NAME out of ``designs/baselines.json`` (``gen3_baselines_registry_v1``).
#: **INTERIM (legacy-manifest B3, 2026-10-04):** the default was ``untaught_meter_opponent`` (rev-1's
#: 24M snapshot, config v101), which does NOT load at HEAD (a pre-generation checkpoint), so a bare
#: ``python -m main.untaught_meter`` died at its first model load (F-LR-2). It is now the v14
#: opponent, N0's 24M snapshot (config v121, ``gen3_event_record_v2``), which loads and is the
#: opponent every post-M5 untaught read (the X5 A/B's, P0's, the sizing study's) already used, so
#: this re-point opens NO new series. It stays interim until the Rustboro opponent replaces it
#: (``designs/ops/legacy_removal_manifest.md`` D-L3 / R0). **A new opponent is a RE-MEASUREMENT, not a
#: rename**: levels are not comparable across opponents, so the registry entry carries that fact
#: forward, and every artifact now stamps ``_meta["series"]`` (see :func:`series_identity`).
DEFAULT_OPPONENT_BASELINE = "untaught_meter_opponent_v14"
#: The module tree every model is loaded against. ``"auto"`` = each model's OWN ``model_config.json``,
#: which is the only value that loads a CURRENT checkpoint (a shared config from another generation
#: cannot) and is what every post-M5 read used. The v101 shared config (``untaught_meter_config``)
#: remains in the registry as history and is still selectable with ``--config untaught_meter_config``.
DEFAULT_CONFIG = "auto"


def default_opponent() -> str:
    """The fixed opponent's run spec, from the registry. Raises if the registry cannot be read."""
    return baselines.spec(DEFAULT_OPPONENT_BASELINE)


def series_identity(opponent_json: dict) -> dict:
    """The identity of the measurement SERIES an artifact belongs to: which opponent it played.

    ``opponent_json`` is a :meth:`ResolvedRef.to_json` (or a banked artifact's ``_meta.opponent``).
    Untaught levels against different opponents are NOT one scale (F-X5-20), so this is what a
    reader compares before putting two levels side by side. ``key`` is the sha256 when the artifact
    recorded one (new artifacts do), else ``<run_base>/<file name>`` (every banked artifact has both).
    """
    f = opponent_json.get("resolved_file") or ""
    run = opponent_json.get("run_base") or ""
    sha = opponent_json.get("sha256")
    return {"opponent_run": run, "opponent_file": os.path.basename(f),
            "opponent_num_timesteps": opponent_json.get("num_timesteps"),
            "opponent_sha256": sha,
            "key": f"sha256:{sha}" if sha else f"{run}/{os.path.basename(f)}"}


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows_artifact_series(path: str) -> Optional[dict]:
    """The series identity a committed per-team artifact recorded, or None when it names no opponent."""
    with open(path) as fh:
        raw = json.load(fh)
    meta = raw.get("_meta") if isinstance(raw, dict) else None
    opp = meta.get("opponent") if isinstance(meta, dict) else None
    if not isinstance(opp, dict) or not (opp.get("resolved_file") or opp.get("run_base")):
        return None
    return series_identity(opp)


def refuse_mixed_series(identities: Sequence[Optional[dict]], labels: Sequence[str]) -> List[str]:
    """Refuse artifacts from DIFFERENT opponents; return warnings for artifacts that name none.

    ``identities[i]`` is :func:`series_identity` of artifact ``i`` or None when it recorded no
    opponent. Two recorded identities that disagree raise :class:`MeterError` (a sha256 beats a
    file name when both sides carry one; a one-sided sha256 falls back to ``run/file``).
    """
    recorded = [(lab, i) for lab, i in zip(labels, identities) if i is not None]
    warnings = [f"{lab}: artifact records no opponent — its series is UNKNOWN, not checked"
                for lab, i in zip(labels, identities) if i is None]

    def same(a: dict, b: dict) -> bool:
        if a.get("opponent_sha256") and b.get("opponent_sha256"):
            return a["opponent_sha256"] == b["opponent_sha256"]
        return (a["opponent_run"], a["opponent_file"]) == (b["opponent_run"], b["opponent_file"])

    for lab, i in recorded[1:]:
        if not same(recorded[0][1], i):
            raise MeterError(
                f"these artifacts were played against DIFFERENT opponents — different measurement "
                f"series, never one scale (F-X5-20): {recorded[0][0]} = {recorded[0][1]['key']}, "
                f"{lab} = {i['key']}. Re-read one side against the other's opponent, or pass "
                f"--allow-opponent-mix to read them side by side knowingly.")
    return warnings


DEFAULT_GAMES_PER_TEAM = 200
DEFAULT_SEED = 0
DEFAULT_BOOTSTRAP_DRAWS = 20000
DEFAULT_BOOTSTRAP_SEED = 20260906

#: A timeout is never a semantic outcome. Above this fraction of attempted battles the run reports
#: INCONCLUSIVE instead of a level.
TIMEOUT_INCONCLUSIVE_FRACTION = 0.25

#: THE PYTHON-BRIDGE SEED CONVENTION (pre-boundary, ``transport = "python_bridge"``). The meter no longer
#: plays through it; it is kept, unchanged, as the record of the committed measurement drivers that replayed
#: the pre-boundary series (``n0_endofrun_2026-09-27/scripts/gu_unit.py``). Those drivers can no longer RUN:
#: the Python runtime they played through (``RLPlayer``, ``run_local_battles``) was deleted in T27 P6 slice 6d-2,
#: and their pin (``untaught_unit_script_test.py``) with it. At ``--seed 0`` the sim dice, pool draw and
#: per-battle policy seeds reproduced ``exploiter_competence/compete.py`` exactly.
_ENV_SEED_OFFSETS = {
    "GEN3AI_PLAYER_SEED": 10000,
    "GEN3AI_TEAM_SEED": 20000,
    "GEN3AI_POLICY_SEED": 30000,
    "GEN3AI_POOL_SEED": 40000,
    "GEN3AI_STALLER_SEED": 50000,
}
_POOL_SEED_BASE = 61000
_PILOT_POLICY_BASE = 71000
_OPP_POLICY_BASE = 72000
_SEED_STRIDE = 1000000

#: The two TRANSPORTS a cell can have been played on — a regime boundary (module docs).
TRANSPORT_RUST = "rust_eval"
TRANSPORT_LEGACY = "python_bridge"


class MeterError(RuntimeError):
    """A refusal the caller should surface verbatim (bad concurrency, unresolvable input)."""


class TransportMixError(MeterError):
    """Reads played on DIFFERENT transports (``rust_eval`` vs the pre-boundary ``python_bridge``) put side by
    side without consent (:func:`cells_transport`)."""

    cause = "transport_mix"


class MirroredPinnedTeamError(MeterError):
    """``--mirrored-pairs`` on this PINNED-TEAM meter (`mirrored_pairs.pinned_team_refusal`)."""

    cause = "mirrored_pinned_team"


# --------------------------------------------------------------------------------------------
# Inputs — teams and model refs
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TeamSlice:
    """One pinned pilot team. ``index`` is the seed offset and is positional, never sorted."""
    index: int
    key: str            # the stable row key ("U_61590463"), matching the committed probe artifacts
    path: str
    sha1: str           # sha1 of the file's RAW bytes, full
    pin_sha: str        # sha1(raw)[:10] — the MatchupSpec pin_sha convention (live-run provenance)
    team_sha: str       # team_archetypes.team_sha — the STRIP-normalized pool join key

    def to_json(self) -> dict:
        return {"index": self.index, "key": self.key, "path": self.path,
                "sha1": self.sha1, "pin_sha": self.pin_sha, "team_sha": self.team_sha}


def _team_key(path: str, prefix: str) -> str:
    """``U_61590463`` from ``data/teams/sample/61590463ee85d456.txt`` — the committed probes' key."""
    return f"{prefix}_{os.path.basename(path).split('.')[0][:8]}"


def load_team_manifest(path: str, *, prefix: str = "U") -> List[TeamSlice]:
    """Read a team manifest into ordered :class:`TeamSlice` records.

    Accepts a bare JSON list, or an object carrying the list under ``teams`` / ``untaught`` /
    ``taught``. Team paths are resolved relative to the repo root when they are not absolute, so a
    manifest works from any cwd. Raises :class:`MeterError` naming every missing file — a manifest
    is an input, and a silently-shortened team set changes the cluster count under the reader.
    """
    with open(path) as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):
        for key in ("teams", "untaught", "taught"):
            if key in raw:
                items = raw[key]
                break
        else:
            raise MeterError(f"{path}: no 'teams'/'untaught'/'taught' list in the manifest")
    else:
        items = raw
    if not isinstance(items, list) or not items:
        raise MeterError(f"{path}: the team list is empty or not a list")

    return team_slices(items, prefix=prefix, source=path)


def team_slices(paths: Sequence[str], *, prefix: str = "U",
                source: str = "team list") -> List[TeamSlice]:
    """Ordered :class:`TeamSlice` records for an explicit list of team files.

    Split out of :func:`load_team_manifest` so a caller that already HOLDS the paths — an
    exploiter's recorded `--trainee-teams` pin, read back through
    `matchup_spec.read_recorded_trainee_teams` — can build the same slices without first
    writing a manifest file. `source` only names the input in the refusal message.
    """
    slices: List[TeamSlice] = []
    missing: List[str] = []
    for i, rel in enumerate(paths):
        full = rel if os.path.isabs(rel) else str(repo_path(rel))
        if not os.path.isfile(full):
            missing.append(full)
            continue
        data = open(full, "rb").read()
        # TWO conventions, both recorded because they answer different questions and DIFFER on a
        # file with a trailing newline: ``pin_sha`` fingerprints the pin file's RAW bytes (what a
        # live run's MatchupSpec records), ``team_sha`` the STRIP-normalized text (what
        # ``data/teams/gen3_team_archetypes.json`` is keyed by). Neither is re-derived here.
        raw_sha = hashlib.sha1(data).hexdigest()
        slices.append(TeamSlice(index=i, key=_team_key(rel, prefix), path=full,
                                sha1=raw_sha, pin_sha=raw_sha[:10],
                                team_sha=team_sha(data.decode())))
    if missing:
        raise MeterError(f"{source}: {len(missing)} team file(s) missing:\n  "
                         + "\n  ".join(missing))
    return slices


@dataclass(frozen=True)
class ResolvedRef:
    """One model ref, resolved through the ONE choke point, with its provenance attached."""
    label: str
    ref: str
    role: str            # "ref" | "baseline" | "control" | "opponent"
    zip_path: str
    config_path: str
    run_base: str
    rung: str
    rule: str
    num_timesteps: Optional[int]

    def to_json(self) -> dict:
        return {"label": self.label, "ref": self.ref, "role": self.role,
                "resolved_file": self.zip_path, "config_path": self.config_path,
                "run_base": self.run_base, "resolution_rung": self.rung,
                "resolution_rule": self.rule, "num_timesteps": self.num_timesteps}

    def provenance(self) -> str:
        steps = f"@{self.num_timesteps:,} steps" if self.num_timesteps is not None else "@steps unknown"
        return f"{self.zip_path} {steps} [rung={self.rung} rule={self.rule}]"


def _candidate_paths(ref: str) -> List[str]:
    """``ai_v9_162_TCUNFA_0903`` and ``models/ai_v9_...`` both work, from a worktree too.

    ``models/`` is NOT committed and exists only in the MAIN checkout, so a bare run name is
    additionally tried under :func:`utils.paths.main_models_dir` (git's shared common dir), which
    is the accessor that reaches across from a linked worktree.
    """
    out = [ref]
    models = main_models_dir()
    if models is not None:
        base = ref.split("@", 1)[0]
        if not os.path.isabs(base) and not os.path.exists(base):
            out.append(str(models / ref))
            if ref.startswith("models/"):
                out.append(str(models.parent / ref))
    return out


def expand_baseline_name(ref: str) -> str:
    """A registry NAME → that baseline's explicit spec; anything else through unchanged.

    So ``--baseline v9_fold_parent`` works wherever a ref does, and a caller that already holds a
    name (``main.critic_gate`` forwarding its ``--parent``) can pass it straight through.
    """
    return baselines.spec(ref) if baselines.is_name(ref) else ref


def resolve_ref(ref: str, *, label: Optional[str] = None, role: str = "ref",
                config_override: Optional[str] = None) -> ResolvedRef:
    """Resolve one ref through :func:`agents.training.fixed_opponent_pool.resolve_model_ref`.

    That is THE choke point every training-side consumer uses (``--stable-opponents``,
    ``--exploiter``, …), so a bare run directory here means what it means
    to a launch: the run's LAST SNAPSHOT (``gen3_last_snapshot_resolution_v1``), not the
    bot-selected ``best_model``. The rung and rule are carried into the artifact so the reader
    never has to infer WHICH FILE was scored — the failure ledger 2026-09-06 records.
    """
    last: Exception = MeterError(f"ref {ref!r}: nothing tried")
    for cand in _candidate_paths(expand_baseline_name(ref)):
        try:
            r = resolve_model_ref(cand, warn=False)
        except (FileNotFoundError, ValueError) as exc:
            last = exc
            continue
        return ResolvedRef(
            label=label or _default_label(ref), ref=ref, role=role, zip_path=r.zip_path,
            config_path=config_override or r.config_path, run_base=r.run_base,
            rung=r.rung, rule=r.rule, num_timesteps=r.num_timesteps)
    raise MeterError(f"ref {ref!r}: {last}")


def _default_label(ref: str) -> str:
    base = ref.split("@", 1)[0].rstrip("/")
    if base.endswith(".zip"):
        parts = base.split(os.sep)
        run = parts[-3] if len(parts) >= 3 and parts[-2] in ("checkpoints", "best_model",
                                                             "snapshots") else parts[-2] if len(parts) >= 2 else parts[-1]
        return f"{run}:{os.path.basename(base)[:-4]}"
    return os.path.basename(base)


# --------------------------------------------------------------------------------------------
# Seeds — every stream this meter can reach, derived from --seed and the team index
# --------------------------------------------------------------------------------------------

def team_env_seeds(seed: int, team_index: int) -> Dict[str, str]:
    """The five global-RNG env seeds for one team's cell (``src/agents/training/CLAUDE.md``)."""
    return {k: str(off + _SEED_STRIDE * seed + team_index)
            for k, off in _ENV_SEED_OFFSETS.items()}


def sim_seed(seed: int, team_index: int, battle_index: int) -> List[int]:
    """The gen-5 PRNG seed for one battle. At ``seed=0`` this is ``compete.py``'s ``[ti+1,j+1,3,4]``."""
    return [seed + team_index + 1, battle_index + 1, 3, 4]


def pool_sequence(seed: int, team_index: int, n_games: int, n_pool: int) -> List[int]:
    """The opponent's team draw for one team's cell — ONE ``Random`` drawn sequentially.

    Prefix-consistent by construction: the first ``k`` entries of a 200-game sequence are the
    200-game sequence's first ``k``, so a cheap ref and an expensive one still play paired games.
    (Re-instantiating the ``Random`` inside a comprehension yields the SAME index every time; that
    bug was written once and caught by the per-battle ``opp_team`` column, which is why it is
    recorded.)
    """
    rng = random.Random(_POOL_SEED_BASE + _SEED_STRIDE * seed + team_index)
    return [rng.randrange(n_pool) for _ in range(n_games)]


def policy_seeds(seed: int, team_index: int, battle_index: int) -> Tuple[int, int]:
    """``(pilot, opponent)`` sampling seeds, re-set per battle so cell (ti, j) starts identically."""
    off = _SEED_STRIDE * seed + team_index * 1000 + battle_index
    return _PILOT_POLICY_BASE + off, _OPP_POLICY_BASE + off


def check_concurrency(concurrency: int) -> None:
    """The ``concurrency`` knob is RETIRED (P2): REFUSE any value but 1, with the reason.

    It existed because interleaved poke-env battles consumed shared RNG streams in a
    scheduling-dependent order (measured 2026-09-03: seeded at concurrency 3, two runs differed by up to
    +0.043 in level). On the Rust eval core every game is a pure function of its key and the core's envs
    ARE the battles in flight (``--n-envs``), so there is nothing left for this knob to set."""
    if concurrency == 1:
        return
    raise MeterError(
        f"REFUSING concurrency={concurrency}: the knob is RETIRED — the meter plays on the Rust eval core,\n"
        "  where the battles in flight are the core's envs (--n-envs) and every game is a pure function of\n"
        "  its key (cycle seed, game index), so no shared stream is consumed in a scheduling-dependent\n"
        "  order. Shard over TEAMS with --workers N, or raise --n-envs.")


def check_impl(impl: str) -> None:
    """``impl`` named the poke-env bridge's sim (``rust`` / ``node``); the meter's games now run on the Rust eval
    core, so only ``rust`` means anything. ``node`` is REFUSED with the reason."""
    if impl == "rust":
        return
    raise MeterError(
        f"REFUSING --impl {impl}: the untaught meter's games run on the Rust EVAL CORE (poke-env retirement P2) —\n"
        "  there is no bridge to pick an implementation for. The pre-boundary node-bridge series is history;\n"
        "  replay it pinned to a commit before the switch.")


# --------------------------------------------------------------------------------------------
# Cells — the raw per-(ref, team) counts
# --------------------------------------------------------------------------------------------

@dataclass
class Cell:
    """One (ref, team) result. ``attempted`` − ``finished`` is the TIMEOUT bucket, never a loss."""
    wins: int = 0
    ties: int = 0
    losses: int = 0
    finished: int = 0
    attempted: int = 0
    opp_teams: List[int] = field(default_factory=list)
    #: MIRRORED TEAM PAIRS (`gen3_mirrored_pairs_v1`): the cell's pentanomial over PAIRS (pairs scoring
    #: 0..4 half-points to the pilot). None = an unmirrored cell; its presence IS the cell's regime.
    pairs: Optional[List[int]] = None
    #: The TRANSPORT the cell was played on (``rust_eval``); None = a pre-boundary cell (``python_bridge``).
    transport: Optional[str] = None
    #: Rust eval core only: the games among ``ties`` that ended AT THE TURN LIMIT (a DRAW, as in training).
    turn_limit_draws: Optional[int] = None

    @property
    def mirrored(self) -> bool:
        return self.pairs is not None

    @property
    def transport_name(self) -> str:
        return self.transport or TRANSPORT_LEGACY

    @property
    def timeouts(self) -> int:
        return self.attempted - self.finished

    @property
    def win_rate(self) -> float:
        return self.wins / self.finished if self.finished else 0.0

    def to_json(self) -> dict:
        return {"wins": self.wins, "ties": self.ties, "losses": self.losses,
                "finished": self.finished, "attempted": self.attempted,
                "timeouts": self.timeouts, "win_rate": self.win_rate,
                "opp_teams": self.opp_teams,
                **({"pairs": list(self.pairs)} if self.pairs is not None else {}),
                **({"transport": self.transport} if self.transport is not None else {}),
                **({"turn_limit_draws": self.turn_limit_draws} if self.turn_limit_draws is not None else {})}


def cell_from_json(d: dict, transport: Optional[str] = None) -> Cell:
    """``transport``: the artifact-level stamp, used when the row carries none of its own."""
    finished = int(d.get("finished", d.get("games", 0)))
    attempted = int(d.get("attempted", finished))
    wins = int(d["wins"])
    ties = int(d.get("ties", 0))
    pairs = d.get("pairs")
    tld = d.get("turn_limit_draws")
    return Cell(wins=wins, ties=ties, losses=max(0, finished - wins - ties),
                finished=finished, attempted=attempted, opp_teams=list(d.get("opp_teams", [])),
                pairs=[int(x) for x in pairs] if pairs is not None else None,
                transport=d.get("transport") or transport,
                turn_limit_draws=int(tld) if tld is not None else None)


def cells_from_rows_artifact(path: str) -> Dict[str, Cell]:
    """Ingest a committed per-team artifact (``untaught_<TAG>_end.json``-shaped) as cells.

    The shape is ``{"_meta": …, "<TEAM_KEY>": {"wins": w, "games": n, …}, "POOLED": {…}}``. The
    ``POOLED`` row is a SUMMARY, not a team cell — counting it reports 9 clusters out of 8.
    """
    with open(path) as fh:
        raw = json.load(fh)
    meta = raw.get("_meta") if isinstance(raw, dict) else None
    transport = meta.get("transport") if isinstance(meta, dict) else None
    cells = {k: cell_from_json(v, transport) for k, v in raw.items()
             if isinstance(v, dict) and "wins" in v and k not in ("POOLED", "_meta")}
    if not cells:
        raise MeterError(f"{path}: no per-team rows (expected objects carrying 'wins'/'games')")
    return cells


# --------------------------------------------------------------------------------------------
# Playing — on the Rust eval core (`untaught_rust`, imported lazily so the maths half stays cheap)
# --------------------------------------------------------------------------------------------

def _teambuilders():
    """Build the two teambuilder subclasses lazily. LEGACY: the python-bridge path only (``gu_unit.py``
    replays the pre-boundary series through it); the meter's own games draw on the Rust eval core.

    They are defined INSIDE a function on purpose: importing ``utils.teambuilder`` at module scope
    would drag poke-env into every consumer of the pure aggregation half, which is torch-free and
    battle-free by design (the unit tests run in 0.1 s because of it).
    """
    from utils.teambuilder import Gen3Teambuilder

    class PinnedTeam(Gen3Teambuilder):
        """MUST subclass Gen3Teambuilder — ``yield_team`` has to return a PACKED team."""

        def __init__(self, path: str):
            super().__init__([open(path).read()])

        def yield_team(self):
            return self.packed_teams[0]

    class PairedPool(Gen3Teambuilder):
        """Indices are into ``packed_teams``, NOT the raw list — the builder SKIPS invalid teams."""

        def __init__(self, teams):
            super().__init__(teams)
            self._seq: List[int] = []
            self._i = 0

        def set_sequence(self, seq: Sequence[int]) -> "PairedPool":
            self._seq, self._i = list(seq), 0
            return self

        def at(self, i: int) -> "PairedPool":
            self._i = i
            return self

        def yield_team(self):
            t = self.packed_teams[self._seq[self._i % len(self._seq)]]
            self._i += 1
            return t

    return PinnedTeam, PairedPool


def _reseed_player(player, seed: int) -> None:
    """Reset a player's private sampling generator (the documented per-instance cache,
    ``gen3_policy_sample_rng_v1``) so battle (ti, j) starts identically for every ref. LEGACY: the
    python-bridge path only (``gu_unit.py``); the meter no longer plays through it."""
    player._policy_seed = int(seed)
    player._policy_gens = {}


def default_compute() -> Any:
    """The meter's default CPU compute (``main.h2h.play.Compute``): 32 envs, 2 core threads, 4 torch threads,
    the process front end, the release build. A test passes its own (``front="ffi", profile="selfcheck"``)."""
    from main.h2h.play import Compute

    return Compute(device="cpu", backend="eager", n_envs=32, threads=2, torch_threads=4, front="proc",
                   profile="release")


@restores_torch_globals          # it sets the forward's thread count for its games, and gives the caller its own back
def play_cells(
    refs: Sequence[ResolvedRef],
    teams: Sequence[TeamSlice],
    opponent: ResolvedRef,
    *,
    games_per_team: int,
    seed: int = DEFAULT_SEED,
    impl: str = "rust",
    concurrency: int = 1,
    stochastic: bool = True,
    progress=None,
    mirrored: bool = False,
    compute: Any = None,
    game_log: Optional[Dict[Tuple[str, str], List[Dict[str, Any]]]] = None,
    info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Cell]]:
    """Play every (ref × team) cell ON THE RUST EVAL CORE and return the raw counts
    (``agents.training.untaught_rust``; every cell is stamped ``transport = "rust_eval"``).

    ``stochastic`` sets BOTH sides' sampling regime and defaults to True — the TRAINING regime
    (both sides sample at T = 1.0, the keyed draw), which is what every untaught-meter level on record
    was measured in. ``False`` is the EVAL regime (argmax both sides), the one a fixed cross-run opponent
    is evaluated in; the two are different populations and a number must never leave either without
    saying which it is (`main.best_response_gap` prints the regime on every row).

    ``impl`` (only ``"rust"``: :func:`check_impl`) and ``concurrency`` (only 1: :func:`check_concurrency`)
    are RETIRED knobs, still checked so an old caller is refused with the reason. ``compute`` is a
    ``main.h2h.play.Compute`` (CPU only; default :func:`default_compute`). ``game_log`` receives each
    cell's trimmed game rows under ``(ref label, team key)``; ``info`` the engine's provenance (the
    transport, the encoder, the core stamp, the eval regime, the compute).

    ``mirrored`` is REFUSED (P13): every cell here is a PILOT on its PINNED team, and mirroring swaps it.
    """
    check_concurrency(concurrency)
    check_impl(impl)
    if mirrored:
        # P13 (owner 2026-10-02): every cell here is a PILOT on its PINNED team — the one place both
        # pinned-team meters (`main.untaught_meter`, `main.best_response_gap --play`) play through.
        from agents.training.mirrored_pairs import pinned_team_refusal
        raise MirroredPinnedTeamError(pinned_team_refusal("untaught_meter.play_cells"))
    import torch as th

    from agents.training.untaught_rust import play_cells_rust

    compute = compute if compute is not None else default_compute()
    if compute.torch_threads:
        th.set_num_threads(int(compute.torch_threads))
    return play_cells_rust(refs, teams, opponent, games_per_team=games_per_team, seed=seed,
                           stochastic=stochastic, compute=compute, progress=progress, game_log=game_log,
                           info=info)


def cells_transport(cells_by_ref: Dict[str, Dict[str, Cell]], *, allow_mix: bool = False) -> str:
    """The ONE transport every cell was played on (``rust_eval`` / the pre-boundary ``python_bridge``).

    A MIX is :class:`TransportMixError` — the transport switch (poke-env retirement P2) is a regime
    boundary: the encoder, the sampler and the turn-limit rule all changed with it — unless
    ``allow_mix``, which returns ``"MIXED:<a>+<b>"`` so the readout says so."""
    kinds = sorted({c.transport_name for t in cells_by_ref.values() for c in t.values()})
    if len(kinds) <= 1:
        return kinds[0] if kinds else TRANSPORT_RUST
    if allow_mix:
        return "MIXED:" + "+".join(kinds)
    by = {k: sorted(lab for lab, t in cells_by_ref.items() if any(c.transport_name == k for c in t.values()))
          for k in kinds}
    raise TransportMixError(
        f"these reads were played on DIFFERENT transports {by} — the Rust eval core (rust_eval) and the "
        "pre-boundary poke-env bridge (python_bridge) are different regimes (encoder, sampler, turn-limit "
        "rule); re-read one side on the other's transport, or pass --allow-transport-mix to read them side "
        "by side knowingly")



def cells_regime(cells_by_ref: Dict[str, Dict[str, Cell]]) -> bool:
    """True iff every cell is MIRRORED; False iff none is. A MIX is REFUSED — the two regimes are not
    one population (a mirrored cell plays half its games on a pool team), so no readout may pool them."""
    kinds = {c.mirrored for t in cells_by_ref.values() for c in t.values()}
    if len(kinds) > 1:
        raise MeterError("these cells mix MIRRORED-PAIR and unmirrored reads (gen3_mirrored_pairs_v1) — "
                         "different populations; re-read one side under the other's regime")
    return bool(kinds and kinds.pop())


# --------------------------------------------------------------------------------------------
# Aggregation — pure numpy, no torch, no battles
# --------------------------------------------------------------------------------------------

def bootstrap_index(n_teams: int, draws: int = DEFAULT_BOOTSTRAP_DRAWS,
                    seed: int = DEFAULT_BOOTSTRAP_SEED) -> np.ndarray:
    """ONE fixed resampling index set, shared by every ref and every contrast.

    That sharing is what makes a ref-vs-ref difference PAIRED on the same team draws. Building a
    fresh index set per contrast would give each one independent noise and silently widen every
    interval — the vacuous comparison this programme retired.
    """
    return np.random.default_rng(seed).integers(0, n_teams, (draws, n_teams))


def cluster_ci(per_team: np.ndarray, idx: np.ndarray) -> Tuple[float, float, float]:
    """``(mean, lo, hi)`` in PERCENTAGE POINTS — the equal-weight cluster bootstrap over teams."""
    boot = per_team[idx].mean(axis=1)
    return (float(per_team.mean() * 100),
            float(np.percentile(boot, 2.5) * 100),
            float(np.percentile(boot, 97.5) * 100))


def verdict(delta: float, lo: float, hi: float, floor: Optional[float]) -> str:
    """WITHIN FLOOR → NOT DETECTED → SIGNIFICANT, in that order (the ruled vocabulary).

    ``WITHIN FLOOR`` comes FIRST and deliberately outranks a CI that excludes zero: a delta smaller
    than the replicate floor says the games are consistent, not that the arm differs.
    """
    if floor is not None and abs(delta) < floor:
        return "WITHIN FLOOR"
    if lo <= 0 <= hi:
        return "NOT DETECTED"
    return "SIGNIFICANT"


def replicate_floor(arms: Sequence[np.ndarray]) -> Optional[Tuple[float, List[dict]]]:
    """The MAX pairwise |Δ| over replicate arms — a floor is a magnitude, so it pools |mean|.

    Returns ``None`` for fewer than two arms: one control arm gives a re-based delta but no floor,
    and a meter that invented one would be asserting a resolution it never measured.
    """
    if len(arms) < 2:
        return None
    pairs: List[dict] = []
    for i in range(len(arms)):
        for k in range(i + 1, len(arms)):
            d = float((arms[i] - arms[k]).mean() * 100)
            pairs.append({"i": i, "j": k, "delta": d, "abs": abs(d)})
    return max(p["abs"] for p in pairs), pairs


def _rates(cells: Dict[str, Cell], team_keys: Sequence[str]) -> np.ndarray:
    return np.array([cells[k].win_rate for k in team_keys], dtype=float)


def aggregate(
    cells_by_ref: Dict[str, Dict[str, Cell]],
    team_keys: Sequence[str],
    *,
    ref_labels: Sequence[str],
    baseline_label: Optional[str],
    control_labels: Sequence[str] = (),
    floor: Optional[float] = None,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    allow_transport_mix: bool = False,
) -> dict:
    """The whole readout: levels, both delta columns, both floors, the INCONCLUSIVE rule.

    ``floor`` is the externally-ruled replicate floor for the BASELINE column (the ledger's 1.66pp
    frozen bar / 4.27pp live bar — regime-specific, and they are never pooled). The CONTROL
    column's floor is computed from the control arms themselves.
    """
    mirrored = cells_regime(cells_by_ref)
    # the TRANSPORT boundary (P2): a rust_eval read and a python_bridge read are never one population
    transport = cells_transport(cells_by_ref, allow_mix=allow_transport_mix)
    idx = bootstrap_index(len(team_keys), draws, bootstrap_seed)
    all_labels = list(dict.fromkeys(list(ref_labels) + ([baseline_label] if baseline_label else [])
                                    + list(control_labels)))

    attempted = sum(c.attempted for lab in all_labels for c in cells_by_ref[lab].values())
    timeouts = sum(c.timeouts for lab in all_labels for c in cells_by_ref[lab].values())
    timeout_fraction = timeouts / attempted if attempted else 0.0
    inconclusive = timeout_fraction > TIMEOUT_INCONCLUSIVE_FRACTION

    levels: Dict[str, dict] = {}
    rates: Dict[str, np.ndarray] = {}
    for lab in all_labels:
        cells = cells_by_ref[lab]
        r = _rates(cells, team_keys)
        rates[lab] = r
        mean, lo, hi = cluster_ci(r, idx)
        att = sum(c.attempted for c in cells.values())
        levels[lab] = {
            "per_team": {k: cells[k].to_json() for k in team_keys},
            "per_team_win_rate": [float(x) for x in r],
            "cluster_mean_pp": mean, "cluster_ci95_pp": [lo, hi],
            "wins": sum(c.wins for c in cells.values()),
            "ties": sum(c.ties for c in cells.values()),
            "finished": sum(c.finished for c in cells.values()),
            "attempted": att,
            "timeouts": sum(c.timeouts for c in cells.values()),
            "timeout_fraction": (sum(c.timeouts for c in cells.values()) / att) if att else 0.0,
        }
        if mirrored:
            from agents.training import mirrored_pairs as MP
            # the PAIR-level score over every team's pairs (the cluster CI above is over TEAMS, which
            # nests the pairs, so both intervals are taken over a unit coarser than the game)
            levels[lab]["pairs"] = MP.summary(MP.pooled(cells[k].pairs for k in team_keys))

    control_rates: Optional[np.ndarray] = None
    control_block: Optional[dict] = None
    if control_labels:
        control_rates = np.mean([rates[lab] for lab in control_labels], axis=0)
        mean, lo, hi = cluster_ci(control_rates, idx)
        fl = replicate_floor([rates[lab] for lab in control_labels])
        control_block = {
            "labels": list(control_labels),
            "pooled_cluster_mean_pp": mean, "pooled_ci95_pp": [lo, hi],
            "replicate_floor_pp": None if fl is None else fl[0],
            "pairwise": [] if fl is None else [
                {"a": control_labels[p["i"]], "b": control_labels[p["j"]],
                 "delta_pp": p["delta"]} for p in fl[1]],
            "floor_note": ("one control arm gives a re-based delta but NO floor — a floor needs at "
                           "least two replicates" if fl is None else
                           "max pairwise |Δ| over the control arms (a floor is a magnitude)"),
        }

    contrasts: List[dict] = []
    for lab in ref_labels:
        row: dict = {"ref": lab}
        if baseline_label is not None:
            d = rates[lab] - rates[baseline_label]
            m, lo, hi = cluster_ci(d, idx)
            row["vs_baseline"] = {
                "baseline": baseline_label, "delta_pp": m, "ci95_pp": [lo, hi],
                "floor_pp": floor,
                "verdict": "INCONCLUSIVE" if inconclusive else verdict(m, lo, hi, floor)}
        if control_rates is not None and control_block is not None:
            d = rates[lab] - control_rates
            m, lo, hi = cluster_ci(d, idx)
            cf = control_block["replicate_floor_pp"]
            row["vs_control"] = {
                "controls": list(control_labels), "delta_pp": m, "ci95_pp": [lo, hi],
                "floor_pp": cf,
                "verdict": "INCONCLUSIVE" if inconclusive else verdict(m, lo, hi, cf)}
        contrasts.append(row)

    return {
        "teams": list(team_keys),
        "mirrored_pairs": mirrored,
        "transport": transport,
        "levels": levels,
        "control": control_block,
        "contrasts": contrasts,
        "baseline": baseline_label,
        "bootstrap": {"draws": draws, "seed": bootstrap_seed,
                      "index_set": "ONE fixed set shared by every ref and contrast (paired)"},
        "baseline_floor_pp": floor,
        "timeouts": {"attempted": attempted, "timeouts": timeouts,
                     "fraction": timeout_fraction,
                     "inconclusive_above": TIMEOUT_INCONCLUSIVE_FRACTION,
                     "inconclusive": inconclusive},
    }


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------

def render_markdown(result: dict, *, title: str = "Untaught meter") -> str:
    """The markdown table a ledger entry can paste."""
    res = result["result"] if "result" in result else result
    meta = result.get("_meta", {})
    lines: List[str] = [f"# {title}", ""]
    if meta:
        lines += [f"**Teams** {meta.get('teams_manifest', '?')} ({len(res['teams'])} clusters) · "
                  f"**opponent** `{meta.get('opponent', {}).get('resolved_file', '?')}` · "
                  f"**{meta.get('games_per_team', '?')} games/team** · seed {meta.get('seed', '?')} · "
                  f"**transport** `{res.get('transport', meta.get('transport', TRANSPORT_LEGACY))}`", ""]
    if str(res.get("transport", "")).startswith("MIXED:"):
        lines += [f"> 🚨 **TRANSPORT MIX** ({res['transport']}) — read side by side under "
                  "`--allow-transport-mix`: the Rust eval core and the poke-env bridge are different regimes "
                  "(encoder, sampler, turn-limit rule); the levels are NOT one population.", ""]
    to = res["timeouts"]
    if to["inconclusive"]:
        lines += [f"> 🚨 **INCONCLUSIVE** — {to['timeouts']}/{to['attempted']} battles timed out "
                  f"({to['fraction']:.1%} > {to['inconclusive_above']:.0%}). A timeout is never a "
                  "semantic outcome; no verdict is reported.", ""]

    lines += ["## Levels — cluster mean over teams (equal weight)", "",
              "| ref | win rate | CI95 | wins/finished | timeouts |",
              "|---|---:|---|---:|---:|"]
    for lab, lv in res["levels"].items():
        lines.append(f"| `{lab}` | {lv['cluster_mean_pp']:.2f}pp | "
                     f"[{lv['cluster_ci95_pp'][0]:.2f}, {lv['cluster_ci95_pp'][1]:.2f}] | "
                     f"{lv['wins']}/{lv['finished']} | {lv['timeouts']} |")

    ctrl = res.get("control")
    if ctrl:
        lines += ["", "## Continuation control", "",
                  f"Arms: {', '.join('`%s`' % c for c in ctrl['labels'])} · pooled "
                  f"{ctrl['pooled_cluster_mean_pp']:.2f}pp "
                  f"[{ctrl['pooled_ci95_pp'][0]:.2f}, {ctrl['pooled_ci95_pp'][1]:.2f}]", ""]
        if ctrl["replicate_floor_pp"] is None:
            lines.append(f"Replicate floor: **none** — {ctrl['floor_note']}.")
        else:
            lines.append(f"Replicate floor: **{ctrl['replicate_floor_pp']:.2f}pp** "
                         f"({ctrl['floor_note']}); pairwise "
                         + ", ".join(f"{p['delta_pp']:+.2f}" for p in ctrl["pairwise"]) + ".")

    lines += ["", "## Deltas", ""]
    has_ctrl = any("vs_control" in c for c in res["contrasts"])
    head = "| ref | Δ vs baseline | verdict |"
    sep = "|---|---|---|"
    if has_ctrl:
        head = "| ref | Δ vs baseline | verdict | Δ vs continuation control | verdict |"
        sep = "|---|---|---|---|---|"
    lines += [head, sep]
    for c in res["contrasts"]:
        cells = [f"`{c['ref']}`"]
        b = c.get("vs_baseline")
        cells += ([f"{b['delta_pp']:+.2f} [{b['ci95_pp'][0]:+.2f}, {b['ci95_pp'][1]:+.2f}]",
                   f"**{b['verdict']}**"] if b else ["—", "—"])
        if has_ctrl:
            k = c.get("vs_control")
            cells += ([f"{k['delta_pp']:+.2f} [{k['ci95_pp'][0]:+.2f}, {k['ci95_pp'][1]:+.2f}]",
                       f"**{k['verdict']}**"] if k else ["—", "—"])
        lines.append("| " + " | ".join(cells) + " |")

    if not has_ctrl:
        lines += ["", "> ⚠️ **No continuation control.** A delta against a FROZEN baseline credits "
                  "an arm with progress the baseline would have made anyway — ledger 2026-09-06 "
                  "(cell 2) measured a plain continuation moving this meter +3.45pp [+0.46, +6.48] "
                  "on its own. Pass `--control` with continuation arms at matched depth."]

    lines += ["", "## Per-team win rate", "",
              "| team | " + " | ".join(f"`{lab}`" for lab in res["levels"]) + " |",
              "|---|" + "---:|" * len(res["levels"])]
    for i, t in enumerate(res["teams"]):
        lines.append(f"| `{t}` | " + " | ".join(
            f"{lv['per_team_win_rate'][i] * 100:.2f}" for lv in res["levels"].values()) + " |")
    return "\n".join(lines) + "\n"


def render_text(result: dict) -> str:
    """A terse console echo of the same content."""
    res = result["result"] if "result" in result else result
    out: List[str] = []
    for lab, lv in res["levels"].items():
        out.append(f"  {lab:28s} {lv['cluster_mean_pp']:7.2f}pp "
                   f"[{lv['cluster_ci95_pp'][0]:+7.2f},{lv['cluster_ci95_pp'][1]:+7.2f}]  "
                   f"{lv['wins']}/{lv['finished']}"
                   + (f"  ({lv['timeouts']} TIMEOUT)" if lv["timeouts"] else ""))
    ctrl = res.get("control")
    if ctrl:
        fl = ctrl["replicate_floor_pp"]
        out.append(f"  CONTROL pooled {ctrl['pooled_cluster_mean_pp']:.2f}pp   floor "
                   + (f"{fl:.2f}pp" if fl is not None else "none (needs >= 2 arms)"))
    for c in res["contrasts"]:
        b, k = c.get("vs_baseline"), c.get("vs_control")
        row = f"  {c['ref']:28s}"
        if b:
            row += (f"  vs baseline {b['delta_pp']:+7.2f} "
                    f"[{b['ci95_pp'][0]:+7.2f},{b['ci95_pp'][1]:+7.2f}] {b['verdict']:<13s}")
        if k:
            row += (f"  vs control {k['delta_pp']:+7.2f} "
                    f"[{k['ci95_pp'][0]:+7.2f},{k['ci95_pp'][1]:+7.2f}] {k['verdict']}")
        out.append(row)
    to = res["timeouts"]
    out.append(f"  timeouts {to['timeouts']}/{to['attempted']} ({to['fraction']:.1%})"
               + ("  ** INCONCLUSIVE **" if to["inconclusive"] else "")
               + f"   transport {res.get('transport', TRANSPORT_LEGACY)}")
    return "\n".join(out)


#: The key a shard file's provenance rides under (never a ref label).
SHARD_META = "_meta"


def merge_cells(shards: Iterable[Dict[str, Dict[str, Any]]], *,
                allow_transport_mix: bool = False) -> Dict[str, Dict[str, Cell]]:
    """Merge per-shard raw cell dicts (JSON-shaped) into one ``label -> team -> Cell`` map. A shard's
    ``_meta`` block (its transport stamp) is provenance, not a ref; a cell without its own transport takes
    the shard's. A set that mixes the mirrored regimes, or the TRANSPORTS, is refused here, not averaged."""
    out: Dict[str, Dict[str, Cell]] = {}
    for shard in shards:
        meta = shard.get(SHARD_META) if isinstance(shard.get(SHARD_META), dict) else {}
        for lab, teams in shard.items():
            if lab == SHARD_META:
                continue
            out.setdefault(lab, {})
            for key, c in teams.items():
                if key in out[lab]:
                    raise MeterError(f"shard overlap: {lab}/{key} produced twice")
                out[lab][key] = c if isinstance(c, Cell) else cell_from_json(c, meta.get("transport"))
    cells_regime(out)
    cells_transport(out, allow_mix=allow_transport_mix)
    return out
