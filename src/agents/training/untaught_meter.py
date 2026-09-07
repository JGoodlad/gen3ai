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

REPRODUCIBILITY — BOTH HALVES, OR THE LEVELS ARE A DRAW.

* **All five global-RNG seams are pinned** (``src/agents/training/CLAUDE.md`` → GLOBAL-RANDOM
  COUPLING): ``$GEN3AI_{PLAYER,TEAM,POLICY,POOL,STALLER}_SEED``, set per team from
  ``--seed`` + the team index, plus a per-BATTLE re-seed of both players' sampling generators and a
  per-battle sim seed. Seeds alone are not enough.
* **``concurrency`` is REFUSED above 1.** Interleaved battles consume the shared streams in a
  scheduling-dependent order: measured 2026-09-03, seeded at concurrency 3 two runs of the offline
  collateral-KL probe still produced 1193 vs 1141 states with arm levels up to +0.043 apart. At
  concurrency 1 they were byte-identical.
* **Sharding is over TEAMS, one single-concurrency process each.** A cell is a pure function of
  (ref, team index, battle index), so the shard split cannot move a number — verified by
  ``exploiter_competence`` (two runs of one cell in two processes → bit-identical per-battle rows)
  before it sharded 3200 battles across six workers, and gated here by
  ``untaught_meter_reproducibility_integration_test.py``.

CRN. For a given (team ``ti``, battle ``j``) the dice, the opponent's team and both players'
sampling-stream starting state are identical across every ref — only the ref's weights change — so
every ref-vs-ref difference is a PAIRED difference on the same games. The opponent's team draw is
prefix-consistent, so a ref measured at 12 games/team plays the first 12 of another ref's 200.

AGGREGATION. Team is the cluster (between-team variance dominates and more games per team does not
shrink it). **One fixed resampling index set is shared by every ref and every contrast**, so a
ref-vs-ref difference is paired on the same team draws instead of each arm carrying independent
noise. A pooled level is the equal-weight mean of the per-team rates — never the state/game-weighted
pool, which can disagree in SIGN (measured: C1−B2 was +0.0309 pooled and −0.0188 clustered).

VERDICT VOCABULARY, as ruled: ``WITHIN FLOOR`` (|Δ| below the replicate floor — the CI may still
exclude zero, which says the games are consistent, not that the arm differs), ``NOT DETECTED``
(|Δ| clears the floor but the CI spans zero), ``SIGNIFICANT``. A run whose timeouts exceed 25% of
attempted battles is ``INCONCLUSIVE`` and reports no verdict at all — a timeout is never a semantic
outcome.
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

# --------------------------------------------------------------------------------------------
# Defaults — the recipe the banked artifacts were produced under
# --------------------------------------------------------------------------------------------

#: The reuse batch's UNTAUGHT 8, IN THE ORDER THE SEEDS ARE CONSUMED. The order is part of the
#: measurement (index = team seed offset); never sort it.
DEFAULT_TEAMS_MANIFEST = repo_path(
    "designs/research_state/measurements/reuse_batch_2026-09-03/offline_collateral_kl",
    "untaught_teams.json")

#: The taught-16 slice — every 2×2 arm's own ``--distill-teacher`` expands to exactly this set.
DEFAULT_TAUGHT_MANIFEST = repo_path(
    "designs/research_state/measurements/teacher_content_2x2_2026-09-04", "taught_teams.json")

#: The fixed opponent and the ONE ``model_config.json`` every model in the series is loaded
#: against, BY NAME out of ``designs/baselines.json`` (``gen3_baselines_registry_v1``). They were
#: string literals here until 2026-09-06, which made "what is the meter's opponent?" a question
#: answerable only by reading this module — and made re-pointing it an edit with no procedure, no
#: validation and no ledger entry. **A new opponent is a RE-MEASUREMENT, not a rename**: levels are
#: not comparable across opponents, so the registry entry is what carries that fact forward.
#: ``--config auto`` resolves each model's own config instead.
DEFAULT_OPPONENT_BASELINE = "untaught_meter_opponent"
DEFAULT_CONFIG_BASELINE = "untaught_meter_config"


def default_opponent() -> str:
    """The fixed opponent's run spec, from the registry. Raises if the registry cannot be read."""
    return baselines.spec(DEFAULT_OPPONENT_BASELINE)


def default_config() -> str:
    """The shared ``model_config.json``'s run spec, from the registry."""
    return baselines.spec(DEFAULT_CONFIG_BASELINE)

DEFAULT_GAMES_PER_TEAM = 200
DEFAULT_SEED = 0
DEFAULT_BOOTSTRAP_DRAWS = 20000
DEFAULT_BOOTSTRAP_SEED = 20260906

#: A timeout is never a semantic outcome. Above this fraction of attempted battles the run reports
#: INCONCLUSIVE instead of a level.
TIMEOUT_INCONCLUSIVE_FRACTION = 0.25

#: Seed offsets, per seam. At ``--seed 0`` the sim dice, pool draw and per-battle policy seeds
#: reproduce ``exploiter_competence/compete.py`` exactly.
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

#: Set to ``"1"`` to accept an unquotable, non-reproducible run at concurrency > 1.
ALLOW_CONCURRENCY_ENV = "GEN3AI_UNTAUGHT_METER_ALLOW_CONCURRENCY"


class MeterError(RuntimeError):
    """A refusal the caller should surface verbatim (bad concurrency, unresolvable input)."""


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

    slices: List[TeamSlice] = []
    missing: List[str] = []
    for i, rel in enumerate(items):
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
        raise MeterError(f"{path}: {len(missing)} team file(s) missing:\n  "
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

    That is THE choke point every training-side consumer uses (``--distill-teacher``,
    ``--stable-opponents``, ``--exploiter``, …), so a bare run directory here means what it means
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
    """REFUSE concurrency > 1 unless the caller explicitly accepts unquotable levels."""
    if concurrency == 1:
        return
    if os.environ.get(ALLOW_CONCURRENCY_ENV) == "1":
        return
    raise MeterError(
        f"REFUSING concurrency={concurrency}: this meter is reproducible only at concurrency=1.\n"
        "  Seeds pin the dice and both players' sampling, but interleaved battles still consume\n"
        "  the shared streams in a scheduling-dependent order (measured 2026-09-03: seeded at\n"
        "  concurrency 3, two runs differed by 52 states and up to +0.043 in level).\n"
        f"  Shard over TEAMS with --workers N instead, or set {ALLOW_CONCURRENCY_ENV}=1 to accept\n"
        "  levels that cannot be quoted.")


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
                "opp_teams": self.opp_teams}


def cell_from_json(d: dict) -> Cell:
    finished = int(d.get("finished", d.get("games", 0)))
    attempted = int(d.get("attempted", finished))
    wins = int(d["wins"])
    ties = int(d.get("ties", 0))
    return Cell(wins=wins, ties=ties, losses=max(0, finished - wins - ties),
                finished=finished, attempted=attempted, opp_teams=list(d.get("opp_teams", [])))


def cells_from_rows_artifact(path: str) -> Dict[str, Cell]:
    """Ingest a committed per-team artifact (``untaught_<TAG>_end.json``-shaped) as cells.

    The shape is ``{"_meta": …, "<TEAM_KEY>": {"wins": w, "games": n, …}, "POOLED": {…}}``. The
    ``POOLED`` row is a SUMMARY, not a team cell — counting it reports 9 clusters out of 8.
    """
    with open(path) as fh:
        raw = json.load(fh)
    cells = {k: cell_from_json(v) for k, v in raw.items()
             if isinstance(v, dict) and "wins" in v and k not in ("POOLED", "_meta")}
    if not cells:
        raise MeterError(f"{path}: no per-team rows (expected objects carrying 'wins'/'games')")
    return cells


# --------------------------------------------------------------------------------------------
# Playing — the battle harness (imports torch/poke-env lazily so the maths half stays cheap)
# --------------------------------------------------------------------------------------------

def _strip_debugger(model):
    """Drop the ObservationDebugger a ``--log-level periodic`` checkpoint carries.

    It ``print()``s a full DEEP TRACE board on EVERY forward. Verified output-neutral 2026-09-02
    (actions and values bit-identical with and without), which matters because these levels are
    compared against baselines measured before the strip existed.
    """
    obj = getattr(model, "policy", model)
    for mod in (obj.modules() if hasattr(obj, "modules") else []):
        if getattr(mod, "_debugger", None) is not None:
            mod._debugger = None
    return model


def _teambuilders():
    """Build the two teambuilder subclasses lazily.

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
    ``gen3_policy_sample_rng_v1``) so battle (ti, j) starts identically for every ref."""
    player._policy_seed = int(seed)
    player._policy_gens = {}


def play_cells(
    refs: Sequence[ResolvedRef],
    teams: Sequence[TeamSlice],
    opponent: ResolvedRef,
    *,
    games_per_team: int,
    seed: int = DEFAULT_SEED,
    impl: str = "rust",
    concurrency: int = 1,
    progress=None,
) -> Dict[str, Dict[str, Cell]]:
    """Play every (ref × team) cell and return the raw counts. ``concurrency`` must be 1."""
    check_concurrency(concurrency)
    import asyncio

    import torch as th
    from poke_env.ps_client import AccountConfiguration
    from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

    from agents.inference.player import RLPlayer
    from agents.model.snapshot import current_model_version, load_foreign_opponent
    from agents.observation.state_encoder import load_mappings
    from utils.bridge.local_battle_runner import run_local_battles
    from utils.team_loader import TeamLoader

    th.set_num_threads(1)
    PinnedTeam, PairedPool = _teambuilders()

    maps = load_mappings()
    cv = current_model_version(maps)
    opp_model = _strip_debugger(load_foreign_opponent(
        opponent.zip_path, current_version=cv, device="cpu",
        config_path=opponent.config_path)[0])

    pool = PairedPool(TeamLoader().get_all_teams())
    n_pool = len(pool.packed_teams)
    seqs = {t.index: pool_sequence(seed, t.index, games_per_team, n_pool) for t in teams}

    out: Dict[str, Dict[str, Cell]] = {}
    for ref in refs:
        model = _strip_debugger(load_foreign_opponent(
            ref.zip_path, current_version=cv, device="cpu", config_path=ref.config_path)[0])
        out[ref.label] = {}
        for team in teams:
            ti = team.index
            pilot = RLPlayer(model=model, team=PinnedTeam(team.path), battle_format="gen3ou",
                             server_configuration=LocalhostServerConfiguration, mappings=maps,
                             account_configuration=AccountConfiguration(f"UM{ti}a", "pw"),
                             stochastic=True, start_listening=False)
            opp = RLPlayer(model=opp_model, team=pool, battle_format="gen3ou",
                           server_configuration=LocalhostServerConfiguration, mappings=maps,
                           account_configuration=AccountConfiguration(f"UM{ti}b", "pw"),
                           stochastic=True, start_listening=False)
            pool.set_sequence(seqs[ti])
            cell = Cell()
            for j in range(games_per_team):
                ps, os_ = policy_seeds(seed, ti, j)
                _reseed_player(pilot, ps)
                _reseed_player(opp, os_)
                pool.at(j)
                pilot.reset_battles()
                opp.reset_battles()
                asyncio.run(run_local_battles(pilot, opp, 1, concurrency=1, impl=impl,
                                              seed=sim_seed(seed, ti, j)))
                cell.attempted += 1
                cell.opp_teams.append(seqs[ti][j])
                if pilot.n_finished_battles != 1:
                    continue          # the TIMEOUT bucket — never scored as a loss
                cell.finished += 1
                cell.wins += int(pilot.n_won_battles)
                cell.ties += int(pilot.n_tied_battles)
                cell.losses += 1 - int(pilot.n_won_battles) - int(pilot.n_tied_battles)
            out[ref.label][team.key] = cell
            if progress is not None:
                progress(ref.label, team.key, cell)
    return out


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
) -> dict:
    """The whole readout: levels, both delta columns, both floors, the INCONCLUSIVE rule.

    ``floor`` is the externally-ruled replicate floor for the BASELINE column (the ledger's 1.66pp
    frozen bar / 4.27pp live bar — regime-specific, and they are never pooled). The CONTROL
    column's floor is computed from the control arms themselves.
    """
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
                  f"concurrency {meta.get('concurrency', 1)}", ""]
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
               + ("  ** INCONCLUSIVE **" if to["inconclusive"] else ""))
    return "\n".join(out)


def merge_cells(shards: Iterable[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Cell]]:
    """Merge per-shard raw cell dicts (JSON-shaped) into one ``label -> team -> Cell`` map."""
    out: Dict[str, Dict[str, Cell]] = {}
    for shard in shards:
        for lab, teams in shard.items():
            out.setdefault(lab, {})
            for key, c in teams.items():
                if key in out[lab]:
                    raise MeterError(f"shard overlap: {lab}/{key} produced twice")
                out[lab][key] = c if isinstance(c, Cell) else cell_from_json(c)
    return out
