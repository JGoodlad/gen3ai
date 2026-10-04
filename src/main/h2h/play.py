"""PLAY ONE EDGE of the head-to-head meter: checkpoint A against checkpoint B on the Rust eval core, MIRRORED.

``main.h2h play`` — the offline, symmetric checkpoint-vs-checkpoint read (X5 design §7 P0 / U0). What is
reused, and what is new:

* REUSED, unchanged: the Rust eval core and its executor (``rust_eval.executor.RustEvalCore.run_cycle``,
  the code the in-loop eval and the SPRT promotion play on), the per-game seed rule and the mirrored-pair
  rule (``rust_eval.seeds.pair_game``, ``gen3_mirrored_pairs_v1``), the eval team builders
  (``rust_eval.build.eval_builders``), the strict checkpoint loader.
* NEW: the host that declares ONE T2 service holding TWO slots (the player's eval slot, one SENTINEL slot
  for the opponent) and ONE eval core over them, plays the plan in BATCHES, scores each batch from the
  executor's game log into a §0b COUNT row (``agents.training.eval_ledger``, ``gen3_eval_count_row_v2``) and
  appends it durably, under a CLAIM, for a REQUEST (§0b.4).

THE LEDGER (eval unit U1, 2026-10-03 — STORAGE ONLY: the games are unchanged, proved by an outcome-digest check in
``designs/research_state/measurements/eval_ledger_u1_2026-10-03/``). Rows go to a ledger ROOT: by default the run
archive's ``<archive>/_ledger/`` (``utils.paths.run_archive_dir``), else the root the caller names (outside
``models/``). Every batch belongs to a REQUEST: by default one derived from (the two players, the regime, the
schedule), so a re-run with the same arguments resumes it; ``--request`` / ``--family`` / ``--request-kind`` put the
edge into a caller's request (an X5 look: ``--purpose ab --family <the A/B>``). The protocol is
:data:`PROTOCOL` — a v1 h2h row upgraded on read carries the same one.

THE REGIME (stated on every row; the readers refuse to mix it). Both sides play GREEDY (argmax of the served
log-probs — the eval regime, ``eval_sentinel_greedy``); MIRRORED pairs; the game ends at the declared turn
limit (a timeout is a DRAW, as in training); BOTH sides draw their teams from the PLAYER's eval team builder
(the default pool with its 10 % sample-team bias — what ``eval_builders(None, [])`` builds and what a
greedy sentinel draws from in the in-loop eval). A player whose run pinned its trainee team is REFUSED: the
meter's team distribution is the trainee's, and a pinned run's is not this one.

THE MIRROR, and what it does NOT cancel. A pair is ONE team pairing ``(T1, T2)`` on ONE battle seed, played
twice: game ``2k`` the player pilots ``T1`` against the opponent on ``T2``; game ``2k+1`` the teams are
handed over. The PLAYER KEEPS SEAT p1 in both games (``rust_eval.seeds.pair_game``: "only the TEAMS change
hands"), so team-draw luck cancels inside the pair but a SEAT effect does not: it adds to every pair. The
instrument measures it directly — a checkpoint against ITSELF reads ``0.5 + seat effect`` — and a reader
comparing two players on one scale must carry it (``main.h2h.runfloor``).

REPRODUCIBLE BY CONSTRUCTION. Every game's teams, battle seed and (greedy: unused) draws are a pure function
of ``(schedule seed, schedule key, batch, game index)`` (``cycle_seed`` below + ``rust_eval.seeds``), never
of the env, the env count, the thread count or the batch schedule. The default schedule key is the digest of
the two players' content hashes, ORDER-INDEPENDENT: ``A vs B`` and ``B vs A`` draw the same team pairs, so
the two directions of an edge are a seat-effect measurement. What a re-run can still change is the FORWARD:
a greedy decision within a rounding error of a tie can flip with the device, the backend or the batch
shape. Every row therefore counts the decisions inside ``NEAR_TIE`` (``compute.near_tie_decisions`` /
``near_tie_games``) — the games whose outcome rests on a rounding error — rather than leaving them to
chance (standing rule 8).

DURABLE AND RESUMABLE. A batch is one row, appended (fsynced) when it ends, under its claim; a re-run with the
same request skips the batches already recorded and plays the rest. A different batch size over an existing
request is REFUSED (the request's spec pins it): the same batch index would replay the same games under another
length. A batch is also never recorded twice across requests: the ledger refuses a second row on one SEED BLOCK.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from agents.training import eval_ledger as L
from agents.training import mirrored_pairs as MP
from agents.training.rust_eval import seeds as SD

#: The opponent's key in the eval plan (the executor's per-game seed key is ``(cycle seed, key, game)``).
ITEM_KEY = "h2h"

CYCLE_SEED_TAG = "gen3_h2h_cycle_seed_v1"
SCHEDULE_TAG = "gen3_h2h_schedule_v1"

#: A decision whose top-2 log-prob margin is below this is "within a rounding error of a tie" — the
#: executor's own constant (``rust_eval.executor.NEAR_TIE``): twice the CPU eager-vs-eager |Δ log-prob| bar
#: of the Lane H gate (``rust_eval.parity.BAR_CPU``).
NEAR_TIE = 2e-5
#: The same census at the GPU bar (``rust_eval.parity.BAR_GPU`` 1e-3, T2 ``graph`` on CUDA vs the Python path,
#: x 2): a decision this close can flip with the device or the batch shape on the card.
NEAR_TIE_WIDE = 2e-3

DEFAULT_BATCH_PAIRS = 500
DEFAULT_PURPOSE = "audit"
#: The game protocol this tool plays (``eval_ledger.schema.PROTOCOLS``). Storage-only changes keep it; a change to
#: what a game measures (seats, teams, turn limit, the core's semantics) bumps it — and is REFUSED while a family
#: pinned to it has no decision row (design_evaluation.md §0c rule 6).
PROTOCOL = "gen3_eval_protocol_v1_h2h"
PRODUCER = "h2h"
#: The margin that defines a near-tie decision for the OUTCOME DIGEST: the GPU bar (``NEAR_TIE_WIDE``), so a game
#: that could flip between CPU and GPU is listed by index instead of hashed and a cross-device replay still matches.
DIGEST_MARGIN = 2e-3
#: The first claim's expected batch wall, in games per second (CPU eager on the production model, P0: 4.77).
EXPECTED_GAMES_PER_S = 4.0

TEAM_SOURCE = "eval_builders default_biased(bias_prob=0.1): both sides draw from the player's eval team builder"

#: The resume read: THIS request's rows (any purpose — a request has one), at this tool's protocol.
H2H_RESUME = L.ReaderDecl(
    name="main.h2h.resume", purposes=L.ALL_PURPOSES,
    regime=L.RegimeFilter(protocol=PROTOCOL, play="greedy", opponent_play="greedy", mirrored=True),
    requests="own", selection="include", flags_ok=frozenset(), inference="conditional")


def eval_team_set() -> str:
    """The TEAM SET identity of the player's eval team builder (``eval_builders(None, [])``'s trainee builder: the
    default pool with its 10 % sample-team bias): the digest of the ordered team lists and the builder parameters."""
    from utils.team_loader import TeamLoader

    loader = TeamLoader()
    return L.team_set_id({"builder": "Gen3Teambuilder", "kind": "default_biased", "bias_prob": 0.1,
                          "teams": list(loader.get_all_teams()), "bias_teams": list(loader.get_sample_teams())})


class H2HError(RuntimeError):
    """A head-to-head read that cannot be run honestly (incompatible players, a pinned-team run, a refused
    resume, a game log that does not score)."""


# ---------------------------------------------------------------------------------------------- players
@dataclass(frozen=True)
class PlayerRef:
    """A checkpoint, resolved: the file, its config, how it was chosen and its content hash."""

    spec: str
    zip_path: str
    config_path: str
    run_dir: str
    run_base: str
    rung: str
    num_timesteps: Optional[int]
    sha256: str

    @property
    def id(self) -> str:
        return f"{self.run_base}@{self.num_timesteps}" if self.num_timesteps is not None else \
            f"{self.run_base}:{Path(self.zip_path).stem}"

    def block(self) -> Dict[str, Any]:
        return {"id": self.id, "sha256": self.sha256, "path": self.zip_path, "run": self.run_base,
                "step": self.num_timesteps, "rung": self.rung}


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_player(spec: str) -> PlayerRef:
    """``spec`` = a ``.zip``, a run dir (its LAST snapshot, ``resolve_model_ref``) or ``<run>@<step>``."""
    from agents.training.fixed_opponent_pool import resolve_model_ref

    ref = resolve_model_ref(spec)
    return PlayerRef(spec=spec, zip_path=ref.zip_path, config_path=ref.config_path, run_dir=ref.run_dir,
                     run_base=ref.run_base, rung=ref.rung, num_timesteps=ref.num_timesteps,
                     sha256=file_sha256(ref.zip_path))


def _read_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def check_team_source(ref: PlayerRef) -> str:
    """REFUSE a player whose run pinned its eval trainee team; say what was checked. The run's
    ``metadata.json`` ``matchup_history[-1].spec.eval_trainee_teams`` is the record."""
    md = Path(ref.run_dir) / "metadata.json"
    if not md.exists():
        md = Path(ref.zip_path).resolve().parent.parent / "metadata.json"
    if not md.exists():
        return "unverified (no metadata.json beside the checkpoint)"
    hist = (_read_json(str(md)).get("matchup_history") or [])
    if not hist:
        return "unverified (metadata.json carries no matchup_history)"
    spec = (hist[-1].get("spec") or {}).get("eval_trainee_teams") or {}
    if spec.get("kind") != "default_biased" or float(spec.get("bias_prob", 0.1)) != 0.1 or spec.get("pin_file"):
        raise H2HError(f"{ref.id}: its eval trainee teams are {spec} — the head-to-head draws from the DEFAULT "
                       "biased pool builder (the trainee's team distribution); a pinned-team player is out of scope")
    return f"default_biased (matchup {hist[-1].get('hash')})"


def check_core_flags(ref: PlayerRef) -> None:
    """REFUSE a player whose run recorded the no-progress clock's deleted variants ON. ``progress_decision_tense`` /
    ``progress_switch_freeze`` are gone from the env core (P11d, 2026-10-03; the recorded fields stay READABLE and a
    recorded True is refused on a resume), so a checkpoint trained under either was trained on a core this tree no
    longer has — its games cannot be played here as they were trained."""
    cfg = _read_json(ref.config_path)
    on = [k for k in ("progress_decision_tense", "progress_switch_freeze") if cfg.get(k)]
    if on:
        raise H2HError(f"{ref.id}: its model_config.json records {on} ON — those no-progress-clock variants were deleted "
                       "with the env core's spec keys (P11d); this tree cannot play such a checkpoint as it was trained")


# ---------------------------------------------------------------------------------------------- seeds
def schedule_key_of(a: PlayerRef, b: PlayerRef) -> str:
    """The default schedule key: the digest of the TWO content hashes, order-independent."""
    lo, hi = sorted([a.sha256, b.sha256])
    return "h2h:" + hashlib.blake2b(f"{SCHEDULE_TAG}:{lo}:{hi}".encode(), digest_size=8).hexdigest()


def cycle_seed(schedule_seed: int, schedule_key: str, batch: int) -> int:
    """One batch's per-game seed base — a hash on its OWN namespace (disjoint from the in-loop cycles'
    ``rust_eval.launch.cycle_seed`` and the SPRT's ``sprt_promotion.sprt_seed``)."""
    d = hashlib.blake2b(f"{CYCLE_SEED_TAG}:{int(schedule_seed)}:{schedule_key}:{int(batch)}".encode(),
                        digest_size=8).digest()
    return int.from_bytes(d, "big") & ((1 << 62) - 1)


def batch_plan(total_pairs: int, batch_pairs: int) -> List[int]:
    """Pairs per batch: ``total_pairs`` split into ``batch_pairs``-sized batches, the last the remainder."""
    if total_pairs < 1 or batch_pairs < 1:
        raise H2HError("pairs and batch-pairs must be positive")
    full, rem = divmod(int(total_pairs), int(batch_pairs))
    return [int(batch_pairs)] * full + ([rem] if rem else [])


# ---------------------------------------------------------------------------------------------- scoring
@dataclass
class BatchScore:
    w: int
    l: int
    d: int
    pair_counts: List[int]
    n_pairs: int
    teams: Dict[str, Dict[str, List[int]]]
    near_tie_decisions: int
    near_tie_games: int
    mirror_checked: int
    near_tie_decisions_wide: int = 0
    near_tie_games_wide: int = 0
    #: per pair, in order: ``(player half-points game 1, game 2, near-tie decisions in the pair, wide)``
    pair_detail: Optional[List[Tuple[int, int, int, int]]] = None
    #: the §0b.2 audit pair: sha256 over the ``(game, W/L/D, turns)`` vector of the games with NO decision inside
    #: :data:`DIGEST_MARGIN`, and the indices of the games that had one
    outcome_digest: Optional[str] = None
    near_tie_idx: Optional[List[int]] = None

    def clean_pairs(self, wide: bool = False) -> int:
        """Pairs with NO decision inside the near-tie margin (narrow, or the GPU-bar ``wide`` one)."""
        i = 3 if wide else 2
        return sum(1 for t in (self.pair_detail or []) if t[i] == 0)

    def clean_pairs_off_center(self, wide: bool = False) -> int:
        """Clean pairs whose score is NOT exactly 1/2 (2 of 4 half-points). For a player against ITSELF the
        mirror cancels exactly, so this is 0 whenever no decision rests on a rounding error."""
        i = 3 if wide else 2
        return sum(1 for t in (self.pair_detail or []) if t[i] == 0 and t[0] + t[1] != 2)


def score_games(games: Sequence[Mapping[str, Any]], team_packed: Sequence[str], n_pairs: int) -> BatchScore:
    """Score one batch's games (the executor's log rows) into counts. EVERY check is an assertion about the
    mirror, never a branch: the batch must hold games ``0 .. 2 n_pairs - 1`` exactly once each; game ``2k``
    must be unswapped and ``2k+1`` swapped; the second game's teams must be the first's HANDED OVER
    (the player's team and the opponent's exchanged). A pair whose games do not satisfy them is not scored —
    it is an :class:`H2HError`, because a mirrored count over broken pairs is not a mirrored count.

    ``games`` rows carry ``game``, ``result`` (``WIN`` / ``LOSS`` / ``DRAW`` from the PLAYER's side),
    ``swapped``, ``teams`` (``[player's team index, opponent's team index]``) and ``near_ties`` (the decisions
    of both sides inside :data:`NEAR_TIE`)."""
    from agents.training.trace_result import DRAW, LOSS, WIN

    by_g: Dict[int, Mapping[str, Any]] = {}
    for g in games:
        k = int(g["game"])
        if k in by_g:
            raise H2HError(f"game {k} appears twice in the batch")
        by_g[k] = g
    want = set(range(2 * n_pairs))
    if set(by_g) != want:
        raise H2HError(f"the batch holds games {sorted(set(by_g) ^ want)[:8]}... off the plan 0..{2 * n_pairs - 1}")
    w = l = d = 0
    teams: Dict[str, Dict[str, List[int]]] = {}
    pts: List[Optional[int]] = []

    def bump(tid: str, side: str, won: bool) -> None:
        t = teams.setdefault(tid, {"p": [0, 0], "o": [0, 0]})
        t[side][0] += 1
        t[side][1] += int(won)

    for k in range(n_pairs):
        a, b = by_g[2 * k], by_g[2 * k + 1]
        if bool(a["swapped"]) or not bool(b["swapped"]):
            raise H2HError(f"pair {k}: games are not (unswapped, swapped): {a['swapped']}, {b['swapped']}")
        ta, tb = tuple(int(x) for x in a["teams"]), tuple(int(x) for x in b["teams"])
        if tb != (ta[1], ta[0]):
            raise H2HError(f"pair {k}: the second game's teams {tb} are not the first's {ta} handed over")
        for g in (a, b):
            r = g["result"]
            if r not in (WIN, LOSS, DRAW):
                raise H2HError(f"pair {k}: unknown result {r!r}")
            w, l, d = w + (r == WIN), l + (r == LOSS), d + (r == DRAW)
            tp, to = L.team_id(team_packed[int(g["teams"][0])]), L.team_id(team_packed[int(g["teams"][1])])
            bump(tp, "p", r == WIN)
            bump(to, "o", r == LOSS)
            pts.append(MP.result_points(r))
    nt = {int(g["game"]): int(g.get("near_ties", 0)) for g in games}
    ntw = {int(g["game"]): int(g.get("near_ties_wide", 0)) for g in games}
    detail = [(int(pts[2 * k]), int(pts[2 * k + 1]), nt[2 * k] + nt[2 * k + 1], ntw[2 * k] + ntw[2 * k + 1])  # type: ignore[arg-type]
              for k in range(n_pairs)]
    near = sorted(k for k, x in ntw.items() if x)
    digest = L.outcome_digest(outcome_vector(games), near)
    return BatchScore(w=w, l=l, d=d, pair_counts=MP.pair_counts(pts), n_pairs=n_pairs, teams=teams,
                      near_tie_decisions=sum(nt.values()), near_tie_games=sum(1 for x in nt.values() if x),
                      mirror_checked=n_pairs, near_tie_decisions_wide=sum(ntw.values()),
                      near_tie_games_wide=sum(1 for x in ntw.values() if x), pair_detail=detail,
                      outcome_digest=digest, near_tie_idx=near)


def outcome_vector(games: Sequence[Mapping[str, Any]]) -> List[Tuple[int, str, Optional[int]]]:
    """``(game, W/L/D, end turn)`` per game, from the PLAYER's side — the outcome digest's input."""
    from agents.training.trace_result import DRAW, LOSS, WIN

    letter = {WIN: "W", LOSS: "L", DRAW: "D"}
    return [(int(g["game"]), letter[g["result"]], None if g.get("end_turn") is None else int(g["end_turn"]))
            for g in games]


class _GameSink(list):
    """The executor's ``game_log`` — trimmed AT THE SOURCE: a game's input log (``script``) and per-decision
    arrays are tens of KB each, and a batch holds ~1,000 games. Keeps what scoring needs and the count of
    each side's decisions inside :data:`NEAR_TIE`."""

    KEEP = ("game", "result", "swapped", "teams", "winner", "end_turn", "forfeit")

    def append(self, row: Mapping[str, Any]) -> None:  # type: ignore[override]
        slim = {k: row[k] for k in self.KEEP if k in row}
        margins = [float(m) for m in row.get("margins", ())] + [float(o[3]) for o in row.get("opp", ())]
        slim["near_ties"] = sum(1 for m in margins if m < NEAR_TIE)
        slim["near_ties_wide"] = sum(1 for m in margins if m < NEAR_TIE_WIDE)
        super().append(slim)


# ---------------------------------------------------------------------------------------------- the engine
@dataclass(frozen=True)
class Compute:
    """Where the forward and the core run. Never part of the regime identity — recorded beside it."""

    device: str = "cpu"
    backend: str = ""             # "" = graph on CUDA, eager on CPU
    n_envs: int = 64
    threads: int = 4              # the Rust core's worker threads
    torch_threads: int = 4        # intra-op threads of a CPU forward
    front: str = "proc"
    profile: str = "release"

    @property
    def resolved_backend(self) -> str:
        return self.backend or ("graph" if self.device.startswith("cuda") else "eager")

    def block(self) -> Dict[str, Any]:
        return {"device": self.device, "backend": self.resolved_backend, "n_envs": self.n_envs,
                "threads": self.threads, "torch_threads": self.torch_threads, "front": self.front,
                "profile": self.profile}


def regime_for(turn_limit: int, team_set: Optional[str] = None) -> Dict[str, Any]:
    """The v2 regime block. ``team_set`` defaults to :func:`eval_team_set` (it reads the team pool)."""
    return L.with_regime_id({
        "play": "greedy", "opponent_play": "greedy", "mirrored": True, "mirror_rule": MP.SCHEMA,
        "eval_core": "rust", "turn_limit": int(turn_limit), "seed_rule": SD.SCHEMA, "team_source": TEAM_SOURCE,
        "protocol": PROTOCOL, "seat_rule": "fixed_p1", "player_temp": None, "opponent_temp": None,
        "team_set": team_set or eval_team_set()})


class H2HEngine:
    """ONE T2 service (two slots) and ONE eval core for one (player, opponent) edge. Build once; play any
    number of batches (each a ``RustEvalCore.run_cycle``); close. The declared lifecycle holds: every
    resource is acquired here, and a cycle only LOADS weights into the two slots."""

    def __init__(self, player: PlayerRef, opponent: PlayerRef, compute: Compute,
                 emit: Callable[[str], None] = lambda m: print(m, file=sys.stderr, flush=True)):
        import torch

        from agents.inference.service import InferenceService, ServiceSpec, SlotGroupSpec
        from agents.model.snapshot import (ModelVersion, arch_toggles_from_model, current_model_version,
                                           historical_load_kwargs, load_checkpoint_strict)
        from agents.observation.state_encoder import load_mappings
        from agents.training.reward_config import RewardConfig
        from agents.training.rust_eval.build import EvalDecl, build_eval_core, eval_builders, eval_extra_slots
        from agents.training.rust_rollout.build import RustEnvDecl, _arch_key
        from utils.rust_env import episode as EP

        self.player, self.opponent, self.compute, self.emit = player, opponent, compute, emit
        check_core_flags(player)
        check_core_flags(opponent)
        # THIS checkout's env core, built (incrementally) before anything loads — the trainer's own startup step
        # (`rust_env_setup`, F-LG-6); the loaders' stamp check refuses a build that is not this tree's, so a stale
        # `target/` (a rebase moved the wire) is "missing / stale" turned into "current", never another checkout's
        from utils.rust_env.build import ensure_built

        ensure_built(compute.profile, emit=emit)
        self.team_check = {player.id: check_team_source(player), opponent.id: check_team_source(opponent)}
        self.turn_limit = EP.stall_threshold()
        self.regime = regime_for(self.turn_limit)
        self._threads0 = torch.get_num_threads()
        try:
            if compute.torch_threads:
                torch.set_num_threads(int(compute.torch_threads))
            t0 = time.perf_counter()
            # the strict loader, for BOTH (`gen3_strict_checkpoint_load_v1`); the opponent stays on the CPU (T2
            # copies it into its slot — a source on the card would be a second copy of it)
            # `historical_load_kwargs` strips the policy / extractor kwargs DELETED since the checkpoint was written
            # (PopArt, the value-dist head) and REFUSES an ON one — a run trained at an older pin still loads at HEAD
            self.historical = {player.id: sorted((historical_load_kwargs(player.zip_path).get("custom_objects") or {})),
                               opponent.id: sorted((historical_load_kwargs(opponent.zip_path).get("custom_objects") or {}))}
            self.pm = load_checkpoint_strict(player.zip_path, device=compute.device,
                                             **historical_load_kwargs(player.zip_path))
            self.om = load_checkpoint_strict(opponent.zip_path, device="cpu", **historical_load_kwargs(opponent.zip_path))
            self.pm.policy.eval()
            self.om.policy.eval()
            toggles = arch_toggles_from_model(self.pm)
            if arch_toggles_from_model(self.om) != toggles:
                raise H2HError("the two players' architecture toggles differ — one T2 slot group serves one architecture")
            version = current_model_version(load_mappings(), **toggles)
            for ref in (player, opponent):
                version.check_opponent_snapshot_compatible(ModelVersion.from_json_file(ref.config_path))
            if _arch_key(self.pm.policy) != _arch_key(self.om.policy):
                raise H2HError("the two players' weight signatures differ — they cannot share one T2 slot group")
            n = int(compute.n_envs)
            decl = EvalDecl(n_envs=n, n_sentinels=1)
            extra = eval_extra_slots(decl, self.pm.policy, {})
            n_slots = len(extra)
            cdecl = RustEnvDecl(n_envs=n, threads=int(compute.threads), front=compute.front, profile=compute.profile,
                                device=compute.device,
                                backend=compute.resolved_backend, turn_limit=self.turn_limit)
            buckets = cdecl.resolved_buckets
            lanes = min(n_slots, 8) if compute.device.startswith("cuda") else 1
            self.svc = InferenceService(ServiceSpec(
                groups=(SlotGroupSpec("eval", n_slots, self.pm.policy),), device=compute.device,
                backend=compute.resolved_backend, buckets=buckets, lanes=lanes,
                max_rows_per_flush=max(1024, n_slots * max(buckets), 4 * n))).startup()
            terminal = EP.terminal_from_reward_config(RewardConfig.from_dict(_read_json(player.config_path)))
            tb, flat, fixed = eval_builders(None, [])
            self.ev = build_eval_core(decl, collector_decl=cdecl, svc=self.svc, extra_ids=list(range(n_slots)),
                                      trainee_builder=tb, opp_builder=flat, fixed_builders=fixed,
                                      turn_limit=self.turn_limit, terminal=terminal, emit=lambda _m: None)
            self.team_packed: List[str] = list(self.ev.team_table.teams)
            self.startup_s = time.perf_counter() - t0
            self.torch_version = torch.__version__
            self.core_stamp = str(getattr(self.ev.core, "stamp", "") or "")
            emit(f"[h2h] engine up in {self.startup_s:.1f}s: {player.id} vs {opponent.id}; {n} envs, "
                 f"T2 {compute.resolved_backend} on {compute.device}; regime {self.regime['regime_id']}")
        except BaseException:
            torch.set_num_threads(self._threads0)
            raise

    def play_batch(self, pairs: int, seed: int, step: int = 0, sink: Optional[List[Dict[str, Any]]] = None
                   ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Play ``pairs`` mirrored pairs at per-game seed base ``seed``; returns the trimmed game log and the
        cycle's stats (``CycleStats.as_dict``). ``sink`` (a diagnostic's own list) receives the executor's
        FULL game rows instead — every action, margin and the input log — and is what is returned."""
        from agents.training.eval_player import ForensicQuota
        from agents.training.eval_sharding import SENTINEL, EvalItem, ShardedEvalPool

        shard_games = max(2, 2 * math.ceil(pairs / int(self.compute.n_envs)))
        item = EvalItem(ITEM_KEY, SENTINEL, 2 * pairs, path=self.opponent.zip_path, step=0)
        pool = ShardedEvalPool([item], shard_games, step=step, mirrored=True)
        run_dir = tempfile.mkdtemp(prefix="h2h_cycle_")
        try:
            pool.write_plan(run_dir)
            sink = _GameSink() if sink is None else sink
            st = self.ev.run_cycle(pool, run_dir, step=step, trainee_policy=self.pm.policy,
                                   sentinel_policies={ITEM_KEY: self.om.policy}, forensic_root=None,
                                   quota=ForensicQuota(0, 0, 0), gamma=float(self.pm.gamma), sentinel_greedy=True,
                                   self_play_temp=1.0, cycle_seed=int(seed), game_log=sink)
            merged, missing = pool.collect(run_dir)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)
        if missing:
            raise H2HError(f"the eval core published no result for {missing}")
        pc = ((merged.get("pairs") or {}).get(ITEM_KEY))
        out = st.as_dict()
        out["executor_pair_counts"] = list(pc) if pc is not None else None
        return list(sink), out

    def close(self) -> None:
        """Close the eval core (the service has no teardown of its own: it ends with the process) and hand
        back the torch thread count this engine changed."""
        import torch

        try:
            self.ev.close()
        finally:
            torch.set_num_threads(self._threads0)


# ---------------------------------------------------------------------------------------------- the edge
def default_request_id(a: PlayerRef, b: PlayerRef, regime_id: str, key: str, sched_seed: int) -> str:
    """The request a plain re-run resumes: a digest of (the two players, the regime, the schedule key and seed) —
    NOT of the pair count (a longer plan extends the same request) nor the batch size (pinned in its spec)."""
    d = hashlib.blake2b(f"{a.sha256}:{b.sha256}:{regime_id}:{key}:{int(sched_seed)}".encode(), digest_size=8)
    return f"h2h:{d.hexdigest()}"


def completed_batches(rows: Sequence[Mapping[str, Any]], a: PlayerRef, b: PlayerRef) -> Dict[int, int]:
    """``{batch index: pairs}`` of this edge among one request's rows."""
    return {int(r["request"]["batch"]): int(r["pairs"]["n_pairs"]) for r in rows
            if r["player"]["sha256"] == a.sha256 and r["opponent"]["sha256"] == b.sha256}


def _loadavg() -> List[float]:
    return [round(x, 2) for x in os.getloadavg()]


def build_row(*, writer: L.LedgerWriter, run_label: str, commit: str, a: PlayerRef, b: PlayerRef,
              eng: "H2HEngine", purpose: str, key: str, sched_seed: int, batch: int, cseed: int,
              score: BatchScore, t_start: str, t_end: str, wall_s: float, load: Dict[str, Any],
              executor: Mapping[str, Any], request: Mapping[str, Any]) -> Dict[str, Any]:
    n_games = 2 * score.n_pairs
    compute = {**eng.compute.block(), "torch": eng.torch_version, "core_stamp": eng.core_stamp,
               "wall_s": round(wall_s, 3), "games_per_s": round(n_games / wall_s, 3) if wall_s > 0 else None,
               "near_tie_decisions": score.near_tie_decisions, "near_tie_game_count": score.near_tie_games,
               "near_tie_decisions_wide": score.near_tie_decisions_wide,
               "near_tie_game_count_wide": score.near_tie_games_wide, "near_tie_margin": NEAR_TIE,
               "near_tie_margin_wide": NEAR_TIE_WIDE, "clean_pairs": score.clean_pairs(),
               "clean_pairs_off_center": score.clean_pairs_off_center(),
               "clean_pairs_wide": score.clean_pairs(True),
               "clean_pairs_off_center_wide": score.clean_pairs_off_center(True), "load_avg_start": load["start"],
               "load_avg_end": load["end"], "contention": load["contention"], "team_check": eng.team_check,
               "sanitized_load": eng.historical,
               "trainee_decisions": executor.get("trainee_decisions"),
               "p2_policy_decisions": executor.get("p2_policy_decisions"),
               "outcome_digest": score.outcome_digest, "near_tie_games": list(score.near_tie_idx or []),
               "digest_margin": DIGEST_MARGIN}
    return {
        "schema": L.SCHEMA, "row_id": writer.next_row_id(), "supersedes": None, "ts": L.utc_now(),
        "t_start": t_start, "t_end": t_end, "run": run_label, "commit": commit,
        "player": {**a.block(), "kind": "checkpoint"}, "opponent": {**b.block(), "kind": "checkpoint"},
        "regime": dict(eng.regime), "compute": compute, "purpose": purpose,
        "request": {"id": request["request_id"], "kind": request["kind"], "family": request["family"],
                    "opened": request["ts"], "batch": int(batch)},
        "counts": {"w": score.w, "l": score.l, "d": score.d, "aborted": 0},
        "pairs": {"counts": score.pair_counts, "n_pairs": score.n_pairs, "voided": 0},
        "teams": score.teams,
        "seed": {"rule": SD.SCHEMA, "schedule_seed": int(sched_seed), "schedule_key": key, "batch": int(batch),
                 "cycle_seed": int(cseed), "item_key": ITEM_KEY, "game_lo": 0, "game_hi": n_games - 1},
        "flags": [], "provenance": None}


def play_edge(out_dir: Optional[str], player: PlayerRef, opponent: PlayerRef, *, pairs: int,
              batch_pairs: int = DEFAULT_BATCH_PAIRS, schedule_seed: int = 0, schedule_key: Optional[str] = None,
              purpose: str = DEFAULT_PURPOSE, run_label: str, compute: Compute, request_id: Optional[str] = None,
              family: Optional[str] = None, request_kind: Optional[str] = None,
              emit: Callable[[str], None] = lambda m: print(m, file=sys.stderr, flush=True)) -> Dict[str, Any]:
    """Play ``pairs`` mirrored pairs of ``player`` vs ``opponent`` in batches, one §0b row per batch appended under
    its claim to the ledger root ``out_dir`` (``None`` = the run archive's ``_ledger``; any other root under
    ``models/`` is refused), for the request ``request_id`` (default: :func:`default_request_id`). Skips the batches
    the request already holds. Returns the edge's pooled summary (``main.h2h.stats.edge_summary``)."""
    from main.h2h import stats as ST
    from utils.git import get_git_hash

    if purpose not in L.PURPOSES:
        raise H2HError(f"purpose {purpose!r} not in {L.PURPOSES}")
    root = L.check_write_root(out_dir) if out_dir is not None else L.archive_ledger_root()
    key = schedule_key or schedule_key_of(player, opponent)
    plan = batch_plan(pairs, batch_pairs)
    check_core_flags(player)
    check_core_flags(opponent)
    from utils.rust_env import episode as EP

    regime = regime_for(EP.stall_threshold())
    rid = request_id or default_request_id(player, opponent, regime["regime_id"], key, schedule_seed)
    kind = request_kind or ("ab_cell" if purpose == "ab" else "adhoc")
    writer = L.LedgerWriter(root, producer=PRODUCER)
    try:
        req = writer.open_request(rid, kind=kind, purpose=purpose, family=family, protocol=PROTOCOL,
                                  spec={"producer": PRODUCER, "batch_pairs": int(batch_pairs),
                                        "schedule_seed": int(schedule_seed)})
    except L.RequestSpecError as e:
        raise H2HError(f"request {rid}: {e} — a different batch size (or schedule seed) over an existing request "
                       "would replay the same batch index under another length; use the same --batch-pairs, or a "
                       "new --request") from None
    existing = L.read(H2H_RESUME, root=root, request_id=rid, regime_id=regime["regime_id"])
    done = completed_batches(existing.rows, player, opponent)
    for b_i, n in done.items():
        if b_i >= len(plan) or plan[b_i] != n:
            raise H2HError(f"request {rid} already holds batch {b_i} of this edge with {n} pairs, but this call's plan "
                           f"is {plan[:3]}... — a different batch size over existing rows would replay the same "
                           "batch index under another length; use the same --batch-pairs / a new --request")
    todo = [b_i for b_i in range(len(plan)) if b_i not in done]
    emit(f"[h2h] {player.id} vs {opponent.id}: {pairs} pairs in {len(plan)} batch(es) of {batch_pairs}; "
         f"{len(done)} already recorded, {len(todo)} to play; schedule {key} seed {schedule_seed}; request {rid} "
         f"({kind}, purpose {purpose}{f', family {family}' if family else ''}) under {root}")
    if todo:
        from utils.contention import cpu_contention_factor

        lock = contextlib.nullcontext()
        if compute.device.startswith("cuda"):
            from utils.gpu_lock import gpu_lock

            lock = gpu_lock(what="main.h2h")
        try:
            commit = get_git_hash()
        except Exception:                                            # noqa: BLE001 - a read still runs
            commit = "unknown"
        with lock:
            eng = H2HEngine(player, opponent, compute, emit)
            try:
                if eng.regime["regime_id"] != regime["regime_id"]:
                    raise H2HError("the engine's regime differs from the one planned")
                last_wall: Optional[float] = None
                for b_i in todo:
                    n = plan[b_i]
                    try:
                        claim = writer.claim(rid, batch=b_i, player=player.sha256, opponent=opponent.sha256,
                                             regime_id=regime["regime_id"],
                                             expected_wall_s=last_wall or 2 * n / EXPECTED_GAMES_PER_S)
                    except L.AlreadyRecordedError as e:
                        emit(f"[h2h] batch {b_i}: {e} — skipped")
                        continue
                    cseed = cycle_seed(schedule_seed, key, b_i)
                    load0, t0 = _loadavg(), time.perf_counter()
                    t_start = L.utc_now()
                    games, st = eng.play_batch(n, cseed)
                    wall = time.perf_counter() - t0
                    last_wall = wall
                    t_end = L.utc_now()
                    score = score_games(games, eng.team_packed, n)
                    if st.get("executor_pair_counts") is not None and list(st["executor_pair_counts"]) != score.pair_counts:
                        raise H2HError(f"batch {b_i}: the executor's pentanomial {st['executor_pair_counts']} != the "
                                       f"game log's {score.pair_counts}")
                    row = build_row(writer=writer, run_label=run_label, commit=commit, a=player, b=opponent,
                                    eng=eng, purpose=purpose, key=key, sched_seed=schedule_seed, batch=b_i,
                                    cseed=cseed, score=score, t_start=t_start, t_end=t_end, wall_s=wall,
                                    load={"start": load0, "end": _loadavg(), "contention": cpu_contention_factor(refresh=True)},
                                    executor=st, request=req)
                    try:
                        writer.append_row(row, claim)
                    except L.ClaimVoidedError as e:
                        emit(f"[h2h] batch {b_i}: {e}")
                        continue
                    emit(f"[h2h] batch {b_i + 1}/{len(plan)}: {n} pairs / {2 * n} games in {wall:.1f}s "
                         f"({2 * n / wall:.2f} games/s); W/L/D {score.w}/{score.l}/{score.d}; pair counts "
                         f"{score.pair_counts}; near-tie games {score.near_tie_games}; digest "
                         f"{(score.outcome_digest or '')[:12]}")
            finally:
                eng.close()
    writer.close()
    got = L.read(H2H_RESUME, root=root, request_id=rid, regime_id=regime["regime_id"],
                 players=[player.sha256], opponents=[opponent.sha256])
    return ST.edge_summary(list(got.rows))
