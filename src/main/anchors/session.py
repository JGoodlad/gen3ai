"""What BOTH of our sides share, and the watchdog that refuses to report nothing.

Our side of a read is played by one of two poke-env-free transports — the in-process core slot
(:mod:`main.anchors.core_side`) or a websocket client on the Rust stack (:mod:`main.anchors.live_side`);
the legacy ``main.play`` → ``RLPlayer`` client was deleted in P6 of the poke-env retirement. This module
holds what they share:

* the **team source** (our 719-team pool, a directory of Showdown exports, or a mirrored-pair
  sequence) with a seeded private draw RNG, and the team actually yielded recorded per game;
* the **per-battle record** (:class:`BattleRecord`) each transport appends as a battle finishes, so a
  series that dies halfway keeps what completed, and the per-half :class:`OurSideState` (the records,
  the draws, the ``stochastic`` value each decision REALLY used, the loader that ran);
* the **foreign loader** a cross-run snapshot needs, and the PEER plumbing (start / await / stop by
  PID, Metamon's own battle CSV for an anchor-vs-anchor cell).

**Watch the pair, and FAIL LOUDLY with a named cause.** This is the half that hazard H-B bought:
when our side forfeits at turn 250, Metamon's long-tail handler force-resets and then calls itself
(~985 levels, then a ``RecursionError``), the process dies, and a naive harness sits there with a
live client, no opponent, and nothing to report. Four named causes, never a silent zero:

=====================  ===================================================================
``peer_never_ready``   the peer process never printed its online banner
``peer_exited``        the peer died before the series finished (rc and its log tail named)
``no_first_game``      nothing finished inside the first-game deadline
``no_progress``        games stopped finishing inside the progress deadline
=====================  ===================================================================

Each is returned as a ``failure`` dict that reaches ``summary.json``'s ``status: "FAILED"`` and a
non-zero exit code, **with whatever games did finish attached** — a partial n is honest only when
it is labelled partial.
"""
from __future__ import annotations

import asyncio
import glob
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from main.anchors.peers import TEAM_FILE_GLOBS, PeerPlan


class SeriesFailure(RuntimeError):
    """A named, reportable failure of the pair — never raised for an ordinary loss."""

    def __init__(self, cause: str, detail: str) -> None:
        super().__init__(f"{cause}: {detail}")
        self.cause = cause
        self.detail = detail

    def as_dict(self) -> Dict[str, str]:
        return {"cause": self.cause, "detail": self.detail}


# ------------------------------------------------------------------ the team source we draw from
def load_team_texts(directory: Path) -> "list[tuple[str, str]]":
    """``(basename, text)`` for every team file in ``directory``, each ``.strip()``ed.

    The glob allowlist lives in :data:`main.anchors.peers.TEAM_FILE_GLOBS` so BOTH sides count the
    same thing — Metamon's `competitive` directory ships an `index.csv` beside its 20 teams, and a
    count that included it would report a 20-vs-21 team-source asymmetry that does not exist.

    🚨 **The ``.strip()`` is load-bearing history, not tidiness.** Before 2026-09-14 our vendored
    fork's ``Teambuilder.parse_showdown_team`` skipped a split chunk only when it was exactly
    ``""``, so a file ending ``"\\n\\n\\n"`` — which every Metamon ``competitive`` file does —
    produced an EMPTY 7th Pokemon, Showdown rejected the packed team, and the challenge never
    became a battle: the series HUNG. The fork is fixed and ``Gen3Teambuilder`` now raises on an
    oversize pack, so this is the third line of defence rather than the first; it stays because
    normalising a file we did not write costs nothing and the failure it prevents costs a session.
    """
    seen: Dict[str, str] = {}
    for pattern in TEAM_FILE_GLOBS:
        for path in sorted(glob.glob(os.path.join(str(directory), pattern))):
            name = os.path.basename(path)
            if name.startswith("."):
                continue
            with open(path) as fh:
                seen[name] = fh.read().strip()
    return sorted(seen.items())


def build_team_source(spec: Dict[str, Any], seed: Optional[int], draw_log: List[Optional[str]]):
    """A ``Gen3Teambuilder`` over the configured source, plus a label for every pool index.

    ``spec`` is ``{"kind": "pool"}`` or ``{"kind": "dir", "path": ...}``.
    """
    from utils.teambuilder import Gen3Teambuilder

    kind = spec.get("kind")
    if kind == "sequence":
        # MIRRORED TEAM PAIRS (T17, `main.anchors.mirrored`): the planned (label, text) per battle, IN
        # ORDER. Every team was validated when the plan was drawn; a builder that still dropped one
        # would shift every later battle onto the wrong team, so that is REFUSED, not absorbed.
        items = list(spec["items"])
        tb = Gen3Teambuilder([text for _, text in items], rng_seed=seed)
        if len(tb.packed_teams) != len(items):
            raise SeriesFailure("bad_team_source",
                                f"the mirrored-pair sequence lost {len(items) - len(tb.packed_teams)} "
                                "team(s) to the builder's validation — the plan would desync")
        order = iter(range(len(items)))

        def yield_sequence():
            i = next(order, None)
            if i is None:
                raise SeriesFailure("bad_team_source", "the mirrored-pair team sequence ran out")
            draw_log.append(items[i][0])
            return tb.packed_teams[i]

        tb.yield_team = yield_sequence
        return tb
    if kind == "pool":
        from utils.team_loader import TeamLoader

        tb = Gen3Teambuilder([t.strip() for t in TeamLoader().get_all_teams()], rng_seed=seed)
        # Our pool's own fingerprints ARE the Metamon export's filenames (`<sha>.gen3ou_team`),
        # so the two sides' team identifiers join with no translation table.
        labels = list(tb.get_pool_team_keys())
    elif kind == "dir":
        directory = Path(spec["path"])
        items = load_team_texts(directory)
        if not items:
            raise SeriesFailure("no_teams",
                                f"{directory} holds no team files matching {TEAM_FILE_GLOBS}")
        tb = Gen3Teambuilder([text for _, text in items], rng_seed=seed)
        # `Gen3Teambuilder` DROPS locally-invalid teams, so labels must follow the SURVIVORS, not
        # the original file order — otherwise every recorded team name is off by the drop count.
        from utils.bridge.team_validator import validate_teams_locally

        verdicts = validate_teams_locally("gen3ou", [text for _, text in items])
        labels = [name for (name, _), v in zip(items, verdicts) if v.get("valid")]
    else:
        raise SeriesFailure("bad_team_source", f"unknown team source kind {kind!r}")

    tb._label_by_idx = labels

    inner_yield = tb.yield_team

    def yield_team():
        packed = inner_yield()
        idx = tb._last_pool_idx
        draw_log.append(labels[idx] if (idx is not None and idx < len(labels)) else None)
        return packed

    tb.yield_team = yield_team
    return tb


# ------------------------------------------------------------------------- the per-battle record
@dataclass
class BattleRecord:
    """One finished battle, as our side saw it. The tool's authoritative per-game view."""

    battle_tag: str
    turns: int
    won: Optional[bool]
    finished: Optional[bool]
    hit_forfeit_limit: bool
    our_team: Optional[str]
    n_decisions: Optional[int]
    n_defaults: Optional[int]
    n_redecides: Optional[int]
    t_finished: float
    #: OUR per-decision argmax check (X22 f3): of the decisions our policy made in this battle,
    #: how many chose the policy's own argmax. Counted per decision from that decision's masked
    #: logits, never assumed from the temperature. Greedy ⇒ equal by construction.
    our_argmax_matches: Optional[int] = None
    our_argmax_decisions: Optional[int] = None

    @property
    def result(self) -> str:
        """``won is None`` with ``finished`` true is a TIE (the server's ``|tie``, never a ``|win|``).

        Ties count in the denominator and not the numerator, as both 2026-09-14 batteries did.
        """
        if self.won is True:
            return "win"
        if self.won is False:
            return "loss"
        return "tie"


@dataclass
class OurSideState:
    records: List[BattleRecord] = field(default_factory=list)
    draws: List[Optional[str]] = field(default_factory=list)
    stochastic_kwargs: List[bool] = field(default_factory=list)
    decision_times: List[float] = field(default_factory=list)
    last_progress: float = field(default_factory=time.time)
    error: Optional[str] = None
    #: Which loader actually built our policy — "bare" / "foreign" / "" (our side is a bot).
    model_loader: str = ""


def foreign_loader(zip_path: str, device: str):
    """Load a frozen snapshot from ANOTHER run — the loader a cross-run anchor read needs.

    🚨 A bare ``MaskablePPO.load`` rebuilds the extractor from the zip's own ``policy_kwargs`` and
    hands every one of them to the CURRENT ``ExtractorBuild``. A snapshot from an older run in the
    SAME observation family therefore dies with ``got an unexpected keyword argument
    'threat_prob_outspeed'`` — measured 2026-09-14 on every ``ai_v9_29_rev1_0823`` node, while
    ``ai_v12``/``ai_v13`` nodes load fine. ``load_foreign_opponent`` is the path the snapshot
    ladder and the fixed-opponent pool already use for exactly this: it reads the zip's OWN
    ``model_config.json``, checks the ``arch_signature`` (observation-family compatibility, which
    is the property that actually matters for a frozen opponent), and refuses a PRE-GENERATION
    checkpoint below ``MIGRATION_FLOOR`` rather than loading it wrong.
    """
    import os

    from agents.model.oracle_reveal import refuse_if_revealed
    from agents.model.snapshot import current_model_version, load_foreign_opponent
    from agents.observation.state_encoder import load_mappings

    refuse_if_revealed(zip_path, tool="main.anchors", reason="The anchor session builds its observation without the reveal.")
    cfg = None
    d = os.path.dirname(os.path.abspath(zip_path))
    for cand_dir in (d, os.path.dirname(d)):
        cand = os.path.join(cand_dir, "model_config.json")
        if os.path.exists(cand):
            cfg = cand
            break
    model, _foreign = load_foreign_opponent(
        zip_path, current_version=current_model_version(load_mappings()),
        device=device, config_path=cfg)
    return model


# ------------------------------------------------------ a PEER as our side (anchor vs anchor)
#: Metamon's own per-battle log, written under ``--results-dir``. The header is
#: ``Player Username, Team File, Opponent Username, Result, Turn Count, Battle ID``.
PEER_RESULT_GLOB = "battle_log_*.csv"


def read_peer_battles(results_dir: Path) -> "list[BattleRecord]":
    """Metamon's OWN per-battle CSV, as :class:`BattleRecord`s.

    This is the only per-game view available when BOTH sides are external peers — there is no
    side of ours in the process to observe. It is a weaker instrument than our own side's record
    (the server's ``|win|`` / ``|tie``) and the difference is NAMED rather than smoothed over: Metamon books a result as
    ``WIN``/``LOSS`` off a BOOLEAN, so a TIE is recorded as its own LOSS. Ties ran 0-1 per 100
    games in the 2026-09-14 batteries, so the bias is small — but it is a bias, and a head-to-head
    edge built from this source says so.

    🚨 The file is APPENDED to across runs of the same ``--results-dir``. A retry must write to a
    fresh directory or it inherits the previous attempt's games; the campaign driver retires a
    failed cell directory rather than reusing it, which is what makes that true here.
    """
    import csv

    out: List[BattleRecord] = []
    for path in sorted(results_dir.glob(PEER_RESULT_GLOB)):
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh, skipinitialspace=True):
                result = (row.get("Result") or "").strip().upper()
                try:
                    turns = int(row.get("Turn Count") or 0)
                except ValueError:
                    turns = 0
                team = os.path.basename((row.get("Team File") or "").strip()) or None
                out.append(BattleRecord(
                    battle_tag=f"battle-{(row.get('Battle ID') or '').strip()}",
                    turns=turns,
                    won=True if result == "WIN" else (False if result == "LOSS" else None),
                    finished=True,
                    hit_forfeit_limit=False,
                    our_team=team,
                    n_decisions=None, n_defaults=None, n_redecides=None,
                    t_finished=time.time(),
                ))
    return out


async def watch_peer_pair(procs: "list[subprocess.Popen]", plans: "list[PeerPlan]",
                          results_dir: Path, state: OurSideState, expected: int,
                          first_game_timeout_s: float, progress_timeout_s: float,
                          poll_s: float = 5.0) -> None:
    """The watchdog for a cell with NO side of ours in it.

    Progress is the row count in our peer's own battle CSV — the same signal `watch` takes from
    `state.records`, read from disk instead of from an observer. Both processes are watched,
    because either one dying leaves the other waiting on a challenge that will never come.
    """
    started = time.time()
    seen = 0
    while True:
        await asyncio.sleep(poll_s)
        state.records = read_peer_battles(results_dir)
        done = len(state.records)
        while seen < done:
            # The same per-game line the observed path prints. Without it an anchor-vs-anchor cell
            # looks IDENTICAL to a hung one from outside — the watchdog knows it is progressing and
            # nobody watching the log does.
            rec = state.records[seen]
            seen += 1
            print(f"[anchors] game {seen}: {rec.result} in {rec.turns} turns "
                  f"(team={rec.our_team})", flush=True)
            state.last_progress = time.time()
        if done >= expected:
            return
        for proc, plan in zip(procs, plans):
            if proc.poll() is not None and proc.returncode != 0:
                raise SeriesFailure(
                    "peer_exited",
                    f"{plan.label} exited rc={proc.returncode} after {done}/{expected} games. "
                    + log_tail(plan.log_path))
        if all(p.poll() is not None for p in procs):
            if done >= expected:
                return
            raise SeriesFailure(
                "short_series",
                f"both peers exited cleanly with only {done}/{expected} games in "
                f"{results_dir}. " + log_tail(plans[0].log_path))
        idle = time.time() - (state.last_progress if done else started)
        budget = progress_timeout_s if done else first_game_timeout_s
        if idle > budget:
            raise SeriesFailure(
                "no_progress" if done else "no_first_game",
                f"{done}/{expected} games finished and nothing has completed for {idle:.0f}s "
                f"(budget {budget:.0f}s). " + log_tail(plans[0].log_path))


# --------------------------------------------------------------------------------- the watchdog
def log_has(path: Path, pattern: str) -> bool:
    if not path.exists():
        return False
    return re.search(pattern, path.read_text(errors="replace")) is not None


def log_tail(path: Path, n: int = 20) -> str:
    if not path.exists():
        return f"(no log at {path})"
    lines = path.read_text(errors="replace").splitlines()[-n:]
    return f"last {len(lines)} lines of {path}:\n" + "\n".join(lines)


def start_peer(plan: PeerPlan, nice: int = 10) -> subprocess.Popen:
    """Start the peer, recording its PID. ``nice`` because a training arm owns this box."""
    plan.log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(plan.log_path, "w", buffering=1)
    argv = (["nice", "-n", str(nice)] if nice else []) + plan.argv
    proc = subprocess.Popen(argv, cwd=str(plan.cwd), env=plan.env,
                            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"[anchors] peer {plan.label} pid={proc.pid} -> {plan.log_path}", flush=True)
    return proc


async def await_peer_exit(proc: Optional[subprocess.Popen], timeout_s: float,
                          poll_s: float = 1.0) -> bool:
    """Let the peer finish on its OWN and write its report before anything terminates it.

    🚨 Measured 2026-09-14: our side finishes its last battle a beat before the peer does, so a
    harness that kills the peer the moment its own games are done kills it MID-REPORT — and the
    cell then has games but no ``argmax_match_rate``, i.e. a win rate whose regime is UNVERIFIED.
    That is precisely the number this tool refuses to emit, so the wait is part of the
    measurement, not tidiness. Both peers exit by themselves once ``--total-battles`` is reached.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is None or proc.poll() is not None:
            return True
        await asyncio.sleep(poll_s)
    return False


def stop_peer(proc: Optional[subprocess.Popen], grace_s: float = 15.0) -> None:
    """Terminate EXACTLY the PID we started. Idempotent, never raises, never `pkill`."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            pass


async def await_peer_ready(proc: subprocess.Popen, plan: PeerPlan, timeout_s: float,
                           poll_s: float = 2.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if log_has(plan.log_path, plan.ready_pattern):
            print(f"[anchors] peer {plan.label} online", flush=True)
            return
        if proc.poll() is not None:
            raise SeriesFailure(
                "peer_exited",
                f"{plan.label} exited rc={proc.returncode} before it came online. "
                + log_tail(plan.log_path))
        await asyncio.sleep(poll_s)
    raise SeriesFailure(
        "peer_never_ready",
        f"{plan.label} never matched /{plan.ready_pattern}/ within {timeout_s:g}s. "
        + log_tail(plan.log_path))


async def watch(proc: subprocess.Popen, plan: PeerPlan, state: OurSideState, expected: int,
                first_game_timeout_s: float, progress_timeout_s: float,
                poll_s: float = 2.0) -> None:
    """Raise a NAMED :class:`SeriesFailure` the moment the pair stops making progress.

    Runs beside our side's task; whichever finishes first decides the half.
    """
    started = time.time()
    while True:
        await asyncio.sleep(poll_s)
        done = len(state.records)
        if done >= expected:
            return
        if proc.poll() is not None:
            raise SeriesFailure(
                "peer_exited",
                f"{plan.label} exited rc={proc.returncode} after {done}/{expected} games. "
                "Metamon answers a 250-turn forfeit with unbounded recursion (hazard H-B), so a "
                "mid-series death is expected there and must never read as 'no result'. "
                + log_tail(plan.log_path))
        idle = time.time() - (state.last_progress if done else started)
        budget = progress_timeout_s if done else first_game_timeout_s
        if idle > budget:
            raise SeriesFailure(
                "no_progress" if done else "no_first_game",
                f"{done}/{expected} games finished and nothing has completed for {idle:.0f}s "
                f"(budget {budget:.0f}s). A rejected team, a dropped challenge or a wedged login "
                "all present as a HANG rather than an error. " + log_tail(plan.log_path))
